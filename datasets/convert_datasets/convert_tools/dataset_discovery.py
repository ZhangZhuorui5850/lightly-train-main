#!/usr/bin/env python3
"""Reusable dataset discovery and lightweight format inspection.

The conversion frontends use this module to find datasets from their contents instead
of relying on directory names such as ``dataset_seg``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetCandidate:
    path: Path
    kind: str
    image_count: int = 0
    annotation_count: int = 0
    class_count: int = 0
    splits: tuple[str, ...] = ()
    config_path: Path | None = None
    issues: tuple[str, ...] = field(default_factory=tuple)


KIND_LABELS = {
    "yolo_instance": "YOLO 实例分割",
    "yolo_detection": "YOLO 目标检测",
    "semantic_mask": "PNG 语义分割",
    "labelme": "LabelMe",
    "generated_mask": "image + fg 掩码",
    "mvtec": "MVTec AD",
    "unknown": "待识别",
}


def auto_datasets_root(start: Path | None = None) -> Path:
    """Locate the repository's datasets directory from a script or cwd."""
    origin = (start or Path(__file__)).resolve()
    parents = (origin, *origin.parents) if origin.is_dir() else origin.parents
    for parent in parents:
        candidate = parent / "datasets"
        if candidate.is_dir():
            return candidate.resolve()
    return (Path.cwd() / "datasets").resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def dataset_root_from_config(config_path: Path, config: dict[str, Any]) -> Path:
    raw_path = config.get("path")
    if raw_path is None:
        return config_path.parent.resolve()
    root = Path(str(raw_path)).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return root.resolve()


def class_names(config: dict[str, Any], root: Path) -> dict[int, str]:
    raw = config.get("names", config.get("classes"))
    if isinstance(raw, list):
        return {idx: str(name) for idx, name in enumerate(raw)}
    if isinstance(raw, dict):
        result: dict[int, str] = {}
        for key, value in raw.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, dict):
                value = value.get("name", f"class_{class_id}")
            result[class_id] = str(value)
        return result
    classes_txt = root / "classes.txt"
    if classes_txt.is_file():
        lines = [line.strip() for line in classes_txt.read_text(encoding="utf-8").splitlines()]
        return {idx: name for idx, name in enumerate(lines) if name}
    return {}


def _split_value(config: dict[str, Any], split: str) -> Any:
    value = config.get(split)
    if isinstance(value, dict):
        return value.get("images")
    return value


def split_image_dirs(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for split in SPLITS:
        value = _split_value(config, split)
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            fallback = root / "images" / split
            if fallback.is_dir():
                result[split] = fallback
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_dir():
            result[split] = path.resolve()
    return result


def label_dir_from_image_dir(image_dir: Path, root: Path, split: str) -> Path:
    parts = list(image_dir.parts)
    image_positions = [idx for idx, part in enumerate(parts) if part.lower() == "images"]
    if image_positions:
        parts[image_positions[-1]] = "labels"
        return Path(*parts)
    return root / "labels" / split


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return (
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _sample_yolo_lines(label_dirs: Iterable[Path], limit: int = 100) -> list[list[str]]:
    rows: list[list[str]] = []
    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            try:
                lines = label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            for line in lines:
                tokens = line.split()
                if tokens:
                    rows.append(tokens)
                    if len(rows) >= limit:
                        return rows
    return rows


def inspect_config_dataset(config_path: Path) -> DatasetCandidate:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    root = dataset_root_from_config(config_path, config)
    names = class_names(config, root)
    image_dirs = split_image_dirs(root, config)
    splits = tuple(split for split in SPLITS if split in image_dirs)
    image_count = sum(sum(1 for _ in _iter_images(path)) for path in image_dirs.values())

    semantic = any(
        isinstance(config.get(split), dict) and config[split].get("masks")
        for split in SPLITS
    ) or any((root / "masks" / split).is_dir() for split in SPLITS)
    if semantic:
        mask_count = sum(
            len(list((root / "masks" / split).glob("*.png")))
            for split in SPLITS
            if (root / "masks" / split).is_dir()
        )
        return DatasetCandidate(
            path=root,
            kind="semantic_mask",
            image_count=image_count,
            annotation_count=mask_count,
            class_count=len(names),
            splits=splits,
            config_path=config_path,
        )

    label_dirs = {
        split: label_dir_from_image_dir(image_dir, root, split)
        for split, image_dir in image_dirs.items()
    }
    rows = _sample_yolo_lines(label_dirs.values())
    polygons = sum(1 for row in rows if len(row) >= 7 and len(row) % 2 == 1)
    boxes = sum(1 for row in rows if len(row) == 5)
    kind = "unknown"
    if polygons:
        kind = "yolo_instance"
    elif boxes:
        kind = "yolo_detection"
    annotation_count = sum(
        len(list(path.glob("*.txt"))) for path in label_dirs.values() if path.is_dir()
    )
    issues: list[str] = []
    if not image_dirs:
        issues.append("缺少可解析的 images split")
    if kind == "unknown":
        issues.append("标签为空或标签行格式无法识别")
    return DatasetCandidate(
        path=root,
        kind=kind,
        image_count=image_count,
        annotation_count=annotation_count,
        class_count=len(names),
        splits=splits,
        config_path=config_path,
        issues=tuple(issues),
    )


def _inspect_unconfigured_dir(path: Path, filenames: set[str]) -> DatasetCandidate | None:
    image_files = [name for name in filenames if Path(name).suffix.lower() in IMAGE_EXTENSIONS]
    json_files = [name for name in filenames if Path(name).suffix.lower() == ".json"]
    if json_files and image_files:
        return DatasetCandidate(
            path=path.resolve(),
            kind="labelme",
            image_count=len(image_files),
            annotation_count=len(json_files),
        )
    return None


def scan_generated_mask_roots(search_root: Path) -> list[Path]:
    """Group ``root/object/defect/{image,fg}`` leaves into reusable source roots."""
    search_root = search_root.expanduser().resolve()
    roots: set[Path] = set()
    for image_dir in search_root.rglob("image"):
        if not image_dir.is_dir() or not (image_dir.parent / "fg").is_dir():
            continue
        leaf = image_dir.parent
        grouped = leaf.parent.parent
        try:
            grouped.relative_to(search_root)
        except ValueError:
            grouped = leaf
        roots.add(grouped)
    return sorted(roots, key=lambda path: str(path).lower())


def _generated_candidate(path: Path) -> DatasetCandidate:
    image_count = 0
    annotation_count = 0
    for image_dir in path.rglob("image"):
        fg_dir = image_dir.parent / "fg"
        if not image_dir.is_dir() or not fg_dir.is_dir():
            continue
        images = {item.stem for item in _iter_images(image_dir)}
        masks = {item.stem for item in _iter_images(fg_dir)}
        image_count += len(images)
        annotation_count += len(images & masks)
    return DatasetCandidate(
        path=path.resolve(),
        kind="generated_mask",
        image_count=image_count,
        annotation_count=annotation_count,
    )


def _mvtec_candidates(
    search_root: Path,
    category_roots: set[Path],
) -> list[DatasetCandidate]:
    """Group MVTec category directories by their dataset root."""
    if not category_roots:
        return []
    grouped: dict[Path, list[Path]] = {}
    if search_root in category_roots:
        grouped[search_root] = [search_root]
    for category in category_roots - {search_root}:
        grouped.setdefault(category.parent, []).append(category)

    candidates: list[DatasetCandidate] = []
    for root, categories in grouped.items():
        image_count = 0
        annotation_count = 0
        defects: set[str] = set()
        for category in categories:
            for split in ("train", "test"):
                split_dir = category / split
                if split_dir.is_dir():
                    image_count += sum(
                        1 for path in split_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                    )
            ground_truth = category / "ground_truth"
            annotation_count += sum(
                1 for path in ground_truth.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            defects.update(
                path.name for path in ground_truth.iterdir() if path.is_dir()
            )
        candidates.append(DatasetCandidate(
            path=root.resolve(),
            kind="mvtec",
            image_count=image_count,
            annotation_count=annotation_count,
            class_count=len(defects),
            splits=("train", "test"),
        ))
    return candidates


def scan_mvtec_datasets(search_root: Path) -> list[DatasetCandidate]:
    """Find MVTec roots without descending into each category's image tree."""
    search_root = search_root.expanduser().resolve()
    if not search_root.is_dir():
        return []
    category_roots: set[Path] = set()
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", "out", "weights"}
    for current, dirnames, _ in os.walk(search_root):
        dirnames[:] = [
            name for name in dirnames
            if name not in skipped_dirs and not name.startswith(".")
        ]
        if "test" in dirnames and "ground_truth" in dirnames:
            category_roots.add(Path(current).resolve())
            dirnames[:] = []
    return sorted(
        _mvtec_candidates(search_root, category_roots),
        key=lambda item: str(item.path).lower(),
    )


def scan_datasets(search_root: Path) -> list[DatasetCandidate]:
    """Recursively find configured and supported raw datasets below ``search_root``."""
    search_root = search_root.expanduser().resolve()
    if not search_root.is_dir():
        return []

    candidates: list[DatasetCandidate] = []
    configured_roots: set[Path] = set()
    mvtec_category_roots: set[Path] = set()
    unconfigured: list[DatasetCandidate] = []
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", "out", "weights"}

    candidates.extend(_generated_candidate(path) for path in scan_generated_mask_roots(search_root))

    for current, dirnames, filenames_list in os.walk(search_root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in skipped_dirs and not name.startswith(".")
        ]
        path = Path(current)
        filenames = set(filenames_list)
        if "test" in dirnames and "ground_truth" in dirnames:
            mvtec_category_roots.add(path.resolve())
        yaml_name = "data.yaml" if "data.yaml" in filenames else (
            "dataset.yaml" if "dataset.yaml" in filenames else None
        )
        if yaml_name:
            candidate = inspect_config_dataset(path / yaml_name)
            candidates.append(candidate)
            configured_roots.add(candidate.path)
            continue
        raw_candidate = _inspect_unconfigured_dir(path, filenames)
        if raw_candidate is not None:
            unconfigured.append(raw_candidate)

    for candidate in unconfigured:
        inside_configured_root = any(
            candidate.path == root or root in candidate.path.parents
            for root in configured_roots
        )
        if inside_configured_root:
            continue
        candidates.append(candidate)
    candidates.extend(_mvtec_candidates(search_root, mvtec_category_roots))

    unique: dict[tuple[Path, str], DatasetCandidate] = {}
    for candidate in candidates:
        unique[(candidate.path, candidate.kind)] = candidate
    return sorted(unique.values(), key=lambda item: (item.kind, str(item.path).lower()))


def conversion_actions(candidate: DatasetCandidate) -> tuple[str, ...]:
    return {
        "yolo_instance": ("to-semantic", "to-mvtec"),
        "labelme": ("to-yolo",),
        "generated_mask": ("to-yolo-seg",),
        "mvtec": ("mvtec-to-yolo",),
    }.get(candidate.kind, ())
