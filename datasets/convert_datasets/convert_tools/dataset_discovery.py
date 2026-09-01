#!/usr/bin/env python3
"""Reusable dataset discovery and lightweight format inspection.

The conversion frontends use this module to find datasets from their contents instead
of relying on directory names such as ``dataset_seg``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

try:
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


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
    modified_time: float = 0.0
    counts_are_exact: bool = True
    image_count_is_exact: bool = True
    annotation_count_is_exact: bool = True


KIND_LABELS = {
    "image_classification": "图片分类",
    "yolo_instance": "YOLO 实例分割",
    "yolo_detection": "YOLO 目标检测",
    "semantic_mask": "PNG 语义分割",
    "labelme": "LabelMe",
    "generated_mask": "image + fg 掩码",
    "mvtec": "MVTec AD",
    "unknown": "待识别",
}

STANDARD_CONFIG_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")


def _declared_task_kind(config: dict[str, Any]) -> str | None:
    task = str(config.get("task", "")).casefold()
    if task in {"classify", "classification", "image_classification", "cls"}:
        return "image_classification"
    if task in {"semantic", "semantic_segmentation"}:
        return "semantic_mask"
    if task in {"segment", "seg", "instance", "instance_segmentation"}:
        return "yolo_instance"
    if task in {"det", "detect", "detection", "object_detection"}:
        return "yolo_detection"
    return None


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _latest_mtime(paths: Iterable[Path]) -> float:
    return max((_safe_mtime(path) for path in paths), default=0.0)


def format_modified_time(timestamp: float) -> str:
    if timestamp <= 0:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def format_candidate_count(value: int, exact: bool) -> str:
    return str(value) if exact else f"≥{value}"


def sort_candidates_by_modified(
    candidates: Iterable[DatasetCandidate],
) -> list[DatasetCandidate]:
    return sorted(
        candidates,
        key=lambda item: (-item.modified_time, item.kind, str(item.path).casefold()),
    )


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
        value = yaml.safe_load(read_text_auto(path)) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    return value if isinstance(value, dict) else {}


def dataset_root_from_config(config_path: Path, config: dict[str, Any]) -> Path:
    raw_path = config.get("path")
    if raw_path is None:
        return config_path.parent.resolve()
    root = Path(str(raw_path)).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    configured_root = root.resolve()
    local_root = config_path.parent.resolve()
    configured_score = _root_image_source_score(configured_root, config)
    local_score = (
        _root_image_source_score(local_root, config)
        if local_root != configured_root
        else configured_score
    )
    if local_score > configured_score:
        return local_root
    if configured_score[0] > 0:
        return configured_root
    if local_score[0] > 0:
        return local_root
    return configured_root


def _root_has_image_source(root: Path, config: dict[str, Any]) -> bool:
    """Return whether ``root`` resolves to at least one real image.

    Directory existence alone is insufficient here: moved datasets often leave an
    empty directory tree at the old absolute ``path:``.  Treating that tree as valid
    prevents the local data.yaml directory from being used.
    """
    return _root_image_source_score(root, config)[0] > 0


def _root_image_source_score(root: Path, config: dict[str, Any]) -> tuple[int, int]:
    """Score a root by resolved splits and configured sources containing images."""
    split_count = 0
    source_count = 0
    for split in SPLITS:
        split_has_images = False
        value, _ = _split_values(config, split)
        values = value if isinstance(value, list) else [value]
        for raw_value in values:
            if raw_value is None:
                continue
            candidate = Path(str(raw_value)).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            candidate = candidate.resolve()
            if candidate.is_file() and candidate.suffix.casefold() in IMAGE_EXTENSIONS:
                split_has_images = True
                source_count += 1
                continue
            if candidate.is_file() and candidate.suffix.casefold() == ".txt":
                if any(
                    image.is_file() and image.suffix.casefold() in IMAGE_EXTENSIONS
                    for image in _read_image_manifest(candidate, root)
                ):
                    split_has_images = True
                    source_count += 1
                    continue
            image_dir = _as_image_dir(candidate)
            if image_dir is not None and _count_files_limited(
                [image_dir], IMAGE_EXTENSIONS, 1
            )[0]:
                split_has_images = True
                source_count += 1
        if not split_has_images:
            for alias in SPLIT_ALIASES[split]:
                for candidate in (root / "images" / alias, root / alias / "images"):
                    if candidate.is_dir() and _count_files_limited(
                        [candidate], IMAGE_EXTENSIONS, 1
                    )[0]:
                        split_has_images = True
                        source_count += 1
                        break
                if split_has_images:
                    break
        if split_has_images:
            split_count += 1
    return split_count, source_count


def dataset_path_diagnostics(
    config_path: Path,
    config: dict[str, Any],
) -> list[str]:
    """Describe every configured and conventional image path tried per split."""
    root = dataset_root_from_config(config_path, config)
    lines = [f"解析后的数据集根目录: {root}"]
    raw_root = config.get("path")
    if raw_root is not None:
        lines.append(f"YAML path: {raw_root}")
        configured_root = Path(str(raw_root)).expanduser()
        if not configured_root.is_absolute():
            configured_root = config_path.parent / configured_root
        configured_root = configured_root.resolve()
        lines.append(f"YAML path 解析结果: {configured_root}")
        if configured_root != root:
            lines.append(f"已采用 data.yaml 所在目录: {root}")
    for split in SPLITS:
        attempted: list[Path] = []
        for value in _configured_values(config, split):
            attempted.append(_resolve_path(root, value))
        attempted.extend(
            candidate
            for alias in SPLIT_ALIASES[split]
            for candidate in (root / "images" / alias, root / alias / "images")
        )
        unique = list(dict.fromkeys(path.resolve() for path in attempted))
        rendered = "; ".join(f"{path} ({_path_status(path)})" for path in unique)
        lines.append(f"{split}: {rendered}")
    return lines


def _path_status(path: Path) -> str:
    if not path.exists():
        return "缺失"
    if path.is_file():
        if path.suffix.casefold() == ".txt":
            return "存在，TXT 图片清单"
        if path.suffix.casefold() in IMAGE_EXTENSIONS:
            return "存在，单张图片"
        return "存在，文件"
    image_dir = _as_image_dir(path)
    if image_dir is None:
        return "存在，目录"
    image_count, exact = _count_files_limited(
        [image_dir], IMAGE_EXTENSIONS, 1
    )
    if image_count:
        return "存在，含图片"
    return "存在，空图片目录" if exact else "存在，含图片"


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
        try:
            text = read_text_auto(classes_txt)
        except (OSError, ValueError):
            return {}
        lines = [line.strip() for line in text.splitlines()]
        return {idx: name for idx, name in enumerate(lines) if name}
    return {}


def _split_values(config: dict[str, Any], split: str) -> tuple[Any, Any]:
    """Return configured image and annotation values for one split.

    Both common layouts are supported:

    ``images/{split}`` + ``labels/{split}`` (or ``masks/{split}``)
    ``{split}/images`` + ``{split}/labels`` (or ``{split}/masks``)

    A scalar split value may point at either the image directory or the split
    directory containing ``images`` and its sibling annotation directory.
    """
    value = next(
        (
            config[key]
            for key in SPLIT_ALIASES[split]
            if config.get(key) is not None
        ),
        None,
    )
    if isinstance(value, dict):
        image_value = value.get("images", value.get("image"))
        annotation_value = value.get(
            "labels", value.get("label", value.get("masks", value.get("mask")))
        )
        return image_value, annotation_value
    return value, None


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _configured_values(config: dict[str, Any], split: str) -> list[Any]:
    """Return every configured image source for a split.

    Ultralytics accepts a directory, a text manifest, or a list containing a
    mixture of both.  Keeping the list intact prevents later entries from being
    silently discarded during inspection and conversion.
    """
    value, _ = _split_values(config, split)
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _read_image_manifest(path: Path, root: Path) -> list[Path]:
    if not path.is_file() or path.suffix.casefold() != ".txt":
        return []
    result: list[Path] = []
    try:
        text = read_text_auto(path)
    except (OSError, ValueError):
        return []
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        item = Path(value).expanduser()
        if not item.is_absolute():
            # YOLO manifests conventionally resolve entries relative to the
            # manifest.  A root-relative fallback covers exported configs.
            local = (path.parent / item).resolve()
            item = local if local.exists() else (root / item).resolve()
        else:
            item = item.resolve()
        result.append(item)
    return result


def split_image_files(
    root: Path,
    config: dict[str, Any],
    *,
    selected_splits: Sequence[str] = SPLITS,
) -> dict[str, list[Path]]:
    """Resolve all configured image files, including list and txt-manifest splits."""
    result: dict[str, list[Path]] = {}
    for split in selected_splits:
        sources = _configured_values(config, split)
        fallback_sources = [
            candidate
            for alias in SPLIT_ALIASES[split]
            for candidate in (root / "images" / alias, root / alias / "images")
        ]
        files: list[Path] = []
        seen: set[Path] = set()
        for source_group in (sources, fallback_sources):
            for value in source_group:
                candidate = (
                    value
                    if isinstance(value, Path)
                    else _resolve_path(root, value)
                )
                candidate = candidate.expanduser().resolve()
                if candidate.suffix.casefold() == ".txt" and candidate.is_file():
                    candidates = _read_image_manifest(candidate, root)
                else:
                    image_dir = _as_image_dir(candidate)
                    candidates = (
                        list(_iter_images(image_dir))
                        if image_dir
                        else (
                            [candidate]
                            if candidate.is_file()
                            and candidate.suffix.casefold() in IMAGE_EXTENSIONS
                            else []
                        )
                    )
                for path in candidates:
                    path = path.resolve()
                    if path not in seen:
                        files.append(path)
                        seen.add(path)
            if files:
                break
        if files:
            result[split] = sorted(files)
    return result


def _logical_image_path(image_path: Path, root: Path, split: str) -> Path:
    """Build a stable split-relative path for outputs and annotation pairing."""
    resolved = image_path.expanduser().resolve()
    parts = list(resolved.parts)
    image_positions = [
        index for index, part in enumerate(parts) if part.casefold() in {"image", "images"}
    ]
    if image_positions:
        rel_parts = parts[image_positions[-1] + 1 :]
        if rel_parts and rel_parts[0].casefold() in SPLIT_ALIASES.get(split, (split,)):
            rel_parts = rel_parts[1:]
        if rel_parts:
            return Path(*rel_parts)
    try:
        return resolved.relative_to(root.resolve())
    except ValueError:
        return Path(resolved.name)


def _inferred_annotation_path(
    image_path: Path,
    *,
    annotation: str,
    root: Path,
    split: str,
    relative_path: Path,
) -> Path:
    suffixes = (".txt",) if annotation == "labels" else tuple(sorted(MASK_EXTENSIONS))
    parts = list(image_path.expanduser().resolve().parts)
    positions = [
        index for index, part in enumerate(parts) if part.casefold() in {"image", "images"}
    ]
    if positions:
        parts[positions[-1]] = annotation
        base = Path(*parts)
    else:
        base = root / annotation / split / relative_path
    for suffix in suffixes:
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate.resolve()
    default_suffix = ".txt" if annotation == "labels" else ".png"
    return base.with_suffix(default_suffix).resolve()


def split_sample_files(
    root: Path,
    config: dict[str, Any],
    split: str,
    *,
    annotation: str = "labels",
) -> list[tuple[Path, Path, Path]]:
    """Resolve the exact configured sample set for one split.

    Returns ``(image_path, annotation_path, relative_path)`` tuples.  Unlike
    ``split_image_dirs`` this preserves TXT manifests, direct image paths and every
    entry in a multi-directory split.
    """
    image_paths = split_image_files(root, config, selected_splits=(split,)).get(split, [])
    if not image_paths:
        return []

    _, raw_annotation = _split_values(config, split)
    annotation_values = (
        list(raw_annotation)
        if isinstance(raw_annotation, list)
        else ([raw_annotation] if raw_annotation is not None else [])
    )
    annotation_roots = [_resolve_path(root, value) for value in annotation_values]
    annotation_suffixes = (
        (".txt",) if annotation == "labels" else tuple(sorted(MASK_EXTENSIONS))
    )

    samples: list[tuple[Path, Path, Path]] = []
    seen_relative: dict[Path, Path] = {}
    for image_path in image_paths:
        image_path = image_path.expanduser().resolve()
        relative_path = _logical_image_path(image_path, root, split)
        previous = seen_relative.get(relative_path)
        if previous is not None and previous != image_path:
            raise ValueError(
                f"split '{split}' 中图片输出路径冲突: {previous} 与 {image_path} -> {relative_path}"
            )
        seen_relative[relative_path] = image_path

        annotation_path: Path | None = None
        for annotation_root in annotation_roots:
            base = (
                annotation_root
                if annotation_root.is_file()
                else annotation_root / relative_path
            )
            for suffix in annotation_suffixes:
                candidate = base.with_suffix(suffix)
                if candidate.exists():
                    annotation_path = candidate.resolve()
                    break
            if annotation_path is not None:
                break
        if annotation_path is None and annotation_roots:
            annotation_path = (annotation_roots[0] / relative_path).with_suffix(
                ".txt" if annotation == "labels" else ".png"
            ).resolve()
        if annotation_path is None:
            annotation_path = _inferred_annotation_path(
                image_path,
                annotation=annotation,
                root=root,
                split=split,
                relative_path=relative_path,
            )
        samples.append((image_path, annotation_path, relative_path))
    return samples


def _as_image_dir(path: Path) -> Path | None:
    """Resolve a configured image path that may actually be a split root."""
    child = path / "images"
    if child.is_dir():
        return child.resolve()
    if path.is_dir():
        return path.resolve()
    return None


def _sibling_dir(image_dir: Path, name: str, root: Path, split: str) -> Path:
    parts = list(image_dir.parts)
    positions = [idx for idx, part in enumerate(parts) if part.casefold() == "images"]
    if positions:
        parts[positions[-1]] = name
        return Path(*parts).resolve()
    return (root / name / split).resolve()


def split_image_dirs(
    root: Path,
    config: dict[str, Any],
    *,
    selected_splits: Sequence[str] = SPLITS,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for split in selected_splits:
        values = _configured_values(config, split)
        candidates: list[Path] = []
        for value in values:
            candidate = _resolve_path(root, value)
            if candidate.is_file() and candidate.suffix.casefold() == ".txt":
                manifest_files = _read_image_manifest(candidate, root)
                if manifest_files:
                    common = Path(os.path.commonpath([str(path) for path in manifest_files]))
                    candidates.append(
                        common.parent if common.is_file() or common.suffix else common
                    )
                continue
            candidates.append(candidate)
        candidates.extend(
            candidate
            for alias in SPLIT_ALIASES[split]
            for candidate in (root / "images" / alias, root / alias / "images")
        )
        for candidate in candidates:
            image_dir = _as_image_dir(candidate)
            if image_dir is not None:
                result[split] = image_dir
                break
    return result


def label_dir_from_image_dir(image_dir: Path, root: Path, split: str) -> Path:
    return _sibling_dir(image_dir, "labels", root, split)


def split_annotation_dirs(
    root: Path,
    config: dict[str, Any],
    image_dirs: dict[str, Path] | None = None,
    *,
    annotation: str,
    include_missing: bool = False,
    selected_splits: Sequence[str] = SPLITS,
) -> dict[str, Path]:
    """Resolve per-split ``labels`` or ``masks`` directories.

    Explicit per-split configuration has priority. The inferred sibling path
    then covers both ``images/{split}`` and ``{split}/images`` layouts.
    """
    image_dirs = image_dirs or split_image_dirs(
        root, config, selected_splits=selected_splits
    )
    result: dict[str, Path] = {}
    default_root_value = config.get(f"{annotation}_dir", annotation)
    for split in selected_splits:
        image_dir = image_dirs.get(split)
        image_value, annotation_value = _split_values(config, split)
        if isinstance(annotation_value, list):
            annotation_value = annotation_value[0] if annotation_value else None
        candidates: list[Path] = []
        if annotation_value is not None:
            candidates.append(_resolve_path(root, annotation_value))
        if image_dir is not None:
            candidates.append(_sibling_dir(image_dir, annotation, root, split))
        base = _resolve_path(root, default_root_value)
        candidates.extend(
            candidate
            for alias in SPLIT_ALIASES[split]
            for candidate in (base / alias, root / alias / annotation)
        )
        for candidate in candidates:
            if candidate.is_dir():
                result[split] = candidate.resolve()
                break
        else:
            if include_missing and candidates and (
                split in image_dirs or annotation_value is not None
            ):
                result[split] = candidates[0].resolve()
    return result


def _iter_images(directory: Path) -> Iterable[Path]:
    if not directory.is_dir():
        return ()
    return (
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _count_files_limited(
    directories: Iterable[Path],
    extensions: set[str],
    limit: int | None,
) -> tuple[int, bool]:
    count = 0
    seen: set[Path] = set()
    for directory in dict.fromkeys(path.resolve() for path in directories):
        if not directory.is_dir():
            continue
        pending = [directory]
        while pending:
            current = pending.pop()
            try:
                with os.scandir(current) as entries:
                    for entry in entries:
                        if entry.name.startswith("."):
                            continue
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                pending.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                        except OSError:
                            continue
                        if Path(entry.name).suffix.casefold() not in extensions:
                            continue
                        path = Path(entry.path).resolve()
                        if path in seen:
                            continue
                        seen.add(path)
                        count += 1
                        if limit is not None and count >= limit:
                            return count, False
            except OSError:
                continue
    return count, True


def _sample_yolo_lines(
    label_dirs: Iterable[Path],
    limit: int = 40,
    *,
    max_sampled_files: int = 24,
    count_limit: int | None = None,
    show_progress: bool = True,
) -> tuple[list[list[str]], int, bool]:
    """Count label files once and parse only a bounded sample for format detection."""
    rows: list[list[str]] = []
    label_count = 0
    sampled_nonempty_files = 0
    directories = [path for path in dict.fromkeys(label_dirs) if path.is_dir()]
    walker = tqdm(
        directories,
        desc="快速识别 YOLO 格式",
        unit="split",
        disable=not show_progress,
        leave=False,
    )
    scanned_dirs = 0
    for label_dir in walker:
        for current, dirnames, filenames in os.walk(label_dir):
            scanned_dirs += 1
            dirnames[:] = [name for name in dirnames if not name.startswith(".")]
            label_names = sorted(
                name for name in filenames if Path(name).suffix.casefold() == ".txt"
            )
            label_count += len(label_names)
            if scanned_dirs == 1 or scanned_dirs % 100 == 0:
                walker.set_postfix_str(
                    f"目录={scanned_dirs} 当前={Path(current).name or label_dir.name} labels={label_count}",
                    refresh=True,
                )
            if sampled_nonempty_files < max_sampled_files and len(rows) < limit:
                for name in label_names:
                    if sampled_nonempty_files >= max_sampled_files or len(rows) >= limit:
                        break
                    try:
                        lines = read_text_auto(Path(current) / name).splitlines()
                    except (OSError, ValueError):
                        continue
                    file_rows = [line.split() for line in lines if line.split()]
                    if not file_rows:
                        continue
                    sampled_nonempty_files += 1
                    for tokens in file_rows:
                        rows.append(tokens)
                        if len(rows) >= limit:
                            break
            if count_limit is not None and label_count >= count_limit:
                return rows, count_limit, False
    return rows, label_count, True


def inspect_config_dataset(
    config_path: Path,
    *,
    show_progress: bool = True,
    count_limit: int | None = None,
) -> DatasetCandidate:
    config_path = config_path.resolve()
    config = load_yaml(config_path)
    root = dataset_root_from_config(config_path, config)
    names = class_names(config, root)
    image_dirs = split_image_dirs(root, config)
    configured_image_files = (
        split_image_files(root, config) if count_limit is None else {}
    )
    splits = tuple(
        split for split in SPLITS if split in image_dirs or split in configured_image_files
    )
    if count_limit is None:
        image_count = sum(len(paths) for paths in configured_image_files.values())
        image_count_exact = True
    else:
        image_count, image_count_exact = _count_files_limited(
            image_dirs.values(), IMAGE_EXTENSIONS, count_limit
        )

    mask_dirs = split_annotation_dirs(root, config, image_dirs, annotation="masks")
    semantic = bool(mask_dirs) or any(
        isinstance(config.get(split), dict) and config[split].get("masks")
        for split in SPLITS
    )
    if semantic:
        mask_count, mask_count_exact = _count_files_limited(
            mask_dirs.values(), MASK_EXTENSIONS, count_limit
        )
        semantic_issues: list[str] = []
        if image_count == 0:
            semantic_issues.append("所选 split 中没有图片")
        if mask_count == 0:
            semantic_issues.append("所选 split 中没有 mask")
        return DatasetCandidate(
            path=root,
            kind="semantic_mask",
            image_count=image_count,
            annotation_count=mask_count,
            class_count=len(names),
            splits=splits,
            config_path=config_path,
            issues=tuple(semantic_issues),
            modified_time=_latest_mtime(
                [
                    config_path,
                    root / "classes.txt",
                    *image_dirs.values(),
                    *mask_dirs.values(),
                ]
            ),
            counts_are_exact=image_count_exact and mask_count_exact,
            image_count_is_exact=image_count_exact,
            annotation_count_is_exact=mask_count_exact,
        )

    task = str(config.get("task", "")).casefold()
    if _declared_task_kind(config) == "image_classification":
        return DatasetCandidate(
            path=root,
            kind="image_classification",
            image_count=image_count,
            class_count=len(names),
            splits=splits,
            config_path=config_path,
            issues=("所选 split 中没有图片",) if image_count == 0 else (),
            modified_time=_latest_mtime(
                [config_path, root / "classes.txt", *image_dirs.values()]
            ),
            counts_are_exact=image_count_exact,
            image_count_is_exact=image_count_exact,
        )

    label_dirs = split_annotation_dirs(
        root,
        config,
        image_dirs,
        annotation="labels",
        include_missing=True,
    )
    rows, annotation_count, annotation_count_exact = _sample_yolo_lines(
        label_dirs.values(),
        count_limit=count_limit,
        show_progress=show_progress,
    )
    polygons = sum(1 for row in rows if len(row) >= 7 and len(row) % 2 == 1)
    boxes = sum(1 for row in rows if len(row) == 5)
    kind = "unknown"
    if polygons:
        kind = "yolo_instance"
    elif boxes:
        kind = "yolo_detection"
    elif task in {"segment", "seg", "instance", "instance_segmentation"}:
        kind = "yolo_instance"
    elif task in {"det", "detect", "detection", "object_detection"}:
        kind = "yolo_detection"
    issues: list[str] = []
    if not image_dirs:
        issues.append("缺少可解析的 images split")
    elif image_count == 0:
        issues.append("所选 split 中没有图片")
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
        modified_time=_latest_mtime(
            [
                config_path,
                root / "classes.txt",
                *image_dirs.values(),
                *label_dirs.values(),
            ]
        ),
        counts_are_exact=image_count_exact and annotation_count_exact,
        image_count_is_exact=image_count_exact,
        annotation_count_is_exact=annotation_count_exact,
    )


def _inspect_unconfigured_dir(
    path: Path,
    filenames: set[str],
    dirnames: set[str],
    *,
    count_limit: int | None = 200,
) -> DatasetCandidate | None:
    direct_images = [
        path / name
        for name in filenames
        if Path(name).suffix.lower() in IMAGE_EXTENSIONS
    ]
    direct_json = [
        path / name for name in filenames if Path(name).suffix.lower() == ".json"
    ]
    image_count = len(direct_images)
    json_count = len(direct_json)
    counts_are_exact = True
    images_exact = True
    annotations_exact = True

    # Support LabelMe exports with separated images/labels directories and
    # split-first layouts such as train/{images,labels}.
    image_dir = next(
        (path / name for name in dirnames if name.casefold() in {"images", "image"}),
        None,
    )
    annotation_dir = next(
        (
            path / name
            for name in dirnames
            if name.casefold() in {"labels", "label", "json", "annotations"}
        ),
        None,
    )
    if image_dir is not None and annotation_dir is not None:
        image_count, images_exact = _count_files_limited(
            [image_dir], IMAGE_EXTENSIONS, count_limit
        )
        json_count, annotations_exact = _count_files_limited(
            [annotation_dir], {".json"}, count_limit
        )
        counts_are_exact = images_exact and annotations_exact
    elif any(name.casefold() in SPLITS for name in dirnames):
        split_image_dirs: list[Path] = []
        split_annotation_dirs: list[Path] = []
        for name in dirnames:
            if name.casefold() not in SPLITS:
                continue
            split_root = path / name
            split_image_dir = next(
                (
                    split_root / child
                    for child in ("images", "image")
                    if (split_root / child).is_dir()
                ),
                None,
            )
            split_annotation_dir = next(
                (
                    split_root / child
                    for child in ("labels", "label", "json", "annotations")
                    if (split_root / child).is_dir()
                ),
                None,
            )
            if split_image_dir is not None and split_annotation_dir is not None:
                split_image_dirs.append(split_image_dir)
                split_annotation_dirs.append(split_annotation_dir)
        if split_image_dirs and split_annotation_dirs:
            image_count, images_exact = _count_files_limited(
                split_image_dirs, IMAGE_EXTENSIONS, count_limit
            )
            json_count, annotations_exact = _count_files_limited(
                split_annotation_dirs, {".json"}, count_limit
            )
            counts_are_exact = images_exact and annotations_exact

    if json_count and image_count:
        return DatasetCandidate(
            path=path.resolve(),
            kind="labelme",
            image_count=image_count,
            annotation_count=json_count,
            modified_time=_latest_mtime(
                [path, image_dir or path, annotation_dir or path]
            ),
            counts_are_exact=counts_are_exact,
            image_count_is_exact=images_exact,
            annotation_count_is_exact=annotations_exact,
        )
    return None


def _inspect_classification_dir(
    path: Path,
    dirnames: set[str],
    *,
    count_limit: int | None = 200,
) -> DatasetCandidate | None:
    """Recognize an ImageFolder dataset rooted at train/val class directories."""
    by_lower = {name.casefold(): name for name in dirnames}
    train_name = next(
        (by_lower[name] for name in ("train", "training") if name in by_lower),
        None,
    )
    val_name = next(
        (by_lower[name] for name in ("val", "valid", "validation") if name in by_lower),
        None,
    )
    if train_name is None or val_name is None:
        return None
    train_root = path / train_name
    val_root = path / val_name
    train_classes = {child.name for child in train_root.iterdir() if child.is_dir()}
    val_classes = {child.name for child in val_root.iterdir() if child.is_dir()}
    classes = train_classes | val_classes
    if not classes:
        return None
    content_dirnames = {
        "image",
        "images",
        "label",
        "labels",
        "mask",
        "masks",
        "annotation",
        "annotations",
    }
    if {name.casefold() for name in classes} & content_dirnames:
        return None
    image_count, exact = _count_files_limited(
        [train_root, val_root], IMAGE_EXTENSIONS, count_limit
    )
    if image_count <= 0:
        return None
    issues = () if train_classes == val_classes else ("train/val 类别目录不一致",)
    return DatasetCandidate(
        path=path.resolve(),
        kind="image_classification",
        image_count=image_count,
        class_count=len(classes),
        splits=("train", "val"),
        issues=issues,
        modified_time=_latest_mtime([path, train_root, val_root]),
        counts_are_exact=exact,
        image_count_is_exact=exact,
    )


def _dataset_config_paths(
    path: Path,
    filenames: set[str],
    *,
    require_dataset_evidence: bool = False,
) -> list[Path]:
    """Return standard and content-identified dataset YAML files in priority order."""
    yaml_names = sorted(
        (
            name
            for name in filenames
            if Path(name).suffix.casefold() in {".yaml", ".yml"}
        ),
        key=str.casefold,
    )
    standard_rank = {
        name: index for index, name in enumerate(STANDARD_CONFIG_NAMES)
    }
    result: list[tuple[int, str, Path]] = []
    for name in yaml_names:
        config_path = path / name
        lowered = name.casefold()
        config = load_yaml(config_path)
        has_split = any(
            config.get(alias) is not None
            for split in SPLITS
            for alias in SPLIT_ALIASES[split]
        )
        has_default_layout = any(
            (path / "images" / alias).is_dir()
            or (path / alias / "images").is_dir()
            for split in SPLITS
            for alias in SPLIT_ALIASES[split]
        )
        is_standard = lowered in standard_rank
        if require_dataset_evidence and not (has_split or has_default_layout):
            continue
        if not is_standard and not has_split and not has_default_layout:
            continue
        rank = standard_rank.get(lowered, len(STANDARD_CONFIG_NAMES))
        result.append((rank, lowered, config_path))
    return [item[2] for item in sorted(result)]


def find_dataset_config(source: Path) -> Path:
    """Resolve a standard or content-identified dataset YAML in one directory."""
    source = source.expanduser().resolve()
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(f"数据集路径不存在: {source}")
    try:
        filenames = {path.name for path in source.iterdir() if path.is_file()}
    except OSError as exc:
        raise FileNotFoundError(f"无法读取数据集目录: {source}") from exc
    candidates = _dataset_config_paths(source, filenames)
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError(f"数据集配置不存在: {source}")


def scan_generated_mask_roots(
    search_root: Path, *, show_progress: bool = True
) -> list[Path]:
    """Group ``root/object/defect/{image,fg}`` leaves into reusable source roots."""
    return [
        candidate.path
        for candidate in scan_datasets(search_root, show_progress=show_progress)
        if candidate.kind == "generated_mask"
    ]


def _generated_candidate(
    path: Path, *, show_progress: bool = True
) -> DatasetCandidate:
    image_count = 0
    annotation_count = 0
    walker = tqdm(
        path.rglob("image"),
        desc=f"分析 {path.name}/image/fg",
        unit="目录",
        disable=not show_progress,
        leave=False,
    )
    for image_dir in walker:
        fg_dir = image_dir.parent / "fg"
        if not image_dir.is_dir() or not fg_dir.is_dir():
            continue
        images = {item.stem for item in _iter_images(image_dir)}
        masks = {item.stem for item in _iter_images(fg_dir)}
        image_count += len(images)
        annotation_count += len(images & masks)
        walker.set_postfix_str(
            f"图片={image_count} 配对={annotation_count}", refresh=False
        )
    return DatasetCandidate(
        path=path.resolve(),
        kind="generated_mask",
        image_count=image_count,
        annotation_count=annotation_count,
    )


def _generated_group_root(search_root: Path, leaf: Path) -> Path:
    """Return the dataset root for one ``object/defect/{image,fg}`` leaf."""
    grouped = leaf.parent.parent
    try:
        grouped.relative_to(search_root)
    except ValueError:
        return leaf.resolve()
    return grouped.resolve()


def _direct_image_stems(directory: Path) -> set[str]:
    try:
        entries = directory.iterdir()
        return {
            path.stem
            for path in entries
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
    except OSError:
        return set()


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
        paths_for_mtime: list[Path] = [root]
        for category in categories:
            for split in ("train", "test"):
                split_dir = category / split
                if split_dir.is_dir():
                    split_files = [
                        path
                        for path in split_dir.rglob("*")
                        if path.is_file()
                        and path.suffix.lower() in IMAGE_EXTENSIONS
                    ]
                    image_count += len(split_files)
                    paths_for_mtime.append(split_dir)
            ground_truth = category / "ground_truth"
            if ground_truth.is_dir():
                ground_truth_files = [
                    path
                    for path in ground_truth.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ]
                annotation_count += len(ground_truth_files)
                paths_for_mtime.append(ground_truth)
                defects.update(
                    path.name for path in ground_truth.iterdir() if path.is_dir()
                )
        candidates.append(
            DatasetCandidate(
                path=root.resolve(),
                kind="mvtec",
                image_count=image_count,
                annotation_count=annotation_count,
                class_count=len(defects),
                splits=("train", "test"),
                modified_time=_latest_mtime(paths_for_mtime),
            )
        )
    return candidates


def scan_mvtec_datasets(
    search_root: Path, *, show_progress: bool = True
) -> list[DatasetCandidate]:
    """Find MVTec roots without descending into each category's image tree."""
    search_root = search_root.expanduser().resolve()
    if not search_root.is_dir():
        return []
    category_roots: set[Path] = set()
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", "out", "weights"}
    walker = tqdm(
        os.walk(search_root),
        desc="检测 MVTec 数据集",
        unit="目录",
        disable=not show_progress,
        leave=False,
    )
    for current, dirnames, _ in walker:
        dirnames[:] = [
            name
            for name in dirnames
            if name.casefold() not in skipped_dirs and not name.startswith(".")
        ]
        lowered = {name.casefold() for name in dirnames}
        walker.set_postfix_str(
            f"当前={Path(current).name or search_root.name} 已发现={len(category_roots)}",
            refresh=False,
        )
        if "test" in lowered and "ground_truth" in lowered:
            category_roots.add(Path(current).resolve())
            dirnames[:] = []
    return sort_candidates_by_modified(
        _mvtec_candidates(search_root, category_roots)
    )


def scan_datasets(
    search_root: Path,
    *,
    show_progress: bool = True,
    count_limit: int | None = 200,
    kinds: Iterable[str] | None = None,
) -> list[DatasetCandidate]:
    """Find datasets with bounded counting by default for responsive menus."""
    search_root = search_root.expanduser().resolve()
    if not search_root.is_dir():
        return []

    candidates: list[DatasetCandidate] = []
    configured_roots: set[Path] = set()
    mvtec_category_roots: set[Path] = set()
    mvtec_counts: dict[Path, dict[str, Any]] = {}
    generated_counts: dict[Path, dict[str, float | int]] = {}
    unconfigured: list[DatasetCandidate] = []
    visited_directories: set[tuple[int, int]] = set()
    skipped_dirs = {".git", "__pycache__", ".pytest_cache", "out", "weights"}
    # A configured dataset can live above another configured dataset, for
    # example ``project/data.yaml`` and ``project/dataset_det/data.yaml``.
    # Keep walking project-like children after finding a config, while pruning
    # the large content trees already inspected through that config.
    configured_content_dirs = {
        "image",
        "images",
        "label",
        "labels",
        "mask",
        "masks",
        "annotation",
        "annotations",
        "ground_truth",
        *(alias for aliases in SPLIT_ALIASES.values() for alias in aliases),
    }
    wanted_kinds = set(kinds) if kinds is not None else None
    yolo_scan_only = wanted_kinds is not None and wanted_kinds.issubset(
        {
            "yolo_detection",
            "yolo_instance",
            "semantic_mask",
            "image_classification",
        }
    )
    inspect_labelme = wanted_kinds is None or "labelme" in wanted_kinds
    inspect_classification = (
        wanted_kinds is None or "image_classification" in wanted_kinds
    )
    inspect_generated = wanted_kinds is None or "generated_mask" in wanted_kinds
    inspect_mvtec = wanted_kinds is None or "mvtec" in wanted_kinds

    walker = tqdm(
        os.walk(search_root, followlinks=True),
        desc=f"检测数据集 {search_root.name}",
        unit="目录",
        disable=not show_progress,
        leave=False,
    )
    for current, dirnames, filenames_list in walker:
        try:
            stat_result = Path(current).stat()
            directory_identity = (stat_result.st_dev, stat_result.st_ino)
        except OSError:
            dirnames[:] = []
            continue
        if directory_identity in visited_directories:
            dirnames[:] = []
            continue
        visited_directories.add(directory_identity)
        dirnames[:] = [
            name
            for name in dirnames
            if name.casefold() not in skipped_dirs and not name.startswith(".")
        ]
        path = Path(current).resolve()
        filenames = set(filenames_list)
        dir_by_lower = {name.casefold(): name for name in dirnames}
        try:
            current_display = path.relative_to(search_root).as_posix() or "."
        except ValueError:
            current_display = path.name
        walker.set_postfix_str(
            f"目录={len(visited_directories)} 发现={len(candidates) + len(unconfigured)} 当前={current_display[-48:]}",
            refresh=False,
        )
        if (
            inspect_mvtec
            and "test" in dir_by_lower
            and "ground_truth" in dir_by_lower
        ):
            mvtec_category_roots.add(path)
            image_count, image_count_exact = _count_files_limited(
                [path / "train", path / "test"],
                IMAGE_EXTENSIONS,
                count_limit,
            )
            annotation_count, annotation_count_exact = _count_files_limited(
                [path / "ground_truth"],
                IMAGE_EXTENSIONS,
                count_limit,
            )
            mvtec_counts.setdefault(
                path,
                {
                    "image_count": image_count,
                    "annotation_count": annotation_count,
                    "defects": {
                        child.name
                        for child in (path / "ground_truth").iterdir()
                        if child.is_dir()
                    },
                    "modified_time": _safe_mtime(path),
                    "counts_are_exact": (
                        image_count_exact and annotation_count_exact
                    ),
                    "image_count_is_exact": image_count_exact,
                    "annotation_count_is_exact": annotation_count_exact,
                },
            )
            dirnames[:] = []
            continue

        # Count generated image/fg pairs while this same os.walk is already at
        # their parent. This replaces the former full-tree rglob plus a second
        # rglob for every grouped generated dataset.
        if inspect_generated and "image" in dir_by_lower and "fg" in dir_by_lower:
            image_dir = path / dir_by_lower["image"]
            mask_dir = path / dir_by_lower["fg"]
            grouped = _generated_group_root(search_root, path)
            counts = generated_counts.setdefault(
                grouped,
                {
                    "image_count": 0,
                    "annotation_count": 0,
                    "modified_time": 0.0,
                    "counts_are_exact": True,
                    "image_count_is_exact": True,
                    "annotation_count_is_exact": True,
                },
            )
            remaining_images = (
                None
                if count_limit is None
                else max(0, count_limit - int(counts["image_count"]))
            )
            remaining_masks = (
                None
                if count_limit is None
                else max(0, count_limit - int(counts["annotation_count"]))
            )
            image_count, image_exact = (
                (0, False)
                if remaining_images == 0
                else _count_files_limited(
                    [image_dir], IMAGE_EXTENSIONS, remaining_images
                )
            )
            mask_count, mask_exact = (
                (0, False)
                if remaining_masks == 0
                else _count_files_limited(
                    [mask_dir], IMAGE_EXTENSIONS, remaining_masks
                )
            )
            counts["image_count"] += image_count
            counts["annotation_count"] += mask_count
            counts["counts_are_exact"] = bool(counts["counts_are_exact"]) and (
                image_exact and mask_exact
            )
            counts["image_count_is_exact"] = bool(
                counts["image_count_is_exact"]
            ) and image_exact
            counts["annotation_count_is_exact"] = bool(
                counts["annotation_count_is_exact"]
            ) and mask_exact
            counts["modified_time"] = max(
                float(counts["modified_time"]),
                _latest_mtime([path, image_dir, mask_dir]),
            )
            dirnames[:] = []
            continue
        config_paths = _dataset_config_paths(
            path,
            filenames,
            require_dataset_evidence=True,
        )
        if config_paths:
            for config_path in config_paths:
                candidate = inspect_config_dataset(
                    config_path,
                    show_progress=False,
                    count_limit=count_limit,
                )
                candidates.append(candidate)
                configured_roots.add(candidate.path)
            # The configured inspector has already visited image/annotation
            # trees. Other children may contain independent datasets and stay
            # in the walk.
            dirnames[:] = [
                name
                for name in dirnames
                if name.casefold() not in configured_content_dirs
            ]
            continue
        if inspect_classification:
            classification_candidate = _inspect_classification_dir(
                path,
                set(dirnames),
                count_limit=count_limit,
            )
            if classification_candidate is not None:
                unconfigured.append(classification_candidate)
                dirnames[:] = []
                continue
        if inspect_labelme:
            raw_candidate = _inspect_unconfigured_dir(
                path,
                filenames,
                set(dirnames),
                count_limit=count_limit,
            )
            if raw_candidate is not None:
                unconfigured.append(raw_candidate)
                dirnames[:] = []
                continue
        if yolo_scan_only and path.name.casefold() in configured_content_dirs:
            # Task-specific menus discover configured datasets. Once the walk
            # reaches a content tree without a config, its files cannot add a
            # candidate and the subtree can be skipped safely.
            dirnames[:] = []

    eligible_unconfigured: list[DatasetCandidate] = []
    for candidate in unconfigured:
        inside_configured_root = any(
            candidate.path == root or root in candidate.path.parents
            for root in configured_roots
        )
        if inside_configured_root:
            continue
        eligible_unconfigured.append(candidate)
    for candidate in eligible_unconfigured:
        inside_grouped_labelme = any(
            other.path in candidate.path.parents
            and candidate.path.relative_to(other.path).parts[0].casefold()
            in {*SPLITS, "images", "image", "labels", "label", "annotations", "json"}
            for other in eligible_unconfigured
            if other.path != candidate.path
        )
        if not inside_grouped_labelme:
            candidates.append(candidate)
    candidates.extend(
        DatasetCandidate(
            path=root,
            kind="generated_mask",
            image_count=int(counts["image_count"]),
            annotation_count=int(counts["annotation_count"]),
            modified_time=float(counts["modified_time"]),
            counts_are_exact=bool(counts["counts_are_exact"]),
            image_count_is_exact=bool(counts["image_count_is_exact"]),
            annotation_count_is_exact=bool(
                counts["annotation_count_is_exact"]
            ),
        )
        for root, counts in generated_counts.items()
    )

    grouped_mvtec: dict[Path, list[Path]] = {}
    if search_root in mvtec_category_roots:
        grouped_mvtec[search_root] = [search_root]
    for category in mvtec_category_roots - {search_root}:
        grouped_mvtec.setdefault(category.parent, []).append(category)
    for root, categories in grouped_mvtec.items():
        raw_image_count = sum(
            int(mvtec_counts[item]["image_count"]) for item in categories
        )
        raw_annotation_count = sum(
            int(mvtec_counts[item]["annotation_count"]) for item in categories
        )
        image_count_is_exact = all(
            bool(mvtec_counts[item]["image_count_is_exact"])
            for item in categories
        )
        annotation_count_is_exact = all(
            bool(mvtec_counts[item]["annotation_count_is_exact"])
            for item in categories
        )
        if count_limit is not None and raw_image_count > count_limit:
            raw_image_count = count_limit
            image_count_is_exact = False
        if count_limit is not None and raw_annotation_count > count_limit:
            raw_annotation_count = count_limit
            annotation_count_is_exact = False
        candidates.append(
            DatasetCandidate(
                path=root.resolve(),
                kind="mvtec",
                image_count=raw_image_count,
                annotation_count=raw_annotation_count,
                class_count=len(
                    set().union(*(mvtec_counts[item]["defects"] for item in categories))
                ),
                splits=("train", "test"),
                modified_time=max(
                    float(mvtec_counts[item]["modified_time"])
                    for item in categories
                ),
                counts_are_exact=(
                    image_count_is_exact and annotation_count_is_exact
                ),
                image_count_is_exact=image_count_is_exact,
                annotation_count_is_exact=annotation_count_is_exact,
            )
        )

    unique: dict[tuple[Path, str], DatasetCandidate] = {}
    for candidate in candidates:
        key = (candidate.path, candidate.kind)
        previous = unique.get(key)
        if previous is None:
            unique[key] = candidate
            continue
        previous_score = (
            not previous.issues,
            previous.image_count,
            previous.annotation_count,
            previous.modified_time,
        )
        candidate_score = (
            not candidate.issues,
            candidate.image_count,
            candidate.annotation_count,
            candidate.modified_time,
        )
        if candidate_score > previous_score:
            unique[key] = candidate
    known_roots = {
        candidate.path for candidate in unique.values() if candidate.kind != "unknown"
    }
    visible = [
        candidate
        for candidate in unique.values()
        if candidate.kind != "unknown" or candidate.path not in known_roots
    ]
    return sort_candidates_by_modified(visible)


def conversion_actions(candidate: DatasetCandidate) -> tuple[str, ...]:
    return {
        "yolo_instance": ("edit-classes", "to-semantic", "to-mvtec"),
        "yolo_detection": ("edit-classes",),
        "semantic_mask": ("edit-classes",),
        "labelme": ("to-yolo",),
        "generated_mask": ("to-yolo-seg",),
        "mvtec": ("mvtec-to-yolo",),
    }.get(candidate.kind, ())
