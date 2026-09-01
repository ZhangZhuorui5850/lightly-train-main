#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert generated ``<object>/<defect>/{image,fg}`` pairs to YOLO-seg.

``image`` contains generated RGB images. ``fg`` contains PNG defect masks with
matching stems. Defect folder names become YOLO classes; object folder names
are retained in output filenames to prevent collisions such as repeated
``0.png`` files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        sys.exit(f"缺少依赖 {exc.name}，请在项目训练环境中运行。当前 Python: {sys.executable}")

try:
    from .dataset_transaction import staged_output, validate_output_location
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    from dataset_transaction import (  # type: ignore[no-redef]
        staged_output,
        validate_output_location,
    )
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Pair:
    object_name: str
    defect_name: str
    image: Path
    mask: Path
    leaf: Path


@dataclass(frozen=True)
class ScanIssue:
    status: str
    object_name: str
    defect_name: str
    stem: str
    image: str = ""
    mask: str = ""
    detail: str = ""


def natural_key(value: str) -> list[object]:
    """Human-friendly stable order: class2 precedes class10."""
    return [int(p) if p.isdigit() else p.casefold() for p in re.split(r"(\d+)", value)]


def _files_by_stem(directory: Path) -> tuple[dict[str, Path], set[str]]:
    grouped: dict[str, list[Path]] = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            grouped.setdefault(path.stem, []).append(path)
    unique = {stem: paths[0] for stem, paths in grouped.items() if len(paths) == 1}
    duplicates = {stem for stem, paths in grouped.items() if len(paths) > 1}
    return unique, duplicates


def discover_pairs(src: Path) -> tuple[list[Pair], list[ScanIssue]]:
    """Discover every leaf containing sibling ``image/`` and ``fg/`` dirs."""
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise ValueError(f"输入目录不存在: {src}")

    image_dirs: list[Path] = []
    if (src / "image").is_dir() and (src / "fg").is_dir():
        image_dirs.append(src / "image")
    image_dirs.extend(
        p for p in src.rglob("image")
        if p.is_dir() and (p.parent / "fg").is_dir() and p not in image_dirs
    )

    pairs: list[Pair] = []
    issues: list[ScanIssue] = []
    for image_dir in tqdm(
        sorted(image_dirs),
        desc="检测 image/fg 数据",
        unit="目录",
        leave=False,
    ):
        leaf = image_dir.parent
        fg_dir = leaf / "fg"
        defect = leaf.name
        obj = leaf.parent.name
        images, dup_images = _files_by_stem(image_dir)
        masks, dup_masks = _files_by_stem(fg_dir)

        for stem in sorted(dup_images | dup_masks, key=natural_key):
            issues.append(ScanIssue(
                "duplicate_stem", obj, defect, stem,
                detail="同一目录中存在相同 stem 的多个扩展名文件",
            ))
        for stem in sorted(images.keys() - masks.keys(), key=natural_key):
            issues.append(ScanIssue(
                "missing_mask", obj, defect, stem, image=str(images[stem]),
                detail="image 中存在图片，fg 中缺少同 stem 掩码",
            ))
        for stem in sorted(masks.keys() - images.keys(), key=natural_key):
            issues.append(ScanIssue(
                "missing_image", obj, defect, stem, mask=str(masks[stem]),
                detail="fg 中存在掩码，image 中缺少同 stem 图片",
            ))
        for stem in sorted(images.keys() & masks.keys(), key=natural_key):
            if stem in dup_images or stem in dup_masks:
                continue
            pairs.append(Pair(obj, defect, images[stem], masks[stem], leaf))
    return pairs, issues


def _names_from_config(path: Path) -> list[str]:
    config = path
    if path.is_dir():
        yaml_path = path / "data.yaml"
        classes_path = path / "classes.txt"
        config = yaml_path if yaml_path.is_file() else classes_path
    if not config.is_file():
        return []
    if config.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(read_text_auto(config)) or {}
        raw = data.get("names", data.get("classes"))
        if isinstance(raw, dict):
            return [str(v).strip() for _, v in sorted(raw.items(), key=lambda kv: int(kv[0]))]
        if isinstance(raw, list):
            return [str(v).strip() for v in raw]
        return []
    return [line.strip() for line in read_text_auto(config).splitlines() if line.strip()]


def _automatic_config(src: Path, defects: set[str]) -> Path | None:
    """Find a local config only when it shares at least one inferred class."""
    candidates: list[Path] = []
    for name in ("data.yaml", "classes.txt"):
        direct = src / name
        if direct.is_file():
            candidates.append(direct)
    for path in src.rglob("data.yaml"):
        if path not in candidates:
            candidates.append(path)
    for path in src.rglob("classes.txt"):
        if path not in candidates:
            candidates.append(path)

    ranked: list[tuple[int, int, Path]] = []
    for path in candidates:
        names = _names_from_config(path)
        overlap = len(defects.intersection(names))
        if overlap:
            ranked.append((-overlap, len(names), path))
    return sorted(ranked, key=lambda row: (row[0], row[1], str(row[2])))[0][2] if ranked else None


def resolve_class_names(
    src: Path,
    pairs: list[Pair],
    names_from: Path | None = None,
) -> tuple[list[str], Path | None]:
    """Return contiguous class names, preserving a useful config order."""
    defects = {pair.defect_name for pair in pairs}
    config = names_from.expanduser().resolve() if names_from else _automatic_config(src, defects)
    configured = _names_from_config(config) if config else []
    names: list[str] = []
    for name in configured:
        if name and name not in names:
            names.append(name)
    for name in sorted(defects, key=natural_key):
        if name not in names:
            names.append(name)
    return names, config


def mask_to_polygons(
    mask: np.ndarray,
    threshold: int = 127,
    min_area: float = 1.0,
    epsilon: float = 0.001,
    auto_invert: bool = True,
) -> tuple[list[np.ndarray], bool, float]:
    """Convert a grayscale/RGB mask to external polygon contours."""
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    inverted = False
    if auto_invert and binary.size:
        border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
        if float(np.count_nonzero(border)) / float(border.size) > 0.5:
            binary = cv2.bitwise_not(binary)
            inverted = True

    foreground_ratio = float(np.count_nonzero(binary)) / float(binary.size) if binary.size else 0.0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        arc = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * arc, True) if epsilon > 0 else contour
        points = approx.reshape(-1, 2)
        if len(points) >= 3:
            polygons.append(points)
    polygons.sort(key=lambda poly: (-cv2.contourArea(poly.reshape(-1, 1, 2)), int(poly[:, 1].min()), int(poly[:, 0].min())))
    return polygons, inverted, foreground_ratio


def _safe_component(value: str) -> str:
    value = re.sub(r"[\\/\s]+", "_", value.strip())
    value = re.sub(r"[^\w.()\-\u4e00-\u9fff]+", "_", value, flags=re.UNICODE)
    return value.strip("._") or "unknown"


def _output_stem(pair: Pair, src: Path, used: set[str]) -> str:
    base = "__".join(map(_safe_component, (pair.object_name, pair.defect_name, pair.image.stem)))
    if base not in used:
        used.add(base)
        return base
    rel = str(pair.image.relative_to(src)) if pair.image.is_relative_to(src) else str(pair.image)
    suffix = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}__{suffix}"
    counter = 2
    while candidate in used:
        candidate = f"{base}__{suffix}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _write_dataset_metadata(out: Path, names: list[str], mapping_sources: dict[str, set[str]]) -> None:
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": "segment",
        "nc": len(names),
        "names": {i: name for i, name in enumerate(names)},
    }
    (out / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (out / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    mapping = {
        str(i): {"name": name, "sources": sorted(mapping_sources.get(name, set()))}
        for i, name in enumerate(names)
    }
    (out / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


REPORT_FIELDS = (
    "status", "object", "defect", "class_id", "stem", "source_image", "source_mask",
    "output_image", "output_label", "polygons", "foreground_ratio", "inverted", "detail",
)


def _convert_unpublished(
    src: Path,
    out: Path,
    *,
    names_from: Path | None = None,
    threshold: int = 127,
    min_area: float = 1.0,
    epsilon: float = 0.001,
    auto_invert: bool = True,
    verbose: bool = True,
    report_output: Path | None = None,
) -> dict:
    """Convert all discovered generated image/mask pairs into one YOLO-seg dataset."""
    src, out = src.expanduser().resolve(), out.expanduser().resolve()
    if not 0 <= threshold <= 255:
        raise ValueError("threshold 必须位于 0..255")
    if min_area < 0 or epsilon < 0:
        raise ValueError("min_area 和 epsilon 必须大于等于 0")

    pairs, scan_issues = discover_pairs(src)
    if not pairs:
        raise ValueError(f"未发现可配对的 image/fg 数据: {src}")
    names, config_source = resolve_class_names(src, pairs, names_from)
    class_ids = {name: i for i, name in enumerate(names)}

    published_root = report_output or out
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    mapping_sources: dict[str, set[str]] = {name: set() for name in names}
    rows: list[dict[str, object]] = []
    for issue in scan_issues:
        rows.append({
            "status": issue.status, "object": issue.object_name, "defect": issue.defect_name,
            "class_id": "", "stem": issue.stem, "source_image": issue.image,
            "source_mask": issue.mask, "output_image": "", "output_label": "",
            "polygons": 0, "foreground_ratio": "", "inverted": "", "detail": issue.detail,
        })

    used_stems: set[str] = set()
    converted = empty_masks = skipped = 0
    for pair in tqdm(
        pairs,
        desc=f"转换 {src.name}",
        unit="图片",
        disable=not verbose,
    ):
        class_id = class_ids[pair.defect_name]
        mapping_sources[pair.defect_name].add(f"{pair.object_name}/{pair.defect_name}")
        image = cv2.imread(str(pair.image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(pair.mask), cv2.IMREAD_UNCHANGED)
        status, detail = "converted", ""
        polygons: list[np.ndarray] = []
        inverted = False
        ratio = 0.0
        if image is None or mask is None:
            status, detail = "unreadable", "图片或掩码读取失败"
        elif image.shape[:2] != mask.shape[:2]:
            status = "size_mismatch"
            detail = f"image={image.shape[1]}x{image.shape[0]}, mask={mask.shape[1]}x{mask.shape[0]}"
        else:
            polygons, inverted, ratio = mask_to_polygons(
                mask, threshold=threshold, min_area=min_area,
                epsilon=epsilon, auto_invert=auto_invert,
            )

        output_image = output_label = ""
        if status == "converted":
            out_stem = _output_stem(pair, src, used_stems)
            image_dst = out / "images" / "train" / f"{out_stem}{pair.image.suffix.lower()}"
            label_dst = out / "labels" / "train" / f"{out_stem}.txt"
            shutil.copy2(pair.image, image_dst)
            h, w = image.shape[:2]
            lines = []
            for poly in polygons:
                coords: list[str] = []
                for x, y in poly:
                    nx = min(max(float(x) / float(w), 0.0), 1.0)
                    ny = min(max(float(y) / float(h), 0.0), 1.0)
                    coords.extend((f"{nx:.6f}", f"{ny:.6f}"))
                lines.append(f"{class_id} " + " ".join(coords))
            label_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            output_image = str(
                published_root / image_dst.relative_to(out)
            )
            output_label = str(
                published_root / label_dst.relative_to(out)
            )
            converted += 1
            if not polygons:
                status, detail = "empty_mask", "阈值处理后未提取到有效轮廓"
                empty_masks += 1
        else:
            skipped += 1

        rows.append({
            "status": status, "object": pair.object_name, "defect": pair.defect_name,
            "class_id": class_id, "stem": pair.image.stem, "source_image": str(pair.image),
            "source_mask": str(pair.mask), "output_image": output_image,
            "output_label": output_label, "polygons": len(polygons),
            "foreground_ratio": f"{ratio:.6f}", "inverted": inverted, "detail": detail,
        })

    _write_dataset_metadata(out, names, mapping_sources)
    with (out / "conversion_report.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "src": src, "out": published_root, "pairs": len(pairs), "converted": converted,
        "empty_masks": empty_masks, "skipped": skipped, "scan_issues": len(scan_issues),
        "names": names, "config_source": config_source,
    }
    if verbose:
        print(f"转换完成: {src} -> {published_root}")
        print(f"配对 {len(pairs)}，写入 {converted}，空掩码 {empty_masks}，跳过 {skipped}，配对问题 {len(scan_issues)}")
        print("类别: " + ", ".join(f"{i}={name}" for i, name in enumerate(names)))
        print(f"类别来源: {config_source or '根据缺陷目录自动生成'}")
        print(f"报告: {published_root / 'conversion_report.csv'}")
    return result


def convert(
    src: Path,
    out: Path,
    *,
    names_from: Path | None = None,
    threshold: int = 127,
    min_area: float = 1.0,
    epsilon: float = 0.001,
    auto_invert: bool = True,
    clean: bool = False,
    verbose: bool = True,
) -> dict:
    """Convert into a validated staging tree and publish it atomically."""
    source = src.expanduser().resolve()
    published_output = validate_output_location(out, [source])
    with staged_output(published_output, clean=clean) as stage:
        result = _convert_unpublished(
            source,
            stage,
            names_from=names_from,
            threshold=threshold,
            min_area=min_area,
            epsilon=epsilon,
            auto_invert=auto_invert,
            verbose=verbose,
            report_output=published_output,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="生成图片 + fg 缺陷掩码 -> YOLO Seg")
    parser.add_argument("--src", required=True, type=Path, help="包含 <物体>/<缺陷>/{image,fg} 的目录")
    parser.add_argument("--out", required=True, type=Path, help="YOLO Seg 输出目录")
    parser.add_argument("--names-from", type=Path, help="可选 data.yaml/classes.txt 类别顺序来源")
    parser.add_argument("--threshold", type=int, default=127, help="掩码二值阈值，默认 127")
    parser.add_argument("--min-area", type=float, default=1.0, help="最小轮廓面积，默认 1 像素")
    parser.add_argument("--epsilon", type=float, default=0.001, help="轮廓简化比例，默认 0.001")
    parser.add_argument("--no-auto-invert", action="store_true", help="关闭白底掩码自动反相")
    parser.add_argument("--clean", action="store_true", help="清空已有输出后重建")
    args = parser.parse_args()
    try:
        convert(
            args.src, args.out, names_from=args.names_from, threshold=args.threshold,
            min_area=args.min_area, epsilon=args.epsilon,
            auto_invert=not args.no_auto_invert, clean=args.clean,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
