"""分割导出共用工具。

定义 seg 版本的数据类与标签 IO。

设计要点：数据类字段名故意与 det_shared 保持一致（class_box_counts、
filtered_lines、total_boxes 等），其中 class_box_counts 实际存放的是
"实例数"，filtered_lines 存放的是 polygon 标签行。这样 det_analysis 里
基于 duck typing 的阈值推导、候选筛选、平衡选图等函数都能直接复用，
无需另写一份近乎相同的实现。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

from . import common as rt
from .det_shared import (
    resolve_export_split_dirs,
    safe_class_name,
)


SEG_MIN_POLYGON_FIELDS = 7  # class_id + 3 vertices (6 floats)


@dataclass(frozen=True)
class SegSourceImageInfo:
    """与 det 版同名字段，class_box_counts 含义为"每类实例数"。"""

    split_name: str
    rel_split_image_dir: Path
    rel_split_label_dir: Path
    rel_path: Path
    src_image_path: Path
    src_label_path: Path
    label_lines: tuple[str, ...]
    class_box_counts: dict[int, int]


@dataclass(frozen=True)
class SegExportImageCandidate:
    """与 det 版同名字段，class_box_counts 含义为"每类实例数"。"""

    split_name: str
    rel_split_image_dir: Path
    rel_split_label_dir: Path
    rel_path: Path
    src_image_path: Path
    src_label_path: Path
    filtered_lines: tuple[str, ...]
    class_box_counts: dict[int, int]

    @property
    def total_boxes(self) -> int:
        return sum(self.class_box_counts.values())


def _is_valid_polygon_line(parts: list[str]) -> bool:
    if len(parts) < SEG_MIN_POLYGON_FIELDS:
        return False
    if (len(parts) - 1) % 2 != 0:
        return False
    if (len(parts) - 1) // 2 < 3:
        return False
    return True


def read_yolo_seg_label_lines(label_path: Path) -> tuple[tuple[str, ...], dict[int, int]]:
    """读取 YOLO 实例分割标签：每行 `class_id x1 y1 ... xN yN`。

    无效行（字段数过少或坐标数为奇数）直接丢弃。
    """
    if not label_path.exists():
        return (), {}
    valid_lines: list[str] = []
    class_instance_counts: Counter[int] = Counter()
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if not _is_valid_polygon_line(parts):
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        class_instance_counts[class_id] += 1
        valid_lines.append(" ".join(parts))
    return tuple(valid_lines), dict(class_instance_counts)


def remap_yolo_seg_label_lines(filtered_lines: tuple[str, ...], class_id_mapping: dict[int, int]) -> list[str]:
    """改写每行的 class_id，保留 polygon 坐标原文。"""
    remapped_lines: list[str] = []
    for line in filtered_lines:
        parts = line.split()
        old_class_id = int(float(parts[0]))
        parts[0] = str(class_id_mapping[old_class_id])
        remapped_lines.append(" ".join(parts))
    return remapped_lines


def polygon_area_normalized(coords: list[float]) -> float:
    """对归一化 polygon 用 shoelace 公式求面积（仍为归一化值）。"""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return 0.0
    area = 0.0
    n = len(coords) // 2
    for i in range(n):
        x1, y1 = coords[2 * i], coords[2 * i + 1]
        x2, y2 = coords[2 * ((i + 1) % n)], coords[2 * ((i + 1) % n) + 1]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) * 0.5


def polygon_bbox_normalized(coords: list[float]) -> tuple[float, float, float, float]:
    """返回归一化 polygon 的紧包围盒 (x_min, y_min, x_max, y_max)。"""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return 0.0, 0.0, 0.0, 0.0
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def parse_polygon_line(line: str) -> tuple[int, list[float]] | None:
    parts = line.split()
    if not _is_valid_polygon_line(parts):
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return class_id, coords


def scan_source_split(
    *,
    split_name: str,
    split_image_dir: Path,
    split_label_dir: Path,
    rel_split_image_dir: Path,
    rel_split_label_dir: Path,
) -> list[SegSourceImageInfo]:
    infos: list[SegSourceImageInfo] = []
    for rel_image in rt.file_helpers.list_image_filenames_from_dir(image_dir=split_image_dir):
        rel_path = Path(rel_image)
        src_image_path = split_image_dir / rel_path
        src_label_path = split_label_dir / rel_path.with_suffix(".txt")
        label_lines, class_instance_counts = read_yolo_seg_label_lines(src_label_path)
        infos.append(
            SegSourceImageInfo(
                split_name=split_name,
                rel_split_image_dir=rel_split_image_dir,
                rel_split_label_dir=rel_split_label_dir,
                rel_path=rel_path,
                src_image_path=src_image_path,
                src_label_path=src_label_path,
                label_lines=label_lines,
                class_box_counts=class_instance_counts,
            )
        )
    infos.sort(key=lambda item: item.rel_path.as_posix())
    return infos


def collect_source_image_infos(
    source_cfg: dict[str, Any],
    source_root: Path,
) -> tuple[dict[str, list[SegSourceImageInfo]], dict[str, str]]:
    source_infos_by_split: dict[str, list[SegSourceImageInfo]] = {}
    export_split_paths: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_value = source_cfg.get(split_name)
        if not split_value:
            continue
        split_image_dir, split_label_dir, _ = rt.resolve_dataset_split_paths(source_cfg, split_name)
        if split_label_dir is None or not split_image_dir.exists():
            continue
        rel_split_image_dir, rel_split_label_dir = resolve_export_split_dirs(
            split_name=split_name,
            source_root=source_root,
            split_image_dir=split_image_dir,
            split_label_dir=split_label_dir,
        )
        source_infos_by_split[split_name] = scan_source_split(
            split_name=split_name,
            split_image_dir=split_image_dir,
            split_label_dir=split_label_dir,
            rel_split_image_dir=rel_split_image_dir,
            rel_split_label_dir=rel_split_label_dir,
        )
        export_split_paths[split_name] = rel_split_image_dir.as_posix()
    return source_infos_by_split, export_split_paths


__all__ = [
    "SEG_MIN_POLYGON_FIELDS",
    "SegSourceImageInfo",
    "SegExportImageCandidate",
    "collect_source_image_infos",
    "parse_polygon_line",
    "polygon_area_normalized",
    "polygon_bbox_normalized",
    "read_yolo_seg_label_lines",
    "remap_yolo_seg_label_lines",
    "safe_class_name",
    "scan_source_split",
]
