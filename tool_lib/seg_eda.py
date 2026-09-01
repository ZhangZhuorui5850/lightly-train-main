"""Segmentation EDA router with automatic dataset-type detection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

from . import common as rt
from . import seg_instance_eda, seg_semantic_eda
from .seg_shared import read_yolo_seg_label_lines


SegmentationType = Literal["semantic", "instance"]

_SEMANTIC_TASK_NAMES = {
    "semantic",
    "semantic_segmentation",
    "semantic-segmentation",
}
_INSTANCE_TASK_NAMES = {
    "segment",
    "seg",
    "instance",
    "instance_segmentation",
    "instance-segmentation",
}


def _load_raw_config(data_path: Path) -> dict[str, Any]:
    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data config does not exist: {data_path}")
    if rt.yaml is None:
        rt.import_data_dependencies()
    with data_path.open("r", encoding="utf-8") as file:
        config = rt.yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid data config: {data_path}")
    return config


def _resolve_config_path(data_path: Path, config: dict[str, Any], value: Any) -> Path:
    root_value = config.get("path")
    if root_value is None:
        root = data_path.parent
    else:
        root = rt.resolve_data_yaml_path(Path(str(root_value)), base_dir=data_path.parent)
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _sample_has_polygon_labels(data_path: Path, config: dict[str, Any]) -> bool:
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not split_value:
            continue
        if isinstance(split_value, dict):
            image_value = split_value.get("images")
            label_value = split_value.get("labels")
        else:
            image_value = split_value
            label_value = None
        if image_value is None:
            continue
        image_dir = _resolve_config_path(data_path, config, image_value)
        if label_value is not None:
            label_dir = _resolve_config_path(data_path, config, label_value)
        else:
            parts = list(image_dir.parts)
            try:
                parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
                label_dir = Path(*parts)
            except ValueError:
                label_dir = image_dir.parent.parent / "labels" / image_dir.name
        if not label_dir.exists():
            continue
        for label_path in sorted(label_dir.rglob("*.txt"))[:50]:
            lines, _ = read_yolo_seg_label_lines(label_path)
            if lines:
                return True
    return False


def _has_semantic_mask_paths(data_path: Path, config: dict[str, Any]) -> bool:
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if not split_value:
            continue
        if isinstance(split_value, dict):
            mask_value = split_value.get("masks") or split_value.get("mask")
            if mask_value and _resolve_config_path(data_path, config, mask_value).exists():
                return True
            image_value = split_value.get("images")
        else:
            configured_mask = config.get(f"{split}_masks")
            if configured_mask and _resolve_config_path(data_path, config, configured_mask).exists():
                return True
            image_value = split_value
        if image_value is None:
            continue
        image_dir = _resolve_config_path(data_path, config, image_value)
        parts = list(image_dir.parts)
        try:
            parts[len(parts) - 1 - parts[::-1].index("images")] = "masks"
            mask_dir = Path(*parts)
        except ValueError:
            mask_dir = image_dir.parent.parent / "masks" / image_dir.name
        if mask_dir.exists() and any(mask_dir.rglob("*.png")):
            return True
    return False


def detect_segmentation_type(
    data_path: Path,
    requested_type: str = "auto",
) -> SegmentationType:
    """Detect PNG semantic masks or YOLO polygon instance labels."""
    requested = str(requested_type or "auto").strip().lower()
    if requested in {"semantic", "instance"}:
        return requested  # type: ignore[return-value]
    if requested != "auto":
        raise ValueError("seg_type must be one of: auto, semantic, instance")

    resolved_data_path = data_path.expanduser().resolve()
    config = _load_raw_config(resolved_data_path)
    # Split structure and real annotation files are the strongest signals. This also
    # handles stale or generic task fields in hand-edited YAML files.
    for split in ("train", "val", "test"):
        split_value = config.get(split)
        if isinstance(split_value, dict):
            if split_value.get("masks") or split_value.get("mask"):
                return "semantic"
            if split_value.get("labels"):
                return "instance"
        if config.get(f"{split}_masks"):
            return "semantic"

    if _has_semantic_mask_paths(resolved_data_path, config):
        return "semantic"
    if _sample_has_polygon_labels(resolved_data_path, config):
        return "instance"

    task_name = str(config.get("task", "")).strip().lower()
    if task_name in _SEMANTIC_TASK_NAMES:
        return "semantic"
    if task_name in _INSTANCE_TASK_NAMES:
        return "instance"

    raise ValueError(
        "无法自动识别分割数据集类型。请在 data.yaml 中设置 "
        "task: semantic_segmentation 或 task: segment，"
        "也可以通过 --seg-type semantic/instance 显式指定。"
    )


def run_seg_eda(args: argparse.Namespace) -> Path:
    data_path = Path(args.data).expanduser().resolve()
    seg_type = detect_segmentation_type(
        data_path,
        str(getattr(args, "seg_type", "auto") or "auto"),
    )
    print(f"[seg/eda] 自动识别结果: {seg_type}")

    common_kwargs = {
        "source_data_path": data_path,
        "output_dir": getattr(args, "output_dir", None),
        "overwrite": bool(getattr(args, "overwrite", False)),
        "min_class_images": int(getattr(args, "min_class_images", 10)),
        "threshold_percentile": float(getattr(args, "threshold_percentile", 0.9)),
    }
    if seg_type == "semantic":
        return seg_semantic_eda.generate_semantic_eda_report(**common_kwargs)
    return seg_instance_eda.generate_instance_eda_report(**common_kwargs)


__all__ = ["detect_segmentation_type", "run_seg_eda"]
