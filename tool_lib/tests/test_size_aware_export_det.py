"""det 尺寸感知导出端到端 + 回归保护。"""
from __future__ import annotations

import json
from pathlib import Path

from tool_lib import common as rt
from tool_lib.det_export import export_filtered_dataset


def _make_dataset(root: Path):
    rt.import_runtime_dependencies()
    # 20 张大目标图 (1.0x1.0→10000px large) + 20 张小目标图 (0.2x0.2→400px small)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for i in range(20):
        stem = f"big{i}"
        rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(root / "images" / "train" / f"{stem}.jpg")
        (root / "labels" / "train" / f"{stem}.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    for i in range(20):
        stem = f"sml{i}"
        rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(root / "images" / "train" / f"{stem}.jpg")
        (root / "labels" / "train" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    rt.dump_yaml(data_yaml, {
        "path": str(root.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test",
        "task": "detect", "nc": 1, "names": ["obj"],
    })
    return data_yaml


def test_size_aware_export_shifts_small_ratio_up(tmp_path):
    data_yaml = _make_dataset(tmp_path / "ds")
    neutral = {"config": {"data_root": str((tmp_path / "ds").resolve())}, "summary": {}, "per_class_ap": {}}
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_A",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_boxes=0, target_images_per_class=0,
        target_total_images=20, split_ratio="8:1:1", target_boxes_per_class=0,
        max_boxes_per_image=0, max_boxes_per_class_per_image=0, box_density_penalty=0.0,
        size_ratio="70:0:30", size_balance_weight=1.0,
        avg_boxes_per_image_min=0, avg_boxes_per_image_max=0,
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
    achieved = summary["size_selection_summary"]["achieved_size_ratio"]
    assert achieved["small"] >= 0.5  # 目标偏小，实际小占比明显升高


def test_size_ratio_empty_uses_legacy_path(tmp_path):
    data_yaml = _make_dataset(tmp_path / "ds")
    neutral = {"config": {"data_root": str((tmp_path / "ds").resolve())}, "summary": {}, "per_class_ap": {}}
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_A",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_boxes=0, target_images_per_class=0,
        target_total_images=10, split_ratio="8:1:1", target_boxes_per_class=0,
        max_boxes_per_image=0, max_boxes_per_class_per_image=0, box_density_penalty=0.0,
        size_ratio="",
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["selection_summary"]["selection_algorithm"] == "celf_lazy_greedy"
