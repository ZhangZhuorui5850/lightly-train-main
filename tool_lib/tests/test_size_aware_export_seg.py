"""seg 尺寸感知导出端到端。"""
from __future__ import annotations

import json
from pathlib import Path

from tool_lib import common as rt
from tool_lib.seg_export import scan_candidate_size_buckets, export_filtered_dataset
from tool_lib.seg_shared import SegExportImageCandidate


def test_seg_scan_mask_area_buckets(tmp_path):
    rt.import_runtime_dependencies()
    img = tmp_path / "a.jpg"
    rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(img)
    # 多边形覆盖整图 → 面积≈10000 large；小三角 → tiny/small
    big = "0 0 0 1 0 1 1 0 1"          # 单位正方形，归一面积≈1 → 10000px large
    small = "0 0.5 0.5 0.55 0.5 0.5 0.55"  # 极小三角
    cand = SegExportImageCandidate(
        split_name="train",
        rel_split_image_dir=Path("images/train"),
        rel_split_label_dir=Path("labels/train"),
        rel_path=Path("a.jpg"),
        src_image_path=img,
        src_label_path=tmp_path / "a.txt",
        filtered_lines=(big, small),
        class_box_counts={0: 2},
    )
    out = scan_candidate_size_buckets([cand], {0: 0}, cache_dir=tmp_path)
    key = "a.jpg|train"
    assert out[key]["large"] == 1


def _make_seg_dataset(root: Path):
    """创建最小 seg 数据集：大多边形(large) + 小多边形(small)。"""
    rt.import_runtime_dependencies()
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    # 大多边形：覆盖整图 → 10000px → large
    big_poly = "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"
    # 小多边形：面积 ≈ 0.0025 → 25px → tiny
    small_poly = "0 0.49 0.49 0.51 0.49 0.51 0.51 0.49 0.51"
    for i in range(20):
        stem = f"big{i}"
        rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(root / "images" / "train" / f"{stem}.jpg")
        (root / "labels" / "train" / f"{stem}.txt").write_text(big_poly + "\n", encoding="utf-8")
    for i in range(20):
        stem = f"sml{i}"
        rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(root / "images" / "train" / f"{stem}.jpg")
        (root / "labels" / "train" / f"{stem}.txt").write_text(small_poly + "\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    rt.dump_yaml(data_yaml, {
        "path": str(root.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test",
        "task": "segment", "nc": 1, "names": ["obj"],
    })
    return data_yaml


def test_seg_size_aware_export_shifts_ratio(tmp_path):
    data_yaml = _make_seg_dataset(tmp_path / "ds")
    neutral = {"config": {"data_root": str((tmp_path / "ds").resolve())}, "summary": {}, "per_class_ap": {}}
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_A",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_instances=0, target_images_per_class=0,
        target_total_images=20, split_ratio="8:1:1", target_instances_per_class=0,
        max_instances_per_image=0, max_instances_per_class_per_image=0, instance_density_penalty=0.0,
        size_ratio="70:0:30", size_balance_weight=1.0,
        avg_instances_per_image_min=0, avg_instances_per_image_max=0,
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["size_selection_summary"] is not None
    assert summary["size_selection_summary"]["selection_algorithm"] == "size_aware_greedy"


def test_seg_size_ratio_empty_uses_legacy_path(tmp_path):
    data_yaml = _make_seg_dataset(tmp_path / "ds")
    neutral = {"config": {"data_root": str((tmp_path / "ds").resolve())}, "summary": {}, "per_class_ap": {}}
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_A",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_instances=0, target_images_per_class=0,
        target_total_images=10, split_ratio="8:1:1", target_instances_per_class=0,
        max_instances_per_image=0, max_instances_per_class_per_image=0, instance_density_penalty=0.0,
        size_ratio="",
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["selection_summary"]["selection_algorithm"] == "celf_lazy_greedy"
