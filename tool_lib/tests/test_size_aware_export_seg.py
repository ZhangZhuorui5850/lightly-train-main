"""seg 尺寸感知导出端到端。"""
from __future__ import annotations

from pathlib import Path

from tool_lib import common as rt
from tool_lib.seg_export import scan_candidate_size_buckets
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
