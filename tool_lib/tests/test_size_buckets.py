"""4 档尺寸分桶边界测试。"""
from __future__ import annotations

from tool_lib.det_export import _bucket_box_area, _empty_size_bucket_counts
from tool_lib.seg_export import _bucket_mask_area
from tool_lib import det_size_supplement as ss


def test_bucket_boundaries_box():
    # 阈值：tiny<16²(256), small<32²(1024), medium<96²(9216), large≥96²
    assert _bucket_box_area(255.0) == "tiny"
    assert _bucket_box_area(256.0) == "small"       # 16²
    assert _bucket_box_area(1023.0) == "small"
    assert _bucket_box_area(1024.0) == "medium"     # 32²
    assert _bucket_box_area(9215.0) == "medium"
    assert _bucket_box_area(9216.0) == "large"      # 96²


def test_bucket_boundaries_mask():
    assert _bucket_mask_area(255.0) == "tiny"
    assert _bucket_mask_area(256.0) == "small"
    assert _bucket_mask_area(1024.0) == "medium"
    assert _bucket_mask_area(9216.0) == "large"


def test_empty_buckets_have_four_keys():
    assert list(_empty_size_bucket_counts().keys()) == ["tiny", "small", "medium", "large"]
    assert list(ss._empty_buckets().keys()) == ["tiny", "small", "medium", "large"]
    assert ss.BUCKET_NAMES == ("tiny", "small", "medium", "large")
