from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import coco_to_synced as converter  # noqa: E402


def test_ratio_counts_cover_positive_splits_when_sample_count_allows() -> None:
    assert converter._ratio_counts(3, (0.8, 0.1, 0.1)) == (1, 1, 1)
    assert converter._ratio_counts(10, (1.0, 0.0, 0.0)) == (10, 0, 0)


def test_stratified_split_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="总和等于 1"):
        converter.stratified_split({1: {0}}, (0.8, 0.3, -0.1), seed=1)
