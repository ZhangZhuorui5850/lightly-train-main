from __future__ import annotations

import numpy as np

from tool_lib import seg_shared


def _roundtrip(mask):
    rle = seg_shared.rle_encode(mask)
    out = seg_shared.rle_decode(rle)
    assert out.dtype == bool
    assert out.shape == mask.shape
    assert np.array_equal(out, mask.astype(bool))


def test_rle_roundtrip_mixed():
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    _roundtrip(mask)


def test_rle_roundtrip_all_zero_and_all_one():
    _roundtrip(np.zeros((4, 5), dtype=bool))
    _roundtrip(np.ones((4, 5), dtype=bool))


def test_rle_roundtrip_non_square():
    rng = np.random.default_rng(0)
    _roundtrip(rng.integers(0, 2, size=(7, 13)).astype(bool))


def test_rle_size_field_matches_shape():
    mask = np.zeros((3, 8), dtype=bool)
    rle = seg_shared.rle_encode(mask)
    assert rle["size"] == [3, 8]
