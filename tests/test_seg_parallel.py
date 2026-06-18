from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tool_lib import common as rt
from tool_lib import seg_shared
from tool_lib import seg_tools


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


def _reference_load(mask_path, classes, ignore_classes):
    # 旧实现逻辑的参考副本，用于等价性对拍
    with Image.open(mask_path) as mask_image:
        mask_np = np.array(mask_image)
    compare_np = mask_np if mask_np.ndim == 3 else mask_np[:, :, None]
    target = np.full(mask_np.shape[:2], -100, dtype=np.int64)
    original_to_internal = {
        cid: i for i, cid in enumerate(sorted(set(classes) - ignore_classes))
    }
    for cid, internal in original_to_internal.items():
        for label in seg_tools._class_labels(classes, cid):
            label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            target[np.all(compare_np == np.array(label_tuple), axis=2)] = internal
    return target


def test_load_semantic_mask_single_channel_matches_reference(tmp_path):
    rt.import_runtime_dependencies()
    classes = {0: "bg", 1: "cat", 2: "dog", 255: "ignore"}
    ignore = {255}
    arr = np.array([[0, 1, 2], [255, 1, 0]], dtype=np.uint8)
    p = tmp_path / "m.png"
    Image.fromarray(arr, mode="L").save(p)
    expected = _reference_load(p, classes, ignore)
    got = seg_tools._load_semantic_mask(p, classes, ignore)
    assert np.array_equal(got, expected)


def test_load_semantic_mask_rgb_labels_matches_reference(tmp_path):
    rt.import_runtime_dependencies()
    classes = {
        0: {"name": "bg", "labels": [[0, 0, 0]]},
        1: {"name": "road", "labels": [[128, 64, 128]]},
        2: {"name": "sky", "labels": [[70, 130, 180]]},
    }
    ignore = set()
    arr = np.array(
        [[[0, 0, 0], [128, 64, 128]], [[70, 130, 180], [0, 0, 0]]], dtype=np.uint8
    )
    p = tmp_path / "m.png"
    Image.fromarray(arr, mode="RGB").save(p)
    expected = _reference_load(p, classes, ignore)
    got = seg_tools._load_semantic_mask(p, classes, ignore)
    assert np.array_equal(got, expected)
