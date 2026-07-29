import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from pytest import MonkeyPatch

import seg_copy_paste
from lightly_train._data.mask_semantic_segmentation_dataset import (
    MaskSemanticSegmentationDatasetArgs,
)


def _get_dataset(tmp_path: Path) -> seg_copy_paste.CopyPasteDataset:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    dataset = seg_copy_paste.CopyPasteDataset.__new__(
        seg_copy_paste.CopyPasteDataset
    )
    dataset.dataset_args = MaskSemanticSegmentationDatasetArgs(
        image_dir=image_dir,
        mask_dir_or_file=str(mask_dir),
        classes={
            0: {"name": "background", "labels": [0]},
            1: {"name": "one", "labels": [1]},
            2: {"name": "two", "labels": [2]},
        },
        ignore_classes=None,
        ignore_index=-100,
    )
    dataset.image_info = [
        {
            "image_filepaths": str(image_dir / "0.png"),
            "mask_filepaths": str(mask_dir / "0.png"),
        },
        {
            "image_filepaths": str(image_dir / "1.png"),
            "mask_filepaths": str(mask_dir / "1.png"),
        },
    ]
    dataset._cp_classes = {1, 2}
    dataset._cp_index = {}
    dataset.map_mask_labels_to_class_ids = lambda mask: mask  # type: ignore[method-assign]
    return dataset


def test_copy_paste_index_cache(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        seg_copy_paste.cache,
        "get_data_cache_dir",
        lambda: cache_dir,
    )
    monkeypatch.setitem(seg_copy_paste._CONFIG, "verbose", False)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    mask_0 = mask_dir / "0.png"
    mask_1 = mask_dir / "1.png"
    cv2.imwrite(str(mask_0), np.array([[0, 1]], dtype=np.uint8))
    cv2.imwrite(str(mask_1), np.array([[0, 2]], dtype=np.uint8))

    open_calls = 0
    original_open_mask = seg_copy_paste.file_helpers.open_mask_numpy

    def count_open_mask(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal open_calls
        open_calls += 1
        return original_open_mask(*args, **kwargs)

    monkeypatch.setattr(
        seg_copy_paste.file_helpers,
        "open_mask_numpy",
        count_open_mask,
    )

    first = _get_dataset(tmp_path)
    first._build_index()
    assert first._cp_index == {1: [0], 2: [1]}
    assert open_calls == 2
    assert len(list((cache_dir / "copy_paste").glob("*.json"))) == 1

    second = _get_dataset(tmp_path)
    second._build_index()
    assert second._cp_index == first._cp_index
    assert open_calls == 2

    cv2.imwrite(str(mask_0), np.array([[0, 2]], dtype=np.uint8))
    stat = mask_0.stat()
    os.utime(
        mask_0,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000),
    )

    third = _get_dataset(tmp_path)
    third._build_index()
    assert third._cp_index == {2: [0, 1]}
    assert open_calls == 4


def test_copy_paste_index_is_shared_across_ranks(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(
        seg_copy_paste.cache,
        "get_data_cache_dir",
        lambda: cache_dir,
    )
    monkeypatch.setitem(seg_copy_paste._CONFIG, "verbose", False)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    cv2.imwrite(
        str(mask_dir / "0.png"),
        np.array([[0, 1]], dtype=np.uint8),
    )
    cv2.imwrite(
        str(mask_dir / "1.png"),
        np.array([[0, 2]], dtype=np.uint8),
    )

    rank = 0
    barrier_calls = 0

    def is_rank_zero() -> bool:
        return rank == 0

    def barrier() -> None:
        nonlocal barrier_calls
        barrier_calls += 1

    monkeypatch.setattr(
        seg_copy_paste,
        "_distributed_is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(seg_copy_paste, "_is_global_rank_zero", is_rank_zero)
    monkeypatch.setattr(torch.distributed, "barrier", barrier)

    primary = _get_dataset(tmp_path)
    primary._build_index()

    rank = 1
    secondary = _get_dataset(tmp_path)
    secondary._build_index()

    assert primary._cp_index == {1: [0], 2: [1]}
    assert secondary._cp_index == primary._cp_index
    assert barrier_calls == 2
