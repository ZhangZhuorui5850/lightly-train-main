import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
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
    dataset._cp_on = True
    dataset._cp_valid_indices = None
    dataset._cp_bad_indices = []
    dataset._cp_bad_examples = []
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


def test_copy_paste_skips_and_caches_unreadable_masks(
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
    (mask_dir / "1.png").write_bytes(b"broken png")

    first = _get_dataset(tmp_path)
    first._build_index()

    assert first._cp_index == {1: [0]}
    assert first._cp_valid_indices == [0]
    assert first._cp_bad_indices == [1]
    assert len(first) == 1
    assert first._cp_bad_examples[0]["path"].endswith("1.png")

    second = _get_dataset(tmp_path)
    second._build_index()
    assert second._cp_index == first._cp_index
    assert second._cp_valid_indices == [0]
    assert second._cp_bad_indices == [1]


def test_copy_paste_resumes_from_partial_index(
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
    monkeypatch.setitem(seg_copy_paste._CONFIG, "index_checkpoint_interval", 1)

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

    original_open_mask = seg_copy_paste.file_helpers.open_mask_numpy
    calls = {"0.png": 0, "1.png": 0}
    interrupted = False

    def interrupt_once(*args: Any, **kwargs: Any) -> np.ndarray:
        nonlocal interrupted
        mask_path = Path(kwargs["mask_path"])
        calls[mask_path.name] += 1
        if mask_path.name == "1.png" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return original_open_mask(*args, **kwargs)

    monkeypatch.setattr(
        seg_copy_paste.file_helpers,
        "open_mask_numpy",
        interrupt_once,
    )

    first = _get_dataset(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        first._build_index()

    partial_files = list((cache_dir / "copy_paste").glob("*.partial.json"))
    assert len(partial_files) == 1

    resumed = _get_dataset(tmp_path)
    resumed._build_index()

    assert resumed._cp_index == {1: [0], 2: [1]}
    assert calls == {"0.png": 1, "1.png": 2}
    assert not partial_files[0].exists()


def test_copy_paste_pastes_only_selected_class(monkeypatch: MonkeyPatch) -> None:
    dataset = seg_copy_paste.CopyPasteDataset.__new__(
        seg_copy_paste.CopyPasteDataset
    )
    monkeypatch.setitem(seg_copy_paste._CONFIG, "min_area_frac", 0.0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "max_paste", 3)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "feather", 0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "background_classes", (0,))
    monkeypatch.setitem(seg_copy_paste._CONFIG, "max_target_overlap", 0.0)

    target_image = torch.zeros((3, 8, 8), dtype=torch.float32)
    target_mask = torch.zeros((8, 8), dtype=torch.long)
    source_image = torch.ones((3, 8, 8), dtype=torch.float32)
    source_mask = torch.zeros((8, 8), dtype=torch.long)
    source_mask[0, 0] = 1
    source_mask[0, 2] = 1
    source_mask[0, 4] = 1
    source_mask[7, 7] = 2

    pasted = dataset._paste(
        target_image,
        target_mask,
        source_image,
        source_mask,
        selected_class=2,
    )

    assert pasted is True
    assert set(target_mask.unique().tolist()) == {0, 2}


def test_copy_paste_protects_existing_foreground(monkeypatch: MonkeyPatch) -> None:
    dataset = seg_copy_paste.CopyPasteDataset.__new__(
        seg_copy_paste.CopyPasteDataset
    )
    monkeypatch.setitem(seg_copy_paste._CONFIG, "min_area_frac", 0.0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "max_paste", 1)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "feather", 0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "background_classes", (0,))
    monkeypatch.setitem(seg_copy_paste._CONFIG, "max_target_overlap", 0.0)

    target_image = torch.zeros((3, 8, 8), dtype=torch.float32)
    target_mask = torch.full((8, 8), 9, dtype=torch.long)
    source_image = torch.ones((3, 8, 8), dtype=torch.float32)
    source_mask = torch.zeros((8, 8), dtype=torch.long)
    source_mask[2:4, 2:4] = 2

    pasted = dataset._paste(
        target_image,
        target_mask,
        source_image,
        source_mask,
        selected_class=2,
    )

    assert pasted is False
    assert torch.all(target_mask == 9)
    assert torch.all(target_image == 0)


def test_copy_paste_config_validation_reset_and_disable() -> None:
    try:
        with pytest.raises(ValueError, match="prob"):
            seg_copy_paste.enable(prob=2.0)
        with pytest.raises(TypeError, match="未知"):
            seg_copy_paste.enable(typo_parameter=1)

        seg_copy_paste.enable(prob=0.25)
        assert seg_copy_paste._CONFIG["prob"] == 0.25
        seg_copy_paste.enable(max_paste=1)
        assert seg_copy_paste._CONFIG["prob"] == 0.5
        assert seg_copy_paste._CONFIG["max_paste"] == 1
    finally:
        seg_copy_paste.disable()

    assert seg_copy_paste._CONFIG == seg_copy_paste._DEFAULT_CONFIG
    assert (
        MaskSemanticSegmentationDatasetArgs.get_dataset_cls()
        is seg_copy_paste.MaskSemanticSegmentationDataset
    )


def test_copy_paste_source_selection_excludes_target(monkeypatch: MonkeyPatch) -> None:
    assert seg_copy_paste._choose_source_index([3], target_index=3) is None
    assert seg_copy_paste._choose_source_index([4], target_index=3) == 4
    seen = []
    monkeypatch.setattr(
        seg_copy_paste.random,
        "choice",
        lambda items: seen.extend(items) or items[0],
    )
    assert seg_copy_paste._choose_source_index([3, 5], target_index=3) == 5
    assert seen == [5]
    assert (
        seg_copy_paste._choose_source_index(
            [3, 5, 7], target_index=3, excluded={5, 7}
        )
        is None
    )


def test_copy_paste_moves_component_to_sampled_position(
    monkeypatch: MonkeyPatch,
) -> None:
    dataset = seg_copy_paste.CopyPasteDataset.__new__(
        seg_copy_paste.CopyPasteDataset
    )
    monkeypatch.setitem(seg_copy_paste._CONFIG, "min_area_frac", 0.0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "max_paste", 1)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "feather", 0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "background_classes", (0,))
    monkeypatch.setitem(seg_copy_paste._CONFIG, "max_target_overlap", 0.0)
    monkeypatch.setattr(
        seg_copy_paste.random,
        "randint",
        lambda _lower, upper: upper,
    )

    target_image = torch.zeros((3, 8, 8), dtype=torch.float32)
    target_mask = torch.zeros((8, 8), dtype=torch.long)
    source_image = torch.ones((3, 8, 8), dtype=torch.float32)
    source_mask = torch.zeros((8, 8), dtype=torch.long)
    source_mask[0, 0] = 2

    dataset._paste(
        target_image,
        target_mask,
        source_image,
        source_mask,
        selected_class=2,
    )

    assert target_mask[0, 0] == 0
    assert target_mask[7, 7] == 2


def test_copy_paste_is_enabled_only_for_training_transform(
    monkeypatch: MonkeyPatch,
) -> None:
    class ExampleTrainTransform:
        pass

    class ExampleValTransform:
        pass

    def fake_parent_init(dataset, _args, _info, transform) -> None:
        dataset._transform = transform
        dataset.class_id_to_internal_class_id = {0: 0, 1: 1}

    built = []
    monkeypatch.setattr(
        seg_copy_paste.MaskSemanticSegmentationDataset,
        "__init__",
        fake_parent_init,
    )
    monkeypatch.setattr(
        seg_copy_paste.CopyPasteDataset,
        "_build_index",
        lambda dataset: built.append(dataset),
    )
    monkeypatch.setitem(seg_copy_paste._CONFIG, "enabled", True)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "paste_classes", None)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "background_classes", (0,))

    train_dataset = seg_copy_paste.CopyPasteDataset(
        None, [], ExampleTrainTransform()
    )
    val_dataset = seg_copy_paste.CopyPasteDataset(None, [], ExampleValTransform())

    assert train_dataset._cp_on is True
    assert val_dataset._cp_on is False
    assert built == [train_dataset]


def test_copy_paste_retries_with_another_source(
    monkeypatch: MonkeyPatch,
) -> None:
    dataset = seg_copy_paste.CopyPasteDataset.__new__(
        seg_copy_paste.CopyPasteDataset
    )
    dataset._cp_on = True
    dataset._cp_valid_indices = None
    dataset._cp_index = {2: [1, 2]}
    dataset.image_info = [{"image_filepaths": "target.png"}]
    target_image = torch.zeros((3, 4, 4), dtype=torch.float32)
    target_mask = torch.zeros((4, 4), dtype=torch.long)
    source_image = torch.ones((3, 4, 4), dtype=torch.float32)
    source_mask = torch.full((4, 4), 2, dtype=torch.long)
    loaded_sources = []

    def load(index, require_classes=None):
        if require_classes is None:
            return target_image, target_mask
        loaded_sources.append(index)
        return source_image, source_mask

    paste_calls = 0

    def paste(_tgt_img, tgt_mask, _src_img, _src_mask, selected_class):
        nonlocal paste_calls
        paste_calls += 1
        if paste_calls == 1:
            return False
        tgt_mask[0, 0] = selected_class
        return True

    dataset._load_transformed = load
    dataset._paste = paste
    dataset.get_binary_masks = lambda mask: {
        "masks": mask.new_zeros((0, *mask.shape), dtype=torch.bool),
        "labels": mask.new_zeros((0,), dtype=torch.long),
    }
    dataset.map_class_id_to_internal_class_id = lambda mask: mask
    monkeypatch.setitem(seg_copy_paste._CONFIG, "prob", 1.0)
    monkeypatch.setitem(seg_copy_paste._CONFIG, "source_sample_tries", 2)
    monkeypatch.setattr(seg_copy_paste.random, "choice", lambda items: items[0])

    item = dataset[0]

    assert paste_calls == 2
    assert loaded_sources == [1, 2]
    assert item["mask"][0, 0] == 2
