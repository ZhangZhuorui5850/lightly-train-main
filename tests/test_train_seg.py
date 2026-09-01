from __future__ import annotations

from pathlib import Path

import pytest

import train_seg
from lightly_train._data import task_batch_collation


def test_load_data_resolves_dataset_root_and_preserves_class_mapping(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "data.yaml"
    config.write_text(
        "path: ../dataset\n"
        "train: {images: images/train, masks: masks/train}\n"
        "val: {images: images/val, masks: masks/val}\n"
        "classes:\n"
        "  0: {name: background, labels: [0]}\n"
        "  1: {name: target, labels: [255]}\n",
        encoding="utf-8",
    )

    data = train_seg.load_data(str(config))

    assert data["train"]["images"] == str(dataset / "images" / "train")
    assert data["val"]["masks"] == str(dataset / "masks" / "val")
    assert data["classes"][1] == {"name": "target", "labels": [255]}


def test_resolve_run_mode_handles_new_resume_and_incomplete_runs(
    tmp_path: Path,
) -> None:
    output = tmp_path / "experiment"
    assert train_seg.resolve_run_mode(str(output), fresh=False) == (False, False)
    assert train_seg.resolve_run_mode(str(output), fresh=True) == (False, True)

    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert train_seg.resolve_run_mode(str(output), fresh=False) == (True, False)

    checkpoint.unlink()
    with pytest.raises(FileExistsError, match="缺少续训权重"):
        train_seg.resolve_run_mode(str(output), fresh=False)


def test_main_starts_new_run_then_resumes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "masks" / split).mkdir(parents=True)
    config = dataset / "data.yaml"
    config.write_text(
        "train: {images: images/train, masks: masks/train}\n"
        "val: {images: images/val, masks: masks/val}\n"
        "classes: {0: background, 1: target}\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    output = tmp_path / "out"
    calls = []

    monkeypatch.setattr(train_seg, "DATA_YAML", str(config))
    monkeypatch.setattr(train_seg, "BACKBONE_WEIGHTS", str(weights))
    monkeypatch.setattr(train_seg, "OUT", str(output))
    monkeypatch.setattr(train_seg, "COPY_PASTE", False)
    monkeypatch.setattr(train_seg, "FRESH", False)
    monkeypatch.setattr(
        train_seg.lightly_train,
        "train_semantic_segmentation",
        lambda **kwargs: calls.append(kwargs),
    )

    train_seg.main()
    assert calls[-1]["resume_interrupted"] is False
    assert calls[-1]["overwrite"] is True

    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train_seg.main()
    assert calls[-1]["resume_interrupted"] is True
    assert calls[-1]["overwrite"] is False


def test_enable_checkpoint_compatibility_registers_legacy_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        from lightly_train._transforms.semantic_segmentation_transform import (
            SemanticSegmentationCollateFunction,
        )
    except ImportError:
        train_seg.enable_checkpoint_compatibility()
        assert hasattr(
            task_batch_collation,
            "MaskSemanticSegmentationCollateFunction",
        )
        return

    monkeypatch.delattr(
        task_batch_collation,
        "MaskSemanticSegmentationCollateFunction",
        raising=False,
    )

    train_seg.enable_checkpoint_compatibility()

    assert (
        task_batch_collation.MaskSemanticSegmentationCollateFunction
        is SemanticSegmentationCollateFunction
    )
