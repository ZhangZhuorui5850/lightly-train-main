from __future__ import annotations

from pathlib import Path

import pytest

import train_face_wider


def test_load_dataset_config_resolves_relative_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
    config = root / "data.yaml"
    config.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: {0: face}\n",
        encoding="utf-8",
    )

    resolved = train_face_wider.load_dataset_config(config)

    assert resolved["path"] == str(root.resolve())


def test_validate_args_rejects_invalid_training_values(monkeypatch) -> None:
    parser = train_face_wider.build_parser()
    with pytest.raises(ValueError, match="steps"):
        train_face_wider.validate_args(parser.parse_args(["--steps", "0"]))

    monkeypatch.setattr(train_face_wider.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_face_wider.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="整除"):
        train_face_wider.validate_args(parser.parse_args(["--batch-size", "3"]))


def test_resolve_run_mode_handles_new_resume_and_incomplete_runs(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    assert train_face_wider.resolve_run_mode(output, fresh=False) == (False, False)
    assert train_face_wider.resolve_run_mode(output, fresh=True) == (False, True)

    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert train_face_wider.resolve_run_mode(output, fresh=False) == (True, False)

    checkpoint.unlink()
    with pytest.raises(FileExistsError, match="缺少续训权重"):
        train_face_wider.resolve_run_mode(output, fresh=False)


def test_main_starts_resumes_and_rejects_changed_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames: {0: face}\n",
        encoding="utf-8",
    )
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"weights")
    output = tmp_path / "out"
    calls = []
    monkeypatch.setitem(train_face_wider.DATASETS, "original", data_yaml)
    monkeypatch.setattr(train_face_wider, "BACKBONE_WEIGHTS", weights)
    monkeypatch.setattr(
        train_face_wider.lightly_train,
        "train_object_detection",
        lambda **kwargs: calls.append(kwargs),
    )

    base_args = [
        "--variant",
        "original",
        "--steps",
        "100",
        "--batch-size",
        "1",
        "--out",
        str(output),
    ]
    assert train_face_wider.main(base_args) == 0
    assert calls[-1]["resume_interrupted"] is False
    assert calls[-1]["overwrite"] is True

    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    assert train_face_wider.main(base_args) == 0
    assert calls[-1]["resume_interrupted"] is True
    assert calls[-1]["overwrite"] is False

    changed_args = ["200" if value == "100" else value for value in base_args]
    with pytest.raises(ValueError, match="steps"):
        train_face_wider.main(changed_args)
