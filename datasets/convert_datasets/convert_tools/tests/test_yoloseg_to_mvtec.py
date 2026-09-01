from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import mirror_det_subset_to_seg as mirror  # noqa: E402
import yoloseg_to_mvtec as converter  # noqa: E402


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((16, 16, 3), 100, dtype=np.uint8))


def _dataset(root: Path, *, class_name: str = "defect") -> Path:
    root.mkdir(parents=True)
    (root / "classes.txt").write_text(class_name + "\n", encoding="utf-8")
    _image(root / "images" / "train" / "good.jpg")
    _image(root / "images" / "val" / "bad.jpg")
    (root / "labels" / "val").mkdir(parents=True)
    (root / "labels" / "val" / "bad.txt").write_text(
        "0 0.2 0.2 0.8 0.2 0.5 0.8\n", encoding="utf-8"
    )
    return root


def test_missing_label_image_is_kept_as_good(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source")
    output = tmp_path / "mvtec"

    result = converter.convert(source, output, verbose=False)

    assert result[0]["train_good"] == 1
    assert (output / "0_defect" / "train" / "good" / "good.png").is_file()


def test_conversion_preserves_manifest_membership(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for stem in ("selected", "extra"):
        _image(source / "images" / "train" / "nested" / f"{stem}.jpg")
        label = source / "labels" / "train" / "nested" / f"{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "0 0.2 0.2 0.8 0.2 0.5 0.8\n",
            encoding="utf-8",
        )
    (source / "train.txt").write_text(
        "images/train/nested/selected.jpg\n",
        encoding="utf-8",
    )
    (source / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train.txt",
                "task": "segment",
                "names": {0: "defect"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "mvtec"

    result = converter.convert(source / "data.yaml", output, verbose=False)

    assert result[0]["defect"] == 1
    assert (output / "0_defect" / "test" / "defect" / "selected.png").is_file()
    assert not list(output.rglob("extra.png"))


def test_output_must_be_isolated_from_source(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source")
    before = sorted(path.relative_to(source) for path in source.rglob("*"))

    with pytest.raises(ValueError, match="输出目录"):
        converter.convert(source, source, clean=True, verbose=False)

    assert sorted(path.relative_to(source) for path in source.rglob("*")) == before


def test_failed_clean_conversion_keeps_previous_output(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source")
    (source / "labels" / "val" / "bad.txt").write_text(
        "0 1.2 0.2 0.8 0.2 0.5 0.8\n", encoding="utf-8"
    )
    output = tmp_path / "mvtec"
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        converter.convert(source, output, clean=True, verbose=False)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_class_names_cannot_escape_output(tmp_path: Path) -> None:
    source = _dataset(tmp_path / "source", class_name="../escape")
    output = tmp_path / "mvtec"

    converter.convert(source, output, verbose=False)

    assert not (tmp_path / "escape").exists()
    assert len([path for path in output.iterdir() if path.is_dir()]) == 1


def test_mirror_place_rejects_same_file_without_deleting_it(tmp_path: Path) -> None:
    path = tmp_path / "sample.jpg"
    path.write_bytes(b"image")

    with pytest.raises(ValueError, match="重合"):
        mirror._place(path, path, "copy", False)

    assert path.read_bytes() == b"image"


def test_mirror_publishes_portable_metadata(tmp_path: Path) -> None:
    det_root = tmp_path / "det"
    seg_root = tmp_path / "seg"
    _image(det_root / "images" / "train" / "a.jpg")
    _image(seg_root / "images" / "train" / "a.jpg")
    (seg_root / "labels" / "train").mkdir(parents=True)
    (seg_root / "labels" / "train" / "a.txt").write_text(
        "0 0.2 0.2 0.8 0.2 0.5 0.8\n", encoding="utf-8"
    )
    output = tmp_path / "mirrored"

    status = mirror.run_mirror(
        det_root=det_root,
        det_cfg={"train": "images/train"},
        seg_root=seg_root,
        seg_cfg={"train": "images/train", "task": "segment", "names": {0: "defect"}},
        out_root=output,
        copy_mode="copy",
        dry_run=False,
    )

    assert status == 0
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["path"] == "."
    assert ".staging-" not in (output / "data.yaml").read_text(encoding="utf-8")


def test_mirror_preserves_det_manifest_membership_and_nested_paths(tmp_path: Path) -> None:
    det_root = tmp_path / "det"
    seg_root = tmp_path / "seg"
    for stem in ("selected", "extra"):
        _image(det_root / "images" / "train" / "nested" / f"{stem}.jpg")
        _image(seg_root / "images" / "train" / "nested" / f"{stem}.jpg")
        seg_label = seg_root / "labels" / "train" / "nested" / f"{stem}.txt"
        seg_label.parent.mkdir(parents=True, exist_ok=True)
        seg_label.write_text(
            "0 0.2 0.2 0.8 0.2 0.5 0.8\n",
            encoding="utf-8",
        )
    (det_root / "train.txt").write_text(
        "images/train/nested/selected.jpg\n",
        encoding="utf-8",
    )
    det_cfg = {"path": ".", "train": "train.txt", "task": "detect", "names": ["defect"]}
    seg_cfg = {
        "path": ".",
        "train": "images/train",
        "task": "segment",
        "names": {0: "defect"},
    }
    output = tmp_path / "mirrored"

    status = mirror.run_mirror(
        det_root=det_root,
        det_cfg=det_cfg,
        seg_root=seg_root,
        seg_cfg=seg_cfg,
        out_root=output,
        copy_mode="copy",
        dry_run=False,
    )

    assert status == 0
    assert (output / "images/train/nested/selected.jpg").is_file()
    assert (output / "labels/train/nested/selected.txt").is_file()
    assert not list(output.rglob("extra.*"))


def test_make_sample_help_has_no_output_side_effect(tmp_path: Path) -> None:
    output = tmp_path / "sample"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_sample_yoloseg.py"),
            "--out",
            str(output),
            "--help",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert not output.exists()
