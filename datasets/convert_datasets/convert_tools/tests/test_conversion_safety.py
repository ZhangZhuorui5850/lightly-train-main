from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import one_click_convert  # noqa: E402
import sync_picture  # noqa: E402


def _pair(root: Path, split: str, class_id: int, color: tuple[int, int, int]) -> None:
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color).save(directory / "a.jpg")
    (directory / "a.txt").write_text(
        f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )


def test_sync_dry_run_has_no_filesystem_side_effects(tmp_path):
    source = tmp_path / "source"
    _pair(source, "raw", 0, (255, 0, 0))
    output = tmp_path / "output"
    stats = sync_picture.copy_files([source], output, "yolo", seed=1, dry_run=True)
    sync_picture.print_report(stats, output, [source], 0.0, 1, dry_run=True)
    assert stats["paired_total"] == 1
    assert not output.exists()


def test_same_basename_from_two_sources_never_cross_pairs(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _pair(left, "raw", 0, (255, 0, 0))
    _pair(right, "raw", 1, (0, 0, 255))
    output = tmp_path / "output"
    stats = sync_picture.copy_files([left, right], output, "yolo", seed=1)
    assert stats["paired_total"] == 2
    for image_path in output.rglob("*.jpg"):
        label_path = image_path.with_suffix(".txt")
        class_id = int(label_path.read_text(encoding="utf-8").split()[0])
        with Image.open(image_path) as image:
            red, _green, blue = image.resize((1, 1)).getpixel((0, 0))
        assert class_id == (0 if red > blue else 1)


def test_oneclick_presplit_dry_run_never_calls_converter(tmp_path, monkeypatch):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root, class_id in ((left, 0), (right, 1)):
        _pair(root, "train", class_id, (255, 0, 0))
        _pair(root, "val", class_id, (0, 0, 255))
        (root / "classes.txt").write_text("class_0\nclass_1\n", encoding="utf-8")

    def forbidden() -> None:
        raise AssertionError("dry-run invoked the mutating converter")

    monkeypatch.setattr(one_click_convert.labelme_to_yolo, "main", forbidden)
    output = tmp_path / "converted"
    one_click_convert.run_conversion(
        [left, right], output, task="det", label_format="yolo",
        preserve_splits="auto", dry_run=True,
    )
    assert not output.exists()


def test_oneclick_publishes_complete_output_without_staging_paths(tmp_path):
    source = tmp_path / "source"
    _pair(source, "train", 0, (255, 0, 0))
    _pair(source, "val", 0, (0, 0, 255))
    (source / "classes.txt").write_text("defect\n", encoding="utf-8")
    output = tmp_path / "converted"

    one_click_convert.run_conversion(
        [source], output, task="det", label_format="yolo",
        preserve_splits="yes",
    )

    data_yaml = output / "dataset_det" / "data.yaml"
    assert data_yaml.is_file()
    assert "path: ." in data_yaml.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".converted.staging-*"))


def test_oneclick_requires_clean_for_existing_output(tmp_path):
    source = tmp_path / "source"
    _pair(source, "train", 0, (255, 0, 0))
    _pair(source, "val", 0, (0, 0, 255))
    (source / "classes.txt").write_text("defect\n", encoding="utf-8")
    output = tmp_path / "converted"
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        one_click_convert.run_conversion(
            [source], output, task="det", label_format="yolo",
            preserve_splits="yes",
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_oneclick_remaps_single_presplit_yolo_by_class_name(tmp_path):
    source = tmp_path / "source"
    _pair(source, "train", 0, (255, 0, 0))  # source id 0 == class_b
    _pair(source, "val", 0, (0, 0, 255))
    (source / "classes.txt").write_text("class_b\nclass_a\n", encoding="utf-8")
    class_ref = tmp_path / "canonical.txt"
    class_ref.write_text("class_a\nclass_b\n", encoding="utf-8")
    output = tmp_path / "converted"

    one_click_convert.run_conversion(
        [source], output, task="det", label_format="yolo",
        preserve_splits="yes", class_ref=str(class_ref),
    )

    labels = list((output / "dataset_det" / "labels").rglob("*.txt"))
    assert labels
    assert all(path.read_text(encoding="utf-8").split()[0] == "1" for path in labels)


def test_semantic_multi_source_remaps_ids_and_avoids_name_collisions(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    for root, names, mask_id, color in (
        (left, ["background", "defect"], 1, (255, 0, 0)),
        (right, ["defect", "background"], 0, (0, 0, 255)),
    ):
        (root / "JPEGImages").mkdir(parents=True)
        (root / "SegmentationClass").mkdir(parents=True)
        Image.new("RGB", (4, 4), color).save(root / "JPEGImages" / "a.jpg")
        Image.fromarray(np.full((4, 4), mask_id, dtype=np.uint8)).save(
            root / "SegmentationClass" / "a.png"
        )
        (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    output = tmp_path / "converted"
    one_click_convert.run_conversion(
        [left, right], output, task="seg", seg_type="semantic",
        preserve_splits="no", seed=1,
    )

    masks = list((output / "dataset_semantic" / "masks").rglob("*.png"))
    assert len(masks) == 2
    assert len({path.name for path in masks}) == 2
    assert all(set(np.unique(np.array(Image.open(path))).tolist()) == {1} for path in masks)


def test_standard_yolo_images_labels_layout_preserves_splits(tmp_path):
    source = tmp_path / "source"
    for split, color in (("train", (255, 0, 0)), ("val", (0, 0, 255))):
        (source / "images" / split).mkdir(parents=True, exist_ok=True)
        (source / "labels" / split).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(source / "images" / split / "a.jpg")
        (source / "labels" / split / "a.txt").write_text(
            "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
        )
    (source / "classes.txt").write_text("defect\n", encoding="utf-8")
    output = tmp_path / "converted"

    one_click_convert.run_conversion(
        [source], output, task="det", label_format="yolo", preserve_splits="auto"
    )

    assert (output / "dataset_det" / "labels" / "train" / "a.txt").is_file()
    assert (output / "dataset_det" / "labels" / "val" / "a.txt").is_file()


def test_oneclick_all_with_semantic_keeps_each_requested_component(tmp_path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "masks").mkdir()
    (source / "labels").mkdir()
    Image.new("RGB", (8, 8), (120, 120, 120)).save(source / "images" / "a.jpg")
    Image.fromarray(np.pad(np.ones((4, 4), dtype=np.uint8), 2)).save(
        source / "masks" / "a.png"
    )
    (source / "labels" / "a.txt").write_text(
        "1 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    (source / "classes.txt").write_text("background\ndefect\n", encoding="utf-8")
    output = tmp_path / "converted"

    one_click_convert.run_conversion(
        [source],
        output,
        task="all",
        label_format="yolo",
        seg_type="semantic",
        preserve_splits="no",
        seed=7,
    )

    assert (output / "dataset_semantic" / "data.yaml").is_file()
    assert (output / "dataset_det" / "data.yaml").is_file()
    assert (output / "dataset_cls" / "data.yaml").is_file()
    assert not (output / "dataset_seg").exists()
    det_yaml = (output / "dataset_det" / "data.yaml").read_text(encoding="utf-8")
    assert '1: "defect"' in det_yaml
