from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import face_wider_prepare  # noqa: E402


def _write_sample(root: Path, split: str, stem: str, box_size: int) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1024, 1024), (120, 160, 190))
    left = 512 - box_size // 2
    ImageDraw.Draw(image).rectangle((left, left, left + box_size, left + box_size), fill=(210, 90, 70))
    image.save(image_dir / f"{stem}.jpg")
    ratio = box_size / 1024
    (label_dir / f"{stem}.txt").write_text(f"0 0.5 0.5 {ratio} {ratio}\n", encoding="utf-8")


def test_combined_generation_and_dry_run(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_sample(source, "train", "donor", 40)
    _write_sample(source, "train", "recipient", 120)
    ratio = 120 / 1024
    (source / "labels" / "train" / "recipient.txt").write_text(
        f"0 0.35 0.5 {ratio} {ratio}\n0 0.65 0.5 {ratio} {ratio}\n",
        encoding="utf-8",
    )
    _write_sample(source, "val", "val", 80)
    _write_sample(source, "test", "test", 80)
    (source / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames: {0: face}\n",
        encoding="utf-8",
    )
    output = tmp_path / "combined"
    parser = face_wider_prepare.build_parser()
    dry_args = parser.parse_args(["--mode", "combined", "--src", str(source), "--out", str(output), "--dry-run"])
    face_wider_prepare.run(dry_args)
    assert not output.exists()

    args = parser.parse_args(
        [
            "--mode", "combined", "--src", str(source), "--out", str(output),
            "--recipient-fraction", "1", "--tile-count", "1", "--tile-min", "512",
            "--tile-max", "512", "--min-sharpness", "0", "--preview-count", "2",
        ]
    )
    face_wider_prepare.run(args)

    train_images = list((output / "images" / "train").glob("*.jpg"))
    assert len(train_images) == 4
    assert (output / "augmentation_preview.jpg").is_file()
    manifest = (output / "augmentation_manifest.jsonl").read_text(encoding="utf-8")
    assert '"kind": "copy_paste"' in manifest
    assert '"kind": "tile"' in manifest
    for line in manifest.splitlines():
        record = json.loads(line)
        if record["kind"] == "copy_paste":
            donors = [
                (paste["donor_image"], paste["donor_box_index"])
                for paste in record["pastes"]
            ]
            assert record["recipient_image"] not in {donor[0] for donor in donors}
            assert len(donors) == len(set(donors))
    source_image = source / "images" / "train" / "donor.jpg"
    output_image = output / "images" / "train" / "donor.jpg"
    source_label = source / "labels" / "train" / "donor.txt"
    output_label = output / "labels" / "train" / "donor.txt"
    assert source_image.stat().st_ino == output_image.stat().st_ino
    assert source_label.stat().st_ino != output_label.stat().st_ino
    for label in (output / "labels" / "train").glob("*.txt"):
        for box in face_wider_prepare.read_boxes(label):
            assert box.x - box.w / 2 >= -1e-6
            assert box.y - box.h / 2 >= -1e-6
            assert box.x + box.w / 2 <= 1 + 1e-6
            assert box.y + box.h / 2 <= 1 + 1e-6


def test_read_boxes_rejects_fractional_class_and_overflow(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    for text in ("0.5 0.5 0.5 0.1 0.1\n", "0 0.95 0.5 0.2 0.1\n"):
        label.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            face_wider_prepare.read_boxes(label)


def test_run_validates_arguments_for_direct_call(tmp_path: Path) -> None:
    parser = face_wider_prepare.build_parser()
    args = parser.parse_args(
        ["--mode", "tile", "--out", str(tmp_path / "out"), "--tile-count", "-1"]
    )
    with pytest.raises(ValueError, match="非负数"):
        face_wider_prepare.run(args)
