from pathlib import Path

import json
import sys

import numpy as np
import pytest
import yaml
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from semantic_to_xanylabeling import (  # noqa: E402
    build_grayscale_mapping,
    create_mapping,
)


def _make_dataset(root: Path) -> Path:
    image_dir = root / "images" / "train"
    mask_dir = root / "masks" / "train"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    Image.new("RGB", (4, 3), "white").save(image_dir / "sample.jpg")
    mask = np.array(
        [[0, 0, 1, 1], [0, 2, 2, 1], [0, 0, 2, 2]], dtype=np.uint8
    )
    Image.fromarray(mask).save(mask_dir / "sample.png")
    config = {
        "task": "semantic_segmentation",
        "classes": {0: "background", 1: "road", 2: "car"},
        "train": {"images": "images/train", "masks": "masks/train"},
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return root


def test_build_mapping_skips_background_by_default() -> None:
    payload = build_grayscale_mapping(
        {0: "background", 1: "road", 2: "car"}
    )
    assert payload == {"type": "grayscale", "colors": {"road": 1, "car": 2}}


def test_create_mapping_validates_and_writes_json(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "semantic")
    output, payload, stats = create_mapping(source)

    assert output == source / "mask_grayscale_map.json"
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert stats is not None
    assert stats.images == 1
    assert stats.masks == 1
    assert stats.observed_values == (0, 1, 2)


def test_create_mapping_dry_run_validates_without_writing(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "semantic")

    output, payload, stats = create_mapping(source, dry_run=True)

    assert output == source / "mask_grayscale_map.json"
    assert not output.exists()
    assert payload["colors"] == {"road": 1, "car": 2}
    assert stats is not None
    assert stats.observed_values == (0, 1, 2)


def test_create_mapping_reports_unknown_mask_value(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "semantic")
    Image.fromarray(
        np.array([[0, 0, 9, 9], [0, 9, 9, 9], [0, 0, 9, 9]], dtype=np.uint8)
    ).save(
        source / "masks" / "train" / "sample.png"
    )
    messages: list[str] = []

    output, payload, stats = create_mapping(source, messages=messages)

    assert output.is_file()
    assert payload["colors"] == {"class_9": 9}
    assert stats is not None
    assert stats.observed_values == (0, 9)
    assert "9->class_9" in messages[0]


def test_duplicate_names_are_disambiguated_by_id() -> None:
    messages: list[str] = []
    payload = build_grayscale_mapping(
        {1: "none", 4: "none", 7: "none"}, messages=messages
    )

    assert payload["colors"] == {
        "none": 1,
        "none__id_4": 4,
        "none__id_7": 7,
    }
    assert len(messages) == 2


def test_strict_classes_reports_unknown_mask_value(tmp_path: Path) -> None:
    source = _make_dataset(tmp_path / "semantic")
    Image.fromarray(np.full((3, 4), 9, dtype=np.uint8)).save(
        source / "masks" / "train" / "sample.png"
    )

    with pytest.raises(ValueError, match="未定义的像素值"):
        create_mapping(source, strict_classes=True)
