from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from tool_lib import common as rt
from tool_lib.seg_eda import detect_segmentation_type, run_seg_eda
from tool_lib.seg_instance_eda import generate_instance_eda_report


def _write_image(path: Path, size: tuple[int, int] = (100, 100)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(32, 64, 96)).save(path)


def _create_instance_dataset(root: Path) -> Path:
    labels = {
        "train/a.txt": [
            "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5",
            "1 0.6 0.6 0.9 0.6 0.9 0.9 0.6 0.9",
        ],
        "val/b.txt": ["1 0.2 0.2 0.4 0.2 0.4 0.4 0.2 0.4"],
        "test/c.txt": [],
    }
    for rel_label, lines in labels.items():
        split = Path(rel_label).parts[0]
        stem = Path(rel_label).stem
        _write_image(root / "images" / split / f"{stem}.jpg")
        label_path = root / "labels" / rel_label
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_path = root / "data.yaml"
    data_path.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "task": "segment",
                "names": {0: "scratch", 1: "dent"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_path


def test_detect_segmentation_type_from_yaml_task(tmp_path: Path) -> None:
    semantic_path = tmp_path / "semantic.yaml"
    semantic_path.write_text(
        yaml.safe_dump({"task": "semantic_segmentation"}), encoding="utf-8"
    )
    instance_path = tmp_path / "instance.yaml"
    instance_path.write_text(yaml.safe_dump({"task": "segment"}), encoding="utf-8")

    assert detect_segmentation_type(semantic_path) == "semantic"
    assert detect_segmentation_type(instance_path) == "instance"


def test_detect_segmentation_type_from_split_structure(tmp_path: Path) -> None:
    semantic_path = tmp_path / "semantic.yaml"
    semantic_path.write_text(
        yaml.safe_dump(
            {"train": {"images": "images/train", "masks": "masks/train"}}
        ),
        encoding="utf-8",
    )
    assert detect_segmentation_type(semantic_path) == "semantic"


def test_generate_instance_eda_report(tmp_path: Path) -> None:
    rt.import_runtime_dependencies()
    data_path = _create_instance_dataset(tmp_path / "dataset")
    output_dir = generate_instance_eda_report(
        source_data_path=data_path,
        output_dir=tmp_path / "eda",
        overwrite=True,
        min_class_images=1,
    )

    report = json.loads(
        (output_dir / "instance_eda_instance.json").read_text(encoding="utf-8")
    )
    assert report["meta"]["segmentation_type"] == "instance"
    assert report["overview"]["images"] == 3
    assert report["overview"]["instances"] == 3
    assert report["classes"]["0"]["all"]["instances"] == 1
    assert report["classes"]["1"]["all"]["instances"] == 2
    assert report["split_summaries"]["test"]["empty_images"] == 1
    assert (output_dir / "instance_eda_instance.md").exists()
    assert (output_dir / "image_instance_inventory_instance.csv").exists()


def test_auto_router_selects_instance_eda(tmp_path: Path, monkeypatch) -> None:
    data_path = _create_instance_dataset(tmp_path / "dataset")
    expected = tmp_path / "selected-instance"

    def fake_generate_instance_eda_report(**kwargs):
        assert kwargs["source_data_path"] == data_path.resolve()
        return expected

    monkeypatch.setattr(
        "tool_lib.seg_eda.seg_instance_eda.generate_instance_eda_report",
        fake_generate_instance_eda_report,
    )
    args = argparse.Namespace(
        data=data_path,
        seg_type="auto",
        output_dir=None,
        overwrite=False,
        min_class_images=10,
        threshold_percentile=0.9,
    )
    assert run_seg_eda(args) == expected
