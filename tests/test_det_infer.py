from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from tool_lib import common as rt
from tool_lib.det_infer import (
    build_split_output_dir,
    export_bad_class_images,
    resolve_dataset_infer_splits,
    select_bad_classes_from_report,
)


def _create_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_resolve_dataset_infer_splits_supports_train_and_all() -> None:
    data_cfg = {"train": "images/train", "val": "images/val", "test": "images/test"}

    assert resolve_dataset_infer_splits(data_cfg, "train") == ["train"]
    assert resolve_dataset_infer_splits(data_cfg, "all") == ["train", "test", "val"]


def test_build_split_output_dir_defaults_to_experiment_infer_split(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    args = SimpleNamespace(output_dir=None, split="test")
    data_cfg = {"_root_dir": tmp_path / "dataset_det"}

    output_dir = build_split_output_dir(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        split="test",
    )

    assert output_dir == experiment_dir / "infer" / "test"


def test_select_bad_classes_from_report_uses_threshold_and_gt() -> None:
    report_payload = {
        "per_class_ap": {
            "0": {"name": "bad", "ap": 0.29, "gt": 2, "pred": 1, "tp": 0},
            "1": {"name": "border", "ap": 0.3, "gt": 2, "pred": 1, "tp": 1},
            "2": {"name": "empty", "ap": 0.0, "gt": 0, "pred": 1, "tp": 0},
        }
    }

    bad_classes = select_bad_classes_from_report(report_payload, threshold=0.3)

    assert set(bad_classes) == {0}
    assert bad_classes[0]["class_name"] == "bad"


def test_export_bad_class_images_writes_named_visualizations_and_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_det"
    image_dir = dataset_root / "images" / "test"
    label_dir = dataset_root / "labels" / "test"
    output_dir = tmp_path / "out" / "exp_det" / "infer" / "test"

    _create_image(image_dir / "a.jpg")
    _create_image(image_dir / "b.jpg", color=(0, 255, 0))
    (label_dir / "a.txt").parent.mkdir(parents=True, exist_ok=True)
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (label_dir / "b.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    _create_image(output_dir / "images" / "a.jpg", color=(0, 0, 255))
    _create_image(output_dir / "images" / "b.jpg", color=(255, 255, 0))

    report_path = output_dir / "test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "per_class_ap": {
                    "0": {"name": "缺陷/类别", "ap": 0.2, "gt": 2, "pred": 1, "tp": 0},
                    "1": {"name": "good", "ap": 0.9, "gt": 1, "pred": 1, "tp": 1},
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    samples = [
        rt.ImageSample(
            image_path=image_dir / "a.jpg",
            relative_path=Path("a.jpg"),
            label_path=label_dir / "a.txt",
        ),
        rt.ImageSample(
            image_path=image_dir / "b.jpg",
            relative_path=Path("b.jpg"),
            label_path=label_dir / "b.txt",
        ),
    ]

    summary = export_bad_class_images(
        args=SimpleNamespace(bad_class_map50_threshold=0.3),
        output_dir=output_dir,
        report_path=report_path,
        data_cfg=None,
        split="test",
        samples=samples,
    )

    bad_images_dir = output_dir / "bad_images"
    assert summary["exported_count"] == 2
    assert (bad_images_dir / "缺陷-类别_1.jpg").exists()
    assert (bad_images_dir / "缺陷-类别_2.jpg").exists()

    with (bad_images_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["split"] == "test"
    assert rows[0]["class_id"] == "0"
    assert rows[0]["class_name"] == "缺陷/类别"
    assert rows[0]["ap"] == "0.2"
    assert Path(rows[0]["source_image_path"]).exists()
    assert Path(rows[0]["source_visualization_path"]).exists()
    assert Path(rows[0]["export_image_path"]).exists()

    manifest_payload = json.loads((bad_images_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["summary"]["bad_class_count"] == 1
