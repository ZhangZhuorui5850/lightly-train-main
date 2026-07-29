from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import dataset_discovery as discovery  # noqa: E402
import conversion_wizard as wizard  # noqa: E402
import output_naming  # noqa: E402
import yoloseg_to_semantic as converter  # noqa: E402


def _image(path: Path, size: tuple[int, int] = (20, 16)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (100, 120, 140)).save(path)


def _make_yolo_seg(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: "scratch", 1: "dent"},
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    _image(root / "images/train/a.jpg")
    _image(root / "images/val/b.png")
    label = root / "labels/train/a.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text(
        "0 0.10 0.10 0.80 0.10 0.80 0.80 0.10 0.80\n"
        "1 0.50 0.50 0.95 0.50 0.95 0.95 0.50 0.95\n"
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    return root


def test_convert_yolo_polygons_to_train_seg_dataset(tmp_path: Path) -> None:
    source = _make_yolo_seg(tmp_path / "dataset_seg")
    output = tmp_path / "dataset_semantic"

    report = converter.convert_dataset(source, output)

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "semantic_segmentation"
    assert config["train"] == {"images": "images/train", "masks": "masks/train"}
    assert config["classes"] == {0: "background", 1: "scratch", 2: "dent"}
    assert (output / "images/train/a.jpg").is_file()
    assert (output / "images/val/b.png").is_file()

    train_mask = np.asarray(Image.open(output / "masks/train/a.png"))
    assert train_mask.shape == (16, 20)
    assert set(np.unique(train_mask)) == {0, 1, 2}
    assert train_mask[10, 12] == 2  # 后一行 polygon 赢得重叠区域。
    val_mask = np.asarray(Image.open(output / "masks/val/b.png"))
    assert np.all(val_mask == 0)
    assert report["class_mapping"] == {"0": 1, "1": 2}
    assert report["splits"]["train"]["invalid_rows"] == 1
    assert report["splits"]["val"]["background_images"] == 1


def test_converter_rejects_detection_labels(tmp_path: Path) -> None:
    source = _make_yolo_seg(tmp_path / "dataset_det")
    (source / "labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="需要 YOLO polygon"):
        converter.convert_dataset(source, tmp_path / "output")


def test_discovery_classifies_supported_datasets(tmp_path: Path) -> None:
    seg = _make_yolo_seg(tmp_path / "seg")
    det = _make_yolo_seg(tmp_path / "det")
    (det / "labels/train/a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    semantic = tmp_path / "semantic"
    _image(semantic / "images/train/a.jpg")
    _image(semantic / "images/val/b.jpg")
    (semantic / "masks/train").mkdir(parents=True)
    Image.new("L", (20, 16), 0).save(semantic / "masks/train/a.png")
    (semantic / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {"images": "images/train", "masks": "masks/train"},
                "val": {"images": "images/val", "masks": "masks/val"},
                "classes": {0: "background"},
            }
        ),
        encoding="utf-8",
    )

    kinds = {item.path: item.kind for item in discovery.scan_datasets(tmp_path)}
    assert kinds[seg.resolve()] == "yolo_instance"
    assert kinds[det.resolve()] == "yolo_detection"
    assert kinds[semantic.resolve()] == "semantic_mask"
    candidate = discovery.inspect_config_dataset(seg / "data.yaml")
    assert discovery.conversion_actions(candidate) == ("to-semantic", "to-mvtec")


def test_output_directory_uses_source_and_operation(tmp_path: Path) -> None:
    source = tmp_path / "dataset_seg"
    assert output_naming.default_output_dir(source, "to-semantic") == (
        tmp_path / "dataset_seg__to_semantic"
    )
    assert output_naming.default_output_dir(source, "to-mvtec") == (
        tmp_path / "dataset_seg__to_mvtec_ad"
    )


def test_wizard_selects_operation_before_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset_seg"
    candidate = discovery.DatasetCandidate(
        path=source,
        kind="yolo_instance",
        image_count=10,
        annotation_count=10,
        class_count=2,
    )
    events: list[str] = []
    answers = iter(["1", "1", ""])

    def fake_input(_prompt: str) -> str:
        events.append("prompt")
        return next(answers)

    def fake_scan(_root: Path) -> list[discovery.DatasetCandidate]:
        events.append("scan")
        return [candidate]

    dispatched: dict[str, object] = {}

    def fake_dispatch(item, action, output):
        dispatched.update(item=item, action=action, output=output)
        return 0

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(wizard, "scan_datasets", fake_scan)
    monkeypatch.setattr(wizard, "dispatch", fake_dispatch)

    assert wizard.interactive(tmp_path) == 0
    assert events.index("prompt") < events.index("scan")
    assert dispatched == {
        "item": candidate,
        "action": "to-semantic",
        "output": tmp_path / "dataset_seg__to_semantic",
    }
