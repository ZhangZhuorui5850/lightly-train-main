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
import mirror_det_subset_to_seg as mirror  # noqa: E402
import semantic_class_editor as semantic  # noqa: E402
import yolo_class_editor as yolo  # noqa: E402


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (20, 40, 60)).save(path)


def test_yolo_detection_discovers_split_root_layout(tmp_path: Path) -> None:
    root = tmp_path / "det"
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {"train": "train", "val": "val", "task": "detect", "names": {0: "box"}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _image(root / "train/images/a.jpg")
    _image(root / "val/images/b.jpg")
    (root / "train/labels/a.txt").parent.mkdir(parents=True)
    (root / "train/labels/a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    candidate = discovery.inspect_config_dataset(root / "data.yaml")
    dataset = yolo.resolve_dataset(root)

    assert candidate.kind == "yolo_detection"
    assert candidate.splits == ("train", "val")
    assert dataset.splits["train"].images == (root / "train/images").resolve()
    assert dataset.splits["train"].labels == (root / "train/labels").resolve()
    assert yolo.analyze_dataset(dataset).annotation_count == 1


def test_semantic_discovers_split_root_layout(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "train": "train",
                "val": "val",
                "task": "semantic_segmentation",
                "classes": {0: "background", 1: "object"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _image(root / "train/images/a.jpg")
    _image(root / "val/images/b.jpg")
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    (root / "train/masks").mkdir(parents=True)
    (root / "val/masks").mkdir(parents=True)
    Image.fromarray(mask).save(root / "train/masks/a.png")
    Image.fromarray(mask).save(root / "val/masks/b.png")

    candidate = discovery.inspect_config_dataset(root / "data.yaml")
    dataset = semantic.resolve_dataset(root)

    assert candidate.kind == "semantic_mask"
    assert candidate.annotation_count == 2
    assert dataset.splits["train"].images == (root / "train/images").resolve()
    assert dataset.splits["train"].masks == (root / "train/masks").resolve()
    assert semantic.analyze_dataset(dataset).mask_count == 2


def test_mirror_tool_uses_split_root_layout(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    root.mkdir()
    config = {
        "train": "train",
        "task": "semantic_segmentation",
        "classes": {0: "background"},
    }
    _image(root / "train/images/a.jpg")
    (root / "train/masks").mkdir(parents=True)
    Image.new("L", (8, 6), 0).save(root / "train/masks/a.png")

    assert mirror._split_image_dir(root, config, "train") == (root / "train/images").resolve()
    assert mirror._detect_seg_format(root, config) == "semantic"
    index = mirror.build_seg_index(root, config, "semantic")
    assert index["a"]["label"] == (root / "train/masks/a.png").resolve()


@pytest.mark.parametrize("layout", ("images_first", "split_first"))
def test_yolo_resolver_recovers_from_stale_yaml_root(
    tmp_path: Path,
    layout: str,
) -> None:
    root = tmp_path / "copied_dataset"
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "/mnt/old-machine/project/dataset_det",
                "train": "images/train" if layout == "images_first" else "train",
                "task": "detect",
                "names": {0: "part"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if layout == "images_first":
        _image(root / "images/train/a.jpg")
        label = root / "labels/train/a.txt"
    else:
        _image(root / "train/images/a.jpg")
        label = root / "train/labels/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    dataset = yolo.resolve_dataset(root)

    assert dataset.root == root.resolve()
    assert len(yolo.split_image_paths(dataset.splits["train"])) == 1


def test_yolo_resolver_error_lists_attempted_paths(tmp_path: Path) -> None:
    root = tmp_path / "empty_dataset"
    root.mkdir()
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": "duplicated_dataset_name",
                "train": "images/train",
                "val": "valid/images",
                "task": "detect",
                "names": {0: "part"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as error:
        yolo.resolve_dataset(root)

    message = str(error.value)
    assert "解析后的数据集根目录" in message
    assert "train:" in message
    assert "val:" in message
    assert "缺失" in message
