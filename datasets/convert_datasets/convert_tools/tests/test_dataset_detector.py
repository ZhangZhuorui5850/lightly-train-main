from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import dataset_detector as detector  # noqa: E402


def _dataset(root: Path, *, task: str, row: str, timestamp: int) -> None:
    (root / "images/train").mkdir(parents=True)
    (root / "labels/train").mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "task": task,
                "names": {0: "part"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    Image.new("RGB", (8, 6)).save(root / "images/train/a.jpg")
    (root / "labels/train/a.txt").write_text(row, encoding="utf-8")
    for path in [*root.rglob("*"), root]:
        os.utime(path, (timestamp, timestamp))


def test_shared_detector_filters_task_and_sorts_by_modified_time(
    tmp_path: Path,
) -> None:
    _dataset(
        tmp_path / "older_det",
        task="detect",
        row="0 0.5 0.5 0.2 0.2\n",
        timestamp=1_000,
    )
    _dataset(
        tmp_path / "newer_det",
        task="detect",
        row="0 0.5 0.5 0.2 0.2\n",
        timestamp=2_000,
    )
    _dataset(
        tmp_path / "seg",
        task="segment",
        row="0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
        timestamp=3_000,
    )

    detected = detector.detect_datasets(
        tmp_path,
        kinds=detector.TASK_KINDS["det"],
        show_progress=False,
    )

    assert [item.path.name for item in detected] == ["newer_det", "older_det"]


def test_shared_detector_recognizes_classification_yaml(tmp_path: Path) -> None:
    root = tmp_path / "classification"
    (root / "train/widget").mkdir(parents=True)
    (root / "train/widget/a.jpg").write_bytes(b"image")
    (root / "project.yml").write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "train",
                "task": "classify",
                "names": {0: "widget"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    detected = detector.detect_datasets(
        tmp_path,
        kinds=detector.TASK_KINDS["cls"],
        show_progress=False,
    )

    assert len(detected) == 1
    assert detected[0].kind == "image_classification"
    assert detected[0].image_count == 1


def test_shared_detector_recognizes_raw_imagefolder_dataset(tmp_path: Path) -> None:
    root = tmp_path / "classification"
    for split in ("train", "val"):
        for class_name in ("cat", "dog"):
            image = root / split / class_name / f"{class_name}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6)).save(image)

    detected = detector.detect_datasets(
        tmp_path,
        kinds=detector.TASK_KINDS["cls"],
        show_progress=False,
    )

    assert len(detected) == 1
    assert detected[0].path == root.resolve()
    assert detected[0].config_path is None
    assert detected[0].image_count == 4


def test_shared_detector_uses_label_content_before_declared_task_filter(
    tmp_path: Path,
) -> None:
    root = tmp_path / "legacy_export"
    _dataset(
        root,
        task="segment",
        row="0 0.5 0.5 0.2 0.2\n",
        timestamp=1_000,
    )

    detected = detector.detect_datasets(
        tmp_path,
        kinds=detector.TASK_KINDS["det"],
        show_progress=False,
    )

    assert [item.path for item in detected] == [root.resolve()]
    assert detected[0].kind == "yolo_detection"


def test_shared_detector_scans_multiple_roots_and_deduplicates_overlap(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _dataset(
        first / "older",
        task="detect",
        row="0 0.5 0.5 0.2 0.2\n",
        timestamp=1_000,
    )
    _dataset(
        second / "newer",
        task="detect",
        row="0 0.5 0.5 0.2 0.2\n",
        timestamp=2_000,
    )

    detected = detector.detect_datasets(
        [tmp_path, second],
        kinds=detector.TASK_KINDS["det"],
        show_progress=False,
    )

    assert [item.path.name for item in detected] == ["newer", "older"]
