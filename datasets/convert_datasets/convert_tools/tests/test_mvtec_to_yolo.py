from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import mvtec_to_yolo as converter  # noqa: E402
import dataset_discovery as discovery  # noqa: E402
import output_naming  # noqa: E402


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((60, 80, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _mask(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((60, 80), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (29, 29), 255, thickness=-1)
    cv2.rectangle(mask, (50, 35), (69, 49), 255, thickness=-1)
    assert cv2.imwrite(str(path), mask)


def _mvtec(root: Path) -> Path:
    _image(root / "gear" / "train" / "good" / "a.png")
    _image(root / "gear" / "test" / "good" / "b.png")
    _image(root / "gear" / "test" / "scratch" / "c.png")
    _mask(root / "gear" / "ground_truth" / "scratch" / "c_mask.png")
    return root


def test_discover_standard_mvtec_root_and_single_category(tmp_path: Path) -> None:
    source = _mvtec(tmp_path / "mvtec")
    samples, issues = converter.discover_samples(source)
    assert issues == []
    assert len(samples) == 3
    assert converter.discover_category_roots(source) == [(source / "gear").resolve()]
    assert converter.discover_category_roots(source / "gear") == [
        (source / "gear").resolve()
    ]


def test_auto_discovery_exposes_mvtec_conversion(tmp_path: Path) -> None:
    source = _mvtec(tmp_path / "mvtec")
    candidates = discovery.scan_datasets(tmp_path)
    candidate = next(item for item in candidates if item.kind == "mvtec")
    assert candidate.path == source.resolve()
    assert candidate.image_count == 3
    assert candidate.annotation_count == 1
    assert candidate.class_count == 1
    assert discovery.conversion_actions(candidate) == ("mvtec-to-yolo",)
    assert output_naming.default_output_dir(source, "mvtec-to-yolo") == (
        tmp_path / "mvtec__mvtec_to_yolo"
    )


def test_convert_both_outputs_seg_and_det_with_empty_good_labels(tmp_path: Path) -> None:
    source = _mvtec(tmp_path / "mvtec")
    output = tmp_path / "yolo"
    result = converter.convert(source, output, verbose=False)

    assert result["converted"] == 3
    seg = output / "dataset_seg"
    det = output / "dataset_det"
    assert yaml.safe_load((seg / "data.yaml").read_text(encoding="utf-8"))["task"] == "segment"
    assert yaml.safe_load((det / "data.yaml").read_text(encoding="utf-8"))["task"] == "detect"
    assert len(list((seg / "images" / "train").glob("*.png"))) == 3
    assert len(list((det / "images" / "train").glob("*.png"))) == 3
    assert (seg / "labels" / "train" / "gear__good__a.txt").read_text() == ""
    assert (det / "labels" / "train" / "gear__good__b.txt").read_text() == ""

    seg_lines = (
        seg / "labels" / "train" / "gear__scratch__c.txt"
    ).read_text().splitlines()
    det_lines = (
        det / "labels" / "train" / "gear__scratch__c.txt"
    ).read_text().splitlines()
    assert len(seg_lines) == 2
    assert all(len(line.split()) >= 7 for line in seg_lines)
    assert len(det_lines) == 2
    assert all(len(line.split()) == 5 for line in det_lines)


def test_preserve_split_and_object_defect_class_mode(tmp_path: Path) -> None:
    source = _mvtec(tmp_path / "mvtec")
    output = tmp_path / "seg"
    result = converter.convert(
        source,
        output,
        task="segment",
        split_mode="preserve",
        class_mode="object-defect",
        verbose=False,
    )
    assert result["names"] == ["gear__scratch"]
    assert (output / "images" / "train" / "gear__good__a.png").is_file()
    assert (output / "images" / "test" / "gear__good__b.png").is_file()
    assert (output / "images" / "test" / "gear__scratch__c.png").is_file()


def test_missing_mask_is_reported_and_anomaly_is_skipped(tmp_path: Path) -> None:
    source = _mvtec(tmp_path / "mvtec")
    (source / "gear" / "ground_truth" / "scratch" / "c_mask.png").unlink()
    _image(source / "gear" / "test" / "dent" / "d.png")
    _mask(source / "gear" / "ground_truth" / "dent" / "d_mask.png")
    output = tmp_path / "det"
    result = converter.convert(source, output, task="detect", verbose=False)
    assert result["scan_issues"] == 1
    rows = list(csv.DictReader(
        (output / "conversion_report.csv").open(encoding="utf-8-sig")
    ))
    assert any(row["status"] == "missing_mask" and row["stem"] == "c" for row in rows)
    assert not (output / "images" / "train" / "gear__scratch__c.png").exists()


def test_cli_runs_end_to_end(tmp_path: Path) -> None:
    source = _mvtec(tmp_path / "mvtec")
    output = tmp_path / "det"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "mvtec_to_yolo.py"),
            "--src",
            str(source),
            "--out",
            str(output),
            "--task",
            "detect",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "data.yaml").is_file()
    assert len(list((output / "labels" / "train").glob("*.txt"))) == 3


def test_interactive_cli_scans_source_and_uses_default_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = _mvtec(tmp_path / "mvtec")
    answers = iter(("", "", ""))
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert converter.main(["--datasets", str(tmp_path)]) == 0
    output = source.parent / "mvtec__mvtec_to_yolo"
    assert (output / "dataset_seg" / "data.yaml").is_file()
    assert (output / "dataset_det" / "data.yaml").is_file()
