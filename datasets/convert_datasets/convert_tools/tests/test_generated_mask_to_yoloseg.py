from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import generated2seg_interactive as gi  # noqa: E402
import generated_mask_to_yoloseg as gm  # noqa: E402


def _write_pair(root: Path, obj: str, defect: str, stem: str, *, inverted: bool = False) -> None:
    image_dir = root / obj / defect / "image"
    fg_dir = root / obj / defect / "fg"
    image_dir.mkdir(parents=True, exist_ok=True)
    fg_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 80, 3), 120, dtype=np.uint8)
    mask = np.full((64, 80), 255 if inverted else 0, dtype=np.uint8)
    cv2.rectangle(mask, (10, 15), (35, 45), 0 if inverted else 255, thickness=-1)
    cv2.imwrite(str(image_dir / f"{stem}.png"), image)
    cv2.imwrite(str(fg_dir / f"{stem}.png"), mask)


def _dataset(root: Path) -> Path:
    _write_pair(root, "bearing", "scratch", "0")
    _write_pair(root, "bearing", "abrasion", "0")
    _write_pair(root, "gear", "scratch", "0")
    return root


def test_discover_pairs_extracts_object_defect_and_matches_stems(tmp_path):
    src = _dataset(tmp_path / "generated")
    pairs, issues = gm.discover_pairs(src)
    assert len(pairs) == 3
    assert issues == []
    assert {(p.object_name, p.defect_name, p.image.stem) for p in pairs} == {
        ("bearing", "scratch", "0"),
        ("bearing", "abrasion", "0"),
        ("gear", "scratch", "0"),
    }


def test_discover_pairs_accepts_single_defect_leaf_as_source(tmp_path):
    src = tmp_path / "generated"
    _write_pair(src, "bearing", "scratch", "5")
    pairs, issues = gm.discover_pairs(src / "bearing" / "scratch")
    assert issues == []
    assert len(pairs) == 1
    assert pairs[0].object_name == "bearing"
    assert pairs[0].defect_name == "scratch"


def test_discover_pairs_accepts_object_directory_as_source(tmp_path):
    src = tmp_path / "generated"
    _write_pair(src, "bearing", "scratch", "5")
    pairs, issues = gm.discover_pairs(src / "bearing")
    assert issues == []
    assert len(pairs) == 1
    assert pairs[0].object_name == "bearing"


def test_discover_pairs_reports_missing_sides(tmp_path):
    leaf = tmp_path / "generated" / "bearing" / "scratch"
    (leaf / "image").mkdir(parents=True)
    (leaf / "fg").mkdir()
    cv2.imwrite(str(leaf / "image" / "image_only.png"), np.zeros((16, 16, 3), np.uint8))
    cv2.imwrite(str(leaf / "fg" / "mask_only.png"), np.zeros((16, 16), np.uint8))
    pairs, issues = gm.discover_pairs(tmp_path / "generated")
    assert pairs == []
    assert {issue.status for issue in issues} == {"missing_mask", "missing_image"}


def test_mask_to_polygons_auto_inverts_white_background():
    mask = np.full((50, 60), 255, np.uint8)
    cv2.rectangle(mask, (10, 10), (30, 30), 0, thickness=-1)
    polygons, inverted, ratio = gm.mask_to_polygons(mask)
    assert inverted is True
    assert len(polygons) == 1
    assert 0.1 < ratio < 0.3


def test_convert_without_yaml_generates_complete_yolo_dataset(tmp_path):
    src = _dataset(tmp_path / "generated")
    out = tmp_path / "output_seg"
    result = gm.convert(src, out, verbose=False)

    assert result["converted"] == 3
    assert result["names"] == ["abrasion", "scratch"]
    assert len(list((out / "images" / "train").glob("*.png"))) == 3
    labels = sorted((out / "labels" / "train").glob("*.txt"))
    assert len(labels) == 3
    assert (out / "images" / "val").is_dir()
    assert (out / "images" / "test").is_dir()

    config = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "segment"
    assert config["nc"] == 2
    assert config["names"] == {0: "abrasion", 1: "scratch"}
    assert (out / "classes.txt").read_text(encoding="utf-8").splitlines() == ["abrasion", "scratch"]

    for label in labels:
        parts = label.read_text(encoding="utf-8").strip().split()
        assert int(parts[0]) in (0, 1)
        coords = [float(value) for value in parts[1:]]
        assert len(coords) >= 6 and len(coords) % 2 == 0
        assert all(0.0 <= value <= 1.0 for value in coords)


def test_convert_merges_same_defect_class_and_avoids_numeric_name_collisions(tmp_path):
    src = _dataset(tmp_path / "generated")
    out = tmp_path / "out"
    gm.convert(src, out, verbose=False)
    names = {path.name for path in (out / "images" / "train").iterdir()}
    assert names == {
        "bearing__abrasion__0.png",
        "bearing__scratch__0.png",
        "gear__scratch__0.png",
    }
    mapping = json.loads((out / "class_mapping.json").read_text(encoding="utf-8"))
    scratch = next(entry for entry in mapping.values() if entry["name"] == "scratch")
    assert scratch["sources"] == ["bearing/scratch", "gear/scratch"]


def test_local_yaml_order_is_used_and_new_defect_is_appended(tmp_path):
    src = _dataset(tmp_path / "generated")
    (src / "data.yaml").write_text(
        yaml.safe_dump({"names": {0: "scratch", 1: "old_class"}}, sort_keys=False),
        encoding="utf-8",
    )
    pairs, _ = gm.discover_pairs(src)
    names, config = gm.resolve_class_names(src, pairs)
    assert config == src / "data.yaml"
    assert names == ["scratch", "old_class", "abrasion"]


def test_size_mismatch_is_skipped_and_written_to_report(tmp_path):
    src = tmp_path / "generated"
    _write_pair(src, "bearing", "scratch", "0")
    cv2.imwrite(
        str(src / "bearing" / "scratch" / "fg" / "0.png"),
        np.zeros((32, 32), np.uint8),
    )
    out = tmp_path / "out"
    result = gm.convert(src, out, verbose=False)
    assert result["skipped"] == 1
    rows = list(csv.DictReader((out / "conversion_report.csv").open(encoding="utf-8-sig")))
    assert rows[0]["status"] == "size_mismatch"
    assert list((out / "images" / "train").iterdir()) == []


def test_scan_candidates_groups_object_defect_leaves_under_dataset_root(tmp_path):
    root = tmp_path / "datasets"
    generated = _dataset(root / "returned_data")
    candidates = gi.scan_candidates(root)
    assert candidates == [generated.resolve()]


def test_core_cli_runs_end_to_end(tmp_path):
    src = _dataset(tmp_path / "generated")
    out = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "generated_mask_to_yoloseg.py"),
            "--src", str(src),
            "--out", str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out / "data.yaml").is_file()
    assert len(list((out / "labels" / "train").glob("*.txt"))) == 3
