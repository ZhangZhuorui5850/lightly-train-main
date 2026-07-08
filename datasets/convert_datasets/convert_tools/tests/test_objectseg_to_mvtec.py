# datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import objectseg_to_mvtec as om  # noqa: E402

NAMES = ["锈蚀", "裂纹", "污迹"]  # cls 0,1,2


def _tri(cx: float, cy: float, r: float = 0.08):
    """一个小三角多边形(归一化坐标)。"""
    return [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]


def _write_label(p: Path, entries):
    """entries: list[(cls, [(x,y),...])];  空 list = good 图(空标签)。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, pts in entries:
        coords = " ".join(f"{x:.4f} {y:.4f}" for x, y in pts)
        lines.append(f"{cls} {coords}")
    p.write_text("\n".join(lines))


def make_src(root: Path) -> Path:
    """最小 YOLO-seg 源:只需要 data.yaml + labels/(images/ 不必要)。"""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": NAMES}, allow_unicode=True), encoding="utf-8"
    )
    _write_label(root / "labels" / "train" / "a.txt", [(0, _tri(0.3, 0.3))])                 # 锈蚀
    _write_label(root / "labels" / "train" / "b.txt", [(0, _tri(0.3, 0.3)), (1, _tri(0.7, 0.7))])  # 锈蚀+裂纹
    _write_label(root / "labels" / "val" / "c.txt", [(2, _tri(0.5, 0.5))])                   # 污迹
    _write_label(root / "labels" / "train" / "d.txt", [])                                    # 空/good
    return root


def _img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.full((64, 64, 3), 128, np.uint8))


def make_staging(root: Path) -> Path:
    """物体 管道: a,b ; 物体 阀门: c,d。"""
    for stem in ("a", "b"):
        _img(root / "管道" / f"{stem}.jpg")
    for stem in ("c", "d"):
        _img(root / "阀门" / f"{stem}.jpg")
    return root


def test_build_label_index_maps_stem_to_split(tmp_path):
    src = make_src(tmp_path / "src")
    index = om.build_label_index(src)
    assert set(index) == {"a", "b", "c", "d"}
    assert index["a"][1] == "train"
    assert index["c"][1] == "val"
    assert index["a"][0].name == "a.txt"


def test_build_label_index_raises_on_duplicate_stem(tmp_path):
    src = make_src(tmp_path / "src")
    _write_label(src / "labels" / "test" / "a.txt", [(0, _tri(0.5, 0.5))])  # 与 train/a 重名
    with pytest.raises(om.DuplicateStemError):
        om.build_label_index(src)


def test_scan_staging_lists_objects_and_images(tmp_path):
    staging = make_staging(tmp_path / "staging")
    objects, conflicts = om.scan_staging(staging)
    assert set(objects) == {"管道", "阀门"}
    assert sorted(p.stem for p in objects["管道"]) == ["a", "b"]
    assert conflicts == {}


def test_scan_staging_flags_stem_in_two_objects(tmp_path):
    staging = make_staging(tmp_path / "staging")
    _img(staging / "阀门" / "a.jpg")  # a 同时在 管道 和 阀门
    objects, conflicts = om.scan_staging(staging)
    assert "a" in conflicts
    assert set(conflicts["a"]) == {"管道", "阀门"}


def test_scan_staging_same_stem_diff_ext_in_one_object_is_not_conflict(tmp_path):
    staging = make_staging(tmp_path / "staging")
    _img(staging / "管道" / "a.png")  # 管道 里已有 a.jpg,再加同名不同扩展的 a.png
    _objects, conflicts = om.scan_staging(staging)
    assert "a" not in conflicts  # 同一物体内不算跨物体冲突


def _run(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    result = om.convert(staging, src, out, clean=True, verbose=False)
    return out, result


def test_convert_builds_object_category_structure(tmp_path):
    out, _ = _run(tmp_path)
    assert (out / "管道" / "test" / "锈蚀" / "a.png").exists()
    assert (out / "管道" / "ground_truth" / "锈蚀" / "a_mask.png").exists()
    assert (out / "管道" / "train" / "good").is_dir()
    assert (out / "阀门" / "train" / "good").is_dir()


def test_convert_duplicates_multidefect_image_into_each_defect(tmp_path):
    out, _ = _run(tmp_path)
    assert (out / "管道" / "test" / "锈蚀" / "b.png").exists()
    assert (out / "管道" / "test" / "裂纹" / "b.png").exists()
    assert (out / "管道" / "ground_truth" / "锈蚀" / "b_mask.png").exists()
    assert (out / "管道" / "ground_truth" / "裂纹" / "b_mask.png").exists()


def test_convert_mask_is_split_per_defect(tmp_path):
    out, _ = _run(tmp_path)
    m_rust = cv2.imread(str(out / "管道" / "ground_truth" / "锈蚀" / "b_mask.png"), cv2.IMREAD_GRAYSCALE)
    m_crack = cv2.imread(str(out / "管道" / "ground_truth" / "裂纹" / "b_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(m_rust)).issubset({0, 255})
    assert m_rust[19, 19] == 255 and m_rust[45, 45] == 0
    assert m_crack[45, 45] == 255 and m_crack[19, 19] == 0


def test_convert_empty_label_goes_to_test_good(tmp_path):
    out, _ = _run(tmp_path)
    assert (out / "阀门" / "test" / "good" / "d.png").exists()


def test_convert_writes_manifest_csv(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    om.convert(staging, src, out, clean=True, verbose=False)

    manifest = out / "object_manifest.csv"
    assert manifest.exists()
    import csv as _csv
    rows = list(_csv.DictReader(manifest.open(encoding="utf-8")))
    by_stem = {r["stem"]: r for r in rows}
    assert by_stem["b"]["object"] == "管道"
    assert by_stem["b"]["defects"] == "锈蚀;裂纹"
    assert by_stem["c"]["orig_split"] == "val"
    assert by_stem["a"]["src_label"] == "labels/train/a.txt"


def test_cli_runs_end_to_end(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    import subprocess
    script = TOOLS / "objectseg_to_mvtec.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--staging", str(staging), "--src", str(src), "--out", str(out), "--clean"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "管道" / "test" / "锈蚀" / "a.png").exists()
    assert (out / "object_manifest.csv").exists()
