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
