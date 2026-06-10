"""Tests for tool_lib/det_review_sample.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_lib import common as rt
from tool_lib import det_review_sample as drs


# ── helpers ───────────────────────────────────────────────


def _make_det_dataset(
    root: Path,
    spec: dict[str, list[tuple[str, list[tuple[int, float, float, float, float]]]]],
    names: list[str] | None = None,
):
    """造一个最小 YOLO 检测数据集。

    spec: {split: [(stem, [(class_id, cx, cy, w, h), ...]), ...]}
    每张图 640x640。返回 data.yaml 路径。
    """
    rt.import_runtime_dependencies()
    names = list(names) if names is not None else ["cls_a", "cls_b", "cls_c"]
    img_mod = rt.Image
    for split, items in spec.items():
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem, boxes in items:
            img_mod.new("RGB", (640, 640), color=(127, 127, 127)).save(
                img_dir / f"{stem}.jpg"
            )
            lines = [f"{cid} {cx} {cy} {w} {h}" for cid, cx, cy, w, h in boxes]
            (lbl_dir / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
    data_yaml = root / "data.yaml"
    split_keys = {i: n for i, n in enumerate(names)}
    cfg_dict: dict[str, Any] = {
        "path": str(root.resolve()),
        "task": "detect",
        "nc": len(names),
        "names": split_keys,
    }
    for split_name in spec:
        cfg_dict[split_name] = f"images/{split_name}"
    rt.dump_yaml(
        data_yaml,
        cfg_dict,
    )
    return data_yaml


# ── TestScanDataset ───────────────────────────────────────


class TestScanDataset:
    def test_basic_scan(self, tmp_path: Path):
        data_yaml = _make_det_dataset(
            tmp_path,
            {
                "train": [
                    ("img0", [(0, 0.5, 0.5, 0.1, 0.1), (1, 0.3, 0.3, 0.2, 0.2)]),
                    ("img1", [(0, 0.4, 0.4, 0.3, 0.3)]),
                ],
                "val": [("v0", [(2, 0.5, 0.5, 0.05, 0.05)])],
            },
        )
        infos, names, nc, report = drs.scan_dataset(data_yaml)
        assert nc == 3
        assert len(names) == 3
        assert report.total_images == 3
        assert report.total_boxes == 4
        assert len(report.zero_instance_classes) == 0

    def test_zero_instance_classes(self, tmp_path: Path):
        data_yaml = _make_det_dataset(
            tmp_path,
            {"train": [("img0", [(0, 0.5, 0.5, 0.1, 0.1)])]},
            names=["a", "b", "c"],
        )
        _, _, _, report = drs.scan_dataset(data_yaml)
        assert 1 in report.zero_instance_classes
        assert 2 in report.zero_instance_classes

    def test_id_out_of_bounds(self, tmp_path: Path):
        data_yaml = _make_det_dataset(
            tmp_path,
            {"train": [("img0", [(0, 0.5, 0.5, 0.1, 0.1)])]},
            names=["a", "b"],
        )
        # 手动写一个越界标签
        lbl = tmp_path / "labels" / "train" / "img0.txt"
        lbl.write_text("0 0.5 0.5 0.1 0.1\n5 0.3 0.3 0.2 0.2\n")
        _, _, _, report = drs.scan_dataset(data_yaml)
        assert any(cid == 5 for _, cid in report.id_out_of_bounds)


# ── TestGeometryFlags ─────────────────────────────────────


class TestGeometryFlags:
    def _make_info(self, boxes_raw: list[tuple[int, float, float, float, float]], w=640, h=640):
        boxes = [drs.BoxAnnotation(cid, cx, cy, bw, bh) for cid, cx, cy, bw, bh in boxes_raw]
        return drs.ImageReviewInfo(
            rel_path=Path("test.jpg"),
            split="train",
            src_image_path=Path("/dev/null"),
            src_label_path=Path("/dev/null"),
            boxes=boxes,
            image_width=w,
            image_height=h,
        )

    def test_dup_detected(self):
        # 两个完全重叠的框
        info = self._make_info([
            (0, 0.5, 0.5, 0.1, 0.1),
            (0, 0.5, 0.5, 0.1, 0.1),
        ])
        drs.detect_geometry_flags([info])
        assert "dup" in info.boxes[0].flags
        assert "dup" in info.boxes[1].flags

    def test_near_dup_not_flagged_at_low_iou(self):
        # 两个不重叠的框
        info = self._make_info([
            (0, 0.2, 0.2, 0.1, 0.1),
            (0, 0.8, 0.8, 0.1, 0.1),
        ])
        drs.detect_geometry_flags([info])
        assert "dup" not in info.boxes[0].flags
        assert "dup" not in info.boxes[1].flags

    def test_conflict_detected_sibling(self):
        # 两个 sibling 类别高度重叠
        groups = [drs.ClassGroup("veh", "sibling", frozenset({0, 1}))]
        info = self._make_info([
            (0, 0.5, 0.5, 0.2, 0.2),
            (1, 0.5, 0.5, 0.2, 0.2),
        ])
        drs.detect_geometry_flags([info], class_groups=groups)
        assert any("conflict" in f for f in info.boxes[0].flags)

    def test_parent_child_no_conflict(self):
        # parent_child 关系不报冲突
        groups = [drs.ClassGroup("veh", "parent_child", frozenset({0, 1}))]
        info = self._make_info([
            (0, 0.5, 0.5, 0.2, 0.2),
            (1, 0.5, 0.5, 0.2, 0.2),
        ])
        drs.detect_geometry_flags([info], class_groups=groups)
        assert all("conflict" not in f for f in info.boxes[0].flags)

    def test_bad_box_tiny(self):
        # 极小框 (< 4px)
        info = self._make_info([(0, 0.5, 0.5, 0.001, 0.001)], w=640, h=640)
        drs.detect_geometry_flags([info])
        assert "bad_box:tiny" in info.boxes[0].flags

    def test_bad_box_degenerate(self):
        # wh = 0
        info = self._make_info([(0, 0.5, 0.5, 0.0, 0.1)])
        drs.detect_geometry_flags([info])
        assert "bad_box:degenerate" in info.boxes[0].flags

    def test_dense_flag(self):
        # 创建 25 张图，前 20 张框数多 → 应该被标 dense
        infos = []
        for i in range(25):
            n = 50 if i < 20 else 1
            boxes = [(0, 0.5, 0.5, 0.05, 0.05)] * n
            infos.append(self._make_info(boxes))
        drs.detect_geometry_flags(infos, dense_top_n=20)
        dense_count = sum(1 for info in infos if "dense" in info.image_flags)
        assert dense_count == 20


# ── TestComputeClassQuotas ────────────────────────────────


class TestComputeClassQuotas:
    def test_basic_quota(self):
        names = {0: "a", 1: "b", 2: "c"}
        counts = {0: 1000, 1: 100, 2: 10}
        quotas = drs.compute_class_quotas(names, counts, k=3, alpha=3.0, cap=10)
        # 1000 张: clamp(3, 3+round(3*log10(1000)), 10) = clamp(3, 3+9, 10) = 10
        assert quotas[0] == 10
        # 100 张: clamp(3, 3+round(3*2), 10) = clamp(3, 9, 10) = 9
        assert quotas[1] == 9
        # 10 张: clamp(3, 3+round(3*1), 10) = clamp(3, 6, 10) = 6
        assert quotas[2] == 6

    def test_rare_class_uses_actual(self):
        names = {0: "a"}
        counts = {0: 2}
        quotas = drs.compute_class_quotas(names, counts, k=3, alpha=3.0, cap=10)
        assert quotas[0] == 2  # 稀有类：有几张拿几张

    def test_zero_class(self):
        names = {0: "a"}
        counts: dict[int, int] = {}
        quotas = drs.compute_class_quotas(names, counts, k=3, alpha=3.0, cap=10)
        assert quotas[0] == 0


# ── TestSelectReviewSubset ────────────────────────────────


class TestSelectReviewSubset:
    def _make_info(self, cid: int, has_problem: bool = False):
        boxes = [drs.BoxAnnotation(cid, 0.5, 0.5, 0.1, 0.1)]
        if has_problem:
            boxes[0].flags.add("dup")
        return drs.ImageReviewInfo(
            rel_path=Path(f"img_{cid}_{id(boxes)}.jpg"),
            split="train",
            src_image_path=Path("/dev/null"),
            src_label_path=Path("/dev/null"),
            boxes=boxes,
            image_width=640,
            image_height=640,
        )

    def test_covers_all_quotas(self):
        # 3 类，每类需要 1 张
        infos = [
            self._make_info(0),
            self._make_info(1),
            self._make_info(2),
        ]
        quotas = {0: 1, 1: 1, 2: 1}
        selected = drs.select_review_subset(infos, quotas, seed=0)
        covered = set()
        for info in selected:
            for box in info.boxes:
                covered.add(box.class_id)
        assert covered == {0, 1, 2}

    def test_problem_images_prioritized(self):
        # 类 0 有 1 张问题图 + 1 张正常图
        normal = self._make_info(0, has_problem=False)
        problem = self._make_info(0, has_problem=True)
        infos = [normal, problem, self._make_info(1)]
        quotas = {0: 1, 1: 1}
        selected = drs.select_review_subset(infos, quotas, seed=0)
        selected_rels = {str(s.rel_path) for s in selected}
        assert str(problem.rel_path) in selected_rels


# ── TestExport ────────────────────────────────────────────


class TestExport:
    def test_export_creates_expected_files(self, tmp_path: Path):
        rt.import_runtime_dependencies()
        data_yaml = _make_det_dataset(
            tmp_path / "src",
            {
                "train": [
                    ("img0", [(0, 0.5, 0.5, 0.1, 0.1), (1, 0.3, 0.3, 0.2, 0.2)]),
                    ("img1", [(2, 0.5, 0.5, 0.05, 0.05)]),
                ],
            },
        )
        infos, names, nc, report = drs.scan_dataset(data_yaml)
        quotas = {0: 1, 1: 1, 2: 1}
        selected = drs.select_review_subset(infos, quotas, seed=0)
        out_dir = tmp_path / "review"
        drs.export_review_subset(selected, out_dir, names, report, quotas)

        assert (out_dir / "classes.txt").exists()
        assert (out_dir / "qc_report.md").exists()
        assert (out_dir / "qc_report.csv").exists()
        assert (out_dir / "images").is_dir()
        assert (out_dir / "labels").is_dir()
        # JSONs now alongside images in images/

        # LabelMe JSON 校验
        json_files = list((out_dir / "images").glob("*.json"))
        assert len(json_files) >= 1
        for jf in json_files:
            data = json.loads(jf.read_text(encoding="utf-8"))
            assert data["version"] == "5.5.0"
            assert "shapes" in data
            assert "flags" in data
            for shape in data["shapes"]:
                assert "label" in shape
                assert "points" in shape
                assert shape["shape_type"] == "rectangle"

    def test_classes_txt_content(self, tmp_path: Path):
        rt.import_runtime_dependencies()
        data_yaml = _make_det_dataset(
            tmp_path / "src",
            {"train": [("img0", [(0, 0.5, 0.5, 0.1, 0.1)])]},
            names=["person", "car", "dog"],
        )
        infos, names, nc, report = drs.scan_dataset(data_yaml)
        quotas = {0: 1, 1: 0, 2: 0}
        selected = drs.select_review_subset(infos, quotas, seed=0)
        out_dir = tmp_path / "review"
        drs.export_review_subset(selected, out_dir, names, report, quotas)

        classes = (out_dir / "classes.txt").read_text(encoding="utf-8").strip().splitlines()
        assert classes == ["person", "car", "dog"]
