"""Tests for the size-supplement tool (tool_lib/det_size_supplement.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_lib import common as rt
from tool_lib import det_size_supplement as ss


def _make_det_dataset(
    root: Path,
    spec: dict[str, list[tuple[str, list[tuple[int, float, float]]]]],
    names: list[str] | None = None,
):
    """在 root 下造一个最小 YOLO 检测数据集。

    spec: {split: [(stem, [(class_id, box_w_norm, box_h_norm), ...]), ...]}
    每张图为 100x100，框中心固定 (0.5,0.5)。返回 data.yaml 路径。
    """
    rt.import_runtime_dependencies()
    names = list(names) if names is not None else ["obj"]
    image_module = rt.Image
    for split, items in spec.items():
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for stem, boxes in items:
            image_module.new("RGB", (100, 100), color=(127, 127, 127)).save(image_dir / f"{stem}.jpg")
            lines = [f"{cls} 0.5 0.5 {bw} {bh}" for cls, bw, bh in boxes]
            (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    rt.dump_yaml(
        data_yaml,
        {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "task": "detect",
            "nc": len(names),
            "names": names,
        },
    )
    return data_yaml


def _count_images(root: Path) -> int:
    return len(list((root / "images").rglob("*.jpg")))


class TestParseSizeRatio:
    def test_slash_separated_normalizes_to_sum_one(self):
        ratio = ss.parse_size_ratio("33/33/33")
        assert pytest.approx(ratio["small"] + ratio["medium"] + ratio["large"], abs=1e-9) == 1.0
        assert ratio["small"] == pytest.approx(1 / 3, abs=1e-6)
        assert ratio["medium"] == pytest.approx(1 / 3, abs=1e-6)
        assert ratio["large"] == pytest.approx(1 / 3, abs=1e-6)

    def test_space_separated_keeps_proportions(self):
        ratio = ss.parse_size_ratio("30 30 40")
        assert ratio["small"] == pytest.approx(0.3, abs=1e-6)
        assert ratio["medium"] == pytest.approx(0.3, abs=1e-6)
        assert ratio["large"] == pytest.approx(0.4, abs=1e-6)

    def test_colon_separated(self):
        ratio = ss.parse_size_ratio("1:1:2")
        assert ratio["small"] == pytest.approx(0.25, abs=1e-6)
        assert ratio["large"] == pytest.approx(0.5, abs=1e-6)

    def test_rejects_all_zero(self):
        with pytest.raises(ValueError):
            ss.parse_size_ratio("0 0 0")

    def test_rejects_wrong_count(self):
        with pytest.raises(ValueError):
            ss.parse_size_ratio("30/70")


def _cand(key: str, small: int = 0, medium: int = 0, large: int = 0, classes=None):
    return ss.SupplementCandidate(
        key=key,
        buckets={"small": small, "medium": medium, "large": large},
        class_box_counts=classes or {},
    )


class TestSelectSupplementCandidates:
    def test_prefers_size_rich_candidates_for_deficit_bucket(self):
        # base is 10% small / 10% medium / 80% large; small & medium are in deficit.
        base = {"small": 10, "medium": 10, "large": 80}
        candidates = [_cand(f"small{i}", small=10) for i in range(5)]
        candidates += [_cand(f"large{i}", large=10) for i in range(5)]
        selected, summary = ss.select_supplement_candidates(
            base_buckets=base,
            candidates=candidates,
            target_ratio=ss.parse_size_ratio("33/33/33"),
            num_to_add=5,
        )
        assert len(selected) == 5
        # Every pick should be a small-rich image, never a large-only one.
        assert all(c.buckets["large"] == 0 for c in selected)
        base_small_ratio = 10 / 100
        achieved_small_ratio = summary["achieved_ratio"]["small"]
        assert achieved_small_ratio > base_small_ratio

    def test_self_corrects_across_buckets_per_batch(self):
        # base is pure large; with equal-third target, small AND medium are both
        # in deficit. Greedy must not dump everything into small — once small
        # catches up, medium becomes the deficit and should get picked too.
        base = {"small": 0, "medium": 0, "large": 100}
        candidates = [_cand(f"s{i}", small=5) for i in range(50)]
        candidates += [_cand(f"m{i}", medium=5) for i in range(50)]
        selected, _ = ss.select_supplement_candidates(
            base_buckets=base,
            candidates=candidates,
            target_ratio=ss.parse_size_ratio("33/33/33"),
            num_to_add=20,
            batch_size=1,
        )
        picked_small = sum(1 for c in selected if c.buckets["small"] > 0)
        picked_medium = sum(1 for c in selected if c.buckets["medium"] > 0)
        assert picked_small > 0
        assert picked_medium > 0

    def test_caps_at_num_to_add(self):
        base = {"small": 0, "medium": 0, "large": 10}
        candidates = [_cand(f"s{i}", small=5) for i in range(100)]
        selected, summary = ss.select_supplement_candidates(
            base_buckets=base,
            candidates=candidates,
            target_ratio=ss.parse_size_ratio("33/33/33"),
            num_to_add=7,
        )
        assert len(selected) == 7
        assert summary["pool_exhausted"] is False

    def test_pool_exhaustion_reports_shortfall(self):
        base = {"small": 0, "medium": 0, "large": 1000}
        candidates = [_cand(f"s{i}", small=5) for i in range(3)]
        selected, summary = ss.select_supplement_candidates(
            base_buckets=base,
            candidates=candidates,
            target_ratio=ss.parse_size_ratio("33/33/33"),
            num_to_add=50,
        )
        assert len(selected) == 3  # only what the pool had
        assert summary["pool_exhausted"] is True
        assert summary["requested_add"] == 50
        assert summary["selected_add"] == 3
        # small is still far below target -> reported as a positive shortfall.
        assert summary["bucket_shortfall"]["small"] > 0

    def test_grows_beyond_floor_to_meet_ratio(self):
        # Floor is tiny (2) but target wants 50% small. The selector should keep
        # adding small-rich images beyond the floor until small ~ 50%.
        base = {"small": 0, "medium": 0, "large": 100}
        candidates = [_cand(f"s{i}", small=5) for i in range(100)]
        selected, summary = ss.select_supplement_candidates(
            base_buckets=base,
            candidates=candidates,
            target_ratio=ss.parse_size_ratio("50/0/50"),
            num_to_add=2,
        )
        assert len(selected) > 2  # grew past the requested floor
        assert summary["achieved_ratio"]["small"] >= 0.45

    def test_class_bonus_breaks_ties_toward_weak_classes(self):
        # Two candidates contribute identical size buckets; one covers an
        # under-represented class. With class bonus on, that one wins.
        base = {"small": 0, "medium": 0, "large": 100}
        base_classes = {0: 100, 1: 0}  # class 1 is starved
        candidates = [
            _cand("covers_strong", small=5, classes={0: 5}),
            _cand("covers_weak", small=5, classes={1: 5}),
        ]
        selected, _ = ss.select_supplement_candidates(
            base_buckets=base,
            candidates=candidates,
            target_ratio=ss.parse_size_ratio("33/33/33"),
            num_to_add=1,
            base_class_box_counts=base_classes,
            class_bonus_weight=1.0,
        )
        assert selected[0].key == "covers_weak"


class TestBucketLabelLines:
    def test_buckets_by_pixel_area_on_100px_image(self):
        # On a 100x100 image: 10x10=100px -> tiny (<16^2=256),
        # 20x20=400px -> small (<32^2=1024),
        # 50x50=2500px -> medium (<96^2=9216), 100x100=10000px -> large.
        lines = (
            "0 0.5 0.5 0.1 0.1",   # tiny
            "1 0.5 0.5 0.2 0.2",   # small
            "2 0.5 0.5 0.5 0.5",   # medium
            "0 0.5 0.5 1.0 1.0",   # large
        )
        buckets = ss.bucket_label_lines(lines, width=100, height=100)
        assert buckets == {"tiny": 1, "small": 1, "medium": 1, "large": 1}

    def test_ignores_malformed_lines(self):
        lines = ("0 0.5 0.5 0.1 0.1", "garbage", "1 0.5 0.5")
        buckets = ss.bucket_label_lines(lines, width=100, height=100)
        assert buckets == {"tiny": 1, "small": 0, "medium": 0, "large": 0}

    def test_class_box_counts_too(self):
        lines = ("0 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1 0.1", "2 0.5 0.5 1.0 1.0")
        buckets, classes = ss.bucket_label_lines(
            lines, width=100, height=100, with_class_counts=True
        )
        assert buckets == {"tiny": 2, "small": 0, "medium": 0, "large": 1}
        assert classes == {0: 2, 2: 1}


class TestImageSizeCache:
    def test_reads_once_then_memoizes(self, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        calls = []

        def reader(path):
            calls.append(path)
            return (640, 480)

        cache = ss.ImageSizeCache(tmp_path / "cache.json", size_reader=reader)
        assert cache.get(img) == (640, 480)
        assert cache.get(img) == (640, 480)
        assert len(calls) == 1

    def test_persists_across_instances(self, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        calls = []

        def reader(path):
            calls.append(path)
            return (100, 200)

        first = ss.ImageSizeCache(tmp_path / "c.json", size_reader=reader)
        first.get(img)
        first.save()
        second = ss.ImageSizeCache(tmp_path / "c.json", size_reader=reader)
        assert second.get(img) == (100, 200)
        assert len(calls) == 1  # loaded from disk, reader not called again

    def test_invalidates_on_mtime_change(self, tmp_path):
        import os
        import time

        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        sizes = [(10, 10), (20, 20)]
        calls = []

        def reader(path):
            calls.append(path)
            return sizes[len(calls) - 1]

        cache = ss.ImageSizeCache(tmp_path / "c.json", size_reader=reader)
        assert cache.get(img) == (10, 10)
        future = time.time() + 10
        os.utime(img, (future, future))
        assert cache.get(img) == (20, 20)
        assert len(calls) == 2


class TestAllocateSupplementsToSplits:
    def test_eight_one_one_split(self):
        alloc = ss.allocate_supplements_to_splits(100, split_ratio="8:1:1")
        assert alloc == {"train": 80, "val": 10, "test": 10}

    def test_allocation_sums_to_total(self):
        alloc = ss.allocate_supplements_to_splits(97, split_ratio="8:1:1")
        assert alloc["train"] + alloc["val"] + alloc["test"] == 97


class TestBuildOutputDirName:
    def test_encodes_ratio_and_total(self):
        name = ss.build_output_dir_name(
            base_name="dataset_det_A_10000",
            achieved_ratio={"small": 0.33, "medium": 0.30, "large": 0.37},
            total_images=15000,
        )
        assert name == "dataset_det_A_10000__szsup_s33m30l37_n15000"


class TestRemapLinesByName:
    def test_restrict_drops_unknown_classes(self):
        lines = ["0 0.5 0.5 0.1 0.1", "1 0.5 0.5 0.1 0.1"]
        out, name_to_id, added = ss.remap_lines_by_name(
            lines,
            source_id_to_name={0: "a", 1: "b"},
            name_to_id={"a": 0},
            allow_new_classes=False,
        )
        assert out == ["0 0.5 0.5 0.1 0.1"]
        assert name_to_id == {"a": 0}
        assert added == set()

    def test_remaps_id_by_name(self):
        # source id 0 is name "b", which is base id 1 -> line must be relabeled to 1.
        lines = ["0 0.5 0.5 0.1 0.1"]
        out, _, _ = ss.remap_lines_by_name(
            lines,
            source_id_to_name={0: "b", 1: "a"},
            name_to_id={"a": 0, "b": 1},
            allow_new_classes=False,
        )
        assert out == ["1 0.5 0.5 0.1 0.1"]

    def test_allow_new_classes_extends(self):
        lines = ["1 0.5 0.5 0.1 0.1"]
        out, name_to_id, added = ss.remap_lines_by_name(
            lines,
            source_id_to_name={0: "a", 1: "c"},
            name_to_id={"a": 0},
            allow_new_classes=True,
        )
        assert out == ["1 0.5 0.5 0.1 0.1"]
        assert name_to_id == {"a": 0, "c": 1}
        assert added == {"c"}

    def test_restrict_tracks_added_names_even_when_dropped(self):
        """allow_new_classes=False 时 added 仍记录「本应新增的名字」，供 fallback 使用。"""
        lines = ["0 0.5 0.5 0.1 0.1", "1 0.5 0.5 0.1 0.1"]
        out, _, added = ss.remap_lines_by_name(
            lines,
            source_id_to_name={0: "a", 1: "z"},
            name_to_id={"a": 0},
            allow_new_classes=False,
        )
        assert out == ["0 0.5 0.5 0.1 0.1"]  # "z" 被丢弃
        assert added == set()  # False 模式下不追加，所以 added 为空

    def test_mixed_classes_keeps_base_boxes(self):
        """混合图：既有 base 类又有非 base 类的框 → 只保留 base 类的框。"""
        lines = ["0 0.5 0.5 0.1 0.1", "1 0.5 0.5 0.1 0.1", "2 0.5 0.5 0.1 0.1"]
        out, _, added = ss.remap_lines_by_name(
            lines,
            source_id_to_name={0: "base_a", 1: "other_x", 2: "base_b"},
            name_to_id={"base_a": 0, "base_b": 1},
            allow_new_classes=False,
        )
        assert len(out) == 2  # base_a + base_b，other_x 被丢弃
        assert added == set()


class TestSummarizeSize:
    def test_reports_counts_and_ratio(self, tmp_path):
        root = tmp_path / "ds"
        yaml_path = _make_det_dataset(
            root,
            {
                "train": [("a", [(0, 0.2, 0.2)]), ("b", [(0, 1.0, 1.0)])],  # 1 small + 1 large
                "val": [],
                "test": [],
            },
        )
        summary = ss.summarize_size(yaml_path)
        assert summary["image_count"] == 2
        assert summary["size_buckets"] == {"tiny": 0, "small": 1, "medium": 0, "large": 1}
        assert summary["size_ratio"]["small"] == pytest.approx(0.5)
        assert summary["size_ratio"]["large"] == pytest.approx(0.5)


class TestInferSourceDatasetName:
    def test_strips_export_suffix_with_count(self):
        assert ss.infer_source_dataset_name("dataset_det_A_10000") == "dataset_det"

    def test_strips_plain_a_suffix(self):
        assert ss.infer_source_dataset_name("dataset_det_A") == "dataset_det"

    def test_strips_our_szsup_suffix(self):
        assert ss.infer_source_dataset_name("dataset_det_A__szsup_s33m30l37_n15000") == "dataset_det"

    def test_leaves_plain_name_unchanged(self):
        assert ss.infer_source_dataset_name("dataset_det") == "dataset_det"


class TestRunSizeSupplementIntegration:
    def _build(self, tmp_path):
        # Base: 10 large-object images (small ratio = 0%), 8/1/1 split.
        base_root = tmp_path / "ds" / "dataset_det_A"
        base_spec = {
            "train": [(f"base_{i}", [(0, 1.0, 1.0)]) for i in range(8)],
            "val": [("base_v0", [(0, 1.0, 1.0)])],
            "test": [("base_t0", [(0, 1.0, 1.0)])],
        }
        base_yaml = _make_det_dataset(base_root, base_spec)
        # Source pool: 40 small-object images (distinct filenames).
        source_root = tmp_path / "ds" / "dataset_det"
        source_spec = {
            "train": [(f"src_{i}", [(0, 0.2, 0.2)]) for i in range(40)],
            "val": [],
            "test": [],
        }
        source_yaml = _make_det_dataset(source_root, source_spec)
        return base_root, base_yaml, source_root, source_yaml

    def test_supplements_and_writes_new_dataset(self, tmp_path):
        base_root, base_yaml, source_root, source_yaml = self._build(tmp_path)

        out_root = ss.run_size_supplement(
            base_data_yaml=base_yaml,
            source_data_yaml=source_yaml,
            target_ratio=ss.parse_size_ratio("33/33/33"),
            target_total_images=20,
        )

        # New dataset created, named to show what happened, separate from originals.
        assert out_root.exists()
        assert out_root.name.startswith("dataset_det_A__szsup_")
        assert out_root.name.endswith("_n20")
        assert out_root.resolve() != base_root.resolve()

        # Originals untouched.
        assert _count_images(base_root) == 10
        assert _count_images(source_root) == 40

        # Final size and split composition.
        assert _count_images(out_root) == 20
        assert len(list((out_root / "images" / "train").glob("*.jpg"))) == 16  # 8 base + 8 supp
        assert len(list((out_root / "images" / "val").glob("*.jpg"))) == 2
        assert len(list((out_root / "images" / "test").glob("*.jpg"))) == 2

        # Required artifacts present.
        assert (out_root / "data.yaml").exists()
        assert (out_root / "classes.txt").exists()
        assert (out_root / "supplement_report.md").exists()
        report = json.loads((out_root / "supplement_report.json").read_text(encoding="utf-8"))

        # Small-object ratio moved up from base 0%.
        assert report["base_size_ratio"]["small"] == pytest.approx(0.0, abs=1e-9)
        assert report["final_size_ratio"]["small"] > report["base_size_ratio"]["small"]
        assert report["supplemented_images"] == 10
        assert report["supplement_split_allocation"] == {"train": 8, "val": 1, "test": 1}

    def test_handles_class_mismatch_and_exceeds_target(self, tmp_path):
        # Base is a remapped subset: classes ["a","b"] at ids 0,1, all LARGE.
        base_root = tmp_path / "ds" / "dataset_det_A"
        _make_det_dataset(
            base_root,
            {
                "train": [(f"base_{i}", [(0, 1.0, 1.0), (1, 1.0, 1.0)]) for i in range(8)],
                "val": [("bv", [(0, 1.0, 1.0)])],
                "test": [("bt", [(1, 1.0, 1.0)])],
            },
            names=["a", "b"],
        )
        # Source: 4 classes ["x","a","b","y"] at ids 0..3 (note "a" is id 1 here,
        # different from base id 0). Small boxes of "a" (in base) and "y" (not in base).
        source_root = tmp_path / "ds" / "dataset_det"
        _make_det_dataset(
            source_root,
            {
                "train": [(f"src_a_{i}", [(1, 0.2, 0.2)]) for i in range(20)]
                + [(f"src_y_{i}", [(3, 0.2, 0.2)]) for i in range(20)],
                "val": [],
                "test": [],
            },
            names=["x", "a", "b", "y"],
        )

        out_root = ss.run_size_supplement(
            base_data_yaml=base_root / "data.yaml",
            source_data_yaml=source_root / "data.yaml",
            target_ratio=ss.parse_size_ratio("60/0/40"),
            target_total_images=20,  # floor; growth may exceed it
        )

        # No new classes added (default restrict mode).
        out_cfg = rt.load_data_config(out_root / "data.yaml")
        names = rt.normalize_names(out_cfg.get("names"))
        assert set(names.values()) == {"a", "b"}

        # Every label id is within the base id space {0,1} (remapped by name).
        ids = set()
        for txt in (out_root / "labels").rglob("*.txt"):
            for line in txt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(int(float(line.split()[0])))
        assert ids <= {0, 1}

        # Grew past the requested floor of 20 to chase the ratio.
        assert _count_images(out_root) > 20

        report = json.loads((out_root / "supplement_report.json").read_text(encoding="utf-8"))
        assert report["final_size_ratio"]["small"] > report["base_size_ratio"]["small"]
        # "src_y" images contributed no base-class boxes -> excluded as candidates.
        assert report["source_candidate_images"] == 20

    def test_rejects_target_not_larger_than_base(self, tmp_path):
        _, base_yaml, _, source_yaml = self._build(tmp_path)
        with pytest.raises(ValueError):
            ss.run_size_supplement(
                base_data_yaml=base_yaml,
                source_data_yaml=source_yaml,
                target_ratio=ss.parse_size_ratio("33/33/33"),
                target_total_images=10,  # == base size, nothing to add
            )

    def test_fallback_allows_new_classes_when_base_candidates_insufficient(self, tmp_path):
        """base-only 候选不足时，allow_new_classes=True 应自动降级，加入非 base 类图片。"""
        base_root = tmp_path / "ds" / "dataset_det_A"
        _make_det_dataset(
            base_root,
            {
                "train": [(f"base_{i}", [(0, 1.0, 1.0)]) for i in range(8)],
                "val": [("bv", [(0, 1.0, 1.0)])],
                "test": [("bt", [(0, 1.0, 1.0)])],
            },
            names=["a"],
        )
        # Source: 2 张含 base 类 "a" 的小目标图，18 张仅含非 base 类 "z" 的小目标图。
        # 如果只用 base-only，最多补 2 张，不够 target=20。
        source_root = tmp_path / "ds" / "dataset_det"
        _make_det_dataset(
            source_root,
            {
                "train": [(f"src_a_{i}", [(0, 0.2, 0.2)]) for i in range(2)]
                + [(f"src_z_{i}", [(1, 0.2, 0.2)]) for i in range(18)],
                "val": [],
                "test": [],
            },
            names=["a", "z"],
        )

        out_root = ss.run_size_supplement(
            base_data_yaml=base_root / "data.yaml",
            source_data_yaml=source_root / "data.yaml",
            target_ratio=ss.parse_size_ratio("50/0/50"),
            target_total_images=20,
            allow_new_classes=True,
        )

        out_cfg = rt.load_data_config(out_root / "data.yaml")
        names = rt.normalize_names(out_cfg.get("names"))
        # "z" 应被加入（fallback 触发）
        assert "z" in set(names.values())
        assert "a" in set(names.values())

        report = json.loads((out_root / "supplement_report.json").read_text(encoding="utf-8"))
        assert report["used_fallback"] is True
        assert report["non_base_only_images"] == 18
        # floor=10, 加 10 张小目标后 small=10/20=50% 已达标 → growth 停在 10
        assert report["supplemented_images"] >= 10
        assert report["final_size_ratio"]["small"] >= 0.45

    def test_no_fallback_when_enough_base_candidates(self, tmp_path):
        """base-only 候选足够时，即使 allow_new_classes=True 也不触发 fallback。"""
        base_root = tmp_path / "ds" / "dataset_det_A"
        _make_det_dataset(
            base_root,
            {
                "train": [(f"base_{i}", [(0, 1.0, 1.0)]) for i in range(8)],
                "val": [("bv", [(0, 1.0, 1.0)])],
                "test": [("bt", [(0, 1.0, 1.0)])],
            },
            names=["a"],
        )
        # Source: 30 张含 base 类 "a" 的小目标图，10 张仅含非 base 类 "z" 的图。
        # base-only 候选 30 张 > target=20，不需要 fallback。
        source_root = tmp_path / "ds" / "dataset_det"
        _make_det_dataset(
            source_root,
            {
                "train": [(f"src_a_{i}", [(0, 0.2, 0.2)]) for i in range(30)]
                + [(f"src_z_{i}", [(1, 0.2, 0.2)]) for i in range(10)],
                "val": [],
                "test": [],
            },
            names=["a", "z"],
        )

        out_root = ss.run_size_supplement(
            base_data_yaml=base_root / "data.yaml",
            source_data_yaml=source_root / "data.yaml",
            target_ratio=ss.parse_size_ratio("50/0/50"),
            target_total_images=20,
            allow_new_classes=True,
        )

        out_cfg = rt.load_data_config(out_root / "data.yaml")
        names = rt.normalize_names(out_cfg.get("names"))
        # "z" 不应被加入（base-only 够用，fallback 未触发）
        assert set(names.values()) == {"a"}

        report = json.loads((out_root / "supplement_report.json").read_text(encoding="utf-8"))
        assert report["used_fallback"] is False
        assert report["non_base_only_images"] == 10
        assert report["source_candidate_images"] == 30

    def test_base_only_mixed_images_keeps_base_boxes(self, tmp_path):
        """混合图（既有 base 类又有非 base 类框）在 base-only 模式下应保留 base 类框。"""
        base_root = tmp_path / "ds" / "dataset_det_A"
        _make_det_dataset(
            base_root,
            {
                "train": [(f"base_{i}", [(0, 1.0, 1.0)]) for i in range(8)],
                "val": [("bv", [(0, 1.0, 1.0)])],
                "test": [("bt", [(0, 1.0, 1.0)])],
            },
            names=["a"],
        )
        # Source: 混合图（class 0 = base "a" 小框, class 1 = 非 base "z" 大框）。
        # base-only 过滤后应保留 "a" 的小框 → 候选有效。
        source_root = tmp_path / "ds" / "dataset_det"
        _make_det_dataset(
            source_root,
            {
                "train": [
                    (f"mixed_{i}", [(0, 0.2, 0.2), (1, 1.0, 1.0)])
                    for i in range(20)
                ],
                "val": [],
                "test": [],
            },
            names=["a", "z"],
        )

        out_root = ss.run_size_supplement(
            base_data_yaml=base_root / "data.yaml",
            source_data_yaml=source_root / "data.yaml",
            target_ratio=ss.parse_size_ratio("50/0/50"),
            target_total_images=20,
            allow_new_classes=False,
        )

        out_cfg = rt.load_data_config(out_root / "data.yaml")
        names = rt.normalize_names(out_cfg.get("names"))
        # 不应新增类别
        assert set(names.values()) == {"a"}

        report = json.loads((out_root / "supplement_report.json").read_text(encoding="utf-8"))
        # 混合图的 base 类框被保留 → 20 张都是候选
        assert report["source_candidate_images"] == 20
        assert report["supplemented_images"] >= 10
        # 非 base 类框被丢弃 → 最终只有 base 类
        ids = set()
        for txt in (out_root / "labels").rglob("*.txt"):
            for line in txt.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(int(float(line.split()[0])))
        assert ids <= {0}  # 只有 base 类 id=0


class TestInteractiveEntryWiring:
    def test_scripted_run_produces_dataset(self, tmp_path, monkeypatch):
        import size_supplement as entry

        base_root = tmp_path / "ds" / "dataset_det_A"
        _make_det_dataset(
            base_root,
            {
                "train": [(f"base_{i}", [(0, 1.0, 1.0)]) for i in range(8)],
                "val": [("base_v0", [(0, 1.0, 1.0)])],
                "test": [("base_t0", [(0, 1.0, 1.0)])],
            },
        )
        source_root = tmp_path / "ds" / "dataset_det"
        _make_det_dataset(
            source_root,
            {
                "train": [(f"src_{i}", [(0, 0.2, 0.2)]) for i in range(40)],
                "val": [],
                "test": [],
            },
        )
        base_yaml = base_root / "data.yaml"
        source_yaml = source_root / "data.yaml"

        monkeypatch.setattr(entry.it, "list_dataset_yaml_candidates", lambda **kw: [base_yaml, source_yaml])
        scripted = iter(["1", "1", "33/33/33", "20", "y"])
        monkeypatch.setattr("builtins.input", lambda prompt="": next(scripted))

        entry.main()

        produced = list((tmp_path / "ds").glob("dataset_det_A__szsup_*"))
        assert len(produced) == 1
        assert (produced[0] / "supplement_report.json").exists()
        assert _count_images(produced[0]) == 20

