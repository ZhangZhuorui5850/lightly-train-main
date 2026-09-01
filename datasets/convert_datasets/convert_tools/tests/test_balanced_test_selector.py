from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import balanced_test_selector as selector  # noqa: E402


def _make_dataset(
    root: Path,
    *,
    names: dict[int, str],
    samples: dict[str, tuple[tuple[int, ...], tuple[int, int, int]]],
    task: str = "detect",
) -> Path:
    (root / "images/train").mkdir(parents=True)
    (root / "labels/train").mkdir(parents=True)
    config = {
        "path": str(root),
        "train": "images/train",
        "task": task,
        "nc": len(names),
        "names": names,
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    for index, (stem, (class_ids, color)) in enumerate(samples.items()):
        Image.new("RGB", (12, 8), color).save(root / f"images/train/{stem}.png")
        if task == "segment":
            rows = [
                f"{class_id} 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9"
                for class_id in class_ids
            ]
        else:
            rows = [
                f"{class_id} 0.5 0.5 0.2 {0.2 + index * 0.01:.2f}"
                for class_id in class_ids
            ]
        (root / f"labels/train/{stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""),
            encoding="utf-8",
        )
    return root / "data.yaml"


def test_previous_is_exclusion_only_and_count_is_final_size(tmp_path: Path) -> None:
    previous_yaml = _make_dataset(
        tmp_path / "previous",
        names={99: "completely_unrelated_taxonomy"},
        samples={"manual": ((99,), (200, 20, 20))},
    )
    (tmp_path / "previous/labels/train/manual.txt").write_text(
        "这份标签故意无法解析，排除集合只读取图片\n",
        encoding="utf-8",
    )
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={3: "cat", 7: "dog"},
        samples={
            "renamed_manual": ((7,), (10, 10, 10)),
            "cat": ((3,), (20, 200, 20)),
            "dog": ((7,), (200, 200, 20)),
            "both": ((3, 7), (20, 20, 200)),
        },
    )
    shutil.copy2(
        tmp_path / "previous/images/train/manual.png",
        tmp_path / "source/images/train/renamed_manual.png",
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        previous_path=previous_yaml,
        count=2,
        seed=11,
        workers=1,
    )

    assert len(plan.excluded_source_samples) == 1
    assert len(plan.selected) == 2
    assert {3, 7}.issubset(
        {class_id for sample in plan.selected for class_id in sample.class_ids}
    )
    assert all(
        sample.image_path.name != "renamed_manual.png" for sample in plan.selected
    )

    output = selector.write_selection(
        plan,
        output=tmp_path / "final_test",
        image_mode="copy",
        clean=False,
    )
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "cat", 1: "dog"}
    assert len(list((output / "images/test").iterdir())) == 2
    report = json_load(output / "selection_report.json")
    assert report["requested_test_images"] == 2
    assert report["final_test_images"] == 2
    assert report["source_images_excluded"] == 1
    assert "base_unique_images" not in report


def test_plain_image_directory_can_be_used_as_exclusion_set(tmp_path: Path) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "cat"},
        samples={
            "excluded": ((0,), (200, 20, 20)),
            "selected": ((0,), (20, 200, 20)),
        },
    )
    exclusion_dir = tmp_path / "manual_images"
    exclusion_dir.mkdir()
    shutil.copy2(
        tmp_path / "source/images/train/excluded.png",
        exclusion_dir / "renamed.png",
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        previous_path=exclusion_dir,
        count=1,
        workers=1,
    )

    assert len(plan.exclusion_images) == 1
    assert len(plan.excluded_source_samples) == 1
    assert plan.selected[0].image_path.name == "selected.png"


def test_instance_segmentation_output_preserves_polygon(tmp_path: Path) -> None:
    exclusion_dir = tmp_path / "manual_images"
    exclusion_dir.mkdir()
    Image.new("RGB", (12, 8), (200, 20, 20)).save(exclusion_dir / "old.png")
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={9: "scratch"},
        samples={"candidate": ((9,), (20, 200, 20))},
        task="segment",
    )
    plan = selector.build_selection_plan(
        source_path=source_yaml,
        previous_path=exclusion_dir,
        count=1,
        workers=1,
        require_all_previous_matched=False,
    )
    output = selector.write_selection(
        plan,
        output=tmp_path / "seg_test",
        image_mode="copy",
        clean=False,
    )

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "segment"
    tokens = next((output / "labels/test").glob("*.txt")).read_text().split()
    assert len(tokens) == 9
    assert tokens[0] == "0"


def test_reference_yaml_controls_full_class_order_and_label_ids(tmp_path: Path) -> None:
    exclusion_dir = tmp_path / "manual_images"
    exclusion_dir.mkdir()
    Image.new("RGB", (12, 8), (200, 20, 20)).save(exclusion_dir / "old.png")
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "scratch", 1: "crack"},
        samples={"candidate": ((0, 1), (20, 200, 20))},
        task="segment",
    )
    reference_yaml = tmp_path / "reference.yaml"
    reference_yaml.write_text(
        yaml.safe_dump(
            {"nc": 3, "names": {0: "crack", 1: "dent", 2: "scratch"}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        previous_path=exclusion_dir,
        reference_path=reference_yaml,
        count=1,
        workers=1,
        require_all_previous_matched=False,
    )
    output = selector.write_selection(
        plan,
        output=tmp_path / "aligned_test",
        image_mode="copy",
        clean=False,
    )

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "crack", 1: "dent", 2: "scratch"}
    rows = next((output / "labels/test").glob("*.txt")).read_text().splitlines()
    assert [int(row.split()[0]) for row in rows] == [2, 0]


def test_reference_yaml_filters_images_and_extra_annotations(tmp_path: Path) -> None:
    exclusion_dir = tmp_path / "manual_images"
    exclusion_dir.mkdir()
    Image.new("RGB", (12, 8), (200, 20, 20)).save(exclusion_dir / "old.png")
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "scratch", 1: "unused"},
        samples={
            "target": ((0,), (20, 200, 20)),
            "irrelevant": ((1,), (20, 20, 200)),
            "mixed": ((0, 1), (200, 200, 20)),
        },
        task="segment",
    )
    reference_yaml = tmp_path / "reference.yaml"
    reference_yaml.write_text(
        yaml.safe_dump(
            {"nc": 2, "names": {0: "dent", 1: "scratch"}},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        previous_path=exclusion_dir,
        reference_path=reference_yaml,
        count=2,
        workers=1,
        require_all_previous_matched=False,
    )
    assert plan.reference_filtered_image_count == 1
    assert plan.reference_filtered_annotation_count == 2
    assert {sample.logical_stem for sample in plan.selected} == {"target", "mixed"}

    output = selector.write_selection(
        plan,
        output=tmp_path / "filtered_test",
        image_mode="copy",
        clean=False,
    )
    labels = sorted((output / "labels/test").glob("*.txt"))
    assert len(labels) == 2
    assert all(
        {int(row.split()[0]) for row in path.read_text().splitlines()} == {1}
        for path in labels
    )


def test_source_same_content_with_complementary_labels_is_merged(
    tmp_path: Path,
) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "cat", 1: "dog"},
        samples={
            "same_cat": ((0,), (200, 20, 20)),
            "same_dog": ((1,), (20, 20, 200)),
        },
    )
    shutil.copy2(
        tmp_path / "source/images/train/same_cat.png",
        tmp_path / "source/images/train/same_dog.png",
    )
    exclusion_dir = tmp_path / "manual_images"
    exclusion_dir.mkdir()
    Image.new("RGB", (12, 8), (20, 200, 20)).save(exclusion_dir / "old.png")

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        previous_path=exclusion_dir,
        count=1,
        workers=1,
    )

    assert plan.source_duplicate_count == 1
    assert len(plan.available_candidates) == 1
    assert plan.selected[0].class_ids == {0, 1}
    assert plan.source_annotation_events[0]["added_annotations"] == 1

    with pytest.raises(ValueError, match="内容相同、标注不同"):
        selector.build_selection_plan(
            source_path=source_yaml,
            previous_path=exclusion_dir,
            count=1,
            workers=1,
            duplicate_annotation_policy="error",
        )


def test_main_interactively_prompts_for_final_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "cat"},
        samples={
            "one": ((0,), (20, 200, 20)),
            "two": ((0,), (20, 20, 200)),
        },
    )
    exclusion_dir = tmp_path / "manual_images"
    exclusion_dir.mkdir()
    Image.new("RGB", (12, 8), (200, 20, 20)).save(exclusion_dir / "old.png")
    answers = iter(["", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    exit_code = selector.main(
        [
            "--source",
            str(source_yaml),
            "--previous",
            str(exclusion_dir),
            "--out",
            str(tmp_path / "interactive_output"),
                "--image-mode",
                "copy",
                "--allow-unmatched-previous",
                "--yes",
        ]
    )

    assert exit_code == 0
    assert len(list((tmp_path / "interactive_output/images/test").iterdir())) == 1


def test_long_tail_threshold_excludes_rare_class_images(tmp_path: Path) -> None:
    samples = {
        **{
            f"common_{index}": ((0,), (20 + index, 100, 30))
            for index in range(5)
        },
        "rare_0": ((1,), (180, 20, 20)),
        "rare_1": ((1,), (190, 20, 20)),
        "mixed": ((0, 1), (200, 30, 20)),
    }
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "common", 1: "rare"},
        samples=samples,
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        count=3,
        workers=1,
        min_class_images=4,
        rare_class_policy="exclude-images",
        long_tail_thresholds=(5, 4, 3),
    )

    assert plan.eligible_class_ids == {0}
    assert plan.rare_class_ids == {1}
    assert plan.rare_filtered_image_count == 3
    assert all(sample.class_ids == {0} for sample in plan.selected)
    summary = {row["threshold"]: row for row in plan.long_tail_summary}
    assert summary[4]["below_class_count"] == 1
    assert summary[4]["below_class_ratio"] == pytest.approx(0.5)


def test_representative_strategy_avoids_forced_rare_class_coverage(
    tmp_path: Path,
) -> None:
    samples = {
        **{
            f"common_{index}": ((0,), (20 + index, 100, 30))
            for index in range(8)
        },
        "rare_0": ((1,), (180, 20, 20)),
        "rare_1": ((1,), (190, 20, 20)),
    }
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "common", 1: "rare"},
        samples=samples,
    )

    representative = selector.build_selection_plan(
        source_path=source_yaml,
        count=2,
        workers=1,
        selection_strategy="representative",
    )
    coverage = selector.build_selection_plan(
        source_path=source_yaml,
        count=2,
        workers=1,
        selection_strategy="coverage",
    )

    assert representative.selected_image_counts == {0: 2}
    assert coverage.selected_image_counts == {0: 1, 1: 1}
    assert representative.missing_selected_classes == [1]


def test_source_split_scope_and_optional_previous(tmp_path: Path) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "train_only", 1: "evaluation"},
        samples={"train_item": ((0,), (20, 100, 30))},
    )
    root = source_yaml.parent
    for split, color in (("val", (30, 120, 30)), ("test", (40, 140, 30))):
        (root / f"images/{split}").mkdir(parents=True)
        (root / f"labels/{split}").mkdir(parents=True)
        Image.new("RGB", (12, 8), color).save(root / f"images/{split}/{split}.png")
        (root / f"labels/{split}/{split}.txt").write_text(
            "1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    config.update({"val": "images/val", "test": "images/test"})
    source_yaml.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        count=2,
        source_splits=("val", "test"),
        workers=1,
    )

    assert plan.exclusion_path is None
    assert plan.source_splits == ("val", "test")
    assert {sample.split for sample in plan.selected} == {"val", "test"}
    assert 0 in plan.rare_class_ids
    assert plan.selected_image_counts == {1: 2}


def test_seg_keep_incidental_preserves_complete_polygon_labels(tmp_path: Path) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "common", 1: "rare"},
        samples={
            "common_0": ((0,), (20, 100, 30)),
            "common_1": ((0,), (30, 110, 30)),
            "mixed": ((0, 1), (40, 120, 30)),
        },
        task="segment",
    )
    plan = selector.build_selection_plan(
        source_path=source_yaml,
        count=3,
        workers=1,
        min_class_images=2,
        rare_class_policy="keep-incidental",
    )
    output = selector.write_selection(
        plan,
        output=tmp_path / "seg_output",
        image_mode="copy",
        clean=False,
    )

    rows = [
        line.split()
        for path in (output / "labels/test").glob("*.txt")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert {int(row[0]) for row in rows} == {0, 1}
    assert all(len(row) == 9 for row in rows)
    report = json_load(output / "selection_report.json")
    assert report["task"] == "segment"
    assert report["rare_class_ids"] == [1]


def test_default_long_tail_report_has_ten_image_stages_to_one_hundred(
    tmp_path: Path,
) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "part"},
        samples={"one": ((0,), (20, 100, 30))},
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        count=1,
        workers=1,
    )

    assert [row["threshold"] for row in plan.long_tail_summary] == list(
        range(10, 101, 10)
    )


def test_fast_mode_skips_image_decode_and_content_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_yaml = _make_dataset(
        tmp_path / "source",
        names={0: "part"},
        samples={"one": ((0,), (20, 100, 30))},
    )

    monkeypatch.setattr(
        selector,
        "_hash_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("快速模式触发了内容哈希")
        ),
    )
    monkeypatch.setattr(
        selector,
        "_analyze_or_raise",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("快速模式触发了逐图完整校验")
        ),
    )

    plan = selector.build_selection_plan(
        source_path=source_yaml,
        count=1,
        workers=1,
        deduplicate_source=False,
        deep_validate=False,
    )

    assert len(plan.selected) == 1
    assert plan.deep_validate is False
    assert plan.deduplicate_source is False


def json_load(path: Path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))
