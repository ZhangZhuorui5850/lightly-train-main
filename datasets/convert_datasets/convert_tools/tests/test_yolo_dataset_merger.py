from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import semantic_class_editor as common  # noqa: E402
import yolo_dataset_merger as merger  # noqa: E402


def _make_dataset(
    root: Path,
    names: dict[int, str],
    labels: dict[str, str],
    *,
    task: str = "detect",
    split: str = "train",
) -> Path:
    root.mkdir(parents=True)
    config = {
        "path": str(root),
        split: f"images/{split}",
        "task": task,
        "nc": len(names),
        "names": names,
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    for stem, text in labels.items():
        image_path = root / "images" / split / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (30, 60, 90)).save(image_path)
        label_path = root / "labels" / split / f"{stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(text, encoding="utf-8")
    return root


def test_merge_two_datasets_collects_full_plan_then_writes_once(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "original",
        {0: "car", 1: "truck"},
        {"same": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"},
    )
    second = _make_dataset(
        tmp_path / "generated",
        {0: "truck", 1: "car", 2: "bus"},
        {"same": "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n2 0.2 0.2 0.1 0.1\n"},
    )
    inventory = merger.build_inventory([first, second])

    assert inventory.class_names == {
        0: "dataset1_car",
        1: "dataset1_truck",
        2: "dataset2_truck",
        3: "dataset2_car",
        4: "dataset2_bus",
    }
    assert [0, 3] in inventory.analysis.similar_name_groups
    assert [1, 2] in inventory.analysis.similar_name_groups

    plan = common.EditPlan(
        merges=[
            common.MergeSpec(ids=[0, 3], name="car"),
            common.MergeSpec(ids=[1, 2], name="truck"),
        ],
        renames={4: "bus"},
    )
    output = tmp_path / "merged"
    report = merger.merge_datasets(inventory, output, plan)

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "detect"
    assert config["nc"] == 3
    assert config["names"] == {0: "car", 1: "truck", 2: "bus"}
    assert (output / "labels/train/same.txt").read_text(
        encoding="utf-8"
    ) == "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"
    assert (output / "labels/train/same__2.txt").read_text(
        encoding="utf-8"
    ) == (
        "1 0.4 0.4 0.2 0.2\n"
        "0 0.6 0.6 0.2 0.2\n"
        "2 0.2 0.2 0.1 0.1\n"
    )
    assert (output / "images/train/same.jpg").is_file()
    assert (output / "images/train/same__2.jpg").is_file()
    assert report["splits"]["train"]["renamed_samples"] == 1
    assert report["sources"]["dataset1"]["old_to_final"] == {0: 0, 1: 1}
    assert report["sources"]["dataset2"]["old_to_final"] == {0: 1, 1: 0, 2: 2}


def test_no_merge_plan_keeps_source_classes_independent(tmp_path: Path) -> None:
    first = _make_dataset(tmp_path / "a", {0: "car"}, {"a": "0 0.5 0.5 0.2 0.2\n"})
    second = _make_dataset(tmp_path / "b", {0: "car"}, {"b": "0 0.5 0.5 0.2 0.2\n"})
    inventory = merger.build_inventory([first, second])
    output = tmp_path / "merged"

    merger.merge_datasets(inventory, output, common.EditPlan())

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "dataset1_car", 1: "dataset2_car"}
    assert (output / "labels/train/a.txt").read_text(
        encoding="utf-8"
    ).startswith("0 ")
    assert (output / "labels/train/b.txt").read_text(
        encoding="utf-8"
    ).startswith("1 ")


def test_unknown_source_id_becomes_user_editable_temporary_class(tmp_path: Path) -> None:
    first = _make_dataset(tmp_path / "a", {0: "car"}, {"a": "3 0.5 0.5 0.2 0.2\n"})
    second = _make_dataset(tmp_path / "b", {0: "car"}, {"b": "0 0.5 0.5 0.2 0.2\n"})

    inventory = merger.build_inventory([first, second])

    first_source = inventory.sources[0]
    provisional = first_source.old_to_provisional[3]
    assert inventory.class_names[provisional] == "dataset1_unknown_3"
    assert inventory.analysis.class_stats[provisional].pixel_count == 1


def test_interactive_loop_collects_multiple_merges_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_dataset(
        tmp_path / "a",
        {0: "car", 1: "truck"},
        {"a": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"},
    )
    second = _make_dataset(
        tmp_path / "b",
        {0: "truck", 1: "car"},
        {"b": "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n"},
    )
    inventory = merger.build_inventory([first, second])
    answers = iter(
        [
            "n",                 # 跳过 car 自动候选
            "n",                 # 跳过 truck 自动候选
            "m 0,3 = car",
            "m 1,2 = truck",
            "done",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    plan = common.interactive_plan(
        SimpleNamespace(class_names=inventory.class_names),
        inventory.analysis,
        annotation_label="标注",
        drop_effect="删除对应标注行",
        unknown_effect="删除对应标注行",
        merge_default_names=inventory.base_names,
    )

    assert plan is not None
    assert [(merge.ids, merge.name) for merge in plan.merges] == [
        ([0, 3], "car"),
        ([1, 2], "truck"),
    ]
    assert not (tmp_path / "merged").exists()


def test_reference_plan_rewrites_swapped_ids_in_one_batch(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "car", 1: "truck"},
        {"a": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"},
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "truck", 1: "car", 2: "bus"},
        {"b": "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n2 0.2 0.2 0.1 0.1\n"},
    )
    inventory = merger.build_inventory(
        [first, second], use_first_source_names=True
    )
    plan = common.recommend_reference_merge_plan(
        inventory.class_names,
        inventory.analysis,
        reference_ids=set(inventory.sources[0].old_to_provisional.values()),
        base_names=inventory.base_names,
    )
    resolved = common.resolve_plan(plan, inventory.class_names)

    assert resolved.output_names == {0: "car", 1: "truck", 2: "bus"}
    assert merger.build_final_source_maps(inventory, resolved) == {
        "dataset1": {0: 0, 1: 1},
        "dataset2": {0: 1, 1: 0, 2: 2},
    }


def test_choose_merge_reference_moves_selected_yaml_to_front(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / "a/data.yaml", tmp_path / "b/data.yaml"]

    ordered = common.choose_merge_reference(paths, selected=2)

    assert ordered == [paths[1], paths[0]]


def test_main_uses_one_yes_for_the_complete_reference_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "car", 1: "truck"},
        {"a": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n"},
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "truck", 1: "car", 2: "bus"},
        {"b": "0 0.4 0.4 0.2 0.2\n1 0.6 0.6 0.2 0.2\n2 0.2 0.2 0.1 0.1\n"},
    )
    output = tmp_path / "merged"
    answers = iter(["2", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = merger.main(
        [
            "--src",
            str(first),
            "--src",
            str(second),
            "--out",
            str(output),
            "--format",
            "yolo-detect",
        ]
    )

    assert result == 0
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["names"] == {0: "truck", 1: "car", 2: "bus"}
    assert (output / "labels/train/a.txt").read_text(encoding="utf-8") == (
        "1 0.5 0.5 0.2 0.2\n0 0.3 0.3 0.1 0.1\n"
    )


def test_v2_plan_locks_untouched_classes_across_source_reordering(
    tmp_path: Path,
) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "cat"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "dog"}, {"b": "0 0.5 0.5 0.2 0.2\n"}
    )
    original = merger.build_inventory([first, second])
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(
            merger.serialize_merge_plan(original, common.EditPlan()),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    reordered = merger.build_inventory([second, first])
    loaded = merger.load_merge_plan(plan_path, reordered)
    resolved = common.resolve_plan(loaded, reordered.class_names)
    final_maps = merger.build_final_source_maps(reordered, resolved)
    by_root = {
        source.dataset.root.name: final_maps[source.key][0]
        for source in reordered.sources
    }

    assert by_root == {"first": 0, "second": 1}
    assert resolved.output_names == {0: "dataset1_cat", 1: "dataset2_dog"}


def test_plan_rejects_taxonomy_mode_change(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "cat"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "cat"}, {"b": "0 0.5 0.5 0.2 0.2\n"}
    )
    original = merger.build_inventory([first, second], taxonomy_mode="namespace")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(merger.serialize_merge_plan(original, common.EditPlan())),
        encoding="utf-8",
    )
    union = merger.build_inventory([first, second], taxonomy_mode="union-by-name")

    with pytest.raises(ValueError, match="taxonomy"):
        merger.load_merge_plan(plan_path, union)


def test_hash_dedupe_rejects_conflicting_rewritten_labels(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "cat"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "dog"}, {"b": "0 0.4 0.4 0.2 0.2\n"}
    )
    inventory = merger.build_inventory([first, second])
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match="label 不一致"):
        merger.merge_datasets(
            inventory,
            output,
            common.EditPlan(),
            duplicate_policy="hash-dedupe",
        )
    assert not output.exists()


def test_all_taxonomy_modes_and_mapping_file(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "vehicle"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "vehicle"}, {"b": "0 0.5 0.5 0.2 0.2\n"}
    )
    assert len(merger.build_inventory([first, second]).class_names) == 2
    assert len(
        merger.build_inventory([first, second], taxonomy_mode="strict").class_names
    ) == 1
    assert len(
        merger.build_inventory(
            [first, second], taxonomy_mode="union-by-name"
        ).class_names
    ) == 1
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text(
        yaml.safe_dump({"dataset1": {0: "vehicle"}, "dataset2": {0: "vehicle"}}),
        encoding="utf-8",
    )
    assert len(
        merger.build_inventory(
            [first, second],
            taxonomy_mode="mapping-file",
            mapping_file=mapping,
        ).class_names
    ) == 1


def test_zero_classes_are_rejected_before_output(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "cat"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "dog"}, {"b": "0 0.5 0.5 0.2 0.2\n"}
    )
    inventory = merger.build_inventory([first, second])
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match="类别为空"):
        merger.merge_datasets(
            inventory,
            output,
            common.EditPlan(drop_ids=set(inventory.class_names)),
        )
    assert not output.exists()


def test_orphan_drop_is_audited(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "cat"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "dog"}, {"b": "0 0.5 0.5 0.2 0.2\n"}
    )
    (first / "labels/train/orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    inventory = merger.build_inventory(
        [first, second], orphan_policy="drop"
    )
    report = merger.merge_datasets(
        inventory,
        tmp_path / "merged",
        common.EditPlan(),
        orphan_policy="drop",
    )

    assert report["splits"]["train"]["orphan_labels"] == 1


@pytest.mark.parametrize("workers", [1, 4])
def test_merge_annotations_unions_detection_rows_deterministically(
    tmp_path: Path,
    workers: int,
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "car"},
        {
            "a": (
                "0 0.2 0.2 0.1 0.1\n"
                "0 0.5 0.5 0.1 0.1\n"
            )
        },
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "car"},
        {
            "b": (
                "0 0.2 0.2 0.1 0.1\n"
                "0 0.8 0.8 0.1 0.1\n"
            )
        },
    )
    inventory = merger.build_inventory(
        [first, second], taxonomy_mode="union-by-name"
    )
    output = tmp_path / f"merged-workers-{workers}"

    report = merger.merge_datasets(
        inventory,
        output,
        common.EditPlan(),
        duplicate_policy="merge-annotations",
        workers=workers,
    )

    assert (output / "labels/train/a.txt").read_text(
        encoding="utf-8"
    ) == (
        "0 0.2 0.2 0.1 0.1\n"
        "0 0.5 0.5 0.1 0.1\n"
        "0 0.8 0.8 0.1 0.1\n"
    )
    assert not (output / "images/train/b.jpg").exists()
    assert not (output / "labels/train/b.txt").exists()
    split_report = report["splits"]["train"]
    assert split_report["images"] == 1
    assert split_report["labels"] == 1
    assert split_report["kept_annotations"] == 3
    assert split_report["duplicate_samples"] == 1
    assert split_report["merged_samples"] == 1
    assert split_report["merged_annotations"] == 1
    assert split_report["deduplicated_annotations"] == 1
    assert split_report["conflicted_annotations"] == 0


def test_merge_annotations_default_conflict_is_transactional(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "cat"},
        {"a": "0 0.5 0.5 0.2 0.2\n"},
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "dog"},
        {"b": "0 0.5 0.5 0.2 0.2\n"},
    )
    inventory = merger.build_inventory([first, second])
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match=r"标注冲突.*type=same"):
        merger.merge_datasets(
            inventory,
            output,
            common.EditPlan(),
            duplicate_policy="merge-annotations",
        )

    assert not output.exists()


@pytest.mark.parametrize(
    ("conflict_policy", "expected", "merged", "replaced"),
    [
        ("keep-first", "0 0.5 0.5 0.2 0.2\n", 0, 0),
        ("keep-last", "1 0.5 0.5 0.2 0.2\n", 1, 1),
        (
            "keep-both",
            "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n",
            1,
            0,
        ),
    ],
)
def test_merge_annotations_detection_conflict_policies(
    tmp_path: Path,
    conflict_policy: str,
    expected: str,
    merged: int,
    replaced: int,
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "cat"},
        {"a": "0 0.5 0.5 0.2 0.2\n"},
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "dog"},
        {"b": "0 0.5 0.5 0.2 0.2\n"},
    )
    inventory = merger.build_inventory([first, second])
    output = tmp_path / "merged"

    report = merger.merge_datasets(
        inventory,
        output,
        common.EditPlan(),
        duplicate_policy="merge-annotations",
        duplicate_annotation_conflict_policy=conflict_policy,
    )

    assert (output / "labels/train/a.txt").read_text(
        encoding="utf-8"
    ) == expected
    split_report = report["splits"]["train"]
    assert split_report["conflicted_annotations"] == 1
    assert split_report["annotation_conflict_pairs"] == 1
    assert split_report["merged_annotations"] == merged
    assert split_report["replaced_annotations"] == replaced


def test_merge_annotations_detection_geometry_conflict_uses_iou(
    tmp_path: Path,
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "car"},
        {"a": "0 0.500 0.5 0.4 0.4\n"},
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "car"},
        {"b": "0 0.501 0.5 0.4 0.4\n"},
    )
    inventory = merger.build_inventory(
        [first, second], taxonomy_mode="union-by-name"
    )

    with pytest.raises(ValueError, match=r"type=conflict.*threshold=0.950000"):
        merger.merge_datasets(
            inventory,
            tmp_path / "merged",
            common.EditPlan(),
            duplicate_policy="merge-annotations",
        )


def test_merge_annotations_polygon_rotation_deduplicates_exact_geometry(
    tmp_path: Path,
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "part"},
        {"a": "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"},
        task="segment",
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "part"},
        {
            "b": (
                "0 0.4 0.4 0.4 0.1 0.1 0.1 0.1 0.4\n"
                "0 0.6 0.6 0.8 0.6 0.8 0.8 0.6 0.8\n"
            )
        },
        task="segment",
    )
    inventory = merger.build_inventory(
        [first, second],
        forced_kind="yolo_instance",
        taxonomy_mode="union-by-name",
    )
    output = tmp_path / "merged"

    report = merger.merge_datasets(
        inventory,
        output,
        common.EditPlan(),
        duplicate_policy="merge-annotations",
    )

    assert (output / "labels/train/a.txt").read_text(
        encoding="utf-8"
    ) == (
        "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"
        "0 0.6 0.6 0.8 0.6 0.8 0.8 0.6 0.8\n"
    )
    assert report["format"] == "yolo_instance"
    assert report["splits"]["train"]["deduplicated_annotations"] == 1
    assert report["splits"]["train"]["merged_annotations"] == 1


def test_merge_annotations_convex_polygon_geometry_conflict(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "part"},
        {"a": "0 0.100 0.1 0.500 0.1 0.500 0.5 0.100 0.5\n"},
        task="segment",
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "part"},
        {"b": "0 0.101 0.1 0.501 0.1 0.501 0.5 0.101 0.5\n"},
        task="segment",
    )
    inventory = merger.build_inventory(
        [first, second],
        forced_kind="yolo_instance",
        taxonomy_mode="union-by-name",
    )

    with pytest.raises(ValueError, match=r"type=conflict.*iou_or_upper_bound"):
        merger.merge_datasets(
            inventory,
            tmp_path / "merged",
            common.EditPlan(),
            duplicate_policy="merge-annotations",
        )


def test_merge_annotations_ambiguous_concave_polygon_uses_conflict_policy(
    tmp_path: Path,
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "part"},
        {
            "a": (
                "0 0.1 0.1 0.5 0.1 0.5 0.2 "
                "0.2 0.2 0.2 0.5 0.1 0.5\n"
            )
        },
        task="segment",
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "part"},
        {
            "b": (
                "0 0.1 0.1 0.5 0.1 0.5 0.21 "
                "0.2 0.21 0.2 0.5 0.1 0.5\n"
            )
        },
        task="segment",
    )
    inventory = merger.build_inventory(
        [first, second],
        forced_kind="yolo_instance",
        taxonomy_mode="union-by-name",
    )

    with pytest.raises(ValueError, match=r"type=ambiguous.*threshold=0.900000"):
        merger.merge_datasets(
            inventory,
            tmp_path / "merged",
            common.EditPlan(),
            duplicate_policy="merge-annotations",
            duplicate_annotation_conflict_iou=0.9,
        )


def test_merge_annotations_stays_within_one_split(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "car"},
        {"a": "0 0.2 0.2 0.1 0.1\n"},
        split="train",
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "car"},
        {"b": "0 0.8 0.8 0.1 0.1\n"},
        split="val",
    )
    inventory = merger.build_inventory(
        [first, second], taxonomy_mode="union-by-name"
    )
    output = tmp_path / "merged"

    report = merger.merge_datasets(
        inventory,
        output,
        common.EditPlan(),
        duplicate_policy="merge-annotations",
    )

    assert (output / "images/train/a.jpg").is_file()
    assert (output / "images/val/b.jpg").is_file()
    assert report["splits"]["train"]["merged_samples"] == 0
    assert report["splits"]["val"]["merged_samples"] == 0
    assert any("跨 split 重复图片" in warning for warning in report["warnings"])


def test_merge_annotations_cli_exposes_conflict_controls_and_reflink() -> None:
    defaults = merger.parse_args([])
    assert defaults.duplicate_annotation_conflict_policy == "error"
    assert defaults.duplicate_annotation_conflict_iou == 0.95

    args = merger.parse_args(
        [
            "--duplicate-policy",
            "merge-annotations",
            "--duplicate-annotation-conflict-policy",
            "keep-last",
            "--duplicate-annotation-conflict-iou",
            "0.8",
            "--image-mode",
            "reflink",
            "--id-policy",
            "preserve",
        ]
    )
    assert args.duplicate_policy == "merge-annotations"
    assert args.duplicate_annotation_conflict_policy == "keep-last"
    assert args.duplicate_annotation_conflict_iou == 0.8
    assert args.image_mode == "reflink"
    assert args.id_policy == "preserve"


@pytest.mark.parametrize("threshold", [0.0, -0.1, 1.01])
def test_merge_annotations_rejects_invalid_conflict_iou(
    tmp_path: Path,
    threshold: float,
) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "car"}, {"a": "0 0.5 0.5 0.2 0.2\n"}
    )
    second = _make_dataset(
        tmp_path / "second", {0: "car"}, {"b": "0 0.5 0.5 0.2 0.2\n"}
    )
    inventory = merger.build_inventory([first, second])

    with pytest.raises(ValueError, match=r"conflict_iou"):
        merger.merge_datasets(
            inventory,
            tmp_path / "merged",
            common.EditPlan(),
            duplicate_policy="merge-annotations",
            duplicate_annotation_conflict_iou=threshold,
        )
