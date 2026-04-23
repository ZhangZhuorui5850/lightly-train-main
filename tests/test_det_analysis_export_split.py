from pathlib import Path

from tool_lib.det_analysis import (
    allocate_box_targets_per_split,
    allocate_image_targets_per_split,
    assign_candidates_to_new_splits,
)
from tool_lib.det_shared import ExportImageCandidate


def _candidate(name: str, class_box_counts: dict[int, int]) -> ExportImageCandidate:
    rel_path = Path(f"{name}.jpg")
    return ExportImageCandidate(
        split_name="train",
        rel_split_image_dir=Path("images/train"),
        rel_split_label_dir=Path("labels/train"),
        rel_path=rel_path,
        src_image_path=Path("/tmp") / rel_path,
        src_label_path=Path("/tmp") / rel_path.with_suffix(".txt"),
        filtered_lines=tuple(
            f"{class_id} 0.5 0.5 0.2 0.2"
            for class_id, count in sorted(class_box_counts.items())
            for _ in range(count)
        ),
        class_box_counts=class_box_counts,
    )


def _covered_classes(candidates: list[ExportImageCandidate]) -> set[int]:
    covered: set[int] = set()
    for candidate in candidates:
        covered.update(candidate.class_box_counts)
    return covered


def test_assign_candidates_preserves_eval_class_coverage_when_possible() -> None:
    selected_candidates = [
        _candidate("all_1", {0: 1, 1: 1, 2: 1}),
        _candidate("all_2", {0: 1, 1: 1, 2: 1}),
        _candidate("all_3", {0: 1, 1: 1, 2: 1}),
        _candidate("train_0", {0: 1}),
        _candidate("train_1", {1: 1}),
        _candidate("train_2", {2: 1}),
    ]
    split_image_targets = {"train": 4, "val": 1, "test": 1}
    kept_class_ids = [0, 1, 2]
    desired_box_targets_by_split = allocate_box_targets_per_split(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )

    assigned_candidates, repartition_summary = assign_candidates_to_new_splits(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        desired_box_targets_by_split=desired_box_targets_by_split,
        kept_class_ids=kept_class_ids,
    )

    assert _covered_classes(assigned_candidates["val"]) == {0, 1, 2}
    assert _covered_classes(assigned_candidates["test"]) == {0, 1, 2}
    assert repartition_summary["missing_image_coverage_counts"] == {"train": 0, "val": 0, "test": 0}


def test_assign_candidates_reports_classes_with_insufficient_split_coverage() -> None:
    selected_candidates = [
        _candidate("class0_a", {0: 1}),
        _candidate("class0_b", {0: 1}),
        _candidate("class1_a", {1: 1}),
        _candidate("class1_b", {1: 1}),
        _candidate("class2_a", {2: 1}),
        _candidate("class2_b", {2: 1}),
    ]
    split_image_targets = {"train": 4, "val": 1, "test": 1}
    kept_class_ids = [0, 1, 2]
    desired_box_targets_by_split = allocate_box_targets_per_split(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )

    _, repartition_summary = assign_candidates_to_new_splits(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        desired_box_targets_by_split=desired_box_targets_by_split,
        kept_class_ids=kept_class_ids,
    )

    assert repartition_summary["limited_coverage_classes"] == {0: 2, 1: 2, 2: 2}
    assert repartition_summary["missing_image_coverage_counts"]["val"] + repartition_summary["missing_image_coverage_counts"]["test"] > 0


def test_allocate_image_targets_per_split_follows_class_level_ratio() -> None:
    selected_candidates = [
        _candidate("all_1", {0: 1}),
        _candidate("all_2", {0: 1}),
        _candidate("all_3", {0: 1}),
        _candidate("all_4", {0: 1}),
        _candidate("all_5", {0: 1}),
        _candidate("all_6", {0: 1}),
        _candidate("all_7", {0: 1}),
        _candidate("all_8", {0: 1}),
        _candidate("all_9", {0: 1}),
        _candidate("all_10", {0: 1}),
    ]
    split_image_targets = {"train": 8, "val": 1, "test": 1}
    kept_class_ids = [0]

    desired_image_targets = allocate_image_targets_per_split(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )

    assert desired_image_targets == {
        "train": {0: 8},
        "val": {0: 1},
        "test": {0: 1},
    }


def test_allocate_image_targets_preserves_split_coverage_for_small_classes() -> None:
    selected_candidates = [
        _candidate("class0_a", {0: 1}),
        _candidate("class0_b", {0: 1}),
        _candidate("class0_c", {0: 1}),
    ]
    split_image_targets = {"train": 8, "val": 1, "test": 1}
    kept_class_ids = [0]

    desired_image_targets = allocate_image_targets_per_split(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )

    assert desired_image_targets == {
        "train": {0: 1},
        "val": {0: 1},
        "test": {0: 1},
    }
