from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import dataset_merger  # noqa: E402
import semantic_class_editor as common  # noqa: E402
import semantic_dataset_merger as merger  # noqa: E402


def _make_dataset(
    root: Path,
    classes: dict[int, object],
    mask: np.ndarray,
) -> Path:
    root.mkdir(parents=True)
    config = {
        "task": "semantic_segmentation",
        "classes": classes,
        "train": {
            "images": "images/train",
            "masks": "masks/train",
        },
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    image_path = root / "images/train/same.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (mask.shape[1], mask.shape[0]), (40, 70, 90)).save(image_path)
    mask_path = root / "masks/train/same.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(mask).save(mask_path)
    return root


def _add_sample(root: Path, stem: str, mask: np.ndarray, color: int) -> None:
    image_path = root / "images/train" / f"{stem}.jpg"
    Image.new(
        "RGB",
        (mask.shape[1], mask.shape[0]),
        (color, (color + 31) % 256, (color + 67) % 256),
    ).save(image_path)
    Image.fromarray(mask).save(root / "masks/train" / f"{stem}.png")


def _mask_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "masks").rglob("*.png"))
    }


def test_semantic_merge_rewrites_masks_after_complete_plan(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "original",
        {0: "background", 1: "road", 2: "tree"},
        np.array([[0, 1, 2, 255]], dtype=np.uint8),
    )
    second = _make_dataset(
        tmp_path / "generated",
        {0: "background", 1: "tree", 2: "road", 3: None},
        np.array([[0, 1, 2, 3]], dtype=np.uint8),
    )
    inventory = merger.build_inventory([first, second])

    assert inventory.class_names == {
        0: "dataset1_background",
        1: "dataset1_road",
        2: "dataset1_tree",
        3: "dataset2_background",
        4: "dataset2_tree",
        5: "dataset2_road",
        6: "",
    }
    assert [0, 3] in inventory.analysis.similar_name_groups
    assert [1, 5] in inventory.analysis.similar_name_groups
    assert [2, 4] in inventory.analysis.similar_name_groups
    assert inventory.analysis.null_like_ids == {6}

    plan = common.EditPlan(
        drop_ids={6},
        merges=[
            common.MergeSpec(ids=[0, 3], name="background"),
            common.MergeSpec(ids=[1, 5], name="road"),
            common.MergeSpec(ids=[2, 4], name="tree"),
        ],
    )
    output = tmp_path / "merged"
    report = merger.merge_datasets(inventory, output, plan)

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["classes"] == {0: "background", 1: "road", 2: "tree"}
    assert config["task"] == "semantic_segmentation"
    assert Path(config["train"]["images"]).is_absolute()
    assert Path(config["train"]["masks"]).is_absolute()
    first_mask = np.asarray(Image.open(output / "masks/train/same.png"))
    second_mask = np.asarray(Image.open(output / "masks/train/same__2.png"))
    assert first_mask.tolist() == [[0, 1, 2, 255]]
    assert second_mask.tolist() == [[0, 2, 1, 255]]
    assert (output / "images/train/same.jpg").is_file()
    assert (output / "images/train/same__2.jpg").is_file()
    assert report["splits"]["train"]["renamed_samples"] == 1
    assert report["sources"]["dataset2"]["old_to_final"] == {
        0: 0,
        1: 2,
        2: 1,
        3: None,
    }


def test_unified_merger_detects_semantic_format(tmp_path: Path) -> None:
    source = _make_dataset(
        tmp_path / "semantic",
        {0: "background", 1: "road"},
        np.array([[0, 1]], dtype=np.uint8),
    )

    assert dataset_merger.detect_format(source) == "semantic"


def test_canonical_duplicate_and_shape_mismatch_are_rejected(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "background"}, np.zeros((2, 3), dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "background"}, np.zeros((2, 3), dtype=np.uint8)
    )
    with pytest.raises(ValueError, match="重复"):
        merger.build_inventory([first, first / "data.yaml"])

    Image.new("RGB", (9, 9)).save(second / "images/train/same.jpg")
    with pytest.raises(ValueError, match="preflight|问题"):
        merger.build_inventory([first, second])


def test_legal_class_255_is_preserved_per_source(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {255: "real"}, np.array([[255]], dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "background"}, np.array([[0]], dtype=np.uint8)
    )
    inventory = merger.build_inventory([first, second], ignore_label=255)
    assert inventory.sources[0].ignore_label is None
    output = tmp_path / "merged"
    merger.merge_datasets(inventory, output, common.EditPlan())
    mask = np.asarray(Image.open(output / "masks/train/same.png"))
    assert mask.tolist() == [[0]]


def test_v2_plan_survives_source_reordering_and_checks_fingerprint(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "road"}, np.array([[0]], dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "street"}, np.array([[0]], dtype=np.uint8)
    )
    original = merger.build_inventory([first, second])
    plan = common.EditPlan(
        merges=[common.MergeSpec(ids=[0, 1], name="road")]
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        yaml.safe_dump(merger.serialize_merge_plan(original, plan), sort_keys=False),
        encoding="utf-8",
    )
    reordered = merger.build_inventory([second, first])
    loaded = merger.load_merge_plan(plan_path, reordered)
    refs = {
        source.dataset.root.name: source.old_to_provisional[0]
        for source in reordered.sources
    }
    assert set(loaded.merges[0].ids) == {refs["first"], refs["second"]}

    config = yaml.safe_load((first / "data.yaml").read_text(encoding="utf-8"))
    config["classes"][0] = "changed"
    (first / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    changed = merger.build_inventory([second, first])
    with pytest.raises(ValueError, match="指纹"):
        merger.load_merge_plan(plan_path, changed)


def test_transaction_preserves_old_output_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "a"}, np.array([[0]], dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "b"}, np.array([[0]], dtype=np.uint8)
    )
    inventory = merger.build_inventory([first, second])
    output = tmp_path / "merged"
    output.mkdir()
    (output / "sentinel").write_text("old", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(common, "_transfer_file", fail)
    with pytest.raises(RuntimeError, match="injected"):
        merger.merge_datasets(inventory, output, common.EditPlan(), clean=True)
    assert (output / "sentinel").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".merged.staging-*"))


def test_v2_plan_locks_untouched_class_ids_after_reordering(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "road"}, np.array([[0]], dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "tree"}, np.array([[0]], dtype=np.uint8)
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
    assert resolved.output_names == {0: "dataset1_road", 1: "dataset2_tree"}


def test_hash_dedupe_rejects_conflicting_remapped_masks(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "background", 1: "object"},
        np.array([[0]], dtype=np.uint8),
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "background", 1: "object"},
        np.array([[1]], dtype=np.uint8),
    )
    inventory = merger.build_inventory(
        [first, second], taxonomy_mode="union-by-name"
    )
    output = tmp_path / "merged"

    with pytest.raises(ValueError, match="mask 不一致"):
        merger.merge_datasets(
            inventory,
            output,
            common.EditPlan(),
            duplicate_policy="hash-dedupe",
        )
    assert not output.exists()


def test_quarantine_counts_exclude_training_split(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "drop", 1: "keep"},
        np.array([[0]], dtype=np.uint8),
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "drop", 1: "keep"},
        np.array([[1]], dtype=np.uint8),
    )
    inventory = merger.build_inventory([first, second])
    first_drop = inventory.sources[0].old_to_provisional[0]
    output = tmp_path / "merged"
    report = merger.merge_datasets(
        inventory,
        output,
        common.EditPlan(drop_ids={first_drop}),
        all_ignore_policy="quarantine",
    )

    assert report["splits"]["train"]["images"] == 1
    assert report["splits"]["train"]["masks"] == 1
    assert report["splits"]["train"]["quarantined_samples"] == 1
    assert (output / "quarantine/train/images/same.jpg").is_file()


def test_declared_source_ignore_is_remapped_to_output_ignore(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "road"}, np.array([[99]], dtype=np.uint8)
    )
    config = yaml.safe_load((first / "data.yaml").read_text(encoding="utf-8"))
    config["ignore_label"] = 99
    (first / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    second = _make_dataset(
        tmp_path / "second", {0: "tree"}, np.array([[0]], dtype=np.uint8)
    )
    inventory = merger.build_inventory([first, second], ignore_label=255)

    assert inventory.sources[0].ignore_label == 99
    output = tmp_path / "merged"
    merger.merge_datasets(inventory, output, common.EditPlan())
    mask = np.asarray(Image.open(output / "masks/train/same.png"))
    assert mask.tolist() == [[255]]


def test_workers_produce_identical_masks_and_counts(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "road"}, np.array([[0]], dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "tree"}, np.array([[0]], dtype=np.uint8)
    )
    inventory = merger.build_inventory([first, second])
    one = tmp_path / "one"
    four = tmp_path / "four"

    report_one = merger.merge_datasets(
        inventory, one, common.EditPlan(), workers=1
    )
    report_four = merger.merge_datasets(
        inventory, four, common.EditPlan(), workers=4
    )

    assert report_one["splits"] == report_four["splits"]
    for relative in (
        Path("masks/train/same.png"),
        Path("masks/train/same__2.png"),
    ):
        assert (one / relative).read_bytes() == (four / relative).read_bytes()


def test_zero_classes_and_dry_run_output_writes(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first", {0: "road"}, np.array([[0]], dtype=np.uint8)
    )
    second = _make_dataset(
        tmp_path / "second", {0: "tree"}, np.array([[0]], dtype=np.uint8)
    )
    inventory = merger.build_inventory([first, second])
    with pytest.raises(ValueError, match="类别为空"):
        merger.merge_datasets(
            inventory,
            tmp_path / "zero",
            common.EditPlan(drop_ids=set(inventory.class_names)),
        )

    dry_output = tmp_path / "absent" / "dry"
    report = merger.merge_datasets(
        inventory, dry_output, common.EditPlan(), dry_run=True
    )
    assert report["dry_run"] is True
    assert not dry_output.parent.exists()


def test_rgb_rgba_workers_are_parallel_and_byte_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rgb_classes = {
        0: {"name": "background", "labels": [[0, 0, 0], [1, 1, 1]]},
        1: {"name": "object", "labels": [[255, 0, 0]]},
    }
    rgba_classes = {
        0: {"name": "background", "values": [[0, 0, 0, 0]]},
        1: {
            "name": "object",
            "values": [[255, 0, 0, 255], [200, 0, 0, 128]],
        },
    }
    rgb_mask = np.array(
        [[[0, 0, 0], [1, 1, 1], [255, 0, 0]]], dtype=np.uint8
    )
    rgba_mask = np.array(
        [[[0, 0, 0, 0], [255, 0, 0, 255], [200, 0, 0, 128]]],
        dtype=np.uint8,
    )
    first = _make_dataset(tmp_path / "rgb", rgb_classes, rgb_mask)
    second = _make_dataset(tmp_path / "rgba", rgba_classes, rgba_mask)
    for index in range(1, 4):
        _add_sample(first, f"rgb_{index}", rgb_mask, 20 + index)
        _add_sample(second, f"rgba_{index}", rgba_mask, 80 + index)
    inventory = merger.build_inventory(
        [first, second], taxonomy_mode="union-by-name"
    )
    one = tmp_path / "one"
    four = tmp_path / "four"
    report_one = merger.merge_datasets(
        inventory, one, common.EditPlan(), workers=1
    )

    original_prepare = merger._prepare_mask_task
    barrier = threading.Barrier(4)
    thread_names: set[str] = set()
    lock = threading.Lock()

    def synchronized_prepare(task: merger._MaskTask) -> merger._PreparedMask:
        with lock:
            thread_names.add(threading.current_thread().name)
        barrier.wait(timeout=10)
        return original_prepare(task)

    monkeypatch.setattr(merger, "_prepare_mask_task", synchronized_prepare)
    report_four = merger.merge_datasets(
        inventory, four, common.EditPlan(), workers=4
    )

    assert len(thread_names) == 4
    assert report_one["splits"] == report_four["splits"]
    assert report_one["warnings"] == report_four["warnings"]
    assert _mask_hashes(one) == _mask_hashes(four)
    for path in (four / "masks").rglob("*.png"):
        assert np.asarray(Image.open(path)).ndim == 2


def test_palette_identity_fast_path_preserves_palette_png(tmp_path: Path) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: "background", 1: "road"},
        np.array([[0, 1]], dtype=np.uint8),
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "background", 1: "road"},
        np.array([[1, 0]], dtype=np.uint8),
    )
    palette_values = [0, 0, 0, 220, 30, 10] + [0] * (256 * 3 - 6)
    for source, values in (
        (first, np.array([[0, 1]], dtype=np.uint8)),
        (second, np.array([[1, 0]], dtype=np.uint8)),
    ):
        image = Image.fromarray(values, mode="P")
        image.putpalette(palette_values)
        image.save(source / "masks/train/same.png")
    inventory = merger.build_inventory(
        [first, second], taxonomy_mode="union-by-name"
    )
    output = tmp_path / "merged"

    report = merger.merge_datasets(inventory, output, common.EditPlan(), workers=4)

    assert report["splits"]["train"]["identity_fast_path_masks"] == 2
    for output_name, source in zip(
        ("same.png", "same__2.png"), (first, second), strict=True
    ):
        target = output / "masks" / "train" / output_name
        assert target.read_bytes() == (source / "masks/train/same.png").read_bytes()
        with Image.open(target) as image:
            assert image.mode == "P"
            assert image.getpalette()[:6] == palette_values[:6]


def test_unknown_rgb_label_obeys_plan_and_worker_failure_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _make_dataset(
        tmp_path / "first",
        {0: {"name": "background", "labels": [[0, 0, 0]]}},
        np.array([[[9, 8, 7]]], dtype=np.uint8),
    )
    second = _make_dataset(
        tmp_path / "second",
        {0: "road"},
        np.array([[0]], dtype=np.uint8),
    )
    inventory = merger.build_inventory([first, second])
    assert inventory.analysis.unknown_ids == {(9, 8, 7)}

    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="未映射"):
        merger.merge_datasets(
            inventory, rejected, common.EditPlan(), workers=4
        )
    assert not rejected.exists()

    output = tmp_path / "merged"
    merger.merge_datasets(
        inventory,
        output,
        common.EditPlan(unknown_policy="ignore"),
        workers=4,
    )
    assert np.asarray(
        Image.open(output / "masks/train/same.png")
    ).tolist() == [[255]]

    sentinel_output = tmp_path / "sentinel-output"
    sentinel_output.mkdir()
    (sentinel_output / "sentinel").write_text("old", encoding="utf-8")

    def fail_in_worker(_task: merger._MaskTask) -> merger._PreparedMask:
        raise RuntimeError("worker injected")

    monkeypatch.setattr(merger, "_prepare_mask_task", fail_in_worker)
    with pytest.raises(RuntimeError, match="worker injected"):
        merger.merge_datasets(
            inventory,
            sentinel_output,
            common.EditPlan(unknown_policy="ignore"),
            workers=4,
            clean=True,
        )
    assert (sentinel_output / "sentinel").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".sentinel-output.staging-*"))


def test_semantic_merger_parser_accepts_reflink() -> None:
    assert merger.parse_args(["--image-mode", "reflink"]).image_mode == "reflink"
