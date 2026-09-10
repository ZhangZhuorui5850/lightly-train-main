from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import rebalance_yolo_splits as rebalance  # noqa: E402


def _make_dataset(root: Path, rows: list[tuple[str, str, tuple[int, ...]]]) -> Path:
    names = {0: "common", 1: "medium", 2: "rare", 3: "very_rare"}
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": "detect",
        "nc": len(names),
        "names": names,
    }
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    for index, (split, stem, class_ids) in enumerate(rows):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        Image.new(
            "RGB",
            (10, 8),
            ((index * 31) % 255, (index * 67) % 255, (index * 101) % 255),
        ).save(root / "images" / split / f"{stem}.png")
        label_rows = [f"{class_id} 0.5 0.5 0.2 0.2" for class_id in class_ids]
        (root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8"
        )
    return root / "data.yaml"


def test_rebalance_has_exact_sizes_and_covers_eligible_classes(tmp_path: Path) -> None:
    rows: list[tuple[str, str, tuple[int, ...]]] = []
    source_splits = ("train", "val", "test")
    for index in range(60):
        classes = [0]
        if index < 15:
            classes.append(1)
        if index < 3:
            classes.append(2)
        if index < 2:
            classes.append(3)
        rows.append((source_splits[index % 3], f"img_{index:03d}", tuple(classes)))
    source = _make_dataset(tmp_path / "source", rows)

    plan = rebalance.build_plan(source, ratios=(0.8, 0.1, 0.1), seed=17)

    assert Counter(plan.assignments.values()) == {"train": 48, "val": 6, "test": 6}
    for class_id in (0, 1, 2):
        assert all(plan.class_image_counts[class_id][split] >= 1 for split in rebalance.SPLITS)
    missing_very_rare = [
        row for row in plan.missing_coverage if row["class_id"] == 3
    ]
    assert len(missing_very_rare) == 1
    assert missing_very_rare[0]["coverage_possible_by_count"] is False


def test_write_plan_preserves_pairs_and_handles_same_stem(tmp_path: Path) -> None:
    source = _make_dataset(
        tmp_path / "source",
        [
            ("train", "same", (0,)),
            ("val", "same", (1,)),
            ("test", "third", (2,)),
            ("train", "fourth", (0, 1)),
            ("val", "fifth", (1, 2)),
            ("test", "sixth", (0, 2)),
        ],
    )
    plan = rebalance.build_plan(source, ratios=(1, 1, 1), seed=3)
    output = rebalance.write_plan(
        plan, tmp_path / "balanced", image_mode="copy"
    )

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["train"] == "images/train"
    assert config["val"] == "images/val"
    assert config["test"] == "images/test"
    image_count = sum(
        len(list((output / "images" / split).glob("*")))
        for split in rebalance.SPLITS
    )
    label_count = sum(
        len(list((output / "labels" / split).glob("*.txt")))
        for split in rebalance.SPLITS
    )
    assert image_count == label_count == 6
    assert (output / "split_report.json").is_file()
    assert (output / "split_mapping.json").is_file()


def test_content_deduplication_is_opt_in(tmp_path: Path) -> None:
    source = _make_dataset(
        tmp_path / "source",
        [("train", "one", (0,)), ("val", "two", (0,)), ("test", "three", (1,))],
    )
    (tmp_path / "source/images/val/two.png").write_bytes(
        (tmp_path / "source/images/train/one.png").read_bytes()
    )

    preserved = rebalance.build_plan(source, ratios=(0.8, 0.1, 0.1), seed=5)
    deduplicated = rebalance.build_plan(
        source, ratios=(0.8, 0.1, 0.1), seed=5, deduplicate=True
    )

    assert len(preserved.samples) == 3
    assert preserved.duplicate_rows == []
    assert len(deduplicated.samples) == 2
    assert len(deduplicated.duplicate_rows) == 1


def test_duplicate_yolo_images_with_different_labels_are_merged(
    tmp_path: Path,
) -> None:
    source = _make_dataset(
        tmp_path / "source",
        [("train", "one", (0,)), ("val", "two", (1,)), ("test", "three", (2,))],
    )
    (tmp_path / "source/images/val/two.png").write_bytes(
        (tmp_path / "source/images/train/one.png").read_bytes()
    )

    plan = rebalance.build_plan(
        source, ratios=(1, 1, 1), seed=5, deduplicate=True
    )

    merged = next(sample for sample in plan.samples if sample.image_path.stem == "one")
    assert merged.class_ids == {0, 1}
    assert len(merged.label_lines) == 2
    assert plan.duplicate_rows[0]["status"] == "merged-yolo-annotations"

    output = rebalance.write_plan(plan, tmp_path / "balanced", image_mode="copy")
    output_rows = [
        line
        for path in (output / "labels").glob("*/*.txt")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(line.startswith("0 ") for line in output_rows)
    assert any(line.startswith("1 ") for line in output_rows)


def _make_semantic_dataset(root: Path) -> Path:
    classes = {
        0: {"name": "background", "labels": [0]},
        1: {"name": "defect", "labels": [10]},
        2: {"name": "rare", "labels": [20]},
    }
    config = {
        "path": str(root),
        "task": "semantic_segmentation",
        "classes": classes,
        "train": {"images": "images/train", "masks": "masks/train"},
        "val": {"images": "images/val", "masks": "masks/val"},
        "test": {"images": "images/test", "masks": "masks/test"},
    }
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    source_splits = ("train", "val", "test")
    for index in range(9):
        split = source_splits[index % 3]
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "masks" / split).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), (index * 20, 30, 60)).save(
            root / "images" / split / f"semantic_{index}.png"
        )
        mask = Image.new("L", (8, 6), 0)
        pixels = mask.load()
        for x in range(3):
            for y in range(2):
                pixels[x, y] = 10
        if index < 3:
            pixels[7, 5] = 20
        mask.save(root / "masks" / split / f"semantic_{index}.png")
    return root / "data.yaml"


def test_semantic_masks_are_stratified_and_written_as_pairs(tmp_path: Path) -> None:
    source = _make_semantic_dataset(tmp_path / "semantic_source")

    plan = rebalance.build_plan(source, ratios=(1, 1, 1), seed=23)

    assert plan.format_kind == "semantic_mask"
    assert plan.amount_unit == "pixels"
    assert Counter(plan.assignments.values()) == {"train": 3, "val": 3, "test": 3}
    assert all(plan.class_image_counts[2][split] == 1 for split in rebalance.SPLITS)

    output = rebalance.write_plan(
        plan, tmp_path / "semantic_balanced", image_mode="copy"
    )
    output_config = yaml.safe_load(
        (output / "data.yaml").read_text(encoding="utf-8")
    )
    assert output_config["task"] == "semantic_segmentation"
    assert output_config["train"]["masks"] == "masks/train"
    for split in rebalance.SPLITS:
        image_stems = {
            path.stem for path in (output / "images" / split).glob("*.png")
        }
        mask_stems = {
            path.stem for path in (output / "masks" / split).glob("*.png")
        }
        assert image_stems == mask_stems
    report = yaml.safe_load((output / "split_report.json").read_text(encoding="utf-8"))
    assert report["amount_unit"] == "pixels"


def test_rgb_semantic_mask_labels_are_supported(tmp_path: Path) -> None:
    root = tmp_path / "rgb_semantic"
    config = {
        "path": str(root),
        "task": "semantic_segmentation",
        "classes": {
            0: {"name": "background", "labels": [[0, 0, 0]]},
            1: {"name": "defect", "labels": [[255, 0, 0]]},
        },
        "train": {"images": "images/train", "masks": "masks/train"},
        "val": {"images": "images/val", "masks": "masks/val"},
        "test": {"images": "images/test", "masks": "masks/test"},
    }
    root.mkdir(parents=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    for index, split in enumerate(rebalance.SPLITS):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        Image.new("RGB", (6, 4), (10 + index, 20, 30)).save(
            root / "images" / split / f"rgb_{index}.png"
        )
        mask = Image.new("RGB", (6, 4), (0, 0, 0))
        mask.putpixel((index, 0), (255, 0, 0))
        mask.save(root / "masks" / split / f"rgb_{index}.png")

    plan = rebalance.build_plan(root / "data.yaml", ratios=(1, 1, 1), seed=7)

    assert plan.format_kind == "semantic_mask"
    assert all(plan.class_image_counts[1][split] == 1 for split in rebalance.SPLITS)
    assert sum(plan.class_amount_counts[1].values()) == 3


def test_cli_rejects_symlink_output_and_dry_run_is_read_only(tmp_path):
    import pytest

    source = _make_dataset(tmp_path / "source", [("train", "a", (0,)), ("val", "b", (0,)), ("test", "c", (0,))])
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)
    args = ["--src", str(source), "--out", str(output), "--clean", "--yes"]
    with pytest.raises(ValueError, match="符号链接"):
        rebalance.main([*args, "--dry-run"])
    with pytest.raises(ValueError, match="符号链接"):
        rebalance.main(args)
    assert marker.read_text() == "keep"
    safe_output = tmp_path / "safe_output"
    assert rebalance.main(["--src", str(source), "--out", str(safe_output), "--dry-run"]) == 0
    assert not safe_output.exists()
