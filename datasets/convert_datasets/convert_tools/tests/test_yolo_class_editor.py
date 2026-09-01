from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import class_editor  # noqa: E402
import semantic_class_editor as common  # noqa: E402
import yolo_class_editor as editor  # noqa: E402


def _save_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (50, 80, 100)).save(path)


def _make_yolo_dataset(root: Path, *, task: str = "detect") -> Path:
    root.mkdir(parents=True)
    config = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "task": task,
        "nc": 5,
        "names": {
            0: "dataset1_car",
            2: "car",
            4: "tree",
            6: None,
            8: "unused",
        },
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _save_image(root / "images/train/a.jpg")
    _save_image(root / "images/val/b.jpg")
    return root


def test_detection_analysis_and_edit_rewrites_class_column(tmp_path: Path) -> None:
    source = _make_yolo_dataset(tmp_path / "det")
    labels = source / "labels"
    (labels / "train").mkdir(parents=True)
    (labels / "val").mkdir(parents=True)
    (labels / "train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n"
        "2 0.4 0.4 0.1 0.1\n"
        "4 0.2 0.2 0.1 0.1\n"
        "9 0.1 0.1 0.1 0.1\n",
        encoding="utf-8",
    )
    (labels / "val/b.txt").write_text("2 0.6 0.6 0.2 0.2\n", encoding="utf-8")

    dataset = editor.resolve_dataset(source)
    analysis = editor.analyze_dataset(dataset)

    assert dataset.kind == "yolo_detection"
    assert analysis.shared.unknown_ids == {9}
    assert analysis.shared.null_like_ids == {6}
    assert analysis.shared.unused_ids == {6, 8}
    assert [0, 2] in analysis.shared.similar_name_groups
    assert analysis.annotation_count == 5

    output = tmp_path / "edited"
    plan = common.EditPlan(
        drop_ids={4, 6, 8},
        merges=[common.MergeSpec(ids=[0, 2], name="vehicle")],
        unknown_policy="ignore",
    )
    report = editor.edit_dataset(dataset, output, plan, analysis=analysis)

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "detect"
    assert config["nc"] == 1
    assert config["names"] == {0: "vehicle"}
    assert (output / "labels/train/a.txt").read_text(encoding="utf-8") == (
        "0 0.5 0.5 0.2 0.3\n"
        "0 0.4 0.4 0.1 0.1\n"
    )
    assert (output / "labels/val/b.txt").read_text(encoding="utf-8") == (
        "0 0.6 0.6 0.2 0.2\n"
    )
    assert report["kept_annotations"] == 3
    assert report["removed_annotations"] == 2
    assert (output / "images/train/a.jpg").is_file()


def test_instance_segmentation_preserves_polygon_coordinates(tmp_path: Path) -> None:
    source = _make_yolo_dataset(tmp_path / "seg", task="segment")
    label = source / "labels/train/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text(
        "2 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n",
        encoding="utf-8",
    )

    dataset = editor.resolve_dataset(source)
    assert dataset.kind == "yolo_instance"
    plan = common.EditPlan(
        drop_ids={0, 6, 8},
        renames={2: "vehicle"},
    )
    output = tmp_path / "edited"
    editor.edit_dataset(dataset, output, plan)

    assert (output / "labels/train/a.txt").read_text(encoding="utf-8") == (
        "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n"
    )
    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["task"] == "segment"
    assert config["names"] == {0: "vehicle", 1: "tree"}


def test_invalid_yolo_row_blocks_output(tmp_path: Path) -> None:
    source = _make_yolo_dataset(tmp_path / "det")
    label = source / "labels/train/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.1 0.2\n", encoding="utf-8")
    dataset = editor.resolve_dataset(source)
    analysis = editor.analyze_dataset(dataset)
    output = tmp_path / "edited"

    assert analysis.malformed_count == 1
    with pytest.raises(ValueError, match="无效 YOLO 标注"):
        editor.edit_dataset(
            dataset,
            output,
            common.EditPlan(drop_ids={6, 8}),
            analysis=analysis,
        )
    assert not output.exists()


def test_unified_editor_detects_supported_formats(tmp_path: Path) -> None:
    detection = _make_yolo_dataset(tmp_path / "det")
    (detection / "labels/train").mkdir(parents=True)
    (detection / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    semantic = tmp_path / "semantic"
    semantic.mkdir()
    (semantic / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "task": "semantic_segmentation",
                "classes": {0: "background"},
                "train": {
                    "images": "images/train",
                    "masks": "masks/train",
                },
            }
        ),
        encoding="utf-8",
    )
    (semantic / "images/train").mkdir(parents=True)
    (semantic / "masks/train").mkdir(parents=True)

    assert class_editor.detect_kind(detection) == "yolo_detection"
    assert class_editor.detect_kind(semantic) == "semantic_mask"
    assert class_editor.detect_kind(detection, "yolo-seg") == "yolo_instance"


def test_txt_manifest_edit_copies_only_listed_images(tmp_path: Path) -> None:
    source = tmp_path / "manifest"
    source.mkdir()
    (source / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(source),
                "train": "train.txt",
                "task": "detect",
                "names": {0: "item"},
            }
        ),
        encoding="utf-8",
    )
    _save_image(source / "images/train/listed.jpg")
    _save_image(source / "images/train/unlisted.jpg")
    (source / "train.txt").write_text(
        "images/train/listed.jpg\n", encoding="utf-8"
    )
    label = source / "labels/train/listed.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    dataset = editor.resolve_dataset(source)
    output = tmp_path / "edited"
    editor.edit_dataset(dataset, output, common.EditPlan())

    assert (output / "images/train/listed.jpg").is_file()
    assert not (output / "images/train/unlisted.jpg").exists()
    assert (output / "labels/train/listed.txt").is_file()


def test_class_edit_transaction_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _make_yolo_dataset(tmp_path / "source")
    label = source / "labels/train/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset = editor.resolve_dataset(source)
    output = tmp_path / "edited"
    output.mkdir()
    (output / "sentinel").write_text("old", encoding="utf-8")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected")

    monkeypatch.setattr(common, "_transfer_file", fail)
    with pytest.raises(RuntimeError, match="injected"):
        editor.edit_dataset(
            dataset,
            output,
            common.EditPlan(drop_ids={6}),
            clean=True,
        )

    assert (output / "sentinel").read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".edited.staging-*"))


def test_dry_run_creates_no_output_parent(tmp_path: Path) -> None:
    source = _make_yolo_dataset(tmp_path / "source")
    label = source / "labels/train/a.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    dataset = editor.resolve_dataset(source)
    output = tmp_path / "absent" / "edited"

    report = editor.edit_dataset(
        dataset,
        output,
        common.EditPlan(drop_ids={6}),
        dry_run=True,
    )

    assert report["dry_run"] is True
    assert not output.parent.exists()


def test_unified_editor_forwards_yes_and_train_val_flags() -> None:
    args = SimpleNamespace(
        out=None,
        plan=None,
        image_mode="copy",
        clean=False,
        dry_run=False,
        yes=True,
        require_train_val=True,
        id_policy="preserve",
    )
    child_args: list[str] = []

    class_editor._append_common_args(args, child_args)

    assert "--yes" in child_args
    assert "--require-train-val" in child_args
    id_policy_index = child_args.index("--id-policy")
    assert child_args[id_policy_index : id_policy_index + 2] == [
        "--id-policy",
        "preserve",
    ]
