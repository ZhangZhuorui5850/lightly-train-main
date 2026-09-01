from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import semantic_class_editor as editor  # noqa: E402


def _save_image(path: Path, size: tuple[int, int] = (4, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (80, 100, 120)).save(path)


def _save_mask(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8)).save(path)


def _make_custom_dataset(
    root: Path,
    classes: dict[int, object],
    mask: np.ndarray,
    *,
    ignore_label: int | None = None,
) -> Path:
    root.mkdir(parents=True)
    config: dict[str, object] = {
        "task": "semantic_segmentation",
        "classes": classes,
        "train": {"images": "images/train", "masks": "masks/train"},
    }
    if ignore_label is not None:
        config["ignore_label"] = ignore_label
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _save_image(root / "images/train/a.jpg", (mask.shape[1], mask.shape[0]))
    mask_path = root / "masks/train/a.png"
    mask_path.parent.mkdir(parents=True)
    Image.fromarray(mask).save(mask_path)
    return root


def _make_semantic_dataset(root: Path) -> Path:
    root.mkdir(parents=True)
    config = {
        "path": ".",
        "task": "semantic_segmentation",
        "train": {"images": "images/train", "masks": "masks/train"},
        "val": {"images": "images/val", "masks": "masks/val"},
        "names": {
            0: "background",
            1: "dataset1_road",
            2: None,
            4: "road",
            7: "unused",
        },
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _save_image(root / "images/train/a.jpg")
    _save_image(root / "images/val/b.jpg")
    _save_mask(
        root / "masks/train/a.png",
        np.array(
            [
                [0, 1, 2, 4],
                [0, 9, 255, 4],
                [1, 1, 2, 0],
            ],
            dtype=np.uint8,
        ),
    )
    _save_mask(
        root / "masks/val/b.png",
        np.array(
            [
                [0, 4, 4, 4],
                [0, 0, 1, 1],
                [255, 255, 2, 2],
            ],
            dtype=np.uint8,
        ),
    )
    return root


def test_analysis_finds_null_unknown_unused_and_similar_names(tmp_path: Path) -> None:
    source = _make_semantic_dataset(tmp_path / "source")
    dataset = editor.resolve_dataset(source)

    analysis = editor.analyze_dataset(dataset)

    assert analysis.null_like_ids == {2}
    assert analysis.unknown_ids == {9}
    assert analysis.unknown_stats[9].image_count == 1
    assert analysis.unknown_stats[9].pixel_count == 1
    assert analysis.unused_ids == {7}
    assert [1, 4] in analysis.similar_name_groups
    assert analysis.class_stats[2].image_count == 2
    assert analysis.class_stats[2].pixel_count == 4
    assert {issue.issue_type for issue in analysis.issues} >= {
        "null_like_names",
        "unknown_mask_ids",
        "unused_yaml_ids",
        "non_contiguous_ids",
        "similar_names",
    }


def test_edit_dataset_drops_merges_and_reindexes_masks(tmp_path: Path) -> None:
    source = _make_semantic_dataset(tmp_path / "source")
    output = tmp_path / "edited"
    dataset = editor.resolve_dataset(source)
    plan = editor.EditPlan(
        drop_ids={2},
        merges=[editor.MergeSpec(ids=[1, 4], name="road")],
        unknown_policy="ignore",
    )

    report = editor.edit_dataset(dataset, output, plan)

    config = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert config["classes"] == {0: "background", 1: "road", 2: "unused"}
    assert "names" not in config
    assert "nc" not in config
    assert "path" not in config
    assert report["source_to_target"] == {0: 0, 1: 1, 2: 255, 4: 1, 7: 2}
    train_mask = np.asarray(Image.open(output / "masks/train/a.png"))
    assert train_mask.tolist() == [
        [0, 1, 255, 1],
        [0, 255, 255, 1],
        [1, 1, 255, 0],
    ]
    assert (output / "images/train/a.jpg").is_file()
    assert (output / "class_edit_mapping.yaml").is_file()
    assert (output / "class_analysis.json").is_file()


def test_edit_dataset_rejects_unknown_mask_ids_by_default(tmp_path: Path) -> None:
    dataset = editor.resolve_dataset(_make_semantic_dataset(tmp_path / "source"))

    with pytest.raises(ValueError, match="未定义"):
        editor.edit_dataset(
            dataset,
            tmp_path / "edited",
            editor.EditPlan(drop_ids={2}),
        )


def test_edit_dataset_rejects_output_nested_in_source(tmp_path: Path) -> None:
    dataset = editor.resolve_dataset(_make_semantic_dataset(tmp_path / "source"))

    with pytest.raises(ValueError, match="分离"):
        editor.edit_dataset(
            dataset,
            dataset.root / "edited",
            editor.EditPlan(drop_ids={2}, unknown_policy="ignore"),
        )


def test_invalid_mask_is_rejected_before_output_is_created(tmp_path: Path) -> None:
    source = _make_semantic_dataset(tmp_path / "source")
    Image.new("RGB", (4, 3), (1, 2, 3)).save(source / "masks/train/a.png")
    dataset = editor.resolve_dataset(source)
    output = tmp_path / "edited"

    with pytest.raises(ValueError, match="无效 mask"):
        editor.edit_dataset(
            dataset,
            output,
            editor.EditPlan(drop_ids={2}, unknown_policy="ignore"),
        )

    assert not output.exists()


def test_plan_rejects_class_in_drop_and_merge() -> None:
    plan = editor.EditPlan(
        drop_ids={2},
        merges=[editor.MergeSpec(ids=[1, 2], name="road")],
    )

    with pytest.raises(ValueError, match="删除和合并"):
        editor.resolve_plan(plan, {0: "background", 1: "road", 2: "street"})


def test_numbered_placeholder_names_are_not_synonym_candidates() -> None:
    exact, similar = editor._name_candidate_groups(
        {
            1: "class_1",
            10: "class_10",
            11: "class_11",
            21: "dataset2_class_1",
            22: "dataset2_class_10",
        }
    )

    assert exact == []
    assert similar == []


def test_output_ignore_value_declared_as_class_is_preserved(tmp_path: Path) -> None:
    source = _make_semantic_dataset(tmp_path / "source")
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    config["names"][255] = "reserved"
    (source / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dataset = editor.resolve_dataset(source)
    analysis = editor.analyze_dataset(dataset)

    assert analysis.class_stats[255].pixel_count == 3
    output = tmp_path / "edited"
    editor.edit_dataset(
        dataset,
        output,
        editor.EditPlan(
            drop_ids={2, 7},
            unknown_policy="ignore",
        ),
        analysis=analysis,
    )
    output_mask = np.asarray(Image.open(output / "masks/train/a.png"))
    assert output_mask[1, 2] == 3


def test_interactive_plan_handles_detected_issues_and_keeps_valid_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = editor.resolve_dataset(_make_semantic_dataset(tmp_path / "source"))
    analysis = editor.analyze_dataset(dataset)
    answers = iter(
        [
            "",                  # 删除 null 类别 2
            "",                  # 删除未使用类别 7
            "y",                 # 合并近似名称 1、4
            "road",
            "y",                 # 未知 ID 9 映射 ignore
            "m 1,2 = invalid",   # 冲突操作，计划保持原状态
            "done",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    plan = editor.interactive_plan(dataset, analysis)

    assert plan is not None
    assert plan.drop_ids == {2, 7}
    assert [merge.ids for merge in plan.merges] == [[1, 4]]
    assert plan.merges[0].name == "road"
    assert plan.unknown_policy == "ignore"


def test_official_uavscenes_blank_ids_can_map_to_ignore() -> None:
    names = {
        0: "background",
        1: "roof",
        7: "",
        8: "",
        12: "",
        21: "",
        22: "",
        23: "",
        24: "truck",
        25: "",
    }
    plan = editor.EditPlan(drop_ids={7, 8, 12, 21, 22, 23, 25})

    resolved = editor.resolve_plan(plan, names)
    mask = np.array([[0, 1, 7, 24, 25]], dtype=np.uint8)
    output = editor.remap_mask(
        mask,
        resolved,
        ignore_label=255,
        unknown_policy="error",
    )

    assert resolved.output_names == {0: "background", 1: "roof", 2: "truck"}
    assert output.tolist() == [[0, 1, 255, 2, 255]]


def test_multi_integer_labels_and_values_alias_decode_uint16(tmp_path: Path) -> None:
    source = _make_custom_dataset(
        tmp_path / "source",
        {
            10: {"name": "road", "labels": [1000, 1001]},
            20: {"name": "tree", "values": [2000]},
        },
        np.array([[1000, 1001, 2000, 3000]], dtype=np.uint16),
    )
    dataset = editor.resolve_dataset(source)

    assert dataset.class_labels == {10: (1000, 1001), 20: (2000,)}
    assert dataset.label_to_class == {1000: 10, 1001: 10, 2000: 20}
    assert dataset.label_kind == "integer"
    analysis = editor.analyze_dataset(dataset, ignore_label=65535)
    assert analysis.class_stats[10].image_count == 1
    assert analysis.class_stats[10].pixel_count == 2
    assert analysis.class_stats[20].pixel_count == 1
    assert analysis.unknown_ids == {3000}

    output = tmp_path / "edited"
    editor.edit_dataset(
        dataset,
        output,
        editor.EditPlan(unknown_policy="ignore"),
        analysis=analysis,
        ignore_label=65535,
    )
    converted = np.asarray(Image.open(output / "masks/train/a.png"))
    assert converted.dtype == np.uint16
    assert converted.tolist() == [[0, 0, 1, 65535]]


def test_rgb_labels_decode_to_single_channel_and_unknown_policy(tmp_path: Path) -> None:
    source = _make_custom_dataset(
        tmp_path / "source",
        {
            3: {"name": "background", "labels": [[0, 0, 0], [1, 1, 1]]},
            8: {"name": "road", "values": [[255, 0, 0]]},
        },
        np.array(
            [[[0, 0, 0], [1, 1, 1]], [[255, 0, 0], [7, 8, 9]]],
            dtype=np.uint8,
        ),
    )
    dataset = editor.resolve_dataset(source)
    analysis = editor.analyze_dataset(dataset)

    assert dataset.label_kind == "rgb"
    assert dataset.label_channels == 3
    assert analysis.class_stats[3].pixel_count == 2
    assert analysis.class_stats[8].pixel_count == 1
    assert analysis.unknown_ids == {(7, 8, 9)}
    with pytest.raises(ValueError, match="未定义/未映射"):
        editor.edit_dataset(dataset, tmp_path / "rejected", editor.EditPlan())

    output = tmp_path / "edited"
    editor.edit_dataset(
        dataset,
        output,
        editor.EditPlan(unknown_policy="ignore"),
        analysis=analysis,
    )
    with Image.open(output / "masks/train/a.png") as image:
        assert image.mode == "L"
        assert np.asarray(image).tolist() == [[0, 0], [1, 255]]


def test_rgba_labels_decode_to_single_channel(tmp_path: Path) -> None:
    source = _make_custom_dataset(
        tmp_path / "source",
        {
            0: {"name": "transparent", "labels": [[0, 0, 0, 0]]},
            1: {
                "name": "object",
                "labels": [[10, 20, 30, 255], [40, 50, 60, 128]],
            },
        },
        np.array(
            [[[0, 0, 0, 0], [10, 20, 30, 255], [40, 50, 60, 128]]],
            dtype=np.uint8,
        ),
    )
    dataset = editor.resolve_dataset(source)
    output = tmp_path / "edited"

    assert dataset.label_kind == "rgba"
    editor.edit_dataset(dataset, output, editor.EditPlan())
    with Image.open(output / "masks/train/a.png") as image:
        assert image.mode == "L"
        assert np.asarray(image).tolist() == [[0, 1, 1]]


def test_palette_mask_keeps_index_labels(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    config = {
        "task": "semantic_segmentation",
        "classes": {0: "background", 1: "road"},
        "train": {"images": "images/train", "masks": "masks/train"},
    }
    (root / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    _save_image(root / "images/train/a.jpg", (2, 1))
    mask_path = root / "masks/train/a.png"
    mask_path.parent.mkdir(parents=True)
    palette = Image.fromarray(np.array([[0, 1]], dtype=np.uint8), mode="P")
    colors = [0, 0, 0, 200, 10, 10] + [0] * (256 * 3 - 6)
    palette.putpalette(colors)
    palette.save(mask_path)

    dataset = editor.resolve_dataset(root)
    raw = editor.read_mask(mask_path)
    analysis = editor.analyze_dataset(dataset)

    assert raw.ndim == 2
    assert raw.tolist() == [[0, 1]]
    assert analysis.class_stats[1].pixel_count == 1


@pytest.mark.parametrize(
    ("classes", "message"),
    [
        (
            {
                0: {"name": "a", "labels": [0]},
                1: {"name": "b", "labels": [[1, 2, 3]]},
            },
            "混合.*整数与颜色",
        ),
        (
            {
                0: {"name": "a", "labels": [[0, 0, 0]]},
                1: {"name": "b", "labels": [[1, 1, 1, 1]]},
            },
            "RGB 与 RGBA",
        ),
        (
            {
                0: {"name": "a", "labels": [7]},
                1: {"name": "b", "values": [7]},
            },
            "同时映射",
        ),
        (
            {0: {"name": "a", "labels": [0], "values": [0]}},
            "同时配置 labels 和 values",
        ),
    ],
)
def test_semantic_label_schema_rejects_ambiguous_mappings(
    tmp_path: Path, classes: dict[int, object], message: str
) -> None:
    source = _make_custom_dataset(
        tmp_path / "source", classes, np.zeros((1, 1), dtype=np.uint8)
    )

    with pytest.raises(ValueError, match=message):
        editor.resolve_dataset(source)


def test_reflink_transfer_copies_content_without_hardlinking(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    target = tmp_path / "nested/target.bin"
    source.write_bytes(b"semantic-reflink")

    editor._transfer_file(source, target, "reflink")

    assert target.read_bytes() == source.read_bytes()
    assert target.stat().st_ino != source.stat().st_ino
    assert editor.parse_args(["--image-mode", "reflink"]).image_mode == "reflink"
