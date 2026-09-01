from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tool_lib import common as rt
from tool_lib import det_infer
from tool_lib.det_infer import (
    build_parallel_child_command,
    build_split_report_path,
    build_split_output_dir,
    export_bad_class_images,
    resolve_dataset_infer_splits,
    rebase_output_artifact_path,
    select_bad_classes_from_report,
    visualization_indices,
)
from tool_lib.artifact_transaction import create_stage, publish_stage
from tool_lib.interactive import parse_cli_args


def _create_image(path: Path, color: tuple[int, int, int] = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)


def test_resolve_dataset_infer_splits_supports_train_and_all() -> None:
    data_cfg = {"train": "images/train", "val": "images/val", "test": "images/test"}

    assert resolve_dataset_infer_splits(data_cfg, "train") == ["train"]
    assert resolve_dataset_infer_splits(data_cfg, "all") == ["train", "test", "val"]


def test_build_split_output_dir_includes_dataset_identity_and_split(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    args = SimpleNamespace(output_dir=None, split="test")
    data_path = tmp_path / "source" / "dataset_det" / "data.yaml"
    data_cfg = {"_root_dir": data_path.parent, "_data_yaml_path": data_path}

    output_dir = build_split_output_dir(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        split="test",
    )

    dataset_tag = rt.path_identity_tag(data_path)
    assert output_dir == experiment_dir / "infer" / dataset_tag / "test"


def test_build_split_output_dir_distinguishes_same_named_datasets(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    args = SimpleNamespace(output_dir=None, split="test")

    outputs = []
    for parent in ("source_a", "source_b"):
        data_path = tmp_path / parent / "dataset_det" / "data.yaml"
        outputs.append(
            build_split_output_dir(
                args=args,
                checkpoint_path=checkpoint_path,
                data_cfg={"_root_dir": data_path.parent, "_data_yaml_path": data_path},
                split="test",
            )
        )

    assert outputs[0] != outputs[1]
    assert outputs[0].name == outputs[1].name == "test"


def test_build_split_output_dir_uses_eval_action(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    data_path = tmp_path / "source" / "dataset_det" / "data.yaml"
    args = SimpleNamespace(output_dir=None, split="test", tool_action="eval")

    output_dir = build_split_output_dir(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg={"_root_dir": data_path.parent, "_data_yaml_path": data_path},
        split="test",
    )

    assert output_dir == experiment_dir / "eval" / rt.path_identity_tag(data_path) / "test"


def test_eval_report_is_written_inside_stage_before_publish(tmp_path: Path) -> None:
    published_output = tmp_path / "eval" / "dataset" / "test"
    published_report = published_output / "0829-test_report.json"
    stage_output = create_stage(published_output, overwrite=False)

    stage_report = rebase_output_artifact_path(
        published_report,
        source_output_dir=published_output,
        target_output_dir=stage_output,
    )
    stage_report.write_text("{}\n", encoding="utf-8")

    assert stage_report.parent == stage_output
    assert published_output.exists() is False

    publish_stage(stage_output, published_output, overwrite=False)

    assert published_report.read_text(encoding="utf-8") == "{}\n"


def test_confusion_index_records_published_paths_while_writing_stage(tmp_path: Path) -> None:
    published_output = (tmp_path / "eval" / "dataset" / "test").resolve()
    stage_output = create_stage(published_output, overwrite=False)
    stage_report = stage_output / "0829-test_report.json"
    stage_report.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        _published_output_dir=published_output,
        data=tmp_path / "data.yaml",
        save_json=False,
        score_threshold=0.3,
        report_iou_threshold=0.5,
    )

    index_path = det_infer.write_confusion_inputs_index(
        output_dir=stage_output,
        report_path=stage_report,
        args=args,
        data_cfg={"_root_dir": tmp_path / "dataset"},
        split="test",
        num_images=2,
    )
    payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert payload["paths"]["output_dir"] == str(published_output)
    assert payload["paths"]["report_path"] == str(
        published_output / "0829-test_report.json"
    )
    assert ".staging-" not in json.dumps(payload)


def test_multi_split_confusion_index_records_all_published_splits(tmp_path: Path) -> None:
    published_root = (tmp_path / "eval" / "dataset").resolve()
    stage_root = create_stage(published_root, overwrite=False)
    for split in ("test", "val"):
        split_dir = stage_root / split
        split_dir.mkdir(parents=True)
        (split_dir / f"0829-{split}_report.json").write_text("{}\n", encoding="utf-8")

    index_path = det_infer.write_multi_split_confusion_inputs_index(
        output_root=stage_root,
        published_output_root=published_root,
        args=SimpleNamespace(
            split="test+val",
            data=tmp_path / "data.yaml",
            save_json=False,
            score_threshold=0.3,
            report_iou_threshold=0.5,
        ),
        data_cfg={"_root_dir": tmp_path / "dataset"},
        splits=["test", "val"],
    )
    payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert payload["splits"] == ["test", "val"]
    assert [item["output_dir"] for item in payload["split_outputs"]] == [
        str(published_root / "test"),
        str(published_root / "val"),
    ]
    assert ".staging-" not in json.dumps(payload)


def test_published_report_generation_reads_the_published_output(
    tmp_path: Path, monkeypatch
) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    output_dir = experiment_dir / "eval" / "dataset" / "test"
    report_path = output_dir / "0829-test_report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text("{}\n", encoding="utf-8")
    (output_dir / "run_meta.json").write_text(
        json.dumps({"paths": {"report_path": str(report_path)}}) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "single_report_0829_test_dataset.md"
    observed: list[Path] = []

    def fake_generate_report(*, experiment_dir: Path, output_dir: Path) -> list[Path]:
        assert output_dir.exists()
        assert (output_dir / "run_meta.json").exists()
        observed.append(output_dir)
        markdown_path.write_text("report\n", encoding="utf-8")
        return [markdown_path]

    monkeypatch.setattr(det_infer, "generate_report_for_infer_output", fake_generate_report)
    monkeypatch.setattr(det_infer, "copy_infer_artifacts_to_important", lambda *_args: [])

    result = det_infer.finalize_published_infer_reports(
        args=SimpleNamespace(
            save_test_report=True,
            skip_important_artifacts=False,
            tool_action="eval",
            split="test",
        ),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    assert observed == [output_dir.resolve()]
    assert result == [markdown_path]


@pytest.mark.parametrize("reuse_test", [False, True])
def test_parallel_test_val_publishes_both_splits_before_generating_reports(
    tmp_path: Path, monkeypatch, reuse_test: bool
) -> None:
    published_root = tmp_path / "out" / "exp_det" / "eval" / "dataset"
    checkpoint_path = tmp_path / "out" / "exp_det" / "exported_models" / "exported_best.pt"
    args = SimpleNamespace(
        tool_action="eval",
        command="eval",
        experiment_dir=checkpoint_path.parents[1],
        checkpoint=None,
        data=tmp_path / "data.yaml",
        image=None,
        image_dir=None,
        split="test+val",
        output_dir=published_root,
        report_path=None,
        score_threshold=0.3,
        report_iou_threshold=0.5,
        bad_class_map50_threshold=0.3,
        device="auto",
        save_visualization=True,
        vis_max_images=50,
        save_json=False,
        save_txt=False,
        compute_metrics=True,
        metric_classwise=False,
        save_test_report=True,
        overwrite=True,
        sahi=False,
        dry_run=False,
        skip_important_artifacts=False,
        selected_splits=None,
        multi_output_root=None,
    )
    samples = [
        det_infer.SplitSample(split, SimpleNamespace(image_path=tmp_path / f"{split}-{i}.jpg"))
        for split in ("test", "val")
        for i in range(2)
    ]
    if reuse_test:
        reused_test_dir = published_root / "test"
        reused_test_dir.mkdir(parents=True)
        (reused_test_dir / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
        (reused_test_dir / "0829-test_report.json").write_text("{}\n", encoding="utf-8")
        (reused_test_dir / "run_meta.json").write_text(
            json.dumps(
                {"paths": {"report_path": str(reused_test_dir / "0829-test_report.json")}}
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        det_infer,
        "query_gpu_inventory",
        lambda: ([{"index": 0}, {"index": 1}], None),
    )
    monkeypatch.setattr(det_infer, "filter_high_memory_gpus", lambda gpus: gpus)
    monkeypatch.setattr(det_infer, "format_gpu_summary", lambda gpu: f"GPU {gpu['index']}")
    monkeypatch.setattr(det_infer.rt, "resolve_checkpoint_path", lambda *_args: checkpoint_path)
    monkeypatch.setattr(
        det_infer.rt,
        "load_data_config",
        lambda *_args: {"test": "images/test", "val": "images/val"},
    )
    monkeypatch.setattr(
        det_infer,
        "get_multi_split_input_samples",
        lambda _args, splits: ([item for item in samples if item.split in splits], {}, {}),
    )
    monkeypatch.setattr(det_infer.rt, "ensure_image_samples", lambda _samples: None)
    monkeypatch.setattr(det_infer, "_sample_weight", lambda _sample: 1.0)
    monkeypatch.setattr(
        det_infer,
        "_det_reuse_precheck",
        lambda split_args, *_args: (
            det_infer.run_reuse.REUSE
            if reuse_test and split_args.split == "test"
            else det_infer.run_reuse.RUN
        ),
    )

    def fake_run_shards(jobs, *, cwd, shard_totals):
        del cwd, shard_totals
        for _shard_index, _gpu_index, command in jobs:
            shard_dir = Path(command[command.index("--output-dir") + 1])
            selected_splits = command[command.index("--selected-splits") + 1].split(",")
            for split in selected_splits:
                split_dir = shard_dir / split
                split_dir.mkdir(parents=True, exist_ok=True)
                (split_dir / "shard_result.json").write_text("{}\n", encoding="utf-8")
        return []

    monkeypatch.setattr(det_infer.gpu_parallel, "run_sharded_subprocesses", fake_run_shards)

    merged: list[str] = []

    def fake_merge(*, final_output_dir: Path, split: str, args, **_kwargs) -> None:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        (final_output_dir / "metrics_summary.json").write_text("{}\n", encoding="utf-8")
        Path(args.report_path).write_text("{}\n", encoding="utf-8")
        merged.append(split)

    monkeypatch.setattr(det_infer, "merge_shard_results", fake_merge)

    def fake_finalize(*, args, final_output_dir: Path, split: str, **_kwargs) -> None:
        published_dir = Path(args._published_output_dir)
        published_report = published_dir / Path(args.report_path).name
        (final_output_dir / "run_meta.json").write_text(
            json.dumps({"paths": {"report_path": str(published_report)}}) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(det_infer, "finalize_parallel_infer_output", fake_finalize)

    indexed_splits: list[list[str]] = []

    def fake_write_index(*, output_root: Path, splits: list[str], **_kwargs) -> Path:
        indexed_splits.append(list(splits))
        index_path = output_root / "confusion_inputs.json"
        index_path.write_text("{}\n", encoding="utf-8")
        return index_path

    monkeypatch.setattr(det_infer, "write_multi_split_confusion_inputs_index", fake_write_index)
    generated: list[str] = []

    def fake_finalize_reports(*, args, output_dir: Path, **_kwargs) -> list[Path]:
        assert output_dir == published_root / args.split
        assert (published_root / "test" / "run_meta.json").exists()
        assert (published_root / "val" / "run_meta.json").exists()
        generated.append(args.split)
        return []

    monkeypatch.setattr(det_infer, "finalize_published_infer_reports", fake_finalize_reports)
    monkeypatch.setattr(det_infer, "print_published_eval_artifacts", lambda *_args: None)

    assert det_infer.run_parallel_all_infer(args, ["test", "val"]) is True
    assert merged == (["val"] if reuse_test else ["test", "val"])
    assert generated == ["test", "val"]
    assert indexed_splits == [["test", "val"]]
    assert (published_root / "test" / "metrics_summary.json").exists()
    assert (published_root / "val" / "metrics_summary.json").exists()


def test_sequential_test_val_writes_dataset_root_index_then_refreshes_reports(
    tmp_path: Path, monkeypatch
) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    data_path = tmp_path / "datasets" / "face_yolo" / "data.yaml"
    data_cfg = {
        "_root_dir": data_path.parent,
        "_data_yaml_path": data_path,
        "test": "images/test",
        "val": "images/val",
    }
    args = SimpleNamespace(
        tool_action="eval",
        command="eval",
        experiment_dir=experiment_dir,
        checkpoint=None,
        data=data_path,
        image=None,
        image_dir=None,
        split="test+val",
        output_dir=None,
        report_path=None,
        device="cpu",
        save_json=False,
        save_test_report=True,
        score_threshold=0.3,
        report_iou_threshold=0.5,
    )
    expected_root = experiment_dir.resolve() / "eval" / rt.path_identity_tag(data_path)

    monkeypatch.setattr(det_infer.rt, "load_data_config", lambda *_args: data_cfg)
    monkeypatch.setattr(det_infer.rt, "resolve_checkpoint_path", lambda *_args: checkpoint_path)
    monkeypatch.setattr(det_infer, "run_parallel_all_infer", lambda *_args: False)

    executed: list[str] = []

    def fake_single(split_args) -> None:
        assert split_args._defer_report_generation is True
        output_dir = expected_root / split_args.split
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"0829-{split_args.split}_report.json"
        report_path.write_text("{}\n", encoding="utf-8")
        (output_dir / "run_meta.json").write_text(
            json.dumps({"paths": {"report_path": str(report_path)}}) + "\n",
            encoding="utf-8",
        )
        split_args.output_dir = output_dir
        executed.append(split_args.split)

    monkeypatch.setattr(det_infer, "run_single_infer", fake_single)
    refreshed: list[tuple[Path, list[str]]] = []

    def fake_refresh(*, output_root: Path, splits: list[str], **_kwargs) -> None:
        assert (output_root / "confusion_inputs.json").exists()
        refreshed.append((output_root, list(splits)))

    monkeypatch.setattr(det_infer, "finalize_published_multi_split_reports", fake_refresh)
    monkeypatch.setattr(det_infer, "print_published_eval_artifacts", lambda *_args: None)

    det_infer.run_infer(args)

    payload = json.loads((expected_root / "confusion_inputs.json").read_text(encoding="utf-8"))
    assert executed == ["test", "val"]
    assert refreshed == [(expected_root, ["test", "val"])]
    assert payload["paths"]["output_dir"] == str(expected_root)
    assert [item["split"] for item in payload["split_outputs"]] == ["test", "val"]


def test_multi_split_dry_run_keeps_output_tree_unchanged(tmp_path: Path, monkeypatch) -> None:
    experiment_dir = tmp_path / "out" / "exp_det"
    checkpoint_path = experiment_dir / "exported_models" / "exported_best.pt"
    data_path = tmp_path / "datasets" / "face_yolo" / "data.yaml"
    data_cfg = {
        "_root_dir": data_path.parent,
        "_data_yaml_path": data_path,
        "test": "images/test",
        "val": "images/val",
    }
    args = SimpleNamespace(
        tool_action="eval",
        experiment_dir=experiment_dir,
        checkpoint=None,
        data=data_path,
        image=None,
        image_dir=None,
        split="test+val",
        output_dir=None,
        report_path=None,
        device="cpu",
        dry_run=True,
    )
    expected_root = experiment_dir.resolve() / "eval" / rt.path_identity_tag(data_path)

    monkeypatch.setattr(det_infer.rt, "load_data_config", lambda *_args: data_cfg)
    monkeypatch.setattr(det_infer.rt, "resolve_checkpoint_path", lambda *_args: checkpoint_path)
    monkeypatch.setattr(det_infer, "run_parallel_all_infer", lambda *_args: False)
    monkeypatch.setattr(det_infer, "run_single_infer", lambda _args: None)

    det_infer.run_infer(args)

    assert expected_root.exists() is False


def test_det_eval_parser_defaults_to_limited_visualizations() -> None:
    args = parse_cli_args(["eval", "--task", "det", "--data", "data.yaml"])

    assert args.tool_action == "eval"
    assert args.split == "test"
    assert args.vis_max_images == rt.DET_EVAL_VIS_MAX_IMAGES
    assert args.save_visualization is True
    assert args.compute_metrics is True
    assert args.save_test_report is True
    assert args.save_json is False


def test_det_eval_parser_normalizes_val_test() -> None:
    args = parse_cli_args(
        ["eval", "--task", "det", "--data", "data.yaml", "--split", "val", "test"]
    )

    assert args.split == "test+val"


def test_eval_visualization_indices_respect_global_shard_quota() -> None:
    shard_sets = [
        visualization_indices(
            100,
            SimpleNamespace(
                tool_action="eval",
                save_visualization=True,
                vis_max_images=5,
                shard_index=shard_index,
                num_shards=3,
            ),
        )
        for shard_index in range(3)
    ]

    assert [len(indices) for indices in shard_sets] == [2, 2, 1]


def test_run_eval_applies_metric_only_profile(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(det_infer, "run_infer", lambda args: captured.update(vars(args)))
    args = SimpleNamespace(
        data=Path("data.yaml"), classwise=True, vis_max_images=12,
        save_visualization=True,
    )

    det_infer.run_eval(args)

    assert captured["tool_action"] == "eval"
    assert captured["compute_metrics"] is True
    assert captured["save_test_report"] is True
    assert captured["save_json"] is False
    assert captured["save_txt"] is False
    assert captured["metric_classwise"] is True


def test_eval_parallel_child_command_reenters_det_eval(tmp_path: Path) -> None:
    args = SimpleNamespace(
        tool_action="eval",
        experiment_dir=tmp_path / "exp",
        checkpoint=None,
        data=tmp_path / "data.yaml",
        image=None,
        image_dir=None,
        score_threshold=0.3,
        report_iou_threshold=0.5,
        bad_class_map50_threshold=0.3,
        device="auto",
        save_visualization=True,
        vis_max_images=50,
        save_json=False,
        save_txt=False,
        compute_metrics=True,
        metric_classwise=False,
        save_test_report=True,
        overwrite=True,
        sahi=False,
        dry_run=False,
        skip_important_artifacts=True,
        selected_splits=None,
        multi_output_root=None,
    )

    command = build_parallel_child_command(
        args=args,
        split="test",
        device="auto",
        output_dir=tmp_path / "shard",
        report_path=tmp_path / "shard" / "test_report.json",
        shard_index=0,
        num_shards=2,
    )

    assert command[2:5] == ["eval", "--task", "det"]
    assert "--vis-max-images" in command
    assert "--save-json" not in command


def test_report_name_tracks_requested_split(tmp_path: Path) -> None:
    output_dir = tmp_path / "infer" / "dataset-id" / "val"
    report = build_split_report_path(
        args=SimpleNamespace(report_path=None, split="val"),
        output_dir=output_dir,
        split="val",
    )

    assert report.name.endswith("-val_report.json")


def test_multi_split_report_path_is_stable_after_staging_rebase(tmp_path: Path) -> None:
    output_dir = (tmp_path / "eval" / "dataset-id" / "test").resolve()
    report = build_split_report_path(
        args=SimpleNamespace(report_path=None, split="test", requested_split="test+val"),
        output_dir=output_dir,
        split="test",
    )

    rebuilt = build_split_report_path(
        args=SimpleNamespace(
            report_path=report,
            split="test",
            requested_split="test+val",
        ),
        output_dir=output_dir,
        split="test",
    )

    assert rebuilt == report


def test_select_bad_classes_from_report_uses_threshold_and_gt() -> None:
    report_payload = {
        "per_class_ap": {
            "0": {"name": "bad", "ap": 0.29, "gt": 2, "pred": 1, "tp": 0},
            "1": {"name": "border", "ap": 0.3, "gt": 2, "pred": 1, "tp": 1},
            "2": {"name": "empty", "ap": 0.0, "gt": 0, "pred": 1, "tp": 0},
        }
    }

    bad_classes = select_bad_classes_from_report(report_payload, threshold=0.3)

    assert set(bad_classes) == {0}
    assert bad_classes[0]["class_name"] == "bad"


def test_export_bad_class_images_writes_named_visualizations_and_manifest(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_det"
    image_dir = dataset_root / "images" / "test"
    label_dir = dataset_root / "labels" / "test"
    output_dir = tmp_path / "out" / "exp_det" / "infer" / "test"

    _create_image(image_dir / "a.jpg")
    _create_image(image_dir / "b.jpg", color=(0, 255, 0))
    (label_dir / "a.txt").parent.mkdir(parents=True, exist_ok=True)
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")
    (label_dir / "b.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    _create_image(output_dir / "images" / "a.jpg", color=(0, 0, 255))
    _create_image(output_dir / "images" / "b.jpg", color=(255, 255, 0))

    report_path = output_dir / "test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "per_class_ap": {
                    "0": {"name": "缺陷/类别", "ap": 0.2, "gt": 2, "pred": 1, "tp": 0},
                    "1": {"name": "good", "ap": 0.9, "gt": 1, "pred": 1, "tp": 1},
                }
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    samples = [
        rt.ImageSample(
            image_path=image_dir / "a.jpg",
            relative_path=Path("a.jpg"),
            label_path=label_dir / "a.txt",
        ),
        rt.ImageSample(
            image_path=image_dir / "b.jpg",
            relative_path=Path("b.jpg"),
            label_path=label_dir / "b.txt",
        ),
    ]

    summary = export_bad_class_images(
        args=SimpleNamespace(bad_class_map50_threshold=0.3),
        output_dir=output_dir,
        report_path=report_path,
        data_cfg=None,
        split="test",
        samples=samples,
    )

    bad_images_dir = output_dir / "bad_images"
    assert summary["exported_count"] == 2
    assert (bad_images_dir / "缺陷-类别_1.jpg").exists()
    assert (bad_images_dir / "缺陷-类别_2.jpg").exists()

    with (bad_images_dir / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["split"] == "test"
    assert rows[0]["class_id"] == "0"
    assert rows[0]["class_name"] == "缺陷/类别"
    assert rows[0]["ap"] == "0.2"
    assert Path(rows[0]["source_image_path"]).exists()
    assert Path(rows[0]["source_visualization_path"]).exists()
    assert Path(rows[0]["export_image_path"]).exists()

    manifest_payload = json.loads((bad_images_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["summary"]["bad_class_count"] == 1
