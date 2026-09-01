from __future__ import annotations

import os
from pathlib import Path

from tool_lib import common


def test_recent_experiment_discovery_filters_task_before_sorting(tmp_path, monkeypatch):
    det_old = tmp_path / "det_old"
    seg_new = tmp_path / "seg_new"
    det_old.mkdir()
    seg_new.mkdir()
    os.utime(det_old, (1, 1))
    os.utime(seg_new, (2, 2))

    monkeypatch.setattr(common, "EXPERIMENT_ROOT_DIR", tmp_path)
    monkeypatch.setattr(common, "is_experiment_dir", lambda path, **_kwargs: path.is_dir())

    assert common.discover_recent_experiment_dirs("det", limit=1) == [det_old]


def test_experiment_discovery_reads_task_from_log_and_keeps_incomplete_runs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    det_run = tmp_path / "dated" / "run_a"
    seg_run = tmp_path / "dated" / "run_b"
    for path, task, timestamp in (
        (det_run, "object_detection", 2_000),
        (seg_run, "instance_segmentation", 3_000),
    ):
        path.mkdir(parents=True)
        log = path / "train.log"
        log.write_text(f'Args: {{"task": "{task}"}}\n', encoding="utf-8")
        os.utime(log, (timestamp, timestamp))

    monkeypatch.setattr(common, "EXPERIMENT_ROOT_DIR", tmp_path)

    assert common.discover_recent_experiment_dirs("det") == [det_run.resolve()]
    assert common.discover_recent_experiment_dirs("seg") == [seg_run.resolve()]


def test_experiment_discovery_accepts_custom_checkpoint_names_and_sorts_by_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    older = tmp_path / "det" / "older"
    newer = tmp_path / "det" / "newer"
    for path, timestamp in ((older, 1_000), (newer, 2_000)):
        checkpoint = path / "checkpoints" / f"epoch_{timestamp}.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        os.utime(checkpoint, (timestamp, timestamp))
        os.utime(path, (10, 10))

    monkeypatch.setattr(common, "EXPERIMENT_ROOT_DIR", tmp_path)

    assert common.discover_recent_experiment_dirs(
        "det",
        require_checkpoint=True,
    ) == [newer.resolve(), older.resolve()]


def test_experiment_discovery_follows_symlinked_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    experiment_root = tmp_path / "out"
    experiment_root.mkdir()
    mounted_run = tmp_path / "mounted_runs" / "run_01"
    mounted_run.mkdir(parents=True)
    (mounted_run / "train.log").write_text(
        'Args: {"task": "object_detection"}\n',
        encoding="utf-8",
    )
    (experiment_root / "run_link").symlink_to(
        mounted_run,
        target_is_directory=True,
    )
    (mounted_run / "loop").symlink_to(experiment_root, target_is_directory=True)
    monkeypatch.setattr(common, "EXPERIMENT_ROOT_DIR", experiment_root)

    assert common.discover_recent_experiment_dirs("det") == [
        mounted_run.resolve()
    ]


def test_experiment_seg_type_reads_training_metadata(tmp_path: Path) -> None:
    semantic = tmp_path / "semantic_run"
    instance = tmp_path / "instance_run"
    semantic.mkdir()
    instance.mkdir()
    (semantic / "train.log").write_text(
        'Args: {"task": "semantic_segmentation"}\n', encoding="utf-8"
    )
    (instance / "train.log").write_text(
        'Args: {"task": "instance_segmentation"}\n', encoding="utf-8"
    )

    assert common.experiment_seg_type(semantic) == "semantic"
    assert common.experiment_seg_type(instance) == "instance"
