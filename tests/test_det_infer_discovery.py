from __future__ import annotations

import json
from pathlib import Path

from tool_lib import interactive
from tool_lib.det_problem_export import discover_infer_runs
from tool_lib.det_report import _discover_infer_runs as discover_report_runs


def _write_run_meta(
    output_dir: Path,
    *,
    recorded_experiment: Path,
    recorded_output: Path,
    data_yaml: Path,
    split: str = "test",
    action: str = "infer",
) -> None:
    output_dir.mkdir(parents=True)
    report = output_dir / f"{split}_report.json"
    report.write_text("{}\n", encoding="utf-8")
    (output_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "task": "det",
                "action": action,
                "split": split,
                "created_at": "2026-01-01T00:00:00+08:00",
                "paths": {
                    "experiment_dir": str(recorded_experiment),
                    "output_dir": str(recorded_output),
                    "report_path": str(recorded_output / report.name),
                    "data_yaml": str(data_yaml),
                    "data_root": str(data_yaml.parent),
                },
            }
        ),
        encoding="utf-8",
    )


def test_discover_infer_runs_rebases_paths_after_project_move(tmp_path: Path) -> None:
    experiment = tmp_path / "new_project" / "out" / "exp"
    output = experiment / "infer-neu-test"
    old_experiment = Path("/old/project/out/exp")
    old_output = old_experiment / "infer-neu-test"
    data_yaml = tmp_path / "datasets" / "neu" / "data.yaml"
    _write_run_meta(
        output,
        recorded_experiment=old_experiment,
        recorded_output=old_output,
        data_yaml=data_yaml,
    )

    runs = discover_infer_runs(experiment)

    assert len(runs) == 1
    assert runs[0]["output_dir"] == output.resolve()
    assert runs[0]["report_path"] == (output / "test_report.json").resolve()

    report_runs = discover_report_runs(experiment)
    assert len(report_runs) == 1
    assert report_runs[0].output_dir == output.resolve()
    assert report_runs[0].report_path == (output / "test_report.json").resolve()


def test_existing_output_discovery_matches_legacy_directory_by_metadata(tmp_path: Path) -> None:
    experiment = tmp_path / "out" / "exp"
    data_yaml = tmp_path / "datasets" / "neu" / "data.yaml"
    other_yaml = tmp_path / "datasets" / "other" / "data.yaml"
    matching = experiment / "infer-neu-test"
    unrelated = experiment / "infer-other-test"
    _write_run_meta(
        matching,
        recorded_experiment=experiment,
        recorded_output=matching,
        data_yaml=data_yaml,
    )
    _write_run_meta(
        unrelated,
        recorded_experiment=experiment,
        recorded_output=unrelated,
        data_yaml=other_yaml,
    )

    outputs = interactive._collect_existing_det_infer_outputs(
        experiment_dir=experiment,
        data_path=data_yaml,
    )

    assert outputs == {"test": matching.resolve()}


def test_report_discovery_accepts_eval_runs(tmp_path: Path) -> None:
    experiment = tmp_path / "out" / "exp"
    data_yaml = tmp_path / "datasets" / "neu" / "data.yaml"
    output = experiment / "eval" / "neu" / "test"
    _write_run_meta(
        output,
        recorded_experiment=experiment,
        recorded_output=output,
        data_yaml=data_yaml,
        action="eval",
    )

    report_runs = discover_report_runs(experiment)

    assert len(report_runs) == 1
    assert report_runs[0].run_meta["action"] == "eval"
    assert discover_infer_runs(experiment) == []
