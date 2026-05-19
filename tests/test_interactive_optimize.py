from __future__ import annotations

import json
import os
from pathlib import Path

from tool_lib import common as rt
from tool_lib import interactive


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _create_optimize_candidate(
    *,
    experiment_dir: Path,
    data_yaml: Path,
    save_prediction_json: bool,
    mtime: int,
    split: str = "test",
) -> Path:
    output_dir = experiment_dir / "infer" / split
    report_path = output_dir / "test_report.json"
    temp_dir = output_dir / rt.INFER_TEMP_DIRNAME
    json_dir = temp_dir / "json"

    _write_json(report_path, {"per_class_ap": {"0": {"name": "defect", "ap": 0.8, "gt": 4, "pred": 4, "tp": 3}}})
    _write_json(
        output_dir / "run_meta.json",
        {
            "task": "det",
            "action": "infer",
            "paths": {
                "experiment_dir": str(experiment_dir),
                "output_dir": str(output_dir),
                "temp_dir": str(temp_dir),
                "data_yaml": str(data_yaml),
                "data_root": str(data_yaml.parent),
            },
            "artifacts": {
                "confusion_inputs": str(output_dir / "confusion_inputs.json"),
            },
        },
    )
    _write_json(
        output_dir / "confusion_inputs.json",
        {
            "paths": {
                "output_dir": str(output_dir),
                "prediction_json_dir": str(json_dir),
                "report_path": str(report_path),
                "data_yaml": str(data_yaml),
                "data_root": str(data_yaml.parent),
            }
        },
    )
    if save_prediction_json:
        _write_json(
            json_dir / "sample.json",
            {
                "image": "sample.jpg",
                "predictions": [{"class_id": 0, "score": 0.9, "bbox_xyxy": [1, 2, 3, 4]}],
            },
        )
    os.utime(report_path, (mtime, mtime))
    return report_path


def _create_prediction_only_output(
    *,
    experiment_dir: Path,
    data_yaml: Path,
    split: str,
) -> Path:
    output_dir = experiment_dir / "infer" / split
    temp_dir = output_dir / rt.INFER_TEMP_DIRNAME
    json_dir = temp_dir / "json"
    _write_json(
        output_dir / "run_meta.json",
        {
            "task": "det",
            "action": "infer",
            "split": split,
            "paths": {
                "experiment_dir": str(experiment_dir),
                "output_dir": str(output_dir),
                "temp_dir": str(temp_dir),
                "data_yaml": str(data_yaml),
                "data_root": str(data_yaml.parent),
            },
        },
    )
    _write_json(
        json_dir / f"{split}_sample.json",
        {
            "image": f"{split}_sample.jpg",
            "predictions": [{"class_id": 0, "score": 0.9, "bbox_xyxy": [1, 2, 3, 4]}],
        },
    )
    return output_dir


def test_collect_optimize_analysis_candidates_marks_prediction_json(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path
    out_dir = tmp_path / "out"
    dataset_root = tmp_path / "datasets" / "steel_dataset" / "dataset_det"
    data_yaml = dataset_root / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  - defect\n", encoding="utf-8")

    monkeypatch.setattr(rt, "ROOT_DIR", root_dir)
    monkeypatch.setattr(rt, "EXPERIMENT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "TEST_OUTPUT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "ALL_REPORT_ROOT_DIR", out_dir / "all_report")

    report_without_pred = _create_optimize_candidate(
        experiment_dir=out_dir / "2026-04-01" / "steel_train",
        data_yaml=data_yaml,
        save_prediction_json=False,
        mtime=100,
    )
    report_with_pred = _create_optimize_candidate(
        experiment_dir=out_dir / "2026-04-02" / "steel_train",
        data_yaml=data_yaml,
        save_prediction_json=True,
        mtime=200,
    )

    candidates = interactive._collect_optimize_analysis_candidates(data_yaml)

    assert [candidate["report_json"] for candidate in candidates] == [report_with_pred, report_without_pred]
    assert candidates[0]["has_prediction_json"] is True
    assert candidates[0]["infer_output_dir"] == report_with_pred.parent
    assert candidates[1]["has_prediction_json"] is False
    assert candidates[1]["infer_output_dir"] is None


def test_pick_best_optimize_analysis_candidate_uses_dataset_relation(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path
    out_dir = tmp_path / "out"
    dataset_root = tmp_path / "datasets" / "steel_dataset" / "dataset_det"
    data_yaml = dataset_root / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text("path: .\ntest: images/test\nnames:\n  - defect\n", encoding="utf-8")

    monkeypatch.setattr(rt, "ROOT_DIR", root_dir)
    monkeypatch.setattr(rt, "EXPERIMENT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "TEST_OUTPUT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "ALL_REPORT_ROOT_DIR", out_dir / "all_report")

    other_dataset_root = tmp_path / "datasets" / "other_dataset" / "dataset_det"
    other_data_yaml = other_dataset_root / "data.yaml"
    other_data_yaml.parent.mkdir(parents=True, exist_ok=True)
    other_data_yaml.write_text("path: .\ntest: images/test\nnames:\n  - defect\n", encoding="utf-8")

    related_report = _create_optimize_candidate(
        experiment_dir=out_dir / "2026-03-31" / "steel_train",
        data_yaml=data_yaml,
        save_prediction_json=True,
        mtime=100,
    )
    unrelated_report = _create_optimize_candidate(
        experiment_dir=out_dir / "2026-04-01" / "steel_train",
        data_yaml=other_data_yaml,
        save_prediction_json=True,
        mtime=200,
    )

    best_candidate = interactive._pick_best_optimize_analysis_candidate(
        interactive._collect_optimize_analysis_candidates(data_yaml)
    )

    assert best_candidate is not None
    assert best_candidate["report_json"] == related_report
    assert best_candidate["report_json"] != unrelated_report


def test_collect_optimize_analysis_candidates_adds_combined_infer_candidate(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path
    out_dir = tmp_path / "out"
    dataset_root = tmp_path / "datasets" / "steel_dataset" / "dataset_det"
    data_yaml = dataset_root / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  - defect\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(rt, "ROOT_DIR", root_dir)
    monkeypatch.setattr(rt, "EXPERIMENT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "TEST_OUTPUT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "ALL_REPORT_ROOT_DIR", out_dir / "all_report")

    experiment_dir = out_dir / "2026-04-03" / "steel_train"
    _create_optimize_candidate(
        experiment_dir=experiment_dir,
        data_yaml=data_yaml,
        save_prediction_json=True,
        mtime=100,
        split="train",
    )
    _create_optimize_candidate(
        experiment_dir=experiment_dir,
        data_yaml=data_yaml,
        save_prediction_json=True,
        mtime=200,
        split="test",
    )
    _create_optimize_candidate(
        experiment_dir=experiment_dir,
        data_yaml=data_yaml,
        save_prediction_json=True,
        mtime=150,
        split="val",
    )

    candidates = interactive._collect_optimize_analysis_candidates(data_yaml)

    combined_candidate = candidates[0]
    assert combined_candidate["candidate_kind"] == "combined"
    assert combined_candidate["infer_output_dir"] == experiment_dir / "infer"
    assert combined_candidate["has_prediction_json"] is True
    assert combined_candidate["split_names"] == ["test", "val", "train"]


def test_collect_optimize_analysis_candidates_supplements_prediction_only_splits(tmp_path: Path, monkeypatch) -> None:
    root_dir = tmp_path
    out_dir = tmp_path / "out"
    dataset_root = tmp_path / "datasets" / "steel_dataset" / "dataset_det"
    data_yaml = dataset_root / "data.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  - defect\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(rt, "ROOT_DIR", root_dir)
    monkeypatch.setattr(rt, "EXPERIMENT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "TEST_OUTPUT_ROOT_DIR", out_dir)
    monkeypatch.setattr(rt, "ALL_REPORT_ROOT_DIR", out_dir / "all_report")

    experiment_dir = out_dir / "2026-04-04" / "steel_train"
    _create_optimize_candidate(
        experiment_dir=experiment_dir,
        data_yaml=data_yaml,
        save_prediction_json=True,
        mtime=200,
        split="test",
    )
    _create_prediction_only_output(experiment_dir=experiment_dir, data_yaml=data_yaml, split="val")
    _create_prediction_only_output(experiment_dir=experiment_dir, data_yaml=data_yaml, split="train")

    candidates = interactive._collect_optimize_analysis_candidates(data_yaml)

    assert candidates[0]["candidate_kind"] == "combined"
    assert candidates[0]["split_names"] == ["test", "val", "train"]
    prediction_only = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_kind") == "single" and candidate.get("report_json") is None
    ]
    assert {candidate["split_name"] for candidate in prediction_only} == {"val", "train"}
