from pathlib import Path

from tool_lib.det_report import InferRunRecord, _build_per_class_table


def _make_run(per_class_ap: dict[str, dict[str, object]]) -> InferRunRecord:
    return InferRunRecord(
        output_dir=Path("out/demo"),
        split="test",
        created_at=None,
        report_path=None,
        metrics_path=None,
        run_meta_path=Path("out/demo/run_meta.json"),
        report_payload={"per_class_ap": per_class_ap},
        metrics_payload=None,
        run_meta={},
        mtime=0.0,
    )


def test_build_per_class_table_sorts_by_ap_ascending() -> None:
    run = _make_run(
        {
            "2": {"name": "class_b", "gt": 12, "pred": 13, "ap": 0.8},
            "0": {"name": "class_a", "gt": 10, "pred": 11, "ap": 0.2},
            "1": {"name": "class_c", "gt": 8, "pred": 9, "ap": 0.5},
        }
    )

    table = _build_per_class_table(run)
    lines = table.splitlines()

    assert lines[0] == "| 类别名称 (Label) | GT | Pred | AP@0.5 |"
    assert lines[2] == "| class_a | 10 | 11 | 0.2000 |"
    assert lines[3] == "| class_c | 8 | 9 | 0.5000 |"
    assert lines[4] == "| class_b | 12 | 13 | 0.8000 |"


def test_build_per_class_table_keeps_only_present_columns() -> None:
    run = _make_run(
        {
            "0": {"name": "class_a", "gt": 10, "ap": 0.2},
            "1": {"name": "class_b", "gt": 8, "ap": 0.4},
        }
    )

    table = _build_per_class_table(run)
    lines = table.splitlines()

    assert lines[0] == "| 类别名称 (Label) | GT | AP@0.5 |"
    assert "Pred" not in lines[0]
    assert lines[2] == "| class_a | 10 | 0.2000 |"
