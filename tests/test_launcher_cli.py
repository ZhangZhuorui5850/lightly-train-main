from __future__ import annotations

import argparse
import sys
from types import ModuleType
from pathlib import Path
from unittest.mock import Mock

import pytest

import launcher
from tool_lib import common
from tool_lib import interactive
from tool_lib.dispatch import dispatch
from tool_lib.interactive import parse_cli_args, validate_args


def test_launcher_no_longer_exposes_convert_subcommand():
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_args(["convert"])
    assert exc_info.value.code == 2


def test_launcher_validates_every_iou_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.SAHI_SETTINGS, "det_sahi_nms_iou", 1.2)

    with pytest.raises(ValueError, match="det_sahi_nms_iou"):
        launcher.build_user_settings()


def test_launcher_validates_dataset_search_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.COMMON_SETTINGS, "dataset_search_roots", ["datasets", 3])

    with pytest.raises(TypeError, match="dataset_search_roots"):
        launcher.build_user_settings()


def test_launcher_accepts_manual_balance_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.DET_SETTINGS, "det_export_balance_ratio", 4.0)
    assert launcher.build_user_settings()["det_export_balance_ratio"] == 4.0


def test_launcher_rejects_fractional_balance_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.DET_SETTINGS, "det_export_balance_ratio", 0.5)
    with pytest.raises(ValueError, match="大于等于 1"):
        launcher.build_user_settings()


def test_launcher_rejects_full_sahi_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.SAHI_SETTINGS, "det_sahi_overlap", 1.0)
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        launcher.build_user_settings()


def test_launcher_rejects_nan_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.CLS_SETTINGS, "cls_threshold", float("nan"))
    with pytest.raises(ValueError, match="cls_threshold"):
        launcher.build_user_settings()


def test_launcher_rejects_string_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(launcher.DET_SETTINGS, "det_eval_vis_max_images", "50")
    with pytest.raises(TypeError, match="det_eval_vis_max_images"):
        launcher.build_user_settings()


def test_validate_args_rejects_negative_seg_visualization_limit() -> None:
    with pytest.raises(ValueError, match="vis-max-images"):
        validate_args(argparse.Namespace(vis_max_images=-1))


def test_interactive_cls_eval_builds_complete_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(interactive, "prompt_choice", Mock(side_effect=["cls", "eval"]))
    monkeypatch.setattr(interactive, "prompt_experiment_dir", Mock(return_value=Path("out/cls")))
    monkeypatch.setattr(interactive, "prompt_required_path", Mock(return_value=Path("data/test")))
    monkeypatch.setattr(interactive, "prompt_text", Mock(side_effect=["out/cls/eval", "cpu"]))
    monkeypatch.setattr(interactive, "prompt_float", Mock(return_value=0.5))
    monkeypatch.setattr(interactive, "prompt_int", Mock(return_value=1))
    monkeypatch.setattr(interactive, "confirm_args", lambda _title, args: args)

    args = interactive.build_interactive_args()

    assert args is not None
    for field in ("checkpoint", "experiment_dir", "output_dir", "device", "test_dir", "topk", "threshold"):
        assert hasattr(args, field)


def test_clean_dispatch_calls_cleaner(monkeypatch: pytest.MonkeyPatch) -> None:
    from tool_lib import exp_cleaner

    run_clean = Mock()
    monkeypatch.setattr(exp_cleaner, "run_clean", run_clean)
    args = argparse.Namespace(tool_task="clean", tool_action="clean", analyses=[])
    dispatch(args)
    run_clean.assert_called_once_with(args)


def test_interactive_clean_reprompts_invalid_preserve_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_lib import exp_cleaner

    analyses = [
        exp_cleaner.ExperimentAnalysis(Path("out/first"), 100),
        exp_cleaner.ExperimentAnalysis(Path("out/second"), 200),
    ]
    monkeypatch.setattr(exp_cleaner, "scan_experiments", Mock(return_value=analyses))
    monkeypatch.setattr(exp_cleaner, "print_clean_report", Mock())
    monkeypatch.setattr(exp_cleaner, "print_clean_preview", Mock())
    monkeypatch.setattr(interactive, "read_input", Mock(side_effect=["abc", "1"]))
    monkeypatch.setattr(interactive, "prompt_yes_no", Mock(return_value=True))

    args = interactive._build_clean_args()

    assert args is not None
    assert args.analyses == [analyses[1]]


def test_optimize_auto_infer_branch_initializes_report_list(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(interactive, "prompt_choice", Mock(side_effect=["det", "optimize"]))
    monkeypatch.setattr(interactive, "prompt_dataset_yaml", Mock(return_value=Path("data.yaml")))
    monkeypatch.setattr(interactive, "_collect_optimize_analysis_candidates", Mock(return_value=[]))
    monkeypatch.setattr(interactive, "_prompt_optimize_report_json", Mock(return_value=None))
    monkeypatch.setattr(
        interactive,
        "_run_auto_infer_for_optimize",
        Mock(return_value=(None, None, None, None)),
    )
    monkeypatch.setattr(interactive, "prompt_yes_no", Mock(return_value=True))

    args = interactive.build_interactive_args()

    assert args is not None
    assert args.report_jsons is None


def test_help_skips_user_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    apply_settings = Mock(side_effect=AssertionError("help must skip settings"))
    monkeypatch.setattr(launcher.rt, "apply_user_settings", apply_settings)
    with pytest.raises(SystemExit) as exc_info:
        launcher.main(["--help"])
    assert exc_info.value.code == 0
    apply_settings.assert_not_called()


def test_clean_and_optimize_cli_are_registered() -> None:
    clean = parse_cli_args(["clean", "--dry-run"])
    optimize = parse_cli_args(["optimize", "--source-data", "data.yaml"])
    assert (clean.tool_task, clean.tool_action) == ("clean", "clean")
    assert (optimize.tool_task, optimize.tool_action) == ("det", "optimize")


def test_negative_toggle_help_describes_disabled_state(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_cli_args(["infer", "--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--skip-visualization" in output
    assert "跳过可视化结果" in output
    assert "--no-sahi-skip-small" in output
    assert "所有图片均按已启用的 SAHI 策略处理" in output
    skip_line = next(line for line in output.splitlines() if "--skip-visualization" in line)
    assert "default:" not in skip_line


def test_runtime_import_rejects_preloaded_external_lightly_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = ModuleType("lightly_train")
    external.__file__ = "/tmp/site-packages/lightly_train/__init__.py"
    monkeypatch.setitem(sys.modules, "lightly_train", external)

    with pytest.raises(RuntimeError, match="仓库 fork"):
        common.import_runtime_dependencies()


def test_repo_source_is_reprioritized(monkeypatch: pytest.MonkeyPatch) -> None:
    source = str(common.SRC_DIR)
    monkeypatch.setattr(common.sys, "path", ["/tmp/external", source, "/tmp/other"])

    common._prioritize_repo_source()

    assert common.sys.path == [source, "/tmp/external", "/tmp/other"]


def test_report_dispatch_stays_on_lightweight_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tool_lib import det_tools

    runtime_import = Mock(side_effect=AssertionError("report must stay lightweight"))
    run_report = Mock()
    monkeypatch.setattr(common, "import_runtime_dependencies", runtime_import)
    monkeypatch.setattr(det_tools, "run_report", run_report)
    args = argparse.Namespace(tool_task="det", tool_action="report")

    dispatch(args)

    runtime_import.assert_not_called()
    run_report.assert_called_once_with(args)


def test_report_cli_auto_selects_without_input(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = Path("out/20260831/det-exp")
    monkeypatch.setattr(interactive, "list_experiment_dirs", Mock(return_value=[selected]))

    args = parse_cli_args(["report"])

    assert args.experiment_dir == selected
