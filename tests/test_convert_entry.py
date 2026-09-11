from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import Mock


CONVERT_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "convert_datasets"
    / "convert.py"
)


def _load_entry() -> dict[str, object]:
    namespace = runpy.run_path(str(CONVERT_PATH), run_name="convert_entry_test")
    return namespace["main"].__globals__


def test_registry_doctor_passes() -> None:
    entry = _load_entry()
    assert entry["main"](["doctor"]) == 0


def test_unrelated_missing_tool_does_not_block_valid_command() -> None:
    entry = _load_entry()
    commands = entry["COMMANDS"]
    old = commands["datasets"]
    commands["datasets"] = entry["Tool"](
        "missing.py", old.desc, old.stage, old.usage, old.interactive,
        old.primary, old.read_only,
    )
    run = Mock(return_value=7)
    entry["_run"] = run

    assert entry["main"](["auto", "--list"]) == 7
    run.assert_called_once()


def test_doctor_reports_missing_tool() -> None:
    entry = _load_entry()
    commands = entry["COMMANDS"]
    old = commands["datasets"]
    commands["datasets"] = entry["Tool"](
        "missing.py", old.desc, old.stage, old.usage, old.interactive,
        old.primary, old.read_only,
    )
    assert entry["main"](["doctor"]) == 2


def test_interactive_reprompts_after_invalid_selection(monkeypatch, capsys) -> None:
    entry = _load_entry()
    answers = iter(["999", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert entry["interactive"]() == 0
    assert "请重新输入" in capsys.readouterr().out


def test_noninteractive_tool_blank_args_shows_help_then_returns(monkeypatch) -> None:
    entry = _load_entry()
    answers = iter(["oneclick", "", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    run = Mock(return_value=0)
    entry["_run"] = run

    assert entry["interactive"]() == 0
    assert run.call_args.args[1] == ["-h"]


def test_all_primary_writers_declare_dry_run() -> None:
    entry = _load_entry()
    assert entry["validate_registry"]() == []


def test_interactive_keyboard_interrupt_returns_130(monkeypatch) -> None:
    entry = _load_entry()

    def interrupted(_prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupted)
    assert entry["interactive"]() == 130


def test_signaled_child_exit_is_normalized(monkeypatch) -> None:
    entry = _load_entry()
    monkeypatch.setattr(entry["subprocess"], "call", Mock(return_value=-2))

    assert entry["_run"](entry["COMMANDS"]["datasets"], []) == 130


def test_doctor_checks_real_cli_contract(monkeypatch):
    import subprocess
    entry = _load_entry()
    monkeypatch.setattr(entry["subprocess"], "run", Mock(return_value=subprocess.CompletedProcess([], 0, "usage: example", "")))
    assert entry["main"](["doctor"]) == 2


def test_interactive_returns_to_menu_after_tool_argument_error(monkeypatch):
    entry = _load_entry()
    answers = iter(["oneclick", "--invalid", "q"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    entry["_run"] = Mock(return_value=2)
    assert entry["interactive"]() == 0


def test_doctor_checks_auxiliary_cli_startup(monkeypatch):
    import subprocess
    entry = _load_entry()

    def run(command, **kwargs):
        broken = Path(command[1]).name == "seg_sample_browse.py"
        return subprocess.CompletedProcess(command, 2 if broken else 0, "--dry-run", "bad import" if broken else "")

    monkeypatch.setattr(entry["subprocess"], "run", run)
    assert entry["main"](["doctor"]) == 2
