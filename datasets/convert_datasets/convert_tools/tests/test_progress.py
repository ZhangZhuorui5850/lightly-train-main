from __future__ import annotations

import sys
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import progress  # noqa: E402


def test_redirected_progress_emits_periodic_snapshot_and_completion(
    monkeypatch, capsys,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(progress, "_isatty", lambda _stream: False)
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("LIGHTLY_PROGRESS_SNAPSHOT_INTERVAL", "5")

    bar = progress.tqdm(["a", "b"], desc="转换", unit="img")
    iterator = iter(bar)
    assert next(iterator) == "a"
    clock[0] = 106.0
    assert next(iterator) == "b"
    clock[0] = 107.0
    try:
        next(iterator)
    except StopIteration:
        pass

    output = capsys.readouterr().err
    assert "[progress] 转换 0/2" in output
    assert "转换 1/2" in output
    assert "转换 2/2" in output
    assert "complete" in output


def test_disabled_progress_stays_quiet(monkeypatch, capsys) -> None:
    monkeypatch.setattr(progress, "_isatty", lambda _stream: False)
    assert list(progress.tqdm([1, 2], disable=True)) == [1, 2]
    assert capsys.readouterr().err == ""


def test_manual_progress_reports_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(progress, "_isatty", lambda _stream: False)
    try:
        with progress.tqdm(total=3, desc="合并") as bar:
            bar.update()
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    output = capsys.readouterr().err
    assert "合并 1/3" in output
    assert "failed" in output
