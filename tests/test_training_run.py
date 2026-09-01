from __future__ import annotations

import json
from pathlib import Path

import pytest

from tool_lib import training_run


def test_prepare_run_guards_resume_config_and_archives_fresh_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "experiment"
    first = {"model": "a", "steps": 100, "batch_size": 2}
    second = {"model": "a", "steps": 200, "batch_size": 2}

    assert training_run.inspect_run_mode(output, fresh=False) == (False, False)
    assert training_run.prepare_run(output, fresh=False, config=first) == (
        False,
        True,
    )
    assert training_run.inspect_run_mode(output, fresh=False) == (False, True)

    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    assert training_run.prepare_run(output, fresh=False, config=first) == (
        True,
        False,
    )
    with pytest.raises(ValueError, match="steps"):
        training_run.prepare_run(output, fresh=False, config=second)

    (output / "old-event").write_text("old", encoding="utf-8")
    assert training_run.prepare_run(output, fresh=True, config=second) == (
        False,
        True,
    )
    archives = list((tmp_path / "_archive").glob("experiment-*"))
    assert len(archives) == 1
    assert (archives[0] / "old-event").is_file()
    payload = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert payload["config"] == second


def test_prepare_run_adopts_legacy_checkpoint_config(tmp_path: Path) -> None:
    output = tmp_path / "legacy"
    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert training_run.prepare_run(
        output,
        fresh=False,
        config={"model": "legacy", "steps": 100},
    ) == (True, False)
    assert (output / "run_config.json").is_file()


def test_prepare_run_rejects_non_mapping_config_file(tmp_path: Path) -> None:
    output = tmp_path / "experiment"
    checkpoint = output / "checkpoints" / "last.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (output / "run_config.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="指纹格式错误"):
        training_run.prepare_run(
            output,
            fresh=False,
            config={"model": "a", "steps": 100},
        )


def test_nonzero_rank_does_not_archive_fresh_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "experiment"
    config = {"model": "a", "steps": 100}
    training_run.prepare_run(output, fresh=False, config=config)
    monkeypatch.setenv("RANK", "1")

    assert training_run.prepare_run(output, fresh=True, config=config) == (
        False,
        True,
    )
    assert output.is_dir()
    assert not (tmp_path / "_archive").exists()
