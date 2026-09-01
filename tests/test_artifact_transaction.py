from __future__ import annotations

import pytest

from tool_lib.artifact_transaction import staged_directory


def test_staged_directory_preserves_previous_output_on_failure(tmp_path):
    output = tmp_path / "infer"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="boom"):
        with staged_directory(output, overwrite=True) as stage:
            (stage / "new.txt").write_text("new", encoding="utf-8")
            raise RuntimeError("boom")

    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (output / "new.txt").exists()


def test_staged_directory_atomically_replaces_successful_output(tmp_path):
    output = tmp_path / "infer"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")

    with staged_directory(output, overwrite=True) as stage:
        (stage / "run_meta.json").write_text('{"complete": true}', encoding="utf-8")

    assert not (output / "old.txt").exists()
    assert (output / "run_meta.json").is_file()
