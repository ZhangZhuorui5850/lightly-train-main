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


def test_symlink_output_preserves_target(tmp_path):
    output = tmp_path / "output"
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep")
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        with staged_directory(output, overwrite=True):
            pytest.fail("symlink output accepted")
    assert marker.read_text() == "keep"


def test_publish_rechecks_concurrent_output(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not empty"):
        with staged_directory(output, overwrite=False) as first:
            (first / "first.txt").write_text("first")
            with staged_directory(output, overwrite=False) as second:
                (second / "second.txt").write_text("second")
    assert (output / "second.txt").read_text() == "second"
    assert not (output / "first.txt").exists()
