from __future__ import annotations

from pathlib import Path

from tool_lib import file_index


def test_find_files_matches_multiple_patterns_in_one_tree_walk(
    tmp_path: Path, monkeypatch,
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = tmp_path / "a" / "test_report.json"
    second = tmp_path / "b" / "seg_eval_summary.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    walk_calls = 0
    original_walk = file_index.os.walk

    def counted_walk(*args, **kwargs):
        nonlocal walk_calls
        walk_calls += 1
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(file_index.os, "walk", counted_walk)
    found = file_index.find_files(
        [tmp_path],
        label="test",
        patterns=["*test_report.json", "*seg_eval_summary.json"],
        show_progress=False,
    )

    assert set(found) == {first.resolve(), second.resolve()}
    assert walk_calls == 1


def test_find_files_prunes_excluded_dirs(tmp_path: Path) -> None:
    keep = tmp_path / "run" / "run_meta.json"
    skipped = tmp_path / "important" / "run_meta.json"
    keep.parent.mkdir()
    skipped.parent.mkdir()
    keep.write_text("{}", encoding="utf-8")
    skipped.write_text("{}", encoding="utf-8")

    found = file_index.find_files(
        [tmp_path],
        label="test",
        filenames={"run_meta.json"},
        skip_dir_names={"important"},
        show_progress=False,
    )

    assert found == [keep.resolve()]


def test_walk_tree_deduplicates_symlink_cycles(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    try:
        (nested / "loop").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        return

    visited = list(
        file_index.walk_tree(
            tmp_path,
            label="test",
            followlinks=True,
            show_progress=False,
        )
    )

    assert [path.resolve() for path, _dirs, _files in visited] == [
        tmp_path.resolve(),
        nested.resolve(),
    ]
