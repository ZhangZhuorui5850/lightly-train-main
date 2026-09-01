"""Shared, cycle-safe filesystem indexing for launcher artifacts.

Dataset format recognition lives in ``convert_tools/dataset_discovery.py``.
This module covers generic launcher artifacts such as experiments, checkpoints,
run metadata and evaluation reports.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Iterator, Sequence

from .progress import track


def _unique_roots(roots: Sequence[Path]) -> list[Path]:
    unique: list[Path] = []
    for value in roots:
        root = Path(value).expanduser().resolve()
        if root.is_dir() and root not in unique:
            unique.append(root)
    return unique


def walk_tree(
    root: Path,
    *,
    label: str,
    followlinks: bool = True,
    skip_dir_names: set[str] | None = None,
    show_progress: bool = True,
) -> Iterator[tuple[Path, list[str], list[str]]]:
    """Yield one directory at a time with inode deduplication and progress."""
    resolved_root = Path(root).expanduser().resolve()
    if not resolved_root.is_dir():
        return
    skipped = {name.casefold() for name in (skip_dir_names or set())}
    visited: set[tuple[int, int]] = set()
    walker = track(
        os.walk(resolved_root, followlinks=followlinks),
        label=label,
        total=None,
        unit="dir",
        enable=show_progress,
    )
    for current, dirnames, filenames in walker:
        path = Path(current)
        try:
            stat_result = path.stat()
            identity = (stat_result.st_dev, stat_result.st_ino)
        except OSError:
            dirnames[:] = []
            continue
        if identity in visited:
            dirnames[:] = []
            continue
        visited.add(identity)
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name.casefold() not in skipped
        ]
        yield path, dirnames, filenames


def find_files(
    roots: Sequence[Path],
    *,
    label: str,
    filenames: set[str] | None = None,
    patterns: Sequence[str] = (),
    suffixes: set[str] | None = None,
    skip_dir_names: set[str] | None = None,
    followlinks: bool = True,
    show_progress: bool = True,
) -> list[Path]:
    """Find matching files with one traversal per root and stable deduplication."""
    resolved_roots = _unique_roots(roots)
    wanted_names = {name.casefold() for name in (filenames or set())}
    wanted_suffixes = {suffix.casefold() for suffix in (suffixes or set())}
    found: dict[Path, Path] = {}
    for index, root in enumerate(resolved_roots, start=1):
        root_label = label
        if len(resolved_roots) > 1:
            root_label = f"{label} {index}/{len(resolved_roots)}"
        for current, _dirnames, current_filenames in walk_tree(
            root,
            label=root_label,
            followlinks=followlinks,
            skip_dir_names=skip_dir_names,
            show_progress=show_progress,
        ):
            for name in current_filenames:
                path = current / name
                folded_name = name.casefold()
                if wanted_names and folded_name in wanted_names:
                    matched = True
                elif patterns and any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    matched = True
                elif wanted_suffixes and path.suffix.casefold() in wanted_suffixes:
                    matched = True
                else:
                    matched = False
                if not matched:
                    continue
                try:
                    resolved = path.resolve()
                    if resolved.is_file():
                        found[resolved] = resolved
                except OSError:
                    continue
    return list(found.values())
