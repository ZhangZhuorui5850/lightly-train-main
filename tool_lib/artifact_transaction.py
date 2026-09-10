"""Atomic publication for inference/evaluation directory artifacts."""

from __future__ import annotations

import os
import atexit
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # Windows retains publication-time overwrite checks.
    fcntl = None


def create_stage(output: Path, *, overwrite: bool) -> Path:
    """Create a sibling stage without touching an existing published output."""
    output = Path(output).expanduser()
    if output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output}. Use overwrite to continue.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    atexit.register(shutil.rmtree, stage, ignore_errors=True)
    return stage


def _publish_stage(stage: Path, output: Path, *, overwrite: bool) -> None:
    """Atomically replace output with a successfully validated stage."""
    stage = Path(stage).expanduser().resolve()
    output = Path(output).expanduser()
    if output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {output}")
    if not stage.is_dir():
        raise FileNotFoundError(f"staging 目录不存在: {stage}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output}. Use overwrite to continue.")
    backup: Path | None = None
    if output.exists():
        backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
        backup.rmdir()
        os.replace(output, backup)
    try:
        os.replace(stage, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def publish_stage(stage: Path, output: Path, *, overwrite: bool) -> None:
    """Serialize publication and recheck overwrite permission under the lock."""
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / f".{output.name}.lock").open("a+") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _publish_stage(stage, output, overwrite=overwrite)
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def staged_directory(output: Path, *, overwrite: bool) -> Iterator[Path]:
    """Generate privately, then publish with a fresh overwrite check."""
    stage = create_stage(output, overwrite=overwrite)
    try:
        yield stage
        publish_stage(stage, output, overwrite=overwrite)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
