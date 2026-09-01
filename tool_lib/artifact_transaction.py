"""Atomic publication for inference/evaluation directory artifacts."""

from __future__ import annotations

import os
import atexit
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def create_stage(output: Path, *, overwrite: bool) -> Path:
    """Create a sibling stage without touching an existing published output."""
    output = Path(output).expanduser().resolve()
    if output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output}. Use overwrite to continue.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    atexit.register(shutil.rmtree, stage, ignore_errors=True)
    return stage


def publish_stage(stage: Path, output: Path, *, overwrite: bool) -> None:
    """Atomically replace output with a successfully validated stage."""
    stage = Path(stage).expanduser().resolve()
    output = Path(output).expanduser().resolve()
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


@contextmanager
def staged_directory(output: Path, *, overwrite: bool) -> Iterator[Path]:
    """Yield an empty sibling directory and atomically publish it on success."""
    output = Path(output).expanduser().resolve()
    if output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {output}")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output}. Use overwrite to continue.")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    backup: Path | None = None
    committed = False
    try:
        yield stage
        if output.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=output.parent))
            backup.rmdir()
            os.replace(output, backup)
        try:
            os.replace(stage, output)
            committed = True
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists():
            if committed:
                shutil.rmtree(backup)
            elif not output.exists():
                os.replace(backup, output)
