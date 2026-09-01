"""Shared safety helpers for dataset conversion output.

The converters write into a sibling staging directory and publish the completed
tree with one rename.  This keeps an existing output intact when validation or
file conversion fails halfway through a run.
"""

from __future__ import annotations

import hashlib
import atexit
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback keeps atomic publication
    fcntl = None  # type: ignore[assignment]


def file_digest(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 digest without loading a whole image in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_fingerprint(value: Any) -> str:
    """Hash JSON-compatible metadata with deterministic key ordering."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def same_filesystem_identity(path: Path) -> tuple[int, int] | None:
    """Return ``(device, inode)`` when the path exists."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def validate_distinct_sources(
    roots_and_configs: list[tuple[Path, Path]],
) -> None:
    """Reject aliases, repeated configs, and nested dataset roots.

    Validation runs after dataset resolution, so ``root`` and ``root/data.yaml``
    resolve to the same canonical source.
    """
    seen_roots: dict[Path, Path] = {}
    seen_configs: dict[Path, Path] = {}
    seen_inodes: dict[tuple[int, int], Path] = {}
    canonical: list[tuple[Path, Path]] = []
    for raw_root, raw_config in roots_and_configs:
        root = raw_root.expanduser().resolve()
        config = raw_config.expanduser().resolve()
        if root in seen_roots:
            raise ValueError(
                f"输入数据集重复: {root}（同时来自 {seen_roots[root]} 和 {config}）"
            )
        if config in seen_configs:
            raise ValueError(f"输入数据集配置重复: {config}")
        for candidate in (root, config):
            identity = same_filesystem_identity(candidate)
            if identity is not None and identity in seen_inodes:
                raise ValueError(
                    f"输入数据集通过链接/别名重复: {candidate} -> {seen_inodes[identity]}"
                )
            if identity is not None:
                seen_inodes[identity] = candidate
        seen_roots[root] = config
        seen_configs[config] = root
        canonical.append((root, config))

    for index, (left, _) in enumerate(canonical):
        for right, _ in canonical[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError(f"输入数据集根目录存在嵌套关系: {left} <-> {right}")


def validate_output_location(output: Path, source_roots: list[Path]) -> Path:
    raw_output = output.expanduser()
    if raw_output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {raw_output}")
    output = raw_output.resolve()
    for root_value in source_roots:
        root = root_value.expanduser().resolve()
        if output == root or output in root.parents or root in output.parents:
            raise ValueError(f"输出目录需要与输入数据集目录分离: {root}")
    return output


def create_output_stage(output: Path, *, clean: bool) -> Path:
    """Create a sibling stage without changing the currently published tree."""
    output = output.expanduser().resolve()
    if output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {output}")
    if output.exists() and any(output.iterdir()) and not clean:
        raise FileExistsError(f"输出目录已有内容: {output}；使用 --clean 重新生成")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    atexit.register(shutil.rmtree, stage, ignore_errors=True)
    return stage


def publish_output_stage(stage: Path, output: Path, *, clean: bool) -> None:
    """Atomically publish a completed stage and restore the old tree on failure."""
    stage = stage.expanduser().resolve()
    output = output.expanduser().resolve()
    if not stage.is_dir():
        raise FileNotFoundError(f"staging 目录不存在: {stage}")
    if output.exists() and any(output.iterdir()) and not clean:
        raise FileExistsError(f"输出目录已有内容: {output}；使用 --clean 重新生成")
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
def _output_lock(output: Path) -> Iterator[None]:
    """Serialize writers targeting the same output directory on POSIX systems."""
    if fcntl is None:
        yield
        return
    lock_path = output.parent / f".{output.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"输出目录正在由另一进程生成: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _staged_output_unlocked(output: Path, *, clean: bool = False) -> Iterator[Path]:
    """Implement staging while the caller holds the per-output writer lock."""
    raw_output = output.expanduser()
    if raw_output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {raw_output}")
    output = raw_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"输出路径已是文件: {output}")
        if any(output.iterdir()) and not clean:
            raise FileExistsError(
                f"输出目录已有内容: {output}；使用 --clean 重新生成"
            )

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    ).resolve()
    backup: Path | None = None
    committed = False
    try:
        yield stage
        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=f".{output.name}.backup-", dir=str(output.parent))
            ).resolve()
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


@contextmanager
def staged_output(output: Path, *, clean: bool = False) -> Iterator[Path]:
    """Yield a staging directory and atomically publish it under a writer lock."""
    output = output.expanduser().resolve()
    with _output_lock(output):
        with _staged_output_unlocked(output, clean=clean) as stage:
            yield stage
