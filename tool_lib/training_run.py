"""Safe first-run, resume, and fresh-run handling for standalone trainers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


_CONFIG_NAME = "run_config.json"
_CONFIG_VERSION = 1


def inspect_run_mode(output: Path, fresh: bool) -> tuple[bool, bool]:
    """Return ``(resume_interrupted, overwrite)`` without changing the filesystem."""
    raw_output = output.expanduser()
    if raw_output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {raw_output}")
    output = raw_output.resolve()
    if fresh:
        return False, True
    checkpoint = output / "checkpoints" / "last.ckpt"
    if checkpoint.is_file():
        return True, False
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"输出路径已经是文件: {output}")
        remaining = [path for path in output.iterdir() if path.name != _CONFIG_NAME]
        if remaining:
            raise FileExistsError(
                f"输出目录已有内容但缺少续训权重: {output}；"
                "请更换输出目录，或确认后启用 fresh"
            )
        return False, bool(list(output.iterdir()))
    return False, False


def _normalize(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _read_config(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise ValueError(f"训练参数指纹损坏: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"训练参数指纹格式错误: {path}")
    if payload.get("version") != _CONFIG_VERSION or not isinstance(
        payload.get("config"), dict
    ):
        raise ValueError(f"训练参数指纹格式错误: {path}")
    return payload["config"]


def _write_config(path: Path, config: dict[str, Any]) -> None:
    payload = {"version": _CONFIG_VERSION, "config": _normalize(config)}
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _archive_output(output: Path) -> Path | None:
    if not output.exists():
        return None
    if output.is_symlink() or not output.is_dir():
        raise ValueError(f"fresh 输出路径需要为普通目录: {output}")
    archive_root = output.parent / "_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    archive = archive_root / f"{output.name}-{timestamp}"
    suffix = 1
    while archive.exists():
        archive = archive_root / f"{output.name}-{timestamp}-{suffix}"
        suffix += 1
    os.replace(output, archive)
    print(f"旧实验已归档: {archive}", flush=True)
    return archive


def prepare_run(
    output: Path,
    *,
    fresh: bool,
    config: dict[str, Any],
) -> tuple[bool, bool]:
    """Prepare metadata and return LightlyTrain resume/overwrite arguments."""
    raw_output = output.expanduser()
    if raw_output.is_symlink():
        raise ValueError(f"输出路径不能是符号链接: {raw_output}")
    output = raw_output.resolve()
    current = _normalize(config)
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return inspect_run_mode(output, fresh=False)
    if fresh:
        _archive_output(output)

    resume_interrupted, overwrite = inspect_run_mode(output, fresh=False)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / _CONFIG_NAME

    if resume_interrupted:
        saved = _read_config(config_path)
        if saved is None:
            _write_config(config_path, current)
            print(
                f"续训目录缺少历史参数指纹，已采用当前配置建立基线: {config_path}",
                flush=True,
            )
        elif saved != current:
            changed = sorted(set(saved) | set(current))
            changed = [key for key in changed if saved.get(key) != current.get(key)]
            raise ValueError(
                "续训参数与原实验不一致: "
                + ", ".join(changed)
                + "；请恢复原参数或启用 fresh"
            )
        return True, False

    _write_config(config_path, current)
    # run_config.json makes a new output directory non-empty. overwrite=True only
    # grants LightlyTrain permission to use it; fresh archiving handled above.
    return False, True
