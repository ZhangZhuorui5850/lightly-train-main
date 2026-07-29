"""Small reusable primitives for conversion command-line wizards."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def prompt_choice(prompt: str, count: int, *, default: int | None = None) -> int | None:
    """Return a zero-based choice, or ``None`` when the user exits."""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw and default is not None:
            return default
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        print(f"请输入 1-{count} 中的序号，输入 q 退出。")


def prompt_path(prompt: str, default: Path) -> Path | None:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if raw.lower() in {"q", "quit", "exit"}:
        return None
    return Path(raw).expanduser().resolve() if raw else default.resolve()


def run_python_tool(script: Path, args: list[str]) -> int:
    return subprocess.call([sys.executable, str(script), *args])
