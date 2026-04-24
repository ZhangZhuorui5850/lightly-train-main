"""脚本执行器。

这个文件的职责很单一：
- 接收一个现成脚本路径
- 检查脚本是否存在
- 用 runpy 方式按 __main__ 执行

目前主要给 launcher.py 菜单里的 train / 部分 eval 使用，
这样可以直接复用已有的 train_cls.py、train_det.py、train_seg.py、test_cls.py 等脚本。
"""

from __future__ import annotations

import runpy
from pathlib import Path

from . import common as rt


def _display_script_path(script_path: Path) -> str:
    try:
        return str(script_path.relative_to(rt.ROOT_DIR))
    except ValueError:
        return script_path.name


def _collect_experiment_mtimes() -> dict[Path, float]:
    if not rt.EXPERIMENT_ROOT_DIR.exists():
        return {}
    mtimes: dict[Path, float] = {}
    for path in rt.EXPERIMENT_ROOT_DIR.rglob("*"):
        if not rt.is_experiment_dir(path):
            continue
        resolved_path = path.resolve()
        mtimes[resolved_path] = resolved_path.stat().st_mtime
    return mtimes


def _sync_changed_experiment_archives(before: dict[Path, float], after: dict[Path, float]) -> None:
    changed_dirs = sorted(
        (
            path
            for path, mtime in after.items()
            if mtime > before.get(path, float("-inf"))
        ),
        key=lambda path: after[path],
        reverse=True,
    )
    for experiment_dir in changed_dirs:
        important_dir, archive_dir, dashboard_path = rt.sync_training_summary_artifacts(experiment_dir)
        print(f"summary important synced to: {important_dir}")
        print(f"summary archive synced to: {archive_dir}")
        if dashboard_path is not None:
            print(f"training_dashboard copied to: {dashboard_path}")


def run_script(script_path: Path) -> None:
    script_path = script_path.expanduser().resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Script does not exist: {script_path}")
    print(f"Running script: {_display_script_path(script_path)}")
    before = _collect_experiment_mtimes()
    runpy.run_path(str(script_path), run_name="__main__")
    after = _collect_experiment_mtimes()
    _sync_changed_experiment_archives(before, after)
