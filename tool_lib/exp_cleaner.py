"""实验清理工具。

扫描 out/ 下所有实验目录，分析文件组成，标记可清理项，
支持 dry-run 预览和确认后删除。清理日志写入 out/.clean_log。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import common as rt


# ============================================================
# 数据结构
# ============================================================


@dataclass
class CleanCandidate:
    """一个可清理的文件或目录。"""

    path: Path
    category: str  # checkpoints / exported_last / infer_images / temp
    size_bytes: int
    description: str


@dataclass
class PreservedItem:
    """一个保留的文件或目录。"""

    path: Path
    category: str  # exported_best / important / infer_report
    size_bytes: int
    description: str


@dataclass
class ExperimentAnalysis:
    """单个实验的分析结果。"""

    exp_dir: Path
    total_size: int
    cleanable: list[CleanCandidate] = field(default_factory=list)
    preserved: list[PreservedItem] = field(default_factory=list)

    @property
    def cleanable_size(self) -> int:
        return sum(item.size_bytes for item in self.cleanable)

    @property
    def preserved_size(self) -> int:
        return sum(item.size_bytes for item in self.preserved)


@dataclass
class CleanLogEntry:
    """一条删除日志。"""

    timestamp: str
    exp_dir: str
    deleted_path: str
    category: str
    size_bytes: int


# ============================================================
# 工具函数
# ============================================================


def format_size(size_bytes: int) -> str:
    """人类可读的大小格式化。"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _dir_size(path: Path) -> int:
    """计算目录总大小（字节）。"""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _file_size(path: Path) -> int:
    """获取文件大小，不存在返回 0。"""
    if path.exists() and path.is_file():
        return path.stat().st_size
    return 0


def _exp_mtime(exp_dir: Path) -> float:
    """取实验目录的最后修改时间。"""
    try:
        return exp_dir.stat().st_mtime
    except OSError:
        return 0


# ============================================================
# 核心分析
# ============================================================


def analyze_experiment(exp_dir: Path) -> ExperimentAnalysis:
    """分析单个实验目录，标记可清理项和保留项。"""
    cleanable: list[CleanCandidate] = []
    preserved: list[PreservedItem] = []

    # --- 保留项 ---

    exported_best = exp_dir / "exported_models" / "exported_best.pt"
    if exported_best.exists():
        preserved.append(
            PreservedItem(
                path=exported_best,
                category="exported_best",
                size_bytes=_file_size(exported_best),
                description="最佳导出模型",
            )
        )

    important_dir = exp_dir / "important"
    if important_dir.is_dir():
        preserved.append(
            PreservedItem(
                path=important_dir,
                category="important",
                size_bytes=_dir_size(important_dir),
                description="训练日志和 dashboard",
            )
        )

    # infer 报告文件（json/md，不含 images/）
    for infer_dir in exp_dir.glob("infer-*"):
        if not infer_dir.is_dir():
            continue
        report_size = 0
        for f in infer_dir.iterdir():
            if f.is_file() and f.suffix in {".json", ".md"}:
                report_size += _file_size(f)
        if report_size > 0:
            preserved.append(
                PreservedItem(
                    path=infer_dir,
                    category="infer_report",
                    size_bytes=report_size,
                    description=f"{infer_dir.name}/报告文件",
                )
            )

    # --- 可清理项 ---

    # 整个 checkpoints/ 目录
    checkpoints_dir = exp_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        cleanable.append(
            CleanCandidate(
                path=checkpoints_dir,
                category="checkpoints",
                size_bytes=_dir_size(checkpoints_dir),
                description="checkpoints/（不再续训）",
            )
        )

    # exported_models/exported_last.pt
    exported_last = exp_dir / "exported_models" / "exported_last.pt"
    if exported_last.exists():
        cleanable.append(
            CleanCandidate(
                path=exported_last,
                category="exported_last",
                size_bytes=_file_size(exported_last),
                description="exported_models/exported_last.pt",
            )
        )

    # 各推理/评估目录下的 images/ 与 compare/（可重新生成，统一视为可清理）。
    # 覆盖 det 的 infer-*，以及 seg 的 infer / eval。
    vis_parent_dirs = list(exp_dir.glob("infer-*")) + [exp_dir / "infer", exp_dir / "eval"]
    for vis_dir in vis_parent_dirs:
        if not vis_dir.is_dir():
            continue
        for sub in ("images", "compare"):
            target = vis_dir / sub
            if target.is_dir():
                cleanable.append(
                    CleanCandidate(
                        path=target,
                        category="infer_images",
                        size_bytes=_dir_size(target),
                        description=f"{vis_dir.name}/{sub}/",
                    )
                )

    total_size = sum(item.size_bytes for item in cleanable) + sum(
        item.size_bytes for item in preserved
    )

    return ExperimentAnalysis(
        exp_dir=exp_dir,
        total_size=total_size,
        cleanable=cleanable,
        preserved=preserved,
    )


def scan_experiments(root_dir: Path | None = None) -> list[ExperimentAnalysis]:
    """扫描所有实验目录，返回分析结果列表（按修改时间倒序）。"""
    if root_dir is None:
        root_dir = rt.EXPERIMENT_ROOT_DIR
    if not root_dir.exists():
        return []

    from .file_index import walk_tree
    from .progress import track

    experiment_dirs: list[Path] = []
    for path, dirnames, _filenames in walk_tree(
        root_dir,
        label="索引待清理实验",
        followlinks=True,
    ):
        if path == root_dir.resolve():
            continue
        if rt.is_experiment_dir(path):
            experiment_dirs.append(path.resolve())
            dirnames[:] = []

    analyses = [
        analyze_experiment(path)
        for path in track(
            experiment_dirs,
            label="分析实验占用",
            total=len(experiment_dirs),
            unit="exp",
        )
    ]

    analyses.sort(key=lambda a: _exp_mtime(a.exp_dir), reverse=True)
    return analyses


# ============================================================
# 报告与执行
# ============================================================


def compact_display(path: Path) -> str:
    """简洁显示路径（从 out/ 开始）。"""
    try:
        parts = path.resolve().parts
        if "out" in parts:
            idx = parts.index("out")
            tail = parts[idx + 1 :]
            if tail:
                return str(Path(*tail))
        return str(path.relative_to(rt.ROOT_DIR))
    except ValueError:
        return str(path.name)


def print_clean_report(analyses: list[ExperimentAnalysis]) -> None:
    """打印清理报告。"""
    if not analyses:
        print("未扫描到任何实验目录。")
        return

    total_size = sum(a.total_size for a in analyses)
    cleanable_size = sum(a.cleanable_size for a in analyses)

    print()
    print("=" * 70)
    print("  实验清理报告")
    print("=" * 70)
    print()
    print(f"扫描到 {len(analyses)} 个实验目录，总计占用 {format_size(total_size)}")
    print()

    for idx, analysis in enumerate(analyses, start=1):
        name = compact_display(analysis.exp_dir)
        mtime = datetime.fromtimestamp(_exp_mtime(analysis.exp_dir)).strftime("%Y-%m-%d")
        print(f" [{idx}] {name}  （总大小 {format_size(analysis.total_size)} | 可清理 {format_size(analysis.cleanable_size)} | {mtime}）")

        # 显示可清理项
        if analysis.cleanable:
            for item in analysis.cleanable:
                print(f"      ❌ {item.description:<36} {format_size(item.size_bytes):>10}")

        # 显示保留项
        if analysis.preserved:
            for item in analysis.preserved:
                print(f"      ✅ {item.description:<36} {format_size(item.size_bytes):>10}")

        print()

    print(f"总计可释放: {format_size(cleanable_size)}")
    print()


def print_experiment_detail(analysis: ExperimentAnalysis) -> None:
    """打印单个实验的详细信息。"""
    name = compact_display(analysis.exp_dir)
    mtime = datetime.fromtimestamp(_exp_mtime(analysis.exp_dir)).strftime("%Y-%m-%d %H:%M")

    print()
    print(f"实验: {name}")
    print(f"总大小: {format_size(analysis.total_size)} | 最后修改: {mtime}")
    print()

    if analysis.preserved:
        print("  ✅ 保留:")
        for item in analysis.preserved:
            print(f"    {item.description:<40} {format_size(item.size_bytes):>10}")
        print()

    if analysis.cleanable:
        print(f"  ❌ 可清理（共 {format_size(analysis.cleanable_size)}）:")
        for item in analysis.cleanable:
            print(f"    {item.description:<40} {format_size(item.size_bytes):>10}")
        print()
        print(f"  执行清理可释放: {format_size(analysis.cleanable_size)}")
    else:
        print("  没有可清理的内容。")
    print()


def print_clean_preview(analyses: list[ExperimentAnalysis]) -> None:
    """打印即将清理的实验预览。"""
    print()
    print("=" * 70)
    print("  即将清理以下实验")
    print("=" * 70)
    print()

    total_cleanable = 0
    for idx, analysis in enumerate(analyses, start=1):
        name = compact_display(analysis.exp_dir)
        clean_size = analysis.cleanable_size
        total_cleanable += clean_size
        print(f"[{idx}] {name}  （可释放 {format_size(clean_size)}）")
        for item in analysis.cleanable:
            print(f"    ├── {item.description:<36} {format_size(item.size_bytes):>10}")
        print()

    print(f"总计释放: {format_size(total_cleanable)}")
    print()


def execute_clean(
    analyses: list[ExperimentAnalysis],
    *,
    dry_run: bool = False,
) -> list[CleanLogEntry]:
    """执行清理，返回删除日志条目。"""
    entries: list[CleanLogEntry] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for analysis in analyses:
        for item in analysis.cleanable:
            if dry_run:
                print(f"  [dry-run] 将删除: {compact_display(item.path)}  ({format_size(item.size_bytes)})")
                entries.append(
                    CleanLogEntry(
                        timestamp=timestamp,
                        exp_dir=str(analysis.exp_dir),
                        deleted_path=str(item.path),
                        category=item.category,
                        size_bytes=item.size_bytes,
                    )
                )
                continue

            if not item.path.exists():
                continue

            try:
                if item.path.is_dir():
                    size = _dir_size(item.path)
                    shutil.rmtree(item.path)
                else:
                    size = _file_size(item.path)
                    item.path.unlink()

                print(f"  ✓ {compact_display(item.path)}  ({format_size(size)})")
                entries.append(
                    CleanLogEntry(
                        timestamp=timestamp,
                        exp_dir=str(analysis.exp_dir),
                        deleted_path=str(item.path),
                        category=item.category,
                        size_bytes=size,
                    )
                )
            except OSError as e:
                print(f"  ✗ 删除失败: {compact_display(item.path)}  ({e})")

    return entries


def write_clean_log(entries: list[CleanLogEntry], log_path: Path | None = None) -> None:
    """写删除日志。"""
    if not entries:
        return
    if log_path is None:
        log_path = rt.OUT_DIR / ".clean_log"

    with open(log_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(
                f"[{entry.timestamp}] {entry.category}: {entry.deleted_path} "
                f"({format_size(entry.size_bytes)}) <- {entry.exp_dir}\n"
            )
    print(f"删除日志已写入: {compact_display(log_path)}")


def run_clean(args) -> None:
    """实验清理主入口（供 dispatch 调用）。"""
    analyses = list(getattr(args, "analyses", []) or [])
    if not analyses:
        analyses = scan_experiments()
        if not analyses:
            print("未扫描到任何实验目录，无需清理。")
            return
        print_clean_report(analyses)
        print_clean_preview(analyses)
    dry_run = bool(getattr(args, "dry_run", False))
    if not dry_run and not bool(getattr(args, "yes", True)):
        raise ValueError("执行清理需要明确确认。")
    print("\n正在预览清理..." if dry_run else "\n正在清理...")
    entries = execute_clean(analyses, dry_run=dry_run)
    if entries and not dry_run:
        write_clean_log(entries)
    released = sum(entry.size_bytes for entry in entries)
    action = "预计释放" if dry_run else "释放空间"
    print(f"\n清理完成！{action}: {format_size(released)}")
