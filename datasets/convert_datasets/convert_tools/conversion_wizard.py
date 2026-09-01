#!/usr/bin/env python3
"""Scan project datasets, explain available conversions, and dispatch one workflow."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path
from typing import Sequence

try:
    from .dataset_discovery import (
        KIND_LABELS,
        DatasetCandidate,
        auto_datasets_root,
        conversion_actions,
        format_candidate_count,
        format_modified_time,
    )
    from .dataset_detector import detect_datasets
    from .interactive_helpers import prompt_choice, prompt_path, run_python_tool
    from .output_naming import default_output_dir
except ImportError:
    from dataset_discovery import (  # type: ignore[no-redef]
        KIND_LABELS,
        DatasetCandidate,
        auto_datasets_root,
        conversion_actions,
        format_candidate_count,
        format_modified_time,
    )
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from interactive_helpers import (  # type: ignore[no-redef]
        prompt_choice,
        prompt_path,
        run_python_tool,
    )
    from output_naming import default_output_dir  # type: ignore[no-redef]


HERE = Path(__file__).resolve().parent
# Compatibility name for callers/tests; implementation comes from the shared detector.
scan_datasets = detect_datasets
ACTION_LABELS = {
    "to-semantic": "转成 train_seg.py 使用的 PNG 语义掩码",
    "edit-classes": "检查并编辑 PNG 语义/YOLO Det/YOLO Seg 数据集的类别",
    "to-mvtec": "转成 MVTec AD",
    "to-yolo": "整理并转成 YOLO det/cls/seg",
    "to-yolo-seg": "image + fg 掩码转成 YOLO Seg",
    "mvtec-to-yolo": "标准 MVTec AD 转成 YOLO Seg + Det",
}
ACTION_ORDER = (
    "edit-classes",
    "to-semantic",
    "to-mvtec",
    "to-yolo",
    "to-yolo-seg",
    "mvtec-to-yolo",
)
ACTION_SOURCE_LABELS = {
    "to-semantic": "输入格式：YOLO 实例分割 polygon txt",
    "edit-classes": "输入格式：data.yaml + PNG mask 或 YOLO TXT",
    "to-mvtec": "输入格式：YOLO 实例分割 polygon txt",
    "to-yolo": "输入格式：LabelMe 图片和 JSON 标注",
    "to-yolo-seg": "输入格式：image + fg 掩码",
    "mvtec-to-yolo": "输入格式：category/{test,ground_truth}",
}


def _roots(value: Path | Sequence[Path]) -> list[Path]:
    values = [value] if isinstance(value, (str, Path)) else list(value)
    roots: list[Path] = []
    for item in values:
        root = Path(item).expanduser().resolve()
        if root not in roots:
            roots.append(root)
    return roots


def _display_path(path: Path, roots: Path | Sequence[Path]) -> Path:
    for root in _roots(roots):
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return path


def print_candidates(
    candidates: list[DatasetCandidate],
    root: Path | Sequence[Path],
    *,
    selected_action: str | None = None,
) -> None:
    roots = _roots(root)
    print("\n已扫描: " + ", ".join(str(item) for item in roots))
    print(f"检索到 {len(candidates)} 个数据集或原始标注目录:\n")
    print(
        f"  {'#':>2}  {'类型':<18}{'图像':>8}{'标注':>8}{'类别':>8}  "
        f"{'最后修改':<16} 可执行转换  路径"
    )
    print("  " + "-" * 118)
    for index, item in enumerate(candidates, start=1):
        image_count = format_candidate_count(
            item.image_count, item.image_count_is_exact
        )
        annotation_count = format_candidate_count(
            item.annotation_count, item.annotation_count_is_exact
        )
        actions = conversion_actions(item)
        action_text = (
            ACTION_LABELS[selected_action]
            if selected_action is not None and selected_action in actions
            else "、".join(ACTION_LABELS[action] for action in actions)
        )
        if not action_text:
            action_text = "已是可用目标格式" if item.kind == "semantic_mask" else "当前仅识别"
        print(
            f"  {index:>2}  {KIND_LABELS.get(item.kind, item.kind):<18}"
            f"{image_count:>8}{annotation_count:>8}{item.class_count:>8}  "
            f"{format_modified_time(item.modified_time):<16} "
            f"{action_text}  {_display_path(item.path, roots)}"
        )


def dispatch(
    candidate: DatasetCandidate,
    action: str,
    output: Path,
    *,
    extra_args: list[str] | None = None,
    clean: bool = False,
    dry_run: bool = False,
) -> int:
    clean_args = ["--clean"] if clean else []
    dry_run_args = ["--dry-run"] if dry_run else []
    if action == "to-semantic":
        return run_python_tool(
            HERE / "yoloseg_to_semantic.py",
            ["--src", str(candidate.config_path or candidate.path), "--out", str(output), *clean_args, *dry_run_args],
        )
    if action == "edit-classes":
        return run_python_tool(
            HERE / "class_editor.py",
            ["--src", str(candidate.config_path or candidate.path), "--out", str(output), *clean_args, *dry_run_args],
        )
    if action == "to-mvtec":
        return run_python_tool(
            HERE / "yoloseg_to_mvtec.py",
            ["--src", str(candidate.config_path or candidate.path), "--out", str(output), *clean_args, *dry_run_args],
        )
    if action == "to-yolo":
        return run_python_tool(
            HERE / "one_click_convert.py",
            [str(candidate.path), "--output-root", str(output), *(extra_args or []), *clean_args, *dry_run_args],
        )
    if action == "to-yolo-seg":
        return run_python_tool(
            HERE / "generated2seg_interactive.py",
            ["--src", str(candidate.path), "--out", str(output), *clean_args, *dry_run_args],
        )
    if action == "mvtec-to-yolo":
        return run_python_tool(
            HERE / "mvtec_to_yolo.py",
            ["--src", str(candidate.path), "--out", str(output), "--task", "both", *clean_args, *dry_run_args],
        )
    raise ValueError(f"未知转换动作: {action}")


def interactive(root: Path | Sequence[Path], *, dry_run: bool = False) -> int:
    roots = _roots(root)
    print("\n第 1 步 · 选择要执行的功能:\n")
    for index, action in enumerate(ACTION_ORDER, start=1):
        print(f"  {index}. {ACTION_LABELS[action]}")
        print(f"     {ACTION_SOURCE_LABELS[action]}")
    action_index = prompt_choice(
        f"\n选择功能 [1-{len(ACTION_ORDER)}，q 退出]: ", len(ACTION_ORDER)
    )
    if action_index is None:
        return 0
    action = ACTION_ORDER[action_index]

    print(f"\n第 2 步 · 检测支持“{ACTION_LABELS[action]}”的数据集...")
    candidates = [
        candidate
        for candidate in scan_datasets(roots)
        if action in conversion_actions(candidate)
    ]
    if not candidates:
        print("以下目录中没有检测到支持该功能的数据集: " + ", ".join(map(str, roots)))
        return 1

    print(f"\n检测到 {len(candidates)} 个兼容数据集:\n")
    print(
        f"  {'#':>2}  {'类型':<18}{'图像':>8}{'标注':>8}{'类别':>8}  "
        f"{'最后修改':<16} 路径"
    )
    print("  " + "-" * 100)
    for index, item in enumerate(candidates, start=1):
        image_count = format_candidate_count(
            item.image_count, item.image_count_is_exact
        )
        annotation_count = format_candidate_count(
            item.annotation_count, item.annotation_count_is_exact
        )
        print(
            f"  {index:>2}  {KIND_LABELS.get(item.kind, item.kind):<18}"
            f"{image_count:>8}{annotation_count:>8}{item.class_count:>8}  "
            f"{format_modified_time(item.modified_time):<16} "
            f"{_display_path(item.path, roots)}"
        )
    selected = prompt_choice(
        f"\n选择数据集 [1-{len(candidates)}，q 退出]: ", len(candidates)
    )
    if selected is None:
        return 0
    candidate = candidates[selected]

    print("\n第 3 步 · 确认输出目录")
    output = prompt_path(
        "输出目录",
        default_output_dir(candidate.path, action),
    )
    if output is None:
        return 0
    clean = False
    if not dry_run and output.exists() and (not output.is_dir() or any(output.iterdir())):
        answer = input(f"输出已有内容，原子替换 {output}？[y/N]: ").strip().lower()
        if answer not in {"y", "yes", "是"}:
            print("已取消。")
            return 0
        clean = True
    print(f"\n开始执行: {ACTION_LABELS[action]}")
    extra_args: list[str] = []
    if action == "to-yolo":
        task_options = ["det", "cls", "seg", "all"]
        task_index = prompt_choice("输出任务 [1=det, 2=cls, 3=seg, 4=all]: ", len(task_options))
        if task_index is None:
            return 0
        task = task_options[task_index]
        format_options = ["auto", "labelme", "yolo"]
        format_index = prompt_choice("标注格式 [1=auto, 2=labelme, 3=yolo]: ", len(format_options))
        if format_index is None:
            return 0
        extra_args.extend(["--task", task, "--label-format", format_options[format_index]])
        if task in {"seg", "all"}:
            seg_options = ["auto", "instance", "semantic"]
            seg_index = prompt_choice(
                "分割类型 [1=auto, 2=instance, 3=semantic]: ", len(seg_options)
            )
            if seg_index is None:
                return 0
            extra_args.extend(["--seg-type", seg_options[seg_index]])
        preserve_options = ["auto", "yes", "no"]
        preserve_index = prompt_choice(
            "保留已有划分 [1=auto, 2=yes, 3=no]: ", len(preserve_options)
        )
        if preserve_index is None:
            return 0
        extra_args.extend(["--preserve-splits", preserve_options[preserve_index]])
        print(
            "命令预览:\n  "
            + shlex.join(
                ["python", str(HERE / "one_click_convert.py"), str(candidate.path),
                 "--output-root", str(output), *extra_args, *(["--dry-run"] if dry_run else [])]
            )
        )
    if extra_args:
        if clean:
            if dry_run:
                return dispatch(candidate, action, output, extra_args=extra_args, clean=True, dry_run=True)
            return dispatch(candidate, action, output, extra_args=extra_args, clean=True)
        if dry_run:
            return dispatch(candidate, action, output, extra_args=extra_args, dry_run=True)
        return dispatch(candidate, action, output, extra_args=extra_args)
    if clean:
        if dry_run:
            return dispatch(candidate, action, output, clean=True, dry_run=True)
        return dispatch(candidate, action, output, clean=True)
    if dry_run:
        return dispatch(candidate, action, output, dry_run=True)
    return dispatch(candidate, action, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="自动扫描项目数据集并选择可执行转换")
    parser.add_argument(
        "--datasets",
        type=Path,
        action="append",
        help="扫描根目录；可重复传入，默认仓库 datasets/",
    )
    parser.add_argument("--list", action="store_true", help="只显示扫描结果")
    parser.add_argument("--action", choices=ACTION_ORDER, help="配合 --list 筛选指定功能")
    parser.add_argument("--dry-run", action="store_true", help="把零写入预览参数传给选中的转换工具")
    args = parser.parse_args(argv)
    roots = _roots(args.datasets or [auto_datasets_root()])
    if args.list:
        candidates = scan_datasets(roots)
        if args.action:
            candidates = [
                candidate
                for candidate in candidates
                if args.action in conversion_actions(candidate)
            ]
        print_candidates(candidates, roots, selected_action=args.action)
        return 0
    return interactive(roots, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
