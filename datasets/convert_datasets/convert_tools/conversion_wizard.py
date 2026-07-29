#!/usr/bin/env python3
"""Scan project datasets, explain available conversions, and dispatch one workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .dataset_discovery import (
        KIND_LABELS,
        DatasetCandidate,
        auto_datasets_root,
        conversion_actions,
        scan_datasets,
    )
    from .interactive_helpers import prompt_choice, prompt_path, run_python_tool
    from .output_naming import default_output_dir
except ImportError:
    from dataset_discovery import (  # type: ignore[no-redef]
        KIND_LABELS,
        DatasetCandidate,
        auto_datasets_root,
        conversion_actions,
        scan_datasets,
    )
    from interactive_helpers import (  # type: ignore[no-redef]
        prompt_choice,
        prompt_path,
        run_python_tool,
    )
    from output_naming import default_output_dir  # type: ignore[no-redef]


HERE = Path(__file__).resolve().parent
ACTION_LABELS = {
    "to-semantic": "转成 train_seg.py 使用的 PNG 语义掩码",
    "to-mvtec": "转成 MVTec AD",
    "to-yolo": "整理并转成 YOLO det/cls/seg",
    "to-yolo-seg": "image + fg 掩码转成 YOLO Seg",
    "mvtec-to-yolo": "标准 MVTec AD 转成 YOLO Seg + Det",
}
ACTION_ORDER = (
    "to-semantic",
    "to-mvtec",
    "to-yolo",
    "to-yolo-seg",
    "mvtec-to-yolo",
)
ACTION_SOURCE_LABELS = {
    "to-semantic": "输入格式：YOLO 实例分割 polygon txt",
    "to-mvtec": "输入格式：YOLO 实例分割 polygon txt",
    "to-yolo": "输入格式：LabelMe 图片和 JSON 标注",
    "to-yolo-seg": "输入格式：image + fg 掩码",
    "mvtec-to-yolo": "输入格式：category/{test,ground_truth}",
}


def _display_path(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def print_candidates(
    candidates: list[DatasetCandidate],
    root: Path,
    *,
    selected_action: str | None = None,
) -> None:
    print(f"\n已扫描: {root}")
    print(f"检索到 {len(candidates)} 个数据集或原始标注目录:\n")
    print(f"  {'#':>2}  {'类型':<18}{'图像':>8}{'标注':>8}{'类别':>8}  可执行转换  路径")
    print("  " + "-" * 100)
    for index, item in enumerate(candidates, start=1):
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
            f"{item.image_count:>8}{item.annotation_count:>8}{item.class_count:>8}  "
            f"{action_text}  {_display_path(item.path, root)}"
        )


def dispatch(candidate: DatasetCandidate, action: str, output: Path) -> int:
    if action == "to-semantic":
        return run_python_tool(
            HERE / "yoloseg_to_semantic.py",
            ["--src", str(candidate.config_path or candidate.path), "--out", str(output)],
        )
    if action == "to-mvtec":
        return run_python_tool(
            HERE / "yoloseg_to_mvtec.py",
            ["--src", str(candidate.path), "--out", str(output)],
        )
    if action == "to-yolo":
        return run_python_tool(
            HERE / "one_click_convert.py",
            [str(candidate.path), "--output-root", str(output)],
        )
    if action == "to-yolo-seg":
        return run_python_tool(
            HERE / "generated2seg_interactive.py",
            ["--src", str(candidate.path), "--out", str(output)],
        )
    if action == "mvtec-to-yolo":
        return run_python_tool(
            HERE / "mvtec_to_yolo.py",
            ["--src", str(candidate.path), "--out", str(output), "--task", "both"],
        )
    raise ValueError(f"未知转换动作: {action}")


def interactive(root: Path) -> int:
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
        for candidate in scan_datasets(root)
        if action in conversion_actions(candidate)
    ]
    if not candidates:
        print(f"在 {root} 中没有检测到支持该功能的数据集。")
        return 1

    print(f"\n检测到 {len(candidates)} 个兼容数据集:\n")
    print(f"  {'#':>2}  {'类型':<18}{'图像':>8}{'标注':>8}{'类别':>8}  路径")
    print("  " + "-" * 82)
    for index, item in enumerate(candidates, start=1):
        print(
            f"  {index:>2}  {KIND_LABELS.get(item.kind, item.kind):<18}"
            f"{item.image_count:>8}{item.annotation_count:>8}{item.class_count:>8}  "
            f"{_display_path(item.path, root)}"
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
    print(f"\n开始执行: {ACTION_LABELS[action]}")
    return dispatch(candidate, action, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="自动扫描项目数据集并选择可执行转换")
    parser.add_argument("--datasets", type=Path, help="扫描根目录，默认仓库 datasets/")
    parser.add_argument("--list", action="store_true", help="只显示扫描结果")
    parser.add_argument("--action", choices=ACTION_ORDER, help="配合 --list 筛选指定功能")
    args = parser.parse_args(argv)
    root = (args.datasets or auto_datasets_root()).expanduser().resolve()
    if args.list:
        candidates = scan_datasets(root)
        if args.action:
            candidates = [
                candidate
                for candidate in candidates
                if args.action in conversion_actions(candidate)
            ]
        print_candidates(candidates, root, selected_action=args.action)
        return 0
    return interactive(root)


if __name__ == "__main__":
    raise SystemExit(main())
