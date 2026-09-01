#!/usr/bin/env python3
"""统一数据集合并入口：支持 YOLO Det、PNG Semantic 和 YOLO 实例分割。

交互运行时先选择标注类型，再多选来源数据集并规划全部类别操作。
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import semantic_dataset_merger, yolo_dataset_merger
    from .dataset_detector import inspect_dataset
except ImportError:
    import semantic_dataset_merger  # type: ignore[no-redef]
    import yolo_dataset_merger  # type: ignore[no-redef]
    from dataset_detector import inspect_dataset  # type: ignore[no-redef]


FORMAT_LABELS = {
    "yolo-detect": "YOLO 目标检测（labels/*.txt，每行 bbox）",
    "semantic": "PNG 语义分割（masks/*.png，类似 dataset_semantic）",
    "yolo-seg": "YOLO 实例分割（labels/*.txt，每行 polygon）",
}


def choose_format() -> str | None:
    print("\n第 1 步 · 选择要合并的数据集格式:\n")
    order = ["yolo-detect", "semantic", "yolo-seg"]
    for index, key in enumerate(order, start=1):
        print(f"  {index}. {FORMAT_LABELS[key]}")
    while True:
        try:
            raw = input("选择格式 [1/2/3/q]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(order):
            return order[int(raw) - 1]
        print("请输入 1、2、3 或 q。")


def detect_format(source: Path) -> str:
    source = source.expanduser().resolve()
    candidate = inspect_dataset(source, count_limit=1)
    return {
        "yolo_detection": "yolo-detect",
        "semantic_mask": "semantic",
        "yolo_instance": "yolo-seg",
    }.get(candidate.kind, "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        action="append",
        help="输入 data.yaml 或数据集目录；至少重复两次",
    )
    parser.add_argument("--out", type=Path, help="最终合并数据集目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描目录，默认仓库 datasets/")
    parser.add_argument("--plan", type=Path, help="读取已有完整类别操作计划 YAML")
    parser.add_argument(
        "--id-policy",
        choices=("compact", "preserve", "explicit"),
        help="覆盖计划中的类别 ID 策略；explicit 的映射取自计划 explicit_ids",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "yolo-detect", "semantic", "yolo-seg"),
        default="auto",
    )
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink", "reflink"),
        default="copy",
    )
    parser.add_argument(
        "--taxonomy",
        choices=("namespace", "strict", "union-by-name", "mapping-file"),
        default="namespace",
        help="跨来源类别策略",
    )
    parser.add_argument("--mapping-file", type=Path, help="taxonomy=mapping-file 的映射 YAML")
    parser.add_argument(
        "--reference-source",
        type=int,
        metavar="N",
        help="以第 N 个 --src 的 data.yaml 作为类别 ID 与名称基准",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("keep", "error", "hash-dedupe", "drop", "merge-annotations"),
        default="keep",
    )
    parser.add_argument(
        "--duplicate-annotation-conflict-policy",
        choices=("error", "keep-first", "keep-last", "keep-both"),
        default="error",
        help="YOLO merge-annotations 的冲突处理策略",
    )
    parser.add_argument(
        "--duplicate-annotation-conflict-iou",
        type=float,
        default=0.95,
        help="YOLO merge-annotations 的几何冲突 IoU 阈值",
    )
    parser.add_argument(
        "--split-leakage-policy",
        choices=("warn", "error", "drop"),
        default="warn",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--source-ignore-label",
        action="append",
        default=[],
        metavar="SOURCE=ID|none",
        help="Semantic 来源级输入 ignore 值，可重复使用",
    )
    parser.add_argument(
        "--all-ignore-policy",
        choices=("keep", "drop", "quarantine"),
        default="keep",
    )
    parser.add_argument("--require-train-val", action="store_true")
    parser.add_argument(
        "--empty-image-policy",
        choices=("keep", "drop", "quarantine"),
        default="keep",
    )
    parser.add_argument("--orphan-policy", choices=("error", "drop"), default="error")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="采用第一个来源及推荐类别方案，并跳过最终确认",
    )
    parser.add_argument("--clean", action="store_true", help="清理已有输出后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只分析并输出合并计划")
    return parser.parse_args(argv)


def _common_child_args(args: argparse.Namespace) -> list[str]:
    child_args: list[str] = []
    for source in args.src or []:
        child_args.extend(["--src", str(source)])
    if args.out is not None:
        child_args.extend(["--out", str(args.out)])
    if args.datasets is not None:
        child_args.extend(["--datasets", str(args.datasets)])
    if args.plan is not None:
        child_args.extend(["--plan", str(args.plan)])
    if args.id_policy is not None:
        child_args.extend(["--id-policy", args.id_policy])
    child_args.extend(["--image-mode", args.image_mode])
    child_args.extend(["--taxonomy", args.taxonomy])
    if args.mapping_file is not None:
        child_args.extend(["--mapping-file", str(args.mapping_file)])
    if args.reference_source is not None:
        child_args.extend(["--reference-source", str(args.reference_source)])
    child_args.extend(["--duplicate-policy", args.duplicate_policy])
    child_args.extend(["--split-leakage-policy", args.split_leakage_policy])
    child_args.extend(["--workers", str(args.workers)])
    if args.yes:
        child_args.append("--yes")
    if args.clean:
        child_args.append("--clean")
    if args.dry_run:
        child_args.append("--dry-run")
    return child_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_format = args.format
    if selected_format == "auto":
        if args.src:
            selected_format = detect_format(args.src[0])
            if not selected_format:
                raise ValueError(
                    "无法自动识别输入格式；请指定 --format yolo-detect、semantic "
                    "或 yolo-seg"
                )
        else:
            choice = choose_format()
            if choice is None:
                return 0
            selected_format = choice

    child_args = _common_child_args(args)
    if selected_format == "semantic":
        child_args.extend(["--ignore-label", str(args.ignore_label)])
        for value in args.source_ignore_label:
            child_args.extend(["--source-ignore-label", value])
        child_args.extend(["--all-ignore-policy", args.all_ignore_policy])
        if args.require_train_val:
            child_args.append("--require-train-val")
        return semantic_dataset_merger.main(child_args)
    child_args.extend(["--empty-image-policy", args.empty_image_policy])
    child_args.extend(["--orphan-policy", args.orphan_policy])
    child_args.extend(
        [
            "--duplicate-annotation-conflict-policy",
            args.duplicate_annotation_conflict_policy,
            "--duplicate-annotation-conflict-iou",
            str(args.duplicate_annotation_conflict_iou),
        ]
    )
    if args.require_train_val:
        child_args.append("--require-train-val")
    child_args.extend(["--format", selected_format])
    return yolo_dataset_merger.main(child_args)


if __name__ == "__main__":
    raise SystemExit(main())
