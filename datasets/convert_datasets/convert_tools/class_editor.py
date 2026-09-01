#!/usr/bin/env python3
"""统一类别编辑器：支持 PNG 语义分割、YOLO 检测和 YOLO 实例分割。

工具自动识别数据格式，共用类别删除、合并、重命名和连续重排 ID 的操作计划，
再按标注格式重写 PNG mask 或 YOLO TXT。
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from . import semantic_class_editor, yolo_class_editor
    from .dataset_discovery import (
        KIND_LABELS,
        auto_datasets_root,
        load_yaml,
    )
    from .dataset_detector import detect_datasets, inspect_dataset
except ImportError:
    import semantic_class_editor  # type: ignore[no-redef]
    import yolo_class_editor  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        KIND_LABELS,
        auto_datasets_root,
        load_yaml,
    )
    from dataset_detector import (  # type: ignore[no-redef]
        detect_datasets,
        inspect_dataset,
    )


SUPPORTED_KINDS = {"semantic_mask", "yolo_detection", "yolo_instance"}
FORMAT_TO_KIND = {
    "semantic": "semantic_mask",
    "yolo-detect": "yolo_detection",
    "yolo-seg": "yolo_instance",
}


def _config_path(source: Path) -> Path:
    return semantic_class_editor._find_config(source)


def detect_kind(source: Path, forced_format: str = "auto") -> str:
    if forced_format != "auto":
        return FORMAT_TO_KIND[forced_format]
    config_path = _config_path(source)
    candidate = inspect_dataset(config_path, count_limit=1)
    if candidate.kind in SUPPORTED_KINDS:
        return candidate.kind
    config = load_yaml(config_path)
    task = str(config.get("task", "")).casefold()
    if task in {"semantic_segmentation", "semantic"}:
        return "semantic_mask"
    if task in {"segment", "seg", "instance_segmentation"}:
        return "yolo_instance"
    if task in {"detect", "detection", "object_detection"}:
        return "yolo_detection"
    raise ValueError(
        "无法自动识别标注格式；请指定 --format semantic、yolo-detect 或 yolo-seg"
    )


def choose_source(search_root: Path) -> tuple[Path, str] | None:
    candidates = [
        candidate
        for candidate in detect_datasets(search_root, kinds=SUPPORTED_KINDS)
        if candidate.kind in SUPPORTED_KINDS
    ]
    print(f"\n在 {search_root} 中检测到 {len(candidates)} 个可编辑数据集:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. {KIND_LABELS.get(candidate.kind, candidate.kind):<18}"
            f"类别 {candidate.class_count:>4}，标注 {candidate.annotation_count:>7}  "
            f"{candidate.path}"
        )
    print("   p. 输入任意 data.yaml 或数据集路径")
    while True:
        try:
            raw = input("选择数据集 [序号/p/q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        if raw.casefold() == "p":
            value = input("data.yaml 或数据集目录: ").strip()
            if not value:
                continue
            path = Path(value).expanduser().resolve()
            return path, detect_kind(path)
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            candidate = candidates[int(raw) - 1]
            return candidate.config_path or candidate.path, candidate.kind
        print("请输入有效序号、p 或 q。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="源 data.yaml 或数据集目录")
    parser.add_argument("--out", type=Path, help="输出目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描目录，默认仓库 datasets/")
    parser.add_argument("--plan", type=Path, help="读取已有类别操作计划 YAML")
    parser.add_argument(
        "--id-policy",
        choices=("compact", "preserve", "explicit"),
        help="覆盖计划中的类别 ID 策略；explicit 的映射取自计划 explicit_ids",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "semantic", "yolo-detect", "yolo-seg"),
        default="auto",
        help="标注格式，默认根据目录和 data.yaml 自动识别",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=255,
        help="语义分割删除类别使用的 ignore 值",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink", "reflink"),
        default="copy",
    )
    parser.add_argument("--clean", action="store_true", help="清理已有输出后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只分析和生成操作计划")
    parser.add_argument("--yes", action="store_true", help="跳过最终交互确认")
    parser.add_argument("--require-train-val", action="store_true")
    return parser.parse_args(argv)


def _append_common_args(args: argparse.Namespace, child_args: list[str]) -> None:
    if args.out is not None:
        child_args.extend(["--out", str(args.out)])
    if args.plan is not None:
        child_args.extend(["--plan", str(args.plan)])
    if getattr(args, "id_policy", None) is not None:
        child_args.extend(["--id-policy", args.id_policy])
    child_args.extend(["--image-mode", args.image_mode])
    if args.clean:
        child_args.append("--clean")
    if args.dry_run:
        child_args.append("--dry-run")
    if args.yes:
        child_args.append("--yes")
    if args.require_train_val:
        child_args.append("--require-train-val")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.src is None:
        selected = choose_source((args.datasets or auto_datasets_root()).resolve())
        if selected is None:
            return 0
        source, detected_kind = selected
        kind = FORMAT_TO_KIND.get(args.format, detected_kind)
    else:
        source = args.src.expanduser().resolve()
        kind = detect_kind(source, args.format)

    child_args = ["--src", str(source)]
    _append_common_args(args, child_args)
    if kind == "semantic_mask":
        child_args.extend(["--ignore-label", str(args.ignore_label)])
        return semantic_class_editor.main(child_args)
    if kind == "yolo_detection":
        child_args.extend(["--format", "yolo-detect"])
        return yolo_class_editor.main(child_args)
    if kind == "yolo_instance":
        child_args.extend(["--format", "yolo-seg"])
        return yolo_class_editor.main(child_args)
    raise ValueError(f"暂不支持的数据格式: {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
