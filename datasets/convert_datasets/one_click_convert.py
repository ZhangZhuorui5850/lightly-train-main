#!/usr/bin/env python3
"""
一键整理并转换数据集：
1. 先把原始数据整理成 train/val/test
2. 再输出统一的 dataset_det / dataset_cls / dataset_seg

示例:
    python one_click_convert.py data_a data_b -o converted_all
    python one_click_convert.py data_a -o converted_all --label-format yolo --seed 42
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SYNC_PATH = THIS_DIR / "sync_picture.py"
CONVERT_PATH = THIS_DIR / "LabelMeToYOLO.py"


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def copy_yolo_metadata(src_roots: list[Path], synced_root: Path) -> None:
    candidates = ["data.yaml", "dataset.yaml", "classes.txt"]
    for src_root in src_roots:
        for name in candidates:
            src = src_root / name
            if src.exists() and src.is_file():
                shutil.copy2(src, synced_root / name)
                return


def detect_pre_split(src_roots: list[Path]) -> bool:
    """每个源目录都至少含 2 个 {train, val, test} 子目录且非空时，视为已预划分。"""
    for root in src_roots:
        present = 0
        for split in ("train", "val", "test"):
            d = root / split
            if d.is_dir() and any(d.iterdir()):
                present += 1
        if present < 2:
            return False
    return True


def merge_pre_split_sources(
    src_roots: list[Path],
    synced_root: Path,
    label_format: str,
) -> dict:
    """把多个已预划分的源目录按 train/val/test 合并到 synced_root（硬链接优先）。"""
    import os

    label_suffix = ".json" if label_format == "labelme" else ".txt"
    stats = {"split_counts": {"train": 0, "val": 0, "test": 0}, "skipped_conflicts": 0}

    for split in ("train", "val", "test"):
        (synced_root / split).mkdir(parents=True, exist_ok=True)

    used_per_split: dict = {"train": set(), "val": set(), "test": set()}

    for root in src_roots:
        for split in ("train", "val", "test"):
            src_dir = root / split
            if not src_dir.is_dir():
                continue
            for path in src_dir.iterdir():
                if not path.is_file():
                    continue
                name = path.name
                if name in used_per_split[split]:
                    stats["skipped_conflicts"] += 1
                    continue
                dst = synced_root / split / name
                try:
                    os.link(path, dst)
                except OSError:
                    shutil.copy2(path, dst)
                used_per_split[split].add(name)
                if path.suffix.lower() == label_suffix or path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    pass
            stats["split_counts"][split] = len(
                [p for p in (synced_root / split).iterdir() if p.suffix.lower() == label_suffix]
            )

    stats["paired_total"] = sum(stats["split_counts"].values())
    return stats


def normalize_selected_tasks(task: str) -> tuple[str, ...]:
    task_key = task.strip().lower()
    if task_key == "all":
        return ("det", "cls", "seg")
    if task_key in {"det", "cls", "seg"}:
        return (task_key,)
    raise ValueError(f"不支持的任务类型: {task}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键整理原始数据并输出统一的 dataset_det / dataset_cls / dataset_seg",
    )
    parser.add_argument("sources", nargs="+", help="原始源目录，可多个")
    parser.add_argument(
        "-o",
        "--output-root",
        required=True,
        help="总输出目录，内部会生成 dataset_det、dataset_cls、dataset_seg",
    )
    parser.add_argument(
        "--task",
        choices=["det", "cls", "seg", "all"],
        default="all",
        help="输出任务类型：det/cls/seg/all",
    )
    parser.add_argument(
        "--label-format",
        choices=["auto", "labelme", "yolo"],
        default="auto",
        help="输入标注格式：auto 自动判断，labelme 为 .json，yolo 为 .txt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="划分 train/val/test 的随机种子",
    )
    parser.add_argument(
        "--preserve-splits",
        choices=["auto", "yes", "no"],
        default="auto",
        help=(
            "源目录已含 train/val/test 时如何处理："
            "auto=检测到则跳过 sync 直接用现有划分（默认），"
            "yes=强制跳过 sync（要求已预划分），"
            "no=始终走 sync 重新 8:1:1 划分"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅模拟整理阶段，不实际写文件",
    )
    return parser.parse_args(argv)


def run_conversion(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    task: str = "all",
    label_format: str = "auto",
    seed: int | None = None,
    dry_run: bool = False,
    preserve_splits: str = "auto",
) -> None:
    sync_module = load_module("sync_picture_module", SYNC_PATH)
    convert_module = load_module("labelme_to_yolo_module", CONVERT_PATH)
    selected_tasks = normalize_selected_tasks(task)

    src_roots = [Path(s).expanduser().resolve() for s in sources]
    for src in src_roots:
        if not src.is_dir():
            print(f"[ERROR] 源目录不存在: {src.resolve()}")
            sys.exit(1)

    output_root = Path(output_root).expanduser().resolve()

    if label_format == "auto":
        label_format = sync_module.detect_label_format(src_roots)
        if label_format == "mixed":
            print("[ERROR] 自动检测到源目录同时包含 JSON 和 TXT，请显式指定 --label-format")
            sys.exit(1)
        if label_format == "unknown":
            print("[ERROR] 未检测到可用标注文件（.json 或 .txt）")
            sys.exit(1)

    is_pre_split = detect_pre_split(src_roots)
    if preserve_splits == "yes" and not is_pre_split:
        print("[ERROR] --preserve-splits=yes 但源目录并未含 train/val/test 子目录")
        sys.exit(1)
    use_preserve = preserve_splits == "yes" or (preserve_splits == "auto" and is_pre_split)

    print("=" * 60)
    print("  一键转换开始")
    print("=" * 60)
    print(f"SOURCES      : {[str(p.resolve()) for p in src_roots]}")
    print(f"TASKS        : {', '.join(selected_tasks)}")
    print(f"LABEL_FORMAT : {label_format}")
    print(f"OUTPUT_ROOT  : {output_root.resolve()}")
    print(f"PRESERVE_SPL : {use_preserve} (mode={preserve_splits}, detected={is_pre_split})")
    if "det" in selected_tasks:
        print(f"DET_ROOT     : {(output_root / 'dataset_det').resolve()}")
    if "cls" in selected_tasks:
        print(f"CLS_ROOT     : {(output_root / 'dataset_cls').resolve()}")
    if "seg" in selected_tasks:
        print(f"SEG_ROOT     : {(output_root / 'dataset_seg').resolve()}")

    def _invoke_labelme_to_yolo(synced_root: Path) -> None:
        original_policy = convert_module.EXISTING_OUTPUT_POLICY
        convert_module.EXISTING_OUTPUT_POLICY = "clean"
        original_argv = sys.argv[:]
        try:
            sys.argv = [
                str(CONVERT_PATH),
                "--source-root",
                str(synced_root),
                "--output-root",
                str(output_root),
                "--task",
                task,
                "--source-format",
                label_format,
            ]
            convert_module.main()
        finally:
            sys.argv = original_argv
            convert_module.EXISTING_OUTPUT_POLICY = original_policy

    # ── 分支 A：已预划分，跳过 sync 的随机洗牌 ─────────────────────────────
    if use_preserve:
        if dry_run:
            print("\n[INFO] 已检测到预划分，跳过 sync。dry-run 结束。")
            return
        if len(src_roots) == 1:
            synced_root = src_roots[0]
            print(f"[INFO] 单源已预划分，直接使用: {synced_root}")
            _invoke_labelme_to_yolo(synced_root)
            return
        # 多源：合并到临时目录，保留各源的 split 归属
        with tempfile.TemporaryDirectory(prefix="dataset_sync_") as temp_dir:
            synced_root = Path(temp_dir) / "synced_source"
            print(f"[INFO] 多源已预划分，合并到: {synced_root}")
            stats = merge_pre_split_sources(src_roots, synced_root, label_format)
            print(f"  合并完成: train={stats['split_counts']['train']} "
                  f"val={stats['split_counts']['val']} test={stats['split_counts']['test']} "
                  f"冲突跳过={stats['skipped_conflicts']}")
            if label_format == "yolo":
                copy_yolo_metadata(src_roots, synced_root)
            if stats["paired_total"] == 0:
                print("\n[ERROR] 合并阶段没有得到任何文件，终止后续转换。")
                sys.exit(1)
            _invoke_labelme_to_yolo(synced_root)
            return

    # ── 分支 B：走原 sync 流程，重新 8:1:1 随机划分 ─────────────────────────
    with tempfile.TemporaryDirectory(prefix="dataset_sync_") as temp_dir:
        synced_root = Path(temp_dir) / "synced_source"
        print(f"TEMP_SYNCED  : {synced_root}")

        stats = sync_module.copy_files(
            src_roots,
            synced_root,
            label_format=label_format,
            seed=seed,
            dry_run=dry_run,
        )

        if label_format == "yolo" and not dry_run:
            copy_yolo_metadata(src_roots, synced_root)

        if dry_run:
            print("\n[INFO] dry-run 模式已结束，未执行后续转换。")
            return

        if stats["paired_total"] == 0:
            print("\n[ERROR] 整理阶段没有得到任何有效配对，终止后续转换。")
            sys.exit(1)

        _invoke_labelme_to_yolo(synced_root)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_conversion(
        args.sources,
        args.output_root,
        task=args.task,
        label_format=args.label_format,
        seed=args.seed,
        dry_run=args.dry_run,
        preserve_splits=args.preserve_splits,
    )


if __name__ == "__main__":
    main()
