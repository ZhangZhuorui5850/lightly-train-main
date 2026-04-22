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
        "--dry-run",
        action="store_true",
        help="仅模拟整理阶段，不实际写文件",
    )
    return parser.parse_args(argv)


def run_conversion(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    label_format: str = "auto",
    seed: int | None = None,
    dry_run: bool = False,
) -> None:
    sync_module = load_module("sync_picture_module", SYNC_PATH)
    convert_module = load_module("labelme_to_yolo_module", CONVERT_PATH)

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

    print("=" * 60)
    print("  一键转换开始")
    print("=" * 60)
    print(f"SOURCES      : {[str(p.resolve()) for p in src_roots]}")
    print(f"LABEL_FORMAT : {label_format}")
    print(f"OUTPUT_ROOT  : {output_root.resolve()}")
    print(f"DET_ROOT     : {(output_root / 'dataset_det').resolve()}")
    print(f"CLS_ROOT     : {(output_root / 'dataset_cls').resolve()}")
    print(f"SEG_ROOT     : {(output_root / 'dataset_seg').resolve()}")

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
                "--source-format",
                label_format,
            ]
            convert_module.main()
        finally:
            sys.argv = original_argv
            convert_module.EXISTING_OUTPUT_POLICY = original_policy


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_conversion(
        args.sources,
        args.output_root,
        label_format=args.label_format,
        seed=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
