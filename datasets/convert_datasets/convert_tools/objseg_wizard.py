#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物体版 MVTec 交互向导:把"生成标注预览 → 人工分图 → 转换"两步用文字引导串起来。

因为中间有"下载预览→本地按物体分图→上传服务器"的人工往返,这不是一个不间断的
脚本会话,而是引导式两步:第1步生成预览,第2步(分好图之后)再来跑转换。

用法:
    python objseg_wizard.py        # 交互:先选第1步还是第2步
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import objectseg_to_mvtec  # noqa: E402
import seg_sample_browse  # noqa: E402
from dataset_detector import choose_dataset  # noqa: E402
from dataset_discovery import auto_datasets_root  # noqa: E402
from output_naming import default_output_dir  # noqa: E402


def run_generate(
    src: Path,
    out: Path,
    limit: int = 500,
    font_path: Path | None = None,
    *,
    clean: bool = False,
) -> int:
    """第1步:生成标注预览,并打印下一步该做什么。"""
    browse_kwargs = {"limit": limit, "font_path": font_path}
    if clean:
        browse_kwargs["clean"] = True
    n = seg_sample_browse.browse(src, out, **browse_kwargs)
    print(
        f"\n[第1步完成] 生成了 {n} 张标注预览到:\n  {out}\n"
        "接下来(在你本地做):\n"
        f"  1) 把 {out} 下载到本地,逐张看图+信息栏,判断每张属于哪个物体;\n"
        "  2) 建【中文物体名】文件夹(如 盖板/ 紧固件/),把【预览图】直接分进去即可,\n"
        "     一张图只放一个物体;多物体/说不清的先别放。(脚本只认文件名,\n"
        "     真正的图和标注都会从源数据集按名字取原图,预览图的合成内容不会进 MVTec)\n"
        "  3) 把分好的 staging/ 上传回服务器;\n"
        "  4) 回来再跑本向导选【第2步】做转换。\n"
    )
    return n


def run_build(staging: Path, src: Path, out: Path, clean: bool = False) -> dict:
    """第2步:按分好的 staging + 源数据生成物体版 MVTec。"""
    result = objectseg_to_mvtec.convert(staging, src, out, clean=clean, verbose=True)
    print(f"\n[第2步完成] MVTec 输出在:\n  {out}\n  审计清单:{out / 'object_manifest.csv'}\n")
    return result


def _ask_path(prompt: str) -> Path:
    return Path(input(prompt).strip())


def _choose_seg_source() -> Path | None:
    candidate = choose_dataset(
        auto_datasets_root(),
        kinds={"yolo_instance"},
        title="选择源 YOLO Seg 数据集",
        include_unknown=True,
    )
    return candidate.path if candidate is not None else None


def _ask_output_path(prompt: str, source: Path, operation: str) -> Path:
    default = default_output_dir(source, operation)
    raw = input(f"{prompt} [默认 {default}]: ").strip()
    return Path(raw).expanduser() if raw else default


def _ask_int(prompt: str, default: int) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  输入的不是数字,用默认 {default}。")
        return default


def _ask_yes(prompt: str) -> bool:
    return input(prompt).strip().lower() in ("y", "yes", "是")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="物体版 MVTec 两步交互向导"
    )
    parser.add_argument("--dry-run", action="store_true", help="完成交互选择并显示计划，保持零写入")
    args = parser.parse_args(argv)
    print(
        "物体版 MVTec 向导(分两步):\n"
        "  1 = 生成标注预览(挑数据用)\n"
        "  2 = 把分好的 staging 转成 MVTec\n"
    )
    try:
        choice = input("选择步骤 [1/2,回车取消]: ").strip()
        if choice == "1":
            src = _choose_seg_source()
            if src is None:
                return 0
            out = _ask_output_path("预览输出目录", src, "sample-preview")
            limit = _ask_int("抽样张数(默认 500): ", 500)
            clean = False
            if out.is_dir() and any(out.iterdir()):
                clean = _ask_yes(f"输出已有内容，原子替换 {out}？[y/N]: ")
                if not clean:
                    print("已取消。")
                    return 0
            if args.dry_run:
                print(f"[dry-run] 生成预览: {src} -> {out}，limit={limit}")
            else:
                run_generate(src, out, limit=limit, clean=clean)
        elif choice == "2":
            staging = _ask_path("staging(物体文件夹根)路径: ")
            src = _choose_seg_source()
            if src is None:
                return 0
            out = _ask_output_path("MVTec 输出目录", src, "to-mvtec-object")
            clean = _ask_yes(f"输出目录 {out} 若已存在且非空要先清空重建吗? [y/N]: ")
            if args.dry_run:
                print(f"[dry-run] 生成 MVTec: staging={staging}, src={src}, out={out}")
            else:
                run_build(staging, src, out, clean=clean)
        else:
            print("已取消。")
    except EOFError:
        print("\n已取消。")
        return 0
    except KeyboardInterrupt:
        print("\n已取消。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
