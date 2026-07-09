#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物体版 MVTec 交互向导:把"生成标注预览 → 人工分图 → 转换"两步用文字引导串起来。

因为中间有"下载预览→本地按物体分图→上传服务器"的人工往返,这不是一个不间断的
脚本会话,而是引导式两步:第1步生成预览,第2步(分好图之后)再来跑转换。

用法:
    python objseg_wizard.py        # 交互:先选第1步还是第2步
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import objectseg_to_mvtec  # noqa: E402
import seg_sample_browse  # noqa: E402


def run_generate(src: Path, out: Path, limit: int = 500, font_path: Path | None = None) -> int:
    """第1步:生成标注预览,并打印下一步该做什么。"""
    n = seg_sample_browse.browse(src, out, limit=limit, font_path=font_path)
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


def main() -> None:
    print(
        "物体版 MVTec 向导(分两步):\n"
        "  1 = 生成标注预览(挑数据用)\n"
        "  2 = 把分好的 staging 转成 MVTec\n"
    )
    try:
        choice = input("选择步骤 [1/2,回车取消]: ").strip()
        if choice == "1":
            src = _ask_path("源 seg 数据集路径: ")
            out = _ask_path("预览输出目录: ")
            limit = _ask_int("抽样张数(默认 500): ", 500)
            run_generate(src, out, limit=limit)
        elif choice == "2":
            staging = _ask_path("staging(物体文件夹根)路径: ")
            src = _ask_path("源 seg 数据集路径: ")
            out = _ask_path("MVTec 输出目录: ")
            clean = _ask_yes(f"输出目录 {out} 若已存在且非空要先清空重建吗? [y/N]: ")
            run_build(staging, src, out, clean=clean)
        else:
            print("已取消。")
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。")


if __name__ == "__main__":
    main()
