#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""物体版 YOLO-seg → MVTec AD 转换器。

输入:
  --staging  人工分好的物体文件夹根目录(staging/<中文物体名>/*.jpg)
  --src      YOLO-seg 源数据集(需 labels/{train,val,test}/*.txt + data.yaml|classes.txt)
  --out      输出的 MVTec AD 根目录

产出(每个物体一个 category,category=中文物体名):
  <out>/<物体>/
    train/good/                        # 空目录(零样本用途,保留以满足格式)
    test/good/                         # 空(无干净图;偶有空标签图落此)
    test/<缺陷>/<stem>.png
    ground_truth/<缺陷>/<stem>_mask.png
  <out>/object_manifest.csv            # 审计/可复现清单

规则:
  * 图属于哪个物体 = 人工分图决定(staging 文件夹名);缺陷由回源查标注自动发现。
  * 一图多缺陷 → 复制进每个缺陷子文件夹,mask 只留该缺陷的多边形。
  * 从不修改源数据。图片内容取自 staging 里的那份(与源同图)。

用法:
  python objectseg_to_mvtec.py --staging <staging> --src <seg源> --out <mvtec输出> [--clean]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
except ModuleNotFoundError as e:
    sys.exit(
        f"\n[依赖缺失] 找不到模块 '{e.name}'。\n"
        "你很可能没激活正确的 conda 环境(比如还在 base 里)。\n"
        "请先运行:  conda activate lightlytrain   然后重试。\n"
        f"(当前 Python: {sys.executable})\n"
    )

# 复用 legacy 转换器里已验证的核心逻辑。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yoloseg_to_mvtec import IMG_EXTS, SPLITS, load_names, make_mask, parse_label  # noqa: E402,F401

try:  # 单行进度条;缺 tqdm 时退化为直接迭代
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(it, **_kw):
        return it


class DuplicateStemError(RuntimeError):
    """源数据里出现重名 stem,无法确定标签归属。"""


def build_label_index(src: Path) -> dict[str, tuple[Path, str]]:
    """扫源 seg 的 labels/{split}/*.txt,建 stem -> (标签路径, split)。

    出现重名 stem 时抛 DuplicateStemError(列出冲突路径),绝不静默取其一。
    """
    index: dict[str, tuple[Path, str]] = {}
    dupes: dict[str, set[Path]] = defaultdict(set)
    for split in SPLITS:
        lbl_dir = src / "labels" / split
        if not lbl_dir.exists():
            continue
        for txt in sorted(lbl_dir.glob("*.txt")):
            if txt.stem in index:
                dupes[txt.stem].add(index[txt.stem][0])
                dupes[txt.stem].add(txt)
            index[txt.stem] = (txt, split)
    if dupes:
        detail = "\n".join(
            f"  {stem}: {sorted(str(p) for p in paths)}" for stem, paths in dupes.items()
        )
        raise DuplicateStemError("源数据存在重名 stem,无法确定标签归属:\n" + detail)
    return index


def scan_staging(staging: Path) -> tuple[dict[str, list[Path]], dict[str, list[str]]]:
    """扫 staging 下的一级子目录(每个=一个物体)。

    返回:
      objects   : {物体名: [图片路径, ...]}
      conflicts : {stem: [物体, ...]}  仅含出现在 >1 个物体里的 stem
    """
    objects: dict[str, list[Path]] = {}
    stem_objs: dict[str, list[str]] = defaultdict(list)
    for obj_dir in sorted(p for p in staging.iterdir() if p.is_dir()):
        imgs: list[Path] = []
        for img in sorted(obj_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                imgs.append(img)
                stem_objs[img.stem].append(obj_dir.name)
        objects[obj_dir.name] = imgs
    conflicts = {stem: objs for stem, objs in stem_objs.items() if len(objs) > 1}
    return objects, conflicts
