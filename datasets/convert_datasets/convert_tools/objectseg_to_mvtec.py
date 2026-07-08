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
                if obj_dir.name not in stem_objs[img.stem]:
                    stem_objs[img.stem].append(obj_dir.name)
        objects[obj_dir.name] = imgs
    conflicts = {stem: objs for stem, objs in stem_objs.items() if len(objs) > 1}
    return objects, conflicts


def _ensure_empty_layout(cat: Path) -> None:
    """建每个物体都要有的空目录(满足 MVTec 格式,即使零样本用途)。"""
    (cat / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat / "test" / "good").mkdir(parents=True, exist_ok=True)


_MANIFEST_FIELDS = ["stem", "object", "orig_split", "defects", "src_label"]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_report(result: dict) -> None:
    stats, missing, conflicts = result["stats"], result["missing"], result["conflicts"]
    print(f"\n{'物体':<16}{'缺陷/good':<16}{'数量':>8}")
    print("-" * 40)
    for obj in sorted(stats):
        for defect in sorted(stats[obj]):
            print(f"{obj:<16}{defect:<16}{stats[obj][defect]:>8}")
    if conflicts:
        print(f"\n[冲突] {len(conflicts)} 个 stem 被分到多个物体(人工分图请修正):")
        for stem, objs in conflicts.items():
            print(f"  {stem}: {objs}")
    if missing:
        print(f"\n[跳过] {len(missing)} 张图在源里查不到标签或读不了:")
        for obj, stem in missing:
            print(f"  {obj}/{stem}")


def convert(
    staging: Path,
    src: Path,
    out: Path,
    clean: bool = False,
    verbose: bool = True,
) -> dict:
    """把 staging(人工分好的物体文件夹) + src(YOLO-seg) 转成物体版 MVTec 到 out。

    返回统计 dict:{stats, missing, conflicts, manifest}。从不修改 src。
    """
    staging, src, out = staging.resolve(), src.resolve(), out.resolve()
    names = load_names(src)
    index = build_label_index(src)
    objects, conflicts = scan_staging(staging)

    if clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # stats[物体][缺陷名 或 "good"] = 计数
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing: list[tuple[str, str]] = []
    manifest: list[dict[str, str]] = []

    for obj, imgs in objects.items():
        cat = out / obj
        _ensure_empty_layout(cat)
        bar = tqdm(imgs, desc=f"物体 {obj}", unit="img", disable=not verbose,
                   dynamic_ncols=True, leave=False)
        for img_path in bar:
            stem = img_path.stem
            if stem not in index:
                missing.append((obj, stem))
                continue
            label_path, split = index[stem]
            img = cv2.imread(str(img_path))
            if img is None:
                missing.append((obj, stem))
                continue
            h, w = img.shape[:2]
            polys = parse_label(label_path)
            present = sorted({cls for cls, _ in polys})

            if not present:  # 空标签 → 对该物体是 good
                dst = cat / "test" / "good"
                cv2.imwrite(str(dst / f"{stem}.png"), img)
                stats[obj]["good"] += 1
            else:
                for cls in present:
                    dname = names[cls]
                    dst = cat / "test" / dname
                    dst.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dst / f"{stem}.png"), img)
                    only = [p for c, p in polys if c == cls]
                    mask = make_mask(only, w, h)
                    gt = cat / "ground_truth" / dname
                    gt.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(gt / f"{stem}_mask.png"), mask)
                    stats[obj][dname] += 1

            manifest.append({
                "stem": stem,
                "object": obj,
                "orig_split": split,
                "defects": ";".join(names[c] for c in present),
                "src_label": str(label_path.relative_to(src)),
            })

    result = {
        "stats": {o: dict(d) for o, d in stats.items()},
        "missing": missing,
        "conflicts": conflicts,
        "manifest": manifest,
    }
    _write_manifest(out / "object_manifest.csv", manifest)
    if verbose:
        _print_report(result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="物体版 YOLO-seg → MVTec AD 转换器")
    ap.add_argument("--staging", required=True, type=Path,
                    help="人工分好的物体文件夹根目录(staging/<物体>/*.jpg)")
    ap.add_argument("--src", required=True, type=Path,
                    help="YOLO-seg 源数据集(labels/{train,val,test}/*.txt + data.yaml)")
    ap.add_argument("--out", required=True, type=Path, help="输出 MVTec AD 根目录")
    ap.add_argument("--clean", action="store_true", help="先清空 --out")
    args = ap.parse_args()
    convert(args.staging, args.src, args.out, clean=args.clean, verbose=True)


if __name__ == "__main__":
    main()
