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
  * 从不修改源数据。staging 里的文件只用来“按文件名选择”;图像和标注都从
    源数据集按 stem 取原图/原标注(所以 staging 里放预览图/占位文件都行)。

用法:
  python objectseg_to_mvtec.py --staging <staging> --src <seg源> --out <mvtec输出> [--clean]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

try:
    import cv2  # noqa: F401
    import numpy as np  # noqa: F401
except ModuleNotFoundError as e:
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        sys.exit(
            f"\n[依赖缺失] 找不到模块 '{e.name}'。\n"
            "你很可能没激活正确的 conda 环境(比如还在 base 里)。\n"
            "请先运行:  conda activate lightlytrain   然后重试。\n"
            f"(当前 Python: {sys.executable})\n"
        )

# 复用 legacy 转换器里已验证的核心逻辑。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from yoloseg_to_mvtec import (  # noqa: E402,F401
    IMG_EXTS,
    SPLITS,
    find_image,
    load_names,
    make_mask,
    parse_label,
)
from dataset_transaction import staged_output, validate_output_location  # noqa: E402
from output_naming import safe_path_component  # noqa: E402
from progress import tqdm  # noqa: E402


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
        seen_stems: set[str] = set()
        for img in sorted(obj_dir.iterdir()):
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                if img.stem in seen_stems:
                    continue
                seen_stems.add(img.stem)
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


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"图片写入失败: {path}")


def _convert_unpublished(
    staging: Path,
    src: Path,
    out: Path,
    *,
    verbose: bool,
) -> dict:
    """把 staging(人工分好的物体文件夹) + src(YOLO-seg) 转成物体版 MVTec 到 out。

    返回统计 dict:{stats, missing, conflicts, manifest}。从不修改 src。
    """
    names = load_names(src)
    index = build_label_index(src)
    objects, conflicts = scan_staging(staging)

    # stats[物体][缺陷名 或 "good"] = 计数
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing: list[tuple[str, str]] = []
    manifest: list[dict[str, str]] = []

    for obj, imgs in objects.items():
        safe_object = safe_path_component(obj, fallback="object")
        cat = out / safe_object
        _ensure_empty_layout(cat)
        bar = tqdm(imgs, desc=f"物体 {obj}", unit="img", disable=not verbose,
                   dynamic_ncols=True, leave=False)
        for img_path in bar:
            stem = img_path.stem
            if stem not in index:
                missing.append((obj, stem))
                continue
            label_path, split = index[stem]
            # 只用 staging 里的“文件名”做选择;图像内容一律取源数据集里的原图,
            # 这样即使你分进去的是带标注的预览图(或占位文件),输出的也是干净原图。
            src_img = find_image(src, split, stem)
            img = cv2.imread(str(src_img)) if src_img is not None else None
            if img is None:
                missing.append((obj, stem))
                continue
            h, w = img.shape[:2]
            polys = parse_label(label_path)
            present = sorted({cls for cls, _ in polys})

            if not present:  # 空标签 → 对该物体是 good
                dst = cat / "test" / "good"
                _write_image(dst / f"{stem}.png", img)
                stats[obj]["good"] += 1
            else:
                for cls in present:
                    if cls < 0 or cls >= len(names):
                        raise SystemExit(
                            f"[类别越界] 标签 {label_path} 出现 cls={cls},"
                            f"但 data.yaml/classes.txt 只有 {len(names)} 个类别(0..{len(names) - 1})。"
                            "请检查是不是 --src 指错了,或 data.yaml 与标签不匹配。"
                        )
                    dname = names[cls]
                    safe_defect = safe_path_component(dname, fallback=f"class_{cls}")
                    dst = cat / "test" / safe_defect
                    _write_image(dst / f"{stem}.png", img)
                    only = [p for c, p in polys if c == cls]
                    mask = make_mask(only, w, h)
                    gt = cat / "ground_truth" / safe_defect
                    _write_image(gt / f"{stem}_mask.png", mask)
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


def convert(
    staging: Path,
    src: Path,
    out: Path,
    clean: bool = False,
    verbose: bool = True,
) -> dict:
    """Convert and atomically publish a complete object-oriented MVTec tree."""
    staging = staging.expanduser().resolve()
    src = src.expanduser().resolve()
    published_out = validate_output_location(out, [staging, src])
    if published_out.is_dir() and any(published_out.iterdir()) and not clean:
        raise SystemExit(
            f"输出目录已有内容: {published_out}；使用 --clean 重新生成"
        )
    with staged_output(published_out, clean=clean) as stage:
        return _convert_unpublished(staging, src, stage, verbose=verbose)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="物体版 YOLO-seg → MVTec AD 转换器")
    ap.add_argument("--staging", required=True, type=Path,
                    help="人工分好的物体文件夹根目录(staging/<物体>/*.jpg)")
    ap.add_argument("--src", required=True, type=Path,
                    help="YOLO-seg 源数据集(labels/{train,val,test}/*.txt + data.yaml)")
    ap.add_argument("--out", required=True, type=Path, help="输出 MVTec AD 根目录")
    ap.add_argument("--clean", action="store_true", help="先清空 --out")
    ap.add_argument("--dry-run", action="store_true", help="检查输入并显示输出计划，保持零写入")
    args = ap.parse_args(argv)
    try:
        if args.dry_run:
            staging = args.staging.expanduser().resolve()
            src = args.src.expanduser().resolve()
            out = validate_output_location(args.out, [staging, src])
            objects, conflicts = scan_staging(staging)
            print("[dry-run] 物体版 YOLO Seg → MVTec AD")
            print(f"  staging: {staging}")
            print(f"  输入: {src}")
            print(f"  输出: {out}")
            print(f"  物体目录: {len(objects)}, 文件名冲突: {len(conflicts)}")
            return 0
        convert(args.staging, args.src, args.out, clean=args.clean, verbose=True)
    except DuplicateStemError as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
