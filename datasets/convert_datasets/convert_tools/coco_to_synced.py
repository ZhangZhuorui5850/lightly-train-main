#!/usr/bin/env python3
"""
coco_to_synced.py
=================
将 COCO 2017 标注转为和 sync_picture.py 同形的目录结构，便于直接喂给
one_click_convert.py / LabelMeToYOLO.py：

    coco_synced/
    ├── train/   *.jpg + *.txt(或 *.json)
    ├── val/     *.jpg + *.txt(或 *.json)
    ├── test/    *.jpg + *.txt(或 *.json)
    ├── classes.txt
    └── _sync_report.txt

特点：
- 合并 instances_train2017.json + instances_val2017.json（共 ~123k 张图）
- 丢弃 test2017（COCO 官方无标注）
- 80 类 category_id (1..90 跳号) 重映射到 0..79 连续
- 分层 8:1:1 划分，保证每个类别都覆盖 train/val/test
- 默认产出 YOLO .txt（class_id cx cy w h），下游用 --source-format yolo
- --include-seg 时产出 LabelMe .json（带 rectangle + polygon），下游用 --source-format labelme

用法:
    python coco_to_synced.py
    python coco_to_synced.py --archive-root ./archive --output ./coco_synced --seed 42
    python coco_to_synced.py --include-seg
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

try:
    from .dataset_transaction import staged_output, validate_output_location
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    from dataset_transaction import staged_output, validate_output_location  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


def _load_json(path):
    return json.loads(read_text_auto(path))
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


SPLITS = ("train", "val", "test")


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    GREY = "\033[90m"


def cprint(color, text):
    print(f"{color}{text}{C.RESET}")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(
        description="Convert COCO 2017 to sync_picture-compatible folder layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--archive-root", default="archive",
                   help="COCO 解压根目录，需含 annotations_trainval2017/、train2017/、val2017/")
    p.add_argument("--output", default="coco_synced", help="输出根目录")
    p.add_argument("--include-seg", action="store_true",
                   help="同时输出 polygon 分割（产出 LabelMe .json）；默认仅 det 产出 YOLO .txt")
    p.add_argument("--ratio", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=None, help="划分随机种子")
    p.add_argument("--copy-mode", choices=["hardlink", "symlink", "copy"], default="hardlink",
                   help="图片落盘方式（默认 hardlink，节省磁盘）")
    p.add_argument("--dry-run", action="store_true", help="只做划分统计，不写文件")
    p.add_argument("--clean", action="store_true", help="安全替换已有输出")
    args = p.parse_args(argv)
    if (
        any(not math.isfinite(value) or value < 0 for value in args.ratio)
        or not math.isclose(sum(args.ratio), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        p.error("--ratio 需要由 3 个非负数组成，且总和等于 1")
    return args


def link_image(src: Path, dst: Path, mode: str):
    if dst.exists():
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _ratio_counts(n: int, ratio: tuple[float, float, float] | list[float]) -> tuple[int, int, int]:
    raw = [n * value for value in ratio]
    counts = [math.floor(value) for value in raw]
    for index in sorted(
        range(3),
        key=lambda item: (raw[item] - counts[item], ratio[item], -item),
        reverse=True,
    )[: n - sum(counts)]:
        counts[index] += 1

    positive = [index for index, value in enumerate(ratio) if value > 0]
    if n >= len(positive):
        for index in positive:
            if counts[index] > 0:
                continue
            donors = [item for item in positive if counts[item] > 1]
            if donors:
                donor = max(donors, key=lambda item: (counts[item], ratio[item]))
                counts[donor] -= 1
                counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def stratified_split(image_classes: dict, ratio, seed):
    """
    迭代分层（rarest-first）：
    按类别频次升序处理，每个类别取其未分配图片按 ratio 划入 train/val/test，
    保证每类 >=3 张图时三个 split 都至少有 1 张。
    """
    if (
        len(ratio) != 3
        or any(not math.isfinite(value) or value < 0 for value in ratio)
        or not math.isclose(sum(ratio), 1.0, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("ratio 需要由 3 个非负数组成，且总和等于 1")
    rng = random.Random(seed)
    image_ids = list(image_classes.keys())

    class_to_imgs: dict = defaultdict(list)
    for img_id, classes in image_classes.items():
        for c in classes:
            class_to_imgs[c].append(img_id)

    assignment: dict = {}
    class_split_count: dict = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    class_order = sorted(class_to_imgs.keys(), key=lambda c: len(class_to_imgs[c]))

    for cls in class_order:
        imgs = [i for i in class_to_imgs[cls] if i not in assignment]
        if not imgs:
            continue
        rng.shuffle(imgs)
        n = len(imgs)
        n_train, n_val, n_test = _ratio_counts(n, ratio)

        for i, img_id in enumerate(imgs):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            assignment[img_id] = split
            for c in image_classes[img_id]:
                class_split_count[c][split] += 1

    # leftover (没含任何类别的图，理论上不会有)
    leftover = [i for i in image_ids if i not in assignment]
    rng.shuffle(leftover)
    n = len(leftover)
    nt, nv, _ = _ratio_counts(n, ratio)
    for i, img_id in enumerate(leftover):
        if i < nt:
            assignment[img_id] = "train"
        elif i < nt + nv:
            assignment[img_id] = "val"
        else:
            assignment[img_id] = "test"

    return assignment, class_split_count


def bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x = max(0.0, min(float(x), img_w))
    y = max(0.0, min(float(y), img_h))
    w = max(0.0, min(float(w), img_w - x))
    h = max(0.0, min(float(h), img_h - y))
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return cx, cy, w / img_w, h / img_h


def write_yolo_label(path: Path, anns, coco_to_new, img_w, img_h):
    lines = []
    for ann in anns:
        if ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        cls = coco_to_new[ann["category_id"]]
        cx, cy, bw, bh = bbox_to_yolo(ann["bbox"], img_w, img_h)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_labelme_label(path: Path, anns, cat_id_to_name, img_w, img_h, image_filename):
    shapes = []
    for ann in anns:
        if ann.get("iscrowd"):
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        label = cat_id_to_name[ann["category_id"]]
        shapes.append({
            "label": label,
            "points": [[float(x), float(y)], [float(x + w), float(y + h)]],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {},
        })
        seg = ann.get("segmentation")
        if isinstance(seg, list):
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = [[float(poly[i]), float(poly[i + 1])] for i in range(0, len(poly) - 1, 2)]
                if len(pts) < 3:
                    continue
                shapes.append({
                    "label": label,
                    "points": pts,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                })

    doc = {
        "version": "5.0.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def _write_output(
    out_root: Path,
    published_out: Path,
    *,
    assignment: dict,
    images: dict,
    anns_by_image: dict,
    new_id_to_name: list[str],
    cat_id_to_name: dict,
    coco_to_new: dict,
    class_split_count: dict,
    archive_root: Path,
    args: argparse.Namespace,
) -> Path:
    for split in SPLITS:
        (out_root / split).mkdir(parents=True, exist_ok=True)
    (out_root / "classes.txt").write_text(
        "\n".join(new_id_to_name) + "\n", encoding="utf-8"
    )

    cprint(C.CYAN, "\n  写入图片 + 标签 ...")
    written = {"train": 0, "val": 0, "test": 0}
    for iid, split in tqdm(
        assignment.items(),
        total=len(assignment),
        desc="写入 COCO",
        unit="图片",
    ):
        src_path, fname, width, height = images[iid]
        stem = Path(fname).stem
        extension = Path(fname).suffix
        dst_img = out_root / split / f"{stem}{extension}"
        link_image(src_path, dst_img, args.copy_mode)
        annotations = anns_by_image[iid]
        if args.include_seg:
            dst_label = out_root / split / f"{stem}.json"
            write_labelme_label(
                dst_label,
                annotations,
                cat_id_to_name,
                width,
                height,
                dst_img.name,
            )
        else:
            dst_label = out_root / split / f"{stem}.txt"
            write_yolo_label(
                dst_label, annotations, coco_to_new, width, height
            )
        written[split] += 1

    report = out_root / "_sync_report.txt"
    with report.open("w", encoding="utf-8") as stream:
        stream.write("coco_to_synced.py 报告\n")
        stream.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        stream.write(f"archive  : {archive_root}\n")
        stream.write(f"output   : {published_out}\n")
        stream.write(f"include_seg: {args.include_seg}\n")
        stream.write(f"label fmt: {'LabelMe .json' if args.include_seg else 'YOLO .txt'}\n")
        stream.write(f"seed     : {args.seed}\n")
        stream.write(f"ratio    : {tuple(args.ratio)}\n\n")
        stream.write("── 各 split 样本数 ──\n")
        for split in SPLITS:
            stream.write(f"  {split:<8} {written[split]}\n")
        stream.write("\n── 每类在 split 的分布 ──\n")
        stream.write(f"{'class':<22}{'train':>10}{'val':>10}{'test':>10}\n")
        for cls, name in enumerate(new_id_to_name):
            counts = class_split_count[cls]
            stream.write(
                f"{name:<22}{counts['train']:>10}{counts['val']:>10}{counts['test']:>10}\n"
            )
    return published_out / report.relative_to(out_root)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    archive_root = Path(args.archive_root).expanduser().resolve()
    out_root = validate_output_location(Path(args.output), [archive_root])

    anno_dir = archive_root / "annotations_trainval2017" / "annotations"

    def resolve_split_dir(name: str) -> Path:
        flat = archive_root / name
        nested = archive_root / name / name
        if nested.is_dir() and any(nested.glob("*.jpg")):
            return nested
        if flat.is_dir() and any(flat.glob("*.jpg")):
            return flat
        return nested if nested.is_dir() else flat

    train_imgs_dir = resolve_split_dir("train2017")
    val_imgs_dir = resolve_split_dir("val2017")

    for p in [anno_dir, train_imgs_dir, val_imgs_dir]:
        if not p.is_dir():
            cprint(C.RED, f"  [ERROR] 目录不存在: {p}")
            sys.exit(1)

    cprint(C.CYAN + C.BOLD, "═" * 62)
    cprint(C.CYAN + C.BOLD, "  COCO 2017 → coco_synced")
    cprint(C.CYAN + C.BOLD, "═" * 62)
    cprint(C.GREY, f"  archive_root : {archive_root}")
    cprint(C.GREY, f"  output       : {out_root}")
    cprint(C.GREY, f"  include_seg  : {args.include_seg}")
    cprint(C.GREY, f"  label format : {'LabelMe .json' if args.include_seg else 'YOLO .txt'}")
    cprint(C.GREY, f"  ratio        : {tuple(args.ratio)}")
    cprint(C.GREY, f"  copy_mode    : {args.copy_mode}")
    cprint(C.GREY, f"  seed         : {args.seed}")
    if args.dry_run:
        cprint(C.YELLOW, "  DRY-RUN（不写文件）")
    print()

    cprint(C.CYAN, f"  读取 instances_train2017.json ...")
    train_json = _load_json(anno_dir / "instances_train2017.json")
    cprint(C.CYAN, f"  读取 instances_val2017.json ...")
    val_json = _load_json(anno_dir / "instances_val2017.json")

    categories = train_json["categories"]
    cat_ids_sorted = sorted(c["id"] for c in categories)
    coco_to_new = {coco_id: i for i, coco_id in enumerate(cat_ids_sorted)}
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    new_id_to_name = [cat_id_to_name[c] for c in cat_ids_sorted]
    cprint(C.GREEN, f"  类别数: {len(new_id_to_name)} (COCO id {cat_ids_sorted[0]}..{cat_ids_sorted[-1]} 跳号 → 0..{len(new_id_to_name)-1})")

    images: dict = {}
    for img in train_json["images"]:
        images[img["id"]] = (train_imgs_dir / img["file_name"], img["file_name"], img["width"], img["height"])
    for img in val_json["images"]:
        images[img["id"]] = (val_imgs_dir / img["file_name"], img["file_name"], img["width"], img["height"])
    cprint(C.GREEN, f"  图片总数: {len(images)}")

    anns_by_image: dict = defaultdict(list)
    for ann in train_json["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    for ann in val_json["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    total_anns = sum(len(v) for v in anns_by_image.values())
    cprint(C.GREEN, f"  标注总数: {total_anns}")

    annotated_ids = [iid for iid in images if anns_by_image.get(iid)]
    dropped = len(images) - len(annotated_ids)
    cprint(C.GREY, f"  有标注图片: {len(annotated_ids)}（丢弃无标注 {dropped} 张）")

    image_classes: dict = {}
    for iid in annotated_ids:
        cls_set = set()
        for ann in anns_by_image[iid]:
            if ann.get("iscrowd"):
                continue
            cls_set.add(coco_to_new[ann["category_id"]])
        if cls_set:
            image_classes[iid] = cls_set

    cprint(C.CYAN, "  分层 8:1:1 划分中 ...")
    assignment, class_split_count = stratified_split(image_classes, args.ratio, args.seed)

    missing = []
    for cls in range(len(new_id_to_name)):
        for s in SPLITS:
            if class_split_count[cls][s] == 0:
                missing.append((new_id_to_name[cls], s))
    if missing:
        cprint(C.YELLOW, f"  ⚠ 有 {len(missing)} 个 类别×split 缺样本:")
        for name, s in missing[:10]:
            print(f"     {name} 缺 {s}")
    else:
        cprint(C.GREEN, "  ✔ 80 类在 train/val/test 全部覆盖")

    split_counts = {s: 0 for s in SPLITS}
    for s in assignment.values():
        split_counts[s] += 1
    total = sum(split_counts.values())
    for s in SPLITS:
        pct = 100 * split_counts[s] / total if total else 0
        cprint(C.GREEN, f"  {s:<6} {split_counts[s]:>7} 张  ({pct:.1f}%)")

    if args.dry_run:
        cprint(C.YELLOW, "\n  DRY-RUN 结束，未写文件。")
        return 0

    with staged_output(out_root, clean=args.clean) as stage:
        report = _write_output(
            stage,
            out_root,
            assignment=assignment,
            images=images,
            anns_by_image=anns_by_image,
            new_id_to_name=new_id_to_name,
            cat_id_to_name=cat_id_to_name,
            coco_to_new=coco_to_new,
            class_split_count=class_split_count,
            archive_root=archive_root,
            args=args,
        )

    cprint(C.GREEN + C.BOLD, f"\n  ✔ 完成。报告: {report}")
    print()
    cprint(C.CYAN, "  下一步：")
    if args.include_seg:
        cprint(C.GREY, f"    python one_click_convert.py {out_root.name} -o converted_all --label-format labelme")
    else:
        cprint(C.GREY, f"    python one_click_convert.py {out_root.name} -o converted_all --label-format yolo")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
