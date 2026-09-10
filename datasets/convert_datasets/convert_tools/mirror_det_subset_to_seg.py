#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按图片名把一个 det 子集(如挑出的 200 张)镜像成对应的 seg 子集。

场景：转换时同一批源图会生成一份 datasets_det 和一份 datasets_seg(文件名 stem 一致)。
当你用 export 从大数据集里挑出 200 张 det 图后，想拿到这 200 张**对应的 seg 数据**，
并组织成和 det 子集一样的目录/划分结构(images/labels 或 images/masks + data.yaml)。

匹配方式：按图片文件名的 stem(去扩展名)。det 子集里每张图，在 seg 全量数据集里找同
名 stem 的图 + 对应标注(实例分割 labels/<stem>.txt，语义分割 masks/<stem>.png)，复制到
输出目录，并**沿用 det 子集给这张图分的 split**(train/val/test)——因为 det export 会重新
划分，同一张图在 seg 源里的 split 未必一样，所以要全量建 stem 索引再按 det 的划分落位。

支持两种 seg 格式，自动识别：
  - 实例分割(YOLO polygon)：images/<split>/<stem>.jpg + labels/<split>/<stem>.txt
                             或 <split>/images/<stem>.jpg + <split>/labels/<stem>.txt
                             data.yaml: task=segment, train/val/test=images/...，names
  - 语义分割(PNG mask)     ：images/<split>/<stem>.jpg + masks/<split>/<stem>.png
                             或 <split>/images/<stem>.jpg + <split>/masks/<stem>.png
                             data.yaml: task=semantic_segmentation, train={images,masks}，classes

用法:
  # 交互式(推荐)：不带路径参数，自动扫描 datasets/，按"图片名重合度"排序供选择
  python datasets/convert_datasets/convert.py det2seg
  python datasets/convert_datasets/convert_tools/mirror_det_subset_to_seg.py

  # 非交互：直接指定两个数据集
  python .../mirror_det_subset_to_seg.py \
      --det-subset  datasets/wuwanPic_dataset/dataset_det_A \
      --seg-source  datasets/wuwanPic_dataset/dataset_seg \
      [--output ...] [--copy-mode copy|symlink|hardlink] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from dataset_discovery import (
    dataset_root_from_config,
    find_dataset_config,
    load_yaml,
    split_annotation_dirs,
    split_image_dirs,
    split_sample_files,
)
from dataset_detector import detect_datasets
from dataset_transaction import staged_output, validate_output_location
from output_naming import default_output_dir
from progress import tqdm
from text_encoding import read_text_auto

SPLITS = ("train", "val", "test")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# 读取配置 / 定位目录
# --------------------------------------------------------------------------- #
def _resolve_root_and_yaml(path: Path) -> tuple[Path, dict]:
    """入参可以是数据集目录，也可以直接是 data.yaml。返回 (根目录, yaml字典)。"""
    path = path.expanduser().resolve()
    try:
        config_path = find_dataset_config(path)
    except FileNotFoundError:
        return path, {}
    cfg = load_yaml(config_path)
    return dataset_root_from_config(config_path, cfg), cfg


def _split_image_dir(root: Path, cfg: dict, split: str) -> Path | None:
    """Compatibility helper for callers that need one conventional split directory."""
    return split_image_dirs(root, cfg).get(split)


def _detect_seg_format(root: Path, cfg: dict) -> str:
    """返回 'semantic' 或 'instance'。"""
    task = str(cfg.get("task", "")).lower()
    if "semantic" in task:
        return "semantic"
    if task in ("segment", "instance", "detect") or "seg" in task:
        # 再用目录结构确认
        pass
    image_dirs = split_image_dirs(root, cfg)
    masks = split_annotation_dirs(
        root, cfg, image_dirs, annotation="masks"
    )
    labels = split_annotation_dirs(
        root, cfg, image_dirs, annotation="labels"
    )
    if masks and not labels:
        return "semantic"
    if labels:
        return "instance"
    if masks:
        return "semantic"
    # 默认按实例
    return "instance"


def _seg_classes(cfg: dict) -> dict[int, str]:
    raw = cfg.get("names")
    if raw is None:
        raw = cfg.get("classes")
    names: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            names[int(k)] = str(v)
    elif isinstance(raw, list):
        for i, v in enumerate(raw):
            names[i] = str(v)
    return names


# --------------------------------------------------------------------------- #
# seg 全量 stem 索引
# --------------------------------------------------------------------------- #
def build_seg_index(seg_root: Path, seg_cfg: dict, seg_format: str) -> dict[str, dict]:
    """stem -> {image, label, split_src}。跨所有 split 扫描。"""
    index: dict[str, dict] = {}
    dup = 0
    for split in SPLITS:
        samples = split_sample_files(
            seg_root,
            seg_cfg,
            split,
            annotation="masks" if seg_format == "semantic" else "labels",
        )
        for img, label, _relative_path in samples:
            stem = img.stem
            if stem in index:
                dup += 1
                continue
            index[stem] = {
                "image": img,
                "label": label if (label and label.is_file()) else None,
                "split_src": split,
            }
    if dup:
        print(f"[warn] seg 源里有 {dup} 个重复 stem，只保留首个。", file=sys.stderr)
    return index


# --------------------------------------------------------------------------- #
# 复制
# --------------------------------------------------------------------------- #
def _place(src: Path, dst: Path, mode: str, dry: bool) -> None:
    if dry:
        return
    if src.resolve() == dst.resolve():
        raise ValueError(f"源文件与目标文件重合: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(os.path.realpath(src), dst)
    elif mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


# --------------------------------------------------------------------------- #
# 核心：给定 det 子集 + seg 源，执行镜像
# --------------------------------------------------------------------------- #
def _run_mirror_unpublished(
    *,
    det_root: Path,
    det_cfg: dict,
    seg_root: Path,
    seg_cfg: dict,
    out_root: Path,
    report_out: Path,
    copy_mode: str,
    dry_run: bool,
) -> int:
    seg_format = _detect_seg_format(seg_root, seg_cfg)
    classes = _seg_classes(seg_cfg)

    print("=" * 64)
    print(f"det 子集    : {det_root}")
    print(f"seg 源      : {seg_root}  (格式: {seg_format})")
    print(f"输出        : {report_out}")
    print(f"复制方式    : {copy_mode}{'  [DRY-RUN]' if dry_run else ''}")
    print(f"类别数      : {len(classes)}")
    print("=" * 64)

    seg_index = build_seg_index(seg_root, seg_cfg, seg_format)
    print(f"seg 源可索引图片: {len(seg_index)}")

    label_sub = "masks" if seg_format == "semantic" else "labels"
    label_ext = ".png" if seg_format == "semantic" else ".txt"

    matched_per_split: dict[str, int] = {s: 0 for s in SPLITS}
    missing: list[str] = []
    missing_label: list[str] = []
    used_splits: list[str] = []
    mapping: list[dict] = []

    for split in SPLITS:
        det_samples = split_sample_files(det_root, det_cfg, split)
        if not det_samples:
            continue
        used_splits.append(split)
        for det_img, _det_label, relative_path in tqdm(
            det_samples,
            desc=f"镜像 {split}",
            unit="图片",
        ):
            stem = det_img.stem
            hit = seg_index.get(stem)
            if hit is None:
                missing.append(f"{split}/{det_img.name}")
                continue
            # 输出文件名沿用 det 子集的图片名，保证 det-200 与 seg-200 文件名一一对应
            out_img = out_root / "images" / split / relative_path
            out_lbl = (out_root / label_sub / split / relative_path).with_suffix(label_ext)
            _place(hit["image"], out_img.with_suffix(hit["image"].suffix), copy_mode, dry_run)
            if hit["label"] is not None:
                _place(hit["label"], out_lbl, copy_mode, dry_run)
            else:
                missing_label.append(f"{split}/{stem}")
            matched_per_split[split] += 1
            mapping.append({
                "stem": stem, "split": split, "seg_split_src": hit["split_src"],
                "image": hit["image"].name,
                "label": hit["label"].name if hit["label"] else None,
            })

    total_matched = sum(matched_per_split.values())
    total_det = total_matched + len(missing)

    if not dry_run and total_matched > 0:
        out_root.mkdir(parents=True, exist_ok=True)
        if seg_format == "semantic":
            cfg_out = {"path": ".", "task": "semantic_segmentation",
                       "classes": {int(k): v for k, v in sorted(classes.items())}}
            for s in used_splits:
                cfg_out[s] = {"images": f"images/{s}", "masks": f"masks/{s}"}
        else:
            cfg_out = {"path": ".", "task": seg_cfg.get("task", "segment"),
                       "nc": len(classes),
                       "names": {int(k): v for k, v in sorted(classes.items())}}
            for s in used_splits:
                cfg_out[s] = f"images/{s}"
        (out_root / "data.yaml").write_text(
            yaml.safe_dump(cfg_out, allow_unicode=True, sort_keys=False), encoding="utf-8")
        if classes:
            (out_root / "classes.txt").write_text(
                "\n".join(classes[i] for i in sorted(classes)) + "\n", encoding="utf-8")
        (out_root / "mirror_mapping.json").write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 64)
    print(f"det 子集图片总数 : {total_det}")
    print(f"匹配到 seg       : {total_matched}   " +
          "  ".join(f"{s}={matched_per_split[s]}" for s in used_splits))
    if missing:
        print(f"[缺] seg 里找不到同名图: {len(missing)}")
        for m in missing[:10]:
            print(f"      - {m}")
        if len(missing) > 10:
            print(f"      ... 其余 {len(missing) - 10} 张")
    if missing_label:
        print(f"[缺] 有图但缺标注({label_sub}): {len(missing_label)}")
        for m in missing_label[:10]:
            print(f"      - {m}")
    if not dry_run and total_matched > 0:
        print(f"\n完成 -> {report_out}")
        print("       data.yaml / classes.txt / mirror_mapping.json 已写入")
    print("=" * 64)
    return 0 if total_matched > 0 else 1


class _NoMatchedImages(RuntimeError):
    pass


def run_mirror(
    *,
    det_root: Path,
    det_cfg: dict,
    seg_root: Path,
    seg_cfg: dict,
    out_root: Path,
    copy_mode: str,
    dry_run: bool,
    clean: bool = False,
) -> int:
    """Mirror a subset through a validated staging directory."""
    det_root = det_root.expanduser().resolve()
    seg_root = seg_root.expanduser().resolve()
    published_out = validate_output_location(out_root, [det_root, seg_root])
    if dry_run:
        return _run_mirror_unpublished(
            det_root=det_root,
            det_cfg=det_cfg,
            seg_root=seg_root,
            seg_cfg=seg_cfg,
            out_root=published_out,
            report_out=published_out,
            copy_mode=copy_mode,
            dry_run=True,
        )
    try:
        with staged_output(published_out, clean=clean) as stage:
            status = _run_mirror_unpublished(
                det_root=det_root,
                det_cfg=det_cfg,
                seg_root=seg_root,
                seg_cfg=seg_cfg,
                out_root=stage,
                report_out=published_out,
                copy_mode=copy_mode,
                dry_run=False,
            )
            if status:
                raise _NoMatchedImages
    except _NoMatchedImages:
        return 1
    return 0


def _default_out_root(det_name: str, seg_root: Path) -> Path:
    det_source = seg_root.parent / det_name
    return default_output_dir(det_source, "mirror-seg")


# --------------------------------------------------------------------------- #
# 交互：扫描 datasets/，按相关性排序供选择(而不是让用户手输路径)
# --------------------------------------------------------------------------- #
def _auto_datasets_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "datasets").is_dir():
            return (p / "datasets").resolve()
    return Path("datasets").resolve()


def _sample_label_tokens(
    root: Path, cfg: dict | None = None, limit: int = 200
) -> list[list[str]]:
    toks: list[list[str]] = []
    cfg = cfg or {}
    image_dirs = split_image_dirs(root, cfg)
    label_dirs = split_annotation_dirs(
        root, cfg, image_dirs, annotation="labels"
    )
    for split in SPLITS:
        d = label_dirs.get(split)
        if d is None:
            continue
        if not d.is_dir():
            continue
        for txt in sorted(d.rglob("*.txt")):
            for ln in read_text_auto(txt).splitlines():
                t = ln.split()
                if t:
                    toks.append(t)
                    if len(toks) >= limit:
                        return toks
    return toks


def _classify(root: Path, cfg: dict | None = None) -> str:
    """det | instance | semantic | unknown —— 读内容判定，不看名字。"""
    cfg = cfg or {}
    image_dirs = split_image_dirs(root, cfg)
    mask_dirs = split_annotation_dirs(
        root, cfg, image_dirs, annotation="masks"
    )
    for mask_dir in mask_dirs.values():
        if mask_dir.is_dir() and any(mask_dir.rglob("*.png")):
            return "semantic"
    toks = _sample_label_tokens(root, cfg)
    poly = sum(1 for t in toks if len(t) >= 7 and len(t) % 2 == 1)
    bbox = sum(1 for t in toks if len(t) == 5)
    if poly > 0 and poly >= bbox:
        return "instance"
    if bbox > 0:
        return "det"
    return "unknown"


def _dataset_stems(root: Path, cfg: dict) -> set[str]:
    stems: set[str] = set()
    for split in SPLITS:
        stems.update(
            image_path.stem
            for image_path, _label_path, _relative_path in split_sample_files(
                root, cfg, split
            )
        )
    return stems


def _scan_datasets(root: Path) -> list[dict]:
    """找出 root 下所有支持的数据集，兼容两种 split 目录布局。"""
    found: list[dict] = []
    for candidate in detect_datasets(
        root,
        kinds={"yolo_detection", "yolo_instance", "semantic_mask"},
    ):
        if candidate.kind not in {"yolo_detection", "yolo_instance", "semantic_mask"}:
            continue
        d = candidate.path
        cfg = {}
        yp = candidate.config_path or (d / "data.yaml")
        if yp.is_file():
            cfg = load_yaml(yp)
        stems = _dataset_stems(d, cfg)
        kind = {
            "yolo_detection": "det",
            "yolo_instance": "instance",
            "semantic_mask": "semantic",
        }[candidate.kind]
        found.append({"path": d, "cfg": cfg, "kind": kind,
                      "n_imgs": len(stems), "stems": stems,
                      "modified_time": candidate.modified_time})
    return found


def _prompt_choice(prompt: str, n: int, default: int | None = None) -> int | None:
    """返回 0-based 选择；空输入用 default；q/空(无default) 取消返回 None。"""
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            if default is not None:
                return default
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        print(f"  无效输入: {raw!r}（输 1-{n}，回车用默认，q 取消）")


def interactive_pick(root: Path) -> dict | None:
    print(f"扫描数据集根目录: {root}")
    all_ds = _scan_datasets(root)
    if not all_ds:
        print("没有在该目录下找到任何 YOLO 数据集(需含 images/ 与 data.yaml)。")
        return None

    # ---- 第 1 步：选 det 子集 ----
    det_cands = [x for x in all_ds if x["kind"] == "det"] or all_ds
    print("\n第 1 步 · 选择 det 子集(你挑出来的那份，比如 200 张):\n")
    print(f"  {'#':>2}  {'类型':<9}{'图数':>7}   数据集")
    print("  " + "-" * 58)
    for i, x in enumerate(det_cands, 1):
        rel = x["path"].relative_to(root)
        print(f"  {i:>2}  {x['kind']:<9}{x['n_imgs']:>7}   {rel}")
    print("  " + "-" * 58)
    di = _prompt_choice(f"\n选 det 子集 [1-{len(det_cands)}, q 取消]: ", len(det_cands))
    if di is None:
        print("已取消。")
        return None
    det = det_cands[di]

    # ---- 第 2 步：按"图片名重合度"给 seg 数据集排序后选 ----
    seg_cands = [x for x in all_ds if x["kind"] in ("instance", "semantic")]
    if not seg_cands:
        print("没找到 seg 数据集(实例或语义)。")
        return None
    det_stems = det["stems"]
    for x in seg_cands:
        inter = len(det_stems & x["stems"])
        x["overlap"] = inter
        x["overlap_frac"] = inter / max(len(det_stems), 1)
    seg_cands.sort(
        key=lambda x: (-x["overlap"], -x["modified_time"], x["path"].name)
    )

    print(f"\n第 2 步 · 选对应的 seg 源(按与该 det 子集的图片名重合度排序):\n")
    print(f"  {'#':>2}  {'类型':<9}{'重合':>12}{'图数':>7}   数据集")
    print("  " + "-" * 66)
    for i, x in enumerate(seg_cands, 1):
        rel = x["path"].relative_to(root)
        star = " ★" if i == 1 and x["overlap"] > 0 else ""
        match = f"{x['overlap']}/{len(det_stems)} ({x['overlap_frac']*100:.0f}%)"
        print(f"  {i:>2}  {x['kind']:<9}{match:>12}{x['n_imgs']:>7}   {rel}{star}")
    print("  " + "-" * 66)
    if seg_cands[0]["overlap"] == 0:
        print("  ⚠ 最高重合度为 0：这些 seg 数据集可能和该 det 子集不是同源。")
    si = _prompt_choice(
        f"\n选 seg 源 [1-{len(seg_cands)}, 回车=第1个★最匹配, q 取消]: ",
        len(seg_cands), default=0)
    if si is None:
        print("已取消。")
        return None
    seg = seg_cands[si]

    # ---- 第 3 步：复制方式 ----
    print("\n第 3 步 · 复制方式:  1) copy 拷贝   2) symlink 软链(省空间)   3) hardlink 硬链")
    mode_map = {0: "copy", 1: "symlink", 2: "hardlink"}
    mi = _prompt_choice("选择 [1-3, 回车=1 copy]: ", 3, default=0)
    copy_mode = mode_map.get(mi if mi is not None else 0, "copy")

    out_root = _default_out_root(det["path"].name, seg["path"])
    print(f"\n将把 {det['path'].name} 的图对应的 seg 数据 -> {out_root}")
    ok = _prompt_choice("确认执行? [回车=是, q 取消]: ", 1, default=0)
    if ok is None:
        print("已取消。")
        return None
    return {"det_root": det["path"], "det_cfg": det["cfg"],
            "seg_root": seg["path"], "seg_cfg": seg["cfg"],
            "out_root": out_root, "copy_mode": copy_mode}


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="把 det 子集按图片名镜像成对应 seg 子集(不带参数=交互式扫描选择)")
    ap.add_argument("--det-subset", type=Path, default=None,
                    help="det 子集目录或其 data.yaml；不填则进入交互式选择")
    ap.add_argument("--seg-source", type=Path, default=None,
                    help="对应的 seg 全量数据集目录或其 data.yaml；不填则进入交互式选择")
    ap.add_argument("--datasets", type=Path, default=None,
                    help="交互扫描的数据集根目录(默认自动定位 ./datasets)")
    ap.add_argument("--output", type=Path, default=None,
                    help="输出目录；缺省时在 seg 源同级按 det 子集名自动命名")
    ap.add_argument("--copy-mode", choices=("copy", "symlink", "hardlink"),
                    default="copy", help="大数据在服务器上建议 symlink/hardlink 省空间")
    ap.add_argument("--dry-run", action="store_true", help="只统计匹配情况，不写文件")
    ap.add_argument("--clean", action="store_true", help="安全替换已有输出")
    args = ap.parse_args()

    # 交互式：只要没同时给 det/seg，就扫描 datasets/ 排序供选择。
    if args.det_subset is None or args.seg_source is None:
        root = (args.datasets or _auto_datasets_root()).expanduser().resolve()
        picked = interactive_pick(root)
        if picked is None:
            return 1
        if args.output is not None:
            picked["out_root"] = args.output.expanduser()
        return run_mirror(dry_run=args.dry_run, clean=args.clean, **picked)

    # 非交互：两个路径都给了，直接跑。
    det_root, det_cfg = _resolve_root_and_yaml(args.det_subset)
    seg_root, seg_cfg = _resolve_root_and_yaml(args.seg_source)
    out_root = (args.output.expanduser() if args.output
                else _default_out_root(
                    args.det_subset.stem if args.det_subset.suffix == ".yaml"
                    else args.det_subset.name, seg_root))
    return run_mirror(det_root=det_root, det_cfg=det_cfg, seg_root=seg_root,
                      seg_cfg=seg_cfg, out_root=out_root,
                      copy_mode=args.copy_mode, dry_run=args.dry_run,
                      clean=args.clean)


if __name__ == "__main__":
    raise SystemExit(main())
