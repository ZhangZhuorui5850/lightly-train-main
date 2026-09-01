#!/usr/bin/env python3
"""
sync_media.py
=============
遍历若干个源文件夹，将其中所有图片和对应的标注文件整理到训练数据集结构：

    moxingxunlian/
    ├── train/   (80%)
    ├── val/     (10%)
    └── test/    (10%)

支持两种标注格式：
- LabelMe: 图片 + .json
- YOLO:    图片 + .txt

该脚本只负责整理原始配对数据到 train/val/test。
后续由 one_click_convert.py / LabelMeToYOLO.py 再导出为 dataset_det / dataset_cls / dataset_seg。

规则：
- 文件名中的中文及特殊字符自动替换为安全 ASCII 字符
- 图片和标注文件始终保持相同的 stem，方便后续使用
- 同名冲突追加序号 __2 __3 ...
- 完全相同的文件（MD5 一致）直接跳过
- 只有图片+标注 完整配对的才进入数据集，孤立图片/标注单独记录
- 随机打乱后按 8:1:1 分配（可用 --seed 固定随机种子保证可复现）
- 复制完成后输出详细彩色统计报告，并写入 _sync_report.txt

用法:
    python sync_media.py <源文件夹1> <源文件夹2> ... [选项]

示例:
    python sync_media.py folderA folderB folderC
    python sync_media.py folderA folderB folderC -o moxingxunlian --seed 42
    python sync_media.py folderA folderB folderC --dry-run
"""

import argparse
import hashlib
import os
import random
import re
import shutil
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional

try:
    from .progress import tqdm
    from .text_encoding import read_text_auto
    from .dataset_transaction import staged_output, validate_output_location
except ImportError:
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]
    from dataset_transaction import staged_output, validate_output_location  # type: ignore[no-redef]

# ── 支持的图片格式 ────────────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
              ".webp", ".heic", ".heif", ".svg", ".ico", ".raw", ".cr2",
              ".nef", ".arw", ".dng"}

# ── ANSI 颜色 ─────────────────────────────────────────────────────────────────
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    GREY   = "\033[90m"
    MAGENTA= "\033[95m"

def cprint(color, text):
    print(f"{color}{text}{C.RESET}")

# ── 文件名清洗 ────────────────────────────────────────────────────────────────
def sanitize_stem(stem: str) -> str:
    nfkd = unicodedata.normalize("NFKD", stem)
    ascii_approx = nfkd.encode("ascii", errors="ignore").decode("ascii")
    base = ascii_approx if ascii_approx.strip() else stem
    safe = re.sub(r"[^\w.\-]", "_", base, flags=re.ASCII)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe if safe else "file"

# ── 工具函数 ──────────────────────────────────────────────────────────────────
def file_md5(path: Path, chunk: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()

def unique_stem(wanted: str, used: set) -> str:
    if wanted not in used:
        return wanted
    i = 2
    while f"{wanted}__{i}" in used:
        i += 1
    return f"{wanted}__{i}"


def detect_label_format(src_roots: list[Path]) -> str:
    # VOC-style semantic segmentation takes priority
    if _has_voc_semantic_masks(src_roots):
        return "voc_seg"

    has_json = False
    has_txt = False
    for root in src_roots:
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                path = Path(dirpath) / fname
                suffix = path.suffix.lower()
                if suffix == ".json":
                    has_json = True
                elif suffix == ".txt" and path.name.lower() not in {"classes.txt"}:
                    has_txt = True
            if has_json and has_txt:
                break
        if has_json and has_txt:
            break

    if has_json and not has_txt:
        return "labelme"
    if has_txt and not has_json:
        return "yolo"
    if has_json and has_txt:
        return "mixed"
    return "unknown"


def _has_voc_semantic_masks(src_roots: list[Path]) -> bool:
    """Check if any source root has a SegmentationClass/ directory with PNG files."""
    for root in src_roots:
        seg_dir = root / "SegmentationClass"
        if seg_dir.is_dir():
            for f in seg_dir.iterdir():
                if f.suffix.lower() == ".png":
                    return True
    return False


def detect_seg_type(src_roots: list[Path]) -> str:
    """Auto-detect seg type: 'semantic' (VOC PNG masks) or 'instance' (JSON/TXT labels)."""
    if _has_voc_semantic_masks(src_roots):
        return "semantic"
    fmt = detect_label_format(src_roots)
    if fmt in {"labelme", "yolo"}:
        return "instance"
    return "unknown"


def collect_voc_semantic_files(src_roots: list[Path]) -> list[tuple[Path, Path, str]]:
    """Collect (image_path, mask_path, stem) tuples from VOC-style semantic seg datasets.

    Looks for:
    - JPEGImages/ or images/ for images
    - SegmentationClass/ or masks/ for PNG masks
    """
    image_dir_names = {"JPEGImages", "images", "Images"}
    mask_dir_names = {"SegmentationClass", "masks", "Masks"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    results: list[tuple[Path, Path, str]] = []
    used_output_stems: set[str] = set()
    for source_index, root in enumerate(src_roots):
        # Find image directory
        img_dir = None
        for name in image_dir_names:
            candidate = root / name
            if candidate.is_dir():
                img_dir = candidate
                break
        if img_dir is None:
            # Fallback: use root itself if it contains images
            img_dir = root

        # Find mask directory
        mask_dir = None
        for name in mask_dir_names:
            candidate = root / name
            if candidate.is_dir():
                mask_dir = candidate
                break
        if mask_dir is None:
            continue

        # Build mask index: stem -> mask_path
        mask_index: dict[str, Path] = {}
        for f in mask_dir.iterdir():
            if f.suffix.lower() == ".png":
                mask_index[f.stem] = f

        # Match images with masks
        for f in img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in image_exts:
                mask_path = mask_index.get(f.stem)
                if mask_path is not None:
                    output_stem = sanitize_stem(f.stem)
                    if output_stem in used_output_stems:
                        output_stem = unique_stem(
                            f"src{source_index + 1}_{output_stem}", used_output_stems
                        )
                    used_output_stems.add(output_stem)
                    results.append((f, mask_path, output_stem))

    return results


def _read_voc_split_file(split_file: Path) -> list[str]:
    """Read a VOC-style split file (one stem per line, no extension)."""
    stems = []
    for line in read_text_auto(split_file).splitlines():
        stem = line.strip()
        if stem:
            stems.append(stem)
    return stems


def split_voc_semantic_files(
    src_roots: list[Path],
    dest: Path,
    seed: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Split VOC-style semantic seg dataset into train/val/test.

    If ImageSets/ with split files exists, uses those.
    Otherwise falls back to 8:1:1 random split.

    Copies images to dest/images/{split}/ and masks to dest/masks/{split}/.
    """
    pairs = collect_voc_semantic_files(src_roots)
    if not pairs:
        return {"paired_total": 0, "split_counts": {"train": 0, "val": 0, "test": 0}}

    # Check for VOC-style ImageSets split files
    split_map: dict[Path, str] = {}  # image path -> split_name
    split_aliases = {"trn": "train", "val": "val", "test": "test", "train": "train"}

    for root in src_roots:
        imagesets_dir = root / "ImageSets"
        if imagesets_dir.is_dir():
            root_split_files: dict[str, list[str]] = {}
            for f in imagesets_dir.iterdir():
                if f.suffix.lower() == ".txt":
                    split_name = split_aliases.get(f.stem.lower(), f.stem.lower())
                    stems = _read_voc_split_file(f)
                    if stems:
                        root_split_files[split_name] = stems
            for split_name, stems in root_split_files.items():
                stem_set = set(stems)
                for image_path, _mask_path, _output_stem in pairs:
                    try:
                        image_path.relative_to(root)
                    except ValueError:
                        continue
                    if image_path.stem in stem_set:
                        split_map[image_path.resolve()] = split_name

    required_splits = {"train", "val", "test"}

    all_splits = required_splits if split_map else required_splits
    if not dry_run:
        for split in all_splits:
            (dest / "images" / split).mkdir(parents=True, exist_ok=True)
            (dest / "masks" / split).mkdir(parents=True, exist_ok=True)

    # Assign splits
    image_paths = [image_path.resolve() for image_path, _, _ in pairs]
    unmatched_in_splits = [path for path in image_paths if path not in split_map]

    if unmatched_in_splits and split_map:
        print(f"[INFO] {len(unmatched_in_splits)} 个样本未出现在 ImageSets 中，将随机补充分配")
        rng = random.Random(seed)
        rng.shuffle(unmatched_in_splits)
        for index, image_path in enumerate(unmatched_in_splits):
            split_map[image_path] = ("train", "val", "test")[index % 10 // 8 if index % 10 < 9 else 2]
    elif not split_map:
        # No split files found - random 8:1:1
        stems_list = sorted(image_paths, key=str)
        rng = random.Random(seed)
        rng.shuffle(stems_list)
        n = len(stems_list)
        n_train = round(n * 0.8)
        n_val = round(n * 0.1)
        for image_path in stems_list[:n_train]:
            split_map[image_path] = "train"
        for image_path in stems_list[n_train:n_train + n_val]:
            split_map[image_path] = "val"
        for image_path in stems_list[n_train + n_val:]:
            split_map[image_path] = "test"

    stats: dict = {
        "paired_total": len(pairs),
        "split_counts": {s: 0 for s in all_splits},
        "errors": [],
        "skipped_no_split": 0,
    }

    for img_path, mask_path, stem in tqdm(
        pairs,
        desc="复制语义样本",
        unit="样本",
        total=len(pairs),
    ):
        split = split_map.get(img_path.resolve())
        if split is None:
            stats["skipped_no_split"] += 1
            continue

        img_dest = dest / "images" / split / img_path.name
        mask_dest = dest / "masks" / split / f"{stem}.png"

        if not dry_run:
            try:
                shutil.copy2(img_path, img_dest)
            except Exception as e:
                stats["errors"].append(f"Image copy failed: {img_path}: {e}")
                continue
            try:
                shutil.copy2(mask_path, mask_dest)
            except Exception as e:
                stats["errors"].append(f"Mask copy failed: {mask_path}: {e}")
                continue

        stats["split_counts"][split] = stats["split_counts"].get(split, 0) + 1

    return stats

# ── 收集文件 ──────────────────────────────────────────────────────────────────
_PAIR_CONTAINER_NAMES = {
    "images", "image", "jpegimages", "imgs",
    "labels", "label", "annotations", "annotation",
}


def _sample_key(root: Path, path: Path, source_index: int) -> tuple[int, str]:
    """Return a source-scoped, relative sample identity.

    ``images/train/a.jpg`` and ``labels/train/a.txt`` intentionally map to the
    same key, while equally named samples from different source roots do not.
    """
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    for index, part in enumerate(parts[:-1]):
        if part.casefold() in _PAIR_CONTAINER_NAMES:
            del parts[index]
            break
    return source_index, Path(*parts).as_posix().casefold()


def collect_files(src_roots: list, label_format: str):
    """递归收集图片和标注，并保留来源及相对路径身份。"""
    image_records = []   # (src_path, orig_stem, ext, sample_key)
    label_records = []   # (src_path, orig_stem, sample_key)

    for source_index, root_value in enumerate(tqdm(
        src_roots,
        desc="扫描来源",
        unit="目录",
        leave=False,
    )):
        root = Path(root_value).expanduser().resolve()
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                ext   = fpath.suffix.lower()
                stem  = fpath.stem
                key = _sample_key(root, fpath, source_index)
                if ext in IMAGE_EXTS:
                    image_records.append((fpath, stem, ext, key))
                elif label_format == "labelme" and ext == ".json":
                    label_records.append((fpath, stem, key))
                elif label_format == "yolo" and ext == ".txt" and fpath.name.lower() not in {"classes.txt"}:
                    label_records.append((fpath, stem, key))

    return image_records, label_records

# ── 数据集划分 ────────────────────────────────────────────────────────────────
def split_indices(n: int, ratio=(0.8, 0.1, 0.1), seed=None):
    """
    将 n 个样本按 ratio 划分为 train/val/test 三组。
    返回 (train_indices, val_indices, test_indices)。
    """
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = round(n * ratio[0])
    n_val   = round(n * ratio[1])
    # test 取剩余，避免舍入误差丢失样本
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    return train_idx, val_idx, test_idx

# ── 主逻辑 ────────────────────────────────────────────────────────────────────
def copy_files(
    src_roots: list,
    dest: Path,
    label_format: str,
    seed=None,
    dry_run: bool = False,
    yolo_remaps: dict[int, dict[int, int]] | None = None,
):

    splits = ["train", "val", "test"]
    if not dry_run:
        for s in splits:
            (dest / s).mkdir(parents=True, exist_ok=True)

    image_records, label_records = collect_files(src_roots, label_format)

    label_suffix = ".json" if label_format == "labelme" else ".txt"
    label_name = "JSON" if label_format == "labelme" else "YOLO TXT"

    # 标注索引：orig_stem -> [src_path, ...]
    label_index: dict = defaultdict(list)
    for src_path, _stem, sample_key in label_records:
        label_index[sample_key].append(src_path)

    all_image_keys = {sample_key for _, _, _, sample_key in image_records}

    stats = {
        "total_images"        : len(image_records),
        "total_labels"        : len(label_records),
        "paired_total"        : 0,
        "images_with_label"   : 0,   # 成功复制的配对数
        "images_without_label": 0,
        "labels_without_image": 0,
        "renamed_sanitized"   : 0,
        "renamed_conflict"    : 0,
        "identical_skipped"   : 0,
        "copied_images"       : 0,
        "copied_labels"       : 0,
        "split_counts"        : {"train": 0, "val": 0, "test": 0},
        "errors"              : [],
        "rename_log"          : [],
        "orphan_images"       : [],
        "orphan_labels"       : [],
        "label_format"        : label_format,
    }

    print()
    cprint(C.CYAN + C.BOLD, "═" * 62)
    cprint(C.CYAN + C.BOLD, "  🔍  扫描文件并整理配对...")
    cprint(C.CYAN + C.BOLD, "═" * 62)

    # ── 第一步：生成所有配对 & 确定目标文件名 ────────────────────────────────
    # paired_items: list of (img_src, label_src, final_stem, ext, source_index)
    paired_items   = []
    used_stems     = set()

    for src_path, orig_stem, ext, sample_key in image_records:
        # 清洗文件名
        clean_stem = sanitize_stem(orig_stem)
        sanitized  = clean_stem != orig_stem

        # 解决冲突
        final_stem = unique_stem(clean_stem, used_stems)
        conflict   = final_stem != clean_stem

        orig_name = f"{orig_stem}{ext}"
        dest_name = f"{final_stem}{ext}"

        if sanitized or conflict:
            reasons = []
            if sanitized: reasons.append("含非ASCII字符")
            if conflict:  reasons.append("目标名冲突")
            reason_str = "、".join(reasons)
            stats["rename_log"].append((orig_name, dest_name, reason_str))
            if sanitized: stats["renamed_sanitized"] += 1
            if conflict:  stats["renamed_conflict"]  += 1

        used_stems.add(final_stem)

        # 查找对应标注
        if sample_key in label_index:
            label_candidates = label_index[sample_key]
            if len(label_candidates) > 1:
                stats["errors"].append(
                    f"同一来源及相对路径存在多个标注，已跳过: {src_path} -> {label_candidates}"
                )
                stats["images_without_label"] += 1
                stats["orphan_images"].append(str(src_path))
                continue
            label_src = label_candidates[0]
            paired_items.append((src_path, label_src, final_stem, ext, sample_key[0]))
        else:
            stats["images_without_label"] += 1
            stats["orphan_images"].append(str(src_path))

    # 孤立标注
    for src_path, _stem, sample_key in label_records:
        if sample_key not in all_image_keys:
            stats["labels_without_image"] += 1
            stats["orphan_labels"].append(str(src_path))

    stats["paired_total"] = len(paired_items)
    n = len(paired_items)

    cprint(C.GREEN, f"  ✔  发现完整配对 {n} 组，开始按 8:1:1 分配...")
    print()

    if n == 0:
        cprint(C.RED, "  ✖  没有可用的图片+标注配对，退出。")
        return stats

    # ── 第二步：划分 train/val/test ──────────────────────────────────────────
    train_idx, val_idx, test_idx = split_indices(n, seed=seed)
    split_map = {}
    for i in train_idx: split_map[i] = "train"
    for i in val_idx:   split_map[i] = "val"
    for i in test_idx:  split_map[i] = "test"

    cprint(C.CYAN + C.BOLD, "═" * 62)
    cprint(C.CYAN + C.BOLD, "  📂  开始复制文件")
    cprint(C.CYAN + C.BOLD, "═" * 62)

    # ── 第三步：复制 ──────────────────────────────────────────────────────────
    # 每个 split 内部单独管理 used_filenames（不同 split 可以同名）
    split_used: dict = {"train": set(), "val": set(), "test": set()}

    for idx, (img_src, label_src, final_stem, ext, source_index) in tqdm(
        enumerate(paired_items),
        total=len(paired_items),
        desc="复制数据",
        unit="样本",
    ):
        split = split_map[idx]
        split_dir = dest / split

        img_dest_name  = f"{final_stem}{ext}"
        label_dest_name = f"{final_stem}{label_suffix}"

        # split 内冲突处理（罕见，但 final_stem 跨 split 可能已经保证唯一）
        s_used = split_used[split]
        if img_dest_name in s_used:
            i = 2
            while f"{final_stem}__{i}{ext}" in s_used:
                i += 1
            img_dest_name  = f"{final_stem}__{i}{ext}"
            label_dest_name = f"{final_stem}__{i}{label_suffix}"

        img_dest_path  = split_dir / img_dest_name
        label_dest_path = split_dir / label_dest_name

        # MD5 查重
        if img_dest_path.exists() and file_md5(img_src) == file_md5(img_dest_path):
            stats["identical_skipped"] += 1
            cprint(C.GREY, f"  [跳过-相同]  {img_src.name}  →  {split}/")
            continue

        if not dry_run:
            try:
                shutil.copy2(img_src, img_dest_path)
            except Exception as e:
                stats["errors"].append(f"复制图片失败: {img_src} → {img_dest_path}: {e}")
                cprint(C.RED, f"  [错误]  {img_src.name}: {e}")
                continue

            try:
                remap = (yolo_remaps or {}).get(source_index)
                if label_format == "yolo" and remap and any(old != new for old, new in remap.items()):
                    converted: list[str] = []
                    for raw_line in read_text_auto(label_src).splitlines():
                        parts = raw_line.split()
                        if not parts:
                            continue
                        old_id = int(float(parts[0]))
                        if old_id not in remap:
                            raise ValueError(
                                f"标注类别 ID {old_id} 不在来源 metadata 中: {label_src}"
                            )
                        parts[0] = str(remap[old_id])
                        converted.append(" ".join(parts))
                    label_dest_path.write_text(
                        "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
                    )
                else:
                    shutil.copy2(label_src, label_dest_path)
            except Exception as e:
                stats["errors"].append(f"复制标注失败: {label_src} → {label_dest_path}: {e}")
                cprint(C.RED, f"  [错误]  {label_src.name}: {e}")
                img_dest_path.unlink(missing_ok=True)
                stats["images_without_label"] += 1
                continue

        stats["copied_images"]        += 1
        stats["copied_labels"]        += 1
        stats["images_with_label"]    += 1
        stats["split_counts"][split]  += 1
        s_used.add(img_dest_name)
        s_used.add(label_dest_name)

    return stats

# ── 报告 ──────────────────────────────────────────────────────────────────────
def print_report(
    stats: dict,
    dest: Path,
    src_roots: list,
    elapsed: float,
    seed,
    *,
    dry_run: bool = False,
    report_dest: Path | None = None,
):
    display_dest = (report_dest or dest).resolve()
    print()
    cprint(C.BLUE + C.BOLD, "═" * 62)
    cprint(C.BLUE + C.BOLD, "  📊  完成 · 统计报告")
    cprint(C.BLUE + C.BOLD, "═" * 62)

    print(f"\n  {'输出目录':<22} {C.CYAN}{display_dest}{C.RESET}")
    print(f"  {'源文件夹数量':<22} {C.CYAN}{len(src_roots)}{C.RESET}")
    print(f"  {'标注格式':<22} {C.CYAN}{stats['label_format']}{C.RESET}")
    print(f"  {'随机种子':<22} {C.CYAN}{seed if seed is not None else '随机（不固定）'}{C.RESET}")
    print(f"  {'耗时':<22} {C.CYAN}{elapsed:.2f} 秒{C.RESET}")

    print()
    cprint(C.BOLD, "  ── 数据集分布 ─────────────────────────────────────")
    total_copied = stats["images_with_label"]
    for split, count in stats["split_counts"].items():
        pct = count / total_copied * 100 if total_copied else 0
        bar = "█" * int(pct / 2)
        print(f"  {split:<8} {C.GREEN}{count:>6} 对{C.RESET}  {C.GREY}{bar} {pct:.1f}%{C.RESET}")

    print()
    cprint(C.BOLD, "  ── 文件数量 ───────────────────────────────────────")
    for label, val, color in [
        ("发现图片总数",        stats["total_images"],    C.GREEN),
        ("发现标注总数",        stats["total_labels"],    C.GREEN),
        ("完整配对总数",        stats["paired_total"],    C.GREEN),
        ("成功复制图片",        stats["copied_images"],   C.GREEN),
        ("成功复制标注",        stats["copied_labels"],   C.GREEN),
    ]:
        print(f"  ✔ {label:<25} {color}{val}{C.RESET}")

    print()
    cprint(C.BOLD, "  ── 异常 / 处理情况 ────────────────────────────────")
    for label, val, color in [
        ("因中文/特殊字符重命名",  stats["renamed_sanitized"],    C.YELLOW),
        ("因名字冲突追加序号",     stats["renamed_conflict"],      C.YELLOW),
        ("图片无对应标注",         stats["images_without_label"],  C.YELLOW),
        ("标注无对应图片",         stats["labels_without_image"],  C.YELLOW),
        ("完全相同文件已跳过",     stats["identical_skipped"],     C.GREY),
        ("发生错误数",             len(stats["errors"]),           C.RED),
    ]:
        flag = "⚠" if val > 0 and color == C.YELLOW else \
               "✖" if val > 0 and color == C.RED else "·"
        print(f"  {flag} {label:<25} {color}{val}{C.RESET}")

    if stats["rename_log"]:
        print()
        cprint(C.BOLD, "  ── 重命名记录（前 40 条）──────────────────────────")
        for orig, new, reason in stats["rename_log"][:40]:
            print(f"  {C.GREY}{orig}{C.RESET}  →  {C.YELLOW}{new}{C.RESET}  {C.GREY}[{reason}]{C.RESET}")
        if len(stats["rename_log"]) > 40:
            cprint(C.GREY, f"  ... 共 {len(stats['rename_log'])} 条，完整记录见日志")

    if stats["orphan_images"]:
        print()
        cprint(C.BOLD, "  ── 无对应标注的图片（前 20 条）─────────────────")
        for p in stats["orphan_images"][:20]:
            print(f"  {C.YELLOW}  {p}{C.RESET}")
        if len(stats["orphan_images"]) > 20:
            cprint(C.GREY, f"  ... 共 {len(stats['orphan_images'])} 条")

    if stats["orphan_labels"]:
        print()
        cprint(C.BOLD, "  ── 无对应图片的标注（前 20 条）───────────────────")
        for p in stats["orphan_labels"][:20]:
            print(f"  {C.YELLOW}  {p}{C.RESET}")
        if len(stats["orphan_labels"]) > 20:
            cprint(C.GREY, f"  ... 共 {len(stats['orphan_labels'])} 条")

    if stats["errors"]:
        print()
        cprint(C.RED + C.BOLD, "  ── 错误详情 ────────────────────────────────────")
        for e in stats["errors"]:
            cprint(C.RED, f"  ✖ {e}")

    if dry_run:
        print()
        cprint(C.YELLOW, "  dry-run：未创建目录、复制文件或写入报告。")
        cprint(C.BLUE + C.BOLD, "═" * 62)
        print()
        return

    # 写日志
    log_path = dest / "_sync_report.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("sync_media.py 报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输出目录: {display_dest}\n")
        f.write(f"随机种子: {seed if seed is not None else '随机'}\n\n")
        f.write("── 数据集分布 ──\n")
        for split, count in stats["split_counts"].items():
            pct = count / stats["images_with_label"] * 100 if stats["images_with_label"] else 0
            f.write(f"  {split:<8} {count} 对  ({pct:.1f}%)\n")
        f.write("\n── 统计数字 ──\n")
        for k, v in [
            ("发现图片总数",           stats["total_images"]),
            ("发现标注总数",           stats["total_labels"]),
            ("完整配对总数",           stats["paired_total"]),
            ("成功复制图片",           stats["copied_images"]),
            ("成功复制标注",           stats["copied_labels"]),
            ("图片无对应标注",         stats["images_without_label"]),
            ("标注无对应图片",         stats["labels_without_image"]),
            ("因中文/特殊字符重命名",  stats["renamed_sanitized"]),
            ("因名字冲突追加序号",     stats["renamed_conflict"]),
            ("完全相同文件已跳过",     stats["identical_skipped"]),
            ("发生错误数",             len(stats["errors"])),
        ]:
            f.write(f"  {k:<28}{v}\n")

        if stats["rename_log"]:
            f.write("\n── 重命名记录 ──\n")
            for orig, new, reason in stats["rename_log"]:
                f.write(f"  {orig}  →  {new}  [{reason}]\n")

        if stats["orphan_images"]:
            f.write("\n── 无对应标注的图片 ──\n")
            for p in stats["orphan_images"]:
                f.write(f"  {p}\n")

        if stats["orphan_labels"]:
            f.write("\n── 无对应图片的标注 ──\n")
            for p in stats["orphan_labels"]:
                f.write(f"  {p}\n")

        if stats["errors"]:
            f.write("\n── 错误详情 ──\n")
            for e in stats["errors"]:
                f.write(f"  {e}\n")

    print()
    cprint(C.GREEN, f"  ✔  完整报告已保存至 {display_dest / log_path.name}")
    cprint(C.BLUE + C.BOLD, "═" * 62)
    print()

# ── 入口 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="整理图片+标注为训练数据集结构（train/val/test = 8:1:1）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sources", nargs="+", help="源文件夹路径（可多个）")
    parser.add_argument(
        "-o", "--output", default="moxingxunlian",
        help="输出根目录（默认: moxingxunlian）"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子，固定后每次分配结果相同（默认: 随机）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="模拟运行，不实际写入文件"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="原子替换已有输出；不指定时拒绝覆盖非空目录",
    )
    parser.add_argument(
        "--label-format",
        choices=["auto", "labelme", "yolo"],
        default="auto",
        help="输入标注格式：auto 自动判断，labelme 表示 .json，yolo 表示 .txt",
    )

    args = parser.parse_args()

    src_roots = [Path(s) for s in args.sources]
    for r in src_roots:
        if not r.is_dir():
            cprint(C.RED, f"错误: 源文件夹不存在: {r}")
            sys.exit(1)

    src_roots = [root.expanduser().resolve() for root in src_roots]
    dest = validate_output_location(Path(args.output), src_roots)
    detected_format: Optional[str] = None
    if args.label_format == "auto":
        detected_format = detect_label_format(src_roots)
        if detected_format == "mixed":
            cprint(C.RED, "错误: 源目录同时发现 .json 和 .txt 标注，请显式指定 --label-format")
            sys.exit(1)
        if detected_format == "unknown":
            cprint(C.RED, "错误: 未检测到可用标注文件（.json 或 .txt）")
            sys.exit(1)
        label_format = detected_format
    else:
        label_format = args.label_format

    if args.dry_run:
        cprint(C.YELLOW + C.BOLD, "\n  ⚡  DRY-RUN 模式，不会实际写入文件\n")

    cprint(C.CYAN, f"\n  [INFO] 使用标注格式: {label_format}\n")

    start = datetime.now()
    if args.dry_run:
        stats = copy_files(
            src_roots, dest, label_format=label_format, seed=args.seed, dry_run=True,
        )
        elapsed = (datetime.now() - start).total_seconds()
        print_report(stats, dest, src_roots, elapsed, args.seed, dry_run=True)
        return

    with staged_output(dest, clean=args.overwrite) as stage:
        stats = copy_files(
            src_roots, stage, label_format=label_format, seed=args.seed, dry_run=False,
        )
        if stats["errors"]:
            raise RuntimeError(f"整理过程中有 {len(stats['errors'])} 个错误，未发布输出")
        elapsed = (datetime.now() - start).total_seconds()
        print_report(
            stats,
            stage,
            src_roots,
            elapsed,
            args.seed,
            report_dest=dest,
        )
    print(f"  原子发布: {dest}")


if __name__ == "__main__":
    main()
