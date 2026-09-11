#!/usr/bin/env python3
"""Convert a YOLO-segmentation dataset (polygon TXT labels) into MVTec-AD format.

Structure  : Option B  -> one top-level MVTec "category" PER defect class.
good policy : relative  -> for class i, every image WITHOUT class i is "good".

For each class C the converter emits:

    <out>/<NN_C>/
      train/good/<stem>.png                 # source-train imgs that lack C
      test/good/<stem>.png                  # source-val/test imgs that lack C
      test/<C>/<stem>.png                   # any img that CONTAINS C
      ground_truth/<C>/<stem>_mask.png      # binary mask, only C's polygons = 255

Rules (faithful to MVTec AD):
  * train/ holds ONLY good images (unsupervised setting).
  * an image containing C is an anomaly for C -> goes to test/<C> (never train).
  * a multi-class image is duplicated into every relevant class category,
    and in each its mask keeps ONLY that class's polygons.
  * "good" is relative to C: such an image may still carry OTHER defects.

Usage:
    python yoloseg_to_mvtec.py --src <yolo_seg_dir> --out <mvtec_dir>
    python yoloseg_to_mvtec.py --src ../sample_yoloseg --out ../sample_mvtec
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

try:
    from .dataset_transaction import staged_output, validate_output_location
    from .text_encoding import read_text_auto
    from .dataset_discovery import (
        dataset_root_from_config,
        find_dataset_config,
        load_yaml,
        split_sample_files,
    )
    from .output_naming import allocate_flat_sample_stem, safe_path_component
    from .progress import tqdm, write as progress_write
except ImportError:
    from dataset_transaction import (  # type: ignore[no-redef]
        staged_output,
        validate_output_location,
    )
    from text_encoding import read_text_auto  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        dataset_root_from_config,
        find_dataset_config,
        load_yaml,
        split_sample_files,
    )
    from output_naming import allocate_flat_sample_stem, safe_path_component  # type: ignore[no-redef]
    from progress import tqdm, write as progress_write  # type: ignore[no-redef]

try:
    import cv2
    import numpy as np
    import yaml
except ModuleNotFoundError as e:
    import sys
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        sys.exit(
            f"\n[依赖缺失] 找不到模块 '{e.name}'。\n"
            "你很可能没激活正确的 conda 环境(比如还在 base 里)。\n"
            "请先运行:  conda activate lightlytrain   然后重试。\n"
            f"(当前 Python: {sys.executable})\n"
        )

SPLITS = ("train", "val", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def load_names(src: Path) -> list[str]:
    try:
        yml = find_dataset_config(src)
    except FileNotFoundError:
        yml = None
    if yml is not None:
        data = yaml.safe_load(read_text_auto(yml))
        if not isinstance(data, dict):
            raise ValueError(f"YAML 顶层需要为字典: {yml}")
        names = data.get("names")
        if isinstance(names, dict):
            try:
                by_id = {int(key): str(value) for key, value in names.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"类别 ID 需要为整数: {yml}") from exc
            expected = list(range(len(by_id)))
            if sorted(by_id) != expected:
                raise ValueError(f"类别 ID 需要从 0 连续编号: {yml}")
            return [by_id[index] for index in expected]
        if isinstance(names, list):
            return [str(name) for name in names]
    cls = src / "classes.txt"
    if cls.exists():
        return [l.strip() for l in read_text_auto(cls).splitlines() if l.strip()]
    raise SystemExit(f"No data.yaml/classes.txt with names in {src}")


def find_image(src: Path, split: str, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = src / "images" / split / f"{stem}{ext}"
        if p.exists():
            return p
    image_dir = src / "images" / split
    if image_dir.is_dir():
        for path in image_dir.iterdir():
            if path.is_file() and path.stem == stem and path.suffix.lower() in IMG_EXTS:
                return path
    return None


def parse_label(txt: Path) -> list[tuple[int, np.ndarray]]:
    """Return [(class_id, polygon Nx2 normalized), ...]; [] for good images."""
    polys = []
    if not txt.exists():
        return polys
    for line_number, line in enumerate(read_text_auto(txt).splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) < 7 or len(parts[1:]) % 2:
            raise ValueError(f"标签格式错误: {txt}:{line_number}（多边形至少需要 3 个坐标点）")
        try:
            cls_value = float(parts[0])
            coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
        except ValueError as exc:
            raise ValueError(f"标签包含非数字字段: {txt}:{line_number}") from exc
        if not np.isfinite(cls_value) or not cls_value.is_integer() or cls_value < 0:
            raise ValueError(f"类别编号无效: {txt}:{line_number}: {parts[0]}")
        if not np.isfinite(coords).all() or ((coords < 0) | (coords > 1)).any():
            raise ValueError(f"归一化坐标超出 [0, 1]: {txt}:{line_number}")
        cls = int(cls_value)
        polys.append((cls, coords))
    return polys


def make_mask(polys: list[np.ndarray], w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polys:
        pts = np.round(poly * np.array([max(w - 1, 0), max(h - 1, 0)])).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=255)
    return mask


def mvtec_defect_names(names: list[str]) -> dict[int, str]:
    """为缺陷分配独立目录，并保留 MVTec 的正常样本目录 good。"""
    used = {"good"}
    return {
        cid: allocate_flat_sample_stem(safe_path_component(name, fallback=f"class_{cid}"), used)
        for cid, name in enumerate(names)
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"图片写入失败: {path}")


def _convert_unpublished(
    src: Path,
    out: Path,
    *,
    report_out: Path,
    verbose: bool,
    dry_run: bool = False,
) -> dict:
    names = load_names(src)

    # category folder names: zero-padded index + class name (stable ordering)
    pad = max(1, len(str(max(0, len(names) - 1))))
    cat_name = {
        i: safe_path_component(f"{i:0{pad}d}_{name}", fallback=f"{i:0{pad}d}_class")
        for i, name in enumerate(names)
    }
    defect_name = mvtec_defect_names(names)
    if not names:
        raise ValueError(f"类别列表为空: {src}")
    for cls in range(len(names)):
        if dry_run:
            continue
        category = out / cat_name[cls]
        (category / "train" / "good").mkdir(parents=True, exist_ok=True)
        (category / "test" / "good").mkdir(parents=True, exist_ok=True)
        (category / "test" / defect_name[cls]).mkdir(parents=True, exist_ok=True)
        (category / "ground_truth" / defect_name[cls]).mkdir(
            parents=True, exist_ok=True
        )

    stats = {i: {"train_good": 0, "test_good": 0, "defect": 0} for i in range(len(names))}

    # Images are authoritative: a missing TXT is a valid empty-label/good sample.
    tasks: list[tuple[str, Path, Path]] = []
    try:
        config_path = find_dataset_config(src)
    except FileNotFoundError:
        config_path = None
    if config_path is not None:
        config = load_yaml(config_path)
        source_root = dataset_root_from_config(config_path, config)
        for split in SPLITS:
            tasks.extend(
                (split, image_path, label_path)
                for image_path, label_path, _relative_path in split_sample_files(
                    source_root, config, split, annotation="labels"
                )
            )
    else:
        for split in SPLITS:
            image_dir = src / "images" / split
            if image_dir.is_dir():
                for path in sorted(image_dir.rglob("*")):
                    if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                        continue
                    relative_path = path.relative_to(image_dir)
                    label_path = (src / "labels" / split / relative_path).with_suffix(".txt")
                    tasks.append((split, path, label_path))
    if not tasks:
        raise ValueError(f"未发现可读取的图片: {src / 'images'}")

    stem_counts = Counter(path.stem for _, path, _ in tasks)
    used_stems: set[str] = set()
    output_stems: dict[Path, str] = {}
    for split, image_path, _label_path in tasks:
        base = image_path.stem if stem_counts[image_path.stem] == 1 else f"{split}__{image_path.stem}"
        candidate = base
        suffix = 2
        while candidate in used_stems:
            candidate = f"{base}__{suffix}"
            suffix += 1
        used_stems.add(candidate)
        output_stems[image_path] = candidate

    bar = tqdm(tasks, desc=f"转换 {src.name}", unit="img",
               disable=not verbose, dynamic_ncols=True, leave=True)
    skipped = 0
    for split, img_path, txt in bar:
        output_stem = output_stems[img_path]
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            progress_write(f"  [skip] 图片无法读取: {img_path}")
            continue
        h, w = img.shape[:2]
        polys = parse_label(txt)
        present = {cls for cls, _ in polys}
        invalid = sorted(cls for cls in present if cls >= len(names))
        if invalid:
            raise ValueError(
                f"类别越界: {txt} 出现 {invalid}，类别数为 {len(names)}"
            )

        for cls in range(len(names)):
            cat = out / cat_name[cls]
            if cls in present:
                # anomaly for this class -> test/<class> + mask
                dst = cat / "test" / defect_name[cls]
                only = [p for c, p in polys if c == cls]
                mask = make_mask(only, w, h)
                gt = cat / "ground_truth" / defect_name[cls]
                if not dry_run:
                    _write_image(dst / f"{output_stem}.png", img)
                    _write_image(gt / f"{output_stem}_mask.png", mask)
                stats[cls]["defect"] += 1
            else:
                # good relative to this class
                sub = "train" if split == "train" else "test"
                dst = cat / sub / "good"
                if not dry_run:
                    _write_image(dst / f"{output_stem}.png", img)
                stats[cls]["train_good" if sub == "train" else "test_good"] += 1

    if verbose:
        if skipped:
            print(f"  ({skipped} 张因缺图/无法读取被跳过)")
        print(f"\n{'[dry-run] 预览' if dry_run else 'Converted'} {src.name} -> {report_out}  (Option B, relative-good)\n")
        print(f"{'category':<22}{'train/good':>11}{'test/good':>11}{'test/defect':>13}")
        for i in range(len(names)):
            s = stats[i]
            print(f"{cat_name[i]:<22}{s['train_good']:>11}{s['test_good']:>11}{s['defect']:>13}")
    return stats


def convert(src: Path, out: Path, clean: bool = False, verbose: bool = True, *, dry_run: bool = False) -> dict:
    """Convert a YOLO-seg dataset and atomically publish the completed output."""
    src = src.expanduser().resolve()
    try:
        config_path = find_dataset_config(src)
    except FileNotFoundError:
        source_root = src
    else:
        source_root = dataset_root_from_config(config_path, load_yaml(config_path))
    published_out = validate_output_location(out, [src, source_root])
    if dry_run:
        return _convert_unpublished(src, published_out, report_out=published_out, verbose=verbose, dry_run=True)
    with staged_output(published_out, clean=clean) as stage:
        return _convert_unpublished(
            src,
            stage,
            report_out=published_out,
            verbose=verbose,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--clean", action="store_true", help="wipe --out first")
    ap.add_argument("--dry-run", action="store_true", help="检查图片和标签并预览转换统计，保持零写入")
    args = ap.parse_args()
    convert(args.src, args.out, clean=args.clean, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
