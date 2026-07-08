#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(可选)探索工具:从 YOLO-seg 源抽样若干图,连同它们的缺陷类别摊到一个浏览目录。

目的:让人肉眼归纳"数据里有哪些物体",据此决定要在 staging/ 下建哪些物体文件夹。
产出的不是 MVTec,只是给人看的中间物:
    <out>/<stem>.<ext>          # 复制出来的原图
    <out>/sample_index.csv      # stem, split, defects(该图缺陷类别,分号分隔)

用法:
    python seg_sample_browse.py --src <seg源> --out <浏览目录> [--limit 500]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yoloseg_to_mvtec import IMG_EXTS, SPLITS, load_names, parse_label  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as e:  # pragma: no cover
    sys.exit(
        f"\n[依赖缺失] 找不到模块 '{e.name}'(需要 Pillow)。\n"
        "请先: conda activate lightlytrain  然后重试。\n"
        f"(当前 Python: {sys.executable})\n"
    )

# 仓库自带中文字体:<repo>/tool_lib/msyh.ttc
_DEFAULT_FONT = Path(__file__).resolve().parents[3] / "tool_lib" / "msyh.ttc"

# 固定调色板:整轮预览里同一类别永远同色,便于肉眼辨认。
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (170, 110, 40), (128, 0, 0),
]


def class_color(cls: int) -> tuple[int, int, int]:
    """类别 id → 固定 RGB 色(超出调色板则循环)。"""
    return _PALETTE[cls % len(_PALETTE)]


def _class_name(names: list[str], cls: int) -> str:
    """安全取类名;越界返回占位名(预览要尽量出图,不硬失败)。"""
    return names[cls] if 0 <= cls < len(names) else f"未知类别{cls}"


def load_cjk_font(size: int, font_path: Path | None = None):
    """加载中文字体(默认仓库自带 msyh.ttc);找不到退回 PIL 默认字体。"""
    path = str(font_path) if font_path else str(_DEFAULT_FONT)
    try:
        return ImageFont.truetype(path, size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _text_size(draw, text: str, font) -> tuple[int, int]:
    """量文字像素宽高(用 textbbox,兼容新版 PIL)。"""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left), max(0, bottom - top)


try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(it, **_kw):
        return it


def _find_image(src: Path, split: str, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = src / "images" / split / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def browse(src: Path, out: Path, limit: int = 500) -> int:
    """抽样最多 limit 张,复制图并写 sample_index.csv。返回实际样本数。"""
    src, out = src.resolve(), out.resolve()
    names = load_names(src)
    out.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, Path]] = []
    for split in SPLITS:
        lbl_dir = src / "labels" / split
        if lbl_dir.exists():
            tasks += [(split, txt) for txt in sorted(lbl_dir.glob("*.txt"))]
    tasks = tasks[:limit]

    rows: list[dict[str, str]] = []
    for split, txt in tqdm(tasks, desc="抽样", unit="img", dynamic_ncols=True, leave=False):
        stem = txt.stem
        img = _find_image(src, split, stem)
        if img is None:
            continue
        shutil.copy2(img, out / img.name)
        present = sorted({cls for cls, _ in parse_label(txt)})
        rows.append({
            "stem": stem,
            "split": split,
            "defects": ";".join(names[c] for c in present),
        })

    with (out / "sample_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "split", "defects"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="抽样摊图,帮人归纳有哪些物体")
    ap.add_argument("--src", required=True, type=Path, help="YOLO-seg 源数据集")
    ap.add_argument("--out", required=True, type=Path, help="浏览目录输出")
    ap.add_argument("--limit", type=int, default=500, help="最多抽样张数(默认 500)")
    args = ap.parse_args()
    n = browse(args.src, args.out, limit=args.limit)
    print(f"\n抽了 {n} 张到 {args.out};看 sample_index.csv 归纳物体,再去建 staging/<物体>/ 文件夹。")


if __name__ == "__main__":
    main()
