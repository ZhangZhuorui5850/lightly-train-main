#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(可选)探索工具:从 YOLO-seg 源抽样若干图,生成【带缺陷标注 + 右侧信息栏】的预览图,
帮人肉眼判断每张图是不是只对应一个物体,据此在 staging/ 下建物体文件夹。

产出(都是给人看的中间物,不是 MVTec):
    <out>/<stem>.png            # 左=原图+缺陷多边形+中文名, 右=信息栏
    <out>/sample_index.csv      # stem, split, defects, n_defect_classes, n_polygons

用法:
    python seg_sample_browse.py --src <seg源> --out <浏览目录> [--limit 500] [--font <ttf>]
"""
from __future__ import annotations

import argparse
import csv
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


def _render_panel(stem: str, split: str, polys: list, names: list[str],
                  *, w: int, h: int, font_path: Path | None = None):
    """右侧信息栏:文件名/来源、缺陷种类数(>1 标 ⚠)、逐类计数带色块、多边形总数。"""
    from collections import Counter

    panel = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(panel)
    title = load_cjk_font(18, font_path)
    body = load_cjk_font(16, font_path)
    counts = Counter(cls for cls, _ in polys)
    n_classes = len(counts)
    x, y = 12, 12

    def row(text: str, font, fill=(30, 30, 30)):
        nonlocal y
        d.text((x, y), text, font=font, fill=fill)
        _, th = _text_size(d, text or "字", font)
        y += th + 6

    def divider():
        nonlocal y
        d.line([x, y, w - 12, y], fill=(210, 210, 210))
        y += 8

    row(f"文件: {stem}", title)
    row(f"来源: {split}", body)
    divider()
    warn = "（多类别）" if n_classes > 1 else ""
    row(f"缺陷种类: {n_classes}{warn}", title,
        fill=(200, 80, 0) if n_classes > 1 else (30, 30, 30))
    for cls in sorted(counts):
        d.rectangle([x, y + 3, x + 12, y + 15], fill=class_color(cls))
        label = f"{_class_name(names, cls)} ×{counts[cls]}"
        d.text((x + 20, y), label, font=body, fill=(30, 30, 30))
        _, th = _text_size(d, label, body)
        y += th + 6
    divider()
    row(f"多边形总数: {len(polys)}", body)
    return panel


def render_preview(img_path: Path, polys: list, names: list[str], split: str,
                   *, font_path: Path | None = None, panel_w: int = 320):
    """左=原图+缺陷多边形(半透明填充+描边+中文名),右=信息栏。返回拼好的 RGB 图。"""
    base = Image.open(img_path).convert("RGB")
    w, h = base.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    label_font = load_cjk_font(max(14, h // 28), font_path)

    for cls, poly in polys:
        color = class_color(cls)
        pts = [(min(max(float(px), 0.0), 1.0) * w, min(max(float(py), 0.0), 1.0) * h)
               for px, py in poly]
        if len(pts) >= 3:
            od.polygon(pts, fill=color + (70,))
            od.line(pts + [pts[0]], fill=color + (255,), width=2)
            # 类别名标在最上顶点附近,带该类色底框
            tx, ty = min(pts, key=lambda p: p[1])
            name = _class_name(names, cls)
            tw, th = _text_size(od, name, label_font)
            tx0 = min(max(0.0, tx), max(0.0, w - tw - 4))   # 水平夹住,别画出右边界
            ty0 = max(0, ty - th - 3)
            od.rectangle([tx0, ty0, tx0 + tw + 4, ty0 + th + 3], fill=color + (220,))
            od.text((tx0 + 2, ty0), name, fill=(255, 255, 255), font=label_font)

    annotated = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    panel = _render_panel(img_path.stem, split, polys, names,
                          w=panel_w, h=h, font_path=font_path)
    canvas = Image.new("RGB", (w + panel.width, h), (255, 255, 255))
    canvas.paste(annotated, (0, 0))
    canvas.paste(panel, (w, 0))
    return canvas


def _find_image(src: Path, split: str, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = src / "images" / split / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def browse(src: Path, out: Path, limit: int = 500, font_path: Path | None = None) -> int:
    """抽样最多 limit 张,生成标注预览 png + sample_index.csv。返回成功渲染的张数。"""
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
    for split, txt in tqdm(tasks, desc="标注抽样", unit="img", dynamic_ncols=True, leave=False):
        stem = txt.stem
        img = _find_image(src, split, stem)
        if img is None:
            continue
        polys = parse_label(txt)
        preview = render_preview(img, polys, names, split, font_path=font_path)
        preview.save(out / f"{stem}.png")
        classes = sorted({cls for cls, _ in polys})
        rows.append({
            "stem": stem,
            "split": split,
            "defects": ";".join(_class_name(names, c) for c in classes),
            "n_defect_classes": str(len(classes)),
            "n_polygons": str(len(polys)),
        })

    fields = ["stem", "split", "defects", "n_defect_classes", "n_polygons"]
    with (out / "sample_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="抽样生成带标注的预览图,帮人归纳有哪些物体")
    ap.add_argument("--src", required=True, type=Path, help="YOLO-seg 源数据集")
    ap.add_argument("--out", required=True, type=Path, help="预览输出目录")
    ap.add_argument("--limit", type=int, default=500, help="最多抽样张数(默认 500)")
    ap.add_argument("--font", type=Path, default=None,
                    help="中文 TTF 字体路径(默认用仓库自带 tool_lib/msyh.ttc)")
    args = ap.parse_args()
    n = browse(args.src, args.out, limit=args.limit, font_path=args.font)
    print(f"\n生成 {n} 张标注预览到 {args.out};"
          f"看图 + sample_index.csv 归纳物体,再建 staging/<物体>/ 分图。")


if __name__ == "__main__":
    main()
