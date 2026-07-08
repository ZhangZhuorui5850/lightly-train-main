# 标注预览 + 物体版向导 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给物体版流水线加"带缺陷标注+右侧信息栏的抽样预览图",并用一个交互向导把"生成预览→人工分图→转换"两步串起来。

**Architecture:** 升级 `seg_sample_browse.py` 用 PIL 渲染标注预览(左=原图+缺陷多边形+中文名,右=信息栏),字体用仓库自带 `tool_lib/msyh.ttc`(把 `load_cjk_font`/文本量取两个小助手复制进本文件,不耦合 tool_lib)。新增 `objseg_wizard.py` 交互向导。都注册进 `convert.py`。

**Tech Stack:** Python 3.10+、Pillow(PIL,已装 12.2)、pytest;复用 `yoloseg_to_mvtec` 的 `IMG_EXTS/SPLITS/load_names/parse_label` 和 `objectseg_to_mvtec.convert`。

参考 spec: `docs/superpowers/specs/2026-07-08-objseg-annotated-preview-wizard-design.md`

---

## 文件结构

- Modify: `datasets/convert_datasets/convert_tools/seg_sample_browse.py` — 加标注渲染(palette/class_color、字体助手、render_preview、升级 browse、--font)
- Create: `datasets/convert_datasets/convert_tools/objseg_wizard.py` — 交互向导(run_generate/run_build/main)
- Modify: `datasets/convert_datasets/convert.py` — 注册 `obj-wizard`
- Modify: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py` — 追加测试
- Modify: `datasets/convert_datasets/README.md` — 用向导的说明

约定:`datasets/` 被 gitignore,提交工具文件用 `git add -f`。单行 tqdm、缺依赖中文提示、从不改源数据。

**背景事实(供实现参考):**
- `parse_label(txt)` 返回 `list[(cls:int, poly:np.ndarray Nx2 归一化)]`;空标签→`[]`。
- `load_names(src)` 返回类名列表(中文)。测试 fixture `make_src` 的 names 为 `["锈蚀","裂纹","污迹"]`。
- `seg_sample_browse.py` 现状见下方 Task 1 的"当前内容"。
- 仓库根 = `Path(__file__).resolve().parents[3]`(convert_tools→convert_datasets→datasets→仓库根);字体在 `<root>/tool_lib/msyh.ttc`。
- 测试文件顶部已有 `TOOLS`(=convert_tools 目录)、`sys`、`make_src`、`_img`、`_write_label`、`_tri`、`NAMES`。

---

## Task 1: 调色板 + class_color + 字体助手(不动 browse)

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/seg_sample_browse.py`
- Test: append to `tests/test_objectseg_to_mvtec.py`

当前 `seg_sample_browse.py` 顶部导入是:
```python
import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yoloseg_to_mvtec import IMG_EXTS, SPLITS, load_names, parse_label  # noqa: E402
```

- [ ] **Step 1: 写失败测试**

```python
def test_class_color_deterministic_and_distinct():
    import seg_sample_browse as sb
    assert sb.class_color(0) == sb.class_color(0)          # 同类同色
    assert sb.class_color(0) != sb.class_color(1)          # 不同类不同色
    c = sb.class_color(2)
    assert isinstance(c, tuple) and len(c) == 3 and all(0 <= v <= 255 for v in c)


def test_class_name_handles_out_of_range():
    import seg_sample_browse as sb
    assert sb._class_name(["锈蚀", "裂纹"], 1) == "裂纹"
    assert sb._class_name(["锈蚀", "裂纹"], 9) == "未知类别9"
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k "class_color or class_name" -v`
Expected: FAIL(`AttributeError`/`ImportError`)

- [ ] **Step 3: 在 `seg_sample_browse.py` 顶部导入区之后(`from yoloseg_to_mvtec import ...` 那行之后)加入 PIL 导入 + 调色板/字体助手**

在 `from yoloseg_to_mvtec import ...` 行之后、`try: from tqdm ...` 之前插入:

```python
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
```

(暂不改 `browse`/`main`,`shutil` 仍在用,保留。)

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k "class_color or class_name" -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/zzr/lightly-train-main
git add -f datasets/convert_datasets/convert_tools/seg_sample_browse.py \
           datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): add color palette + CJK font helpers to seg_sample_browse"
```
提交信息末尾加一行:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 2: render_preview + 信息栏

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/seg_sample_browse.py`
- Test: append to `tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试**

```python
def test_render_preview_appends_info_panel(tmp_path):
    import numpy as np
    import seg_sample_browse as sb
    from PIL import Image

    img_path = tmp_path / "x.jpg"
    _img(img_path)  # 64x64 灰图
    polys = [
        (0, np.array([[0.2, 0.2], [0.4, 0.2], [0.3, 0.4]])),   # 锈蚀
        (1, np.array([[0.6, 0.6], [0.8, 0.6], [0.7, 0.8]])),   # 裂纹
    ]
    canvas = sb.render_preview(img_path, polys, ["锈蚀", "裂纹", "污迹"], "train")
    base_w = Image.open(img_path).width
    assert canvas.height == 64                 # 高度=原图高
    assert canvas.width > base_w               # 右边拼了信息栏
    assert canvas.mode == "RGB"
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k render_preview -v`
Expected: FAIL(`AttributeError: ... 'render_preview'`)

- [ ] **Step 3: 追加 `_render_panel` 和 `render_preview`(放在 `_text_size` 之后、`_find_image` 之前)**

```python
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
    warn = " ⚠" if n_classes > 1 else ""
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
            ty0 = max(0, ty - th - 3)
            od.rectangle([tx, ty0, tx + tw + 4, ty0 + th + 3], fill=color + (220,))
            od.text((tx + 2, ty0), name, fill=(255, 255, 255), font=label_font)

    annotated = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    panel = _render_panel(img_path.stem, split, polys, names,
                          w=panel_w, h=h, font_path=font_path)
    canvas = Image.new("RGB", (w + panel.width, h), (255, 255, 255))
    canvas.paste(annotated, (0, 0))
    canvas.paste(panel, (w, 0))
    return canvas
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k render_preview -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
cd /home/zzr/lightly-train-main
git add -f datasets/convert_datasets/convert_tools/seg_sample_browse.py \
           datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): render annotated preview with defect polygons + info panel"
```
末尾加:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 3: 升级 browse(输出标注预览 + 新 CSV 列)+ --font

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/seg_sample_browse.py`
- Test: append to `tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试**

```python
def test_browse_writes_annotated_previews_and_new_csv_columns(tmp_path):
    import csv as _csv
    from PIL import Image
    import seg_sample_browse as sb

    src = make_src(tmp_path / "src")  # a=锈蚀, b=锈蚀+裂纹, c=污迹, d=空
    for split, stem in [("train", "a"), ("train", "b"), ("val", "c"), ("train", "d")]:
        _img(src / "images" / split / f"{stem}.jpg")
    out = tmp_path / "browse"
    n = sb.browse(src, out, limit=10)
    assert n == 4

    # 预览是 png,且比原图宽(拼了信息栏)
    assert (out / "b.png").exists()
    base_w = Image.open(src / "images" / "train" / "b.jpg").width
    assert Image.open(out / "b.png").width > base_w

    rows = {r["stem"]: r for r in _csv.DictReader((out / "sample_index.csv").open(encoding="utf-8"))}
    assert rows["b"]["defects"] == "锈蚀;裂纹"
    assert rows["b"]["n_defect_classes"] == "2"
    assert rows["b"]["n_polygons"] == "2"
    assert rows["d"]["n_defect_classes"] == "0"  # 空标签


def test_browse_skips_missing_image(tmp_path):
    import seg_sample_browse as sb
    src = make_src(tmp_path / "src")
    _img(src / "images" / "train" / "a.jpg")  # 只给 a 配图;b/c/d 缺图
    out = tmp_path / "browse"
    n = sb.browse(src, out, limit=10)
    assert n == 1
    assert (out / "a.png").exists()
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k "browse_writes or browse_skips" -v`
Expected: FAIL(旧 browse 输出的是拷贝的 `.jpg`,没有 `.png`/新列)

- [ ] **Step 3: 替换 `browse` 函数体、更新 `main`、更新模块 docstring、删除不再用的 `import shutil`**

3a. 删除顶部的 `import shutil` 那一行。

3b. 把模块顶部 docstring(第 3-12 行那段三引号)整体替换为:

```python
"""(可选)探索工具:从 YOLO-seg 源抽样若干图,生成【带缺陷标注 + 右侧信息栏】的预览图,
帮人肉眼判断每张图是不是只对应一个物体,据此在 staging/ 下建物体文件夹。

产出(都是给人看的中间物,不是 MVTec):
    <out>/<stem>.png            # 左=原图+缺陷多边形+中文名, 右=信息栏
    <out>/sample_index.csv      # stem, split, defects, n_defect_classes, n_polygons

用法:
    python seg_sample_browse.py --src <seg源> --out <浏览目录> [--limit 500] [--font <ttf>]
"""
```

3c. 把整个 `browse(...)` 函数替换为:

```python
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
```

3d. 把 `main()` 替换为(加 `--font`):

```python
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
```

- [ ] **Step 4: 删除被取代的旧测试**

之前 Task 7 写的 `test_sample_browse_writes_index_csv` 断言旧行为(`(out / "b.jpg").exists()` 和旧三列 CSV),已被本任务的 `test_browse_writes_annotated_previews_and_new_csv_columns` 取代,升级后会失败。**先删掉这个旧测试函数整段**(它在 `tests/test_objectseg_to_mvtec.py` 里,函数名 `test_sample_browse_writes_index_csv`)。

- [ ] **Step 4b: 运行整个测试文件,确认全过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -v`
Expected: PASS(全部;新的 browse 测试通过,旧测试已删)

- [ ] **Step 5: 提交**

```bash
cd /home/zzr/lightly-train-main
git add -f datasets/convert_datasets/convert_tools/seg_sample_browse.py \
           datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): seg_sample_browse emits annotated previews + richer index CSV"
```
末尾加:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 4: 交互向导 objseg_wizard.py

**Files:**
- Create: `datasets/convert_datasets/convert_tools/objseg_wizard.py`
- Test: append to `tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试(用 monkeypatch 验证参数透传,不测 input())**

```python
def test_wizard_run_generate_forwards(monkeypatch, tmp_path):
    import objseg_wizard as wiz
    calls = {}

    def fake_browse(src, out, limit=500, font_path=None):
        calls.update(src=src, out=out, limit=limit, font_path=font_path)
        return 7

    monkeypatch.setattr(wiz.seg_sample_browse, "browse", fake_browse)
    n = wiz.run_generate(tmp_path / "s", tmp_path / "o", limit=3)
    assert n == 7
    assert calls["limit"] == 3 and calls["src"] == tmp_path / "s"


def test_wizard_run_build_forwards(monkeypatch, tmp_path):
    import objseg_wizard as wiz
    calls = {}

    def fake_convert(staging, src, out, clean=False, verbose=True):
        calls.update(staging=staging, src=src, out=out, clean=clean)
        return {"stats": {}, "missing": [], "conflicts": {}, "manifest": []}

    monkeypatch.setattr(wiz.objectseg_to_mvtec, "convert", fake_convert)
    result = wiz.run_build(tmp_path / "stg", tmp_path / "src", tmp_path / "out", clean=True)
    assert calls["clean"] is True and calls["staging"] == tmp_path / "stg"
    assert "manifest" in result
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k wizard -v`
Expected: FAIL(`ModuleNotFoundError: objseg_wizard`)

- [ ] **Step 3: 创建 `objseg_wizard.py`**

```python
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
        "  2) 建【中文物体名】文件夹(如 盖板/ 紧固件/),把图分进去,一张图只放一个物体;\n"
        "     多物体/说不清的图先别放,避免污染;\n"
        "  3) 把分好的 staging/ 上传回服务器;\n"
        "  4) 回来再跑本向导选【第2步】做转换。\n"
    )
    return n


def run_build(staging: Path, src: Path, out: Path, clean: bool = True) -> dict:
    """第2步:按分好的 staging + 源数据生成物体版 MVTec。"""
    result = objectseg_to_mvtec.convert(staging, src, out, clean=clean, verbose=True)
    print(f"\n[第2步完成] MVTec 输出在:\n  {out}\n  审计清单:{out / 'object_manifest.csv'}\n")
    return result


def _ask_path(prompt: str) -> Path:
    return Path(input(prompt).strip())


def main() -> None:
    print(
        "物体版 MVTec 向导(分两步):\n"
        "  1 = 生成标注预览(挑数据用)\n"
        "  2 = 把分好的 staging 转成 MVTec\n"
    )
    try:
        choice = input("选择步骤 [1/2,回车取消]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return
    if choice == "1":
        src = _ask_path("源 seg 数据集路径: ")
        out = _ask_path("预览输出目录: ")
        raw = input("抽样张数(默认 500): ").strip()
        limit = int(raw) if raw else 500
        run_generate(src, out, limit=limit)
    elif choice == "2":
        staging = _ask_path("staging(物体文件夹根)路径: ")
        src = _ask_path("源 seg 数据集路径: ")
        out = _ask_path("MVTec 输出目录: ")
        run_build(staging, src, out, clean=True)
    else:
        print("已取消。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k wizard -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd /home/zzr/lightly-train-main
git add -f datasets/convert_datasets/convert_tools/objseg_wizard.py \
           datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): interactive two-step object-version wizard"
```
末尾加:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 5: 注册 obj-wizard 进 convert.py

**Files:**
- Modify: `datasets/convert_datasets/convert.py`

- [ ] **Step 1: 读文件,找到 `to-mvtec-obj` 那个 `COMMANDS` 条目**

Run: `Read datasets/convert_datasets/convert.py`(定位 `"to-mvtec-obj": Tool(...)`)

- [ ] **Step 2: 在 `"to-mvtec-obj"` 条目之后插入 `obj-wizard` 条目**

```python
    "obj-wizard": Tool(
        "objseg_wizard.py",
        "物体版向导(交互):分两步——①生成标注预览挑数据 ②分好后转 MVTec",
        "转 MVTec AD",
        "回车进入向导,按提示选第1步或第2步",
        interactive=True,
    ),
```

- [ ] **Step 3: 冒烟测试**

Run: `cd datasets/convert_datasets && python convert.py list`
Expected: `[转 MVTec AD]` 分组下出现 `obj-wizard`(以及已有的 `to-mvtec`、`to-mvtec-obj`)。

- [ ] **Step 4: 提交**

```bash
cd /home/zzr/lightly-train-main
git add -f datasets/convert_datasets/convert.py
git commit -m "feat(convert): register obj-wizard interactive entry"
```
末尾加:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## Task 6: README 更新

**Files:**
- Modify: `datasets/convert_datasets/README.md`

- [ ] **Step 1: 读 README,找到"物体版 MVTec AD"章节**

Run: `Read datasets/convert_datasets/README.md`

- [ ] **Step 2: 把该章节里"第0步/sample-browse"的描述改成用向导 + 预览图**

在"物体版 MVTec AD"章节里,把原来讲 `sample-browse` 的三步说明替换/补充为:

```markdown
推荐用交互向导一站式走完:

    python convert.py obj-wizard

- **第1步(生成标注预览)**:输入源 seg 数据集 + 输出目录,生成每张图的**标注预览**
  (左=原图画上缺陷多边形+中文类别名,右=信息栏:文件名、缺陷种类数、逐类计数、多边形总数)
  和 `sample_index.csv`(含 `n_defect_classes` 列,可先筛多类别图)。
- **人工分图**:把预览下载到本地,逐张看图判断属于哪个物体,建**中文物体名**文件夹分好
  (一张图只放一个物体;多物体/说不清的先别放),再把 `staging/` 上传回服务器。
- **第2步(转换)**:再跑一次向导选第2步,按文件名回源数据查标注、生成物体版 MVTec。

也可跳过向导直接用单命令:`python convert.py sample-browse`(生成预览)、
`python convert.py to-mvtec-obj`(转换)。中文字体默认用仓库自带 `tool_lib/msyh.ttc`,
可用 `--font` 覆盖。
```

(若原章节里 `sample-browse` 只出现在工具表/命令列表,保留它,只需在正文说明里加上向导用法即可。)

- [ ] **Step 3: 提交**

```bash
cd /home/zzr/lightly-train-main
git add -f datasets/convert_datasets/README.md
git commit -m "docs(convert): document obj-wizard + annotated preview usage"
```
末尾加:`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## 最终验证

- [ ] **全量测试通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **端到端冒烟:真出一张预览图看看**

Run:
```bash
cd /home/zzr/lightly-train-main/datasets/convert_datasets
python convert_tools/make_sample_yoloseg.py            # 造 sample_yoloseg/(含 images/)
python convert_tools/seg_sample_browse.py --src sample_yoloseg --out /tmp/obj_preview --limit 20
ls /tmp/obj_preview | head
```
Expected: `/tmp/obj_preview/*.png` 是拼了右侧信息栏的标注图,`sample_index.csv` 有 5 列。

---

## 自查清单(Spec 覆盖)

- 标注画在图上(多边形+中文名):Task 2 render_preview ✅
- 右侧丰富信息栏(种类数⚠/逐类计数/多边形数):Task 2 _render_panel ✅
- 颜色一致(class_color 固定调色板):Task 1 ✅
- 中文字体复用 tool_lib/msyh.ttc + --font + 兜底:Task 1 load_cjk_font、Task 3 --font ✅
- CSV 新增 n_defect_classes/n_polygons:Task 3 ✅
- 交互向导分两步 + 引导文案:Task 4 ✅
- convert.py 注册 obj-wizard:Task 5 ✅
- README:Task 6 ✅
- 越界类别不崩(预览尽量出图):Task 1 _class_name ✅
- 复制字体助手、不 import tool_lib.common:Task 1 ✅
- 删除被取代的旧 browse 测试:Task 3 Step 4b ✅
