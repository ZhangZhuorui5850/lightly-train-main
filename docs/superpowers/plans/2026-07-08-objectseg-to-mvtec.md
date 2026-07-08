# 物体版 YOLO-seg → MVTec AD 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把"人工按物体分好的图片文件夹" + "YOLO-seg 源数据集" 转成物体为中心的 MVTec AD 数据集(category=中文物体,物体内按缺陷分子文件夹)。

**Architecture:** 新工具 `objectseg_to_mvtec.py` 建全局 `stem→标签` 索引(查重名),遍历 `staging/<物体>/` 里的图,按文件名回源查标注、自动发现缺陷、写出 MVTec + 审计清单。复用 legacy `yoloseg_to_mvtec.py` 里的 `load_names/parse_label/make_mask`。旧工具保留不动。

**Tech Stack:** Python 3.10+、opencv-python(cv2)、numpy、pyyaml、tqdm、pytest。

参考 spec: `docs/superpowers/specs/2026-07-08-objectseg-to-mvtec-design.md`

---

## 文件结构

- Create: `datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py` — 物体版转换核心 + CLI
- Create: `datasets/convert_datasets/convert_tools/seg_sample_browse.py` — 可选探索工具(抽样摊图)
- Create: `datasets/convert_datasets/convert_tools/tests/__init__.py` — 空文件
- Create: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py` — 测试
- Modify: `datasets/convert_datasets/convert.py` — 注册新菜单项 + legacy 标注
- Modify: `datasets/convert_datasets/README.md` — 物体版用法

约定(与仓库现有代码一致):单行 tqdm 进度条;缺依赖时中文提示;`--src/--out/--clean` 风格;从不修改源数据。

---

## Task 0: 测试脚手架(fixtures)

**Files:**
- Create: `datasets/convert_datasets/convert_tools/tests/__init__.py`
- Create: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 建空 `__init__.py`**

```bash
: > datasets/convert_datasets/convert_tools/tests/__init__.py
```

- [ ] **Step 2: 写 fixtures 助手(测试文件顶部)**

```python
# datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import objectseg_to_mvtec as om  # noqa: E402

NAMES = ["锈蚀", "裂纹", "污迹"]  # cls 0,1,2


def _tri(cx: float, cy: float, r: float = 0.08):
    """一个小三角多边形(归一化坐标)。"""
    return [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]


def _write_label(p: Path, entries):
    """entries: list[(cls, [(x,y),...])];  空 list = good 图(空标签)。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, pts in entries:
        coords = " ".join(f"{x:.4f} {y:.4f}" for x, y in pts)
        lines.append(f"{cls} {coords}")
    p.write_text("\n".join(lines))


def make_src(root: Path) -> Path:
    """最小 YOLO-seg 源:只需要 data.yaml + labels/(images/ 不必要)。"""
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": NAMES}, allow_unicode=True), encoding="utf-8"
    )
    _write_label(root / "labels" / "train" / "a.txt", [(0, _tri(0.3, 0.3))])                 # 锈蚀
    _write_label(root / "labels" / "train" / "b.txt", [(0, _tri(0.3, 0.3)), (1, _tri(0.7, 0.7))])  # 锈蚀+裂纹
    _write_label(root / "labels" / "val" / "c.txt", [(2, _tri(0.5, 0.5))])                   # 污迹
    _write_label(root / "labels" / "train" / "d.txt", [])                                    # 空/good
    return root


def _img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.full((64, 64, 3), 128, np.uint8))


def make_staging(root: Path) -> Path:
    """物体 管道: a,b ; 物体 阀门: c,d。"""
    for stem in ("a", "b"):
        _img(root / "管道" / f"{stem}.jpg")
    for stem in ("c", "d"):
        _img(root / "阀门" / f"{stem}.jpg")
    return root
```

- [ ] **Step 3: 提交脚手架(先不含被测模块,后续任务补)**

暂不提交,进入 Task 1 一起提交(模块尚不存在,import 会失败)。

---

## Task 1: 全局标签索引 + 重名检测

**Files:**
- Create: `datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py`
- Test: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试**

```python
def test_build_label_index_maps_stem_to_split(tmp_path):
    src = make_src(tmp_path / "src")
    index = om.build_label_index(src)
    assert set(index) == {"a", "b", "c", "d"}
    assert index["a"][1] == "train"
    assert index["c"][1] == "val"
    assert index["a"][0].name == "a.txt"


def test_build_label_index_raises_on_duplicate_stem(tmp_path):
    src = make_src(tmp_path / "src")
    _write_label(src / "labels" / "test" / "a.txt", [(0, _tri(0.5, 0.5))])  # 与 train/a 重名
    with pytest.raises(om.DuplicateStemError):
        om.build_label_index(src)
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k build_label_index -v`
Expected: FAIL(`ModuleNotFoundError: objectseg_to_mvtec` 或 `AttributeError`)

- [ ] **Step 3: 写模块骨架 + `build_label_index`**

```python
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
from yoloseg_to_mvtec import IMG_EXTS, SPLITS, load_names, make_mask, parse_label  # noqa: E402

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
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k build_label_index -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py \
        datasets/convert_datasets/convert_tools/tests/
git commit -m "feat(convert): object-seg label index with duplicate-stem detection"
```

---

## Task 2: 扫描 staging → 物体→图 映射 + 跨物体冲突检测

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py`
- Test: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试**

```python
def test_scan_staging_lists_objects_and_images(tmp_path):
    staging = make_staging(tmp_path / "staging")
    objects, conflicts = om.scan_staging(staging)
    assert set(objects) == {"管道", "阀门"}
    assert sorted(p.stem for p in objects["管道"]) == ["a", "b"]
    assert conflicts == {}


def test_scan_staging_flags_stem_in_two_objects(tmp_path):
    staging = make_staging(tmp_path / "staging")
    _img(staging / "阀门" / "a.jpg")  # a 同时在 管道 和 阀门
    objects, conflicts = om.scan_staging(staging)
    assert "a" in conflicts
    assert set(conflicts["a"]) == {"管道", "阀门"}
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k scan_staging -v`
Expected: FAIL(`AttributeError: module ... has no attribute 'scan_staging'`)

- [ ] **Step 3: 实现 `scan_staging`(追加到模块)**

```python
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
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k scan_staging -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
git add datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py \
        datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): scan staging object folders with cross-object conflict detection"
```

---

## Task 3: 转换核心 —— 目录结构、一图多缺陷、mask 分拆、空标签→good

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py`
- Test: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试**

```python
def _run(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    result = om.convert(staging, src, out, clean=True, verbose=False)
    return out, result


def test_convert_builds_object_category_structure(tmp_path):
    out, _ = _run(tmp_path)
    # 单缺陷图 a → 管道/test/锈蚀
    assert (out / "管道" / "test" / "锈蚀" / "a.png").exists()
    assert (out / "管道" / "ground_truth" / "锈蚀" / "a_mask.png").exists()
    # 空目录保留
    assert (out / "管道" / "train" / "good").is_dir()
    assert (out / "阀门" / "train" / "good").is_dir()


def test_convert_duplicates_multidefect_image_into_each_defect(tmp_path):
    out, _ = _run(tmp_path)
    # b 同时有 锈蚀+裂纹 → 两个子文件夹都出现
    assert (out / "管道" / "test" / "锈蚀" / "b.png").exists()
    assert (out / "管道" / "test" / "裂纹" / "b.png").exists()
    assert (out / "管道" / "ground_truth" / "锈蚀" / "b_mask.png").exists()
    assert (out / "管道" / "ground_truth" / "裂纹" / "b_mask.png").exists()


def test_convert_mask_is_split_per_defect(tmp_path):
    out, _ = _run(tmp_path)
    m_rust = cv2.imread(str(out / "管道" / "ground_truth" / "锈蚀" / "b_mask.png"), cv2.IMREAD_GRAYSCALE)
    m_crack = cv2.imread(str(out / "管道" / "ground_truth" / "裂纹" / "b_mask.png"), cv2.IMREAD_GRAYSCALE)
    # 二值
    assert set(np.unique(m_rust)).issubset({0, 255})
    # 锈蚀在左上(0.3,0.3),裂纹在右下(0.7,0.7):各自区域有前景,对方区域没有
    assert m_rust[19, 19] == 255 and m_rust[45, 45] == 0     # (0.3*64≈19, 0.7*64≈45)
    assert m_crack[45, 45] == 255 and m_crack[19, 19] == 0


def test_convert_empty_label_goes_to_test_good(tmp_path):
    out, _ = _run(tmp_path)
    # d 是空标签 → 阀门/test/good
    assert (out / "阀门" / "test" / "good" / "d.png").exists()
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k convert -v`
Expected: FAIL(`AttributeError: ... 'convert'`)

- [ ] **Step 3: 实现 `convert`(追加到模块)**

```python
def _ensure_empty_layout(cat: Path) -> None:
    """建每个物体都要有的空目录(满足 MVTec 格式,即使零样本用途)。"""
    (cat / "train" / "good").mkdir(parents=True, exist_ok=True)
    (cat / "test" / "good").mkdir(parents=True, exist_ok=True)


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

    return {
        "stats": {o: dict(d) for o, d in stats.items()},
        "missing": missing,
        "conflicts": conflicts,
        "manifest": manifest,
    }
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k convert -v`
Expected: PASS(4 passed)

- [ ] **Step 5: 提交**

```bash
git add datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py \
        datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): object-centric MVTec build with per-defect mask split"
```

---

## Task 4: 审计清单 CSV + 统计/报告输出

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py`
- Test: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试**

```python
def test_convert_writes_manifest_csv(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    om.convert(staging, src, out, clean=True, verbose=False)

    manifest = out / "object_manifest.csv"
    assert manifest.exists()
    import csv as _csv
    rows = list(_csv.DictReader(manifest.open(encoding="utf-8")))
    by_stem = {r["stem"]: r for r in rows}
    assert by_stem["b"]["object"] == "管道"
    assert by_stem["b"]["defects"] == "锈蚀;裂纹"
    assert by_stem["c"]["orig_split"] == "val"
    assert by_stem["a"]["src_label"] == "labels/train/a.txt"
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k manifest -v`
Expected: FAIL(`object_manifest.csv` 不存在)

- [ ] **Step 3: 加清单写出 + 报告助手,并在 `convert` 末尾调用**

在模块里新增两个助手:

```python
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
```

在 `convert` 的 `return` 之前插入:

```python
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
```

并删除原来 Task 3 里 `convert` 结尾直接 `return {...}` 的那段(用上面这段替换)。

- [ ] **Step 4: 运行,确认通过(含之前所有测试)**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -v`
Expected: PASS(全部通过)

- [ ] **Step 5: 提交**

```bash
git add datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py \
        datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): write object_manifest.csv and print stats/conflict/skip report"
```

---

## Task 5: CLI(argparse + --clean)

**Files:**
- Modify: `datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py`
- Test: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`

- [ ] **Step 1: 写失败测试(用 subprocess 跑 CLI)**

```python
def test_cli_runs_end_to_end(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    import subprocess
    script = TOOLS / "objectseg_to_mvtec.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--staging", str(staging), "--src", str(src), "--out", str(out), "--clean"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "管道" / "test" / "锈蚀" / "a.png").exists()
    assert (out / "object_manifest.csv").exists()
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k cli -v`
Expected: FAIL(脚本无 `__main__` / 无参数解析,或 returncode != 0)

- [ ] **Step 3: 加 `main()` + `__main__`(追加到模块末尾)**

```python
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
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -v`
Expected: PASS(全部通过)

- [ ] **Step 5: 提交**

```bash
git add datasets/convert_datasets/convert_tools/objectseg_to_mvtec.py \
        datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
git commit -m "feat(convert): CLI for objectseg_to_mvtec with --clean"
```

---

## Task 6: 注册进 convert.py 菜单 + legacy 标注

**Files:**
- Modify: `datasets/convert_datasets/convert.py:53-66`(在 `to-mvtec` 项附近)

- [ ] **Step 1: 加新命令项**

在 `COMMANDS` 字典的 `to-mvtec` 项**之后**插入(缩进对齐现有项):

```python
    "to-mvtec-obj": Tool(
        "objectseg_to_mvtec.py",
        "物体版:按人工分好的物体文件夹 + 源seg 生成 MVTec(category=物体,内含各缺陷)",
        "转 MVTec AD",
        "需 --staging <物体文件夹根> --src <seg源> --out <输出> [--clean]",
        interactive=False,
    ),
```

- [ ] **Step 2: 给旧的缺陷版加 legacy 说明**

把现有 `to-mvtec` 项的 `desc` 改成明确它是**缺陷版**:

```python
    "to-mvtec": Tool(
        "seg2mvtec_interactive.py",
        "缺陷版(legacy):扫描 *seg 数据集,每个缺陷=一个 category,交互式转 MVTec AD",
        "转 MVTec AD",
        "回车=交互扫描并选择;或加 --all --yes 转全部",
        interactive=True,
    ),
```

- [ ] **Step 3: 冒烟测试菜单能列出**

Run: `cd datasets/convert_datasets && python convert.py list`
Expected: `[转 MVTec AD]` 分组下同时出现 `to-mvtec`(缺陷版 legacy)和 `to-mvtec-obj`(物体版)。

- [ ] **Step 4: 冒烟测试分发能跑起来**

Run: `cd datasets/convert_datasets && python convert.py to-mvtec-obj -h`
Expected: 打印 `objectseg_to_mvtec.py` 的 argparse 帮助(含 `--staging/--src/--out/--clean`)。

- [ ] **Step 5: 提交**

```bash
git add datasets/convert_datasets/convert.py
git commit -m "feat(convert): register object-version MVTec tool; mark defect-version as legacy"
```

---

## Task 7: 探索工具 `seg_sample_browse.py`(抽样摊图给人定物体)

**Files:**
- Create: `datasets/convert_datasets/convert_tools/seg_sample_browse.py`
- Test: `datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py`(追加)

- [ ] **Step 1: 写失败测试**

```python
def test_sample_browse_writes_index_csv(tmp_path):
    src = make_src(tmp_path / "src")
    # 需要真实图片供抽样:补 images/
    for split, stem in [("train", "a"), ("train", "b"), ("val", "c"), ("train", "d")]:
        _img(src / "images" / split / f"{stem}.jpg")
    out = tmp_path / "browse"
    import sys as _sys
    _sys.path.insert(0, str(TOOLS))
    import seg_sample_browse as sb
    n = sb.browse(src, out, limit=10)
    assert n == 4
    idx = out / "sample_index.csv"
    assert idx.exists()
    import csv as _csv
    rows = {r["stem"]: r for r in _csv.DictReader(idx.open(encoding="utf-8"))}
    assert rows["b"]["defects"] == "锈蚀;裂纹"
    # 图片被复制出来供人肉眼看
    assert (out / "b.jpg").exists()
```

- [ ] **Step 2: 运行,确认失败**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -k sample_browse -v`
Expected: FAIL(`ModuleNotFoundError: seg_sample_browse`)

- [ ] **Step 3: 实现 `seg_sample_browse.py`**

```python
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
```

- [ ] **Step 4: 运行,确认通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/test_objectseg_to_mvtec.py -v`
Expected: PASS(全部通过)

- [ ] **Step 5: 在 convert.py 菜单登记(可选工具,归到"更多")**

在 `COMMANDS` 里加(`primary=False`):

```python
    "sample-browse": Tool(
        "seg_sample_browse.py",
        "物体版第0步(可选):抽样摊图,帮你归纳源数据里有哪些物体",
        "更多",
        "需 --src <seg源> --out <浏览目录> [--limit 500]",
        interactive=False, primary=False,
    ),
```

- [ ] **Step 6: 提交**

```bash
git add datasets/convert_datasets/convert_tools/seg_sample_browse.py \
        datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py \
        datasets/convert_datasets/convert.py
git commit -m "feat(convert): optional seg_sample_browse discovery tool"
```

---

## Task 8: README 更新(物体版流水线用法)

**Files:**
- Modify: `datasets/convert_datasets/README.md`

- [ ] **Step 1: 读现有 README,找到 MVTec 相关章节**

Run: `sed -n '1,200p' datasets/convert_datasets/README.md`

- [ ] **Step 2: 追加"物体版 MVTec AD"小节**

在 MVTec 相关内容附近追加(若无则加到末尾):

```markdown
## 物体版 MVTec AD(推荐,category=物体)

适用:多模态/零样本异常检测。category 是**物体**,物体内按缺陷分子文件夹。

三步:
1.(可选)抽样定物体:
   `python convert.py sample-browse` → 看 `<out>/sample_index.csv` 归纳有哪些物体。
2. 人工分图:在 `staging/` 下建**中文物体名**文件夹,把对应图片放进去(一张图只放一个物体)。
3. 生成:
   `python convert.py to-mvtec-obj`(或直接
   `python convert_tools/objectseg_to_mvtec.py --staging <staging> --src <seg源> --out <输出> --clean`)

产出 `<out>/<物体>/{train/good(空), test/good(空), test/<缺陷>/, ground_truth/<缺陷>/}`,
外加 `<out>/object_manifest.csv` 审计清单。脚本按文件名回源查标注、自动发现缺陷、一图多缺陷会复制进各缺陷子文件夹(mask 分拆)。

> 旧的 `to-mvtec`(缺陷版,每个缺陷=一个 category)保留为 legacy。
> 说明:`test/good` 为空 → 图像级 AUROC 无法计算,仅支持像素级定位(符合零样本用途)。
```

- [ ] **Step 3: 提交**

```bash
git add datasets/convert_datasets/README.md
git commit -m "docs(convert): document object-version MVTec pipeline"
```

---

## 最终验证

- [ ] **全量测试通过**

Run: `cd datasets/convert_datasets/convert_tools && python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **端到端冒烟(用真实小数据)**

Run:
```bash
cd datasets/convert_datasets
python convert_tools/make_sample_yoloseg.py            # 造 sample_yoloseg/(有 images/)
mkdir -p /tmp/obj_staging/表面 && cp sample_yoloseg/images/train/*.jpg /tmp/obj_staging/表面/ 2>/dev/null
python convert.py to-mvtec-obj --staging /tmp/obj_staging --src sample_yoloseg --out /tmp/obj_mvtec --clean
ls -R /tmp/obj_mvtec | head -40
```
Expected: `/tmp/obj_mvtec/表面/test/<缺陷>/*.png`、`ground_truth/<缺陷>/*_mask.png`、`object_manifest.csv` 都在。

---

## 自查清单(Spec 覆盖)

- 三段式流水线:Task 7(探索)+ 人工分图(用户)+ Task 1-5(生成) ✅
- 物体=category(中文):Task 3 ✅
- 全局标签索引 + 重名检测:Task 1 ✅
- staging 扫描 + 跨物体冲突:Task 2 ✅
- 一图多缺陷复制 + mask 分拆:Task 3 ✅
- 空标签→test/good、train/good 留空:Task 3 ✅
- object_manifest.csv + 统计/报告:Task 4 ✅
- CLI + --clean:Task 5 ✅
- convert.py 菜单 + legacy 标注:Task 6、Task 7 Step 5 ✅
- README:Task 8 ✅
- 交付局限说明(test/good 空):Task 8 Step 2 ✅
