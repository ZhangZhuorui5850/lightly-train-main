# 尺寸感知导出（size-aware export）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 det/seg 导出在选图与划分阶段把"目标尺寸占比（小/中/大）+ 平均框密度"作为优化目标，默认开启 `30:40:30`，可一键退回旧行为。

**Architecture:** 双引擎——`size_ratio` 留空走现有 CELF（零行为变化），填了则切到分批赤字贪心（复用 `det_size_supplement` 的打分结构），把类别赤字 + 尺寸桶赤字 + 平均密度偏离联合加权。尺寸桶扩成 4 档（tiny/small/medium/large），tiny`<16²` 只统计不进比例。划分阶段给每个 split 加尺寸桶目标，类别覆盖优先、尺寸作附加权重。seg 复用同一份 det_analysis 选图/划分逻辑，只在面积口径上用掩码面积。

**Tech Stack:** Python 3.10+，pytest，Pillow（已用），YOLO 格式数据集。测试用合成 100×100 数据集（已有范式见 `tool_lib/tests/test_det_size_supplement.py`）。

**运行环境：** WSL 下 `lightlytrain` conda 环境。测试命令统一用 `conda run -n lightlytrain python -m pytest ...`（在 WSL shell 内执行）。

**设计依据：** `docs/superpowers/specs/2026-06-03-size-aware-export-design.md`

---

## File Structure

| 文件 | 角色 | 改动 |
|---|---|---|
| `tool_lib/det_export.py` | det 4 档分桶常量/函数、候选尺寸扫描、参数透传、报告 | 修改 |
| `tool_lib/seg_export.py` | seg 4 档分桶、掩码面积候选扫描、参数透传 | 修改 |
| `tool_lib/det_size_supplement.py` | 4 档 `BUCKET_NAMES`/`_empty_buckets`（被 det_export 复用） | 修改 |
| `tool_lib/det_analysis.py` | 尺寸赤字+密度打分、分批贪心引擎、双引擎开关、split 尺寸目标 | 修改 |
| `tool_lib/common.py` | det/seg 新参数的默认常量与 `settings.get` 读取 | 修改 |
| `tool_lib/interactive.py` | det/seg 新参数的 argparse、interactive args、preview 打印 | 修改 |
| `launcher.py` | det/seg 新参数默认值 + 注释 | 修改 |
| `tool_lib/tests/test_size_buckets.py` | 4 档分桶边界测试 | 新建 |
| `tool_lib/tests/test_size_aware_selection.py` | 尺寸赤字/密度打分 + 分批贪心 + split 单测 | 新建 |
| `tool_lib/tests/test_size_aware_export_det.py` | det 端到端集成测试 + 回归保护 | 新建 |
| `tool_lib/tests/test_size_aware_export_seg.py` | seg 端到端集成测试 | 新建 |

**约定：** 4 档桶顺序固定 `("tiny","small","medium","large")`；比例桶 = `("small","medium","large")`（排除 tiny）。新增模块级常量集中放 `det_analysis.py`，seg 通过 import 复用。

---

## Phase 0 — 4 档分桶基础

### Task 1: 分桶阈值与 4 档分桶函数

**Files:**
- Modify: `tool_lib/det_export.py:45-46`（阈值常量）、`tool_lib/det_export.py:156-175`（`_empty_size_bucket_counts` / `_bucket_box_area`）
- Modify: `tool_lib/det_size_supplement.py:30-35`（`BUCKET_NAMES` / `_empty_buckets`）
- Modify: `tool_lib/seg_export.py:190-205`（seg 镜像）
- Test: `tool_lib/tests/test_size_buckets.py`

- [ ] **Step 1: 写失败测试**

新建 `tool_lib/tests/test_size_buckets.py`：

```python
"""4 档尺寸分桶边界测试。"""
from __future__ import annotations

from tool_lib.det_export import _bucket_box_area, _empty_size_bucket_counts
from tool_lib.seg_export import _bucket_mask_area
from tool_lib import det_size_supplement as ss


def test_bucket_boundaries_box():
    # 阈值：tiny<16²(256), small<32²(1024), medium<96²(9216), large≥96²
    assert _bucket_box_area(255.0) == "tiny"
    assert _bucket_box_area(256.0) == "small"       # 16²
    assert _bucket_box_area(1023.0) == "small"
    assert _bucket_box_area(1024.0) == "medium"     # 32²
    assert _bucket_box_area(9215.0) == "medium"
    assert _bucket_box_area(9216.0) == "large"      # 96²


def test_bucket_boundaries_mask():
    assert _bucket_mask_area(255.0) == "tiny"
    assert _bucket_mask_area(256.0) == "small"
    assert _bucket_mask_area(1024.0) == "medium"
    assert _bucket_mask_area(9216.0) == "large"


def test_empty_buckets_have_four_keys():
    assert list(_empty_size_bucket_counts().keys()) == ["tiny", "small", "medium", "large"]
    assert list(ss._empty_buckets().keys()) == ["tiny", "small", "medium", "large"]
    assert ss.BUCKET_NAMES == ("tiny", "small", "medium", "large")
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_buckets.py -v`
Expected: FAIL（tiny 不存在 / `_empty_buckets` 只有 3 键）

- [ ] **Step 3: 实现 4 档**

在 `tool_lib/det_export.py:45-46` 阈值后补 tiny 阈值：

```python
TINY_OBJECT_AREA_THRESHOLD = 16.0 * 16.0
SMALL_OBJECT_AREA_THRESHOLD = 32.0 * 32.0
MEDIUM_OBJECT_AREA_THRESHOLD = 96.0 * 96.0

BUCKET_NAMES = ("tiny", "small", "medium", "large")
RATIO_BUCKET_NAMES = ("small", "medium", "large")
```

替换 `det_export.py` 的 `_empty_size_bucket_counts` 与 `_bucket_box_area`：

```python
def _empty_size_bucket_counts() -> dict[str, int]:
    return {name: 0 for name in BUCKET_NAMES}


def _bucket_box_area(area_pixels: float) -> str:
    if area_pixels < TINY_OBJECT_AREA_THRESHOLD:
        return "tiny"
    if area_pixels < SMALL_OBJECT_AREA_THRESHOLD:
        return "small"
    if area_pixels < MEDIUM_OBJECT_AREA_THRESHOLD:
        return "medium"
    return "large"
```

在 `tool_lib/seg_export.py`：把 `_empty_size_bucket_counts` 改为 `{name: 0 for name in ("tiny","small","medium","large")}`，并在 `_bucket_mask_area`（:200-205）开头加 `if area_pixels < 16.0*16.0: return "tiny"`（阈值常量 `SMALL_OBJECT_AREA_THRESHOLD` 等 seg 已有；新增 `TINY_OBJECT_AREA_THRESHOLD = 16.0*16.0`）。

在 `tool_lib/det_size_supplement.py:30-35`：

```python
BUCKET_NAMES = ("tiny", "small", "medium", "large")


def _empty_buckets() -> dict[str, int]:
    return {name: 0 for name in BUCKET_NAMES}
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_buckets.py -v`
Expected: PASS

- [ ] **Step 5: 跑既有 size-supplement 测试确保没破坏**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_det_size_supplement.py -v`
Expected: PASS（若有断言写死 3 键比例分母，改测试用 `RATIO_BUCKET_NAMES`；tiny 默认 0 不影响既有合成数据）

- [ ] **Step 6: Commit**

```bash
git add tool_lib/det_export.py tool_lib/seg_export.py tool_lib/det_size_supplement.py tool_lib/tests/test_size_buckets.py
git commit -m "feat(export): 4-bucket sizing with tiny(<16²) separate"
```

---

## Phase 1 — det_analysis：尺寸/密度打分 + 分批贪心引擎

### Task 2: 尺寸占比与密度的纯函数打分

**Files:**
- Modify: `tool_lib/det_analysis.py`（在 `score_export_candidate` 附近，约 :1201 之后新增）
- Test: `tool_lib/tests/test_size_aware_selection.py`

- [ ] **Step 1: 写失败测试**

新建 `tool_lib/tests/test_size_aware_selection.py`：

```python
"""尺寸感知选图：纯函数打分 + 分批贪心 + split 尺寸目标。"""
from __future__ import annotations

from tool_lib.det_analysis import (
    ratio_bucket_props,
    size_deficit_score,
    density_steer_term,
)


def test_ratio_bucket_props_excludes_tiny():
    # tiny 不进分母：small=2 medium=1 large=1 tiny=10 → small 0.5
    props = ratio_bucket_props({"tiny": 10, "small": 2, "medium": 1, "large": 1})
    assert props["small"] == 0.5
    assert props["medium"] == 0.25
    assert props["large"] == 0.25


def test_size_deficit_score_rewards_deficit_bucket():
    target = {"small": 0.3, "medium": 0.4, "large": 0.3}
    # 当前全是 large → small/medium 有赤字
    current = {"tiny": 0, "small": 0, "medium": 0, "large": 10}
    cand_small = {"tiny": 0, "small": 3, "medium": 0, "large": 0}
    cand_large = {"tiny": 0, "small": 0, "medium": 0, "large": 3}
    s_small = size_deficit_score(cand_small, current, target, over_penalty=1.0)
    s_large = size_deficit_score(cand_large, current, target, over_penalty=1.0)
    assert s_small > s_large  # 缺小目标时小目标图得分更高


def test_density_steer_term_direction():
    # 平均偏低(2)→鼓励多框图(+)，偏高(20)→鼓励少框图
    low = density_steer_term(total_boxes=8, current_avg=2.0, lo=5.0, hi=15.0)
    high = density_steer_term(total_boxes=8, current_avg=20.0, lo=5.0, hi=15.0)
    inband = density_steer_term(total_boxes=8, current_avg=10.0, lo=5.0, hi=15.0)
    assert low > 0.0
    assert high < 0.0
    assert inband == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py -v`
Expected: FAIL（函数未定义）

- [ ] **Step 3: 实现纯函数**

在 `tool_lib/det_analysis.py` 顶部 import 区加 `from .det_export import RATIO_BUCKET_NAMES`（注意循环依赖：det_export 已 import det_analysis。若触发循环，改为在 det_analysis 内本地定义 `RATIO_BUCKET_NAMES = ("small","medium","large")` 常量，不 import）。采用**本地定义**避免循环：

```python
RATIO_BUCKET_NAMES = ("small", "medium", "large")


def ratio_bucket_props(buckets: dict[str, int]) -> dict[str, float]:
    """small/medium/large 占比（分母排除 tiny）。"""
    total = sum(int(buckets.get(name, 0)) for name in RATIO_BUCKET_NAMES)
    if total <= 0:
        return {name: 0.0 for name in RATIO_BUCKET_NAMES}
    return {name: buckets.get(name, 0) / total for name in RATIO_BUCKET_NAMES}


def size_deficit_score(
    cand_buckets: dict[str, int],
    current_buckets: dict[str, int],
    target_ratio: dict[str, float],
    *,
    over_penalty: float = 1.0,
) -> float:
    """候选图对"当前亏空尺寸桶"的贡献减去对超标桶的惩罚（赤字驱动）。"""
    props = ratio_bucket_props(current_buckets)
    score = 0.0
    for name in RATIO_BUCKET_NAMES:
        deficit = max(0.0, target_ratio.get(name, 0.0) - props[name])
        over = max(0.0, props[name] - target_ratio.get(name, 0.0))
        boxes = cand_buckets.get(name, 0)
        score += deficit * boxes - over_penalty * over * boxes
    return score


def density_steer_term(
    *,
    total_boxes: int,
    current_avg: float,
    lo: float,
    hi: float,
) -> float:
    """平均框数软导向：低于 lo 奖励多框图，高于 hi 奖励少框图，带内为 0。

    返回带符号的方向项，量纲为"框数"，由调用方乘以内置权重后并入总分。
    """
    if lo <= 0.0 and hi <= 0.0:
        return 0.0
    if current_avg < lo:
        return float(total_boxes)
    if hi > 0.0 and current_avg > hi:
        return -float(total_boxes)
    return 0.0
```

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tool_lib/det_analysis.py tool_lib/tests/test_size_aware_selection.py
git commit -m "feat(export): size-deficit and density-steer scoring primitives"
```

---

### Task 3: 分批赤字贪心引擎 + 双引擎开关

**Files:**
- Modify: `tool_lib/det_analysis.py`（新增 `_select_size_aware_candidates`；改 `select_balanced_train_candidates` 签名加可选参数与分支）
- Test: `tool_lib/tests/test_size_aware_selection.py`

- [ ] **Step 1: 追加失败测试**

在 `test_size_aware_selection.py` 追加：

```python
from pathlib import Path
from tool_lib.det_shared import ExportImageCandidate
from tool_lib.det_analysis import select_balanced_train_candidates


def _cand(stem, class_box_counts):
    return ExportImageCandidate(
        split_name="train",
        rel_split_image_dir=Path("images/train"),
        rel_split_label_dir=Path("labels/train"),
        rel_path=Path(f"{stem}.jpg"),
        src_image_path=Path(f"/src/{stem}.jpg"),
        src_label_path=Path(f"/src/{stem}.txt"),
        filtered_lines=tuple(),
        class_box_counts=class_box_counts,
    )


def test_size_aware_selection_moves_ratio_toward_target():
    # 10 张大目标图 + 10 张小目标图，目标小占比高 → 应多选小目标图
    cands, buckets = [], {}
    for i in range(10):
        c = _cand(f"big{i}", {0: 3})
        cands.append(c)
        buckets[c.rel_path.as_posix() + "|train"] = {"tiny": 0, "small": 0, "medium": 0, "large": 3}
    for i in range(10):
        c = _cand(f"small{i}", {0: 3})
        cands.append(c)
        buckets[c.rel_path.as_posix() + "|train"] = {"tiny": 0, "small": 3, "medium": 0, "large": 0}

    selected, summary = select_balanced_train_candidates(
        candidates=cands,
        kept_class_ids=[0],
        target_total_images=10,
        target_boxes_per_class=0,
        balance_ratio=0.0,
        target_images_per_class=0,
        box_density_penalty=0.0,
        size_buckets_by_candidate=buckets,
        target_size_ratio={"small": 0.7, "medium": 0.0, "large": 0.3},
        size_balance_weight=1.0,
    )
    n_small = sum(1 for c in selected if c.rel_path.as_posix().startswith("small"))
    assert n_small >= 6  # 偏向小目标
    assert summary["selection_algorithm"] == "size_aware_greedy"


def test_size_ratio_off_uses_legacy_celf():
    c = _cand("a", {0: 2})
    selected, summary = select_balanced_train_candidates(
        candidates=[c],
        kept_class_ids=[0],
        target_total_images=0,
        target_boxes_per_class=0,
        balance_ratio=0.0,
        target_images_per_class=0,
        box_density_penalty=0.0,
        size_buckets_by_candidate=None,
        target_size_ratio=None,
        size_balance_weight=0.0,
    )
    assert summary["selection_algorithm"] == "celf_lazy_greedy"
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py::test_size_aware_selection_moves_ratio_toward_target -v`
Expected: FAIL（`select_balanced_train_candidates` 不接受 `size_buckets_by_candidate`）

- [ ] **Step 3: 加签名与分支**

在 `select_balanced_train_candidates`（:1422）签名末尾追加参数（都带默认值，保证旧调用不变）：

```python
    size_buckets_by_candidate: dict[str, dict[str, int]] | None = None,
    target_size_ratio: dict[str, float] | None = None,
    size_balance_weight: float = 0.0,
    avg_boxes_per_image_min: float = 0.0,
    avg_boxes_per_image_max: float = 0.0,
```

在函数体最前面加分支（在 `effective_target_total_images` 计算之前）：

```python
    if target_size_ratio:
        return _select_size_aware_candidates(
            candidates=candidates,
            kept_class_ids=kept_class_ids,
            target_total_images=target_total_images,
            target_boxes_per_class=target_boxes_per_class,
            balance_ratio=balance_ratio,
            target_images_per_class=target_images_per_class,
            box_density_penalty=box_density_penalty,
            size_buckets_by_candidate=size_buckets_by_candidate or {},
            target_size_ratio=target_size_ratio,
            size_balance_weight=size_balance_weight,
            avg_boxes_per_image_min=avg_boxes_per_image_min,
            avg_boxes_per_image_max=avg_boxes_per_image_max,
            progress_callback=progress_callback,
        )
```

新增 `_select_size_aware_candidates`（放在 `select_balanced_train_candidates` 之后）。复用 Task 2 的纯函数，分批结构对齐 `det_size_supplement.select_supplement_candidates`。`DENSITY_STEER_WEIGHT = 0.5` 为内置常数（不暴露，符合 spec §5.4）：

```python
DENSITY_STEER_WEIGHT = 0.5


def _candidate_size_key(candidate: ExportImageCandidate) -> str:
    return candidate.rel_path.as_posix() + "|" + candidate.split_name


def _select_size_aware_candidates(
    *,
    candidates: list[ExportImageCandidate],
    kept_class_ids: list[int],
    target_total_images: int,
    target_boxes_per_class: int,
    balance_ratio: float,
    target_images_per_class: int,
    box_density_penalty: float,
    size_buckets_by_candidate: dict[str, dict[str, int]],
    target_size_ratio: dict[str, float],
    size_balance_weight: float,
    avg_boxes_per_image_min: float,
    avg_boxes_per_image_max: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
    batch_size: int = 200,
) -> tuple[list[ExportImageCandidate], dict[str, Any]]:
    """分批赤字贪心：类别赤字 + size_balance_weight·尺寸赤字 + 密度软导向。

    每批按当前已选状态重算尺寸占比与平均框数，对剩余候选打分取 top，直到达
    target_total_images（>0）或候选耗尽。target_total_images<=0 时退化为"取尽
    所有正分候选"。
    """
    available_box_counts = count_candidate_boxes_per_class(candidates, kept_class_ids)
    n = len(candidates)
    target_n = target_total_images if target_total_images > 0 else n

    selected: list[ExportImageCandidate] = []
    selected_box_counts = {class_id: 0 for class_id in kept_class_ids}
    current_buckets = {name: 0 for name in ("tiny", "small", "medium", "large")}
    total_selected_boxes = 0
    remaining = list(candidates)
    chosen_keys: set[str] = set()

    def _class_deficit_score(c: ExportImageCandidate) -> float:
        s = 0.0
        for class_id, cnt in c.class_box_counts.items():
            avail = max(available_box_counts.get(class_id, 0), 1)
            s += cnt / avail
        return s

    def _score(c: ExportImageCandidate) -> float:
        key = _candidate_size_key(c)
        buckets = size_buckets_by_candidate.get(key, {})
        class_score = _class_deficit_score(c)
        size_score = size_deficit_score(buckets, current_buckets, target_size_ratio)
        cur_avg = (total_selected_boxes / len(selected)) if selected else 0.0
        density = density_steer_term(
            total_boxes=c.total_boxes,
            current_avg=cur_avg,
            lo=avg_boxes_per_image_min,
            hi=avg_boxes_per_image_max,
        )
        return class_score + size_balance_weight * size_score + DENSITY_STEER_WEIGHT * density

    while remaining and len(selected) < target_n:
        ranked = sorted(remaining, key=lambda c: (-_score(c), c.total_boxes, c.rel_path.as_posix()))
        took_any = False
        for c in ranked:
            if len(selected) >= target_n or (len(selected) % batch_size == 0 and took_any):
                break
            key = _candidate_size_key(c)
            buckets = size_buckets_by_candidate.get(key, {})
            for name in ("tiny", "small", "medium", "large"):
                current_buckets[name] += buckets.get(name, 0)
            for class_id, cnt in c.class_box_counts.items():
                selected_box_counts[class_id] += cnt
            total_selected_boxes += c.total_boxes
            selected.append(c)
            chosen_keys.add(key)
            took_any = True
        remaining = [c for c in remaining if _candidate_size_key(c) not in chosen_keys]
        if not took_any:
            break
        if progress_callback is not None:
            progress_callback(min(len(selected), target_n), max(target_n, 1),
                              f"size-aware pick={len(selected)}/{target_n}")

    selected.sort(key=lambda item: item.rel_path.as_posix())
    achieved = ratio_bucket_props(current_buckets)
    summary = {
        "selection_algorithm": "size_aware_greedy",
        "available_total_images": n,
        "effective_target_total_images": target_n,
        "selected_boxes_per_class": selected_box_counts,
        "achieved_size_buckets": dict(current_buckets),
        "achieved_size_ratio": achieved,
        "target_size_ratio": dict(target_size_ratio),
        "size_balance_weight": size_balance_weight,
        "avg_boxes_per_image_achieved": (total_selected_boxes / len(selected)) if selected else 0.0,
        "avg_boxes_per_image_min": avg_boxes_per_image_min,
        "avg_boxes_per_image_max": avg_boxes_per_image_max,
    }
    return selected, summary
```

> 注：`count_candidate_boxes_per_class` 已存在于 det_analysis；`Callable`/`Any` 已 import。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py -v`
Expected: PASS（两个新测试 + Task 2 的都过）

- [ ] **Step 5: Commit**

```bash
git add tool_lib/det_analysis.py tool_lib/tests/test_size_aware_selection.py
git commit -m "feat(export): size-aware batched greedy selection engine + dual-engine switch"
```

---

### Task 4: split 尺寸桶目标 + 划分集成（类别优先）

**Files:**
- Modify: `tool_lib/det_analysis.py`（新增 `allocate_size_bucket_targets_per_split`；`assign_candidates_to_new_splits` 加可选 size 参数与附加打分项）
- Test: `tool_lib/tests/test_size_aware_selection.py`

- [ ] **Step 1: 追加失败测试**

```python
from tool_lib.det_analysis import allocate_size_bucket_targets_per_split


def test_split_size_targets_proportional_and_exclude_tiny():
    cands = [_cand(f"c{i}", {0: 1}) for i in range(10)]
    buckets = {}
    for i, c in enumerate(cands):
        # 每图 1 个 small 框；外加前 5 张各 1 个 tiny（tiny 不计目标）
        b = {"tiny": 1 if i < 5 else 0, "small": 1, "medium": 0, "large": 0}
        buckets[c.rel_path.as_posix() + "|train"] = b
    targets = allocate_size_bucket_targets_per_split(
        selected_candidates=cands,
        size_buckets_by_candidate=buckets,
        split_image_targets={"train": 8, "val": 1, "test": 1},
    )
    # small 总数 10，按 8:1:1 → train 8 / val 1 / test 1
    assert targets["train"]["small"] == 8
    assert targets["val"]["small"] == 1
    assert targets["test"]["small"] == 1
    # tiny 不进目标
    assert "tiny" not in targets["train"]
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py::test_split_size_targets_proportional_and_exclude_tiny -v`
Expected: FAIL（函数未定义）

- [ ] **Step 3: 实现 split 尺寸目标 + 接入划分**

新增（用与 `allocate_box_targets_per_split` 相同的 Hare-Niemeyer 余数法）：

```python
def allocate_size_bucket_targets_per_split(
    *,
    selected_candidates: list[ExportImageCandidate],
    size_buckets_by_candidate: dict[str, dict[str, int]],
    split_image_targets: dict[str, int],
) -> dict[str, dict[str, int]]:
    """按 split 图数比例给每 split 分配 small/medium/large 目标框数（排除 tiny）。"""
    desired: dict[str, dict[str, int]] = {
        split_name: {name: 0 for name in RATIO_BUCKET_NAMES}
        for split_name in ("train", "val", "test")
    }
    total_images = len(selected_candidates)
    if total_images <= 0:
        return desired
    bucket_totals = {name: 0 for name in RATIO_BUCKET_NAMES}
    for c in selected_candidates:
        b = size_buckets_by_candidate.get(_candidate_size_key(c), {})
        for name in RATIO_BUCKET_NAMES:
            bucket_totals[name] += int(b.get(name, 0))
    for name in RATIO_BUCKET_NAMES:
        total = bucket_totals[name]
        raw = {
            s: total * split_image_targets.get(s, 0) / total_images
            for s in ("train", "val", "test")
        }
        alloc = {s: int(raw[s]) for s in raw}
        assigned = sum(alloc.values())
        rema = sorted(((raw[s] - alloc[s], s) for s in ("train", "val", "test")))
        while assigned < total:
            _, s = rema.pop()
            alloc[s] += 1
            assigned += 1
            rema.append((0.0, s))
            rema.sort()
        for s in ("train", "val", "test"):
            desired[s][name] = alloc[s]
    return desired
```

在 `assign_candidates_to_new_splits`（:1897）签名末尾加可选参数：

```python
    size_buckets_by_candidate: dict[str, dict[str, int]] | None = None,
```

在函数内构造 size 目标与每 split 已分配尺寸计数（仅当传入时启用）：

```python
    size_targets_by_split = (
        allocate_size_bucket_targets_per_split(
            selected_candidates=selected_candidates,
            size_buckets_by_candidate=size_buckets_by_candidate,
            split_image_targets=split_image_targets,
        )
        if size_buckets_by_candidate
        else None
    )
    assigned_size_counts = {
        s: {name: 0 for name in RATIO_BUCKET_NAMES} for s in ("train", "val", "test")
    }
```

在主分配循环对每个 (candidate, split) 计算分数处，**在类别覆盖判定之后**追加 size_gain 作为 tie-break 级附加项（类别优先不变）：找到给某 split 打分的地方（`coverage_gain` 计算附近，约 :1959），把 size 增益并入次级排序键，使尺寸只在类别覆盖同等时起作用。最小改法——在选中 candidate 写入 split 后更新计数：

```python
        if size_buckets_by_candidate:
            b = size_buckets_by_candidate.get(_candidate_size_key(candidate), {})
            for name in RATIO_BUCKET_NAMES:
                assigned_size_counts[best_split][name] += int(b.get(name, 0))
```

并在候选-split 评分的 tie-break（现有 `best_score` 比较链）里，对"该 split 仍亏空的尺寸桶"给一个小附加分：

```python
        def _size_gain(cand, split_name):
            if not size_targets_by_split:
                return 0.0
            b = size_buckets_by_candidate.get(_candidate_size_key(cand), {})
            g = 0.0
            for name in RATIO_BUCKET_NAMES:
                if assigned_size_counts[split_name][name] < size_targets_by_split[split_name][name]:
                    g += int(b.get(name, 0))
            return g
```

把 `_size_gain(candidate, split_name)` 作为**低于类别覆盖、高于 path 排序**的次级键并入现有比较（量级远小于 coverage，不改变类别覆盖优先级）。

> 实现注意：保持 `size_buckets_by_candidate=None` 时该函数行为与现状逐字节一致（不进任何新分支）。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tool_lib/det_analysis.py tool_lib/tests/test_size_aware_selection.py
git commit -m "feat(export): per-split size-bucket targets with class-priority assignment"
```

---

## Phase 2 — det_export 接线

### Task 5: 候选尺寸桶扫描（带缓存）

**Files:**
- Modify: `tool_lib/det_export.py`（新增 `scan_candidate_size_buckets`）
- Test: `tool_lib/tests/test_size_aware_selection.py`

- [ ] **Step 1: 追加失败测试**

```python
def test_scan_candidate_size_buckets(tmp_path):
    from pathlib import Path as _P
    from tool_lib import common as rt
    from tool_lib.det_export import scan_candidate_size_buckets
    from tool_lib.det_shared import ExportImageCandidate
    rt.import_runtime_dependencies()
    img = tmp_path / "a.jpg"
    rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(img)
    # 一个 small 框(20x20=400px∈[256,1024)) + 一个 large 框(50x50=2500? 不,要≥9216) → 用 100x100? 归一化
    # 100x100 图：归一化 0.2 → 20px，面积 400 → small；归一化 1.0×1.0 → 10000px → large
    cand = ExportImageCandidate(
        split_name="train",
        rel_split_image_dir=_P("images/train"),
        rel_split_label_dir=_P("labels/train"),
        rel_path=_P("a.jpg"),
        src_image_path=img,
        src_label_path=tmp_path / "a.txt",
        filtered_lines=("0 0.5 0.5 0.2 0.2", "0 0.5 0.5 1.0 1.0"),
        class_box_counts={0: 2},
    )
    out = scan_candidate_size_buckets([cand], {0: 0}, cache_dir=tmp_path)
    key = "a.jpg|train"
    assert out[key]["small"] == 1
    assert out[key]["large"] == 1
    assert out[key]["tiny"] == 0
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py::test_scan_candidate_size_buckets -v`
Expected: FAIL（`scan_candidate_size_buckets` 未定义）

- [ ] **Step 3: 实现扫描（复用 ImageSizeCache + bucket_label_lines）**

在 `tool_lib/det_export.py` 新增（import `from .det_size_supplement import ImageSizeCache, bucket_label_lines`；注意 det_size_supplement 已 import det_export，需在函数内局部 import 避免顶层循环）：

```python
def scan_candidate_size_buckets(
    candidates: list[Any],
    class_id_mapping: dict[int, int],
    *,
    cache_dir: Path,
) -> dict[str, dict[str, int]]:
    """对候选逐图算 tiny/small/medium/large 框数，键为 rel_path|split。带 mtime 缓存。"""
    from .det_size_supplement import ImageSizeCache, bucket_label_lines
    cache = ImageSizeCache(Path(cache_dir) / ".imgsize_cache.json")
    result: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        width, height = cache.get(candidate.src_image_path)
        buckets = bucket_label_lines(candidate.filtered_lines, width=width, height=height)
        result[candidate.rel_path.as_posix() + "|" + candidate.split_name] = buckets
    cache.save()
    return result
```

> `bucket_label_lines` 用 `_bucket_box_area`（Task 1 已 4 档），故输出含 tiny。`class_id_mapping` 形参保留以便将来按重映射后类别细分（当前不需要，传入即可）。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_selection.py::test_scan_candidate_size_buckets -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tool_lib/det_export.py tool_lib/tests/test_size_aware_selection.py
git commit -m "feat(export): cached candidate size-bucket scan"
```

---

### Task 6: 在 det 导出主管线串起尺寸感知

**Files:**
- Modify: `tool_lib/det_export.py`（`export_filtered_dataset` / `_export_filtered_dataset_impl` 加参数；选图与划分调用处传入；`run_export` 透传 args；`export_summary.json` 增字段）

- [ ] **Step 1: 加参数到对外入口**

在 `export_filtered_dataset`（:733）与 `_export_filtered_dataset_impl`（:785）签名末尾加：

```python
    size_ratio: str = "",
    size_balance_weight: float = 1.0,
    avg_boxes_per_image_min: float = 0.0,
    avg_boxes_per_image_max: float = 0.0,
```

并在 `export_filtered_dataset` 的 `_export_filtered_dataset_impl(...)` 调用里把这 4 个透传下去。

- [ ] **Step 2: 在 impl 内解析比例并扫描尺寸**

在 `_export_filtered_dataset_impl` 拿到 `filtered_candidates_by_split` 之后、选图之前（约 :1100 `pooled_candidates = pool_candidates(...)` 附近）：

```python
    from .det_size_supplement import parse_size_ratio
    target_size_ratio = parse_size_ratio(size_ratio) if size_ratio.strip() else None
    size_buckets_by_candidate = None
    if target_size_ratio is not None:
        size_buckets_by_candidate = scan_candidate_size_buckets(
            pooled_candidates, class_id_mapping, cache_dir=source_root,
        )
```

> `parse_size_ratio` 返回 `{small,medium,large}`（已归一），与 Task 2/3 的 `target_size_ratio` 口径一致。

- [ ] **Step 3: 把尺寸参数传进选图**

在 `_select_balanced_train_candidates_compat(...)` 调用（:1101）补传新参数（同时给 compat 包装加这些 keyword 的透传，沿用其 `_supports_keyword_arg` 模式或直接加形参）：

```python
        size_buckets_by_candidate=size_buckets_by_candidate,
        target_size_ratio=target_size_ratio,
        size_balance_weight=size_balance_weight,
        avg_boxes_per_image_min=avg_boxes_per_image_min,
        avg_boxes_per_image_max=avg_boxes_per_image_max,
```

- [ ] **Step 4: 把尺寸传进划分**

在 `assign_candidates_to_new_splits(...)` 调用（:1130）补 `size_buckets_by_candidate=size_buckets_by_candidate,`。

- [ ] **Step 5: 报告增字段**

在 `export_summary.json` 写入处（:1259 起的 dict）追加：

```python
                "size_ratio_requested": size_ratio,
                "target_size_ratio": target_size_ratio,
                "size_balance_weight": size_balance_weight,
                "avg_boxes_per_image_min": avg_boxes_per_image_min,
                "avg_boxes_per_image_max": avg_boxes_per_image_max,
                "size_selection_summary": selection_summary,
```

（`selection_summary` 已含 `achieved_size_ratio` / `avg_boxes_per_image_achieved`，size-aware 模式下自动带出。）

- [ ] **Step 6: run_export 透传**

在 `run_export`（:1377）的 `export_filtered_dataset(...)` 调用末尾加：

```python
        size_ratio=args.size_ratio,
        size_balance_weight=args.size_balance_weight,
        avg_boxes_per_image_min=args.avg_boxes_per_image_min,
        avg_boxes_per_image_max=args.avg_boxes_per_image_max,
```

- [ ] **Step 7: 冒烟（无新单测，靠 Task 8 集成测试覆盖）**

Run: `conda run -n lightlytrain python -c "import tool_lib.det_export"`
Expected: 无 ImportError

- [ ] **Step 8: Commit**

```bash
git add tool_lib/det_export.py
git commit -m "feat(export): wire size-aware selection/splitting through det export pipeline"
```

---

## Phase 3 — det 参数（launcher / common / interactive）

### Task 7: det 新参数贯通配置链

**Files:**
- Modify: `launcher.py:96-108`（DET_SETTINGS 加 4 个键 + 注释）
- Modify: `tool_lib/common.py`（:124 区默认常量；:293 区 `global`；:437 区 `settings.get`）
- Modify: `tool_lib/interactive.py`（:2247 区 interactive args；:2599 区 argparse；:531 区 preview 打印）

- [ ] **Step 1: launcher 默认值**

在 `launcher.py` DET_SETTINGS（`det_export_box_density_penalty` 之后、`det_export_suffix` 之前）加：

```python
    # det_export_size_ratio:
    #   目标尺寸占比 小:中:大（排除 tiny<16²）。默认 "30:40:30"，填 "" 关闭尺寸感知、回退旧选图。
    "det_export_size_ratio": "30:40:30",
    # det_export_size_balance_weight: 尺寸赤字相对类别赤字的联合权重，默认 1.0。
    "det_export_size_balance_weight": 1.0,
    # det_export_avg_boxes_per_image_min/max: 平均每图框数软目标区间，默认 5~15；都填 0 关闭。
    "det_export_avg_boxes_per_image_min": 5,
    "det_export_avg_boxes_per_image_max": 15,
```

- [ ] **Step 2: common.py 默认常量**

在 `common.py:124`（`EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE = 0` 邻近）加模块级默认：

```python
EXPORT_DEFAULT_SIZE_RATIO = "30:40:30"
EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = 1.0
EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN = 5.0
EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX = 15.0
```

在 `apply_user_settings` 的 `global` 声明区（:293 附近）加这 4 个名字到 `global`。在 `:437`（`EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE` 读取邻近）加：

```python
    EXPORT_DEFAULT_SIZE_RATIO = str(
        settings.get("det_export_size_ratio", EXPORT_DEFAULT_SIZE_RATIO)
    )
    EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = float(
        settings.get("det_export_size_balance_weight", EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT)
    )
    EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN = float(
        settings.get("det_export_avg_boxes_per_image_min", EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN)
    )
    EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX = float(
        settings.get("det_export_avg_boxes_per_image_max", EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX)
    )
```

- [ ] **Step 3: interactive args + argparse + preview**

在 `interactive.py` `build_interactive_args` 的 det export Namespace（:2247 `max_boxes_per_image=...` 邻近）加：

```python
            size_ratio=rt.EXPORT_DEFAULT_SIZE_RATIO,
            size_balance_weight=rt.EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT,
            avg_boxes_per_image_min=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN,
            avg_boxes_per_image_max=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX,
```

在 `parse_cli_args` det export_parser（:2599 `--max-boxes-per-image` 邻近）加：

```python
    export_parser.add_argument("--size-ratio", type=str, default=rt.EXPORT_DEFAULT_SIZE_RATIO)
    export_parser.add_argument("--size-balance-weight", type=float, default=rt.EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT)
    export_parser.add_argument("--avg-boxes-per-image-min", type=float, default=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN)
    export_parser.add_argument("--avg-boxes-per-image-max", type=float, default=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX)
```

在 `print_det_export_preview`（:531 邻近）加：

```python
    print(f"  size_ratio: {compact_display_value(args.size_ratio)}")
    print(_det_export_strategy_text("size_balance_weight", args.size_balance_weight, enabled=bool(str(args.size_ratio).strip())))
    print(_det_export_strategy_text("avg_boxes_per_image_min", args.avg_boxes_per_image_min, enabled=bool(str(args.size_ratio).strip())))
    print(_det_export_strategy_text("avg_boxes_per_image_max", args.avg_boxes_per_image_max, enabled=bool(str(args.size_ratio).strip())))
```

- [ ] **Step 4: 冒烟**

Run: `conda run -n lightlytrain python launcher.py export --help`
Expected: 输出含 `--size-ratio`，无报错

- [ ] **Step 5: Commit**

```bash
git add launcher.py tool_lib/common.py tool_lib/interactive.py
git commit -m "feat(export): expose det size-aware params through config/cli/interactive"
```

---

## Phase 4 — det 端到端集成测试

### Task 8: det 集成 + 回归保护

**Files:**
- Test: `tool_lib/tests/test_size_aware_export_det.py`

- [ ] **Step 1: 写集成测试**

新建 `tool_lib/tests/test_size_aware_export_det.py`（合成数据集范式同 `test_det_size_supplement.py`：100×100 图，`bw/bh` 归一化使面积 = `bw*100 * bh*100`）：

```python
"""det 尺寸感知导出端到端 + 回归保护。"""
from __future__ import annotations

import json
from pathlib import Path

from tool_lib import common as rt
from tool_lib.det_export import export_filtered_dataset


def _make_dataset(root: Path):
    rt.import_runtime_dependencies()
    # 20 张大目标图 (1.0x1.0→10000px large) + 20 张小目标图 (0.2x0.2→400px small)
    spec = {"train": [], "val": [], "test": []}
    for i in range(20):
        spec["train"].append((f"big{i}", 1.0, 1.0))
    for i in range(20):
        spec["train"].append((f"sml{i}", 0.2, 0.2))
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for stem, bw, bh in spec["train"]:
        rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(img_dir / f"{stem}.jpg")
        (lbl_dir / f"{stem}.txt").write_text(f"0 0.5 0.5 {bw} {bh}\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    rt.dump_yaml(data_yaml, {
        "path": str(root.resolve()), "train": "images/train",
        "val": "images/val", "test": "images/test",
        "task": "detect", "nc": 1, "names": ["obj"],
    })
    return data_yaml


def test_size_aware_export_shifts_small_ratio_up(tmp_path):
    data_yaml = _make_dataset(tmp_path / "ds")
    neutral = {"config": {"data_root": str((tmp_path / "ds").resolve())}, "summary": {}, "per_class_ap": {}}
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_A",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_boxes=0, target_images_per_class=0,
        target_total_images=20, split_ratio="8:1:1", target_boxes_per_class=0,
        max_boxes_per_image=0, max_boxes_per_class_per_image=0, box_density_penalty=0.0,
        size_ratio="70:0:30", size_balance_weight=1.0,
        avg_boxes_per_image_min=0, avg_boxes_per_image_max=0,
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
    achieved = summary["size_selection_summary"]["achieved_size_ratio"]
    assert achieved["small"] >= 0.5  # 目标偏小，实际小占比明显升高


def test_size_ratio_empty_uses_legacy_path(tmp_path):
    data_yaml = _make_dataset(tmp_path / "ds")
    neutral = {"config": {"data_root": str((tmp_path / "ds").resolve())}, "summary": {}, "per_class_ap": {}}
    out = export_filtered_dataset(
        data_yaml, neutral, 0.0, "_A",
        auto_balance=True, auto_relax_class_threshold=True, balance_ratio=0.0,
        min_class_images=0, min_class_boxes=0, target_images_per_class=0,
        target_total_images=10, split_ratio="8:1:1", target_boxes_per_class=0,
        max_boxes_per_image=0, max_boxes_per_class_per_image=0, box_density_penalty=0.0,
        size_ratio="",
    )
    summary = json.loads((out / "export_summary.json").read_text(encoding="utf-8"))
    assert summary["selection_summary"]["selection_algorithm"] == "celf_lazy_greedy"
```

- [ ] **Step 2: 运行**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_export_det.py -v`
Expected: PASS（若断言阈值因贪心细节略有出入，按实际 achieved 调整阈值但保持"明显升高"语义）

- [ ] **Step 3: 跑全量 det 相关测试**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/ -v`
Expected: PASS（含既有 size-supplement 测试）

- [ ] **Step 4: Commit**

```bash
git add tool_lib/tests/test_size_aware_export_det.py
git commit -m "test(export): det size-aware integration + legacy regression guard"
```

---

## Phase 5 — seg 对齐

### Task 9: seg 候选尺寸扫描（掩码面积）+ 主管线串接

**Files:**
- Modify: `tool_lib/seg_export.py`（新增 `scan_candidate_size_buckets`（掩码面积版）；主管线传入选图/划分；签名加 4 参数；report 增字段）

- [ ] **Step 1: 写失败测试**

在新建 `tool_lib/tests/test_size_aware_export_seg.py` 先放扫描单测：

```python
from pathlib import Path
from tool_lib import common as rt
from tool_lib.seg_export import scan_candidate_size_buckets
from tool_lib.seg_shared import SegExportImageCandidate


def test_seg_scan_mask_area_buckets(tmp_path):
    rt.import_runtime_dependencies()
    img = tmp_path / "a.jpg"
    rt.Image.new("RGB", (100, 100), (127, 127, 127)).save(img)
    # 多边形覆盖整图 → 面积≈10000 large；小三角 → tiny/small
    big = "0 0 0 1 0 1 1 0 1"          # 单位正方形，归一面积≈1 → 10000px large
    small = "0 0.5 0.5 0.55 0.5 0.5 0.55"  # 极小三角
    cand = SegExportImageCandidate(
        split_name="train",
        rel_split_image_dir=Path("images/train"),
        rel_split_label_dir=Path("labels/train"),
        rel_path=Path("a.jpg"),
        src_image_path=img,
        src_label_path=tmp_path / "a.txt",
        filtered_lines=(big, small),
        class_box_counts={0: 2},
    )
    out = scan_candidate_size_buckets([cand], {0: 0}, cache_dir=tmp_path)
    key = "a.jpg|train"
    assert out[key]["large"] == 1
```

- [ ] **Step 2: 运行确认失败**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_export_seg.py::test_seg_scan_mask_area_buckets -v`
Expected: FAIL

- [ ] **Step 3: 实现 seg 扫描 + 串接**

在 `seg_export.py` 新增（用 `parse_polygon_line` + `polygon_area_normalized` + `_bucket_mask_area`）：

```python
def scan_candidate_size_buckets(
    candidates: list[Any],
    class_id_mapping: dict[int, int],
    *,
    cache_dir: Path,
) -> dict[str, dict[str, int]]:
    """seg：逐图按掩码面积算 tiny/small/medium/large 实例数，键 rel_path|split。"""
    from .det_size_supplement import ImageSizeCache
    cache = ImageSizeCache(Path(cache_dir) / ".imgsize_cache.json")
    result: dict[str, dict[str, int]] = {}
    for candidate in candidates:
        width, height = cache.get(candidate.src_image_path)
        image_area = max(width * height, 1)
        buckets = _empty_size_bucket_counts()
        for line in candidate.filtered_lines:
            parsed = parse_polygon_line(line)
            if parsed is None:
                continue
            _, coords = parsed
            pixel_area = polygon_area_normalized(coords) * image_area
            buckets[_bucket_mask_area(pixel_area)] += 1
        result[candidate.rel_path.as_posix() + "|" + candidate.split_name] = buckets
    cache.save()
    return result
```

在 seg 主导出管线（`seg_export.py` 对应 `_export_filtered_dataset_impl` 等价处）：加签名参数 `size_ratio="" / size_balance_weight=1.0 / avg_instances_per_image_min=0.0 / avg_instances_per_image_max=0.0`；解析 `parse_size_ratio`；扫描；把 `size_buckets_by_candidate / target_size_ratio / size_balance_weight / avg_*` 传进 `_select_balanced_train_candidates_compat`（参数名复用 det_analysis 的 `avg_boxes_per_image_min/max`，seg 侧把实例 min/max 映射进去）与 `assign_candidates_to_new_splits`。report 增同名字段。

> seg 选图/划分**直接复用 det_analysis 的 `select_balanced_train_candidates` / `assign_candidates_to_new_splits`**（seg_export 已 import 它们），因此 Task 3/4 的逻辑 seg 免费获得，只需喂入掩码面积桶。

- [ ] **Step 4: 运行确认通过**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/test_size_aware_export_seg.py::test_seg_scan_mask_area_buckets -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tool_lib/seg_export.py tool_lib/tests/test_size_aware_export_seg.py
git commit -m "feat(export): seg mask-area size scan + wire size-aware pipeline"
```

---

### Task 10: seg 参数贯通配置链

**Files:**
- Modify: `launcher.py:179-191`（SEG_SETTINGS 加 4 键 + 注释）
- Modify: `tool_lib/common.py`（seg export 默认常量区，约 :453+ SEG_ 命名空间）
- Modify: `tool_lib/interactive.py`（seg interactive args、seg argparse、`print_seg_export_preview` :574 邻近）

- [ ] **Step 1: launcher seg 默认值**

在 `launcher.py` SEG_SETTINGS（`seg_export_instance_density_penalty` 之后）加：

```python
    # seg_export_size_ratio: 目标掩码尺寸占比 小:中:大（排除 tiny）。默认 "30:40:30"，"" 关闭。
    "seg_export_size_ratio": "30:40:30",
    "seg_export_size_balance_weight": 1.0,
    "seg_export_avg_instances_per_image_min": 5,
    "seg_export_avg_instances_per_image_max": 15,
```

- [ ] **Step 2: common.py seg 默认常量**

在 common.py seg export 默认区加 `SEG_EXPORT_DEFAULT_SIZE_RATIO = "30:40:30"`、`SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = 1.0`、`SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN = 5.0`、`SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX = 15.0`，加入 `global`，并在 settings 读取区加对应 `settings.get("seg_export_size_ratio", ...)` 等（命名与 det 对称，前缀 `SEG_`）。

- [ ] **Step 3: interactive seg args + argparse + preview**

seg interactive Namespace 加 `size_ratio=rt.SEG_EXPORT_DEFAULT_SIZE_RATIO` 等 4 项；seg export_parser 加 `--size-ratio` / `--size-balance-weight` / `--avg-instances-per-image-min` / `--avg-instances-per-image-max`；`print_seg_export_preview` 加 4 行打印（参照 Task 7 Step 3 det 版）。

- [ ] **Step 4: 冒烟**

Run: `conda run -n lightlytrain python launcher.py seg-export --help`（命令名以现有 seg 导出子命令为准）
Expected: 含 `--size-ratio`，无报错

- [ ] **Step 5: Commit**

```bash
git add launcher.py tool_lib/common.py tool_lib/interactive.py
git commit -m "feat(export): expose seg size-aware params through config/cli/interactive"
```

---

### Task 11: seg 端到端集成测试

**Files:**
- Test: `tool_lib/tests/test_size_aware_export_seg.py`（追加端到端）

- [ ] **Step 1: 追加集成测试**

合成 seg 数据集（polygon 标签，大多边形→large，小多边形→small），调用 seg 导出主入口，断言 `export_summary.json` 的 `achieved_size_ratio["small"]` 在目标偏小时升高；并断言 `size_ratio=""` 时走 `celf_lazy_greedy`。结构镜像 Task 8。

```python
def test_seg_size_aware_export_shifts_ratio(tmp_path):
    # 见 Task 8 范式：造大/小多边形各若干，target small 偏高，断言 achieved small 升高
    ...
```

> 实现时用与 `test_det_size_supplement.py` 一致的 seg 合成范式（若仓库已有 seg 合成 helper 则复用）；断言阈值按实际 achieved 校准，保持"明显升高"语义。

- [ ] **Step 2: 运行全量**

Run: `conda run -n lightlytrain python -m pytest tool_lib/tests/ -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tool_lib/tests/test_size_aware_export_seg.py
git commit -m "test(export): seg size-aware integration test"
```

---

## 自检结果（spec 覆盖）

| spec 章节 | 对应 Task |
|---|---|
| §1b 4 档分桶（tiny 排除） | Task 1 |
| §4 配置参数（det/seg） | Task 7、Task 10 |
| §5.1 候选尺寸扫描（缓存） | Task 5（det）、Task 9（seg） |
| §5.2 双引擎开关 | Task 3 |
| §5.3 联合赤字打分 | Task 2、Task 3 |
| §5.4 密度软目标（内置权重） | Task 2、Task 3 |
| §6 split 尺寸均衡（类别优先） | Task 4 |
| §7 seg 对齐 | Task 9、Task 10、Task 11 |
| §8 报告 | Task 6、Task 9 |
| §9 默认开 / 可退回 | Task 3（开关）、Task 7/10（默认值） |
| §10 测试 | Task 1/2/3/4/5/8/9/11 |
| §11 实现切分（det 先） | Phase 0-4 det，Phase 5 seg |

**已知需实现时确认的点（非阻塞）：**
- Task 3 的 `batch_size` 分批停止条件按实测调（合成小数据集下 batch 触发逻辑需保证至少能选满 target）。
- Task 4 把 size_gain 并入现有比较链时，务必置于类别覆盖之下，跑回归测试确认 `size_buckets_by_candidate=None` 行为不变。
- Task 9/10 的 seg 主入口与子命令确切名称以仓库现状为准（`run_export` 等价物）。
