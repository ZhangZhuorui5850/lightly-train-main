# seg eval/infer 多卡并行与升级 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 seg 的 eval 与 infer 从串行单卡升级为多卡分片 + 负载感知选卡 + 吞吐优化 + 工程对齐，对外产物与现状逐位兼容。

**Architecture:** 先抽出 det/seg 共用的 GPU 调度内核到 `tool_lib/gpu_parallel.py`（纯重构，det 行为零变化）；再在 seg 侧实现 RLE 掩码编解码、语义掩码 LUT、混淆矩阵/实例预测的 shard 序列化与合并、eval/infer 的子进程并行入口、预取重叠与工程对齐字段。

**Tech Stack:** Python 3, numpy, torch, PIL, pytest；子进程通过 `launcher.py eval/infer` 自调用并 pin `CUDA_VISIBLE_DEVICES`。

**测试运行约定：** 全部 pytest 在 WSL 的 `lightlytrain` conda 环境里跑，例如：
`conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py -v`
（纯函数测试不需要 GPU/模型）

---

## 文件结构

- **Create** `tool_lib/gpu_parallel.py` — GPU 探测/负载过滤/选卡/分片索引/子进程编排（det+seg 共用）
- **Modify** `tool_lib/det_infer.py` — 改为从 `gpu_parallel` import，删除被抽走的实现，`filter_samples_for_shard` 变薄封装，两处并行编排改用 `run_sharded_subprocesses`
- **Modify** `tool_lib/seg_shared.py` — 新增 `rle_encode` / `rle_decode`
- **Modify** `tool_lib/seg_tools.py` — LUT 优化、shard 累积/写出/合并、eval/infer 并行入口、预取、计时/run_meta/dry-run/容错
- **Modify** `tool_lib/dispatch.py` — eval/infer 的 seg shard 子进程路由
- **Modify** `tool_lib/interactive.py` — `eval` parser 新增 shard/dry-run/SUPPRESS 旗标
- **Create** `tests/test_gpu_parallel.py` — 调度内核单测
- **Modify** `tests/test_seg_semantic_eda_curate.py` 同目录新增 `tests/test_seg_parallel.py` — RLE/LUT/混淆合并/实例合并/并行退化单测

---

## Phase 0：抽取共享 GPU 调度模块（纯重构，det 零行为变化）

### Task 0.1：创建 `gpu_parallel.py`，迁移 GPU 探测/选卡函数

**Files:**
- Create: `tool_lib/gpu_parallel.py`
- Modify: `tool_lib/det_infer.py:34`（删除常量与函数定义，改为 import）
- Test: `tests/test_gpu_parallel.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_gpu_parallel.py
from __future__ import annotations

from tool_lib import gpu_parallel


def _gpu(index, used_ratio, memory_used=0, memory_total=10000, utilization=0):
    return {
        "index": index,
        "memory_used": memory_used,
        "memory_total": memory_total,
        "utilization": utilization,
        "used_ratio": used_ratio,
    }


def test_filter_high_memory_gpus_drops_busy_cards():
    gpus = [_gpu(0, 0.10), _gpu(1, 0.80), _gpu(2, 0.50)]
    eligible = gpu_parallel.filter_high_memory_gpus(gpus)
    assert [g["index"] for g in eligible] == [0, 2]


def test_select_fallback_single_gpu_picks_first_eligible():
    gpus = [_gpu(0, 0.10), _gpu(1, 0.05)]
    eligible = [gpus[1]]
    device, message = gpu_parallel.select_fallback_single_gpu(gpus, eligible)
    assert device == "cuda:1"
    assert "单卡顺序模式" in message


def test_select_fallback_single_gpu_handles_empty():
    device, message = gpu_parallel.select_fallback_single_gpu([], [])
    assert device is None
    assert "未检测到可用 GPU" in message
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tool_lib.gpu_parallel'`

- [ ] **Step 3: 创建 `gpu_parallel.py`，把 `det_infer.py` 的实现剪切过来**

把 `det_infer.py` 现有的 `GPU_HIGH_MEMORY_RATIO_THRESHOLD`（`det_infer.py:34`）、`query_gpu_inventory`（`det_infer.py:692`）、`filter_high_memory_gpus`（`det_infer.py:757`）、`format_gpu_summary`（`det_infer.py:774`）、`select_fallback_single_gpu`（`det_infer.py:782`）**整体移动**到新文件，开头加：

```python
# tool_lib/gpu_parallel.py
"""det / seg 共用的 GPU 调度内核：探测、负载过滤、选卡、分片、子进程编排。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from . import common as rt

GPU_HIGH_MEMORY_RATIO_THRESHOLD = 0.75
```

（`query_gpu_inventory` / `filter_high_memory_gpus` / `format_gpu_summary` / `select_fallback_single_gpu` 函数体原样粘贴，不改逻辑。）

- [ ] **Step 4: `det_infer.py` 改为 import**

删除 `det_infer.py` 中这 5 个定义，在文件顶部 import 区（`det_infer.py:32` 之后）加：

```python
from .gpu_parallel import (
    GPU_HIGH_MEMORY_RATIO_THRESHOLD,
    filter_high_memory_gpus,
    format_gpu_summary,
    query_gpu_inventory,
    select_fallback_single_gpu,
)
```

- [ ] **Step 5: 运行新测试 + det 回归测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py tests/test_det_infer.py -v`
Expected: PASS（全部）

- [ ] **Step 6: 提交**

```bash
git add tool_lib/gpu_parallel.py tool_lib/det_infer.py tests/test_gpu_parallel.py
git commit -m "refactor(infer): extract GPU scheduling kernel to gpu_parallel"
```

---

### Task 0.2：泛化分片索引 `filter_indices_for_shard`

**Files:**
- Modify: `tool_lib/gpu_parallel.py`
- Modify: `tool_lib/det_infer.py:49`（`filter_samples_for_shard` 改薄封装）
- Test: `tests/test_gpu_parallel.py`

- [ ] **Step 1: 写失败测试**

```python
def test_filter_indices_for_shard_round_robin():
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=0, num_shards=3) == [0, 3, 6]
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=1, num_shards=3) == [1, 4]
    assert gpu_parallel.filter_indices_for_shard(7, shard_index=2, num_shards=3) == [2, 5]


def test_filter_indices_for_shard_single_shard_returns_all():
    assert gpu_parallel.filter_indices_for_shard(4, shard_index=None, num_shards=1) == [0, 1, 2, 3]
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py -k filter_indices -v`
Expected: FAIL — `AttributeError: ... filter_indices_for_shard`

- [ ] **Step 3: 实现**

加到 `gpu_parallel.py`：

```python
def filter_indices_for_shard(count: int, *, shard_index: int | None, num_shards: int) -> list[int]:
    num_shards = int(num_shards or 1)
    if shard_index is None or num_shards <= 1:
        return list(range(count))
    shard_index = int(shard_index)
    return [i for i in range(count) if i % num_shards == shard_index]
```

- [ ] **Step 4: `det_infer.py` 复用**

把 `det_infer.py:49` 的 `filter_samples_for_shard` 改为：

```python
def filter_samples_for_shard(samples: list[Any], args) -> list[Any]:
    if not is_shard_child(args):
        return samples
    indices = gpu_parallel.filter_indices_for_shard(
        len(samples), shard_index=int(args.shard_index), num_shards=int(args.num_shards)
    )
    return [samples[i] for i in indices]
```

并在 `det_infer.py` import 区加 `from . import gpu_parallel`。

- [ ] **Step 5: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py tests/test_det_infer.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add tool_lib/gpu_parallel.py tool_lib/det_infer.py tests/test_gpu_parallel.py
git commit -m "refactor(infer): generalize shard index filtering"
```

---

### Task 0.3：抽取子进程编排 `run_sharded_subprocesses`

**Files:**
- Modify: `tool_lib/gpu_parallel.py`
- Modify: `tool_lib/det_infer.py`（`run_parallel_split_infer` / `run_parallel_all_infer` 的起进程+wait 循环改用它）
- Test: `tests/test_gpu_parallel.py`

- [ ] **Step 1: 写失败测试**

```python
def test_run_sharded_subprocesses_reports_failures(monkeypatch):
    calls = []

    class _FakeProc:
        def __init__(self, returncode):
            self._rc = returncode
        def wait(self):
            return self._rc

    def _fake_popen(command, cwd, env):
        calls.append((command, env.get("CUDA_VISIBLE_DEVICES")))
        rc = 1 if "shard-1" in command else 0
        return _FakeProc(rc)

    monkeypatch.setattr(gpu_parallel.subprocess, "Popen", _fake_popen)
    jobs = [
        (0, 3, ["python", "x", "shard-0"]),
        (1, 5, ["python", "x", "shard-1"]),
    ]
    failed = gpu_parallel.run_sharded_subprocesses(jobs, cwd="/tmp")
    assert failed == ["shard_01(exit=1)"]
    assert calls[0][1] == "3" and calls[1][1] == "5"
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py -k sharded_subprocesses -v`
Expected: FAIL — `AttributeError: run_sharded_subprocesses`

- [ ] **Step 3: 实现**

加到 `gpu_parallel.py`：

```python
def run_sharded_subprocesses(
    jobs: list[tuple[int, int, list[str]]],
    *,
    cwd: str | Path,
) -> list[str]:
    """起一组子进程，每个 pin 到一张卡，等待全部完成，返回失败 shard 描述列表。

    jobs: (shard_index, gpu_index, command) 列表。
    """
    processes: list[tuple[int, subprocess.Popen[str]]] = []
    for shard_index, gpu_index, command in jobs:
        child_env = os.environ.copy()
        child_env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
        process = subprocess.Popen(command, cwd=str(cwd), env=child_env)
        processes.append((shard_index, process))

    failed: list[str] = []
    for shard_index, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append(f"shard_{shard_index:02d}(exit={return_code})")
    return failed
```

- [ ] **Step 4: det 两处编排改用它**

`run_parallel_split_infer`（`det_infer.py:1318-1348`）里构造 `command` 与 `CUDA_VISIBLE_DEVICES` 的循环、以及随后的 `process.wait()` 失败收集，替换为：先构造 `jobs = [(shard_index, int(gpu["index"]), command), ...]`（command 仍由 `build_parallel_child_command` 生成），打印保留，然后：

```python
    failed_shards = gpu_parallel.run_sharded_subprocesses(jobs, cwd=rt.ROOT_DIR)
    if failed_shards:
        raise RuntimeError(f"shard infer 失败: {', '.join(failed_shards)}")
```

`run_parallel_all_infer`（`det_infer.py:1422-1458`）同样改造（错误信息保留原文"并行 infer 失败"）。注意保留原有的打印（保留 GPU 摘要、shard→device 映射）。

- [ ] **Step 5: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py tests/test_det_infer.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add tool_lib/gpu_parallel.py tool_lib/det_infer.py tests/test_gpu_parallel.py
git commit -m "refactor(infer): extract sharded subprocess runner"
```

---

## Phase 1：RLE 掩码编解码

### Task 1.1：`seg_shared.py` 新增 `rle_encode` / `rle_decode`

**Files:**
- Modify: `tool_lib/seg_shared.py`
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_seg_parallel.py
from __future__ import annotations

import numpy as np

from tool_lib import seg_shared


def _roundtrip(mask):
    rle = seg_shared.rle_encode(mask)
    out = seg_shared.rle_decode(rle)
    assert out.dtype == bool
    assert out.shape == mask.shape
    assert np.array_equal(out, mask.astype(bool))


def test_rle_roundtrip_mixed():
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=bool)
    _roundtrip(mask)


def test_rle_roundtrip_all_zero_and_all_one():
    _roundtrip(np.zeros((4, 5), dtype=bool))
    _roundtrip(np.ones((4, 5), dtype=bool))


def test_rle_roundtrip_non_square():
    rng = np.random.default_rng(0)
    _roundtrip(rng.integers(0, 2, size=(7, 13)).astype(bool))


def test_rle_size_field_matches_shape():
    mask = np.zeros((3, 8), dtype=bool)
    rle = seg_shared.rle_encode(mask)
    assert rle["size"] == [3, 8]
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k rle -v`
Expected: FAIL — `AttributeError: ... rle_encode`

- [ ] **Step 3: 实现**

加到 `seg_shared.py`（顶部已 `import numpy as np` 或用 `rt.np`；该文件已 import 何种请遵循其现状，若无则 `import numpy as np`）：

```python
def rle_encode(mask: "np.ndarray") -> dict:
    """列优先（COCO 约定）游程编码二值掩码。counts 首段为 0 像素游程长度。"""
    import numpy as np

    mask = np.asfortranarray(mask.astype(bool))
    flat = mask.flatten(order="F")
    # counts 交替表示 0 段、1 段的游程长度，首段始终代表 0。
    # 不手动预置 0：循环从 prev=False 起步，遇到起始的 1 会自然 append 一个长度 0 的 0 段。
    counts: list[int] = []
    if flat.size == 0:
        return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}
    prev = False
    run = 0
    for value in flat:
        if bool(value) == prev:
            run += 1
        else:
            counts.append(run)
            prev = bool(value)
            run = 1
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def rle_decode(rle: dict) -> "np.ndarray":
    import numpy as np

    height, width = int(rle["size"][0]), int(rle["size"][1])
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for count in rle["counts"]:
        if value:
            flat[position : position + count] = True
        position += count
        value = not value
    return flat.reshape((height, width), order="F")
```

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k rle -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tool_lib/seg_shared.py tests/test_seg_parallel.py
git commit -m "feat(seg): add RLE mask encode/decode for shard serialization"
```

---

## Phase 2：语义掩码 LUT 向量化

### Task 2.1：`_load_semantic_mask` 改 LUT（与旧实现逐元素等价）

**Files:**
- Modify: `tool_lib/seg_tools.py:114-127`
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试**（先把旧逻辑复制成参考实现，断言新实现与之一致）

```python
from pathlib import Path

from PIL import Image

from tool_lib import seg_tools
from tool_lib import common as rt


def _reference_load(mask_path, classes, ignore_classes):
    # 旧实现逻辑的参考副本，用于等价性对拍
    with Image.open(mask_path) as mask_image:
        mask_np = np.array(mask_image)
    compare_np = mask_np if mask_np.ndim == 3 else mask_np[:, :, None]
    target = np.full(mask_np.shape[:2], -100, dtype=np.int64)
    original_to_internal = {
        cid: i for i, cid in enumerate(sorted(set(classes) - ignore_classes))
    }
    for cid, internal in original_to_internal.items():
        for label in seg_tools._class_labels(classes, cid):
            label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            target[np.all(compare_np == np.array(label_tuple), axis=2)] = internal
    return target


def test_load_semantic_mask_single_channel_matches_reference(tmp_path):
    classes = {0: "bg", 1: "cat", 2: "dog", 255: "ignore"}
    ignore = {255}
    arr = np.array([[0, 1, 2], [255, 1, 0]], dtype=np.uint8)
    p = tmp_path / "m.png"
    Image.fromarray(arr, mode="L").save(p)
    expected = _reference_load(p, classes, ignore)
    got = seg_tools._load_semantic_mask(p, classes, ignore)
    assert np.array_equal(got, expected)


def test_load_semantic_mask_rgb_labels_matches_reference(tmp_path):
    classes = {
        0: {"name": "bg", "labels": [[0, 0, 0]]},
        1: {"name": "road", "labels": [[128, 64, 128]]},
        2: {"name": "sky", "labels": [[70, 130, 180]]},
    }
    ignore = set()
    arr = np.array(
        [[[0, 0, 0], [128, 64, 128]], [[70, 130, 180], [0, 0, 0]]], dtype=np.uint8
    )
    p = tmp_path / "m.png"
    Image.fromarray(arr, mode="RGB").save(p)
    expected = _reference_load(p, classes, ignore)
    got = seg_tools._load_semantic_mask(p, classes, ignore)
    assert np.array_equal(got, expected)
```

- [ ] **Step 2: 运行验证（旧实现下应通过，作为基线）**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k load_semantic_mask -v`
Expected: PASS（当前旧实现已能通过——这两个测试是等价性护栏，重构后必须仍 PASS）

- [ ] **Step 3: 用 LUT 重写 `_load_semantic_mask`**

替换 `seg_tools.py:114-127`：

```python
def _load_semantic_mask(mask_path: Path, classes: dict[int, Any], ignore_classes: set[int]) -> Any:
    with rt.Image.open(mask_path) as mask_image:
        mask_np = rt.np.array(mask_image)
    original_to_internal = {
        class_id: internal_id
        for internal_id, class_id in enumerate(sorted(set(classes) - ignore_classes))
    }

    # 收集 (label_tuple, internal_id)；单通道标签长度 1，RGB 长度 3。
    single_pairs: list[tuple[int, int]] = []
    rgb_pairs: list[tuple[tuple[int, int, int], int]] = []
    for class_id, internal_id in original_to_internal.items():
        for label in _class_labels(classes, class_id):
            values = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            if len(values) == 1:
                single_pairs.append((values[0], internal_id))
            else:
                rgb_pairs.append((tuple(int(v) for v in values[:3]), internal_id))

    if mask_np.ndim == 2 and single_pairs and not rgb_pairs:
        max_label = max(label for label, _ in single_pairs)
        lut = rt.np.full(max_label + 1, -100, dtype=rt.np.int64)
        for label, internal_id in single_pairs:
            lut[label] = internal_id
        clipped = rt.np.clip(mask_np, 0, max_label)
        target = rt.np.where(mask_np <= max_label, lut[clipped], -100).astype(rt.np.int64)
        return target

    # RGB（或混合）：打包成 int 后用排序键映射。
    compare_np = mask_np if mask_np.ndim == 3 else rt.np.repeat(mask_np[:, :, None], 3, axis=2)
    packed = (
        compare_np[:, :, 0].astype(rt.np.int64) << 16
    ) | (compare_np[:, :, 1].astype(rt.np.int64) << 8) | compare_np[:, :, 2].astype(rt.np.int64)
    keys: list[int] = []
    vals: list[int] = []
    for (r, g, b), internal_id in rgb_pairs:
        keys.append((r << 16) | (g << 8) | b)
        vals.append(internal_id)
    for label, internal_id in single_pairs:
        keys.append((label << 16) | (label << 8) | label)
        vals.append(internal_id)
    target = rt.np.full(mask_np.shape[:2], -100, dtype=rt.np.int64)
    if keys:
        keys_arr = rt.np.array(keys, dtype=rt.np.int64)
        vals_arr = rt.np.array(vals, dtype=rt.np.int64)
        order = rt.np.argsort(keys_arr)
        keys_sorted = keys_arr[order]
        vals_sorted = vals_arr[order]
        idx = rt.np.searchsorted(keys_sorted, packed)
        idx_clipped = rt.np.clip(idx, 0, len(keys_sorted) - 1)
        match = keys_sorted[idx_clipped] == packed
        target = rt.np.where(match, vals_sorted[idx_clipped], -100).astype(rt.np.int64)
    return target
```

- [ ] **Step 4: 运行等价性测试 + 语义 eda 回归**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k load_semantic_mask tests/test_seg_semantic_eda_curate.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tool_lib/seg_tools.py tests/test_seg_parallel.py
git commit -m "perf(seg): vectorize semantic mask loading with lookup table"
```

---

## Phase 3：语义评估的混淆矩阵 shard 化

### Task 3.1：拆出"累积混淆矩阵"与"写 summary"，新增 shard 写出/合并

**Files:**
- Modify: `tool_lib/seg_tools.py`（重构 `_evaluate_semantic_split`，新增 `_write_semantic_shard_result` / `_merge_semantic_shard_results`）
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试（混淆矩阵合并 == 单进程）**

```python
def test_merge_semantic_confusion_equals_single(tmp_path):
    k = 3
    full = np.array([[5, 1, 0], [0, 4, 2], [1, 0, 6]], dtype=np.int64)
    # 拆成两个 shard：行方向任意切分后元素相加应还原
    a = np.array([[3, 1, 0], [0, 1, 1], [0, 0, 3]], dtype=np.int64)
    b = full - a
    d1 = tmp_path / "s0"; d2 = tmp_path / "s1"
    d1.mkdir(); d2.mkdir()
    seg_tools._write_semantic_shard_result(d1, split="test", confusion=a, rows=[], class_names={0: "a", 1: "b", 2: "c"}, num_samples=2, infer_time_sum_ms=10.0, failed=0)
    seg_tools._write_semantic_shard_result(d2, split="test", confusion=b, rows=[], class_names={0: "a", 1: "b", 2: "c"}, num_samples=3, infer_time_sum_ms=20.0, failed=0)
    merged = seg_tools._merge_semantic_shard_results([d1, d2], split="test")
    assert np.array_equal(merged["confusion"], full)
    assert merged["num_samples"] == 5
    assert merged["infer_time_sum_ms"] == 30.0
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k merge_semantic_confusion -v`
Expected: FAIL — `AttributeError: _write_semantic_shard_result`

- [ ] **Step 3: 实现 shard 写出/合并**

加到 `seg_tools.py`：

```python
def _write_semantic_shard_result(
    shard_dir: Path,
    *,
    split: str,
    confusion: Any,
    rows: list[dict[str, Any]],
    class_names: dict[int, str],
    num_samples: int,
    infer_time_sum_ms: float,
    failed: int,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "confusion": confusion.astype(int).tolist(),
        "rows": rows,
        "class_names": {str(k): v for k, v in class_names.items()},
        "num_samples": int(num_samples),
        "infer_time_sum_ms": float(infer_time_sum_ms),
        "failed": int(failed),
    }
    path = shard_dir / "seg_semantic_shard_result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _merge_semantic_shard_results(shard_dirs: list[Path], *, split: str) -> dict[str, Any]:
    confusion = None
    rows: list[dict[str, Any]] = []
    class_names: dict[int, str] = {}
    num_samples = 0
    infer_time_sum_ms = 0.0
    failed = 0
    for shard_dir in shard_dirs:
        payload = json.loads((shard_dir / "seg_semantic_shard_result.json").read_text(encoding="utf-8"))
        mat = rt.np.array(payload["confusion"], dtype=rt.np.int64)
        confusion = mat if confusion is None else confusion + mat
        rows.extend(payload.get("rows", []))
        for cid_raw, name in payload.get("class_names", {}).items():
            class_names[int(cid_raw)] = str(name)
        num_samples += int(payload.get("num_samples", 0))
        infer_time_sum_ms += float(payload.get("infer_time_sum_ms", 0.0))
        failed += int(payload.get("failed", 0))
    return {
        "confusion": confusion,
        "rows": rows,
        "class_names": class_names,
        "num_samples": num_samples,
        "infer_time_sum_ms": infer_time_sum_ms,
        "failed": failed,
    }
```

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k merge_semantic_confusion -v`
Expected: PASS

- [ ] **Step 5: 重构 `_evaluate_semantic_split` 抽出累积逻辑**

把 `_evaluate_semantic_split`（`seg_tools.py:295-364`）的"遍历 samples 累积 confusion + rows + 计时 + 容错"抽成 `_accumulate_semantic_confusion(model, data_path, split, threshold, shard_index, num_shards) -> (confusion, rows, class_names, num_classes, num_samples, infer_time_sum_ms, failed)`：

- 在样本遍历前用 `gpu_parallel.filter_indices_for_shard(len(samples), shard_index=..., num_shards=...)` 取子集；
- 每张图 `predict` 包 `try/except Exception`，失败 `failed += 1` 并 `print` warning，`continue`；
- 用 `time.perf_counter()` 累加 `infer_time_sum_ms`（顶部 `import time`）。

`_evaluate_semantic_split` 改为：调用累积函数 → 调 `_compute_semantic_iou` → 写 summary（summary 里新增 `avg_infer_time_ms = infer_time_sum_ms / max(num_samples,1)` 与 `failed_images`）。

- [ ] **Step 6: 运行语义 eda/eval 相关回归**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py tests/test_seg_semantic_eda_curate.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add tool_lib/seg_tools.py tests/test_seg_parallel.py
git commit -m "feat(seg): shard-able semantic confusion accumulation + merge"
```

---

## Phase 4：实例评估的 RLE shard 化

### Task 4.1：实例 shard 预测序列化 + 父进程合并喂单个 metric

**Files:**
- Modify: `tool_lib/seg_tools.py`（新增 `_serialize_instance_entry` / `_write_instance_shard_result` / `_merge_instance_shard_results`，重构 `run_eval` 累积部分）
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试（往返序列化 + 合并条目计数）**

```python
import torch


def test_serialize_instance_entry_roundtrip():
    prediction = {
        "labels": torch.tensor([0, 2], dtype=torch.int64),
        "scores": torch.tensor([0.9, 0.5], dtype=torch.float32),
        "masks": torch.tensor(
            np.stack([np.eye(4, dtype=bool), np.ones((4, 4), dtype=bool)]),
        ),
    }
    target = {
        "labels": torch.tensor([0], dtype=torch.int64),
        "masks": torch.tensor(np.eye(4, dtype=bool)[None]),
    }
    entry = seg_tools._serialize_instance_entry(prediction, target)
    pred2, tgt2 = seg_tools._deserialize_instance_entry(entry)
    assert torch.equal(pred2["labels"], prediction["labels"])
    assert torch.allclose(pred2["scores"], prediction["scores"])
    assert torch.equal(pred2["masks"], prediction["masks"])
    assert torch.equal(tgt2["masks"], target["masks"])
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k serialize_instance -v`
Expected: FAIL — `AttributeError: _serialize_instance_entry`

- [ ] **Step 3: 实现序列化/反序列化**

加到 `seg_tools.py`（顶部 `from .seg_shared import rle_encode, rle_decode`）：

```python
def _serialize_instance_entry(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    pred_masks = prediction["masks"].detach().cpu().numpy().astype(bool)
    gt_masks = target["masks"].detach().cpu().numpy().astype(bool)
    return {
        "pred_labels": prediction["labels"].detach().cpu().to(rt.torch.int64).tolist(),
        "pred_scores": prediction["scores"].detach().cpu().to(rt.torch.float32).tolist(),
        "pred_masks_rle": [rle_encode(m) for m in pred_masks],
        "gt_labels": target["labels"].detach().cpu().to(rt.torch.int64).tolist(),
        "gt_masks_rle": [rle_encode(m) for m in gt_masks],
    }


def _stack_rle(rle_list: list[dict], *, height_width: tuple[int, int] | None) -> Any:
    if rle_list:
        masks = rt.np.stack([rle_decode(rle) for rle in rle_list])
        return rt.torch.as_tensor(masks, dtype=rt.torch.bool)
    h, w = height_width if height_width is not None else (1, 1)
    return rt.torch.zeros((0, h, w), dtype=rt.torch.bool)


def _deserialize_instance_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    all_rle = entry.get("pred_masks_rle", []) + entry.get("gt_masks_rle", [])
    hw = (all_rle[0]["size"][0], all_rle[0]["size"][1]) if all_rle else None
    prediction = {
        "labels": rt.torch.as_tensor(entry["pred_labels"], dtype=rt.torch.int64),
        "scores": rt.torch.as_tensor(entry["pred_scores"], dtype=rt.torch.float32),
        "masks": _stack_rle(entry["pred_masks_rle"], height_width=hw),
    }
    target = {
        "labels": rt.torch.as_tensor(entry["gt_labels"], dtype=rt.torch.int64),
        "masks": _stack_rle(entry["gt_masks_rle"], height_width=hw),
    }
    return prediction, target
```

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k serialize_instance -v`
Expected: PASS

- [ ] **Step 5: 加 shard 写出/合并 + 重构 `run_eval` 累积**

加 `_write_instance_shard_result(shard_dir, *, split, class_names, entries, images_with_labels, num_images, infer_time_sum_ms, failed)`（写 `seg_instance_shard_result.json`，结构同语义 shard 但用 `entries`）。

加 `_merge_instance_shard_results(shard_dirs, *, classwise) -> (metric_values, class_names, num_images, images_with_labels, infer_time_sum_ms, failed)`：合并所有 entries 与 class_names → `create_metric(class_names, classwise)` → 对每个 entry `_deserialize_instance_entry` 后 `metric.update_with_predictions(...)`（复用 `update_metric` 的 remap 逻辑：直接调用 `update_metric(metric, label_mapping, pred, tgt)`）→ `metric.compute_aggregated_values().metric_values`。

重构 `run_eval`（`seg_tools.py:397-448`）：把样本遍历抽成 `_accumulate_instance_eval(model, data_cfg, split, classwise, shard_index, num_shards)`，遍历前用 `filter_indices_for_shard` 取子集；每图 `try/except` 容错并计 `failed`；计时累加。shard 子进程模式写 entries；非 shard 模式直接喂 metric 并写 summary（summary 新增 `avg_infer_time_ms`、`failed_images`）。

- [ ] **Step 6: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add tool_lib/seg_tools.py tests/test_seg_parallel.py
git commit -m "feat(seg): RLE shard serialization + merge for instance eval"
```

---

## Phase 5：eval 并行入口 + CLI 旗标 + dispatch 路由

### Task 5.1：`eval` parser 新增 shard/dry-run 旗标

**Files:**
- Modify: `tool_lib/interactive.py:2689-2713`
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试**

```python
from tool_lib.interactive import parse_cli_args


def test_eval_parser_accepts_shard_flags():
    args = parse_cli_args([
        "eval", "--task", "seg", "--data", "d.yaml",
        "--shard-index", "1", "--num-shards", "3", "--dry-run",
        "--skip-important-artifacts",
    ])
    assert args.shard_index == 1
    assert args.num_shards == 3
    assert args.dry_run is True
    assert args.skip_important_artifacts is True


def test_eval_parser_shard_defaults():
    args = parse_cli_args(["eval", "--task", "seg", "--data", "d.yaml"])
    assert args.shard_index is None
    assert args.num_shards == 1
    assert args.dry_run is False
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k eval_parser -v`
Expected: FAIL — `AttributeError: 'Namespace' object has no attribute 'shard_index'`

- [ ] **Step 3: 实现**

在 `eval_parser`（`interactive.py:2713` 的 `--overwrite` 之后）追加：

```python
    eval_parser.add_argument("--dry-run", action="store_true", default=False)
    eval_parser.add_argument("--skip-important-artifacts", action="store_true", default=False, help=argparse.SUPPRESS)
    eval_parser.add_argument("--selected-splits", type=str, default=None, help=argparse.SUPPRESS)
    eval_parser.add_argument("--multi-output-root", type=Path, default=None, help=argparse.SUPPRESS)
    eval_parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    eval_parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
```

并确认交互模式构造 eval args 的 `Namespace`（搜索 `build_interactive_args` 里 eval 分支）补齐 `shard_index=None, num_shards=1, dry_run=False, skip_important_artifacts=False, selected_splits=None, multi_output_root=None`，避免交互路径缺字段。

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k eval_parser tests/test__cli.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tool_lib/interactive.py tests/test_seg_parallel.py
git commit -m "feat(seg): add shard/dry-run flags to eval parser"
```

---

### Task 5.2：`run_parallel_seg_eval` 调度 + dispatch shard 路由

**Files:**
- Modify: `tool_lib/seg_tools.py`（新增 `_is_seg_shard_child`、`run_parallel_seg_eval`、`_build_seg_eval_child_command`，改 `run_eval`/`run_semantic_eval` 入口）
- Modify: `tool_lib/dispatch.py:70-78`
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试（mock GPU 清单：单卡退化返回 False）**

```python
from types import SimpleNamespace


def test_run_parallel_seg_eval_falls_back_when_few_gpus(monkeypatch):
    monkeypatch.setattr(
        seg_tools.gpu_parallel, "query_gpu_inventory", lambda: ([{"index": 0, "used_ratio": 0.1, "memory_used": 0, "memory_total": 1, "utilization": 0}], "")
    )
    args = SimpleNamespace(device="auto", shard_index=None, num_shards=1, dry_run=False, data="d.yaml")
    assert seg_tools.run_parallel_seg_eval(args) is False


def test_run_parallel_seg_eval_skips_when_device_fixed():
    args = SimpleNamespace(device="cuda:0", shard_index=None, num_shards=1, dry_run=False, data="d.yaml")
    assert seg_tools.run_parallel_seg_eval(args) is False
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k run_parallel_seg_eval -v`
Expected: FAIL — `AttributeError: run_parallel_seg_eval`

- [ ] **Step 3: 实现调度入口**

在 `seg_tools.py` 顶部加 `from . import gpu_parallel` 与 `import os, subprocess, sys, tempfile`。新增：

```python
def _is_seg_shard_child(args) -> bool:
    return getattr(args, "shard_index", None) is not None and int(getattr(args, "num_shards", 1) or 1) > 1


def _build_seg_eval_child_command(args, *, shard_index, num_shards, output_dir, device="auto") -> list[str]:
    command = [sys.executable, str(rt.ROOT_DIR / "launcher.py"), "eval", "--task", "seg"]
    command += ["--seg-train-type", str(getattr(args, "seg_train_type", "instance"))]
    if getattr(args, "experiment_dir", None) is not None:
        command += ["--experiment-dir", str(args.experiment_dir)]
    if getattr(args, "checkpoint", None) is not None:
        command += ["--checkpoint", str(args.checkpoint)]
    command += ["--data", str(args.data)]
    for split in _normalize_splits(args.split):
        command += ["--split", split]
    command += ["--output-dir", str(output_dir), "--device", device]
    command += ["--shard-index", str(shard_index), "--num-shards", str(num_shards)]
    command += ["--skip-important-artifacts"]
    if getattr(args, "classwise", False):
        command += ["--classwise"]
    if getattr(args, "overwrite", False):
        command += ["--overwrite"]
    return command


def run_parallel_seg_eval(args) -> bool:
    if getattr(args, "device", "auto") != "auto":
        return False
    if _is_seg_shard_child(args) or getattr(args, "dry_run", False):
        return False
    all_gpus, message = gpu_parallel.query_gpu_inventory()
    if message:
        print(f"[seg/eval] {message}")
        return False
    eligible = gpu_parallel.filter_high_memory_gpus(all_gpus)
    if len(eligible) < 2:
        print(f"[seg/eval] 可用 GPU 数量为 {len(eligible)}，进入单卡顺序模式。")
        return False
    # 真正分片在 Task 5.3 串起；此处先满足退化语义，返回 False 由 Task 5.3 替换实现体。
    return False
```

（说明：本 step 仅落地"退化判定"，让两个退化测试通过；完整分片在 Step 5 接上。）

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k run_parallel_seg_eval -v`
Expected: PASS

- [ ] **Step 5: 接上完整分片体**

把 `run_parallel_seg_eval` 末尾的 `return False` 替换为完整分片：解析 checkpoint，`build` 各 shard 输出目录于 `tempfile`/最终目录下的 `_shards/shard_NN`，对每个 split 与每个 eligible GPU 构造 `_build_seg_eval_child_command`，组 `jobs` 调 `gpu_parallel.run_sharded_subprocesses(jobs, cwd=rt.ROOT_DIR)`；失败则 `raise RuntimeError`。子进程完成后，按 `seg_train_type` 调 `_merge_semantic_shard_results` 或 `_merge_instance_shard_results`，写最终 summary（与串行同名、同 schema），再（若非 skip）写 `run_meta.json`。返回 `True`。

子进程侧：`run_eval`/`run_semantic_eval` 在 `_is_seg_shard_child(args)` 为真时，走"累积 → 写 shard 结果"分支（Task 3/4 已备好累积函数与写出函数），不写最终 summary。

- [ ] **Step 6: 入口接线**

`run_eval`（`seg_tools.py:397`）与 `run_semantic_eval`（`seg_tools.py:367`）开头加：

```python
    if not _is_seg_shard_child(args) and run_parallel_seg_eval(args):
        return
```

（注意：`run_eval` 已先判 semantic 转发；并行判定应在转发**之前**做一次即可，避免双判。把并行入口放在 `run_eval` 顶部、`_normalize_seg_type` 之后、semantic 转发之前。）

- [ ] **Step 7: dispatch 路由（子进程通过 `launcher.py eval` 进来即可，无需特判）**

确认 `dispatch.py:70-78` 的 seg eval 分支对 shard 子进程透明（它们带 `--shard-index`，`run_eval`/`run_semantic_eval` 内部据此走 shard 分支）。无需改 dispatch 逻辑；若 `run_semantic_eval` 的多 split 在 shard 模式下需要 `selected-splits`，在 `_build_seg_eval_child_command` 已透传 `--split` 多值，足够。

- [ ] **Step 8: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py tests/test_seg_semantic_eda_curate.py -v`
Expected: PASS

- [ ] **Step 9: 提交**

```bash
git add tool_lib/seg_tools.py tool_lib/dispatch.py tests/test_seg_parallel.py
git commit -m "feat(seg): multi-GPU sharded eval with load-aware fallback"
```

---

## Phase 6：infer 并行入口

### Task 6.1：seg infer 多卡分片（按图分片，汇集可视化）

**Files:**
- Modify: `tool_lib/seg_tools.py`（新增 `run_parallel_seg_infer`，改 `run_infer`）
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试（退化判定）**

```python
def test_run_parallel_seg_infer_falls_back_single_gpu(monkeypatch):
    monkeypatch.setattr(
        seg_tools.gpu_parallel, "query_gpu_inventory",
        lambda: ([{"index": 0, "used_ratio": 0.1, "memory_used": 0, "memory_total": 1, "utilization": 0}], ""),
    )
    args = SimpleNamespace(device="auto", shard_index=None, num_shards=1, dry_run=False)
    assert seg_tools.run_parallel_seg_infer(args) is False
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k run_parallel_seg_infer -v`
Expected: FAIL — `AttributeError: run_parallel_seg_infer`

- [ ] **Step 3: 实现**

新增 `run_parallel_seg_infer(args) -> bool`：退化判定与 `run_parallel_seg_eval` 一致（device 固定 / shard 子进程 / dry-run / <2 卡 → False）。满足时：解析输入图列表（复用 `_seg_infer_image_paths` 配合 `filter_indices_for_shard` 由子进程各取子集），子进程 `launcher.py infer --task seg ... --shard-index i --num-shards n --output-dir <shard>`，父进程 `gpu_parallel.run_sharded_subprocesses`，完成后把各 shard `output_dir` 内容用 `rt.copy_tree_contents`（det 同款，见 `det_infer.copy_tree_contents`）汇集到最终目录。返回 `True`。

`run_infer`（`seg_tools.py:207`）开头加：

```python
    if not _is_seg_shard_child(args) and run_parallel_seg_infer(args):
        return
```

子进程侧：`run_infer` 在 shard 模式下用 `gpu_parallel.filter_indices_for_shard(len(image_paths), ...)` 取子集，写入各自 `output_dir`，不做汇集。

注意 `infer` parser 已含 `--shard-index/--num-shards`（`interactive.py:2686-2687`），无需改 CLI。

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k run_parallel_seg_infer -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tool_lib/seg_tools.py tests/test_seg_parallel.py
git commit -m "feat(seg): multi-GPU sharded infer"
```

---

## Phase 7：吞吐——预取重叠

### Task 7.1：图片/标签预取迭代器，eval/infer 内循环接入

**Files:**
- Modify: `tool_lib/seg_tools.py`（新增 `_prefetch_iter`）
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试（顺序与产出不变）**

```python
def test_prefetch_iter_preserves_order_and_payload():
    def _load(x):
        return x * 10
    items = [1, 2, 3, 4]
    out = list(seg_tools._prefetch_iter(items, _load))
    assert out == [(1, 10), (2, 20), (3, 30), (4, 40)]


def test_prefetch_iter_empty():
    assert list(seg_tools._prefetch_iter([], lambda x: x)) == []
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k prefetch -v`
Expected: FAIL — `AttributeError: _prefetch_iter`

- [ ] **Step 3: 实现**

```python
def _prefetch_iter(items: list[Any], load_fn):
    """单 worker 预取：在消费当前项时后台解码下一项。产出 (item, loaded) 保序。"""
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_fn, items[0])
        for index in range(len(items)):
            loaded = future.result()
            if index + 1 < len(items):
                future = executor.submit(load_fn, items[index + 1])
            yield items[index], loaded
```

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k prefetch -v`
Expected: PASS

- [ ] **Step 5: 接入累积循环**

在 `_accumulate_semantic_confusion` 与 `_accumulate_instance_eval` 的样本遍历中，用 `_prefetch_iter` 包裹样本列表，`load_fn` 负责 `Image.open(...).convert("RGB")`（语义额外预读 mask、实例额外读 label），predict 时把预解码 PIL 图传给 `predict_model`（`predict_model` 已接受路径或 PIL；确认 `model.predict` 支持 PILImage——上游签名为 `PathLike | PILImage | Tensor`，成立）。保持产物与顺序不变。

- [ ] **Step 6: 运行回归**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py tests/test_seg_semantic_eda_curate.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add tool_lib/seg_tools.py tests/test_seg_parallel.py
git commit -m "perf(seg): overlap image decode with GPU via prefetch"
```

---

## Phase 8：工程对齐——run_meta / dry-run

### Task 8.1：seg eval/infer 的 run_meta 与 dry-run

**Files:**
- Modify: `tool_lib/seg_tools.py`（新增 `_write_seg_run_meta`、dry-run 打印）
- Test: `tests/test_seg_parallel.py`

- [ ] **Step 1: 写失败测试**

```python
def test_write_seg_run_meta_records_core_fields(tmp_path):
    args = SimpleNamespace(
        seg_train_type="semantic", data="d.yaml", split=["test"],
        device="auto", overwrite=False, threshold=None,
    )
    path = tmp_path / "run_meta.json"
    seg_tools._write_seg_run_meta(
        path, action="eval", checkpoint_path=tmp_path / "ck.pt",
        output_dir=tmp_path, args=args, num_images=12, device_mode="cuda:0",
    )
    import json
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "seg"
    assert payload["action"] == "eval"
    assert payload["num_images"] == 12
    assert payload["settings"]["device_mode"] == "cuda:0"
```

- [ ] **Step 2: 运行验证失败**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k run_meta -v`
Expected: FAIL — `AttributeError: _write_seg_run_meta`

- [ ] **Step 3: 实现**

```python
def _write_seg_run_meta(meta_path, *, action, checkpoint_path, output_dir, args, num_images, device_mode):
    payload = {
        "task": "seg",
        "action": action,
        "created_at": rt.timestamp_now_iso(),
        "seg_train_type": str(getattr(args, "seg_train_type", "instance")),
        "split": getattr(args, "split", None) if isinstance(getattr(args, "split", None), str) else _normalize_splits(getattr(args, "split", [])),
        "num_images": int(num_images),
        "paths": {
            "output_dir": str(output_dir),
            "checkpoint_path": str(checkpoint_path),
            "data": str(getattr(args, "data", None)) if getattr(args, "data", None) is not None else None,
        },
        "settings": {
            "device": getattr(args, "device", None),
            "device_mode": device_mode,
            "threshold": getattr(args, "threshold", None),
            "overwrite": getattr(args, "overwrite", None),
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
```

- [ ] **Step 4: 运行测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py -k run_meta -v`
Expected: PASS

- [ ] **Step 5: 接线 run_meta 与 dry-run**

非 shard 子进程、非 dry-run 的 `run_eval`/`run_semantic_eval`/`run_infer` 在写产物时调用 `_write_seg_run_meta(output_dir / "run_meta.json", ...)`，`device_mode` 取实际选定的设备字符串（单卡退化时来自 `select_fallback_single_gpu`，多卡时为 `"sharded"`）。

dry-run：`run_eval`/`run_semantic_eval`/`run_infer` 在解析完输入、加载模型**之前**，若 `getattr(args, "dry_run", False)`，打印计划（checkpoint、seg_train_type、split、样本数、device 模式、计划产物路径）后 `return`。

- [ ] **Step 6: 单卡负载感知选卡接线**

`run_eval`/`run_semantic_eval`/`run_infer` 在 `device=="auto"` 且并行未启用（`run_parallel_*` 返回 False）时，调 `gpu_parallel.query_gpu_inventory` + `filter_high_memory_gpus` + `select_fallback_single_gpu`，把返回的 `cuda:N` 作为实际 device 传给 `rt.resolve_device`；并打印所选卡摘要。CUDA 不可用时维持现状（`select_fallback_single_gpu` 返回 None → 走 `resolve_device("auto")`）。

- [ ] **Step 7: 运行全量 seg 相关回归**

Run: `conda run -n lightlytrain python -m pytest tests/test_seg_parallel.py tests/test_seg_semantic_eda_curate.py tests/test_gpu_parallel.py -v`
Expected: PASS

- [ ] **Step 8: 提交**

```bash
git add tool_lib/seg_tools.py tests/test_seg_parallel.py
git commit -m "feat(seg): run_meta, dry-run, load-aware single-GPU selection"
```

---

## 最终验收

- [ ] **全量测试**

Run: `conda run -n lightlytrain python -m pytest tests/test_gpu_parallel.py tests/test_seg_parallel.py tests/test_det_infer.py tests/test_seg_semantic_eda_curate.py tests/test__cli.py -v`
Expected: 全 PASS

- [ ] **真实环境冒烟（有 GPU 时手动）**
  - 语义：`conda run -n lightlytrain python launcher.py eval --task seg --seg-train-type semantic --data <semantic.yaml> --experiment-dir <exp> --split val test`
  - 实例：`conda run -n lightlytrain python launcher.py eval --task seg --seg-train-type instance --data <seg.yaml> --experiment-dir <exp>`
  - 对比：在 `--device cuda:0`（强制单卡）与默认 auto（多卡）下，summary 的 mIoU / mAP 应一致。
