# 尺寸感知导出（size-aware export）设计

- 日期：2026-06-03
- 作者：zhangzhuorui5850 + Claude
- 状态：待评审

## 1. 背景与问题

检测/分割数据集导出（`tool_lib/det_export.py`、`tool_lib/seg_export.py`）当前按**类别**做均衡选图，并对单图框/实例密度做**硬上限**控制（`max_boxes_per_image` 等）。目标尺寸在导出流程里**只被统计、写进 EDA 报告，从不作为选图或划分目标**（见 [det_export.py:156-175,440-444](../../../tool_lib/det_export.py)）。当前分桶只有 3 档（small `<32²` / medium `32²~96²` / large `≥96²`），且把所有 `<32²` 的框都算作 small。

后果：导出的均衡集尺寸分布完全由源数据决定，常见的小目标占比偏低；train/val/test 各 split 的尺寸分布也不受控。

已有一个独立后处理工具 `tool_lib/det_size_supplement.py`（贪心赤字驱动，从源池增量补小/中目标）能做尺寸 shaping，但它是 export **之后**的单独一步。本设计把尺寸感知**内化进 export 本身**，让 export 一步直接产出尺寸均衡、且每个 split 尺寸一致、平均框密度落带的数据集。

> 用户决策（已确认）：
> 1. 尺寸目标**内化进 export 选图阶段**（不是只做后处理）。
> 2. 目标尺寸比例与密度做成**可配置参数 + 给默认值**。
> 3. 选图阶段类别与尺寸冲突时**联合加权**（两者同等重要、权重可调）。
> 4. "平均 5-15 GT/图"按**数据集平均软目标**落地（保留每图硬上限，新增软导向）。
> 5. **det + seg 同期实现**，写进同一份 spec。
> 6. 选图引擎走**方案 A 双引擎**：尺寸目标关闭时走现有 CELF（零行为变化），开启时切到分批赤字贪心。

## 1b. 尺寸分桶口径（4 档，重要）

按框/实例的**像素面积**分 4 档，阈值 16² / 32² / 96²：

| 档 | 面积区间 | 处理 |
|---|---|---|
| `tiny` | `< 16²`（<256px） | **只统计，不进比例分母**。极小框基本是噪声/难标，模型也学不动，不让它占小目标配额 |
| `small` | `16² ≤ area < 32²` | "真·小目标"，进比例 |
| `medium` | `32² ≤ area < 96²` | 主力，进比例 |
| `large` | `≥ 96²` | 进比例 |

- `size_ratio="25:50:25"` 的分母 = `small + medium + large` 框数（**排除 tiny**）。
- tiny 框仍然照常**保留在导出的标签里**（不删框），只是不参与尺寸占比目标。
- 需要把现有 `_bucket_box_area`（det）/ `_bucket_mask_area`（seg）从 3 档扩成 4 档，并保证依赖它的 EDA、size-supplement 行为相应更新（EDA 表新增 tiny 列）。

> 向后兼容澄清：本设计承诺的"逐字节一致"针对**数据集内容**（选中哪些图、各 split 归属、重映射后的标签）。EDA 报告文本会新增 tiny 列属于**报告外观变化**，不影响数据集本身。

## 2. 关键技术约束

1. **`ExportImageCandidate` 不携带尺寸信息**——只有 `class_box_counts`（[det_shared.py:29-42](../../../tool_lib/det_shared.py)）。尺寸桶目前在 EDA 阶段才通过打开图片计算。尺寸感知选图必须新增一个**带缓存的尺寸桶扫描**。
2. **现有选图器是 CELF 懒贪心**（[det_analysis.py:1422](../../../tool_lib/det_analysis.py)），其正确性**依赖 `score_export_candidate` 单调不增**（类别框赤字只减不增）。尺寸占比目标与平均 GT/图目标都依赖**全局已选比例**，**非单调**，直接塞进 CELF 会破坏其正确性。故采用双引擎。
3. **seg 已具备平行构件**：`SegExportImageCandidate`、`polygon_area_normalized`（shoelace）、`_bucket_mask_area`（同 32²/96² 阈值，掩码面积口径），且 seg_export **已复用** det_analysis 的 `select_balanced_train_candidates` / `assign_candidates_to_new_splits`（duck typing）。因此把 det_analysis 的选图/划分函数改造成接受**通用的"候选→尺寸桶"映射**后，seg 只需喂入掩码面积桶即可复用同一路径。

## 3. 设计目标与非目标

**目标**
- export 在开启尺寸目标时，产出的数据集按**框/实例数量计**的小/中/大占比逼近用户指定比例。
- train/val/test **每个 split** 的尺寸分布都贴近整体目标。
- 选中样本的**平均 GT/图**落在 `[min,max]` 区间（软导向）。
- 类别均衡与尺寸均衡**联合加权**，权重可调。
- 尺寸目标**留空时，行为与今天逐字节一致**（CELF 路径不变）。
- det 与 seg 共用同一套核心逻辑。

**非目标（YAGNI / 本期外）**
- 整数规划/精确最优求解（沿用贪心）。
- 按图片数口径的尺寸占比（统一用框/实例数量口径，与现有 `size_bucket_ratio` 一致）。
- 改动 optimize（合并/删类）流程。
- 废弃 size-supplement（它退居"补丁/二次微调"角色，保留）。

## 4. 新增配置（launcher，可配置 + 默认；det 与 seg 同构）

det 侧（`launcher.py` DEFAULT_SETTINGS）：

| 键 | 默认 | 语义 |
|---|---|---|
| `det_export_size_ratio` | `"30:40:30"` | **默认开启**=小:中:大目标占比；填 `""` 才回退现状 CELF。解析复用 `det_size_supplement.parse_size_ratio` |
| `det_export_size_balance_weight` | `1.0` | 尺寸赤字相对类别赤字的联合权重 |
| `det_export_avg_boxes_per_image_min` | `5` | 平均 GT/图 软目标下限；0=关 |
| `det_export_avg_boxes_per_image_max` | `15` | 平均 GT/图 软目标上限；0=关 |

seg 侧镜像（关键词"框"→"实例"，"boxes"→"instances"）：
`seg_export_size_ratio`、`seg_export_size_balance_weight`、`seg_export_avg_instances_per_image_min`、`seg_export_avg_instances_per_image_max`，默认值同上。

> seg 的"尺寸" = 实例**掩码面积**（`polygon_area_normalized × 图面积` → `_bucket_mask_area`），不是框 w·h。两侧阈值一致（32²/96²px）。

新增的 `args` 透传链：`launcher → build_user_settings → dispatch/interactive → run_export → export_filtered_dataset`。det 与 seg 各自的 `run_export` 都要把新参数透传进核心管线。

## 5. 选图：尺寸/密度感知（det_analysis）

### 5.1 候选尺寸桶扫描（新增，带缓存）

新增函数（det_export 或新模块），对 filtered 候选逐图计算 `size_buckets={tiny,small,medium,large}`（tiny 仅统计，比例计算时排除）：
- 复用 `det_size_supplement.ImageSizeCache`（键=绝对路径+mtime，落盘 `.imgsize_cache.json`）拿图片像素尺寸。
- 复用 `det_size_supplement.bucket_label_lines` 对 YOLO 行分桶。
- 产出 `size_buckets_by_candidate: dict[candidate_key, dict[str,int]]`（key 用 `rel_path.as_posix()+split`，与候选稳定对应）。

seg 侧用 `_bucket_mask_area + polygon_area_normalized` 产出同结构的 map。两侧把这个 map 作为参数传给下面的通用选图/划分函数。

### 5.2 双引擎入口

`select_balanced_train_candidates` 增加可选参数：
```python
size_buckets_by_candidate: dict[str, dict[str,int]] | None = None,
target_size_ratio: dict[str,float] | None = None,
size_balance_weight: float = 0.0,
avg_boxes_per_image_min: float = 0.0,
avg_boxes_per_image_max: float = 0.0,
```
- 当 `target_size_ratio` 为 None/空 → **完全走现有 CELF 路径**（既有代码、既有结果，零变化）。
- 否则 → 走新的**分批赤字贪心**（§5.3）。

### 5.3 分批赤字贪心（尺寸模式）

直接搬 `det_size_supplement.select_supplement_candidates` 的批处理结构，扩展打分项。每批开始按当前已选状态重算：

```
score(cand) =  类别赤字项                       # 原 score_export_candidate 的类别短缺逻辑
             + size_balance_weight · 尺寸桶赤字项 # Σ_b max(0, r_b − cur_prop_b)·buckets_b[cand]
             − over_penalty · 尺寸桶超标惩罚      # Σ_b max(0, cur_prop_b − r_b)·buckets_b[cand]
             + density_term(cand)                # §5.4
```
- `cur_prop_b` = 当前已选集合按框数量计的 b 桶占比，**分母排除 tiny**（b ∈ {small,medium,large}）。
- 类别赤字项沿用现有口径（`desired_box_counts − selected_box_counts`，availability 归一）。
- 批大小约 200；每批选 top、更新状态、重算，直到达 `target_total_images`（或类别+尺寸双双达标、或候选耗尽）。
- 候选耗尽仍未达比例 → 记 `size_shortfall`，不报错，按已选产出。

### 5.4 密度软目标（平均 GT/图）

`density_term` 按当前已选样本的**平均 GT/图** `cur_avg` 偏离 `[min,max]` 的方向给奖惩：
- `cur_avg < min`：奖励 `total_boxes` 多的候选（推平均往上）。
- `cur_avg > max`：奖励 `total_boxes` 少的候选（推平均往下）。
- 落在带内：该项为 0。
- 量纲与赤字项可比，权重内置常数（可后续提为参数，本期不暴露）。
- 现有 `max_boxes_per_image` / `box_density_penalty` 硬上限**保留不动**，与软目标叠加。

## 6. 划分：每个 split 尺寸均衡（det_analysis）

现有 `assign_candidates_to_new_splits` 只按**类别框数**在 split 间均衡。新增：
1. `allocate_size_bucket_targets_per_split(selected_candidates, size_buckets_by_candidate, split_image_targets)`：按 split 图数比例给每 split 算 `{small,medium,large}` 目标桶数，用与 `allocate_box_targets_per_split` 相同的 Hare-Niemeyer 余数法。
2. 在 `assign_candidates_to_new_splits` 主分配循环的打分里**追加一项 size-bucket coverage gain**：候选放入某 split 时，若能补该 split 仍亏空的尺寸桶则加分。类别覆盖优先级（现有 `allocate_image_coverage_targets_per_split`）保持不变，尺寸项作为附加权重。
3. 当 `size_buckets_by_candidate` 为 None（尺寸模式关）→ 该项不参与，划分行为与今天一致。

## 7. seg 对齐（seg_export / seg_analysis）

- seg_export 新增与 det 同构的参数透传与尺寸桶扫描（用掩码面积桶）。
- 复用同一份改造后的 det_analysis 选图/划分函数，喂入 seg 的 `size_buckets_by_candidate`。
- seg 的 `_bucket_mask_area`、`polygon_area_normalized` 已存在，直接用。
- seg EDA 已有尺寸桶表，补 before/after 对照同 det。

## 8. 报告

- `export_summary.json` 增加：`size_ratio`(requested/effective)、`achieved_size_ratio`（small/medium/large，排除 tiny）、`tiny_box_count`（单列统计）、`size_shortfall`、`avg_boxes_per_image`(min/max/achieved)、每 split 尺寸占比。
- EDA markdown 增加 tiny 列、"目标 vs 实际尺寸占比"对照段与平均密度落点（参照 `det_size_supplement._render_supplement_markdown` 风格）。

## 9. 错误处理与兼容

- `size_ratio` 解析失败 → 报错并提示格式（复用 `parse_size_ratio` 的报错）。
- `avg_*_min > max`（且都 >0）→ 报错。
- 源池尺寸供给不足 → **不报错**，记 `size_shortfall` 产出。
- 缺 Pillow → 复用 `rt.ensure_plot_dependencies()` 报错路径。
- **默认行为变化（已确认）**：`size_ratio` 默认 `"30:40:30"`（**默认开启尺寸感知**），所以默认导出结果**会与改动前不同**——这正是本特性的目的。
- **可退回保证**：把 `size_ratio` 显式填 `""` 时，det 与 seg 的导出结果与改动前**逐字节一致**（CELF 路径与现有划分完全不走新分支）。即旧行为仍可一键取回。

## 10. 测试策略

用小型合成数据集（十几张图、人工构造已知尺寸的框/多边形）：
1. 分桶边界正确（16²、32²、96² 临界），tiny 不进比例分母。
2. **回归保护**：`size_ratio` 关闭时，选图与划分结果与现有实现逐字节一致（直接对比 CELF 输出）。
3. 尺寸模式下，达到的小/中/大占比朝目标**单调推进**，源池受限时正确记 `size_shortfall`。
4. 每个 split 的尺寸占比近似一致（在容差内）。
5. 平均 GT/图 落入 `[min,max]` 带（供给允许时）。
6. seg 掩码面积分桶正确、seg 走同路径产出尺寸均衡集。
7. 图尺寸缓存命中与 mtime 失效正确。
8. 原数据集目录零改动。

## 11. 实现切分（同 spec，det 先落地）

1. det_analysis：选图/划分函数加 size 参数（默认关=现状）+ 分批贪心引擎 + split 尺寸目标。
2. det_export：尺寸桶扫描 + 参数透传 + 报告增强。
3. launcher + dispatch/interactive：det 新参数。
4. det 测试（含回归保护）。
5. seg_export + seg launcher 参数：喂掩码面积桶走同路径。
6. seg 测试。
