# 语义分割数据集 EDA + 交互式类别整理 — 设计文档

- 日期：2026-06-13
- 任务：为**语义分割**数据集提供「详细 EDA → 交互式删除类别 / 按图片数压缩超阈值类 → 导出新数据集」的两步式工具链。
- 范围：仅语义分割（mask PNG 标签）。实例分割已有 `seg_export.py`，不在本设计内。

## 1. 背景与动机

现有代码里：

- `det_eda.py` 是检测 EDA 的成熟范式：扫描数据集，产出 JSON + Markdown + CSV，包含按 split / 类别的分布、不平衡指标（Gini / 熵 / CV / JS 漂移）、问题样本导出。
- `seg_export.py` 已为**实例分割**（polygon / YOLO 标签）做类别过滤 + 平衡，复用 `det_analysis` 的 box 逻辑。
- **语义分割**只有 `eval` 路径，没有 EDA，也没有类别编辑 / 阈值处理工具。

语义分割的标签是 **mask 图片**，像素值编码类别（`data.yaml` 的 `classes` 给出 id→名称 + label 值/RGB 元组，`ignore_classes` 列出忽略类）。它与检测/实例分割是**逐像素范式**，不能像「删一个 box」那样删单个像素，因此需要专门设计。

关键代码事实（已核对）：

- 语义 mask 加载器 `seg_tools._load_semantic_mask` 用 `original_to_internal = enumerate(sorted(set(classes) - ignore_classes))` 建立**内部连续映射**；**不在保留类列表里的像素值会保持初始值 -100（即被忽略）**。
- 因此「删除一个类」= 在新 `data.yaml` 的 `classes` 里去掉它即可，**mask 不用重写、像素值不用重编号**。
- (图, mask) 配对逻辑见 `seg_tools._semantic_image_mask_samples`；语义配置加载见 `train_tools.load_semantic_segmentation_data_config` / `load_semantic_segmentation_split_config`。
- dispatch 约定：`tool_task` + `tool_action`（`tool_lib/dispatch.py`）。seg 现有 `infer/eval/export`；det 有 `eda`。本设计新增 seg 的 `eda`（语义）与 `curate` 两个 action。

## 2. 已确定的决策

| 维度 | 决策 | 说明 |
|---|---|---|
| 统计 / 处理口径 | **按图片数** | 每类「出现在多少张图里」。压缩 = 对含该类的图片随机/启发式下采样到阈值张数。「几万」≈ 图片张数。 |
| 交互方式 | **两步式** | EDA 报告（落盘 md/json/csv）→ CLI 输入类别 ID + 阈值 → 导出新数据集。Linux 命令行，输入 ID（逗号分隔）+ 数字，非网页勾选。 |
| 删除作用范围 | **全 split** | train/val/test 都生效，该类彻底消失。 |
| 压缩作用范围 | **仅 train** | val/test 保持完整，保证评估公平。 |
| 删除语义 | **当作 ignore** | 从新 yaml 的 `classes` 去掉（加载器自动忽略）；不污染背景类；mask 不重写。 |
| 压缩策略 | **共现感知，保护稀有类** | 下采样时优先丢「只含优势类、不含稀有类」的图，尽量不连带损伤其他类。 |
| 架构 | **方案 A** | 两个独立新模块 + 持久化逐图类别清单 CSV，分析/交互解耦。 |

## 3. 总体架构

```
seg eda (semantic)                      seg curate
┌─────────────────────────┐            ┌──────────────────────────────┐
│ seg_semantic_eda.py      │            │ seg_semantic_curate.py        │
│  扫描 mask (一次性、贵)   │  inventory │  读 inventory CSV (秒级)       │
│  → 报告 md/json/csv       │ ─────────▶ │  → 交互选删除类 + 阈值        │
│  → image_class_inventory  │   CSV      │  → 共现感知下采样 train       │
│  → 推荐删除类 / 推荐阈值   │            │  → 硬链接 materialize 新数据集 │
└─────────────────────────┘            └──────────────────────────────┘
```

分析与交互解耦：贵的 mask 扫描只在 EDA 做一次并落盘逐图清单；curate 读清单做选择与导出，秒级响应，可反复试不同删除/阈值组合。

## 4. 模块 1 — `tool_lib/seg_semantic_eda.py`

### 4.1 入口

`run_semantic_eda(args)` → `generate_semantic_eda_report(*, source_data_path, output_dir, overwrite)`，由 `dispatch.py` 在 `seg` + `eda` + `--seg-train-type semantic` 时调用。

### 4.2 流程

1. 用 `train_tools.load_semantic_segmentation_data_config` 加载 `classes`（id→名称 + label 值/RGB 元组）、`ignore_classes`、各 split 的 images/masks 目录；用 `seg_tools._semantic_image_mask_samples` 配对 (图, mask)。
2. 逐 mask：
   - 打开 mask 取 `np.array`；单通道索引 mask → `np.unique(return_counts=True)` 得到像素值→计数。
   - RGB 元组 mask → 走 `_class_labels` 的 label 映射，把元组归到类 id（与加载器口径一致）。
   - 该图：含哪些类 id（集合）、各类像素数、总标注像素、ignore 像素、图像宽高。
3. 聚合（all + 每 split）：
   - 每类：**图片数**、像素数、图片占比、像素占比、出现/缺失于哪些 split。
   - 不平衡指标：对「每类图片数」和「每类像素数」两套口径各算 max/min、median、p90、Gini、熵均衡度、CV。**复用** `det_eda` 的 `_distribution_metrics / _gini / _entropy_evenness / _distribution / _js_divergence`（直接 import；这些是纯函数，无 det 语义耦合）。
   - 跨 split：每类图片分布的 JS 漂移。
4. 推荐（写进报告，curate 默认值取自这里）：
   - **推荐删除**：全局图片数 < `min_class_images`（默认下限，初值 10，可 `--min-class-images` 配）的类；以及缺失于 val/test（无法评估）的类，单独标记。
   - **推荐压缩阈值**：train「每类图片数」的 **p90**（向上取整到可读数；可 `--threshold-percentile` 配）。train 图片数 > 阈值的类标为压缩候选。

### 4.3 产物（写入 EDA 输出目录，命名/去重沿用 `det_eda` 的 `_default_eda_output_dir` 风格）

- `semantic_eda_<tag>.json` — 完整报告（含 overview / splits / classes / imbalance / recommendations）。
- `semantic_eda_<tag>.md` — 人读报告，含**带编号类别表**（`# | id | 名称 | train 图数 | all 图数 | 图片占比 | 像素占比 | 出现 split | 标记`）+ 推荐删除/阈值。
- `class_summary_<tag>.csv` — 每类每 split 的图片数/像素数/占比 + 推荐标记。
- `split_summary_<tag>.csv` — 每 split 概览。
- **`image_class_inventory_<tag>.csv`**（curate 消费的关键中间产物）：
  - 列：`split, image, image_path, mask_path, width, height, class_ids, per_class_pixels, total_labeled_pixels`
  - `class_ids` 用 `|` 连接（如 `0|3|7`）；`per_class_pixels` 用 JSON 串（如 `{"0":1234,"3":56}`）。
- 控制台：打印类别表 + 推荐 + inventory CSV 路径，提示下一步可运行 curate。

## 5. 模块 2 — `tool_lib/seg_semantic_curate.py`

### 5.1 入口

`run_semantic_curate(args)`，由 `dispatch.py` 在 `seg` + `curate` 时调用。

输入来源（按序）：
1. 显式 `--eda-dir` 指向的 EDA 输出目录里的 inventory CSV；
2. 未指定时，自动找该数据集最近一次 EDA 的 inventory；
3. 找不到时回退：现场扫描 mask（与模块 1 共用扫描函数），但提示用户「建议先跑 eda」。

### 5.2 交互（复用 `interactive.py` 的 `prompt_text / prompt_int`，逗号解析参照 GPU 选择写法）

1. 打印带编号类别表，对推荐项标 `[建议删除]` / `[建议压缩]`，并显示推荐阈值。
2. 输入删除 ID：`prompt_text("输入要删除的类别 ID(逗号分隔)", default=<推荐稀有类>)`；解析、去重、校验是否为已知 id；非法报错重输。
3. 输入压缩阈值：`prompt_int("压缩阈值(train 每类最多保留图片数), 0=不压缩", default=<推荐值>)`。
4. 删除语义固定为 ignore（按决策，不再追问）。
5. 计算保留类 = 全部类 − 删除类。

### 5.3 共现感知下采样（仅 train）

1. 从 inventory 重算：删除类后，每张 train 图的「保留类集合」与每个保留类的 train 图片数。
2. 标记**超阈值类**（excess）= train 图片数 > 阈值的保留类。无 excess（阈值=0 或无类超）则跳过下采样。
3. **可丢池** = 保留类集合 ⊆ excess 的 train 图（丢它只减 excess 类，不碰任何稀有/非 excess 类）。
4. 在可丢池里按「覆盖最多 excess 类、最少稀有类」排序，贪心丢弃，直到每个 excess 类的 train 图片数 ≤ 阈值。
5. 若可丢池耗尽但仍有 excess 类超阈值 → **停止并告警**（不牺牲稀有类去硬压），`curate_manifest.json` 记录残余超额。
6. 该贪心为图片数口径的专用实现；不强行复用 `det_analysis.select_balanced_train_candidates`（那是 box-count 口径），仅在思路上对齐。

### 5.4 materialize 新数据集（自包含，硬链接）

- 新目录：`<source_root>__curated`（用 `rt.deduplicate_path` 去重）。
- `train`：选中图 + 对应 mask；`val/test`：全量图 + mask。
- 链接方式：**硬链接**（复用现有数据集转换的硬链接工具，见 `convert_tools` / `file_helpers`；跨设备回退拷贝）。
- 新 `data.yaml`：
  - `classes` 只保留「保留类」（删除类省略 → 加载器自动 ignore）；
  - **像素值不重写、不重编号**（加载器内部建连续映射）；
  - `path` + 各 split 指向新结构；保留原 `ignore_classes`。
- `curate_manifest.json`：删除类列表、阈值、各保留类 train 图片数 before/after、残余超额告警、丢弃图片数、新数据集路径、源 EDA 引用。
- 控制台：打印整理前后对照表（数据取自 inventory，无需重扫）。

## 6. 接线

- `dispatch.py`：
  - `seg` + `eda` → `seg_tools.run_semantic_eda`（或新模块直连）；在调用前确保 `rt.import_runtime_dependencies()`（PIL/numpy）。
  - `seg` + `curate` → `seg_semantic_curate.run_semantic_curate`。
- `launcher.py`：在 argparse 里为 seg 增加 `eda` / `curate` 子动作及参数（`--data --output-dir --overwrite --min-class-images --threshold-percentile`，curate 加 `--eda-dir --drop-classes --image-threshold --export-suffix`，支持 CLI 覆盖交互默认值）。
- `interactive.py`：seg 菜单新增「语义 EDA」与「语义整理」两个交互入口。

## 7. 测试（pytest，WSL `lightlytrain` conda env）

构造微型语义数据集（数张已知像素值的 PNG mask + 最小 `data.yaml`）：

1. **EDA**：每类图片数 / 像素数计数正确；推荐删除/阈值符合预期；inventory CSV 字段正确（含 `class_ids` / `per_class_pixels`）。
2. **删除**：导出 yaml 的 `classes` 不含被删类；被删类在加载器里被映射为 -100（ignore）。
3. **压缩**：超阈值类的 train 图片数 ≤ 阈值；阈值=0 时不压缩。
4. **共现保护**：含稀有类的 train 图被保留；可丢池耗尽时告警、不过度压。
5. **范围**：val/test 图片/掩码数量不变。
6. **导出**：硬链接生成（同设备）；产出 yaml 能被 `load_semantic_segmentation_data_config` 加载且 `_load_semantic_mask` 正确映射。
7. 边界：无删除无阈值 → 警告/跳过导出。

## 8. 边界与处理

- 单通道索引 mask 与 RGB 元组 mask 都支持（统一走 `_class_labels` 映射口径）。
- 已有 `ignore_classes`：分析时按加载器口径排除。
- 仅存在于 val/test、train 无图的类：无法压缩，标记为可能删除候选。
- 阈值 = 0：只删不压。
- 无删除且无阈值：产出与源相同 → 告警并跳过导出。
- 删除后只剩背景/全 ignore 的图：默认保留（背景类仍有效，作负样本）。
- mask 缺失 / 图掩码数量不匹配：配对时跳过（沿用 `_semantic_image_mask_samples`）。
- 跨设备硬链接失败：回退为拷贝。
