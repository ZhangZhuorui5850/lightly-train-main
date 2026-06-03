# 目标尺寸均衡补充工具（size-supplement）设计

- 日期：2026-05-29
- 作者：zhangzhuorui5850 + Claude
- 状态：待评审

## 1. 背景与问题

检测数据集导出（`tool_lib/det_export.py`）按**类别**做均衡选图，并对单图框密度做上限控制。目标尺寸（小 `area < 32²px`、中 `32²px ≤ area < 96²px`、大 `area ≥ 96²px`）在导出流程里**只被统计和写进 EDA 报告，从不作为选图目标**。

后果：之前导出的 1w 张均衡集（如 `datasets/coco2017_dataset/dataset_det_A_10000`，用户称为 `jsmb_dataset/dataset_det_A`）小目标占比只有约 10%。

需要一个**可交互、可复用**的工具：以已有均衡集为基底，从源全量库增量补充“小/中目标丰富”的图，把按框数量计的尺寸占比提升到用户指定的比例（例如各三分之一），产出一套**全新**数据集（不改原数据），并写好报告。

> 现状提示：当前 `datasets/` 下没有 `jsmb_dataset/`，可发现的数据集为 `military_dataset`、`wuwanPic_dataset`、`neu_dataset`、`coco2017_dataset`、`aeroscapes`。与“1w 均衡集”最匹配的是 `coco2017_dataset/dataset_det_A_10000`。工具自动检索现有数据集，不硬编码该路径。

## 2. 已确认的需求决策

1. **采样模式：基底 + 增量补充**。保留基底全部图，从源全量库增量挑图加入，总量变大。
2. **占比口径：按目标框数量**（`small_boxes / 总框数`），与现有 `size_bucket_ratio` 一致。
3. **总量控制：用户给“目标比例 + 目标总图数”**，算法贪心逼近比例。
4. **补充去向：按 8:1:1 分到 train/val/test**，整体数据集口径达标，val/test 也随之变化。
5. **类别均衡：尺寸优先、类别软兼顾**。同等尺寸贡献下优先选能补弱势类别的图。
6. **选图算法：贪心赤字驱动（方案 A）**。
7. **代码组织：根目录交互入口 + tool_lib 逻辑**。

## 3. 总体架构

```
size_supplement.py                  # 根目录交互入口（薄壳：收集用户意图 + 调用 run）
tool_lib/det_size_supplement.py     # 全部核心逻辑（可 import / 可单测）
```

`det_size_supplement.py` 复用的现有构件：

| 复用对象 | 来源 | 用途 |
|---|---|---|
| `collect_source_image_infos` | `det_shared` | 扫描数据集各 split 为逐图标签信息（不拷贝） |
| `read_yolo_label_lines` / `remap_yolo_label_lines` / `safe_class_name` | `det_shared` | YOLO 标签读取与重映射 |
| `_bucket_box_area` / `_open_image_size` / `_analyze_candidate_boxes` | `det_export` | 框尺寸分桶（小/中/大）逻辑 |
| `build_export_dataset_eda` / `render_export_dataset_eda_markdown` | `det_export` | 标准 EDA JSON + Markdown |
| `list_dataset_yaml_candidates` / `prompt_choice` / `prompt_int` / `prompt_float` / `prompt_yes_no` / `compact_display_path` | `interactive` | 数据集自动发现 + 交互提示 |
| `load_data_config` / `dump_yaml` / `deduplicate_path` / `normalize_names` | `common` (rt) | 配置读写、目录去重命名 |

> 私有函数（`_bucket_box_area` 等）将在 `det_export.py` 中提升为可导入的公开函数（去掉下划线或在模块内 re-export），保持其行为不变，避免逻辑复制。

## 4. 交互流程（`size_supplement.py`）

1. **选基底**：调用 `list_dataset_yaml_candidates(task="det")` 列出候选，用户选已有均衡集（如 `dataset_det_A_10000`）。
2. **选源池**：智能默认 = 同族去掉 `_A...` 后缀的目录（如 `dataset_det`）；列出候选供改选，支持 custom。
3. **展示基底现状**：打印基底当前小/中/大框数与占比。
4. **输入目标比例**：形如 `33/33/33` 或 `30 30 40`，自动归一化为和为 1。
5. **输入目标总图数**：例如 `15000`（必须 > 基底图数，否则提示并重输或转“固定总量”不在本期范围）。
6. **确认预览**：基底图数、源池可用图数、目标比例、目标总图数、预计补充图数、输出目录名 → `prompt_yes_no` 确认。
7. 执行管线并打印分阶段进度。

## 5. 核心管线（`run_size_supplement`）

1. **加载配置**：`load_data_config` 读基底与源池的 `data.yaml`；统一类别名（`normalize_names`），校验两者类别一致（不一致则报错并列出差异）。
2. **扫描**：`collect_source_image_infos` 分别扫描基底与源池各 split。
3. **去重**：源池中**文件名 stem 已存在于基底**的图剔除（基底图本就来自源池）；记录去重数量。
4. **尺寸分桶**：对基底所有图与源池候选图，逐框用 `_open_image_size` + `_bucket_box_area` 统计 `(small, medium, large)` 框数。
   - **图尺寸缓存**：在源池根目录写 `.imgsize_cache.json`，键为 `相对路径+mtime`，避免重复 PIL 打开（72k 图一次性扫描，重跑秒级）。
5. **目标与赤字**：
   - 基底当前桶计数 `c = (c_s, c_m, c_l)`。
   - 需补充图数 `K = N_target − |base|`（K ≤ 0 直接报错退出）。
   - 目标比例 `r = (r_s, r_m, r_l)`，和为 1。
6. **贪心赤字驱动选图（方案 A）**：
   - 每批开始重算赤字权重：`w_b = max(0, r_b − cur_prop_b)`，其中 `cur_prop_b = c_b / sum(c)`。
   - 候选图打分：`score = Σ_b w_b · boxes_b − over_penalty · Σ_b over_b · boxes_b + λ_cls · class_bonus`
     - `over_b = max(0, cur_prop_b − r_b)`：对已超标桶的惩罚。
     - `class_bonus`：图中覆盖“当前弱势类别”（低于类别中位框数）的程度，软兼顾类别。
   - 按分数取该批 top 图（批大小约 200），更新 `c` 与已选集合；每批后重算权重，重复直到选满 `K` 张或源池耗尽。
   - 若源池耗尽仍未达比例 → 记录“shortfall”（缺口），不报错，按已选结果产出。
7. **划分补充图**：把选中的 `K` 张按 8:1:1 分配到 train/val/test（沿用 `det_export` 的 `allocate_split_targets_from_source` 思路或等价实现）。基底原划分保持不动，补充图叠加到对应 split。
8. **写新数据集**（异常安全：先写临时目录 `__supplement_tmp`，全部成功后 rename；失败清理）：
   - 拷贝基底全部图与标签（保持其 split 与类别 id）。
   - 拷贝补充图与标签到各 split；文件名冲突用 `resolve_destination_rel_path` 思路去重。
   - 写 `data.yaml`、`classes.txt`（类别集合与基底一致，nc/names 不变）。
9. **生成报告**（见 §7）。

## 6. 输出数据集命名

在基底目录名后追加可读后缀，体现“做了什么”：

```
<base_name>__szsup_s<小>m<中>l<大>_n<总图数>
例：dataset_det_A_10000__szsup_s33m30l37_n15000
```

- `szsup` = size supplement；`s/m/l` = 实际达到的小/中/大占比（四舍五入整数百分比）；`n` = 实际总图数。
- 用 `deduplicate_path` 防止重名覆盖。原基底与源池目录**不被修改**。

## 7. 报告（写入新数据集根目录）

1. **标准 EDA**：`build_export_dataset_eda` + `render_export_dataset_eda_markdown` 生成
   `export_dataset_eda_<tag>.json` / `.md`（split 总览、类别分布、尺寸桶、高密度样本）。
2. **补充专项报告** `supplement_report.md` + `supplement_report.json`，含：
   - 基底 vs 最终的尺寸占比 **before/after 对照表**（小/中/大框数与百分比）。
   - 补充图数、补充框数、按 split 与按类别的增量 delta。
   - 源池路径、去重数量、目标比例、目标总图数、批大小等全部参数。
   - 与目标的接近度；若源池供给不足导致 shortfall，明确写出缺口（哪个桶差多少）。
   - 输出目录名解释。

## 8. 错误处理

- 基底/源池类别不一致 → 报错并列出差异，终止。
- `N_target ≤ |base|` → 报错，提示本工具只增量补充。
- 源池去重后为空 / 无可补图 → 报错。
- 源池供给不足以达标 → **不报错**，按已选产出并在报告写明 shortfall。
- 写盘失败 → 清理临时目录，原数据集不受影响。
- 缺 Pillow → 复用 `rt.ensure_plot_dependencies()` 的报错路径。

## 9. 测试策略

- 用一个**小型合成数据集**（十几张图、人工构造已知大小的框）做单元/集成测试：
  - 分桶计数正确（边界 32²、96²）。
  - 去重逻辑正确（stem 命中剔除）。
  - 贪心选图确实把小目标占比朝目标推进，且在源池受限时正确报告 shortfall。
  - 8:1:1 划分数量正确。
  - 原数据集目录零改动（前后哈希/清单一致）。
  - 输出含 `data.yaml`、`classes.txt`、两份报告。
- 图尺寸缓存命中与失效（mtime 变化）正确。

## 10. 不做（YAGNI / 本期范围外）

- 固定总量“置换式”模式（决策选的是增量补充）。
- 按图片数口径的占比。
- 类别硬约束模式。
- 整数规划求解器。
- 分割（seg）任务的对应工具（结构可平移，但本期只做 det）。
