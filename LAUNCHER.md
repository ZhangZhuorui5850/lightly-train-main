# launcher.py 使用说明

`launcher.py` 是训练与实验工具入口，负责 cls、det、seg 的训练、推理、评估、EDA、筛选和报告。
数据集格式转换使用 `python datasets/convert_datasets/convert.py`。两个入口共用同一套数据集检测与路径解析组件。

---

## 快速开始

```bash
# 交互模式（直接运行，进入菜单选择）
python launcher.py

# CLI 模式（直接指定命令和参数）
python launcher.py <command> [options]
```

---

## 两种运行模式

### 交互模式

直接运行 `python launcher.py`，进入菜单。程序会逐步询问任务类型、功能、路径等，并在执行前打印等价的 CLI 命令供参考。

菜单结构：

```
请选择任务类型
  1. cls  分类
  2. det  检测
  3. seg  分割
  4. clean 实验清理

↓ 根据任务，继续选择功能
  train / infer / eval / eda / curate / export / optimize / review-sample / report
```

交互选择数据集时，launcher 会按标注内容识别 Cls、Det、YOLO Seg 和 PNG Semantic，
兼容 `images/train` 与 `train/images` 两类布局、自定义 YAML、迁移后的 YAML path，并按
相关实验和最后修改时间排序。TXT 图片清单、多目录 split、原始分类 ImageFolder 都会保留
精确成员关系。列表支持中文或英文关键词筛选。数据集目录可通过软链接接入，嵌套项目中的配置会独立显示。

交互选择实验时，launcher 递归检查 `out/` 下的训练日志、TensorBoard 记录、
`exported_models/` 和 `checkpoints/`。任务类型优先读取 `train.log`，权重支持标准名称和
自定义 `.pt/.pth/.ckpt` 名称；列表显示最新产物时间并按该时间降序排列。软链接实验目录
使用 inode 去重，EDA、汇总报告和推理内部目录会从实验候选中排除。

实验目录、checkpoint、`run_meta.json`、infer/eval 报告统一通过 `tool_lib/file_index.py`
遍历；多个报告文件名模式共享一次目录扫描。交互终端显示已访问目录进度，批处理日志按
快照间隔持续输出。数据集格式识别继续由共享的 `convert_tools/dataset_detector.py` 负责。

多卡推理在交互终端中固定显示一条合计进度和每张 GPU 一条进度，默认最多展示 8 张卡。
每卡行包含物理 GPU 标识、shard、图片数量、速度、ETA 和当前阶段。批处理日志环境每 5 秒
输出一次逐卡快照。设置 `LIGHTLY_PROGRESS_LAYOUT=compact` 可切换为单行紧凑布局，设置
`LIGHTLY_PROGRESS_MAX_CARD_BARS=<数量>` 可调整多行布局的卡数上限。

数据集索引会显示当前扫描根、已访问目录数和已发现数据集数。临时扫描仓库外目录可设置
`LIGHTLY_DATASET_SEARCH_ROOTS=/mnt/data1/datasets:/mnt/data2/datasets`，多个路径使用系统路径分隔符。
`LIGHTLY_PROGRESS_SNAPSHOT_INTERVAL=<秒>` 可调整批处理日志的进度快照间隔。

### CLI 模式

向 `launcher.py` 传入子命令即进入 CLI 模式，跳过菜单。每条命令的参数见下方各节。

---

## 统一配置区

`launcher.py` 顶部定义了五组配置：COMMON、CLS、DET、SAHI、SEG。修改这里即可改变全局默认值。

### COMMON_SETTINGS — 公共路径

| 键 | 默认值 | 说明 |
|---|---|---|
| `out_dir` | `out` | 统一输出根目录 |
| `experiment_root_dir` | `out` | 实验目录扫描根 |
| `dataset_search_roots` | `["datasets"]` | 数据集递归扫描根列表；可加入挂载盘和仓库外目录 |
| `test_output_root_dir` | `out` | 历史 infer/eval 结果扫描根；新结果归档到对应实验目录 |
| `eda_output_root_dir` | `out/EDA` | EDA 报告输出根 |
| `all_report_root_dir` | `out/all_report` | 全量实验报告目录 |

### DET_SETTINGS — 检测配置

**最常修改的两项**（改完后，绝大多数检测默认路径自动跟随变化）：

| 键 | 默认值 | 说明 |
|---|---|---|
| `det_dataset_dir` | `datasets/convert_datasets/NEU-DET811kuozeng1bei` | 检测数据集根目录 |
| `det_experiment_dir` | `None`（自动检索最近实验） | 实验目录 |

**阈值**：

| 键 | 默认值 | 说明 |
|---|---|---|
| `det_score_threshold` | `0.3` | 推理置信度阈值，偏少漏检；误检多时调高 |
| `det_report_iou_threshold` | `0.5` | 评估时预测框与标注框的 IoU 命中阈值 |
| `det_eval_vis_max_images` | `50` | eval 每个 split 抽样保存的对比图上限；0 = 全量 |

**Export 参数**（填 `0` 或留空时自动推导）：

| 键 | 默认值 | 说明 |
|---|---|---|
| `det_export_target_total_images` | `0` | 导出总图数目标；0 = 自动 |
| `det_export_good_class_threshold` | `0.0` | 类别 AP 参考阈值 |
| `det_export_auto_balance` | `True` | 开启自动平衡分析 |
| `det_export_auto_relax_class_threshold` | `True` | 开启后 AP 仅供参考，不强制删类 |
| `det_export_split_ratio` | `"8:1:1"` | 重划分比例 train:val:test |
| `det_export_suffix` | `"_A"` | 导出数据集目录名后缀 |
| `det_export_balance_ratio` | `0.0` | 最大类/最小类框数比例上限；0 = 自动，人工值 ≥ 1 |
| `det_export_min_class_images` | `0` | 每类最少图数；0 = 自动 |
| `det_export_min_class_boxes` | `0` | 每类最少框数；0 = 自动 |
| `det_export_target_images_per_class` | `0` | 每类目标图数；0 = 取中位数 |
| `det_export_target_boxes_per_class` | `0` | 每类目标框数；0 = 自动 |
| `det_export_max_boxes_per_image` | `0` | 每张图最大总框数；0 = 自动 |
| `det_export_max_boxes_per_class_per_image` | `0` | 每张图同类最大框数；0 = 自动 |
| `det_export_box_density_penalty` | `0.0` | 高框密度图片的惩罚强度；0 = 自动 |

### CLS_SETTINGS — 分类配置

| 键 | 默认值 | 说明 |
|---|---|---|
| `cls_threshold` | `0.5` | 分类推理/评估置信度阈值 |

### SEG_SETTINGS — 分割配置

| 键 | 默认值 | 说明 |
|---|---|---|
| `seg_threshold` | `0.8` | 分割掩码阈值；掩码太少调低，噪声多调高 |

## CLI 命令参考

运行 `python launcher.py --help` 查看全部命令，运行 `python launcher.py <command> -h`
查看参数、默认值与范围。`--debug` 可放在命令前或末尾。CLI 的百分比、数量、比例、
分片编号和平均密度范围会在执行前统一校验。

### `infer` — 目标检测推理

```bash
python launcher.py infer [options]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--experiment-dir` | 自动检索最近 det 实验 | 实验目录 |
| `--checkpoint` | `None`（取实验 best） | 指定权重文件 |
| `--image` | — | 单张图片路径（与 `--image-dir`/`--data` 三选一） |
| `--image-dir` | — | 文件夹批量推理 |
| `--data` | `det_dataset_dir/data.yaml` | 数据集 yaml（数据集评测模式） |
| `--split` | `test` | train / val / test / test+val / all（train+test+val） |
| `--output-dir` | 自动生成 | 推理结果输出目录 |
| `--score-threshold` | `0.3` | 置信度阈值 |
| `--device` | `auto` | 推理设备 |
| `--save-visualization` / `--skip-visualization` | 保存 | 是否保存可视化结果 |
| `--save-json` / `--skip-json` | 保存 | 是否保存 JSON 预测结果 |
| `--save-txt` | `False` | 是否保存 TXT 预测结果 |
| `--overwrite` | `False` | 输出目录非空时是否覆盖 |

**数据集批量推理**示例：

```bash
python launcher.py infer --data datasets/mydata/data.yaml --split test
```

**文件夹批量推理**示例：

```bash
python launcher.py infer --image-dir path/to/images --score-threshold 0.4
```

---

infer/eval 复用指纹覆盖所选 split 的图片、标注、清单和 YAML 配置，包含外部路径与软链接目标；
SAHI 的 NMS、全局局部合并阈值和小图策略参与指纹比较。输出目录采用软链接检查，
发布时通过同目录文件锁串行执行，并重新校验覆盖权限。

### `eval` — 目标检测评估

eval 对全部样本计算指标，每个 split 默认抽样保存 50 张 `[原图|GT|预测]` 对比图。

```bash
python launcher.py eval --task det --data datasets/mydata/data.yaml --split test
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--experiment-dir` | 自动检索最近 det 实验 | 实验目录 |
| `--checkpoint` | `None`（取实验 best） | 指定权重文件 |
| `--data` | `det_dataset_dir/data.yaml` | 数据集 yaml |
| `--split` | `test` | 单个 split，或 `--split val test` / `--split train val test` |
| `--score-threshold` | `0.3` | test_report 的预测置信度阈值 |
| `--report-iou-threshold` | `0.5` | TP/FP/FN 的 IoU 命中阈值 |
| `--classwise` | `False` | 输出按类 mAP 指标 |
| `--vis-max-images` | `50` | 每个 split 的对比图上限；0 = 全量 |
| `--save-visualization` / `--skip-visualization` | 保存 | 控制抽样对比图 |
| `--save-json` / `--skip-json` | `False` | 是否保存每图预测 JSON（`<输出目录>/_tempfile/json`），供分档指标等后处理 |
| `--device` | `auto` | 评估设备 |
| `--overwrite` | `False` | 输出目录非空时是否覆盖 |

评估目录包含 `metrics_summary.json`、`*_report.json`、`run_meta.json`、报告 Markdown 和限额后的 `compare/`。
默认不保存预测明细；需要时加 `--save-json`（或把 `DET_SETTINGS` 的 `det_eval_save_json` 置 `True`）。
注意 JSON 里的预测会被 `--score-threshold` 过滤，做指标后处理时应同时传 `--score-threshold 0`，
否则低置信度的（尤其是小脸）预测会在写入前被砍掉。

---

### `eda` — 数据集探索分析

对检测数据集做全面的分布分析，输出报告到 `out/EDA/` 或指定目录。

分析内容：split 对照、类别分布、不平衡分析、目标尺寸、框密度、分辨率、逐图清单。

```bash
python launcher.py eda --data datasets/mydata/data.yaml [--output-dir out/EDA/mydata] [--overwrite]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data` | `det_dataset_dir/data.yaml` | 数据集配置文件 |
| `--output-dir` | 自动生成 | EDA 报告输出目录 |
| `--overwrite` | `False` | 目录非空时是否覆盖 |

---

### `export` — 数据集筛选导出

根据类别质量、标签数量、框密度等策略，从原始检测数据集中筛选并重新划分子集。

交互模式下只需输入总图数，其余阈值由系统自动联合推导；CLI 模式支持精细控制所有参数。

```bash
python launcher.py export --export-source-data datasets/mydata/data.yaml \
    --target-total-images 500
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--export-source-data` | `det_dataset_dir/data.yaml` | 源数据集 yaml |
| `--report-json` | 自动匹配最近 test_report.json | 推理评估报告（可选） |
| `--target-total-images` | `0`（自动） | 导出总图数目标 |
| `--split-ratio` | `8:1:1` | 重划分比例 |
| `--good-class-threshold` | `0.0` | 类别 AP 参考阈值 |
| `--auto-balance` / `--no-auto-balance` | 开启 | 自动平衡分析 |
| `--auto-relax-class-threshold` / `--strict-class-threshold` | 宽松 | AP 阈值是否参与删类 |
| `--balance-ratio` | `0.0`（自动） | 类间框数最大比例 |
| `--min-class-images` | `0`（自动） | 每类最少图数 |
| `--min-class-boxes` | `0`（自动） | 每类最少框数 |
| `--target-images-per-class` | `0`（自动） | 每类目标图数 |
| `--target-boxes-per-class` | `0`（自动） | 每类目标框数 |
| `--max-boxes-per-image` | `0`（自动） | 每张图最大总框数 |
| `--max-boxes-per-class-per-image` | `0`（自动） | 每张图同类最大框数 |
| `--box-density-penalty` | `0.0`（自动） | 高框密度图片惩罚强度 |
| `--export-suffix` | `_A` | 导出目录名后缀 |

---

### `report` — 生成实验报告

扫描实验目录，汇总训练曲线、推理结果等，生成单实验报告或全量对比报告。

```bash
python launcher.py report --experiment-dir out/2026-04-14/my_exp
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--experiment-dir` | 自动选择最近 det 实验 | 指定实验目录 |
| `--search` | `None` | 关键字筛选实验目录 |
| `--output-dir` | 自动生成 | 报告输出目录 |
| `--dry-run` | `False` | 只打印，不写文件 |

---

### 数据集检测与转换

```bash
python datasets/convert_datasets/convert.py datasets
python datasets/convert_datasets/convert.py auto
python datasets/convert_datasets/convert.py oneclick \
  my_raw_dataset --output-root datasets/my_dataset
```

`datasets` 用于快速识别和诊断数据集，`auto` 根据数据格式列出可执行操作，其他命令通过
`python datasets/convert_datasets/convert.py list` 查看。

常用写入命令统一支持 `--dry-run`。`python datasets/convert_datasets/convert.py doctor`
会检查命令注册、脚本存在性，并运行常用写入工具的 `--help` 检查 dry-run 参数。

---

### `clean` — 实验产物清理

```bash
python launcher.py clean                 # 交互选择并确认
python launcher.py clean --dry-run       # CLI 扫描和预览全部计划项
python launcher.py clean --execute --yes # CLI 执行全部计划项
```

交互层只生成清理计划，删除和日志写入由清理执行器统一完成。

### `optimize` — 训练后数据集优化

```bash
python launcher.py optimize --source-data datasets/mydata/data.yaml \
  --report-json out/exp/eval/test_report.json \
  --infer-output-dir out/exp/infer/mydata/test
```

`--report-json` 可重复提供。命令分析类别质量和混淆候选，随后进入人工决策流程。

### `review-sample` — 数据集质检抽样

该功能同时出现在交互菜单和 CLI。运行 `python launcher.py review-sample -h` 查看几何检查、
模型分析、类别异常和抽样数量参数。

---

## 模块架构

```
launcher.py              ← 用户入口：配置区 + main()
  ↓
tool_lib/interactive.py  ← 菜单交互 / CLI 参数解析，返回 Namespace
tool_lib/dataset_adapter.py ← 复用 convert_tools 的统一数据集检测和路径解析
  ↓
tool_lib/dispatch.py     ← 路由层，将请求分发到对应工具模块
  ↓
tool_lib/
  cls_tools.py           ← 分类推理、评估
  det_tools.py           ← 检测推理、EDA、Export、Report
  seg_tools.py           ← 分割推理、评估
  dataset_adapter.py     ← 复用 datasets/convert_datasets/convert_tools 的数据集检测器
  common.py              ← 全局常量、运行时依赖、通用函数
  det_infer.py           ← 检测推理/评估共享核心
  det_export.py          ← 检测导出核心
  det_eda.py             ← EDA 分析核心
  det_report.py          ← 报告生成核心
  det_analysis.py        ← 分布分析工具
  script_runner.py       ← 调用外部训练/测试脚本
```

EDA、报告和数据整理加载 NumPy、Pillow、PyYAML 等轻量依赖；训练、推理、评估和模型质检
加载 LightlyTrain 与 PyTorch。launcher 会校验 `lightly_train` 的导入来源，确保模型功能使用
仓库 `src/` 中的 fork。
