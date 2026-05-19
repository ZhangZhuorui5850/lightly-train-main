# launcher.py 使用说明

`launcher.py` 是整个工具系统的统一入口。它汇聚了所有用户可调的配置，并将训练、推理、评估、数据转换等功能串联起来。

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
  4. data 数据集转换

↓ 根据任务，继续选择功能
  train / infer / eval / eda / export / report / convert
```

### CLI 模式

向 `launcher.py` 传入子命令即进入 CLI 模式，跳过菜单。每条命令的参数见下方各节。

---

## 统一配置区

`launcher.py` 顶部定义了四组配置，修改这里即可改变全局默认值，无需修改底层模块。

### COMMON_SETTINGS — 公共路径

| 键 | 默认值 | 说明 |
|---|---|---|
| `out_dir` | `out` | 统一输出根目录 |
| `experiment_root_dir` | `out` | 实验目录扫描根 |
| `infer_output_root_dir` | `out` | 推理输出根 |
| `eda_output_root_dir` | `out/EDA` | EDA 报告输出根 |
| `all_report_root_dir` | `out/all_report` | 全量实验报告目录 |

### DET_SETTINGS — 检测配置

**最常修改的两项**（改完后，绝大多数检测默认路径自动跟随变化）：

| 键 | 默认值 | 说明 |
|---|---|---|
| `det_dataset_dir` | `datasets/wuwanPic_dataset/dataset_det` | 检测数据集根目录 |
| `det_experiment_dir` | `None`（自动检索最近实验） | 实验目录 |

**阈值**：

| 键 | 默认值 | 说明 |
|---|---|---|
| `det_score_threshold` | `0.3` | 推理置信度阈值，偏少漏检；误检多时调高 |
| `det_report_iou_threshold` | `0.5` | 评估时预测框与标注框的 IoU 命中阈值 |

**Export 参数**（填 `0` 或留空时自动推导）：

| 键 | 默认值 | 说明 |
|---|---|---|
| `det_export_target_total_images` | `0` | 导出总图数目标；0 = 自动 |
| `det_export_good_class_threshold` | `0.0` | 类别 AP 参考阈值 |
| `det_export_auto_balance` | `True` | 开启自动平衡分析 |
| `det_export_auto_relax_class_threshold` | `True` | 开启后 AP 仅供参考，不强制删类 |
| `det_export_split_ratio` | `"8:1:1"` | 重划分比例 train:val:test |
| `det_export_suffix` | `"_A"` | 导出数据集目录名后缀 |
| `det_export_balance_ratio` | `0.0` | 类间框数最大比例上限；0 = 自动 |
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

### SCRIPT_SETTINGS — 脚本路径

| 键 | 默认值 |
|---|---|
| `train_cls_script` | `train_cls.py` |
| `train_det_script` | `train_det.py` |
| `train_seg_script` | `train_seg.py` |
| `test_cls_script` | `test_cls.py` |
| `test_det_script` | `test_det_v3.py` |

---

## CLI 命令参考

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
| `--compute-metrics` | `False` | 是否计算评估指标（dataset 模式自动开启） |
| `--metric-classwise` | `False` | 是否输出按类指标 |
| `--report-iou-threshold` | `0.5` | 评估 IoU 阈值 |
| `--bad-class-map50-threshold` | `0.3` | 差类别 AP@0.5 阈值 |
| `--save-test-report` | `True`（dataset 模式） | 是否保存 test_report.json |
| `--overwrite` | `False` | 输出目录非空时是否覆盖 |

**数据集评测模式**示例：

```bash
python launcher.py infer --data datasets/mydata/data.yaml --split test
```

**文件夹批量推理**示例：

```bash
python launcher.py infer --image-dir path/to/images --score-threshold 0.4
```

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
| `--experiment-dir` | 自动检索 | 指定实验目录 |
| `--search` | `None` | 关键字筛选实验目录 |
| `--output-dir` | 自动生成 | 报告输出目录 |
| `--dry-run` | `False` | 只打印，不写文件 |

---

### `convert` — 数据集格式转换

将 labelme JSON 或 YOLO TXT 格式的标注转换为标准 YOLO 数据集结构（含 train/val/test 划分）。

```bash
python launcher.py convert my_raw_dataset [--task det] [--label-format labelme]
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `source_dir`（位置参数） | — | 源数据集目录名或路径 |
| `--output-name` | 同源目录名 | 输出数据集目录名 |
| `--output-root` | `datasets/<output-name>` | 输出根目录 |
| `--task` | `all` | det / cls / seg / all |
| `--label-format` | `auto` | auto / labelme / yolo |
| `--seed` | `None` | 随机种子（固定划分） |
| `--dry-run` | `False` | 只打印，不写文件 |

---

## 模块架构

```
launcher.py              ← 用户入口：配置区 + main()
  ↓
tool_lib/interactive.py  ← 菜单交互 / CLI 参数解析，返回 Namespace
  ↓
tool_lib/dispatch.py     ← 路由层，将请求分发到对应工具模块
  ↓
tool_lib/
  cls_tools.py           ← 分类推理、评估
  det_tools.py           ← 检测推理、EDA、Export、Report
  seg_tools.py           ← 分割推理、评估
  convert_tools.py       ← 数据集格式转换
  common.py              ← 全局常量、运行时依赖、通用函数
  det_infer.py           ← 检测推理核心
  det_export.py          ← 检测导出核心
  det_eda.py             ← EDA 分析核心
  det_report.py          ← 报告生成核心
  det_analysis.py        ← 分布分析工具
  script_runner.py       ← 调用外部训练/测试脚本
```

**运行时依赖**（`lightly_train`、`torch`、`PIL` 等）只在需要时才导入，`eda`、`report`、`convert` 等轻量功能无需安装完整深度学习环境即可运行。
