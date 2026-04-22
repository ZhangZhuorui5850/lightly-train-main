# 项目结构梳理

## 1. 项目概览

当前项目位于 `lightly-train-main/`，整体由两部分组成：

- 业务使用层：顶层的训练、测试、推理脚本，以及本地数据集、权重和实验输出目录。
- 框架源码层：`src/lightly_train/` 下的 LightlyTrain 源码，实现分类、检测、分割、导出、评估等核心能力。

从当前工作区看，这个仓库已经不仅是官方框架源码，还叠加了本地实验数据、训练权重和自定义工具脚本，属于“框架源码 + 本地项目实践”的混合结构。

## 2. 顶层目录结构

下面是适合周总结展示的精简目录树，省略了大量图片、标注文件、`__pycache__` 和测试细节：

```text
lightly-train-main/
├── README.md
├── pyproject.toml
├── det_tool.py
├── train_det.py
├── train_cls.py
├── train_seg.py
├── test_det.py
├── test_det_v3.py
├── test_cls.py
├── load_datasets.py
├── datasets/
│   ├── convert_datasets/
│   │   ├── LabelMeToYOLO_v4.py
│   │   └── TLPD/
│   ├── dataset_det/
│   │   ├── data.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── labels/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   └── dataset_seg/
│       ├── data.yaml
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
├── out/
│   ├── my_experiment_cls/
│   │   ├── checkpoints/
│   │   ├── exported_models/
│   │   ├── test_results.csv
│   │   └── train.log
│   ├── my_experiment_det/
│   │   ├── checkpoints/
│   │   ├── exported_models/
│   │   ├── test_report.json
│   │   └── train.log
│   └── pic/
├── weights/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── src/
│   └── lightly_train/
│       ├── _commands/
│       ├── _configs/
│       ├── _data/
│       ├── _embedding/
│       ├── _export/
│       ├── _methods/
│       ├── _metrics/
│       ├── _models/
│       ├── _optim/
│       ├── _task_models/
│       └── _transforms/
├── tests/
├── docs/
├── examples/
├── docker/
├── dev_tools/
└── lightly-train-main/
    └── （内嵌的一份近似完整源码副本）
```

## 3. 关键目录说明

### 3.1 业务脚本层

- `det_tool.py`
  - 当前项目里最完整的检测后处理工具。
  - 统一提供 3 个子命令：
    - `infer`：单图或目录推理，可保存可视化图、JSON、TXT。
    - `eval`：对 `val/test` 数据集进行评估，可生成 `test_report.json`，也可选计算 torchmetrics 风格指标。
    - `export-good-dataset`：根据评估报告中各类 AP 表现筛选“高质量类别”，导出新的数据集。
  - 特点是把推理、评估、报告生成和数据集筛选串成了一条检测实验闭环。

- `train_det.py`
  - 检测训练入口脚本。
  - 调用 `lightly_train.train_object_detection(...)`。
  - 当前配置使用 `dinov3/vitl16-ltdetr`，并显式加载本地预训练权重 `weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。
  - 输出目录是 `out/my_experiment_det/`。

- `test_det.py`
  - 检测模型测试脚本。
  - 主要作用是加载导出的 `exported_best.pt`，遍历测试集图片，与 YOLO 标签计算 IoU、统计预测结果，并输出 `test_report.json`。
  - 实现里包含图片尺寸读取、GT 解析、预测结果兼容解析等逻辑，适合做离线测试报告。

- `test_det_v3.py`
  - 与 `test_det.py` 当前内容一致。
  - 可以视为检测测试脚本的一个平行版本/备份版本，但当前工作区中两者没有差异。

- `train_cls.py`
  - 图像分类训练脚本。
  - 当前示例是猫狗二分类，数据路径指向 `datasets/pet_split_250/images/...`，输出在 `out/my_experiment_cls/`。

- `test_cls.py`
  - 分类测试脚本。
  - 读取 `out/my_experiment_cls/exported_models/exported_best.pt`，对测试集逐张预测，并将结果保存为 `test_results.csv`。

- `train_seg.py`
  - 实例分割训练脚本。
  - 调用 `lightly_train.train_instance_segmentation(...)`。
  - 目前脚本中 `data="datasets/dateset_seg/data.yaml"` 存在路径拼写问题，和实际目录 `datasets/dataset_seg/` 不一致，后续使用前需要校对。

- `load_datasets.py`
  - 用 Hugging Face `datasets` 库拉取 `evan6007/TLPD` 数据集的测试脚本。
  - 更偏向数据准备/数据源验证，不参与主训练流程。

### 3.2 数据目录层

- `datasets/dataset_det/`
  - 检测任务数据集，采用 YOLO 检测格式。
  - `data.yaml` 已配置为单类别检测任务：
    - `task: detect`
    - `nc: 1`
    - `names[0]: carplate`
  - 当前图片数量：
    - train: 2408
    - val: 303
    - test: 303
  - 图片和标签数量一致，都是 3014。

- `datasets/dataset_seg/`
  - 分割任务数据集，采用 YOLO 分割格式。
  - `data.yaml` 配置为：
    - `task: segment`
    - `nc: 1`
    - `names[0]: carplate`
  - 当前图片数量：
    - train: 2408
    - val: 303
    - test: 303
  - 图片和标签数量一致，都是 3014。

- `datasets/convert_datasets/LabelMeToYOLO_v4.py`
  - 数据转换脚本。
  - 作用是把 LabelMe 标注一次性转换成两套输出：
    - `dataset_det/`：检测标签
    - `dataset_seg/`：分割标签
  - 这个脚本是数据生产链路中的关键入口，说明项目的数据格式已经从原始标注层和训练输入层做了清晰拆分。

- `datasets/convert_datasets/TLPD/`
  - 项目内保留的一份 TLPD 数据资源目录，更像原始或中间数据源。

### 3.3 实验产物层

- `out/my_experiment_det/`
  - 检测实验输出目录。
  - 已包含：
    - `checkpoints/best.ckpt`
    - `checkpoints/last.ckpt`
    - `exported_models/exported_best.pt`
    - `exported_models/exported_last.pt`
    - `test_report.json`
    - `train.log`
    - 多个 TensorBoard 事件文件

- `out/my_experiment_cls/`
  - 分类实验输出目录。
  - 已包含：
    - `checkpoints/best.ckpt`
    - `checkpoints/last.ckpt`
    - `exported_models/exported_best.pt`
    - `exported_models/exported_last.pt`
    - `test_results.csv`
    - `train.log`

- `weights/`
  - 本地预训练权重目录。
  - 当前可见核心权重为 `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。

### 3.4 框架源码层

- `src/lightly_train/`
  - 项目真正的 Python 包源码目录。
  - `__init__.py` 对外暴露的核心能力包括：
    - `train_object_detection`
    - `train_image_classification`
    - `train_instance_segmentation`
    - `train_panoptic_segmentation`
    - `train_semantic_segmentation`
    - `load_model`
    - `export`
    - `embed`
  - 当前源码版本号为 `0.14.2`。

- `src/lightly_train/_commands/`
  - 命令层封装，负责训练、导出、预测、嵌入等统一入口。

- `src/lightly_train/_task_models/`
  - 任务模型实现目录。
  - 包含分类、目标检测、实例分割、全景分割、语义分割等任务实现。
  - 与当前项目最直接相关的是：
    - `dinov3_ltdetr_object_detection/`
    - `picodet_object_detection/`
    - `image_classification/`
    - `object_detection_components/`

- `src/lightly_train/_models/`
  - 预训练模型和骨干网络封装目录。
  - 包含 DINOv2、DINOv3、RT-DETR、RF-DETR、torchvision、timm、ultralytics 等适配。

- `src/lightly_train/_metrics/`
  - 各任务评估指标实现目录。

- `src/lightly_train/_data/`
  - 数据加载、文件处理、YOLO 数据辅助函数目录。
  - `det_tool.py` 对其中的 `file_helpers` 和 `yolo_helpers` 有直接依赖。

- `tests/`
  - 测试目录。
  - 当前外层 `tests/` 下约有 72 个测试文件，用于覆盖命令、模型、指标、数据与任务模块。

## 4. 当前项目工作流梳理

按目前目录来看，项目大致形成了下面这条工作链路：

1. 原始标注数据准备
   - 通过 `datasets/convert_datasets/LabelMeToYOLO_v4.py` 将 LabelMe 数据转换成检测/分割两套训练格式。

2. 模型训练
   - 检测：`train_det.py`
   - 分类：`train_cls.py`
   - 分割：`train_seg.py`

3. 模型导出与测试
   - 训练结果统一输出到 `out/.../`
   - 检测测试使用 `test_det.py`
   - 分类测试使用 `test_cls.py`

4. 检测后处理与数据闭环
   - `det_tool.py infer` 做推理和可视化
   - `det_tool.py eval` 做评估与报告输出
   - `det_tool.py export-good-dataset` 根据评估结果筛选并导出新数据集

这说明你的项目已经具备了“数据准备 -> 训练 -> 测试 -> 评估 -> 数据再筛选”的基本闭环。

## 5. 当前结构中的注意点

- 当前仓库存在一个内嵌目录 `lightly-train-main/lightly-train-main/`，里面又是一份近似完整的源码副本。周总结里可以标注为“源码镜像/嵌套副本”，后续需要确认它是否仍然参与实际运行。
- `det_tool.py` 默认配置中引用的 `datasets/wuwanPic_dataset/dataset_det` 和 `out/my_experiment_det_0402` 在当前工作区中不存在，说明这个脚本的默认参数和当前实际目录有偏差，使用前需要调整。
- `train_seg.py` 中的 `datasets/dateset_seg/data.yaml` 与当前真实目录 `datasets/dataset_seg/data.yaml` 不一致，属于明显的配置拼写问题。
- `test_det.py` 和 `test_det_v3.py` 当前无差异，如果后续长期维护，建议只保留一个主版本，避免脚本分叉。
- 你 IDE 中打开的 `eval_infer_det.py`、`split_dataset.py` 在当前工作区里没有找到，因此这份梳理没有把它们纳入正式目录说明。

## 6. 可直接用于周总结的简版表述

本周我完成了对 `lightly-train-main` 项目的整体结构梳理。当前项目由顶层业务脚本、自建数据集目录、训练输出目录以及 `src/lightly_train` 框架源码四部分组成。业务侧已经覆盖目标检测、图像分类、实例分割的训练与测试，其中检测部分额外实现了统一后处理工具 `det_tool.py`，支持推理、评估和基于评估结果的数据再筛选。数据侧已经形成从 LabelMe 标注到 YOLO 检测/分割格式的转换链路，检测集和分割集均按 `train/val/test` 划分，当前单类车牌数据总量为 3014 张。整体上，项目已经具备较完整的“数据转换 -> 模型训练 -> 模型测试 -> 评估分析 -> 数据筛选”的实验闭环。
