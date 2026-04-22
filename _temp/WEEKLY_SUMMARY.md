# 本周项目结构梳理

## 1. 当前项目结构

```text
lightly-train-main/
├── det_tool.py
├── train_det.py
├── train_cls.py
├── train_seg.py
├── test_det.py
├── test_det_v3.py
├── test_cls.py
├── load_datasets.py
├── README.md
├── pyproject.toml
├── datasets/
│   ├── convert_datasets/
│   │   ├── LabelMeToYOLO_v4.py
│   │   ├── sync_picture.py
│   │   └── TLPD/
│   ├── military_dataset/
│   │   ├── dataset_det/
│   │   │   ├── data.yaml
│   │   │   ├── classes.txt
│   │   │   ├── images/train|val|test
│   │   │   └── labels/train|val|test
│   │   ├── dataset_seg/
│   │   ├── dataset_cls/
│   │   └── dataset_det_A/
│   │       └── ...
│   └── wuwanPic_dataset/
│       └── （目录结构与 military_dataset 同理）
├── out/
│   ├── my_experiment_det_0402/
│   │   ├── checkpoints/
│   │   ├── exported_models/
│   │   └── infer/
│   │       ├── test/
│   │       └── val/
│   ├── my_experiment_det/
│   │   ├── checkpoints/
│   │   └── exported_models/
│   ├── my_experiment_cls/
│   │   ├── checkpoints/
│   │   ├── exported_models/
│   │   ├── test_results.csv
│   │   └── train.log
│   └── pic/
├── weights/
│   └── dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
├── src/
│   └── lightly_train/
│       ├── _commands/
│       ├── _data/
│       ├── _metrics/
│       ├── _models/
│       ├── _task_models/
│       └── ...
├── tests/
├── docs/
├── examples/
├── docker/
├── dev_tools/
└── lightly-train-main/
```

## 2. 主要文件说明

- `train_cls.py`
  - 图像分类训练脚本。

- `train_det.py`
  - 目标检测训练脚本。
  - 使用 `lightly_train.train_object_detection()`。

- `train_seg.py`
  - 实例分割训练脚本。

- `det_tool.py`
  - 检测后处理工具。
  - 目前包含 `infer` 和 `export` 2 个功能。
  - `infer` 负责推理、可视化和测试集评估。
  - `export` 会根据评估报告导出筛选后的新数据集，目录名后缀为 `_A`。

- `datasets/convert_datasets/sync_picture.py`
  - 数据同步脚本。
  - 先把多来源图片和对应 JSON 同步整理到 `moxingxunlian/train|val|test`。

- `datasets/convert_datasets/LabelMeToYOLO_v4.py`
  - 数据转换脚本。
  - 在 `sync_picture.py` 整理完数据后，再把 LabelMe 标注转换成检测集和分割集。

## 3. 服务器目录说明

- `datasets/military_dataset/dataset_det/`
  - 军事数据集。

- `datasets/military_dataset/`
  - 实际上包含 `dataset_det`、`dataset_seg`、`dataset_cls` 三类任务目录。
  - 文档里只展开展示了 `dataset_det` 的结构，其他两类目录结构同理。

- `datasets/military_dataset/dataset_det_A/`
  - 军事数据集筛选导出目录。
  - 对应 `det_tool.py export` 生成的新数据集位置。

- `datasets/wuwanPic_dataset/dataset_det/`
  - 五万张图片数据集原始检测目录。
  - 这是 `det_tool.py` 当前默认使用的数据集路径，后面根据项目自己修改

- `datasets/wuwanPic_dataset/`
  - 结构同军事数据集

- `datasets/wuwanPic_dataset/dataset_det_A/`
  - 五万张图片数据集筛选导出目录。
  - 对应 `det_tool.py export` 生成的新数据集位置。

- `out/my_experiment_det_0402/`
  - 改名字了，4月2号晚跑的10000轮
  - `det_tool.py` 当前默认实验输出目录。


## 4. 本周整理结论

当前流程：

```text
原始图片 + JSON
    ↓
sync_picture.py
    ↓
moxingxunlian/train|val|test
    ↓
LabelMeToYOLO_v4.py
    ↓
datasets/.../dataset_det
    ↓
train_det.py
    ↓
out/.../exported_models/exported_best.pt
    ↓
det_tool.py infer
    ↓
推理结果 + 可视化结果 + test_report.json
    ↓
det_tool.py export
    ↓
datasets/.../dataset_det_A
```

- `sync_picture.py`
  - 先整理原始图片和对应 JSON。

- `LabelMeToYOLO_v4.py`
  - 再把标注转换成训练用的 YOLO 检测数据。

- `train_det.py`
  - 用检测数据训练模型，并生成后续推理要用的权重文件。

- `det_tool.py infer`
  - 用训练好的权重做推理、可视化和评估。

- `det_tool.py export`
  - 根据评估结果导出筛选后的 `_A` 数据集。
