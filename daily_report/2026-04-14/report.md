# 每日训练测试汇报 - 2026-04-14

## 今日总览

- 记录数量: 1
- 训练记录: 1
- 独立测试记录: 0
- 任务分布: object_detection=1

## 今日工作摘要

- `NEU_train`, 类型=`train_run`, 状态=`completed`, model=`dinov3/vits16-ltdetr`, tests=1, map=0.0000

## 实验明细

## NEU_train

- 类型: `train_run`
- 状态: `completed`
- 开始时间: `2026-04-14 15:02:30`
- 结束时间: `2026-04-14 15:06:17`
- 来源目录: `/home/zzr/lightly-train-main/out/0414/NEU_train`
### 训练信息
- 任务: `object_detection`
- 数据集: train=`images/train`, val=`images/val`, test=`images/test`, path=`/home/zzr/lightly-train-main/datasets/neu_dataset/dataset_det`, images(train/val)=2592/324, classes=6
- 训练概览: model=`dinov3/vits16-ltdetr`, batch_size=4, steps=200, devices=1

### 训练结果表
| 模型 | steps | batch | train_loss | val_loss | val主指标 | best |
| --- | --- | --- | --- | --- | --- | --- |
| dinov3/vits16-ltdetr | 200 | 4 | 46.5101 | 6.0966 | 0.0000 | 0.0000 |
- 训练结果: last_train_loss=46.5101, last_val_loss=6.0966, last_val_map=0.0000, last_val_map50=0.0000, best=val_metric/map=0.0000
- 耗时: total_time=3.6 min, train_time=3.4 min (93.4%) (1.02 s/step), val_time=0.2 min (6.6%) (0.18 s/step)
### 训练关键参数
```json
{
  "model": "dinov3/vits16-ltdetr",
  "batch_size": 4,
  "steps": 200,
  "devices": 1,
  "accelerator": "CUDAAccelerator",
  "precision": "bf16-mixed",
  "strategy": "SingleDeviceStrategy",
  "num_workers": 8,
  "num_nodes": 1,
  "seed": 0,
  "overwrite": true,
  "resume_interrupted": false,
  "reuse_class_head": false
}
```
### 训练次要参数
```json
{
  "float32_matmul_precision": "highest",
  "backbone_weights": "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
  "lr": 0.0001,
  "weight_decay": 0.0001,
  "ema_momentum": 0.9999,
  "use_ema_model": true,
  "loss_alpha": 0.75,
  "loss_gamma": 2.0,
  "val_every_num_steps": 200,
  "log_every_num_steps": 20,
  "val_log_every_num_steps": 20,
  "save_best": true,
  "save_last": true,
  "save_every_num_steps": 1000,
  "watch_metric": "val_metric/map"
}
```
### 测试信息
- 已关联测试数: 1

#### 测试结果表
| 序号 | 数据集 | split | 类型 | 主指标 | map_50 | recall/正确数 | 样本数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | neu_dataset | test | detection_report | 0.0000 | 0.0000 | 0.0000 | 324 |

#### 测试 1
- 测试对象: `neu_dataset / test`
- 测试运行名: `0414-NEU-train-neu-test`
- 测试结果: map=0.0000, map_50=0.0000, map_75=0.0000, map_05=0.0000, precision_05=0.0000, recall_05=0.0000, images=324, avg_infer_time_ms=45.16
- Top AP 类别: crazing=0.0000, inclusion=0.0000, patches=0.0000, pitted_surface=0.0000, rolled-in_scale=0.0000
### 测试关键参数
```json
{
  "input_mode": "dataset",
  "split": "test",
  "score_threshold": 0.6,
  "report_iou_threshold": 0.5,
  "checkpoint_path": "/home/zzr/lightly-train-main/out/0414/NEU_train/exported_models/exported_best.pt",
  "data_yaml": "/home/zzr/lightly-train-main/datasets/neu_dataset/dataset_det/data.yaml"
}
```
### 测试次要参数
```json
{
  "device": "auto",
  "save_visualization": true,
  "save_json": false,
  "save_txt": false,
  "compute_metrics": true,
  "metric_classwise": false,
  "data_root": "/home/zzr/lightly-train-main/datasets/neu_dataset/dataset_det"
}
```

### 工作笔记
- linked infer run: /home/zzr/lightly-train-main/out/infer/det/0414-NEU-train-neu-test
- 检测 mAP@0.5 为 0，建议优先检查标签类别映射、阈值、权重是否匹配当前数据集。
