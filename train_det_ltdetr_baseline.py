#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LTDETR (DINOv2 + LT-DETR) 目标检测训练 — Baseline (AdamW)

原版 LTDETR 训练，使用默认 AdamW 优化器，不做任何修改。
用于与 train_det_ltdetr_musgd.py (MuSGD版) 进行 ablation 对比。

使用方法:
    # vitb14 单卡
    python train_det_ltdetr_baseline.py --config /path/to/dataset.yaml

    # vitl14 + 6卡 H100
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_det_ltdetr_baseline.py \
        --config dataset.yaml --model dinov2/vitl14-ltdetr --devices 6 --batch-size 18

    # 快速调试
    python train_det_ltdetr_baseline.py --config dataset.yaml --steps 500 --devices 1

输出:
    out_det_ltdetr_baseline_<dataset_name>/
    ├── exported_models/
    │   ├── exported_best.pt
    │   └── exported_last.pt
    └── training_curves/
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Any, Dict

import torch

import lightly_train


# ============================================================
# 模型配置 — 可选模型列表
# ============================================================
# DINOv2:
#   dinov2/vits14-ltdetr  (ViT-S/14,  ~22M)
#   dinov2/vitb14-ltdetr  (ViT-B/14,  ~86M)
#   dinov2/vitl14-ltdetr  (ViT-L/14, ~304M)
#   dinov2/vitg14-ltdetr  (ViT-g/14, ~1.1B)
# DINOv3 (patch_size=16, 需要 --backbone-weights):
#   dinov3/vitl16-ltdetr  (ViT-L/16,  默认)
#   dinov3/vitb16-ltdetr  (ViT-B/16)
#   dinov3/vits16-ltdetr  (ViT-S/16)
MODEL_NAME = "dinov3/vitl16-ltdetr"


# ============================================================
# 默认训练参数
# ============================================================
DEFAULT_MAX_STEPS = 100_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_DEVICES = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_VAL_EVERY = 4000


# ============================================================
# 数据集配置加载
# ============================================================
def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """从 YOLO 格式 YAML 文件加载数据集配置。"""
    try:
        import yaml
    except ImportError:
        print("需要安装 PyYAML: pip install pyyaml")
        sys.exit(1)

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required_fields = ["path", "train", "nc", "names"]
    for field in required_fields:
        if field not in cfg:
            raise ValueError(f"数据集配置缺少必需字段: {field}")

    names = {}
    for k, v in cfg["names"].items():
        names[int(k)] = str(v)

    if len(names) != cfg["nc"]:
        print(f"  [警告] nc={cfg['nc']} 与 names 数量={len(names)} 不一致，以 names 为准")

    val_split = cfg.get("val", cfg["train"])
    dataset_name = config_path.stem
    if dataset_name.startswith("dataset_"):
        dataset_name = dataset_name[len("dataset_"):]

    return {
        "path": cfg["path"],
        "train": cfg["train"],
        "val": val_split,
        "test": cfg.get("test"),
        "nc": len(names),
        "names": names,
        "dataset_name": dataset_name,
        "num_train_images": _count_images(Path(cfg["path"]) / cfg["train"]),
    }


def _count_images(img_dir: Path) -> int:
    if not img_dir.exists():
        return 0
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sum(1 for f in img_dir.rglob("*") if f.suffix.lower() in img_exts)


def suggest_hyperparams(
    num_train_images: int, nc: int, batch_size: int = None
) -> Dict[str, Any]:
    if num_train_images <= 0:
        num_train_images = 5000

    if num_train_images < 5000:
        target_epochs = 300
    elif num_train_images < 50000:
        target_epochs = 150
    else:
        target_epochs = 80

    class_factor = max(1.0, nc / 80.0)
    target_epochs = int(target_epochs * math.sqrt(class_factor))

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    steps_per_epoch = max(1, num_train_images // batch_size)
    suggested_steps = steps_per_epoch * target_epochs
    suggested_steps = max(5000, min(suggested_steps, 300000))

    val_every = max(1000, min(steps_per_epoch, 5000))

    return {
        "suggested_steps": suggested_steps,
        "suggested_epochs": target_epochs,
        "steps_per_epoch": steps_per_epoch,
        "val_every": val_every,
        "num_train_images": num_train_images,
    }


# ============================================================
# 收敛曲线收集器
# ============================================================
class MetricCollectorCallback:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.curves_dir = self.output_dir / "training_curves"
        self.metrics_file = self.curves_dir / "metrics.json"
        self.train_losses = []
        self.val_losses = []
        self.maps_50 = []
        self.maps_5095 = []
        self.lrs = []
        self._last_step = 0

    def __call__(self, line: str):
        try:
            self._parse_line(line.strip())
        except Exception:
            pass

    def _parse_line(self, line: str):
        if not line:
            return
        if "Train Step" in line and "train_loss" in line:
            parts = [p.strip() for p in line.split("|")]
            step = int(parts[0].split()[2].split("/")[0])
            self._last_step = step
            for part in parts[1:]:
                kv = part.split(":")
                if len(kv) == 2:
                    key, val = kv[0].strip(), kv[1].strip()
                    if key == "train_loss":
                        self.train_losses.append((step, float(val)))
                    elif key == "lr":
                        self.lrs.append((step, float(val)))
        elif "Val Step" in line and "val_loss" in line:
            parts = [p.strip() for p in line.split("|")]
            step = self._last_step
            for part in parts[1:]:
                kv = part.split(":")
                if len(kv) == 2:
                    key, val = kv[0].strip(), kv[1].strip()
                    if key == "val_loss":
                        self.val_losses.append((step, float(val)))
        elif "val_metric/map_50" in line:
            clean = line.replace("|", "").strip()
            if clean.endswith("val_metric/map_50"):
                return
            tokens = clean.split()
            for t in tokens:
                try:
                    f = float(t)
                    if 0 <= f <= 1:
                        self.maps_50.append((self._last_step, f))
                        break
                except ValueError:
                    continue
        elif "val_metric/map_50:95" in line:
            clean = line.replace("|", "").strip()
            if clean.endswith("val_metric/map_50:95"):
                return
            tokens = clean.split()
            for t in tokens:
                try:
                    f = float(t)
                    if 0 <= f <= 1:
                        self.maps_5095.append((self._last_step, f))
                        break
                except ValueError:
                    continue

    def save_metrics(self):
        self.curves_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "train_losses": [{"step": s, "value": v} for s, v in self.train_losses],
            "val_losses": [{"step": s, "value": v} for s, v in self.val_losses],
            "maps_50": [{"step": s, "value": v} for s, v in self.maps_50],
            "maps_5095": [{"step": s, "value": v} for s, v in self.maps_5095],
            "lrs": [{"step": s, "value": v} for s, v in self.lrs],
        }
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ 指标数据已保存: {self.metrics_file}")

    def generate_plots(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("\n  [!] 未安装 matplotlib，跳过绘图")
            return

        if not any([self.train_losses, self.val_losses, self.maps_50]):
            print("\n  [!] 没有足够的指标数据")
            return

        self.curves_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid")
        dpi = 150

        # ── Loss ──
        fig, ax = plt.subplots(figsize=(14, 7))
        if self.train_losses:
            s, v = zip(*self.train_losses)
            ax.plot(s, v, "#2196F3", lw=0.8, alpha=0.85, label="Train Loss")
        if self.val_losses:
            s, v = zip(*self.val_losses)
            ax.plot(s, v, "#F44336", lw=2, marker="o", ms=3, label="Val Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("LTDETR Baseline (AdamW) Loss Curve", fontweight="bold")
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(self.curves_dir / "loss_curve.png"), dpi=dpi, facecolor="white")
        plt.close(fig)

        # ── mAP@50 ──
        if self.maps_50:
            fig, ax = plt.subplots(figsize=(14, 7))
            s, v = zip(*self.maps_50)
            ax.plot(s, v, "#4CAF50", lw=2.5, marker="D", ms=5, label="mAP@50")
            best_i = int(np.argmax(v))
            ax.scatter(
                [s[best_i]], [v[best_i]],
                color="red", s=200, marker="*", zorder=5,
                label=f"Best = {v[best_i]:.4f}",
            )
            ax.set_xlabel("Steps")
            ax.set_ylabel("mAP@50")
            ax.set_title("LTDETR Baseline (AdamW) mAP@50 Curve", fontweight="bold")
            ax.legend()
            ax.set_ylim(0, min(max(v) * 1.15, 1.05))
            fig.tight_layout()
            fig.savefig(str(self.curves_dir / "map50_curve.png"), dpi=dpi, facecolor="white")
            plt.close(fig)

        # ── mAP@50:95 ──
        if self.maps_5095:
            fig, ax = plt.subplots(figsize=(14, 7))
            s, v = zip(*self.maps_5095)
            ax.plot(s, v, "#FF9800", lw=2.5, marker="D", ms=5, label="mAP@50:95")
            best_i = int(np.argmax(v))
            ax.scatter(
                [s[best_i]], [v[best_i]],
                color="red", s=200, marker="*", zorder=5,
                label=f"Best = {v[best_i]:.4f}",
            )
            ax.set_xlabel("Steps")
            ax.set_ylabel("mAP@50:95")
            ax.set_title("LTDETR Baseline (AdamW) mAP@50:95 Curve", fontweight="bold")
            ax.legend()
            ax.set_ylim(0, min(max(v) * 1.15, 1.05))
            fig.tight_layout()
            fig.savefig(str(self.curves_dir / "map5095_curve.png"), dpi=dpi, facecolor="white")
            plt.close(fig)

        # ── Combined ──
        n_plots = sum([
            bool(self.train_losses or self.val_losses),
            bool(self.maps_50),
            bool(self.maps_5095),
        ])
        if n_plots >= 2:
            fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 6))
            if n_plots == 1:
                axes = [axes]
            idx = 0
            if self.train_losses or self.val_losses:
                ax = axes[idx]
                if self.train_losses:
                    s, v = zip(*self.train_losses)
                    ax.plot(s, v, "#2196F3", lw=0.8, label="Train")
                if self.val_losses:
                    s, v = zip(*self.val_losses)
                    ax.plot(s, v, "#F44336", lw=2, label="Val")
                ax.set_title("Loss", fontweight="bold")
                ax.legend()
                idx += 1
            if self.maps_50:
                ax = axes[idx]
                s, v = zip(*self.maps_50)
                ax.plot(s, v, "#4CAF50", lw=2.5, marker="o", ms=4)
                best_i = int(np.argmax(v))
                ax.scatter([s[best_i]], [v[best_i]], color="red", s=150, marker="*")
                ax.set_title("mAP@50", fontweight="bold")
                ax.set_ylim(0, 1)
                idx += 1
            if self.maps_5095:
                ax = axes[idx]
                s, v = zip(*self.maps_5095)
                ax.plot(s, v, "#FF9800", lw=2.5, marker="o", ms=4)
                best_i = int(np.argmax(v))
                ax.scatter([s[best_i]], [v[best_i]], color="red", s=150, marker="*")
                ax.set_title("mAP@50:95", fontweight="bold")
                ax.set_ylim(0, 1)
            fig.suptitle(
                "LTDETR Baseline (AdamW) Training Summary",
                fontsize=15, fontweight="bold",
            )
            fig.tight_layout()
            fig.savefig(
                str(self.curves_dir / "combined_curves.png"),
                dpi=dpi, facecolor="white",
            )
            plt.close(fig)

        print(f"  ✓ 收敛曲线已生成: {self.curves_dir}/")


# ============================================================
# 训练入口
# ============================================================
def train():
    """LTDETR Baseline — 原版 AdamW 优化器，不做任何改动。"""

    import argparse
    parser = argparse.ArgumentParser(
        description="LTDETR Baseline (AdamW) — 用于 MuSGD ablation 对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # DINOv3 vitl16 + 本地权重 + 6卡 H100
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python train_det_ltdetr_baseline.py \\
      --config dataset.yaml --model dinov3/vitl16-ltdetr --devices 6 \\
      --backbone-weights dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth

  # DINOv2 vitb14 单卡
  python train_det_ltdetr_baseline.py --config dataset.yaml --model dinov2/vitb14-ltdetr

  # 快速调试
  python train_det_ltdetr_baseline.py --config dataset.yaml --steps 500 --devices 1

DINOv2 模型:
  dinov2/vits14-ltdetr  – ViT-Small
  dinov2/vitb14-ltdetr  – ViT-Base
  dinov2/vitl14-ltdetr  – ViT-Large
  dinov2/vitg14-ltdetr  – ViT-Giant

DINOv3 模型 (需要 --backbone-weights):
  dinov3/vitl16-ltdetr  – ViT-Large/16 (默认)
  dinov3/vitb16-ltdetr  – ViT-Base/16
  dinov3/vits16-ltdetr  – ViT-Small/16

对比脚本:
  train_det_ltdetr_baseline.py  → AdamW (本脚本)
  train_det_ltdetr_musgd.py     → MuSGD
        """,
    )
    parser.add_argument("--config", type=str, required=True,
                        help="YOLO 格式数据集配置文件路径 (YAML)")
    parser.add_argument("--steps", type=int, default=None,
                        help="覆盖训练步数")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"覆盖 batch size (默认: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--devices", type=int, default=None,
                        help=f"覆盖 GPU 数量 (默认: {DEFAULT_NUM_DEVICES})")
    parser.add_argument("--val-every", type=int, default=None,
                        help="覆盖验证频率")
    parser.add_argument("--output", type=str, default=None,
                        help="覆盖输出目录 (默认: out_det_ltdetr_baseline_<name>)")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"模型名称 (默认: {MODEL_NAME})")
    parser.add_argument("--backbone-weights", type=str, default=None,
                        help="DINOv3 pretrained backbone 权重文件路径")
    parser.add_argument("--resume", action="store_true",
                        help="从中断的训练续训")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="从检查点微调，只加载权重")
    parser.add_argument("--early-stop", type=int, default=0, metavar="N",
                        help="早停: 连续 N 次无提升则停止。0=禁用。推荐: 8-10")
    parser.add_argument("--early-stop-delta", type=float, default=0.001,
                        help="最小 mAP 提升量")
    parser.add_argument("--early-stop-metric", type=str, default="val_metric/map",
                        choices=["val_metric/map", "val_metric/map_50"],
                        help="早停监控指标")
    parser.add_argument("--backbone-freeze", action="store_true",
                        help="冻结 ViT backbone")
    args = parser.parse_args()

    # ──── 加载数据集配置 ────
    print(f"\n加载数据集配置: {args.config}")
    ds_config = load_dataset_config(args.config)

    DATA_PATH = ds_config["path"]
    NAMES = ds_config["names"]
    NC = ds_config["nc"]
    TRAIN_SPLIT = ds_config["train"]
    VAL_SPLIT = ds_config["val"]
    DATASET_NAME = ds_config["dataset_name"]
    NUM_TRAIN = ds_config["num_train_images"]

    OUTPUT_DIR = args.output or f"out_det_ltdetr_baseline_{DATASET_NAME}"
    MODEL_NAME_VAL = args.model

    # ──── 自动建议超参数 ────
    BATCH_SIZE_VAL = args.batch_size or DEFAULT_BATCH_SIZE
    hyperparams = suggest_hyperparams(NUM_TRAIN, NC, batch_size=BATCH_SIZE_VAL)

    MAX_STEPS_VAL = args.steps or hyperparams["suggested_steps"]
    NUM_DEVICES_VAL = args.devices or DEFAULT_NUM_DEVICES
    VAL_EVERY = args.val_every or hyperparams["val_every"]

    # ──── 打印配置摘要 ────
    print(f"\n{'=' * 70}")
    print(f"  LTDETR Baseline (AdamW) — 不做任何优化器改动")
    print(f"{'=' * 70}")
    print(f"\n  数据集:        {DATASET_NAME}")
    print(f"  配置文件:      {args.config}")
    print(f"  数据路径:      {DATA_PATH}")
    print(f"  训练图片数:    {NUM_TRAIN:,}")
    print(f"  类别数:        {NC}")
    print(f"  模型:          {MODEL_NAME_VAL}")
    if args.backbone_weights:
        print(f"  Backbone权重:  {args.backbone_weights}")
    print(f"  Backbone冻结:  {'是' if args.backbone_freeze else '否'}")
    print(f"\n  训练参数:")
    print(f"    最大Steps:   {MAX_STEPS_VAL:,}")
    print(f"    Batch Size:  {BATCH_SIZE_VAL} ({NUM_DEVICES_VAL}卡×每卡{BATCH_SIZE_VAL // NUM_DEVICES_VAL})")
    print(f"    验证频率:    每 {VAL_EVERY} 步")
    print(f"    Steps/Epoch: {hyperparams['steps_per_epoch']}")
    print(f"    总Epochs:    ~{MAX_STEPS_VAL // max(hyperparams['steps_per_epoch'], 1)}")

    print(f"\n  {'─' * 50}")
    print(f"  优化器:           AdamW (原版默认)")
    print(f"    lr:             1e-4 (sqrt 缩放)")
    print(f"    betas:          (0.9, 0.999)")
    print(f"    weight_decay:   1e-4")
    print(f"    backbone_lr:    lr × 0.01")
    print(f"    scheduler:      LinearLR warmup (2000 steps)")
    print(f"    gradient_clip:  0.1")
    print(f"  {'─' * 50}")

    print(f"\n  输出目录:      {OUTPUT_DIR}")

    # ──── 类别列表 ────
    sorted_names = sorted(NAMES.items())
    print(f"\n  类别列表 ({NC} 类):")
    for k, v in sorted_names[:10]:
        print(f"    {k}: {v}")
    if NC > 15:
        print(f"    ... (省略 {NC - 15} 个)")
    for k, v in sorted_names[-5:]:
        print(f"    {k}: {v}")

    # ──── 早停 ────
    if args.early_stop > 0:
        from early_stopping_patch import apply_early_stopping_patch
        apply_early_stopping_patch(
            patience=args.early_stop,
            min_delta=args.early_stop_delta,
            monitor=args.early_stop_metric,
        )
        print(f"\n  ✓ 早停已启用: patience={args.early_stop}, delta={args.early_stop_delta}")
    else:
        print("\n  早停: 禁用 (使用 --early-stop N 启用)")

    # ──── 指标收集器 ────
    metric_collector = MetricCollectorCallback(OUTPUT_DIR)
    print(f"✓ 收敛曲线收集器已初始化")

    # ──── Hook stdout ────
    original_write = sys.stdout.write
    sys.stdout.write = lambda s: (
        original_write(s),
        metric_collector(s.rstrip("\n")) if isinstance(s, str) else None,
    )[0]

    # ──── 开始训练 ────
    start_time = time.time()

    resume_interrupted = args.resume
    checkpoint_path = args.checkpoint
    model_arg = MODEL_NAME_VAL

    if checkpoint_path:
        model_arg = checkpoint_path
        print(f"\n  [微调] 从检查点加载权重: {checkpoint_path}")
    if resume_interrupted:
        print(f"\n  [续训] 将从中断处继续: {OUTPUT_DIR}")

    print(f"\n{'=' * 70}")
    print(f"开始 LTDETR Baseline 训练...  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据路径:  {DATA_PATH}")
    print(f"  训练集:    {TRAIN_SPLIT}")
    print(f"  验证集:    {VAL_SPLIT}")
    print(f"  GPU 数:    {NUM_DEVICES_VAL} (DDP)")
    print(f"  优化器:    AdamW")
    if resume_interrupted:
        print(f"  模式:      续训")
    elif checkpoint_path:
        print(f"  模式:      微调")
    print(f"{'=' * 70}\n")

    try:
        train_kwargs = dict(
            out=OUTPUT_DIR,
            model=model_arg,
            data={
                "path": DATA_PATH,
                "train": TRAIN_SPLIT,
                "val": VAL_SPLIT,
                "names": NAMES,
            },
            steps=MAX_STEPS_VAL,
            batch_size=BATCH_SIZE_VAL,
            devices=NUM_DEVICES_VAL,
            num_workers=DEFAULT_NUM_WORKERS,
            strategy="ddp",
            precision="bf16-mixed",
            logger_args={
                "val_every_num_steps": VAL_EVERY,
            },
        )

        model_args = {}
        if args.backbone_freeze:
            model_args["backbone_freeze"] = True
        if args.backbone_weights:
            model_args["backbone_weights"] = args.backbone_weights
        if model_args:
            train_kwargs["model_args"] = model_args

        if resume_interrupted:
            train_kwargs["resume_interrupted"] = True
        else:
            train_kwargs["overwrite"] = True

        lightly_train.train_object_detection(**train_kwargs)
    finally:
        sys.stdout.write = original_write
        if args.early_stop > 0:
            from early_stopping_patch import remove_early_stopping_patch
            remove_early_stopping_patch()

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    print(f"\n{'=' * 70}")
    print(f"  训练完成!")
    print(f"  总耗时: {hours}小时 {minutes}分钟 ({elapsed / 60:.1f} 分钟)")
    print(f"{'=' * 70}")

    print(f"\n  正在生成收敛曲线...")
    metric_collector.save_metrics()
    metric_collector.generate_plots()

    print(f"\n  输出文件:")
    print(f"    最佳模型: {OUTPUT_DIR}/exported_models/exported_best.pt")
    print(f"    最后模型: {OUTPUT_DIR}/exported_models/exported_last.pt")
    print(f"    指标数据: {metric_collector.metrics_file}")
    print(f"    收敛曲线: {metric_collector.curves_dir}/")


if __name__ == "__main__":
    train()
