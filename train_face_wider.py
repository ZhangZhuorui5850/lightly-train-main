#!/usr/bin/env python3
"""使用针对 face_yolo_wider 调好的参数训练 DINOv3 ViT-S/16+ LTDETR。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import lightly_train  # noqa: E402
from tool_lib.training_run import inspect_run_mode, prepare_run  # noqa: E402


# ======================== 常用设置：直接改这里 ========================

DATASETS_ROOT = PROJECT_ROOT / "datasets" / "face_detect"
OUTPUT_ROOT = PROJECT_ROOT / "out" / "face_wider"

VARIANT = "combined"         # original / copy-paste / tile / combined
OUT: Path | None = None   # None = out/face_wider/<variant>_<image_size>
STEPS = 80000
IMAGE_SIZE = 640          # 640 / 800
BATCH_SIZE = 32           # 全局batch；0=按可见GPU数量和总显存自动选择
FRESH = False             # True=归档同名旧实验并重新训练

MODEL = "dinov3/vits16-ltdetr"
BACKBONE_WEIGHTS = (
    PROJECT_ROOT
    / "weights"
    / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)
BACKBONE_FREEZE = True
LEARNING_RATE = 5e-5
RANDOM_ZOOM_OUT = None

# ===================================================================

DATASETS = {
    "original": DATASETS_ROOT / "face_yolo_wider" / "data.yaml",
    "copy-paste": DATASETS_ROOT / "face_yolo_wider_copypaste_v1" / "data.yaml",
    "tile": DATASETS_ROOT / "face_yolo_wider_tiles_v1" / "data.yaml",
    "combined": DATASETS_ROOT / "face_yolo_wider_combined_v1" / "data.yaml",
}


def tuned_global_batch_size(image_size: int) -> int:
    """Return a conservative global batch divisible by the visible GPU count."""
    if not torch.cuda.is_available():
        return 1
    devices = max(1, torch.cuda.device_count())
    memory_gib = min(
        torch.cuda.get_device_properties(index).total_memory / 1024**3
        for index in range(devices)
    )
    if image_size >= 800:
        per_device = 1 if memory_gib < 12 else 2 if memory_gib < 20 else 4 if memory_gib < 36 else 8
    else:
        per_device = 2 if memory_gib < 12 else 4 if memory_gib < 20 else 8 if memory_gib < 36 else 16
    return min(per_device * devices, max(devices, (16 // devices) * devices))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(DATASETS), default=VARIANT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--image-size", type=int, choices=(640, 800), default=IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--fresh", action="store_true", default=FRESH)
    parser.add_argument("--print-config", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps需要为正整数")
    if args.batch_size < 0:
        raise ValueError("--batch-size需要为非负整数")
    devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.batch_size and args.batch_size % devices:
        raise ValueError(f"--batch-size={args.batch_size}需要被可见设备数{devices}整除")


def resolve_run_mode(output: Path, fresh: bool) -> tuple[bool, bool]:
    """Return ``(resume_interrupted, overwrite)`` for an experiment directory."""
    return inspect_run_mode(output, fresh)


def resolved_config(args: argparse.Namespace) -> dict[str, object]:
    data = DATASETS[args.variant].resolve()
    output = (args.out or OUTPUT_ROOT / f"{args.variant}_{args.image_size}").resolve()
    batch_size = args.batch_size or tuned_global_batch_size(args.image_size)
    resume_interrupted, overwrite = resolve_run_mode(output, args.fresh)
    return {
        "data": str(data),
        "out": str(output),
        "model": MODEL,
        "backbone_weights": str(BACKBONE_WEIGHTS.resolve()),
        "image_size": args.image_size,
        "steps": args.steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": "auto",
        "random_zoom_out": RANDOM_ZOOM_OUT,
        "resume_interrupted": resume_interrupted,
        "overwrite": overwrite,
    }


def load_dataset_config(path: Path) -> dict[str, object]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"数据配置需要为YAML映射: {path}")
    required = {"train", "val", "names"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"数据配置缺少字段: {', '.join(missing)}")
    configured_root = Path(config.get("path", "."))
    dataset_root = configured_root if configured_root.is_absolute() else path.parent / configured_root
    resolved = {
        "path": str(dataset_root.resolve()),
        "train": config["train"],
        "val": config["val"],
        "test": config.get("test"),
        "names": config["names"],
    }
    for split in ("train", "val"):
        split_path = Path(str(resolved[split]))
        image_path = split_path if split_path.is_absolute() else dataset_root / split_path
        label_parts = list(split_path.parts)
        try:
            label_parts[label_parts.index("images")] = "labels"
        except ValueError as exc:
            raise ValueError(f"{split}路径需要包含images目录: {split_path}") from exc
        label_relative = Path(*label_parts)
        label_path = label_relative if label_relative.is_absolute() else dataset_root / label_relative
        if not image_path.is_dir() or not label_path.is_dir():
            raise FileNotFoundError(f"{split}图片或标签目录不存在: {image_path} / {label_path}")
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    config = resolved_config(args)
    data = Path(str(config["data"]))
    if not data.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data}")
    if not BACKBONE_WEIGHTS.is_file():
        raise FileNotFoundError(f"backbone 权重不存在: {BACKBONE_WEIGHTS}")
    data_config = load_dataset_config(data)
    config["dataset_root"] = data_config["path"]
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return 0

    resume_interrupted, overwrite = prepare_run(
        Path(str(config["out"])),
        fresh=args.fresh,
        config={
            "task": "object_detection",
            "variant": args.variant,
            "model": MODEL,
            "backbone_weights": str(BACKBONE_WEIGHTS.resolve()),
            "backbone_freeze": BACKBONE_FREEZE,
            "lr": LEARNING_RATE,
            "data_yaml": str(data),
            "data": data_config,
            "image_size": args.image_size,
            "steps": args.steps,
            "batch_size": int(config["batch_size"]),
            "gradient_accumulation_steps": "auto",
            "random_zoom_out": RANDOM_ZOOM_OUT,
            "precision": "bf16-mixed",
            "seed": 42,
        },
    )
    config["resume_interrupted"] = resume_interrupted
    config["overwrite"] = overwrite
    print(json.dumps(config, ensure_ascii=False, indent=2))

    lightly_train.train_object_detection(
        out=str(config["out"]),
        model=MODEL,
        model_args={
            "backbone_weights": str(BACKBONE_WEIGHTS),
            "backbone_freeze": BACKBONE_FREEZE,
            "lr": LEARNING_RATE,
        },
        data=data_config,
        transform_args={
            "image_size": (args.image_size, args.image_size),
            "random_zoom_out": RANDOM_ZOOM_OUT,
        },
        metric_args={"classwise": True, "watch_metric": "val_metric/map"},
        overwrite=bool(config["overwrite"]),
        resume_interrupted=bool(config["resume_interrupted"]),
        steps=args.steps,
        batch_size=int(config["batch_size"]),
        gradient_accumulation_steps="auto",
        num_workers="auto",
        devices="auto",
        precision="bf16-mixed",
        seed=42,
        save_checkpoint_args={"save_every_num_steps": 1000},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
