from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))


# ---------------------------
# Experiment settings
# ---------------------------

OUT = Path("out/seg/NEU_train")

# Pick the EoMT backbone here. Keep this in sync with BACKBONE_WEIGHTS.
# Examples:
#   "dinov3/vits16-eomt"
#   "dinov3/vitl16-eomt"
#   "dinov2/vitb14-eomt"
MODEL = "dinov3/vits16-eomt"

# Upstream pretrained backbone weights. Set to None to use the model default weights.
# This must be a backbone checkpoint, not a full segmentation model checkpoint.
BACKBONE_WEIGHTS: Path | None = Path(
    "weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

# Optional full segmentation checkpoint for continuing fine-tuning.
# Use this when you have out/.../exported_models/exported_best.pt from LightlyTrain.
CHECKPOINT: Path | None = None


# ---------------------------
# Dataset settings
# ---------------------------

# Semantic segmentation expects PNG masks. Pixel values in the mask must be class IDs.
# The mask filename should match the image stem, for example:
#   images/train/foo.jpg -> masks/train/foo.png
DATA_ROOT = Path("datasets/neu_dataset/dataset_semantic")
TRAIN_IMAGES = DATA_ROOT / "images/train"
TRAIN_MASKS = DATA_ROOT / "masks/train"
VAL_IMAGES = DATA_ROOT / "images/val"
VAL_MASKS = DATA_ROOT / "masks/val"

# If you already have class names in a YOLO-style data.yaml, this script can reuse
# only the "names" mapping. It does not use YOLO txt labels for semantic training.
CLASS_YAML: Path | None = Path("datasets/neu_dataset/dataset_seg/data.yaml")

# Used when CLASS_YAML is None.
CLASSES: dict[int, str] = {
    0: "crazing",
    1: "inclusion",
    2: "patches",
    3: "pitted_surface",
    4: "rolled-in_scale",
    5: "scratches",
}

# Set this if one or more mask class IDs should not contribute to training/metrics.
IGNORE_CLASSES: list[int] = []


# ---------------------------
# Training settings
# ---------------------------

STEPS: int | str = 200
BATCH_SIZE: int | str = 4
NUM_WORKERS: int | str = "auto"
DEVICES: int | str | list[int] = "auto"
OVERWRITE = True
RESUME_INTERRUPTED = False

# Useful knobs for backbone comparisons.
MODEL_EXTRA_ARGS: dict[str, Any] = {
    # "patch_size": 32,
    # "num_queries": 200,
    # "backbone_freeze": True,
    # "lr": 1e-4,
}

TRANSFORM_ARGS: dict[str, Any] = {
    # "image_size": (512, 512),
    # "num_channels": 3,
}

METRIC_ARGS: dict[str, Any] = {
    "classwise": True,
}


def load_classes(class_yaml: Path | None, fallback: dict[int, str]) -> dict[int, str]:
    if class_yaml is None:
        return fallback
    with class_yaml.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    names = cfg.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    raise ValueError(f"Unsupported class mapping in {class_yaml}: {names!r}")


def validate_dir(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a directory: {path}")


def build_model_args() -> dict[str, Any] | None:
    model_args = dict(MODEL_EXTRA_ARGS)
    if BACKBONE_WEIGHTS is not None:
        if not BACKBONE_WEIGHTS.exists():
            raise FileNotFoundError(f"Backbone weights not found: {BACKBONE_WEIGHTS}")
        model_args["backbone_weights"] = str(BACKBONE_WEIGHTS)
    return model_args or None


def main() -> None:
    import lightly_train

    validate_dir(TRAIN_IMAGES, "TRAIN_IMAGES")
    validate_dir(TRAIN_MASKS, "TRAIN_MASKS")
    validate_dir(VAL_IMAGES, "VAL_IMAGES")
    validate_dir(VAL_MASKS, "VAL_MASKS")
    if CHECKPOINT is not None and not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    data: dict[str, Any] = {
        "train": {
            "images": str(TRAIN_IMAGES),
            "masks": str(TRAIN_MASKS),
        },
        "val": {
            "images": str(VAL_IMAGES),
            "masks": str(VAL_MASKS),
        },
        "classes": load_classes(CLASS_YAML, CLASSES),
    }
    if IGNORE_CLASSES:
        data["ignore_classes"] = IGNORE_CLASSES

    train_kwargs: dict[str, Any] = {
        "out": str(OUT),
        "model": MODEL,
        "model_args": build_model_args(),
        "data": data,
        "overwrite": OVERWRITE,
        "resume_interrupted": RESUME_INTERRUPTED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "devices": DEVICES,
        "transform_args": TRANSFORM_ARGS or None,
        "metric_args": METRIC_ARGS or None,
    }
    if CHECKPOINT is not None:
        train_kwargs["checkpoint"] = str(CHECKPOINT)

    lightly_train.train_semantic_segmentation(**train_kwargs)


if __name__ == "__main__":
    main()
