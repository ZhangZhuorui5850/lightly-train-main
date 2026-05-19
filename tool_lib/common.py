"""公共运行时与工具函数。

这个文件主要放三类内容：
- 全局默认配置和路径常量
- 运行时依赖的延迟导入，例如 lightly_train / torch / PIL
- 各任务都会复用的通用函数，例如解析 checkpoint、读取 data.yaml、列出图片等

设计上它不直接负责某个具体任务，只给 cls / det / seg 模块提供基础能力。
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import ModuleType

    import numpy as np_types
    import torch as torch_types
    import yaml as yaml_types
    from PIL import Image as PILImageModule
    from PIL import ImageDraw as PILImageDrawModule
    from PIL import ImageFont as PILImageFontModule
else:
    ModuleType = Any
    torch_types = Any
    yaml_types = Any
    np_types = Any
    PILImageModule = Any
    PILImageDrawModule = Any
    PILImageFontModule = Any

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

lightly_train: ModuleType | None = None
np: ModuleType | None = None
torch: ModuleType | None = None
yaml: ModuleType | None = None
Image: Any = None
ImageDraw: Any = None
ImageFont: Any = None
file_helpers: Any = None
yolo_helpers: Any = None
ObjectDetectionTaskMetric: Any = None
ObjectDetectionTaskMetricArgs: Any = None
InstanceSegmentationTaskMetric: Any = None
InstanceSegmentationTaskMetricArgs: Any = None

OUT_DIR = ROOT_DIR / "out"
EXPERIMENT_ROOT_DIR = OUT_DIR
TEST_OUTPUT_ROOT_DIR = OUT_DIR
EDA_OUTPUT_ROOT_DIR = OUT_DIR / "EDA"
ALL_REPORT_ROOT_DIR = OUT_DIR / "all_report"
REPORT_ARCHIVE_ROOT_DIR = OUT_DIR / "test_reports"
DATASET_DIR = ROOT_DIR / "datasets" / "military_dataset" / "dataset_det"
EXPERIMENT_DIR = OUT_DIR / "my_experiment_det_0402"

DEFAULT_CHECKPOINT = None
DEFAULT_DEVICE = "auto"
DEFAULT_OVERWRITE = False
DEFAULT_CLS_THRESHOLD = 0.5
DEFAULT_SEG_THRESHOLD = 0.8
DEFAULT_SCORE_THRESHOLD = 0.3

VISUALIZATION_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

INFER_DEFAULT_EXPERIMENT_DIR = EXPERIMENT_DIR
INFER_DEFAULT_IMAGE = None
INFER_DEFAULT_IMAGE_DIR = DATASET_DIR / "images" / "test"
INFER_DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "infer-test"
INFER_DEFAULT_DATA = DATASET_DIR / "data.yaml"
INFER_DEFAULT_SPLIT = "test"

INFER_DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
INFER_DEFAULT_SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
INFER_DEFAULT_DEVICE = DEFAULT_DEVICE
INFER_DEFAULT_OVERWRITE = DEFAULT_OVERWRITE

INFER_DEFAULT_SAVE_VISUALIZATION = True
INFER_DEFAULT_SAVE_JSON = True
INFER_DEFAULT_SAVE_TXT = False
INFER_DEFAULT_REPORT_IOU_THRESHOLD = 0.5
INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD = 0.3
INFER_DEFAULT_COMPUTE_METRICS = False
INFER_DEFAULT_METRIC_CLASSWISE = False
INFER_DEFAULT_SAVE_TEST_REPORT = True
INFER_DEFAULT_REPORT_PATH = INFER_DEFAULT_OUTPUT_DIR / "test_report.json"

EVAL_DEFAULT_DATA = DATASET_DIR / "data.yaml"
EVAL_DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "infer-test"
EVAL_DEFAULT_REPORT_PATH = EVAL_DEFAULT_OUTPUT_DIR / "test_report.json"

EXPORT_DEFAULT_REPORT_JSON = EVAL_DEFAULT_REPORT_PATH
EXPORT_DEFAULT_SOURCE_DATA = EVAL_DEFAULT_DATA
EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD = 0.0
EXPORT_DEFAULT_AUTO_BALANCE = True
EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD = True
EXPORT_DEFAULT_BALANCE_RATIO = 0.0
EXPORT_DEFAULT_MIN_CLASS_IMAGES = 0
EXPORT_DEFAULT_MIN_CLASS_BOXES = 0
EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS = 0
EXPORT_DEFAULT_TARGET_TOTAL_IMAGES = 0
EXPORT_DEFAULT_SPLIT_RATIO = "8:1:1"
EXPORT_DEFAULT_TARGET_BOXES_PER_CLASS = 0
EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE = 0
EXPORT_DEFAULT_MAX_BOXES_PER_CLASS_PER_IMAGE = 0
EXPORT_DEFAULT_BOX_DENSITY_PENALTY = 0.0
EXPORT_DEFAULT_EXPORT_SUFFIX = "_A"

TRAIN_CLS_SCRIPT = ROOT_DIR / "train_cls.py"
TRAIN_DET_SCRIPT = ROOT_DIR / "train_det.py"
TRAIN_SEG_SCRIPT = ROOT_DIR / "train_seg.py"
TEST_CLS_SCRIPT = ROOT_DIR / "test_cls.py"
TEST_DET_SCRIPT = ROOT_DIR / "test_det_v3.py"

TASK_NAME_MARKERS = {
    "cls": ("cls", "class"),
    "det": ("det", "detect"),
    "seg": ("seg",),
}

TRAINING_CURVE_FILENAMES = {
    "loss": "training_curve_loss.png",
    "map": "training_curve_map.png",
    "lr": "training_curve_lr.png",
    "dashboard": "training_dashboard.png",
}
IMPORTANT_ARTIFACT_DIRNAME = "important"
TRAINING_TEMP_DIRNAME = "_temp"
INFER_TEMP_DIRNAME = "_tempfile"
TENSORBOARD_EVENT_GLOB = "events.out.tfevents.*"


@dataclass
class ImageSample:
    image_path: Path
    relative_path: Path
    label_path: Path | None = None


def experiment_checkpoint_candidates(experiment_dir: Path) -> list[Path]:
    return [
        experiment_dir / "exported_models" / "exported_best.pt",
        experiment_dir / "exported_models" / "exported_last.pt",
        experiment_dir / "checkpoints" / "best.ckpt",
        experiment_dir / "checkpoints" / "last.ckpt",
    ]


def is_task_experiment_dir(path: Path, task: str, *, require_checkpoint: bool = False) -> bool:
    if not path.is_dir():
        return False

    has_artifact_dir = (path / "exported_models").is_dir() or (path / "checkpoints").is_dir()
    if not has_artifact_dir:
        return False

    if not require_checkpoint:
        markers = TASK_NAME_MARKERS.get(task, (task,))
        try:
            path_text = str(path.relative_to(EXPERIMENT_ROOT_DIR)).lower()
        except ValueError:
            path_text = str(path).lower()
        return any(marker in path_text for marker in markers)

    markers = TASK_NAME_MARKERS.get(task, (task,))
    try:
        path_text = str(path.relative_to(EXPERIMENT_ROOT_DIR)).lower()
    except ValueError:
        path_text = str(path).lower()
    if any(marker in path_text for marker in markers):
        return any(candidate.exists() for candidate in experiment_checkpoint_candidates(path))

    return False


def is_experiment_dir(path: Path, *, require_checkpoint: bool = False) -> bool:
    if not path.is_dir():
        return False

    has_artifact_dir = (path / "exported_models").is_dir() or (path / "checkpoints").is_dir()
    if not has_artifact_dir:
        return False

    if not require_checkpoint:
        return True
    return any(candidate.exists() for candidate in experiment_checkpoint_candidates(path))


def discover_recent_experiment_dirs(
    task: str,
    *,
    limit: int | None = None,
    require_checkpoint: bool = False,
) -> list[Path]:
    if not EXPERIMENT_ROOT_DIR.exists():
        return []

    all_candidates = [
        path
        for path in EXPERIMENT_ROOT_DIR.rglob("*")
        if is_experiment_dir(path, require_checkpoint=require_checkpoint)
    ]
    candidates = all_candidates
    candidates.sort(
        key=lambda p: (
            p.stat().st_mtime,
            1 if is_task_experiment_dir(p, task, require_checkpoint=require_checkpoint) else 0,
        ),
        reverse=True,
    )
    if limit is not None:
        return candidates[:limit]
    return candidates


def auto_resolve_experiment_dir(task: str, fallback: Path) -> Path:
    candidates = discover_recent_experiment_dirs(task, limit=1, require_checkpoint=True)
    if candidates:
        return candidates[0]
    return fallback


def apply_user_settings(settings: dict[str, Any]) -> None:
    global OUT_DIR
    global EXPERIMENT_ROOT_DIR
    global TEST_OUTPUT_ROOT_DIR
    global EDA_OUTPUT_ROOT_DIR
    global ALL_REPORT_ROOT_DIR
    global DATASET_DIR
    global EXPERIMENT_DIR
    global REPORT_ARCHIVE_ROOT_DIR
    global INFER_DEFAULT_EXPERIMENT_DIR
    global INFER_DEFAULT_IMAGE_DIR
    global INFER_DEFAULT_OUTPUT_DIR
    global INFER_DEFAULT_DATA
    global INFER_DEFAULT_SPLIT
    global EVAL_DEFAULT_DATA
    global EVAL_DEFAULT_OUTPUT_DIR
    global EVAL_DEFAULT_REPORT_PATH
    global EXPORT_DEFAULT_REPORT_JSON
    global EXPORT_DEFAULT_SOURCE_DATA
    global EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD
    global EXPORT_DEFAULT_AUTO_BALANCE
    global EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD
    global EXPORT_DEFAULT_BALANCE_RATIO
    global EXPORT_DEFAULT_MIN_CLASS_IMAGES
    global EXPORT_DEFAULT_MIN_CLASS_BOXES
    global EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS
    global EXPORT_DEFAULT_TARGET_TOTAL_IMAGES
    global EXPORT_DEFAULT_SPLIT_RATIO
    global EXPORT_DEFAULT_TARGET_BOXES_PER_CLASS
    global EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE
    global EXPORT_DEFAULT_MAX_BOXES_PER_CLASS_PER_IMAGE
    global EXPORT_DEFAULT_BOX_DENSITY_PENALTY
    global EXPORT_DEFAULT_EXPORT_SUFFIX
    global TRAIN_CLS_SCRIPT
    global TRAIN_DET_SCRIPT
    global TRAIN_SEG_SCRIPT
    global TEST_CLS_SCRIPT
    global TEST_DET_SCRIPT
    global DEFAULT_CLS_THRESHOLD
    global DEFAULT_SEG_THRESHOLD
    global DEFAULT_SCORE_THRESHOLD
    global INFER_DEFAULT_SCORE_THRESHOLD
    global INFER_DEFAULT_REPORT_IOU_THRESHOLD

    def _path(key: str, current: Path) -> Path:
        value = settings.get(key)
        if value is None:
            return current
        path = Path(value)
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        return path

    OUT_DIR = _path("out_dir", OUT_DIR)
    EXPERIMENT_ROOT_DIR = _path("experiment_root_dir", EXPERIMENT_ROOT_DIR)
    TEST_OUTPUT_ROOT_DIR = _path(
        "infer_output_root_dir",
        _path("test_output_root_dir", TEST_OUTPUT_ROOT_DIR),
    )
    EDA_OUTPUT_ROOT_DIR = _path("eda_output_root_dir", EDA_OUTPUT_ROOT_DIR)
    ALL_REPORT_ROOT_DIR = _path("all_report_root_dir", ALL_REPORT_ROOT_DIR)
    REPORT_ARCHIVE_ROOT_DIR = _path("report_archive_root_dir", REPORT_ARCHIVE_ROOT_DIR)
    DATASET_DIR = _path("det_dataset_dir", DATASET_DIR)
    DEFAULT_CLS_THRESHOLD = float(settings.get("cls_threshold", DEFAULT_CLS_THRESHOLD))
    DEFAULT_SEG_THRESHOLD = float(settings.get("seg_threshold", DEFAULT_SEG_THRESHOLD))
    DEFAULT_SCORE_THRESHOLD = float(settings.get("det_score_threshold", DEFAULT_SCORE_THRESHOLD))
    det_experiment_value = settings.get("det_experiment_dir")
    if det_experiment_value in {None, "", "auto"}:
        EXPERIMENT_DIR = auto_resolve_experiment_dir("det", EXPERIMENT_DIR)
    else:
        EXPERIMENT_DIR = _path("det_experiment_dir", EXPERIMENT_DIR)
    INFER_DEFAULT_SPLIT = settings.get("det_default_split", INFER_DEFAULT_SPLIT)
    INFER_DEFAULT_SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
    INFER_DEFAULT_REPORT_IOU_THRESHOLD = float(
        settings.get("det_report_iou_threshold", INFER_DEFAULT_REPORT_IOU_THRESHOLD)
    )

    default_data_yaml = DATASET_DIR / "data.yaml"
    default_image_dir = DATASET_DIR / "images" / INFER_DEFAULT_SPLIT
    default_infer_output_dir = build_task_infer_root("det") / f"manual_infer-{INFER_DEFAULT_SPLIT}"
    default_report_path = build_det_report_path(default_infer_output_dir)

    INFER_DEFAULT_EXPERIMENT_DIR = _path("det_infer_experiment_dir", EXPERIMENT_DIR)
    INFER_DEFAULT_IMAGE_DIR = _path("det_infer_image_dir", default_image_dir)
    INFER_DEFAULT_OUTPUT_DIR = _path("det_infer_output_dir", default_infer_output_dir)
    INFER_DEFAULT_DATA = _path("det_data_yaml", default_data_yaml)

    EVAL_DEFAULT_DATA = _path("det_eval_data_yaml", INFER_DEFAULT_DATA)
    EVAL_DEFAULT_OUTPUT_DIR = _path("det_eval_output_dir", default_infer_output_dir)
    EVAL_DEFAULT_REPORT_PATH = _path("det_eval_report_path", default_report_path)

    EXPORT_DEFAULT_REPORT_JSON = _path("det_export_report_json", EVAL_DEFAULT_REPORT_PATH)
    EXPORT_DEFAULT_SOURCE_DATA = _path("det_export_source_data", EVAL_DEFAULT_DATA)
    EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD = float(
        settings.get("det_export_good_class_threshold", EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD)
    )
    EXPORT_DEFAULT_AUTO_BALANCE = bool(
        settings.get("det_export_auto_balance", EXPORT_DEFAULT_AUTO_BALANCE)
    )
    EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD = bool(
        settings.get(
            "det_export_auto_relax_class_threshold",
            EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD,
        )
    )
    EXPORT_DEFAULT_BALANCE_RATIO = float(
        settings.get("det_export_balance_ratio", EXPORT_DEFAULT_BALANCE_RATIO)
    )
    EXPORT_DEFAULT_MIN_CLASS_IMAGES = int(
        settings.get("det_export_min_class_images", EXPORT_DEFAULT_MIN_CLASS_IMAGES)
    )
    EXPORT_DEFAULT_MIN_CLASS_BOXES = int(
        settings.get("det_export_min_class_boxes", EXPORT_DEFAULT_MIN_CLASS_BOXES)
    )
    EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS = int(
        settings.get(
            "det_export_target_images_per_class",
            EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS,
        )
    )
    EXPORT_DEFAULT_TARGET_TOTAL_IMAGES = int(
        settings.get(
            "det_export_target_total_images",
            EXPORT_DEFAULT_TARGET_TOTAL_IMAGES,
        )
    )
    EXPORT_DEFAULT_SPLIT_RATIO = str(
        settings.get("det_export_split_ratio", EXPORT_DEFAULT_SPLIT_RATIO)
    )
    EXPORT_DEFAULT_TARGET_BOXES_PER_CLASS = int(
        settings.get(
            "det_export_target_boxes_per_class",
            EXPORT_DEFAULT_TARGET_BOXES_PER_CLASS,
        )
    )
    EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE = int(
        settings.get("det_export_max_boxes_per_image", EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE)
    )
    EXPORT_DEFAULT_MAX_BOXES_PER_CLASS_PER_IMAGE = int(
        settings.get(
            "det_export_max_boxes_per_class_per_image",
            EXPORT_DEFAULT_MAX_BOXES_PER_CLASS_PER_IMAGE,
        )
    )
    EXPORT_DEFAULT_BOX_DENSITY_PENALTY = float(
        settings.get("det_export_box_density_penalty", EXPORT_DEFAULT_BOX_DENSITY_PENALTY)
    )
    EXPORT_DEFAULT_EXPORT_SUFFIX = str(
        settings.get("det_export_suffix", EXPORT_DEFAULT_EXPORT_SUFFIX)
    )

    TRAIN_CLS_SCRIPT = _path("train_cls_script", TRAIN_CLS_SCRIPT)
    TRAIN_DET_SCRIPT = _path("train_det_script", TRAIN_DET_SCRIPT)
    TRAIN_SEG_SCRIPT = _path("train_seg_script", TRAIN_SEG_SCRIPT)
    TEST_CLS_SCRIPT = _path("test_cls_script", TEST_CLS_SCRIPT)
    TEST_DET_SCRIPT = _path("test_det_script", TEST_DET_SCRIPT)


def import_runtime_dependencies() -> None:
    global Image
    global ImageDraw
    global ImageFont
    global InstanceSegmentationTaskMetric
    global InstanceSegmentationTaskMetricArgs
    global ObjectDetectionTaskMetric
    global ObjectDetectionTaskMetricArgs
    global file_helpers
    global lightly_train
    global np
    global torch
    global yaml
    global yolo_helpers

    try:
        import lightly_train as lightly_train_module
        import numpy as np_module
        import torch as torch_module
        import yaml as yaml_module
        from PIL import Image as image_module
        from PIL import ImageDraw as image_draw_module
        from PIL import ImageFont as image_font_module
        from lightly_train._data import (
            file_helpers as file_helpers_module,
            yolo_helpers as yolo_helpers_module,
        )
        from lightly_train._metrics.detection.task_metric import (
            ObjectDetectionTaskMetric as object_detection_task_metric_module,
        )
        from lightly_train._metrics.detection.task_metric import (
            ObjectDetectionTaskMetricArgs as object_detection_task_metric_args_module,
        )
        from lightly_train._metrics.instance_segmentation.task_metric import (
            InstanceSegmentationTaskMetric as instance_segmentation_task_metric_module,
        )
        from lightly_train._metrics.instance_segmentation.task_metric import (
            InstanceSegmentationTaskMetricArgs as instance_segmentation_task_metric_args_module,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing runtime dependency. Please run this script inside the lightly-train "
            "training environment where numpy, torch, Pillow, PyYAML and lightly_train "
            "are installed."
        ) from exc

    lightly_train = lightly_train_module
    np = np_module
    torch = torch_module
    yaml = yaml_module
    Image = image_module
    ImageDraw = image_draw_module
    ImageFont = image_font_module
    file_helpers = file_helpers_module
    yolo_helpers = yolo_helpers_module
    ObjectDetectionTaskMetric = object_detection_task_metric_module
    ObjectDetectionTaskMetricArgs = object_detection_task_metric_args_module
    InstanceSegmentationTaskMetric = instance_segmentation_task_metric_module
    InstanceSegmentationTaskMetricArgs = instance_segmentation_task_metric_args_module


def resolve_device(device: str) -> Any:
    if device == "auto":
        return None
    if torch is None:
        raise ModuleNotFoundError("torch is not imported. Call import_runtime_dependencies() first.")
    return torch.device(device)


def resolve_checkpoint_path(checkpoint: Path | None, experiment_dir: Path | None) -> Path:
    if checkpoint is not None and checkpoint.is_dir():
        experiment_dir = checkpoint
        checkpoint = None

    if checkpoint is not None:
        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint

    if experiment_dir is None:
        raise ValueError("One of --checkpoint or --experiment-dir must be provided.")

    experiment_dir = experiment_dir.expanduser().resolve()
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    candidates = experiment_checkpoint_candidates(experiment_dir)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No supported checkpoint/model file found under: {experiment_dir}")


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output_dir}. Use overwrite to continue.")
    output_dir.mkdir(parents=True, exist_ok=True)


def load_data_config(data_path: Path) -> dict[str, Any]:
    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data config does not exist: {data_path}")
    if yaml is None:
        raise ModuleNotFoundError("yaml is not imported. Call import_runtime_dependencies() first.")
    with data_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid data config: {data_path}")

    base_dir = data_path.parent
    root = cfg.get("path")
    root_dir = base_dir if root is None else resolve_data_yaml_path(Path(root), base_dir=base_dir)
    cfg["_data_yaml_path"] = data_path
    cfg["_base_dir"] = base_dir
    cfg["_root_dir"] = resolve_dataset_root_dir(
        configured_root_dir=root_dir,
        data_yaml_dir=base_dir,
        split_values={
            "train": cfg.get("train"),
            "val": cfg.get("val"),
            "test": cfg.get("test"),
        },
    )
    return cfg


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        raise ModuleNotFoundError("yaml is not imported. Call import_runtime_dependencies() first.")
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def normalize_names(raw_names: Any) -> dict[int, str]:
    if raw_names is None:
        return {}
    if isinstance(raw_names, list):
        return {idx: str(name) for idx, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        return {int(idx): str(name) for idx, name in raw_names.items()}
    raise ValueError(f"Unsupported names format: {type(raw_names)!r}")


def resolve_data_yaml_path(path_value: Path, *, base_dir: Path) -> Path:
    """Resolve paths in data.yaml.

    Priority:
    1. absolute path
    2. path relative to project root, e.g. datasets/...
    3. path relative to the data.yaml directory
    """
    path_value = path_value.expanduser()
    if path_value.is_absolute():
        return path_value.resolve()

    root_relative = (ROOT_DIR / path_value).resolve()
    base_relative = (base_dir / path_value).resolve()

    path_text = path_value.as_posix()
    if path_text.startswith("./") or path_text.startswith("../"):
        return base_relative
    if path_text == ".":
        return base_relative
    if root_relative.exists() and not base_relative.exists():
        return root_relative
    if path_text.startswith("datasets/") or path_text.startswith("out/") or path_text.startswith("weights/"):
        return root_relative
    if base_relative.exists():
        return base_relative
    return root_relative


def resolve_dataset_root_dir(
    *,
    configured_root_dir: Path,
    data_yaml_dir: Path,
    split_values: dict[str, Any] | None = None,
) -> Path:
    """Resolve the effective dataset root.

    When a dataset directory is renamed manually, `data.yaml` may still contain an
    absolute `path:` pointing to the old directory. In that case, prefer the
    directory that currently contains `data.yaml` if it already looks like a valid
    dataset root for the declared split paths.
    """
    resolved_root_dir = configured_root_dir.expanduser().resolve()
    resolved_data_yaml_dir = data_yaml_dir.expanduser().resolve()
    if resolved_root_dir.exists():
        return resolved_root_dir
    if not resolved_data_yaml_dir.exists():
        return resolved_root_dir

    declared_splits = split_values or {}
    for split_name in ("train", "val", "test"):
        split_value = declared_splits.get(split_name)
        if split_value in {None, ""}:
            continue
        split_path = Path(str(split_value))
        candidate_dir = resolve_split_dir_path(
            split_path=split_path,
            root_dir=resolved_data_yaml_dir,
            base_dir=resolved_data_yaml_dir,
        )
        if candidate_dir.exists():
            return resolved_data_yaml_dir
    return resolved_root_dir


def resolve_dataset_split_paths(data_cfg: dict[str, Any], split: str) -> tuple[Path, Path | None, dict[int, str]]:
    root_dir = Path(data_cfg["_root_dir"])
    base_dir = Path(data_cfg.get("_base_dir", data_cfg["_data_yaml_path"]).parent)
    names = normalize_names(data_cfg.get("names"))
    train = Path(str(data_cfg.get("train", "")))
    val = Path(str(data_cfg.get("val", "")))
    test_value = data_cfg.get("test")
    test = Path(str(test_value)) if test_value else None

    split_path_map = {
        "train": train,
        "val": val,
        "test": test,
    }
    split_path = split_path_map[split]
    if split_path is None:
        raise ValueError(f"Split '{split}' is not defined in the data config.")

    image_dir = resolve_split_dir_path(
        split_path=split_path,
        root_dir=root_dir,
        base_dir=base_dir,
    )
    label_dir = infer_label_dir_from_image_dir(image_dir=image_dir, root_dir=root_dir)
    if image_dir is None:
        raise ValueError(f"Split '{split}' is not defined in the data config.")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if label_dir is not None and not label_dir.exists():
        label_dir = None
    return image_dir, label_dir, names


def resolve_split_dir_path(*, split_path: Path, root_dir: Path, base_dir: Path) -> Path:
    if split_path.is_absolute():
        return split_path.resolve()

    split_text = split_path.as_posix()
    if split_text.startswith("./") or split_text.startswith("../") or split_text == ".":
        return (base_dir / split_path).resolve()

    root_joined = (root_dir / split_path).resolve()
    project_joined = (ROOT_DIR / split_path).resolve()

    if split_text.startswith("datasets/") or split_text.startswith("out/") or split_text.startswith("weights/"):
        return project_joined
    if root_joined.exists():
        return root_joined
    if project_joined.exists() and not root_joined.exists():
        return project_joined
    return root_joined


def infer_label_dir_from_image_dir(*, image_dir: Path, root_dir: Path) -> Path | None:
    parts = list(image_dir.parts)
    for index, part in enumerate(parts):
        if part == "images":
            label_parts = parts.copy()
            label_parts[index] = "labels"
            candidate = Path(*label_parts)
            return candidate.resolve()

    try:
        rel = image_dir.resolve().relative_to(root_dir.resolve())
    except ValueError:
        return None
    candidate = (root_dir / rel).resolve()
    return candidate if candidate != image_dir.resolve() else None


def list_dataset_samples(data_cfg: dict[str, Any], split: str) -> tuple[list[ImageSample], dict[int, str]]:
    image_dir, label_dir, names = resolve_dataset_split_paths(data_cfg=data_cfg, split=split)
    samples: list[ImageSample] = []
    for rel_image in file_helpers.list_image_filenames_from_dir(image_dir=image_dir):
        rel_path = Path(rel_image)
        samples.append(
            ImageSample(
                image_path=image_dir / rel_path,
                relative_path=rel_path,
                label_path=(label_dir / rel_path).with_suffix(".txt") if label_dir else None,
            )
        )
    return samples, names


def list_directory_samples(image_dir: Path) -> list[ImageSample]:
    image_dir = image_dir.expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    samples: list[ImageSample] = []
    for rel_image in file_helpers.list_image_filenames_from_dir(image_dir=image_dir):
        rel_path = Path(rel_image)
        samples.append(ImageSample(image_path=image_dir / rel_path, relative_path=rel_path))
    return samples


def get_model_class_names(model: Any) -> dict[int, str]:
    classes = getattr(model, "classes", None)
    if isinstance(classes, dict):
        return {int(class_id): str(name) for class_id, name in classes.items()}
    if isinstance(classes, list):
        return {idx: str(name) for idx, name in enumerate(classes)}
    return {}


def merge_class_names(model_class_names: dict[int, str], data_class_names: dict[int, str]) -> dict[int, str]:
    merged = dict(model_class_names)
    merged.update(data_class_names)
    return merged


def ensure_image_samples(samples: list[ImageSample]) -> None:
    if not samples:
        raise ValueError("No images found.")


def save_records_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_image_files(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in VISUALIZATION_SUFFIXES else []
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return sorted([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VISUALIZATION_SUFFIXES])


def scalar_int(value: Any) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def scalar_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def build_metric_label_mapping(class_names: dict[int, str]) -> tuple[dict[int, int], list[str]]:
    class_ids = sorted(class_names.keys())
    mapping = {class_id: idx for idx, class_id in enumerate(class_ids)}
    metric_class_names = [class_names[class_id] for class_id in class_ids]
    return mapping, metric_class_names


def remap_labels(labels: Any, mapping: dict[int, int]) -> Any:
    if torch is None:
        raise ModuleNotFoundError("torch is not imported. Call import_runtime_dependencies() first.")
    if labels.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64)
    return torch.as_tensor([mapping[int(label)] for label in labels.tolist()], dtype=torch.int64)


def visualization_suffix(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return suffix if suffix in VISUALIZATION_SUFFIXES else ".png"


def sanitize_tag(value: str) -> str:
    cleaned = []
    for ch in value.strip():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("-")
    text = "".join(cleaned).strip("-_")
    while "--" in text:
        text = text.replace("--", "-")
    return text or "unknown"


def dataset_tag_from_dir(dataset_dir: Path) -> str:
    dataset_dir = dataset_dir.expanduser().resolve()
    name = dataset_dir.name
    parent_name = dataset_dir.parent.name
    parent_tag = sanitize_tag(parent_name.removesuffix("_dataset"))
    if name.startswith("dataset_"):
        suffix = name.removeprefix("dataset_")
        if suffix in {"det", "detect", "detection", "seg", "cls", "class", "classification"}:
            return parent_tag
        return sanitize_tag(f"{parent_tag}_{suffix}")
    return sanitize_tag(name.removesuffix("_dataset"))


def extract_date_token(value: str) -> str | None:
    text = value.strip()
    if len(text) == 4 and text.isdigit():
        return text
    if len(text) == 8 and text.isdigit():
        return text[4:]
    return None


def compact_tag_tokens(value: str, *, drop_tokens: set[str] | None = None) -> list[str]:
    normalized = sanitize_tag(value).replace("_", "-")
    tokens = [token for token in normalized.split("-") if token]
    if drop_tokens is None:
        drop_tokens = set()
    compacted: list[str] = []
    for token in tokens:
        if token in drop_tokens:
            continue
        if extract_date_token(token) is not None:
            continue
        compacted.append(token)
    return compacted


def experiment_tag_from_checkpoint_path(checkpoint_path: Path) -> str | None:
    stem = checkpoint_path.stem.lower()
    if stem not in {"exported_best", "exported_last", "best", "last"}:
        return None

    experiment_dir = checkpoint_path.parent.parent
    tokens = compact_tag_tokens(
        experiment_dir.name,
        drop_tokens={
            "my",
            "experiment",
            "exp",
            "det",
            "detect",
            "detection",
            "exported",
            "models",
            "checkpoints",
            "weights",
            "weight",
            "model",
        },
    )
    if tokens:
        return sanitize_tag("-".join(tokens[:3]))

    raw_tokens = [token for token in sanitize_tag(experiment_dir.name).replace("_", "-").split("-") if token]
    for token in reversed(raw_tokens):
        date_token = extract_date_token(token)
        if date_token is not None:
            return date_token
    return sanitize_tag(experiment_dir.name)


def checkpoint_tag_from_path(checkpoint_path: Path) -> str:
    experiment_tag = experiment_tag_from_checkpoint_path(checkpoint_path)
    if experiment_tag:
        return experiment_tag
    return sanitize_tag(checkpoint_path.stem)


def date_tag_now() -> str:
    return datetime.now().strftime("%m%d")


def date_folder_today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def timestamp_now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def build_task_infer_root(task: str, *, output_root: Path | None = None, date_tag: str | None = None) -> Path:
    root = TEST_OUTPUT_ROOT_DIR if output_root is None else output_root.expanduser().resolve()
    day = date_tag or date_tag_now()
    return root / day


def build_det_run_name(*, checkpoint_path: Path, dataset_dir: Path, split: str) -> str:
    return sanitize_tag(f"infer-{dataset_tag_from_dir(dataset_dir)}-{split}")


def build_det_report_path(output_dir: Path) -> Path:
    date_tag = date_tag_now()
    output_name = output_dir.name
    if output_name.startswith(f"{date_tag}-"):
        report_name = f"{output_name}-test_report.json"
    else:
        report_name = f"{date_tag}-{output_name}-test_report.json"
    return output_dir / report_name


def experiment_important_dir(experiment_dir: Path) -> Path:
    return experiment_dir / IMPORTANT_ARTIFACT_DIRNAME


def all_report_experiment_dir(experiment_dir: Path) -> Path:
    return ALL_REPORT_ROOT_DIR / experiment_dir.name


def infer_temp_dir(output_dir: Path) -> Path:
    return output_dir / INFER_TEMP_DIRNAME


def copy_file_if_exists(source_path: Path, destination_path: Path) -> Path | None:
    if not source_path.exists() or not source_path.is_file():
        return None
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return destination_path


def copy_tree_contents(source_dir: Path, target_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []
    copied_paths: list[Path] = []
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(source_dir)
        destination_path = target_dir / rel_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def sync_important_to_all_report(experiment_dir: Path) -> Path:
    resolved_experiment_dir = experiment_dir.expanduser().resolve()
    important_dir = experiment_important_dir(resolved_experiment_dir)
    archive_dir = all_report_experiment_dir(resolved_experiment_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    copy_tree_contents(important_dir, archive_dir)
    return archive_dir


def sync_training_summary_artifacts(experiment_dir: Path) -> tuple[Path, Path, Path | None]:
    resolved_experiment_dir = experiment_dir.expanduser().resolve()
    important_dir = experiment_important_dir(resolved_experiment_dir)
    important_dir.mkdir(parents=True, exist_ok=True)

    copy_file_if_exists(resolved_experiment_dir / "train.log", important_dir / "train.log")
    curve_paths = generate_training_curve_artifacts(resolved_experiment_dir)
    dashboard_source = next(
        (path for path in curve_paths if path.name == TRAINING_CURVE_FILENAMES["dashboard"]),
        None,
    )
    dashboard_path = None
    if dashboard_source is not None:
        dashboard_path = copy_file_if_exists(
            dashboard_source,
            important_dir / TRAINING_CURVE_FILENAMES["dashboard"],
        )

    archive_dir = sync_important_to_all_report(resolved_experiment_dir)
    return important_dir, archive_dir, dashboard_path


def build_report_archive_path(report_path: Path, archive_root: Path | None = None) -> Path:
    root = REPORT_ARCHIVE_ROOT_DIR if archive_root is None else archive_root.expanduser().resolve()
    return root / date_folder_today() / report_path.name


def deduplicate_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter:02d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def default_det_export_dir_suffix(*, image_count: int) -> str:
    safe_count = max(int(image_count), 0)
    return f"{EXPORT_DEFAULT_EXPORT_SUFFIX}_{safe_count}"


def resolve_det_export_dir_suffix(*, export_suffix: str, image_count: int) -> str:
    normalized_suffix = str(export_suffix).strip() or EXPORT_DEFAULT_EXPORT_SUFFIX
    if normalized_suffix == EXPORT_DEFAULT_EXPORT_SUFFIX:
        return default_det_export_dir_suffix(image_count=image_count)
    return normalized_suffix


def archive_report_copy(report_path: Path, archive_root: Path | None = None) -> Path:
    archive_path = deduplicate_path(build_report_archive_path(report_path, archive_root=archive_root))
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, archive_path)
    return archive_path


def experiment_dir_from_checkpoint_path(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if checkpoint_path.parent.name in {"exported_models", "checkpoints"}:
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def training_temp_dir(experiment_dir: Path) -> Path:
    return experiment_dir / TRAINING_TEMP_DIRNAME


def consolidate_training_artifacts(experiment_dir: Path) -> list[Path]:
    resolved_experiment_dir = experiment_dir.expanduser().resolve()
    if not resolved_experiment_dir.exists():
        return []

    temp_dir = training_temp_dir(resolved_experiment_dir)
    candidate_paths = [
        *resolved_experiment_dir.glob(TENSORBOARD_EVENT_GLOB),
        *(resolved_experiment_dir / filename for filename in TRAINING_CURVE_FILENAMES.values()),
    ]

    moved_paths: list[Path] = []
    for source_path in candidate_paths:
        if not source_path.exists() or not source_path.is_file():
            continue
        temp_dir.mkdir(parents=True, exist_ok=True)
        destination_path = temp_dir / source_path.name
        if destination_path.exists():
            destination_path.unlink()
        shutil.move(str(source_path), str(destination_path))
        moved_paths.append(destination_path)
    return moved_paths


def list_event_files(experiment_dir: Path) -> list[Path]:
    if not experiment_dir.exists():
        return []
    files = list(experiment_dir.glob(TENSORBOARD_EVENT_GLOB))
    temp_dir = training_temp_dir(experiment_dir)
    if temp_dir.exists():
        files.extend(temp_dir.glob(TENSORBOARD_EVENT_GLOB))
    unique_files = {path.resolve(): path.resolve() for path in files}
    files = list(unique_files.values())
    files.sort(key=lambda path: (path.stat().st_mtime, path.name), reverse=True)
    return files


def load_event_accumulator(event_file: Path):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ModuleNotFoundError:
        return None

    try:
        accumulator = EventAccumulator(str(event_file))
        accumulator.Reload()
        return accumulator
    except Exception:
        return None


def ensure_plot_dependencies() -> bool:
    global Image
    global ImageDraw
    global ImageFont

    if Image is not None and ImageDraw is not None and ImageFont is not None:
        return True
    try:
        from PIL import Image as image_module
        from PIL import ImageDraw as image_draw_module
        from PIL import ImageFont as image_font_module
    except ModuleNotFoundError:
        return False

    Image = image_module
    ImageDraw = image_draw_module
    ImageFont = image_font_module
    return True


def ensure_matplotlib_dependencies():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ModuleNotFoundError:
        return None
    except Exception:
        return None


def select_best_event_file(experiment_dir: Path) -> Path | None:
    best_file = None
    best_score = (-1, -1.0)
    for event_file in list_event_files(experiment_dir):
        accumulator = load_event_accumulator(event_file)
        if accumulator is None:
            continue
        scalar_tags = accumulator.Tags().get("scalars", [])
        preferred_tags = {"train_loss", "val_metric/map"}
        score = (
            sum(1 for tag in scalar_tags if tag in preferred_tags),
            len(scalar_tags),
        )
        if score > best_score:
            best_score = score
            best_file = event_file
    return best_file


def load_scalar_series(event_file: Path, tag: str) -> list[tuple[int, float]]:
    accumulator = load_event_accumulator(event_file)
    if accumulator is None:
        return []
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    values = accumulator.Scalars(tag)
    return [(int(item.step), float(item.value)) for item in values]


def _safe_text_size(draw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left), max(0, bottom - top)


def _temporary_image_output_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp{output_path.suffix}")


def _is_valid_image_file(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
        return False
    if not ensure_plot_dependencies():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except OSError:
        return False


def _finalize_image_output(temp_path: Path, output_path: Path) -> bool:
    if not _is_valid_image_file(temp_path):
        temp_path.unlink(missing_ok=True)
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(output_path)
    return True


def render_scalar_plot_matplotlib(
    output_path: Path,
    *,
    title: str,
    series_map: dict[str, list[tuple[int, float]]],
) -> bool:
    plt = ensure_matplotlib_dependencies()
    non_empty = {name: values for name, values in series_map.items() if values}
    if plt is None or not non_empty:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_image_output_path(output_path)

    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=120)
    fig.patch.set_facecolor("#fafbfc")
    ax.set_facecolor("#ffffff")
    palette = ["#2c7bb6", "#d73027", "#27ae60", "#8e44ad", "#e67e22"]

    for index, (name, values) in enumerate(non_empty.items()):
        steps = [step for step, _ in values]
        metrics = [value for _, value in values]
        ax.plot(
            steps,
            metrics,
            label=name.replace("_", " "),
            linewidth=2.2,
            color=palette[index % len(palette)],
        )

    ax.set_title(title, fontsize=18, pad=16)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    ax.grid(True, which="major", color="#e9edf2", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False, fontsize=10)
    try:
        fig.tight_layout()
        fig.savefig(temp_path, bbox_inches="tight")
    finally:
        plt.close(fig)
    return _finalize_image_output(temp_path, output_path)


def render_scalar_plot_pillow(
    output_path: Path,
    *,
    title: str,
    series_map: dict[str, list[tuple[int, float]]],
) -> bool:
    non_empty = {name: values for name, values in series_map.items() if values}
    if not non_empty:
        return False

    width = 1280
    height = 720
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 80
    plot_left = margin_left
    plot_top = margin_top
    plot_right = width - margin_right
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(str(Path(__file__).resolve().parent / "msyh.ttc"), 20)
        small_font = ImageFont.truetype(str(Path(__file__).resolve().parent / "msyh.ttc"), 16)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(180, 186, 194), width=2)

    all_points = [point for values in non_empty.values() for point in values]
    max_step = max(step for step, _ in all_points)
    min_step = min(step for step, _ in all_points)
    min_value = min(value for _, value in all_points)
    max_value = max(value for _, value in all_points)
    if max_step == min_step:
        max_step += 1
    if max_value == min_value:
        delta = abs(max_value) * 0.05 or 1.0
        min_value -= delta
        max_value += delta

    y_padding = (max_value - min_value) * 0.08
    min_value -= y_padding
    max_value += y_padding

    for idx in range(6):
        y_ratio = idx / 5
        y = plot_bottom - y_ratio * plot_height
        value = min_value + y_ratio * (max_value - min_value)
        draw.line((plot_left, y, plot_right, y), fill=(225, 229, 233), width=1)
        label = f"{value:.4f}"
        text_w, text_h = _safe_text_size(draw, label, small_font)
        draw.text((plot_left - text_w - 12, y - text_h / 2), label, fill=(80, 87, 96), font=small_font)

    for idx in range(6):
        x_ratio = idx / 5
        x = plot_left + x_ratio * plot_width
        step = int(round(min_step + x_ratio * (max_step - min_step)))
        draw.line((x, plot_top, x, plot_bottom), fill=(235, 238, 242), width=1)
        label = str(step)
        text_w, _ = _safe_text_size(draw, label, small_font)
        draw.text((x - text_w / 2, plot_bottom + 12), label, fill=(80, 87, 96), font=small_font)

    colors = [
        (44, 123, 182),
        (215, 48, 39),
        (39, 174, 96),
        (142, 68, 173),
        (230, 126, 34),
    ]
    legend_x = plot_left
    legend_y = 24
    for index, (name, values) in enumerate(non_empty.items()):
        color = colors[index % len(colors)]
        points: list[tuple[float, float]] = []
        for step, value in values:
            x = plot_left + ((step - min_step) / (max_step - min_step)) * plot_width
            y = plot_bottom - ((value - min_value) / (max_value - min_value)) * plot_height
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        else:
            x, y = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

        label = name.replace("_", " ")
        draw.line((legend_x, legend_y + 10, legend_x + 28, legend_y + 10), fill=color, width=4)
        draw.text((legend_x + 36, legend_y), label, fill=(35, 39, 42), font=small_font)
        legend_x += 220

    draw.text((plot_left, 18), title, fill=(24, 28, 32), font=font)
    draw.text((plot_right - 90, plot_bottom + 12), "step", fill=(80, 87, 96), font=small_font)
    draw.text((18, plot_top - 8), "value", fill=(80, 87, 96), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temporary_image_output_path(output_path)
    image.save(temp_path)
    return _finalize_image_output(temp_path, output_path)


def render_scalar_plot(
    output_path: Path,
    *,
    title: str,
    series_map: dict[str, list[tuple[int, float]]],
) -> bool:
    if render_scalar_plot_matplotlib(
        output_path,
        title=title,
        series_map=series_map,
    ):
        return True
    if not ensure_plot_dependencies():
        return False
    return render_scalar_plot_pillow(
        output_path,
        title=title,
        series_map=series_map,
    )


def compose_training_dashboard(
    output_path: Path,
    *,
    source_paths: list[Path],
    title: str = "Training Dashboard",
) -> bool:
    existing_paths = [path for path in source_paths if path.exists()]
    if not existing_paths or not ensure_plot_dependencies():
        return False

    opened_images = []
    try:
        for path in existing_paths:
            if not _is_valid_image_file(path):
                continue
            try:
                with Image.open(path) as img:
                    opened_images.append((path, img.convert("RGB")))
            except OSError:
                continue
        if not opened_images:
            return False

        card_width = max(image.width for _, image in opened_images)
        gap = 28
        padding = 36
        header_height = 78
        footer_height = 20
        total_height = (
            header_height
            + footer_height
            + padding * 2
            + sum(image.height for _, image in opened_images)
            + gap * (len(opened_images) - 1)
        )
        total_width = card_width + padding * 2

        canvas = Image.new("RGB", (total_width, total_height), (245, 247, 250))
        draw = ImageDraw.Draw(canvas)
        try:
            title_font = ImageFont.truetype(str(Path(__file__).resolve().parent / "msyh.ttc"), 30)
            small_font = ImageFont.truetype(str(Path(__file__).resolve().parent / "msyh.ttc"), 18)
        except OSError:
            title_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        draw.text((padding, 22), title, fill=(20, 24, 28), font=title_font)
        draw.text(
            (padding, 54),
            "loss, map and learning-rate curves for the selected training run",
            fill=(92, 99, 108),
            font=small_font,
        )

        cursor_y = header_height + padding
        for path, image in opened_images:
            x = padding + (card_width - image.width) // 2
            draw.rounded_rectangle(
                (
                    x - 10,
                    cursor_y - 10,
                    x + image.width + 10,
                    cursor_y + image.height + 10,
                ),
                radius=18,
                fill=(255, 255, 255),
                outline=(225, 229, 233),
            )
            canvas.paste(image, (x, cursor_y))
            label = path.name
            draw.text((padding, cursor_y - 34), label, fill=(60, 66, 74), font=small_font)
            cursor_y += image.height + gap

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _temporary_image_output_path(output_path)
        canvas.save(temp_path)
        return _finalize_image_output(temp_path, output_path)
    finally:
        for _, image in opened_images:
            image.close()


def generate_training_curve_artifacts(experiment_dir: Path) -> list[Path]:
    experiment_dir = experiment_dir.expanduser().resolve()
    consolidate_training_artifacts(experiment_dir)
    event_file = select_best_event_file(experiment_dir)
    if event_file is None or not ensure_plot_dependencies():
        return []

    temp_dir = training_temp_dir(experiment_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    loss_path = temp_dir / TRAINING_CURVE_FILENAMES["loss"]
    if render_scalar_plot(
        loss_path,
        title="Training and Validation Loss",
        series_map={
            "train_loss": load_scalar_series(event_file, "train_loss"),
            "val_loss": load_scalar_series(event_file, "val_loss"),
        },
    ):
        created_paths.append(loss_path)

    map_path = temp_dir / TRAINING_CURVE_FILENAMES["map"]
    if render_scalar_plot(
        map_path,
        title="Validation mAP Curves",
        series_map={
            "val_metric/map": load_scalar_series(event_file, "val_metric/map"),
            "val_metric/map_50": load_scalar_series(event_file, "val_metric/map_50"),
            "val_metric/map_small": load_scalar_series(event_file, "val_metric/map_small"),
            "val_metric/map_medium": load_scalar_series(event_file, "val_metric/map_medium"),
            "val_metric/map_large": load_scalar_series(event_file, "val_metric/map_large"),
        },
    ):
        created_paths.append(map_path)

    lr_path = temp_dir / TRAINING_CURVE_FILENAMES["lr"]
    if render_scalar_plot(
        lr_path,
        title="Learning Rate Curves",
        series_map={
            "learning_rate/backbone": load_scalar_series(event_file, "learning_rate/backbone"),
            "learning_rate/backbone_no_wd": load_scalar_series(event_file, "learning_rate/backbone_no_wd"),
            "learning_rate/detector": load_scalar_series(event_file, "learning_rate/detector"),
            "learning_rate/detector_no_wd": load_scalar_series(event_file, "learning_rate/detector_no_wd"),
        },
    ):
        created_paths.append(lr_path)

    dashboard_path = temp_dir / TRAINING_CURVE_FILENAMES["dashboard"]
    ordered_curve_paths = [
        temp_dir / TRAINING_CURVE_FILENAMES["loss"],
        temp_dir / TRAINING_CURVE_FILENAMES["map"],
        temp_dir / TRAINING_CURVE_FILENAMES["lr"],
    ]
    if compose_training_dashboard(
        dashboard_path,
        source_paths=ordered_curve_paths,
    ):
        created_paths.append(dashboard_path)

    return created_paths


def copy_training_curve_artifacts(*, checkpoint_path: Path, output_dir: Path) -> list[Path]:
    experiment_dir = experiment_dir_from_checkpoint_path(checkpoint_path)
    source_paths = generate_training_curve_artifacts(experiment_dir)
    copied_paths: list[Path] = []
    for source_path in source_paths:
        destination_path = output_dir / source_path.name
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied_paths.append(destination_path)
    return copied_paths


def derive_det_run_dir(
    *,
    checkpoint_path: Path,
    dataset_dir: Path,
    split: str,
    output_root: Path | None = None,
) -> Path:
    if output_root is not None:
        root = output_root.expanduser().resolve()
    else:
        root = experiment_dir_from_checkpoint_path(checkpoint_path)
    run_name = build_det_run_name(
        checkpoint_path=checkpoint_path,
        dataset_dir=dataset_dir,
        split=split,
    )
    return root / run_name
