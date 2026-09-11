"""公共运行时与工具函数。

这个文件主要放三类内容：
- 全局默认配置和路径常量
- 运行时依赖的延迟导入，例如 lightly_train / torch / PIL
- 各任务都会复用的通用函数，例如解析 checkpoint、读取 data.yaml、列出图片等

设计上它不直接负责某个具体任务，只给 cls / det / seg 模块提供基础能力。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import dataset_adapter

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


def _prioritize_repo_source() -> None:
    """Keep the repository fork ahead of environment-installed packages."""
    source = str(SRC_DIR)
    sys.path[:] = [entry for entry in sys.path if entry != source]
    sys.path.insert(0, source)


_prioritize_repo_source()

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
DATASET_SEARCH_ROOTS: list[Path] = []
EXPERIMENT_DIR = OUT_DIR / "my_experiment_det_0402"

DEFAULT_CHECKPOINT = None
DEFAULT_DEVICE = "auto"
DEFAULT_OVERWRITE = False
DEFAULT_CLS_THRESHOLD = 0.5
DEFAULT_SEG_THRESHOLD = 0.8
DEFAULT_SCORE_THRESHOLD = 0.3
# det eval 每个 split 默认抽样输出的对比图数量；0 表示关闭数量限制。
DET_EVAL_VIS_MAX_IMAGES = 50
# seg eval 对比图默认最多出多少张（好/差各半，类别尽量全）；0 表示不限制、出全部。
SEG_EVAL_VIS_MAX_IMAGES = 100

VISUALIZATION_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

INFER_DEFAULT_EXPERIMENT_DIR = EXPERIMENT_DIR
# seg infer/eval 未显式给 --experiment-dir/--checkpoint 时的默认实验目录。
# 由 launcher.py 的 seg_experiment_dir 配置覆盖（"auto" 时自动发现最近一次含 checkpoint 的 seg 实验）。
SEG_DEFAULT_EXPERIMENT_DIR = EXPERIMENT_ROOT_DIR / "my_experiment_seg"
INFER_DEFAULT_IMAGE = None
INFER_DEFAULT_IMAGE_DIR = DATASET_DIR / "images" / "test"
INFER_DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "infer-test"
INFER_OUTPUT_DIR_CONFIGURED = False
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
INFER_DEFAULT_SAVE_TEST_REPORT = False
INFER_DEFAULT_REPORT_PATH = INFER_DEFAULT_OUTPUT_DIR / "test_report.json"

# SAHI 切片推理默认参数（仅 infer --sahi 时生效）
INFER_DEFAULT_SAHI = False
INFER_DEFAULT_SAHI_OVERLAP = 0.2
INFER_DEFAULT_SAHI_NMS_IOU = 0.3
INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU = 0.1
# True 时：短边 < 模型 tile 的小图自动跳过 SAHI、回退普通 predict（小图上 SAHI 会更差）。
INFER_DEFAULT_SAHI_SKIP_SMALL = True

EVAL_DEFAULT_DATA = DATASET_DIR / "data.yaml"
EVAL_DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "eval-test"
EVAL_DEFAULT_REPORT_PATH = EVAL_DEFAULT_OUTPUT_DIR / "test_report.json"
EVAL_OUTPUT_DIR_CONFIGURED = False
EVAL_REPORT_PATH_CONFIGURED = False

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
EXPORT_DEFAULT_SIZE_RATIO = "30:40:30"
EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = 1.0
EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN = 5.0
EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX = 15.0
EXPORT_DEFAULT_TRIM_BOXES = False
EXPORT_DEFAULT_EXPORT_SUFFIX = "_A"

SEG_DATASET_DIR = ROOT_DIR / "datasets" / "neu_dataset" / "dataset_seg"
SEMANTIC_SEG_DATASET_DIR = ROOT_DIR / "datasets" / "neu_dataset" / "dataset_semantic"
SEMANTIC_SEG_DEFAULT_DATA = SEMANTIC_SEG_DATASET_DIR / "data.yaml"
SEG_TRAIN_TYPE = "instance"
SEG_EXPORT_DEFAULT_SOURCE_DATA = SEG_DATASET_DIR / "data.yaml"
SEG_EXPORT_DEFAULT_REPORT_JSON: Path | None = None
SEG_EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD = 0.0
SEG_EXPORT_DEFAULT_AUTO_BALANCE = True
SEG_EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD = True
SEG_EXPORT_DEFAULT_BALANCE_RATIO = 0.0
SEG_EXPORT_DEFAULT_MIN_CLASS_IMAGES = 0
SEG_EXPORT_DEFAULT_MIN_CLASS_INSTANCES = 0
SEG_EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS = 0
SEG_EXPORT_DEFAULT_TARGET_TOTAL_IMAGES = 0
SEG_EXPORT_DEFAULT_SPLIT_RATIO = "8:1:1"
SEG_EXPORT_DEFAULT_TARGET_INSTANCES_PER_CLASS = 0
SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_IMAGE = 0
SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_CLASS_PER_IMAGE = 0
SEG_EXPORT_DEFAULT_INSTANCE_DENSITY_PENALTY = 0.0
SEG_EXPORT_DEFAULT_SIZE_RATIO = "30:40:30"
SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = 1.0
SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN = 5.0
SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX = 15.0
SEG_EXPORT_DEFAULT_EXPORT_SUFFIX = "_A"

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

TASK_VALUE_ALIASES = {
    "cls": "cls",
    "classify": "cls",
    "classification": "cls",
    "image_classification": "cls",
    "det": "det",
    "detect": "det",
    "detection": "det",
    "object_detection": "det",
    "seg": "seg",
    "segment": "seg",
    "semantic": "seg",
    "semantic_segmentation": "seg",
    "instance": "seg",
    "instance_segmentation": "seg",
}
EXPERIMENT_ARTIFACT_DIRNAMES = ("exported_models", "checkpoints")
EXPERIMENT_EXCLUDED_TOP_LEVEL = {"eda", "all_report", "test_reports"}

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
EXPERIMENT_INTERNAL_DIRNAMES = {
    IMPORTANT_ARTIFACT_DIRNAME,
    TRAINING_TEMP_DIRNAME,
    INFER_TEMP_DIRNAME,
}


@dataclass
class ImageSample:
    image_path: Path
    relative_path: Path
    label_path: Path | None = None


def experiment_checkpoint_candidates(experiment_dir: Path) -> list[Path]:
    from .file_index import find_files

    experiment_dir = experiment_dir.expanduser().resolve()
    preferred = [
        experiment_dir / "exported_models" / "exported_best.pt",
        experiment_dir / "exported_models" / "exported_last.pt",
        experiment_dir / "checkpoints" / "best.ckpt",
        experiment_dir / "checkpoints" / "last.ckpt",
    ]
    discovered: list[Path] = []
    for dirname in EXPERIMENT_ARTIFACT_DIRNAMES:
        artifact_dir = experiment_dir / dirname
        if not artifact_dir.is_dir():
            continue
        discovered.extend(
            find_files(
                [artifact_dir],
                label="索引 checkpoint",
                suffixes={".pt", ".ckpt", ".pth"},
                show_progress=False,
            )
        )
    preferred_resolved = [path.resolve() for path in preferred]
    extras = sorted(
        set(discovered) - set(preferred_resolved),
        key=lambda path: (-_safe_path_mtime(path), str(path).casefold()),
    )
    return [*preferred_resolved, *extras]


def _safe_path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _experiment_relative_parts(path: Path) -> tuple[str, ...]:
    try:
        return path.resolve().relative_to(EXPERIMENT_ROOT_DIR.resolve()).parts
    except (OSError, ValueError):
        return path.resolve().parts


def _is_excluded_experiment_location(path: Path) -> bool:
    parts = _experiment_relative_parts(path)
    if not parts:
        return True
    if parts[0].casefold() in EXPERIMENT_EXCLUDED_TOP_LEVEL:
        return True
    return path.name.casefold() in EXPERIMENT_INTERNAL_DIRNAMES


def _task_from_train_log(path: Path) -> str | None:
    log_paths = (path / "train.log", path / IMPORTANT_ARTIFACT_DIRNAME / "train.log")
    for log_path in log_paths:
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as stream:
                text = stream.read(262_144)
        except OSError:
            continue
        match = re.search(r'["\']task["\']\s*:\s*["\']([^"\']+)["\']', text)
        if match:
            task = TASK_VALUE_ALIASES.get(match.group(1).strip().casefold())
            if task is not None:
                return task
    return None


def _task_from_path(path: Path) -> str | None:
    tokens = {
        token
        for part in _experiment_relative_parts(path)
        for token in re.split(r"[^0-9a-z]+", part.casefold())
        if token
    }
    matches = [
        task
        for task, markers in TASK_NAME_MARKERS.items()
        if tokens.intersection(markers)
    ]
    return matches[0] if len(matches) == 1 else None


def experiment_task(path: Path) -> str | None:
    """Infer an experiment task from train metadata, then from path tokens."""
    return _task_from_train_log(path) or _task_from_path(path)


def experiment_seg_type(path: Path) -> str | None:
    """Infer the segmentation subtype recorded by an experiment."""
    path = path.expanduser().resolve()
    for log_path in (path / "train.log", path / IMPORTANT_ARTIFACT_DIRNAME / "train.log"):
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")[:262_144]
        except OSError:
            continue
        match = re.search(r'["\']task["\']\s*:\s*["\']([^"\']+)["\']', text)
        if match:
            task_name = match.group(1).strip().casefold()
            if "semantic" in task_name:
                return "semantic"
            if "instance" in task_name:
                return "instance"
    return None


def experiment_modified_time(path: Path) -> float:
    """Return the latest meaningful direct activity time for an experiment."""
    mtimes = [_safe_path_mtime(path)]
    try:
        entries = list(path.iterdir())
    except OSError:
        entries = []
    mtimes.extend(_safe_path_mtime(entry) for entry in entries)
    mtimes.extend(
        _safe_path_mtime(checkpoint)
        for checkpoint in experiment_checkpoint_candidates(path)
        if checkpoint.exists()
    )
    return max(mtimes, default=0.0)


def format_experiment_modified_time(path: Path) -> str:
    modified = experiment_modified_time(path)
    if modified <= 0:
        return "-"
    return datetime.fromtimestamp(modified).strftime("%Y-%m-%d %H:%M")


def is_task_experiment_dir(path: Path, task: str, *, require_checkpoint: bool = False) -> bool:
    if not is_experiment_dir(path, require_checkpoint=require_checkpoint):
        return False
    detected_task = experiment_task(path)
    return detected_task is None or detected_task == task


def is_experiment_dir(path: Path, *, require_checkpoint: bool = False) -> bool:
    if not path.is_dir() or _is_excluded_experiment_location(path):
        return False
    if require_checkpoint:
        return any(
            candidate.is_file()
            for candidate in experiment_checkpoint_candidates(path)
        )
    has_artifact_dir = any(
        (path / dirname).is_dir() for dirname in EXPERIMENT_ARTIFACT_DIRNAMES
    )
    has_train_record = (
        (path / "train.log").is_file()
        or (path / IMPORTANT_ARTIFACT_DIRNAME / "train.log").is_file()
        or any(path.glob(TENSORBOARD_EVENT_GLOB))
    )
    return has_artifact_dir or has_train_record


def discover_recent_experiment_dirs(
    task: str | None,
    *,
    limit: int | None = None,
    require_checkpoint: bool = False,
) -> list[Path]:
    from .file_index import walk_tree

    if not EXPERIMENT_ROOT_DIR.exists():
        return []

    candidates: list[Path] = []
    for path, dirnames, _filenames in walk_tree(
        EXPERIMENT_ROOT_DIR,
        label="索引实验目录",
        followlinks=True,
    ):
        if path.resolve() == EXPERIMENT_ROOT_DIR.resolve():
            dirnames[:] = [
                name
                for name in dirnames
                if not name.startswith(".")
                and name.casefold() not in EXPERIMENT_EXCLUDED_TOP_LEVEL
            ]
            continue
        if _is_excluded_experiment_location(path):
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith(".") and name not in EXPERIMENT_INTERNAL_DIRNAMES
        ]
        if not is_experiment_dir(path, require_checkpoint=require_checkpoint):
            continue
        detected_task = experiment_task(path) if task is not None else None
        if task is None or detected_task is None or detected_task == task:
            candidates.append(path.resolve())
        dirnames[:] = []
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(
        key=lambda path: (-experiment_modified_time(path), str(path).casefold())
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
    global DATASET_SEARCH_ROOTS
    global EXPERIMENT_DIR
    global REPORT_ARCHIVE_ROOT_DIR
    global INFER_DEFAULT_EXPERIMENT_DIR
    global SEG_DEFAULT_EXPERIMENT_DIR
    global INFER_DEFAULT_IMAGE_DIR
    global INFER_DEFAULT_OUTPUT_DIR
    global INFER_OUTPUT_DIR_CONFIGURED
    global INFER_DEFAULT_DATA
    global INFER_DEFAULT_SPLIT
    global EVAL_DEFAULT_DATA
    global EVAL_DEFAULT_OUTPUT_DIR
    global EVAL_DEFAULT_REPORT_PATH
    global EVAL_OUTPUT_DIR_CONFIGURED
    global EVAL_REPORT_PATH_CONFIGURED
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
    global EXPORT_DEFAULT_SIZE_RATIO
    global EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT
    global EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN
    global EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX
    global EXPORT_DEFAULT_TRIM_BOXES
    global EXPORT_DEFAULT_EXPORT_SUFFIX
    global SEG_DATASET_DIR
    global SEMANTIC_SEG_DATASET_DIR
    global SEMANTIC_SEG_DEFAULT_DATA
    global SEG_TRAIN_TYPE
    global SEG_EXPORT_DEFAULT_SOURCE_DATA
    global SEG_EXPORT_DEFAULT_REPORT_JSON
    global SEG_EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD
    global SEG_EXPORT_DEFAULT_AUTO_BALANCE
    global SEG_EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD
    global SEG_EXPORT_DEFAULT_BALANCE_RATIO
    global SEG_EXPORT_DEFAULT_MIN_CLASS_IMAGES
    global SEG_EXPORT_DEFAULT_MIN_CLASS_INSTANCES
    global SEG_EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS
    global SEG_EXPORT_DEFAULT_TARGET_TOTAL_IMAGES
    global SEG_EXPORT_DEFAULT_SPLIT_RATIO
    global SEG_EXPORT_DEFAULT_TARGET_INSTANCES_PER_CLASS
    global SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_IMAGE
    global SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_CLASS_PER_IMAGE
    global SEG_EXPORT_DEFAULT_INSTANCE_DENSITY_PENALTY
    global SEG_EXPORT_DEFAULT_SIZE_RATIO
    global SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT
    global SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN
    global SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX
    global SEG_EXPORT_DEFAULT_EXPORT_SUFFIX
    global TRAIN_CLS_SCRIPT
    global TRAIN_DET_SCRIPT
    global TRAIN_SEG_SCRIPT
    global TEST_CLS_SCRIPT
    global TEST_DET_SCRIPT
    global DEFAULT_CLS_THRESHOLD
    global DEFAULT_SEG_THRESHOLD
    global SEG_EVAL_VIS_MAX_IMAGES
    global DEFAULT_SCORE_THRESHOLD
    global DET_EVAL_VIS_MAX_IMAGES
    global INFER_DEFAULT_SCORE_THRESHOLD
    global INFER_DEFAULT_REPORT_IOU_THRESHOLD
    global INFER_DEFAULT_SAHI
    global INFER_DEFAULT_SAHI_OVERLAP
    global INFER_DEFAULT_SAHI_NMS_IOU
    global INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU
    global INFER_DEFAULT_SAHI_SKIP_SMALL

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
    # 自动模式先保留稳定 fallback；具体命令在真正需要实验时再执行发现，避免启动菜单和 --help 扫盘。
    seg_experiment_value = settings.get("seg_experiment_dir")
    if seg_experiment_value in {None, "", "auto"}:
        SEG_DEFAULT_EXPERIMENT_DIR = EXPERIMENT_ROOT_DIR / "my_experiment_seg"
    else:
        SEG_DEFAULT_EXPERIMENT_DIR = _path("seg_experiment_dir", EXPERIMENT_ROOT_DIR / "my_experiment_seg")
    TEST_OUTPUT_ROOT_DIR = _path(
        "infer_output_root_dir",
        _path("test_output_root_dir", TEST_OUTPUT_ROOT_DIR),
    )
    EDA_OUTPUT_ROOT_DIR = _path("eda_output_root_dir", EDA_OUTPUT_ROOT_DIR)
    ALL_REPORT_ROOT_DIR = _path("all_report_root_dir", ALL_REPORT_ROOT_DIR)
    REPORT_ARCHIVE_ROOT_DIR = _path("report_archive_root_dir", REPORT_ARCHIVE_ROOT_DIR)
    DATASET_DIR = _path("det_dataset_dir", DATASET_DIR)
    raw_search_roots = settings.get("dataset_search_roots", DATASET_SEARCH_ROOTS)
    if isinstance(raw_search_roots, (str, Path)):
        raw_search_roots = [raw_search_roots]
    elif not isinstance(raw_search_roots, (list, tuple)):
        raise TypeError("dataset_search_roots 必须是路径字符串或路径列表")
    DATASET_SEARCH_ROOTS = []
    for raw_root in raw_search_roots:
        if not isinstance(raw_root, (str, Path)):
            raise TypeError("dataset_search_roots 中的每一项都必须是路径")
        path = Path(raw_root).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        resolved = path.resolve()
        if resolved not in DATASET_SEARCH_ROOTS:
            DATASET_SEARCH_ROOTS.append(resolved)
    DEFAULT_CLS_THRESHOLD = float(settings.get("cls_threshold", DEFAULT_CLS_THRESHOLD))
    DEFAULT_SEG_THRESHOLD = float(settings.get("seg_threshold", DEFAULT_SEG_THRESHOLD))
    SEG_EVAL_VIS_MAX_IMAGES = int(settings.get("seg_eval_vis_max_images", SEG_EVAL_VIS_MAX_IMAGES))
    DEFAULT_SCORE_THRESHOLD = float(settings.get("det_score_threshold", DEFAULT_SCORE_THRESHOLD))
    DET_EVAL_VIS_MAX_IMAGES = int(
        settings.get("det_eval_vis_max_images", DET_EVAL_VIS_MAX_IMAGES)
    )
    det_experiment_value = settings.get("det_experiment_dir")
    if det_experiment_value in {None, "", "auto"}:
        EXPERIMENT_DIR = EXPERIMENT_ROOT_DIR / "my_experiment_det"
    else:
        EXPERIMENT_DIR = _path("det_experiment_dir", EXPERIMENT_DIR)
    INFER_DEFAULT_SPLIT = settings.get("det_default_split", INFER_DEFAULT_SPLIT)
    INFER_DEFAULT_SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
    INFER_DEFAULT_REPORT_IOU_THRESHOLD = float(
        settings.get("det_report_iou_threshold", INFER_DEFAULT_REPORT_IOU_THRESHOLD)
    )
    INFER_DEFAULT_SAHI = bool(settings.get("det_sahi_enabled", INFER_DEFAULT_SAHI))
    INFER_DEFAULT_SAHI_OVERLAP = float(
        settings.get("det_sahi_overlap", INFER_DEFAULT_SAHI_OVERLAP)
    )
    INFER_DEFAULT_SAHI_NMS_IOU = float(
        settings.get("det_sahi_nms_iou", INFER_DEFAULT_SAHI_NMS_IOU)
    )
    INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU = float(
        settings.get("det_sahi_global_local_iou", INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU)
    )
    INFER_DEFAULT_SAHI_SKIP_SMALL = bool(
        settings.get("det_sahi_skip_small_images", INFER_DEFAULT_SAHI_SKIP_SMALL)
    )

    default_data_yaml = DATASET_DIR / "data.yaml"
    default_image_dir = DATASET_DIR / "images" / INFER_DEFAULT_SPLIT
    default_infer_output_dir = build_task_infer_root("det") / f"manual_infer-{INFER_DEFAULT_SPLIT}"

    INFER_DEFAULT_EXPERIMENT_DIR = _path("det_infer_experiment_dir", EXPERIMENT_DIR)
    INFER_DEFAULT_IMAGE_DIR = _path("det_infer_image_dir", default_image_dir)
    INFER_DEFAULT_OUTPUT_DIR = _path("det_infer_output_dir", default_infer_output_dir)
    INFER_OUTPUT_DIR_CONFIGURED = settings.get("det_infer_output_dir") not in {None, ""}
    INFER_DEFAULT_DATA = _path("det_data_yaml", default_data_yaml)

    EVAL_DEFAULT_DATA = _path("det_eval_data_yaml", INFER_DEFAULT_DATA)
    default_eval_output_dir = build_action_output_dir(
        EXPERIMENT_DIR,
        "eval",
        input_path=EVAL_DEFAULT_DATA,
        split=INFER_DEFAULT_SPLIT,
    )
    EVAL_DEFAULT_OUTPUT_DIR = _path("det_eval_output_dir", default_eval_output_dir)
    EVAL_DEFAULT_REPORT_PATH = _path(
        "det_eval_report_path",
        build_det_report_path(EVAL_DEFAULT_OUTPUT_DIR, split=INFER_DEFAULT_SPLIT),
    )
    EVAL_OUTPUT_DIR_CONFIGURED = settings.get("det_eval_output_dir") not in {None, ""}
    EVAL_REPORT_PATH_CONFIGURED = settings.get("det_eval_report_path") not in {None, ""}

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
    EXPORT_DEFAULT_SIZE_RATIO = str(
        settings.get("det_export_size_ratio", EXPORT_DEFAULT_SIZE_RATIO)
    )
    EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = float(
        settings.get("det_export_size_balance_weight", EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT)
    )
    EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN = float(
        settings.get("det_export_avg_boxes_per_image_min", EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN)
    )
    EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX = float(
        settings.get("det_export_avg_boxes_per_image_max", EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX)
    )
    EXPORT_DEFAULT_TRIM_BOXES = bool(
        settings.get("det_export_trim_boxes", EXPORT_DEFAULT_TRIM_BOXES)
    )
    EXPORT_DEFAULT_EXPORT_SUFFIX = str(
        settings.get("det_export_suffix", EXPORT_DEFAULT_EXPORT_SUFFIX)
    )

    SEG_DATASET_DIR = _path("seg_dataset_dir", SEG_DATASET_DIR)
    SEMANTIC_SEG_DATASET_DIR = _path("semantic_seg_dataset_dir", SEMANTIC_SEG_DATASET_DIR)
    SEMANTIC_SEG_DEFAULT_DATA = _path(
        "semantic_seg_data_yaml",
        SEMANTIC_SEG_DATASET_DIR / "data.yaml",
    )
    SEG_TRAIN_TYPE = str(settings.get("seg_train_type", SEG_TRAIN_TYPE)).lower()
    default_seg_data_yaml = SEG_DATASET_DIR / "data.yaml"
    SEG_EXPORT_DEFAULT_SOURCE_DATA = _path("seg_export_source_data", default_seg_data_yaml)
    seg_report_value = settings.get("seg_export_report_json")
    if seg_report_value in {None, "", "auto"}:
        SEG_EXPORT_DEFAULT_REPORT_JSON = None
    else:
        SEG_EXPORT_DEFAULT_REPORT_JSON = _path(
            "seg_export_report_json",
            SEG_EXPORT_DEFAULT_REPORT_JSON or default_seg_data_yaml,
        )
    SEG_EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD = float(
        settings.get("seg_export_good_class_threshold", SEG_EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD)
    )
    SEG_EXPORT_DEFAULT_AUTO_BALANCE = bool(
        settings.get("seg_export_auto_balance", SEG_EXPORT_DEFAULT_AUTO_BALANCE)
    )
    SEG_EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD = bool(
        settings.get(
            "seg_export_auto_relax_class_threshold",
            SEG_EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD,
        )
    )
    SEG_EXPORT_DEFAULT_BALANCE_RATIO = float(
        settings.get("seg_export_balance_ratio", SEG_EXPORT_DEFAULT_BALANCE_RATIO)
    )
    SEG_EXPORT_DEFAULT_MIN_CLASS_IMAGES = int(
        settings.get("seg_export_min_class_images", SEG_EXPORT_DEFAULT_MIN_CLASS_IMAGES)
    )
    SEG_EXPORT_DEFAULT_MIN_CLASS_INSTANCES = int(
        settings.get("seg_export_min_class_instances", SEG_EXPORT_DEFAULT_MIN_CLASS_INSTANCES)
    )
    SEG_EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS = int(
        settings.get(
            "seg_export_target_images_per_class",
            SEG_EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS,
        )
    )
    SEG_EXPORT_DEFAULT_TARGET_TOTAL_IMAGES = int(
        settings.get(
            "seg_export_target_total_images",
            SEG_EXPORT_DEFAULT_TARGET_TOTAL_IMAGES,
        )
    )
    SEG_EXPORT_DEFAULT_SPLIT_RATIO = str(
        settings.get("seg_export_split_ratio", SEG_EXPORT_DEFAULT_SPLIT_RATIO)
    )
    SEG_EXPORT_DEFAULT_TARGET_INSTANCES_PER_CLASS = int(
        settings.get(
            "seg_export_target_instances_per_class",
            SEG_EXPORT_DEFAULT_TARGET_INSTANCES_PER_CLASS,
        )
    )
    SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_IMAGE = int(
        settings.get(
            "seg_export_max_instances_per_image",
            SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_IMAGE,
        )
    )
    SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_CLASS_PER_IMAGE = int(
        settings.get(
            "seg_export_max_instances_per_class_per_image",
            SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_CLASS_PER_IMAGE,
        )
    )
    SEG_EXPORT_DEFAULT_INSTANCE_DENSITY_PENALTY = float(
        settings.get(
            "seg_export_instance_density_penalty",
            SEG_EXPORT_DEFAULT_INSTANCE_DENSITY_PENALTY,
        )
    )
    SEG_EXPORT_DEFAULT_SIZE_RATIO = str(
        settings.get("seg_export_size_ratio", SEG_EXPORT_DEFAULT_SIZE_RATIO)
    )
    SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT = float(
        settings.get("seg_export_size_balance_weight", SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT)
    )
    SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN = float(
        settings.get("seg_export_avg_instances_per_image_min", SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN)
    )
    SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX = float(
        settings.get("seg_export_avg_instances_per_image_max", SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX)
    )
    SEG_EXPORT_DEFAULT_EXPORT_SUFFIX = str(
        settings.get("seg_export_suffix", SEG_EXPORT_DEFAULT_EXPORT_SUFFIX)
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

    _prioritize_repo_source()
    loaded_module = sys.modules.get("lightly_train")
    if loaded_module is not None:
        loaded_file = Path(str(getattr(loaded_module, "__file__", ""))).resolve()
        if not loaded_file.is_relative_to(SRC_DIR.resolve()):
            raise RuntimeError(
                "当前进程已从其他位置导入 lightly_train: "
                f"{loaded_file}。launcher 需要仓库 fork: {SRC_DIR.resolve()}。"
            )

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


def import_data_dependencies() -> None:
    """Load lightweight dependencies used by EDA, reports and dataset curation."""
    global Image
    global ImageDraw
    global ImageFont
    global np
    global yaml

    try:
        import numpy as np_module
        import yaml as yaml_module
        from PIL import Image as image_module
        from PIL import ImageDraw as image_draw_module
        from PIL import ImageFont as image_font_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing data-tool dependency. Install numpy, Pillow and PyYAML."
        ) from exc
    np = np_module
    yaml = yaml_module
    Image = image_module
    ImageDraw = image_draw_module
    ImageFont = image_font_module


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


def prepare_output_dir(output_dir: Path, overwrite: bool, *, clean: bool = False) -> None:
    """准备输出目录。

    overwrite=False 且目录非空时报错（保持原有保护）。
    overwrite=True 时：
      - clean=False（默认，向后兼容）：直接复用目录，新文件覆盖/并入旧文件。
      - clean=True：清空全部旧产物。缓存也属于某次运行，未通过完整指纹验证时
        不能跨运行保留，否则旧预测会混入新报告。
    """
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {output_dir}. Use overwrite to continue.")
    if clean and overwrite and output_dir.exists():
        for child in output_dir.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
            except OSError as exc:
                raise RuntimeError(f"无法清理旧输出: {child}: {exc}") from exc
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
    cfg["_data_yaml_path"] = data_path
    cfg["_base_dir"] = base_dir
    cfg["_root_dir"] = dataset_adapter.resolve_dataset_root(data_path, cfg)
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

    def has_split_source(root: Path) -> bool:
        for split_name, raw_value in (split_values or {}).items():
            if raw_value is None or raw_value == "":
                continue
            if isinstance(raw_value, dict):
                raw_value = raw_value.get("images", raw_value.get("image"))
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in values:
                if value is None or value == "":
                    continue
                candidate = Path(str(value)).expanduser()
                if not candidate.is_absolute():
                    candidate = root / candidate
                if candidate.exists() or (candidate / "images").is_dir():
                    return True
            if (root / "images" / split_name).is_dir():
                return True
            if (root / split_name / "images").is_dir():
                return True
        return False

    if has_split_source(resolved_root_dir):
        return resolved_root_dir
    if has_split_source(resolved_data_yaml_dir):
        return resolved_data_yaml_dir
    return resolved_root_dir


def resolve_dataset_split_paths(data_cfg: dict[str, Any], split: str) -> tuple[Path, Path | None, dict[int, str]]:
    names = normalize_names(data_cfg.get("names"))
    config_path = Path(data_cfg["_data_yaml_path"])
    image_dir, label_dir = dataset_adapter.resolve_split_paths(
        config_path,
        data_cfg,
        split,
        annotation="labels",
    )
    if image_dir is None:
        diagnostics = "\n  ".join(
            dataset_adapter.path_diagnostics(config_path, data_cfg)
        )
        raise ValueError(
            f"Split '{split}' 无法解析。\n  {diagnostics}"
        )
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
    names = normalize_names(data_cfg.get("names"))
    config_path = Path(data_cfg["_data_yaml_path"])
    resolved = dataset_adapter.resolve_split_samples(
        config_path,
        data_cfg,
        split,
        annotation="labels",
    )
    if not resolved:
        diagnostics = "\n  ".join(
            dataset_adapter.path_diagnostics(config_path, data_cfg)
        )
        raise ValueError(f"Split '{split}' 没有可读取的图片。\n  {diagnostics}")
    return (
        [
            ImageSample(
                image_path=image_path,
                relative_path=relative_path,
                label_path=label_path if label_path.exists() else None,
            )
            for image_path, label_path, relative_path in resolved
        ],
        names,
    )


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
    """按 fieldnames 导出指定列，保留原始记录中的内部辅助字段。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
    label_ids = [int(label) for label in labels.tolist()]
    missing = sorted(set(label_ids) - set(mapping))
    if missing:
        known = sorted(mapping)
        raise ValueError(
            f"标签 id {missing} 不在类别映射中（当前 data.yaml 的类别 id 为 {known}）。"
            "通常是评测数据集与 checkpoint 训练时的类别集合不一致："
            "请确认 --data 指向的数据集与该模型训练时使用的数据集相同，"
            "并检查标签 txt 中是否存在超出 nc 范围的类别 id。"
        )
    return torch.as_tensor([mapping[label] for label in label_ids], dtype=torch.int64)


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


def path_identity_tag(path: Path, *, prefix: str | None = None) -> str:
    """Build a readable, collision-resistant tag for an input path."""
    resolved = path.expanduser().resolve()
    identity_dir = resolved.parent if resolved.is_file() or resolved.suffix else resolved
    readable = dataset_tag_from_dir(identity_dir)
    digest = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:8]
    parts = [prefix, readable, digest] if prefix else [readable, digest]
    return sanitize_tag("-".join(str(part) for part in parts if part))


def build_action_output_dir(
    experiment_dir: Path,
    action: str,
    *,
    input_path: Path | None = None,
    split: str | list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Build a default action directory carrying input identity and split identity."""
    output = experiment_dir.expanduser().resolve() / sanitize_tag(action)
    if input_path is not None:
        output /= path_identity_tag(input_path)
    else:
        output /= "manual"
    if split is not None:
        split_items = [split] if isinstance(split, str) else list(split)
        output /= sanitize_tag("-".join(str(item) for item in split_items))
    return output


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


def build_det_report_path(output_dir: Path, *, split: str = "test") -> Path:
    date_tag = date_tag_now()
    output_name = output_dir.name
    split_tag = sanitize_tag(split)
    if output_name.casefold() == split_tag.casefold():
        report_name = f"{date_tag}-{split_tag}_report.json"
    elif output_name.startswith(f"{date_tag}-"):
        report_name = f"{output_name}-{split_tag}_report.json"
    else:
        report_name = f"{date_tag}-{output_name}-{split_tag}_report.json"
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


def copy_tree_contents(
    source_dir: Path,
    target_dir: Path,
    *,
    progress_label: str | None = None,
) -> list[Path]:
    if not source_dir.exists():
        return []
    source_paths = [path for path in sorted(source_dir.rglob("*")) if path.is_file()]
    if not source_paths:
        return []
    if progress_label:
        from .progress import track

        paths = track(
            source_paths,
            label=progress_label,
            total=len(source_paths),
            unit="file",
        )
    else:
        paths = iter(source_paths)
    copied_paths: list[Path] = []
    for path in paths:
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


def default_seg_export_dir_suffix(*, image_count: int) -> str:
    safe_count = max(int(image_count), 0)
    return f"{SEG_EXPORT_DEFAULT_EXPORT_SUFFIX}_{safe_count}"


def resolve_seg_export_dir_suffix(*, export_suffix: str, image_count: int) -> str:
    normalized_suffix = str(export_suffix).strip() or SEG_EXPORT_DEFAULT_EXPORT_SUFFIX
    if normalized_suffix == SEG_EXPORT_DEFAULT_EXPORT_SUFFIX:
        return default_seg_export_dir_suffix(image_count=image_count)
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


def load_merged_scalar_series(
    experiment_dir: Path,
    tags: tuple[str, ...],
) -> dict[str, list[tuple[int, float]]]:
    """Merge scalar histories from every training session by tag and global step."""
    merged: dict[str, dict[int, tuple[float, float]]] = {tag: {} for tag in tags}
    for event_file in list_event_files(experiment_dir):
        accumulator = load_event_accumulator(event_file)
        if accumulator is None:
            continue
        scalar_tags = set(accumulator.Tags().get("scalars", []))
        for tag in tags:
            if tag not in scalar_tags:
                continue
            for item in accumulator.Scalars(tag):
                step = int(item.step)
                wall_time = float(item.wall_time)
                current = merged[tag].get(step)
                if current is None or wall_time > current[0]:
                    merged[tag][step] = (wall_time, float(item.value))

    return {
        tag: [(step, value) for step, (_wall_time, value) in sorted(points.items())]
        for tag, points in merged.items()
    }


def _safe_text_size(draw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return max(0, right - left), max(0, bottom - top)


def load_cjk_font(size: int = 15):
    """加载随仓库自带的中文字体 msyh.ttc；找不到时退回 PIL 默认字体。"""
    ensure_plot_dependencies()
    font_path = str(Path(__file__).resolve().parent / "msyh.ttc")
    try:
        return ImageFont.truetype(font_path, size)
    except (OSError, IOError):
        return ImageFont.load_default()


def make_comparison_panel(
    panels: list[tuple[str, Any]],
    output_path: Path,
    *,
    gap: int = 8,
    title_height: int = 30,
    background: tuple[int, int, int] = (245, 245, 245),
) -> None:
    """把多张图横向拼成一张对比图，每张上方带标题条。

    panels: [(标题, PIL.Image), ...]，按给定顺序从左到右排列。
    各图会等比缩放到统一高度后拼接，便于汇报展示。
    """
    if not ensure_plot_dependencies() or not panels:
        return
    images: list[tuple[str, Any]] = [(title, img.convert("RGB")) for title, img in panels]
    max_h = max(img.height for _, img in images)
    normalized: list[tuple[str, Any]] = []
    for title, img in images:
        if img.height != max_h:
            new_w = max(1, round(img.width * max_h / img.height))
            img = img.resize((new_w, max_h))
        normalized.append((title, img))

    total_w = sum(img.width for _, img in normalized) + gap * (len(normalized) + 1)
    total_h = max_h + title_height + gap
    canvas = Image.new("RGB", (total_w, total_h), background)
    draw = ImageDraw.Draw(canvas)
    font = load_cjk_font(18)

    x = gap
    for title, img in normalized:
        text_w, text_h = _safe_text_size(draw, title, font)
        text_x = x + max(0, (img.width - text_w) // 2)
        text_y = max(0, (title_height - text_h) // 2)
        draw.text((text_x, text_y), title, fill=(30, 30, 30), font=font)
        canvas.paste(img, (x, title_height))
        x += img.width + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


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
    fig.set_facecolor("#fafbfc")
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
        fig.savefig(str(temp_path), bbox_inches="tight")
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
    if not list_event_files(experiment_dir) or not ensure_plot_dependencies():
        return []

    scalar_tags = (
        "train_loss",
        "val_loss",
        "val_metric/map",
        "val_metric/map_50",
        "val_metric/map_small",
        "val_metric/map_medium",
        "val_metric/map_large",
        "learning_rate/backbone",
        "learning_rate/backbone_no_wd",
        "learning_rate/detector",
        "learning_rate/detector_no_wd",
    )
    scalar_series = load_merged_scalar_series(experiment_dir, scalar_tags)

    temp_dir = training_temp_dir(experiment_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    created_paths: list[Path] = []
    loss_path = temp_dir / TRAINING_CURVE_FILENAMES["loss"]
    if render_scalar_plot(
        loss_path,
        title="Training and Validation Loss",
        series_map={
            "train_loss": scalar_series["train_loss"],
            "val_loss": scalar_series["val_loss"],
        },
    ):
        created_paths.append(loss_path)

    map_path = temp_dir / TRAINING_CURVE_FILENAMES["map"]
    if render_scalar_plot(
        map_path,
        title="Validation mAP Curves",
        series_map={
            "val_metric/map": scalar_series["val_metric/map"],
            "val_metric/map_50": scalar_series["val_metric/map_50"],
            "val_metric/map_small": scalar_series["val_metric/map_small"],
            "val_metric/map_medium": scalar_series["val_metric/map_medium"],
            "val_metric/map_large": scalar_series["val_metric/map_large"],
        },
    ):
        created_paths.append(map_path)

    lr_path = temp_dir / TRAINING_CURVE_FILENAMES["lr"]
    if render_scalar_plot(
        lr_path,
        title="Learning Rate Curves",
        series_map={
            "learning_rate/backbone": scalar_series["learning_rate/backbone"],
            "learning_rate/backbone_no_wd": scalar_series["learning_rate/backbone_no_wd"],
            "learning_rate/detector": scalar_series["learning_rate/detector"],
            "learning_rate/detector_no_wd": scalar_series["learning_rate/detector_no_wd"],
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
