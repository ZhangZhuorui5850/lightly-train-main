"""训练工具。

负责通过 lightly_train API 直接执行 det / cls / seg 训练，
替代原来依赖外部脚本（train_det.py 等）的 runpy 方式。
"""

from __future__ import annotations

import json
import math
import re
import traceback
from pathlib import Path
from typing import Any, Literal

from . import common as rt

# task → lightly_train 函数名
_TRAIN_FUNC_NAME: dict[str, str] = {
    "det": "train_object_detection",
    "cls": "train_image_classification",
    "seg_instance": "train_instance_segmentation",
    "seg_semantic": "train_semantic_segmentation",
}

_WEIGHT_EXTENSIONS = {".pth", ".pt", ".ckpt", ".safetensors"}


# ---------------------------------------------------------------------------
# 发现
# ---------------------------------------------------------------------------

def detect_available_gpus() -> list[tuple[int, str]]:
    """检测当前可用 GPU，返回 [(index, name), ...]，无 GPU 时返回空列表。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
        return [(i, torch.cuda.get_device_name(i)) for i in range(count)]
    except Exception:
        return []


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read data yaml files.") from exc
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml payload in {path}: expected a mapping.")
    return cfg


def _resolve_data_root(cfg: dict[str, Any], data_yaml: Path) -> Path:
    raw = cfg.get("path")
    if raw is None:
        # 没有 path 字段时根目录就是 yaml 所在目录；不能再和 data_yaml.parent
        # 二次拼接，否则相对路径的 yaml 会得到 .../parent/.../parent 的重复路径。
        return data_yaml.parent.expanduser().resolve()
    root = Path(raw).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def _resolve_data_path(root: Path, value: Any) -> str:
    text = str(value)
    if "{image_path" in text:
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    return str(path)


def _infer_semantic_mask_path(image_value: Any) -> str:
    image_path = Path(str(image_value))
    if "images" in image_path.parts:
        parts = ["masks" if part == "images" else part for part in image_path.parts]
        return str(Path(*parts))
    return str(image_value).replace("images", "masks", 1)


def _class_mapping_from_yaml(cfg: dict[str, Any]) -> dict[int, Any]:
    classes = cfg.get("classes", cfg.get("names"))
    if classes is None:
        raise ValueError(
            "Semantic segmentation data yaml must define either 'classes' or 'names'."
        )
    if isinstance(classes, list):
        return {idx: str(name) for idx, name in enumerate(classes)}
    if isinstance(classes, dict):
        return {int(idx): value for idx, value in classes.items()}
    raise ValueError(f"Unsupported class mapping in data yaml: {classes!r}")


def _semantic_split_from_yaml(
    cfg: dict[str, Any], root: Path, split_name: str
) -> dict[str, str]:
    split_value = cfg.get(split_name)
    if split_value is None:
        raise ValueError(f"Semantic segmentation data yaml is missing '{split_name}'.")

    if isinstance(split_value, dict):
        image_value = split_value.get("images")
        mask_value = split_value.get("masks")
        if image_value is None or mask_value is None:
            raise ValueError(
                f"Semantic split '{split_name}' must contain both 'images' and 'masks'."
            )
    else:
        image_value = split_value
        mask_value = cfg.get(f"{split_name}_masks") or _infer_semantic_mask_path(
            image_value
        )

    images = _resolve_data_path(root, image_value)
    masks = _resolve_data_path(root, mask_value)

    image_dir = Path(images)
    if not image_dir.exists():
        raise FileNotFoundError(
            f"Semantic {split_name} images directory does not exist: {image_dir}"
        )
    if "{image_path" not in masks:
        mask_dir = Path(masks)
        if not mask_dir.exists():
            raise FileNotFoundError(
                f"Semantic {split_name} masks directory does not exist: {mask_dir}. "
                "Semantic segmentation requires PNG masks, not YOLO txt labels."
            )
    return {"images": images, "masks": masks}


def load_semantic_segmentation_data_config(data_yaml: Path) -> dict[str, Any]:
    cfg = _load_yaml_dict(data_yaml)
    root = _resolve_data_root(cfg, data_yaml)
    data: dict[str, Any] = {
        "train": _semantic_split_from_yaml(cfg, root, "train"),
        "val": _semantic_split_from_yaml(cfg, root, "val"),
        "classes": _class_mapping_from_yaml(cfg),
    }
    ignore_classes = cfg.get("ignore_classes")
    if ignore_classes:
        data["ignore_classes"] = ignore_classes
    return data


def load_semantic_segmentation_split_config(data_yaml: Path, split: str) -> dict[str, Any]:
    cfg = _load_yaml_dict(data_yaml)
    root = _resolve_data_root(cfg, data_yaml)
    data: dict[str, Any] = {
        split: _semantic_split_from_yaml(cfg, root, split),
        "classes": _class_mapping_from_yaml(cfg),
    }
    ignore_classes = cfg.get("ignore_classes")
    if ignore_classes:
        data["ignore_classes"] = ignore_classes
    return data


def get_seg_train_type(args: Any) -> Literal["instance", "semantic"]:
    value = str(getattr(args, "seg_train_type", "instance") or "instance").lower()
    if value not in {"instance", "semantic"}:
        raise ValueError("seg_train_type must be either 'instance' or 'semantic'.")
    return value  # type: ignore[return-value]


def get_train_func_name(args: Any) -> str:
    task = str(args.tool_task)
    if task == "seg":
        return _TRAIN_FUNC_NAME[f"seg_{get_seg_train_type(args)}"]
    try:
        return _TRAIN_FUNC_NAME[task]
    except KeyError:
        raise ValueError(f"Unsupported training task: {task}") from None


def _count_train_images(data_yaml: Path) -> int:
    """从 data.yaml 直接统计 train split 图片数，无需 import_runtime_dependencies。

    `train` 字段可以是单个路径或路径列表（YOLO 格式允许多目录），都正确处理。
    """
    data_yaml = data_yaml.expanduser().resolve()
    if data_yaml.is_dir():
        train_root = next(
            (
                data_yaml / name
                for name in ("train", "training")
                if (data_yaml / name).is_dir()
            ),
            None,
        )
        if train_root is None:
            return 0
        return sum(
            1
            for path in train_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES
        )
    try:
        import yaml  # PyYAML，比 common.lazy-load 更早可用
    except ImportError:
        return 0
    try:
        with open(data_yaml, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return 0
    if not isinstance(cfg, dict):
        return 0
    try:
        return len(
            rt.dataset_adapter.resolve_split_samples(
                data_yaml,
                cfg,
                "train",
                annotation="labels",
            )
        )
    except (OSError, TypeError, ValueError):
        pass
    root = Path(cfg.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    train_val = cfg.get("train", "images/train")
    if isinstance(train_val, dict):
        train_val = train_val.get("images", "images/train")
    train_paths = train_val if isinstance(train_val, list) else [train_val]

    total = 0
    for tv in train_paths:
        if not tv:
            continue
        image_dir = (root / str(tv)).resolve()
        if not image_dir.exists():
            continue
        total += sum(
            1 for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
    return total


def load_classification_directory_data_config(root: Path) -> dict[str, Any]:
    """Build a classification config from train/val ImageFolder directories."""
    root = root.expanduser().resolve()

    def split_dir(names: tuple[str, ...], *, required: bool) -> Path | None:
        path = next((root / name for name in names if (root / name).is_dir()), None)
        if path is None and required:
            raise ValueError(f"分类目录缺少 {'/'.join(names)} split: {root}")
        return path

    train_dir = split_dir(("train", "training"), required=True)
    val_dir = split_dir(("val", "valid", "validation"), required=True)
    test_dir = split_dir(("test", "testing"), required=False)
    assert train_dir is not None and val_dir is not None
    classes = sorted(
        {
            child.name
            for split_root in (train_dir, val_dir)
            for child in split_root.iterdir()
            if child.is_dir()
        },
        key=str.casefold,
    )
    if not classes:
        raise ValueError(f"分类目录中没有类别子目录: {root}")
    data: dict[str, Any] = {
        "train": str(train_dir),
        "val": str(val_dir),
        "classes": {index: name for index, name in enumerate(classes)},
    }
    if test_dir is not None:
        data["test"] = str(test_dir)
    return data


def estimate_steps_from_epochs(
    data_yaml: Path, epochs: int, batch_size: int
) -> tuple[int | None, int]:
    """根据数据集 train split 图片数和 batch_size 估算总 steps。

    返回 (steps, num_train_images)。steps=None 表示无法读取数据集图片数，
    调用方必须改用 step 模式输入（不能再用 epoch 兜底，否则 steps 会被严重低估）。
    """
    num_train = _count_train_images(data_yaml)
    if num_train <= 0 or batch_size <= 0:
        return None, num_train
    return math.ceil(num_train / batch_size) * epochs, num_train


def read_original_train_params(experiment_dir: Path) -> dict[str, Any]:
    """从 train.log 解析原始训练参数（model/steps/batch_size/devices/data）。"""
    log_path = experiment_dir / "train.log"
    if not log_path.exists():
        log_path = experiment_dir / "important" / "train.log"
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return {}

    marker_index = text.find("Args:")
    if marker_index < 0:
        return {}
    start = text.find("{", marker_index)
    if start < 0:
        return {}
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return {}
                result: dict[str, Any] = {}
                for key in ("model", "steps", "batch_size", "devices", "task"):
                    if key in payload:
                        result[key] = payload[key]
                data_val = payload.get("data")
                if isinstance(data_val, dict) and "path" in data_val:
                    result["data"] = data_val["path"]
                    result["data_config"] = data_val
                elif isinstance(data_val, dict):
                    result["data_config"] = data_val
                elif isinstance(data_val, str):
                    result["data"] = data_val
                return result
    return {}


def discover_weight_files() -> list[Path]:
    """递归扫描 weights/ 目录，返回所有权重文件，按修改时间倒序。"""
    weights_dir = rt.ROOT_DIR / "weights"
    if not weights_dir.exists():
        return []
    files = [
        p for p in weights_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in _WEIGHT_EXTENSIONS
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def discover_recent_models(task: str, limit: int = 30) -> list[str]:
    """从现有 train.log 中提取使用过的模型字符串，去重，按最近使用排序。"""
    seen: dict[str, float] = {}
    dirs = rt.discover_recent_experiment_dirs(task, limit=limit, require_checkpoint=False)
    for exp_dir in dirs:
        log_path = exp_dir / "train.log"
        if not log_path.exists():
            # 也检查 important/ 子目录
            log_path = exp_dir / "important" / "train.log"
        if not log_path.exists():
            continue
        try:
            text = log_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        match = re.search(r'"model"\s*:\s*"([^"]+)"', text)
        if match:
            model_str = match.group(1)
            mtime = log_path.stat().st_mtime
            if model_str not in seen or seen[model_str] < mtime:
                seen[model_str] = mtime
    return sorted(seen, key=lambda m: -seen[m])


# ---------------------------------------------------------------------------
# 命名辅助
# ---------------------------------------------------------------------------

def extract_model_short(model_str: str) -> str:
    """从模型字符串提取简短标识。

    "dinov3/vits16-ltdetr"  → "vits16"
    "torchvision/resnet50"  → "resnet50"
    "dinov2/vitb14"         → "vitb14"
    """
    name = model_str.split("/")[-1]   # "vits16-ltdetr"
    return name.split("-")[0]          # "vits16"


_DATASET_NAME_SUFFIXES = ("_dataset", "_data", "_det", "_cls", "_seg")

# 这些是「纯任务目录」名：不携带项目信息，向上找时要跳过。
_PURE_TASK_DIRS = {
    "dataset", "data", "dataset_det", "dataset_cls", "dataset_seg",
    "data_det", "data_cls", "data_seg",
}


def _strip_dataset_suffix(name: str) -> str:
    for suffix in _DATASET_NAME_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def extract_dataset_short(data_yaml: Path) -> str:
    """从 data.yaml 路径提取数据集简称。

    datasets/wuwanPic_dataset/dataset_det/data.yaml → "wuwanPic"
    datasets/NEU_dataset/dataset_det/data.yaml      → "NEU"
    datasets/my_dataset/data.yaml                   → "my"
    /tmp/random/foo/data.yaml                       → "foo"
    """
    resolved = data_yaml.expanduser().resolve()
    for ancestor in resolved.parents:
        name = ancestor.name
        if not name:
            break
        lower = name.lower()
        if lower in {"datasets"}:
            break  # 到 datasets/ 根停下
        if lower in _PURE_TASK_DIRS:
            continue  # 跳过 dataset_det 这种只标注任务的目录
        stripped = _strip_dataset_suffix(name)
        if stripped:
            return stripped
    return "dataset"


def build_default_out_dir(data_yaml: Path, model_str: str) -> Path:
    """生成默认输出目录：out/MMDD/MMDD-{dataset}-{model}/。"""
    from datetime import datetime
    mmdd = datetime.now().strftime("%m%d")
    dataset_short = extract_dataset_short(data_yaml)
    model_short = extract_model_short(model_str)
    return rt.EXPERIMENT_ROOT_DIR / mmdd / f"{mmdd}-{dataset_short}-{model_short}"


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------

def run_train(args) -> None:
    """通过 lightly_train API 执行训练，完成后同步摘要文件到 important/。"""
    rt.import_runtime_dependencies()

    task: str = args.tool_task
    func_name = get_train_func_name(args)
    if func_name is None:
        raise ValueError(f"不支持的任务类型: {task}")
    train_func = getattr(rt.lightly_train, func_name)

    out_dir = Path(args.out_dir).expanduser().resolve()

    data_config = getattr(args, "data_config", None)
    data_yaml = getattr(args, "data_yaml", None) or rt.ROOT_DIR
    data_yaml_path = Path(data_yaml).expanduser().resolve()
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {data_yaml_path}")
    data = str(data_yaml_path)
    data_display = str(data_yaml_path)
    if data_config is not None:
        data = data_config
        data_display = "<data_config>"
    elif task == "cls" and data_yaml_path.is_dir():
        data = load_classification_directory_data_config(data_yaml_path)
    elif task == "seg" and get_seg_train_type(args) == "semantic":
        data = load_semantic_segmentation_data_config(data_yaml_path)

    model_str = str(getattr(args, "model", "") or "").strip()
    if not model_str:
        raise ValueError("model 为空：请确认训练日志里能解析到 model，或显式提供。")

    backbone_weights: Path | None = getattr(args, "backbone_weights", None)
    if backbone_weights is not None:
        backbone_weights = Path(backbone_weights).expanduser().resolve()
        if not backbone_weights.exists():
            raise FileNotFoundError(f"骨干权重文件不存在: {backbone_weights}")
    model_args: dict[str, Any] | None = (
        {"backbone_weights": str(backbone_weights)} if backbone_weights is not None else None
    )

    steps = getattr(args, "steps", "auto")
    batch_size = getattr(args, "batch_size", "auto")
    num_workers = getattr(args, "num_workers", "auto")
    overwrite = getattr(args, "overwrite", False)
    resume_interrupted = getattr(args, "resume_interrupted", False)
    devices = getattr(args, "devices", "auto")
    checkpoint: Path | None = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint 文件不存在: {checkpoint}")

    if overwrite and resume_interrupted:
        raise ValueError("overwrite 与 resume_interrupted 不能同时为 True。")
    if checkpoint is not None and backbone_weights is not None:
        print("  ⚠ 同时提供 checkpoint 和 backbone_weights；通常 checkpoint 会覆盖 backbone_weights。")
    if checkpoint is not None and resume_interrupted:
        print("  ⚠ 同时设置 checkpoint 和 resume_interrupted；按 lightly_train 行为以 resume_interrupted 为准。")

    print(f"\n[{task}/train] 开始训练")
    print(f"  out          : {out_dir}")
    if task == "seg":
        print(f"  seg_type     : {get_seg_train_type(args)}")
    print(f"  data         : {data_display}")
    print(f"  model        : {args.model}")
    if backbone_weights is not None:
        print(f"  backbone     : {backbone_weights}")
    if checkpoint is not None:
        print(f"  checkpoint   : {checkpoint}")
    print(f"  steps        : {steps}")
    print(f"  batch_size   : {batch_size}")
    print(f"  num_workers  : {num_workers}")
    print(f"  devices      : {devices}")
    print(f"  overwrite    : {overwrite}")
    if resume_interrupted:
        print(f"  resume_interrupted: True")

    train_kwargs: dict[str, Any] = dict(
        out=out_dir,
        data=data,
        model=model_str,
        steps=steps,
        batch_size=batch_size,
        num_workers=num_workers,
        devices=devices,
        overwrite=overwrite,
        resume_interrupted=resume_interrupted,
        model_args=model_args,
    )
    if checkpoint is not None:
        train_kwargs["checkpoint"] = str(checkpoint)
    train_func(**train_kwargs)

    # 训练完成后同步摘要文件到 important/ 和 all_report/
    if out_dir.exists():
        try:
            important_dir, archive_dir, dashboard_path = rt.sync_training_summary_artifacts(out_dir)
            print(f"\n[{task}/train] 摘要已同步")
            print(f"  important : {important_dir}")
            print(f"  archive   : {archive_dir}")
            if dashboard_path is not None:
                print(f"  dashboard : {dashboard_path}")
        except Exception as exc:
            print(f"[{task}/train] 警告：摘要同步失败: {exc}")
            traceback.print_exc()
