from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

lightly_train = None
np = None
torch = None
yaml = None
Image = None
ImageDraw = None
ImageFont = None
file_helpers = None
yolo_helpers = None
ObjectDetectionTaskMetric = None
ObjectDetectionTaskMetricArgs = None


# ============================================================
# 默认配置区
# 只改下面这几个公共根路径，其他默认值会自动跟着变
# ============================================================

# ---------- 公共根路径 ----------
OUT_DIR = ROOT_DIR / "out"
DATASET_DIR = ROOT_DIR / "datasets" / "wuwanPic_dataset" / "dataset_det"
EXPERIMENT_DIR = OUT_DIR / "my_experiment_det_0402"

# ---------- 公共默认值 ----------
DEFAULT_CHECKPOINT = None
DEFAULT_DEVICE = "auto"
DEFAULT_OVERWRITE = False
DEFAULT_SCORE_THRESHOLD = 0.6

VISUALIZATION_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

# -------------------------
# infer：推理 + 评测 + 出图（合并命令）
# -------------------------
INFER_DEFAULT_EXPERIMENT_DIR = EXPERIMENT_DIR
INFER_DEFAULT_IMAGE = None
INFER_DEFAULT_IMAGE_DIR = DATASET_DIR / "images" / "test"
INFER_DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "infer" / "test"
INFER_DEFAULT_DATA = DATASET_DIR / "data.yaml"
INFER_DEFAULT_SPLIT = "test"

INFER_DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT
INFER_DEFAULT_SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
INFER_DEFAULT_DEVICE = DEFAULT_DEVICE
INFER_DEFAULT_OVERWRITE = DEFAULT_OVERWRITE

INFER_DEFAULT_SAVE_VISUALIZATION = True
INFER_DEFAULT_SAVE_JSON = False
INFER_DEFAULT_SAVE_TXT = False
INFER_DEFAULT_REPORT_IOU_THRESHOLD = 0.5
INFER_DEFAULT_COMPUTE_METRICS = False
INFER_DEFAULT_METRIC_CLASSWISE = False
INFER_DEFAULT_SAVE_TEST_REPORT = True
INFER_DEFAULT_REPORT_PATH = INFER_DEFAULT_OUTPUT_DIR / "test_report.json"

# eval 默认值保留供 export-good-dataset 引用
EVAL_DEFAULT_DATA = DATASET_DIR / "data.yaml"
EVAL_DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "infer" / "test"
EVAL_DEFAULT_REPORT_PATH = EVAL_DEFAULT_OUTPUT_DIR / "test_report.json"

# -------------------------
# export-good-dataset：导出筛选数据集
# -------------------------
EXPORT_DEFAULT_REPORT_JSON = EVAL_DEFAULT_REPORT_PATH
EXPORT_DEFAULT_SOURCE_DATA = EVAL_DEFAULT_DATA
EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD = 0.7
EXPORT_DEFAULT_EXPORT_SUFFIX = "_A"

@dataclass
class ImageSample:
    image_path: Path
    relative_path: Path
    label_path: Path | None = None


def import_runtime_dependencies() -> None:
    global Image
    global ImageDraw
    global ImageFont
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training object detection tools for inference, evaluation, and dataset export."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run inference, save visualizations, and optionally compute evaluation metrics.",
    )
    infer_parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=INFER_DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory used to auto-resolve best/last weights.",
    )
    infer_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=INFER_DEFAULT_CHECKPOINT,
        help="Specific checkpoint/exported model path. Overrides --experiment-dir.",
    )
    infer_input = infer_parser.add_mutually_exclusive_group(required=False)
    infer_input.add_argument(
        "--image",
        type=Path,
        default=INFER_DEFAULT_IMAGE,
        help="Run inference on one image.",
    )
    infer_input.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Run inference on all images under a directory recursively.",
    )
    infer_input.add_argument(
        "--data",
        type=Path,
        default=None,
        help="YOLO data.yaml; loads dataset split and enables evaluation metrics.",
    )
    infer_parser.add_argument(
        "--split",
        choices=("val", "test"),
        default=INFER_DEFAULT_SPLIT,
        help="Dataset split to use when --data is provided.",
    )
    infer_parser.add_argument(
        "--output-dir",
        type=Path,
        default=INFER_DEFAULT_OUTPUT_DIR,
        help="Directory to save outputs.",
    )
    infer_parser.add_argument(
        "--score-threshold",
        type=float,
        default=INFER_DEFAULT_SCORE_THRESHOLD,
        help="Score threshold used to filter predictions.",
    )
    infer_parser.add_argument(
        "--device",
        type=str,
        default=INFER_DEFAULT_DEVICE,
        help="Inference device: auto, cpu, cuda, cuda:0, mps.",
    )
    infer_viz_group = infer_parser.add_mutually_exclusive_group(required=False)
    infer_viz_group.add_argument(
        "--save-visualization",
        dest="save_visualization",
        action="store_true",
        help="Save visualized images with boxes.",
    )
    infer_viz_group.add_argument(
        "--skip-visualization",
        dest="save_visualization",
        action="store_false",
        help="Do not save visualized images.",
    )
    infer_parser.set_defaults(save_visualization=INFER_DEFAULT_SAVE_VISUALIZATION)
    infer_parser.add_argument(
        "--save-json",
        action="store_true",
        default=INFER_DEFAULT_SAVE_JSON,
        help="Save one JSON file per image with raw predictions.",
    )
    infer_parser.add_argument(
        "--save-txt",
        action="store_true",
        default=INFER_DEFAULT_SAVE_TXT,
        help="Save one TXT file per image with raw predictions.",
    )
    infer_parser.add_argument(
        "--report-iou-threshold",
        type=float,
        default=INFER_DEFAULT_REPORT_IOU_THRESHOLD,
        help="IoU threshold for test_report.json (only used with --data).",
    )
    infer_parser.add_argument(
        "--compute-metrics",
        action="store_true",
        default=INFER_DEFAULT_COMPUTE_METRICS,
        help="Compute torchmetrics-style mAP metrics (only used with --data).",
    )
    infer_parser.add_argument(
        "--metric-classwise",
        action="store_true",
        default=INFER_DEFAULT_METRIC_CLASSWISE,
        help="Also output classwise metrics (only used with --compute-metrics).",
    )
    infer_parser.add_argument(
        "--save-test-report",
        action="store_true",
        default=INFER_DEFAULT_SAVE_TEST_REPORT,
        help="Save test_report.json (only used with --data).",
    )
    infer_parser.add_argument(
        "--report-path",
        type=Path,
        default=INFER_DEFAULT_REPORT_PATH,
        help="Path for test_report.json.",
    )
    infer_parser.add_argument(
        "--overwrite",
        action="store_true",
        default=INFER_DEFAULT_OVERWRITE,
        help="Allow writing into a non-empty output directory.",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export a filtered dataset based on per-class AP in test_report.json.",
    )
    export_parser.add_argument(
        "--report-json",
        type=Path,
        default=EXPORT_DEFAULT_REPORT_JSON,
        help="Path to an existing test_report.json.",
    )
    export_parser.add_argument(
        "--export-source-data",
        type=Path,
        default=EXPORT_DEFAULT_SOURCE_DATA,
        help="Source data.yaml for dataset export.",
    )
    export_parser.add_argument(
        "--good-class-threshold",
        type=float,
        default=EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD,
        help="Keep classes whose AP is >= this threshold.",
    )
    export_parser.add_argument(
        "--export-suffix",
        type=str,
        default=EXPORT_DEFAULT_EXPORT_SUFFIX,
        help="Suffix appended to the exported dataset directory name.",
    )

    args = parser.parse_args()
    if args.command == "infer" and args.image is None and args.image_dir is None and args.data is None:
        args.data = INFER_DEFAULT_DATA
    return args


def resolve_device(device: str) -> str | torch.device | None:
    if device == "auto":
        return None
    return torch.device(device)


def resolve_checkpoint_path(
    checkpoint: Path | None,
    experiment_dir: Path | None,
) -> Path:
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

    candidates = [
        experiment_dir / "exported_models" / "exported_best.pt",
        experiment_dir / "exported_models" / "exported_last.pt",
        experiment_dir / "checkpoints" / "best.ckpt",
        experiment_dir / "checkpoints" / "last.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return path

    joined = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        "No supported checkpoint/model file found under the experiment directory. "
        f"Tried:\n{joined}"
    )


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to continue."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def load_data_config(data_path: Path) -> dict[str, Any]:
    data_path = data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data config does not exist: {data_path}")

    with data_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid data config: {data_path}")

    base_dir = data_path.parent
    root = cfg.get("path")
    if root is None:
        root_dir = base_dir
    else:
        root_dir = Path(root)
        if not root_dir.is_absolute():
            root_dir = (base_dir / root_dir).resolve()
    cfg["_data_yaml_path"] = data_path
    cfg["_root_dir"] = root_dir
    return cfg


def resolve_report_path(
    experiment_dir: Path | None,
    report_path: Path | None,
) -> Path:
    if report_path is not None:
        return report_path.expanduser().resolve()
    if experiment_dir is not None:
        return experiment_dir.expanduser().resolve() / "test_report.json"
    return ROOT_DIR / "out" / "test_report.json"


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def resolve_dataset_split_paths(
    data_cfg: dict[str, Any],
    split: str,
) -> tuple[Path, Path | None, dict[int, str]]:
    root_dir = Path(data_cfg["_root_dir"])
    names = normalize_names(data_cfg.get("names"))

    train = Path(str(data_cfg.get("train", "")))
    val = Path(str(data_cfg.get("val", "")))
    test_value = data_cfg.get("test")
    test = Path(str(test_value)) if test_value else None

    image_dir, label_dir = yolo_helpers.get_image_and_labels_dirs(
        path=root_dir,
        train=train,
        val=val,
        test=test,
        mode=split,  # type: ignore[arg-type]
    )
    if image_dir is None:
        raise ValueError(f"Split '{split}' is not defined in the data config.")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if label_dir is not None and not label_dir.exists():
        label_dir = None

    return image_dir, label_dir, names


def list_dataset_samples(
    data_cfg: dict[str, Any],
    split: str,
) -> tuple[list[ImageSample], dict[int, str]]:
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
    if not image_dir.is_dir():
        raise ValueError(f"Expected a directory: {image_dir}")

    samples: list[ImageSample] = []
    for rel_image in file_helpers.list_image_filenames_from_dir(image_dir=image_dir):
        rel_path = Path(rel_image)
        samples.append(
            ImageSample(
                image_path=image_dir / rel_path,
                relative_path=rel_path,
            )
        )
    return samples


def get_input_samples(args: argparse.Namespace) -> tuple[list[ImageSample], dict[int, str], str]:
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        return [ImageSample(image_path=image_path, relative_path=Path(image_path.name))], {}, "image"

    if args.image_dir is not None:
        return list_directory_samples(args.image_dir), {}, "dir"

    assert args.data is not None
    data_cfg = load_data_config(args.data)
    samples, class_names = list_dataset_samples(data_cfg=data_cfg, split=args.split)
    return samples, class_names, "dataset"


def get_model_class_names(model: Any) -> dict[int, str]:
    classes = getattr(model, "classes", None)
    if isinstance(classes, dict):
        return {int(class_id): str(name) for class_id, name in classes.items()}
    return {}


def merge_class_names(
    model_class_names: dict[int, str],
    data_class_names: dict[int, str],
) -> dict[int, str]:
    merged = dict(model_class_names)
    merged.update(data_class_names)
    return merged


def ensure_image_samples(samples: list[ImageSample]) -> None:
    if not samples:
        raise ValueError("No images found for inference.")


def ensure_object_detection_model(model: Any) -> None:
    class_name = model.__class__.__name__.lower()
    if "objectdetection" not in class_name:
        raise ValueError(
            f"Expected an object detection model, got '{model.__class__.__name__}'."
        )



def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    return [x1, y1, x2, y2]


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def legacy_class_name(class_names: dict[int, str], class_id: int) -> str:
    return class_names.get(class_id, f"class_{class_id}")


def compute_ap(pred_pairs: list[tuple[float, int]], total_gt: int) -> float:
    if total_gt == 0:
        return 0.0

    pred_pairs = sorted(pred_pairs, key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for score, is_tp in pred_pairs:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])

    ap = 0.0
    for idx in range(len(mrec) - 1):
        if mrec[idx + 1] != mrec[idx]:
            ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
    return ap


def gt_records(gt_boxes: torch.Tensor, gt_labels: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {
            "class_id": int(class_id),
            "bbox": [float(v) for v in box],
        }
        for class_id, box in zip(gt_labels.tolist(), gt_boxes.tolist())
    ]


def update_legacy_report_state(
    class_data: dict[int, dict[str, Any]],
    gts: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    total_gt = 0
    total_pred = 0
    total_tp = 0

    for gt in gts:
        class_id = gt["class_id"]
        class_data.setdefault(class_id, {"gt": 0, "pairs": []})
        class_data[class_id]["gt"] += 1
        total_gt += 1

    gt_by_class: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for idx, gt in enumerate(gts):
        gt_by_class.setdefault(gt["class_id"], []).append((idx, gt))

    matched_gt: set[int] = set()
    for pred in sorted(preds, key=lambda x: x["score"], reverse=True):
        total_pred += 1
        class_id = pred["class_id"]
        class_data.setdefault(class_id, {"gt": 0, "pairs": []})

        best_iou = 0.0
        best_idx = -1
        for gt_idx, gt in gt_by_class.get(class_id, []):
            if gt_idx in matched_gt:
                continue
            iou = box_iou(pred["bbox_xyxy"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = gt_idx

        is_tp = best_idx >= 0 and best_iou >= iou_threshold
        if is_tp:
            matched_gt.add(best_idx)
            total_tp += 1
        class_data[class_id]["pairs"].append((float(pred["score"]), 1 if is_tp else 0))

    return total_gt, total_pred, total_tp


def build_legacy_report(
    *,
    checkpoint_path: Path,
    data_cfg: dict[str, Any] | None,
    split: str,
    score_threshold: float,
    iou_threshold: float,
    class_names: dict[int, str],
    class_data: dict[int, dict[str, Any]],
    num_images: int,
    processed_images: int,
    total_gt: int,
    total_pred: int,
    total_tp: int,
    avg_infer_time_ms: float,
) -> dict[str, Any]:
    ap_per_class = {
        class_id: compute_ap(info["pairs"], info["gt"])
        for class_id, info in sorted(class_data.items())
    }
    map50 = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    data_root = None
    split_images = None
    split_labels = None
    if data_cfg is not None:
        data_root = str(Path(data_cfg["_root_dir"]))
        split_images_path, split_labels_path, _ = resolve_dataset_split_paths(
            data_cfg=data_cfg,
            split=split,
        )
        split_images = str(split_images_path)
        split_labels = str(split_labels_path) if split_labels_path is not None else None

    return {
        "config": {
            "model_path": str(checkpoint_path),
            "data_root": data_root,
            "split": split,
            "test_images": split_images,
            "test_labels": split_labels,
            "conf_threshold": score_threshold,
            "iou_threshold": iou_threshold,
        },
        "summary": {
            "num_images": num_images,
            "processed_images": processed_images,
            "skipped_images": num_images - processed_images,
            "num_gt_boxes": total_gt,
            "num_preds": total_pred,
            "precision_05": precision,
            "recall_05": recall,
            "map_05": map50,
            "avg_infer_time_ms": avg_infer_time_ms,
        },
        "dataset_check": {
            "missing_txt_images": 0,
            "empty_txt_images": 0,
            "bad_label_lines": 0,
            "bad_images": 0,
            "predict_errors": 0,
            "processed_images": processed_images,
            "skipped_images": num_images - processed_images,
            "negative_images": 0,
        },
        "class_names": {str(k): v for k, v in class_names.items()},
        "per_class_ap": {
            str(class_id): {
                "name": legacy_class_name(class_names, class_id),
                "ap": ap,
                "gt": class_data[class_id]["gt"],
                "pred": len(class_data[class_id]["pairs"]),
            }
            for class_id, ap in ap_per_class.items()
        },
        "bad_cases": [],
    }


def prediction_records(
    prediction: dict[str, torch.Tensor],
    class_names: dict[int, str],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    width, height = image_size
    labels = prediction["labels"].detach().cpu().tolist()
    boxes = prediction["bboxes"].detach().cpu().tolist()
    scores = prediction["scores"].detach().cpu().tolist()

    records = []
    for class_id, box, score in zip(labels, boxes, scores):
        clipped_box = clip_box(box=box, width=width, height=height)
        records.append(
            {
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "score": float(score),
                "bbox_xyxy": [round(float(v), 2) for v in clipped_box],
            }
        )
    return records


def save_prediction_json(
    json_path: Path,
    sample: ImageSample,
    image_size: tuple[int, int],
    records: list[dict[str, Any]],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image_path": str(sample.image_path),
        "width": image_size[0],
        "height": image_size[1],
        "predictions": records,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_prediction_txt(txt_path: Path, records: list[dict[str, Any]]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        x1, y1, x2, y2 = record["bbox_xyxy"]
        lines.append(
            f"{record['class_id']} {record['score']:.6f} "
            f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {record['class_name']}"
        )
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def draw_predictions(
    image_path: Path,
    output_path: Path,
    records: list[dict[str, Any]],
    gt_items: list[dict[str, Any]] | None = None,
    class_names: dict[int, str] | None = None,
) -> None:
    COLOR_PRED = (0, 200, 0)    # 绿色：推理框
    COLOR_GT   = (220, 30, 30)  # 红色：GT 标签框

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        width, height = image.size
        line_width = max(2, round(min(width, height) / 400))
        text_padding = 2

        def _draw_box(
            box_xyxy: list[float],
            color: tuple[int, int, int],
            label: str,
            dashed: bool = False,
        ) -> None:
            x1, y1, x2, y2 = box_xyxy
            if dashed:
                # 用短线段模拟虚线框
                dash = max(6, line_width * 4)
                gap  = max(4, line_width * 2)
                def _hline(y: float) -> None:
                    x = x1
                    toggle = True
                    while x < x2:
                        x_end = min(x + dash, x2)
                        if toggle:
                            draw.line([(x, y), (x_end, y)], fill=color, width=line_width)
                        x += dash + gap
                        toggle = not toggle
                def _vline(x: float) -> None:
                    y = y1
                    toggle = True
                    while y < y2:
                        y_end = min(y + dash, y2)
                        if toggle:
                            draw.line([(x, y), (x, y_end)], fill=color, width=line_width)
                        y += dash + gap
                        toggle = not toggle
                _hline(y1); _hline(y2)
                _vline(x1); _vline(x2)
            else:
                draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

            left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
            text_w = right - left
            text_h = bottom - top
            text_x1 = x1
            text_y1 = max(0, y1 - text_h - (2 * text_padding))
            text_x2 = min(width, text_x1 + text_w + (2 * text_padding))
            text_y2 = min(height, text_y1 + text_h + (2 * text_padding))
            draw.rectangle((text_x1, text_y1, text_x2, text_y2), fill=color)
            draw.text(
                (text_x1 + text_padding, text_y1 + text_padding),
                label,
                fill=(255, 255, 255),
                font=font,
            )

        # 先画 GT（红色虚线），避免被推理框遮住
        for gt in (gt_items or []):
            name = (class_names or {}).get(gt["class_id"], f"class_{gt['class_id']}")
            _draw_box(
                box_xyxy=gt["bbox"],
                color=COLOR_GT,
                label=f"[GT] {name}",
                dashed=True,
            )

        # 再画推理框（绿色实线）
        for record in records:
            _draw_box(
                box_xyxy=record["bbox_xyxy"],
                color=COLOR_PRED,
                label=f"{record['class_name']} {record['score']:.3f}",
                dashed=False,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)


def xywhn_to_xyxy_tensor(
    boxes: np.ndarray,
    image_size: tuple[int, int],
) -> torch.Tensor:
    width, height = image_size
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    x_center = torch.as_tensor(boxes[:, 0], dtype=torch.float32) * width
    y_center = torch.as_tensor(boxes[:, 1], dtype=torch.float32) * height
    box_width = torch.as_tensor(boxes[:, 2], dtype=torch.float32) * width
    box_height = torch.as_tensor(boxes[:, 3], dtype=torch.float32) * height

    x1 = x_center - (box_width / 2)
    y1 = y_center - (box_height / 2)
    x2 = x_center + (box_width / 2)
    y2 = y_center + (box_height / 2)
    return torch.stack((x1, y1, x2, y2), dim=1)


def load_ground_truth(
    label_path: Path | None,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if label_path is None or not label_path.exists():
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
            False,
        )

    boxes_np, labels_np = file_helpers.open_yolo_object_detection_label_numpy(label_path)
    return (
        xywhn_to_xyxy_tensor(boxes=boxes_np, image_size=image_size),
        torch.as_tensor(labels_np, dtype=torch.int64),
        True,
    )


def build_metric_label_mapping(class_names: dict[int, str]) -> tuple[dict[int, int], list[str]]:
    class_ids = sorted(class_names.keys())
    if not class_ids:
        return {}, []
    mapping = {class_id: idx for idx, class_id in enumerate(class_ids)}
    metric_class_names = [class_names[class_id] for class_id in class_ids]
    return mapping, metric_class_names


def remap_labels(labels: torch.Tensor, mapping: dict[int, int]) -> torch.Tensor:
    if labels.numel() == 0:
        return torch.zeros((0,), dtype=torch.int64)
    return torch.as_tensor([mapping[int(label)] for label in labels.tolist()], dtype=torch.int64)


def filter_yolo_label_lines(
    label_path: Path,
    class_id_mapping: dict[int, int],
) -> list[str]:
    if not label_path.exists():
        return []

    filtered_lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        old_class_id = int(float(parts[0]))
        if old_class_id not in class_id_mapping:
            continue
        parts[0] = str(class_id_mapping[old_class_id])
        filtered_lines.append(" ".join(parts))
    return filtered_lines


def export_filtered_dataset(
    *,
    source_data_path: Path,
    report: dict[str, Any],
    class_threshold: float,
    export_suffix: str,
) -> Path:
    source_cfg = load_data_config(source_data_path)
    source_root = Path(source_cfg["_root_dir"])
    export_root = source_root.parent / f"{source_root.name}{export_suffix}"
    export_root.mkdir(parents=True, exist_ok=True)

    per_class_ap = report.get("per_class_ap", {})
    if not isinstance(per_class_ap, dict):
        raise ValueError("Invalid report JSON: missing per_class_ap.")

    kept_class_ids = sorted(
        int(class_id)
        for class_id, info in per_class_ap.items()
        if isinstance(info, dict) and float(info.get("ap", 0.0)) >= class_threshold
    )
    if not kept_class_ids:
        raise ValueError(
            f"No classes satisfy AP >= {class_threshold:.4f}, dataset export aborted."
        )

    class_names = normalize_names(source_cfg.get("names"))
    class_id_mapping = {class_id: idx for idx, class_id in enumerate(kept_class_ids)}
    kept_names = {idx: class_names[class_id] for class_id, idx in class_id_mapping.items()}

    copied_images = 0
    copied_labels = 0
    for split_name in ("train", "val", "test"):
        split_value = source_cfg.get(split_name)
        if not split_value:
            continue

        split_image_dir, split_label_dir, _ = resolve_dataset_split_paths(
            data_cfg=source_cfg,
            split=split_name,
        )
        if split_label_dir is None or not split_image_dir.exists():
            continue

        rel_split_image_dir = Path(str(split_value))
        rel_split_label_dir = Path(
            str(split_value).replace("images", "labels", 1)
        )
        dst_image_dir = export_root / rel_split_image_dir
        dst_label_dir = export_root / rel_split_label_dir
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for rel_image in file_helpers.list_image_filenames_from_dir(image_dir=split_image_dir):
            rel_path = Path(rel_image)
            src_image_path = split_image_dir / rel_path
            src_label_path = split_label_dir / rel_path.with_suffix(".txt")
            filtered_lines = filter_yolo_label_lines(
                label_path=src_label_path,
                class_id_mapping=class_id_mapping,
            )
            if not filtered_lines:
                continue

            dst_image_path = dst_image_dir / rel_path
            dst_label_path = dst_label_dir / rel_path.with_suffix(".txt")
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            dst_label_path.parent.mkdir(parents=True, exist_ok=True)
            dst_image_path.write_bytes(src_image_path.read_bytes())
            dst_label_path.write_text("\n".join(filtered_lines) + "\n", encoding="utf-8")
            copied_images += 1
            copied_labels += 1

    export_cfg = {
        "path": str(export_root),
        "train": source_cfg.get("train"),
        "val": source_cfg.get("val"),
        "test": source_cfg.get("test"),
        "task": source_cfg.get("task", "detect"),
        "nc": len(kept_names),
        "names": kept_names,
    }
    dump_yaml(export_root / "data.yaml", export_cfg)
    (export_root / "classes.txt").write_text(
        "\n".join(kept_names[idx] for idx in range(len(kept_names))) + "\n",
        encoding="utf-8",
    )
    (export_root / "export_summary.json").write_text(
        json.dumps(
            {
                "source_data": str(source_data_path),
                "source_root": str(source_root),
                "report_threshold": class_threshold,
                "selected_class_ids": kept_class_ids,
                "selected_class_names": [class_names[class_id] for class_id in kept_class_ids],
                "copied_images": copied_images,
                "copied_labels": copied_labels,
                "export_root": str(export_root),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return export_root


def maybe_create_metric(
    compute_metrics: bool,
    class_names: dict[int, str],
    classwise: bool,
) -> tuple[ObjectDetectionTaskMetric | None, dict[int, int]]:
    if not compute_metrics:
        return None, {}
    if not class_names:
        raise ValueError("Metrics require class names. Provide --data with a names field.")

    label_mapping, metric_class_names = build_metric_label_mapping(class_names=class_names)
    metric = ObjectDetectionTaskMetric(
        task_metric_args=ObjectDetectionTaskMetricArgs(
            watch_metric="eval_metric/map",
            classwise=classwise,
            train=False,
        ),
        split="eval",
        class_names=metric_class_names,
        box_format="xyxy",
        loss_names=[],
        init_metrics=True,
    )
    return metric, label_mapping


def update_metric(
    metric: ObjectDetectionTaskMetric | None,
    label_mapping: dict[int, int],
    prediction: dict[str, torch.Tensor],
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> None:
    if metric is None:
        return

    pred_labels = prediction["labels"].detach().cpu().to(torch.int64)
    pred_boxes = prediction["bboxes"].detach().cpu().to(torch.float32)
    pred_scores = prediction["scores"].detach().cpu().to(torch.float32)

    metric.update_with_predictions(
        preds=[
            {
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": remap_labels(pred_labels, label_mapping),
            }
        ],
        target=[
            {
                "boxes": gt_boxes.to(torch.float32),
                "labels": remap_labels(gt_labels.to(torch.int64), label_mapping),
            }
        ],
    )


def relative_output_path(sample: ImageSample, suffix: str) -> Path:
    return sample.relative_path.with_suffix(suffix)


def visualization_suffix(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix in VISUALIZATION_SUFFIXES:
        return suffix
    return ".png"


def run_infer(args: argparse.Namespace) -> None:
    # --data 模式：从 data.yaml 加载数据集，同时支持评测指标
    # --image-dir / --image 模式：纯推理，不算指标
    use_dataset = args.data is not None

    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        experiment_dir=args.experiment_dir,
    )
    prepare_output_dir(output_dir=args.output_dir, overwrite=args.overwrite)

    samples, input_class_names, input_mode = get_input_samples(args)
    ensure_image_samples(samples)

    model = lightly_train.load_model(
        model=checkpoint_path,
        device=resolve_device(args.device),
    )
    model.eval()
    ensure_object_detection_model(model)

    class_names = merge_class_names(
        model_class_names=get_model_class_names(model),
        data_class_names=input_class_names,
    )

    # 评测相关初始化（仅 dataset 模式）
    data_cfg = None
    report_path = None
    metric, label_mapping = None, {}
    metrics_meta = {"images_with_labels": 0, "images_without_labels": 0}
    legacy_class_data: dict[int, dict[str, Any]] = {}
    legacy_total_gt = legacy_total_pred = legacy_total_tp = 0
    infer_time_sum_ms = 0.0

    if use_dataset:
        data_cfg = load_data_config(args.data)
        report_path = resolve_report_path(
            experiment_dir=args.experiment_dir,
            report_path=args.report_path,
        )
        metric, label_mapping = maybe_create_metric(
            compute_metrics=args.compute_metrics,
            class_names=class_names,
            classwise=args.metric_classwise,
        )

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Input mode: {input_mode}")
    print(f"Images to process: {len(samples)}")
    print(f"Output directory: {args.output_dir.resolve()}")

    for idx, sample in enumerate(samples, start=1):
        with Image.open(sample.image_path) as image:
            image_size = image.size

            # 关键：推理前先保证是 RGB
            predict_path = sample.image_path
            if image.mode != "RGB":
                tmp_dir = args.output_dir / "_tmp_rgb"
                tmp_dir.mkdir(parents=True, exist_ok=True)

                predict_path = tmp_dir / relative_output_path(sample, ".jpg")
                predict_path.parent.mkdir(parents=True, exist_ok=True)

                image.convert("RGB").save(predict_path)

        infer_start = time.perf_counter()
        prediction = model.predict(predict_path, threshold=args.score_threshold)
        infer_time_sum_ms += (time.perf_counter() - infer_start) * 1000.0

        records = prediction_records(
            prediction=prediction,
            class_names=class_names,
            image_size=image_size,
        )

        # GT：dataset 模式用 sample.label_path，--image-dir 模式推算
        if sample.label_path is not None:
            label_path_cur: Path | None = sample.label_path
        elif args.image_dir is not None:
            labels_dir = args.image_dir.expanduser().resolve().parent.parent / "labels" / args.image_dir.name
            label_path_cur = (labels_dir / sample.relative_path).with_suffix(".txt")
        else:
            label_path_cur = None

        gt_boxes, gt_labels, has_label = load_ground_truth(
            label_path=label_path_cur,
            image_size=image_size,
        )

        # 可视化
        if args.save_visualization:
            vis_path = args.output_dir / relative_output_path(
                sample=sample,
                suffix=visualization_suffix(sample.image_path),
            )
            gt_items_vis = gt_records(gt_boxes=gt_boxes, gt_labels=gt_labels) if has_label else []
            draw_predictions(
                image_path=sample.image_path,
                output_path=vis_path,
                records=records,
                gt_items=gt_items_vis,
                class_names=class_names,
            )

        if args.save_json:
            save_prediction_json(
                json_path=args.output_dir / relative_output_path(sample=sample, suffix=".json"),
                sample=sample,
                image_size=image_size,
                records=records,
            )
        if args.save_txt:
            save_prediction_txt(
                txt_path=args.output_dir / relative_output_path(sample=sample, suffix=".txt"),
                records=records,
            )

        # 评测指标（仅 dataset 模式）
        if use_dataset:
            if has_label:
                metrics_meta["images_with_labels"] += 1
            else:
                metrics_meta["images_without_labels"] += 1

            update_metric(
                metric=metric,
                label_mapping=label_mapping,
                prediction=prediction,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
            )

            if args.save_test_report:
                gt_items = gt_records(gt_boxes=gt_boxes, gt_labels=gt_labels)
                image_total_gt, image_total_pred, image_total_tp = update_legacy_report_state(
                    class_data=legacy_class_data,
                    gts=gt_items,
                    preds=records,
                    iou_threshold=args.report_iou_threshold,
                )
                legacy_total_gt += image_total_gt
                legacy_total_pred += image_total_pred
                legacy_total_tp += image_total_tp

        if idx == 1 or idx % 20 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] processed: {sample.image_path}")

    # 评测结果汇总（仅 dataset 模式）
    if use_dataset:
        if metric is not None and metrics_meta["images_with_labels"] > 0:
            metric_result = metric.compute_aggregated_values().metric_values
            metrics_payload = {
                "checkpoint": str(checkpoint_path),
                "input_mode": input_mode,
                "num_images": len(samples),
                **metrics_meta,
                "metrics": metric_result,
            }
            metrics_path = args.output_dir / "metrics_summary.json"
            metrics_path.write_text(
                json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print("Metrics:")
            for key in sorted(metric_result):
                print(f"  {key}: {metric_result[key]:.6f}")
            print(f"Metrics saved to: {metrics_path}")
        elif metric is not None:
            print("Metrics skipped because no label files were found.")

        if args.save_test_report and report_path is not None:
            report_payload = build_legacy_report(
                checkpoint_path=checkpoint_path,
                data_cfg=data_cfg,
                split=args.split,
                score_threshold=args.score_threshold,
                iou_threshold=args.report_iou_threshold,
                class_names=class_names,
                class_data=legacy_class_data,
                num_images=len(samples),
                processed_images=len(samples),
                total_gt=legacy_total_gt,
                total_pred=legacy_total_pred,
                total_tp=legacy_total_tp,
                avg_infer_time_ms=infer_time_sum_ms / max(len(samples), 1),
            )
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"test_report saved to: {report_path}")


def run_export_good_dataset(args: argparse.Namespace) -> None:
    if args.report_json is None:
        raise ValueError("The export-good-dataset subcommand requires --report-json.")
    if args.export_source_data is None:
        raise ValueError(
            "The export-good-dataset subcommand requires --export-source-data."
        )

    report_payload = json.loads(
        args.report_json.expanduser().resolve().read_text(encoding="utf-8")
    )
    export_root = export_filtered_dataset(
        source_data_path=args.export_source_data,
        report=report_payload,
        class_threshold=args.good_class_threshold,
        export_suffix=args.export_suffix,
    )
    print(f"Filtered dataset exported to: {export_root}")


def main() -> None:
    args = parse_args()
    import_runtime_dependencies()

    if args.command == "infer":
        run_infer(args)
    elif args.command == "export":
        run_export_good_dataset(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()