"""检测任务共用工具。

这里放 infer / export / analysis 共用的数据结构和基础函数。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import common as rt


@dataclass(frozen=True)
class SourceImageInfo:
    split_name: str
    rel_split_image_dir: Path
    rel_split_label_dir: Path
    rel_path: Path
    src_image_path: Path
    src_label_path: Path
    label_lines: tuple[str, ...]
    class_box_counts: dict[int, int]


@dataclass(frozen=True)
class ExportImageCandidate:
    split_name: str
    rel_split_image_dir: Path
    rel_split_label_dir: Path
    rel_path: Path
    src_image_path: Path
    src_label_path: Path
    filtered_lines: tuple[str, ...]
    class_box_counts: dict[int, int]

    @property
    def total_boxes(self) -> int:
        return sum(self.class_box_counts.values())


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


def prediction_records(prediction: dict[str, Any], class_names: dict[int, str], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    labels = prediction["labels"].detach().cpu().tolist()
    boxes = prediction["bboxes"].detach().cpu().tolist()
    scores = prediction["scores"].detach().cpu().tolist()
    records = []
    for class_id, box, score in zip(labels, boxes, scores):
        clipped_box = clip_box(box, width, height)
        records.append(
            {
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), str(class_id)),
                "score": float(score),
                "bbox_xyxy": [round(float(v), 2) for v in clipped_box],
            }
        )
    return records


def gt_records(gt_boxes, gt_labels) -> list[dict[str, Any]]:
    return [{"class_id": int(class_id), "bbox": [float(v) for v in box]} for class_id, box in zip(gt_labels.tolist(), gt_boxes.tolist())]


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
    for _, is_tp in pred_pairs:
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


def compute_precision_recall(pred_pairs: list[tuple[float, int]], total_gt: int) -> tuple[int, float, float]:
    tp = sum(int(is_tp) for _, is_tp in pred_pairs)
    total_pred = len(pred_pairs)
    precision = tp / total_pred if total_pred > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    return tp, precision, recall


def update_legacy_report_state(class_data: dict[int, dict[str, Any]], gts: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float) -> tuple[int, int, int]:
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
    metric_values: dict[str, float] | None = None,
    metrics_meta: dict[str, int] | None = None,
) -> dict[str, Any]:
    ap_per_class = {class_id: compute_ap(info["pairs"], info["gt"]) for class_id, info in sorted(class_data.items())}
    map50 = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    data_root = None
    split_images = None
    split_labels = None
    if data_cfg is not None:
        data_root = str(Path(data_cfg["_root_dir"]))
        split_images_path, split_labels_path, _ = rt.resolve_dataset_split_paths(data_cfg=data_cfg, split=split)
        split_images = str(split_images_path)
        split_labels = str(split_labels_path) if split_labels_path is not None else None

    merged_summary = {
        "num_images": num_images,
        "processed_images": processed_images,
        "skipped_images": num_images - processed_images,
        "num_gt_boxes": total_gt,
        "num_preds": total_pred,
        "precision_05": precision,
        "recall_05": recall,
        "map_05": map50,
        "avg_infer_time_ms": avg_infer_time_ms,
    }
    if metric_values:
        metric_aliases = {
            "eval_metric/map": "map",
            "eval_metric/map_50": "map_50",
            "eval_metric/map_75": "map_75",
            "eval_metric/map_small": "map_small",
            "eval_metric/map_medium": "map_medium",
            "eval_metric/map_large": "map_large",
            "eval_metric/mar_small": "mar_small",
            "eval_metric/mar_medium": "mar_medium",
            "eval_metric/mar_large": "mar_large",
            "eval_metric/mar_1": "mar_1",
            "eval_metric/mar_10": "mar_10",
            "eval_metric/mar_100": "mar_100",
        }
        for raw_key, summary_key in metric_aliases.items():
            if raw_key in metric_values:
                merged_summary[summary_key] = metric_values[raw_key]

    per_class_summary = {}
    for class_id, ap in ap_per_class.items():
        pred_pairs = class_data[class_id]["pairs"]
        gt_count = class_data[class_id]["gt"]
        tp_count, class_precision, class_recall = compute_precision_recall(pred_pairs, gt_count)
        per_class_summary[str(class_id)] = {
            "name": legacy_class_name(class_names, class_id),
            "ap": ap,
            "gt": gt_count,
            "pred": len(pred_pairs),
            "tp": tp_count,
            "precision": class_precision,
            "recall": class_recall,
        }

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
        "summary": merged_summary,
        "metrics": metric_values or {},
        "metrics_meta": metrics_meta or {},
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
        "per_class_ap": per_class_summary,
        "bad_cases": [],
    }


DET_COLOR_PRED = (0, 200, 0)
DET_COLOR_GT = (220, 30, 30)


def _render_det_boxes(
    base_image: Any,
    *,
    records: list[dict[str, Any]] | None = None,
    gt_items: list[dict[str, Any]] | None = None,
    class_names: dict[int, str] | None = None,
    font: Any = None,
) -> Any:
    """在 base_image 的副本上画框并返回新图（不改动入参）。

    GT：红色虚线 + 标签贴框下沿；pred：绿色实线 + 标签贴框上沿。
    GT 与 pred 框常常高度重合，把两者标签分到上下两侧，避免互相遮挡。
    """
    image = base_image.convert("RGB")
    draw = rt.ImageDraw.Draw(image)
    if font is None:
        font = rt.load_cjk_font(15)

    width, height = image.size
    line_width = max(2, round(min(width, height) / 400))

    def _draw_dashed_rectangle(box_xyxy: list[float], color: tuple[int, int, int], dash: int = 9, gap: int = 6) -> None:
        x1, y1, x2, y2 = box_xyxy
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
            length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
            if length == 0:
                continue
            ux, uy = (bx - ax) / length, (by - ay) / length
            pos = 0.0
            while pos < length:
                seg_end = min(pos + dash, length)
                draw.line(
                    (ax + ux * pos, ay + uy * pos, ax + ux * seg_end, ay + uy * seg_end),
                    fill=color,
                    width=line_width,
                )
                pos += dash + gap

    def draw_box(box_xyxy: list[float], color: tuple[int, int, int], label: str, dashed: bool = False, label_pos: str = "top") -> None:
        x1, y1, x2, y2 = box_xyxy
        if dashed:
            _draw_dashed_rectangle(box_xyxy, color)
        else:
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_h = bottom - top
        text_w = right - left
        if label_pos == "bottom":
            rect_y0 = min(max(y2, 0), height - text_h - 4)
            rect_y1 = rect_y0 + text_h + 4
        else:
            rect_y0 = max(0, y1 - text_h - 4)
            rect_y1 = max(y1, rect_y0 + text_h + 4)
        draw.rectangle((x1, rect_y0, x1 + text_w + 4, rect_y1), fill=color)
        draw.text((x1 + 2, rect_y0 + 2), label, fill=(255, 255, 255), font=font)

    for gt in gt_items or []:
        name = (class_names or {}).get(gt["class_id"], f"class_{gt['class_id']}")
        draw_box(gt["bbox"], DET_COLOR_GT, f"[GT] {name}", dashed=True, label_pos="bottom")
    for record in records or []:
        draw_box(record["bbox_xyxy"], DET_COLOR_PRED, f"{record['class_name']} {record['score']:.3f}", label_pos="top")
    return image


def draw_predictions(image_path: Path, output_path: Path, records: list[dict[str, Any]], gt_items: list[dict[str, Any]] | None = None, class_names: dict[int, str] | None = None) -> None:
    with rt.Image.open(image_path) as image:
        rendered = _render_det_boxes(image, records=records, gt_items=gt_items, class_names=class_names)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)


def draw_comparison(image_path: Path, output_path: Path, records: list[dict[str, Any]], gt_items: list[dict[str, Any]] | None = None, class_names: dict[int, str] | None = None) -> None:
    """生成 [原图 | 真值 GT | 预测 Pred] 三联对比图，便于汇报展示。"""
    font = rt.load_cjk_font(15)
    with rt.Image.open(image_path) as image:
        base = image.convert("RGB")
        original = base.copy()
        gt_panel = _render_det_boxes(base, gt_items=gt_items, class_names=class_names, font=font)
        pred_panel = _render_det_boxes(base, records=records, class_names=class_names, font=font)
    rt.make_comparison_panel(
        [("原图", original), ("真值 GT", gt_panel), ("预测 Pred", pred_panel)],
        output_path,
    )


def xywhn_to_xyxy_tensor(boxes, image_size: tuple[int, int]):
    width, height = image_size
    if len(boxes) == 0:
        return rt.torch.zeros((0, 4), dtype=rt.torch.float32)
    x_center = rt.torch.as_tensor(boxes[:, 0], dtype=rt.torch.float32) * width
    y_center = rt.torch.as_tensor(boxes[:, 1], dtype=rt.torch.float32) * height
    box_width = rt.torch.as_tensor(boxes[:, 2], dtype=rt.torch.float32) * width
    box_height = rt.torch.as_tensor(boxes[:, 3], dtype=rt.torch.float32) * height
    x1 = x_center - (box_width / 2)
    y1 = y_center - (box_height / 2)
    x2 = x_center + (box_width / 2)
    y2 = y_center + (box_height / 2)
    return rt.torch.stack((x1, y1, x2, y2), dim=1)


def load_ground_truth(label_path: Path | None, image_size: tuple[int, int]):
    if label_path is None or not label_path.exists():
        return rt.torch.zeros((0, 4), dtype=rt.torch.float32), rt.torch.zeros((0,), dtype=rt.torch.int64), False
    boxes_np, labels_np = rt.file_helpers.open_yolo_object_detection_label_numpy(label_path)
    return xywhn_to_xyxy_tensor(boxes_np, image_size), rt.torch.as_tensor(labels_np, dtype=rt.torch.int64), True


def relative_output_path(sample: rt.ImageSample, suffix: str) -> Path:
    return sample.relative_path.with_suffix(suffix)


def read_yolo_label_lines(label_path: Path) -> tuple[tuple[str, ...], dict[int, int]]:
    if not label_path.exists():
        return (), {}
    valid_lines: list[str] = []
    class_box_counts: Counter[int] = Counter()
    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        class_box_counts[class_id] += 1
        valid_lines.append(" ".join(parts))
    return tuple(valid_lines), dict(class_box_counts)


def remap_yolo_label_lines(filtered_lines: tuple[str, ...], class_id_mapping: dict[int, int]) -> list[str]:
    remapped_lines: list[str] = []
    for line in filtered_lines:
        parts = line.split()
        old_class_id = int(float(parts[0]))
        parts[0] = str(class_id_mapping[old_class_id])
        remapped_lines.append(" ".join(parts))
    return remapped_lines


def safe_class_name(class_names: dict[int, str], class_id: int) -> str:
    return class_names.get(class_id, f"class_{class_id}")


def resolve_export_split_dirs(
    *,
    split_name: str,
    source_root: Path,
    split_image_dir: Path,
    split_label_dir: Path,
) -> tuple[Path, Path]:
    try:
        rel_image_dir = split_image_dir.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel_image_dir = Path("images") / split_name

    try:
        rel_label_dir = split_label_dir.resolve().relative_to(source_root.resolve())
    except ValueError:
        if "images" in rel_image_dir.parts:
            rel_parts = list(rel_image_dir.parts)
            rel_parts[rel_parts.index("images")] = "labels"
            rel_label_dir = Path(*rel_parts)
        else:
            rel_label_dir = Path("labels") / split_name
    return rel_image_dir, rel_label_dir


def scan_source_split(
    *,
    split_name: str,
    split_image_dir: Path,
    split_label_dir: Path,
    rel_split_image_dir: Path,
    rel_split_label_dir: Path,
) -> list[SourceImageInfo]:
    infos: list[SourceImageInfo] = []
    for rel_image in rt.file_helpers.list_image_filenames_from_dir(image_dir=split_image_dir):
        rel_path = Path(rel_image)
        src_image_path = split_image_dir / rel_path
        src_label_path = split_label_dir / rel_path.with_suffix(".txt")
        label_lines, class_box_counts = read_yolo_label_lines(src_label_path)
        infos.append(
            SourceImageInfo(
                split_name=split_name,
                rel_split_image_dir=rel_split_image_dir,
                rel_split_label_dir=rel_split_label_dir,
                rel_path=rel_path,
                src_image_path=src_image_path,
                src_label_path=src_label_path,
                label_lines=label_lines,
                class_box_counts=class_box_counts,
            )
        )
    infos.sort(key=lambda item: item.rel_path.as_posix())
    return infos


def collect_source_image_infos(source_cfg: dict[str, Any], source_root: Path) -> tuple[dict[str, list[SourceImageInfo]], dict[str, str]]:
    source_infos_by_split: dict[str, list[SourceImageInfo]] = {}
    export_split_paths: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_value = source_cfg.get(split_name)
        if not split_value:
            continue
        split_image_dir, split_label_dir, _ = rt.resolve_dataset_split_paths(source_cfg, split_name)
        if split_label_dir is None or not split_image_dir.exists():
            continue
        rel_split_image_dir, rel_split_label_dir = resolve_export_split_dirs(
            split_name=split_name,
            source_root=source_root,
            split_image_dir=split_image_dir,
            split_label_dir=split_label_dir,
        )
        source_infos_by_split[split_name] = scan_source_split(
            split_name=split_name,
            split_image_dir=split_image_dir,
            split_label_dir=split_label_dir,
            rel_split_image_dir=rel_split_image_dir,
            rel_split_label_dir=rel_split_label_dir,
        )
        export_split_paths[split_name] = rel_split_image_dir.as_posix()
    return source_infos_by_split, export_split_paths


def collect_candidate_class_summary(
    entries_by_split: Mapping[str, Sequence[SourceImageInfo | ExportImageCandidate]],
    class_ids: list[int],
) -> dict[int, dict[str, Any]]:
    summary = {
        class_id: {"images": 0, "boxes": 0, "splits": {}}
        for class_id in class_ids
    }
    for split_name, entries in entries_by_split.items():
        for entry in entries:
            for class_id, box_count in entry.class_box_counts.items():
                class_summary = summary.setdefault(class_id, {"images": 0, "boxes": 0, "splits": {}})
                split_summary = class_summary["splits"].setdefault(split_name, {"images": 0, "boxes": 0})
                class_summary["images"] += 1
                class_summary["boxes"] += box_count
                split_summary["images"] += 1
                split_summary["boxes"] += box_count
    return summary
