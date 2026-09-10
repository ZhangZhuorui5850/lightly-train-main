"""检测任务推理与评估入口。

infer 与 eval 共用模型前向和多卡聚合逻辑，通过 action profile 控制落盘产物。
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import common as rt
from . import gpu_parallel
from . import run_reuse
from .artifact_transaction import create_stage, publish_stage
from .progress import track
from .det_shared import (
    build_legacy_report,
    draw_comparison,
    draw_predictions,
    gt_records,
    load_ground_truth,
    prediction_records,
    read_yolo_label_lines,
    relative_output_path,
    update_legacy_report_state,
)
from .det_report import generate_report_for_infer_output
from .gpu_parallel import (
    GPU_HIGH_MEMORY_RATIO_THRESHOLD,
    filter_high_memory_gpus,
    format_gpu_summary,
    query_gpu_inventory,
    select_fallback_single_gpu,
)


@dataclass
class SplitSample:
    split: str
    sample: rt.ImageSample


def is_shard_child(args) -> bool:
    shard_index = getattr(args, "shard_index", None)
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    return shard_index is not None and num_shards > 1


def det_action(args) -> str:
    action = str(getattr(args, "tool_action", getattr(args, "command", "infer")))
    return "eval" if action == "eval" else "infer"


def det_label(args) -> str:
    return f"det/{det_action(args)}"


def visualization_indices(count: int, args) -> set[int]:
    """返回当前进程需要渲染的局部样本下标。

    infer 渲染全部样本。eval 按 vis_max_images 为每个 split 分配全局配额；
    多卡时把配额稳定分摊给各 shard，最终总量最多为配置值。
    """
    if count <= 0 or not bool(getattr(args, "save_visualization", False)):
        return set()
    if det_action(args) == "infer":
        return set(range(count))
    limit = int(getattr(args, "vis_max_images", rt.DET_EVAL_VIS_MAX_IMAGES) or 0)
    if limit <= 0:
        return set(range(count))
    quota = limit
    if is_shard_child(args):
        num_shards = int(args.num_shards)
        shard_index = int(args.shard_index)
        quota = limit // num_shards + (1 if shard_index < limit % num_shards else 0)
    quota = min(quota, count)
    if quota <= 0:
        return set()
    if quota == 1:
        return {count // 2}
    return {
        round(index * (count - 1) / (quota - 1))
        for index in range(quota)
    }


def _sample_weight(item: Any) -> float:
    sample = item.sample if isinstance(item, SplitSample) else item
    try:
        return float(sample.image_path.stat().st_size)
    except OSError:
        return 1.0


def filter_samples_for_shard(samples: list[Any], args) -> list[Any]:
    if not is_shard_child(args):
        return samples
    indices = gpu_parallel.filter_indices_for_shard(
        len(samples), shard_index=int(args.shard_index), num_shards=int(args.num_shards),
        weights=[_sample_weight(sample) for sample in samples],
    )
    return [samples[i] for i in indices]


def get_input_samples(args) -> tuple[list[rt.ImageSample], dict[int, str], str]:
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        samples = [rt.ImageSample(image_path=image_path, relative_path=Path(image_path.name))]
        return filter_samples_for_shard(samples, args), {}, "image"
    if args.image_dir is not None:
        samples = rt.list_directory_samples(args.image_dir)
        return filter_samples_for_shard(samples, args), {}, "dir"
    data_cfg = rt.load_data_config(args.data)
    samples, class_names = rt.list_dataset_samples(data_cfg=data_cfg, split=args.split)
    return filter_samples_for_shard(samples, args), class_names, "dataset"


def get_multi_split_input_samples(
    args,
    splits: list[str],
) -> tuple[list[SplitSample], dict[int, str], dict[str, Any]]:
    data_cfg = rt.load_data_config(args.data)
    class_names = rt.normalize_names(data_cfg.get("names"))
    split_samples: list[SplitSample] = []
    for split in splits:
        samples, _ = rt.list_dataset_samples(data_cfg=data_cfg, split=split)
        split_samples.extend(SplitSample(split=split, sample=sample) for sample in samples)
    return filter_samples_for_shard(split_samples, args), class_names, data_cfg


def important_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    return rt.experiment_important_dir(experiment_dir)


def build_important_dashboard_path(checkpoint_path: Path) -> Path:
    return important_dir_from_checkpoint(checkpoint_path) / rt.TRAINING_CURVE_FILENAMES["dashboard"]


def sync_important_artifacts(checkpoint_path: Path) -> Path | None:
    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    _, _, dashboard_path = rt.sync_training_summary_artifacts(experiment_dir)
    return dashboard_path


def copy_infer_artifacts_to_important(checkpoint_path: Path, artifact_paths: list[Path]) -> list[Path]:
    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    important_dir = rt.experiment_important_dir(experiment_dir)
    copied_paths: list[Path] = []
    for artifact_path in artifact_paths:
        copied_path = rt.copy_file_if_exists(artifact_path, important_dir / artifact_path.name)
        if copied_path is not None:
            copied_paths.append(copied_path)
    if copied_paths:
        rt.sync_important_to_all_report(experiment_dir)
    return copied_paths


def print_published_eval_artifacts(output_dir: Path, args) -> None:
    if det_action(args) != "eval":
        return
    output_dir = Path(output_dir).expanduser().resolve()
    artifacts = sorted(output_dir.rglob("metrics_summary.json"))
    artifacts.extend(sorted(output_dir.rglob("*_report.json")))
    artifacts.extend(sorted(output_dir.rglob("single_report_*.md")))
    print(f"[det/eval] 最终输出目录: {output_dir}")
    for artifact in artifacts:
        print(f"[det/eval] 最终报告产物: {artifact}")


def ensure_object_detection_model(model: Any) -> None:
    class_name = model.__class__.__name__.lower()
    if "objectdetection" not in class_name:
        raise ValueError(f"Expected an object detection model, got '{model.__class__.__name__}'.")


def _prepare_rgb_predict_path(image: Any, sample: Any, scratch_dir: Path) -> Path:
    """非 RGB 图转 RGB 并写入 scratch_dir；返回预测使用的路径。

    保留原扩展名，避免静默把 PNG/TIFF/BMP 有损压成 JPEG。
    PIL 无法以原扩展名保存 RGB 时回退到 .png（无损）。scratch_dir 由调用方
    用 tempfile.TemporaryDirectory 管理，函数返回后图片随上下文清理。
    """
    orig_ext = sample.image_path.suffix or ".png"
    predict_path = scratch_dir / relative_output_path(sample, orig_ext)
    predict_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_image = image.convert("RGB")
    try:
        rgb_image.save(predict_path)
    except (OSError, ValueError, KeyError):
        predict_path = predict_path.with_suffix(".png")
        rgb_image.save(predict_path)
    return predict_path


def save_prediction_json(json_path: Path, sample: rt.ImageSample, image_size: tuple[int, int], records: list[dict[str, Any]]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"image_path": str(sample.image_path), "width": image_size[0], "height": image_size[1], "predictions": records}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def save_prediction_txt(txt_path: Path, records: list[dict[str, Any]]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        x1, y1, x2, y2 = record["bbox_xyxy"]
        lines.append(
            f"{record['class_id']} {record['score']:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {record['class_name']}"
        )
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def sanitize_bad_image_class_name(value: str) -> str:
    cleaned = []
    for ch in str(value).strip():
        if ch.isalnum() or ch in {"-", "_"}:
            cleaned.append(ch)
        else:
            cleaned.append("-")
    text = "".join(cleaned).strip("-_")
    while "--" in text:
        text = text.replace("--", "-")
    return text or "class"


def primary_prediction_class_name(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no_pred"
    best_record = max(records, key=lambda item: float(item.get("score", 0.0) or 0.0))
    return sanitize_bad_image_class_name(str(best_record.get("class_name") or best_record.get("class_id") or "unknown"))


def visualization_output_path(
    images_dir: Path,
    sample: rt.ImageSample,
    suffix: str,
    records: list[dict[str, Any]],
) -> Path:
    rel_path = relative_output_path(sample, suffix)
    class_name = primary_prediction_class_name(records)
    return images_dir / rel_path.with_name(f"{class_name}_{rel_path.name}")


def comparison_output_path(output_dir: Path, sample: rt.ImageSample, suffix: str) -> Path:
    """[原图|GT|预测] 三联对比图统一落到 <output_dir>/compare/ 下，文件名 <图名>_compare.png。"""
    rel_path = relative_output_path(sample, suffix)
    return output_dir / "compare" / rel_path.with_name(f"{rel_path.stem}_compare.png")


def find_saved_visualization_path(output_dir: Path, sample: rt.ImageSample) -> Path | None:
    rel_path = relative_output_path(sample, rt.visualization_suffix(sample.image_path))
    image_dir = output_dir / "images" / rel_path.parent
    old_path = image_dir / rel_path.name
    if old_path.exists():
        return old_path
    candidates = sorted(image_dir.glob(f"*_{rel_path.name}")) if image_dir.exists() else []
    return candidates[0] if candidates else None


def select_bad_classes_from_report(
    report_payload: dict[str, Any],
    *,
    threshold: float,
) -> dict[int, dict[str, Any]]:
    per_class_ap = report_payload.get("per_class_ap", {})
    if not isinstance(per_class_ap, dict):
        return {}

    bad_classes: dict[int, dict[str, Any]] = {}
    for class_id_raw, info in per_class_ap.items():
        if not isinstance(info, dict):
            continue
        try:
            class_id = int(class_id_raw)
            ap = float(info.get("ap", 0.0))
            gt_count = int(info.get("gt", 0))
        except (TypeError, ValueError):
            continue
        if gt_count > 0 and ap < threshold:
            bad_classes[class_id] = {
                "class_id": class_id,
                "class_name": str(info.get("name") or f"class_{class_id}"),
                "ap": ap,
                "gt": gt_count,
                "pred": int(info.get("pred", 0) or 0),
                "tp": int(info.get("tp", 0) or 0),
            }
    return bad_classes


def export_bad_class_images(
    *,
    args,
    output_dir: Path,
    report_path: Path,
    data_cfg: dict[str, Any] | None,
    split: str,
    samples: list[rt.ImageSample] | None = None,
) -> dict[str, Any]:
    threshold = float(getattr(args, "bad_class_map50_threshold", rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD))
    bad_images_dir = output_dir / "bad_images"
    manifest_csv_path = bad_images_dir / "manifest.csv"
    manifest_json_path = bad_images_dir / "manifest.json"
    bad_images_dir.mkdir(parents=True, exist_ok=True)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    bad_classes = select_bad_classes_from_report(report_payload, threshold=threshold)

    if samples is None and data_cfg is not None:
        samples, _ = rt.list_dataset_samples(data_cfg=data_cfg, split=split)
    samples = samples or []

    exported_rows: list[dict[str, Any]] = []
    skipped_visualizations = 0
    counters: dict[int, int] = defaultdict(int)
    for sample in samples:
        if sample.label_path is None:
            continue
        _, class_box_counts = read_yolo_label_lines(sample.label_path)
        matching_class_ids = [class_id for class_id in sorted(class_box_counts) if class_id in bad_classes]
        if not matching_class_ids:
            continue

        source_visualization_path = find_saved_visualization_path(output_dir, sample)
        if source_visualization_path is None:
            skipped_visualizations += len(matching_class_ids)
            continue

        for class_id in matching_class_ids:
            info = bad_classes[class_id]
            counters[class_id] += 1
            class_name = str(info["class_name"])
            safe_class_name = sanitize_bad_image_class_name(class_name)
            suffix = source_visualization_path.suffix or rt.visualization_suffix(sample.image_path)
            export_image_path = bad_images_dir / f"{safe_class_name}_{counters[class_id]}{suffix}"
            while export_image_path.exists():
                counters[class_id] += 1
                export_image_path = bad_images_dir / f"{safe_class_name}_{counters[class_id]}{suffix}"
            shutil.copy2(source_visualization_path, export_image_path)
            exported_rows.append(
                {
                    "split": split,
                    "image": sample.relative_path.as_posix(),
                    "class_id": class_id,
                    "class_name": class_name,
                    "ap": info["ap"],
                    "gt": info["gt"],
                    "pred": info["pred"],
                    "tp": info["tp"],
                    "source_image_path": str(sample.image_path),
                    "source_label_path": str(sample.label_path),
                    "source_visualization_path": str(source_visualization_path),
                    "export_image_path": str(export_image_path),
                }
            )

    summary = {
        "directory": str(bad_images_dir),
        "threshold": threshold,
        "split": split,
        "bad_class_count": len(bad_classes),
        "exported_count": len(exported_rows),
        "skipped_visualizations": skipped_visualizations,
        "bad_classes": list(bad_classes.values()),
        "manifest_csv": str(manifest_csv_path),
        "manifest_json": str(manifest_json_path),
    }
    rt.save_records_csv(
        manifest_csv_path,
        exported_rows,
        [
            "split",
            "image",
            "class_id",
            "class_name",
            "ap",
            "gt",
            "pred",
            "tp",
            "source_image_path",
            "source_label_path",
            "source_visualization_path",
            "export_image_path",
        ],
    )
    manifest_json_path.write_text(
        json.dumps({"summary": summary, "samples": exported_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"bad_images saved to: {bad_images_dir}")
    return summary


def write_confusion_inputs_index(
    *,
    output_dir: Path,
    report_path: Path,
    args,
    data_cfg: dict[str, Any] | None,
    split: str,
    num_images: int,
) -> Path | None:
    if data_cfg is None:
        return None
    temp_dir = rt.infer_temp_dir(output_dir)
    json_dir = temp_dir / "json"
    prediction_json_count = len(list(json_dir.rglob("*.json"))) if json_dir.exists() else 0
    recorded_output_dir = Path(getattr(args, "_published_output_dir", output_dir))
    recorded_json_dir = rt.infer_temp_dir(recorded_output_dir) / "json"
    recorded_report_path = Path(report_path)
    try:
        recorded_report_path = recorded_output_dir / recorded_report_path.relative_to(output_dir)
    except ValueError:
        pass
    payload = {
        "task": "det",
        "artifact": "confusion_inputs",
        "created_at": rt.timestamp_now_iso(),
        "split": split,
        "num_images": num_images,
        "prediction_json_count": prediction_json_count,
        "paths": {
            "output_dir": str(recorded_output_dir),
            "prediction_json_dir": str(recorded_json_dir),
            "report_path": str(recorded_report_path),
            "data_yaml": str(args.data.expanduser().resolve()) if args.data is not None else None,
            "data_root": str(data_cfg["_root_dir"]),
        },
        "settings": {
            "save_json": args.save_json,
            "score_threshold": args.score_threshold,
            "report_iou_threshold": args.report_iou_threshold,
        },
    }
    index_path = output_dir / "confusion_inputs.json"
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"confusion_inputs saved to: {index_path}")
    return index_path


def write_multi_split_confusion_inputs_index(
    *,
    output_root: Path,
    args,
    data_cfg: dict[str, Any] | None,
    splits: list[str],
    published_output_root: Path | None = None,
) -> Path | None:
    if data_cfg is None:
        return None
    recorded_output_root = (
        Path(published_output_root).expanduser().resolve()
        if published_output_root is not None
        else Path(output_root).expanduser().resolve()
    )
    split_entries: list[dict[str, Any]] = []
    total_prediction_json_count = 0
    for split in splits:
        output_dir = output_root / split
        json_dir = rt.infer_temp_dir(output_dir) / "json"
        recorded_split_dir = recorded_output_root / split
        recorded_json_dir = rt.infer_temp_dir(recorded_split_dir) / "json"
        prediction_json_count = len(list(json_dir.rglob("*.json"))) if json_dir.exists() else 0
        total_prediction_json_count += prediction_json_count
        split_entries.append(
            {
                "split": split,
                "output_dir": str(recorded_split_dir),
                "prediction_json_dir": str(recorded_json_dir),
                "prediction_json_count": prediction_json_count,
                "report_candidates": [
                    str(recorded_split_dir / path.name)
                    for path in sorted(output_dir.glob("*_report.json"))
                ],
            }
        )
    payload = {
        "task": "det",
        "artifact": "confusion_inputs",
        "created_at": rt.timestamp_now_iso(),
        "split": getattr(args, "split", None),
        "splits": splits,
        "prediction_json_count": total_prediction_json_count,
        "paths": {
            "output_dir": str(recorded_output_root),
            "data_yaml": str(args.data.expanduser().resolve()) if args.data is not None else None,
            "data_root": str(data_cfg["_root_dir"]),
        },
        "split_outputs": split_entries,
        "settings": {
            "save_json": args.save_json,
            "score_threshold": args.score_threshold,
            "report_iou_threshold": args.report_iou_threshold,
        },
    }
    index_path = output_root / "confusion_inputs.json"
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"confusion_inputs saved to: {index_path}")
    return index_path


def write_run_meta(
    meta_path: Path,
    *,
    checkpoint_path: Path,
    output_dir: Path,
    report_path: Path,
    input_mode: str,
    args,
    data_cfg: dict[str, Any] | None,
    num_images: int,
    dashboard_path: Path | None,
) -> None:
    recorded_output_dir = Path(output_dir)
    recorded_report_path = Path(report_path)
    published_output_dir = getattr(args, "_published_output_dir", None)
    if published_output_dir is not None:
        try:
            report_relative = recorded_report_path.relative_to(recorded_output_dir)
            recorded_report_path = Path(published_output_dir) / report_relative
        except ValueError:
            pass
        recorded_output_dir = Path(published_output_dir)
    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    important_dir = important_dir_from_checkpoint(checkpoint_path)
    temp_dir = rt.infer_temp_dir(recorded_output_dir)
    action = det_action(args)
    payload = {
        "task": "det",
        "action": action,
        "complete": True,
        "created_at": rt.timestamp_now_iso(),
        "run_name": recorded_output_dir.name,
        "input_mode": input_mode,
        "split": getattr(args, "split", None),
        "num_images": num_images,
        "paths": {
            "output_dir": str(recorded_output_dir),
            "report_path": str(recorded_report_path),
            "checkpoint_path": str(checkpoint_path),
            "experiment_dir": str(experiment_dir),
            "important_dir": str(important_dir),
            "temp_dir": str(temp_dir),
            "data_yaml": str(args.data.expanduser().resolve()) if args.data is not None else None,
            "data_root": str(data_cfg["_root_dir"]) if data_cfg is not None else None,
            "image": str(args.image.expanduser().resolve()) if args.image is not None else None,
            "image_dir": str(args.image_dir.expanduser().resolve()) if args.image_dir is not None else None,
        },
        "settings": {
            "device": args.device,
            "score_threshold": args.score_threshold,
            "report_iou_threshold": args.report_iou_threshold,
            "bad_class_map50_threshold": getattr(
                args,
                "bad_class_map50_threshold",
                rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD,
            ),
            "save_visualization": args.save_visualization,
            "save_json": args.save_json,
            "save_txt": args.save_txt,
            "save_test_report": args.save_test_report,
            "compute_metrics": args.compute_metrics,
            "metric_classwise": args.metric_classwise,
            "vis_max_images": getattr(args, "vis_max_images", None),
            "overwrite": args.overwrite,
        },
        "artifacts": {
            "metrics_summary": str(recorded_output_dir / "metrics_summary.json") if args.data is not None else None,
            "compare": str(recorded_output_dir / "compare") if args.save_visualization else None,
            "bad_images": (
                str(recorded_output_dir / "bad_images")
                if action == "infer" and args.data is not None and args.save_visualization
                else None
            ),
            "confusion_inputs": str(recorded_output_dir / "confusion_inputs.json") if args.data is not None else None,
            "training_dashboard": str(dashboard_path) if dashboard_path is not None else None,
        },
    }
    payload["fingerprint"] = run_reuse.make_fingerprint(
        task="det", action=action, checkpoint=checkpoint_path,
        data=getattr(args, "data", None), split=getattr(args, "split", None),
        threshold=getattr(args, "score_threshold", None), input_mode=input_mode,
        image=getattr(args, "image", None), image_dir=getattr(args, "image_dir", None),
        options=_det_fingerprint_options(args),
    )
    temp_meta = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
    temp_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_meta.replace(meta_path)

def print_dry_run_summary(
    *,
    checkpoint_path: Path,
    output_dir: Path,
    report_path: Path,
    run_meta_path: Path,
    input_mode: str,
    args,
    data_cfg: dict[str, Any] | None,
    num_images: int,
) -> None:
    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    important_dir = important_dir_from_checkpoint(checkpoint_path)
    temp_dir = rt.infer_temp_dir(output_dir)
    report_target = build_split_report_path(
        args=args,
        output_dir=output_dir,
        split=getattr(args, "split", None),
        checkpoint_path=checkpoint_path,
    )
    planned_artifacts = [
        run_meta_path,
        important_dir / "train.log",
        build_important_dashboard_path(checkpoint_path),
    ]
    action = det_action(args)
    if args.save_visualization:
        if action == "infer":
            planned_artifacts.append(output_dir / "images")
        planned_artifacts.append(output_dir / "compare")
    if args.save_json:
        planned_artifacts.append(temp_dir / "json")
    if args.save_txt:
        planned_artifacts.append(temp_dir / "txt")
    if args.data is not None and (args.compute_metrics or args.save_test_report):
        planned_artifacts.append(output_dir / "metrics_summary.json")
    if args.save_test_report:
        planned_artifacts.append(report_target)
        planned_artifacts.append(important_dir / report_target.name)
        if args.save_visualization and action == "infer":
            planned_artifacts.append(output_dir / "bad_images")

    print(f"[dry-run] detection {action} plan")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  experiment_dir: {experiment_dir}")
    print(f"  input_mode: {input_mode}")
    print(f"  split: {getattr(args, 'split', None)}")
    print(f"  num_images: {num_images}")
    if action == "eval" and args.save_visualization:
        print(f"  vis_max_images: {getattr(args, 'vis_max_images', rt.DET_EVAL_VIS_MAX_IMAGES)} per split")
    if data_cfg is not None:
        print(f"  data_yaml: {args.data.expanduser().resolve()}")
        print(f"  data_root: {Path(data_cfg['_root_dir'])}")
    if args.image is not None:
        print(f"  image: {args.image.expanduser().resolve()}")
    if args.image_dir is not None:
        print(f"  image_dir: {args.image_dir.expanduser().resolve()}")
    print(f"  output_dir: {output_dir}")
    print(f"  report_path: {report_target}")
    print(f"  run_meta: {run_meta_path}")
    print("  planned_artifacts:")
    for artifact in planned_artifacts:
        print(f"    - {artifact}")

def maybe_create_metric(compute_metrics: bool, class_names: dict[int, str], classwise: bool):
    if not compute_metrics:
        return None, {}
    if not class_names:
        raise ValueError("Metrics require class names.")
    label_mapping, metric_class_names = rt.build_metric_label_mapping(class_names)
    metric = rt.ObjectDetectionTaskMetric(
        task_metric_args=rt.ObjectDetectionTaskMetricArgs(
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

# mAP 是基于排序的指标：必须吃全量带分预测，由指标内部用 score 画 PR 曲线。
# 若在喂指标前按 score 阈值过滤，会截断 PR 曲线、人为压低 mAP，且与训练内部
# 验证不一致——训练 validation_step 直接把 postprocessor 的全量输出喂给 map_metric，
# 不做任何 score 阈值过滤。因此推理评估时对预测恒用 0.0 阈值（保留全部预测）。
# 注意：lightly_train 的 model.predict 使用 `scores > threshold`（严格大于），
# 0.0 会丢弃恰好为 0 的预测，这对 mAP 无影响（0 分预测排在末尾不改变插值 AP）。
METRIC_PREDICT_THRESHOLD = 0.0


def predict_for_infer(model, predict_path, args):
    """统一的前向入口：默认走 model.predict；args.sahi 为真时走切片推理。

    普通 predict 用 METRIC_PREDICT_THRESHOLD（0.0）做全量预测，score 过滤交给下游
    filter_records_by_score，保证不开 --sahi 时行为与旧版完全一致。

    SAHI 不能传 0.0：predict_sahi 内部要做 tile 间 NMS + 全局/局部合并，喂 0.0 会把
    每个 tile 的几百个近零分框灌进合并逻辑、污染结果（实测比传真实阈值少框且更差）。
    因此 SAHI 直接传真实 score_threshold，与独立脚本/实际部署行为一致。
    """
    if not getattr(args, "sahi", False):
        return model.predict(predict_path, threshold=METRIC_PREDICT_THRESHOLD)

    sahi_threshold = float(getattr(args, "score_threshold", rt.INFER_DEFAULT_SCORE_THRESHOLD))

    # 小图处理：图比模型 tile 小时有两种策略（由 det_sahi_skip_small_images 开关决定）。
    #   1) 跳过 SAHI、回退普通 predict（默认）：小图上 SAHI 会更差，直接用整图推理更稳。
    #   2) 兜底放大：用 PIL 放大到至少一个 tile（保持 uint8，绕开 lightly tile_image 对
    #      uint8 做 F.interpolate 的 "Byte" 报错），切片后再把框坐标按比例缩回原图。
    image_arg: Any = str(predict_path)
    scale = 1.0
    tile_size = getattr(model, "image_size", (640, 640))
    with rt.Image.open(predict_path) as im:
        w, h = im.size
        tile_h, tile_w = int(tile_size[0]), int(tile_size[1])
        if h < tile_h or w < tile_w:
            if getattr(args, "sahi_skip_small", rt.INFER_DEFAULT_SAHI_SKIP_SMALL):
                return model.predict(predict_path, threshold=METRIC_PREDICT_THRESHOLD)
            import math

            scale = max(tile_h / h, tile_w / w)
            new_w, new_h = math.ceil(w * scale), math.ceil(h * scale)
            image_arg = im.convert("RGB").resize(
                (new_w, new_h), rt.Image.Resampling.BILINEAR
            )

    prediction = model.predict_sahi(
        image=image_arg,
        threshold=sahi_threshold,
        overlap=getattr(args, "sahi_overlap", rt.INFER_DEFAULT_SAHI_OVERLAP),
        nms_iou_threshold=getattr(args, "sahi_nms_iou", rt.INFER_DEFAULT_SAHI_NMS_IOU),
        global_local_iou_threshold=getattr(
            args, "sahi_global_local_iou", rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU
        ),
    )
    if scale != 1.0 and len(prediction["bboxes"]) > 0:
        prediction["bboxes"] = prediction["bboxes"] / scale  # 框坐标缩回原图尺寸
    return prediction


def filter_records_by_score(records: list[dict[str, Any]], score_threshold: float) -> list[dict[str, Any]]:
    """按 score 阈值过滤记录，供可视化 / JSON / TXT / legacy 报表使用。

    与 model.predict 的 `scores > threshold` 语义保持一致（严格大于），
    以保证开启该阈值后这些产物的输出与旧行为完全相同。
    """
    return [record for record in records if record["score"] > score_threshold]


def update_metric(metric, label_mapping: dict[int, int], prediction, gt_boxes, gt_labels) -> None:
    if metric is None:
        return
    pred_labels = prediction["labels"].detach().cpu().to(rt.torch.int64)
    pred_boxes = prediction["bboxes"].detach().cpu().to(rt.torch.float32)
    pred_scores = prediction["scores"].detach().cpu().to(rt.torch.float32)
    metric.update_with_predictions(
        preds=[{"boxes": pred_boxes, "scores": pred_scores, "labels": rt.remap_labels(pred_labels, label_mapping)}],
        target=[{"boxes": gt_boxes.to(rt.torch.float32), "labels": rt.remap_labels(gt_labels.to(rt.torch.int64), label_mapping)}],
    )


MULTI_SPLIT_MODES = {"test+val", "all"}


def resolve_dataset_infer_splits(data_cfg: dict[str, Any], requested_split: str) -> list[str]:
    split_groups = {
        "train": ("train",),
        "test": ("test",),
        "val": ("val",),
        "test+val": ("test", "val"),
        "all": ("train", "test", "val"),
    }
    if requested_split not in split_groups:
        raise ValueError(f"Unsupported split: {requested_split}")

    splits = [split for split in split_groups[requested_split] if data_cfg.get(split)]
    if not splits:
        raise ValueError(f"data.yaml 中未找到可用于 infer 的 split 配置: {requested_split}")
    return splits


def build_split_output_dir(
    *,
    args,
    checkpoint_path: Path,
    data_cfg: dict[str, Any] | None,
    split: str,
) -> Path:
    action = det_action(args)
    split_mode = getattr(args, "requested_split", getattr(args, "split", None))
    if args.output_dir is not None:
        return args.output_dir / split if split_mode in MULTI_SPLIT_MODES else args.output_dir

    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    if data_cfg is None:
        input_path = getattr(args, "image", None) or getattr(args, "image_dir", None)
        return rt.build_action_output_dir(
            experiment_dir, action, input_path=input_path,
        )
    data_path = Path(data_cfg.get("_data_yaml_path", data_cfg["_root_dir"]))
    return rt.build_action_output_dir(
        experiment_dir, action, input_path=data_path, split=split,
    )


def build_multi_split_output_root(
    *,
    args,
    checkpoint_path: Path,
    data_cfg: dict[str, Any],
) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir).expanduser().resolve()
    data_path = Path(data_cfg.get("_data_yaml_path", args.data or data_cfg["_root_dir"]))
    return rt.build_action_output_dir(
        rt.experiment_dir_from_checkpoint_path(checkpoint_path),
        det_action(args),
        input_path=data_path,
    )


def build_split_report_path(
    *,
    args,
    output_dir: Path,
    split: str,
    checkpoint_path: Path | None = None,
) -> Path:
    split_mode = getattr(args, "requested_split", getattr(args, "split", None))
    if args.report_path is None:
        return rt.build_det_report_path(output_dir, split=split)
    if split_mode not in MULTI_SPLIT_MODES:
        return args.report_path
    report_path = Path(args.report_path).expanduser().resolve()
    if report_path.parent == Path(output_dir).expanduser().resolve():
        return report_path
    stem = args.report_path.stem
    suffix = args.report_path.suffix or ".json"
    return args.report_path.with_name(f"{stem}-{split}{suffix}")


def rebase_output_artifact_path(
    path: Path,
    *,
    source_output_dir: Path,
    target_output_dir: Path,
) -> Path:
    """把输出目录内的产物路径从发布目录映射到 staging 目录。"""
    path = Path(path).expanduser().resolve()
    source_output_dir = Path(source_output_dir).expanduser().resolve()
    target_output_dir = Path(target_output_dir).expanduser().resolve()
    try:
        return target_output_dir / path.relative_to(source_output_dir)
    except ValueError:
        return path


def build_parallel_child_command(
    *,
    args,
    split: str,
    device: str,
    output_dir: Path,
    report_path: Path,
    shard_index: int | None = None,
    num_shards: int = 1,
) -> list[str]:
    command = [
        sys.executable,
        str(rt.ROOT_DIR / "launcher.py"),
        det_action(args),
        "--task",
        "det",
    ]
    if args.experiment_dir is not None:
        command.extend(["--experiment-dir", str(args.experiment_dir)])
    if args.checkpoint is not None:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.data is not None:
        command.extend(["--data", str(args.data)])
    if args.image is not None:
        command.extend(["--image", str(args.image)])
    if args.image_dir is not None:
        command.extend(["--image-dir", str(args.image_dir)])

    command.extend(
        [
            "--split",
            split,
            "--output-dir",
            str(output_dir),
            "--report-path",
            str(report_path),
            "--score-threshold",
            str(args.score_threshold),
            "--report-iou-threshold",
            str(args.report_iou_threshold),
            "--bad-class-map50-threshold",
            str(getattr(args, "bad_class_map50_threshold", rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD)),
            "--device",
            device,
        ]
    )

    command.append("--save-visualization" if args.save_visualization else "--skip-visualization")
    if det_action(args) == "eval":
        command.extend(["--vis-max-images", str(getattr(args, "vis_max_images", rt.DET_EVAL_VIS_MAX_IMAGES))])
        if args.metric_classwise:
            command.append("--classwise")
    else:
        command.append("--save-json" if args.save_json else "--skip-json")
        if args.save_txt:
            command.append("--save-txt")
        if args.compute_metrics:
            command.append("--compute-metrics")
        if args.metric_classwise:
            command.append("--metric-classwise")
        if args.save_test_report:
            command.append("--save-test-report")
    if args.overwrite:
        command.append("--overwrite")
    if getattr(args, "sahi", False):
        command.extend(
            [
                "--sahi",
                "--sahi-overlap",
                str(getattr(args, "sahi_overlap", rt.INFER_DEFAULT_SAHI_OVERLAP)),
                "--sahi-nms-iou",
                str(getattr(args, "sahi_nms_iou", rt.INFER_DEFAULT_SAHI_NMS_IOU)),
                "--sahi-global-local-iou",
                str(getattr(args, "sahi_global_local_iou", rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU)),
            ]
        )
        command.append(
            "--sahi-skip-small"
            if getattr(args, "sahi_skip_small", rt.INFER_DEFAULT_SAHI_SKIP_SMALL)
            else "--no-sahi-skip-small"
        )
    if getattr(args, "dry_run", False):
        command.append("--dry-run")
    if getattr(args, "skip_important_artifacts", False):
        command.append("--skip-important-artifacts")
    if getattr(args, "selected_splits", None):
        command.extend(["--selected-splits", str(args.selected_splits)])
    if getattr(args, "multi_output_root", None) is not None:
        command.extend(["--multi-output-root", str(args.multi_output_root)])
    if shard_index is not None:
        command.extend(["--shard-index", str(shard_index), "--num-shards", str(num_shards)])
    return command


def build_metric_entry(prediction, gt_boxes, gt_labels) -> dict[str, Any]:
    return {
        "pred_boxes": prediction["bboxes"].detach().cpu().to(rt.torch.float32).tolist(),
        "pred_scores": prediction["scores"].detach().cpu().to(rt.torch.float32).tolist(),
        "pred_labels": prediction["labels"].detach().cpu().to(rt.torch.int64).tolist(),
        "gt_boxes": gt_boxes.detach().cpu().to(rt.torch.float32).tolist(),
        "gt_labels": gt_labels.detach().cpu().to(rt.torch.int64).tolist(),
    }


def serialize_legacy_class_data(class_data: dict[int, dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for class_id, info in class_data.items():
        payload[str(class_id)] = {
            "gt": int(info.get("gt", 0)),
            "pairs": [[float(score), int(flag)] for score, flag in info.get("pairs", [])],
        }
    return payload


def deserialize_legacy_class_data(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    class_data: dict[int, dict[str, Any]] = {}
    for class_id_raw, info in payload.items():
        if not isinstance(info, dict):
            continue
        try:
            class_id = int(class_id_raw)
        except ValueError:
            continue
        pairs_raw = info.get("pairs", [])
        pairs: list[tuple[float, int]] = []
        if isinstance(pairs_raw, list):
            for pair in pairs_raw:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    try:
                        pairs.append((float(pair[0]), int(pair[1])))
                    except (TypeError, ValueError):
                        continue
        class_data[class_id] = {
            "gt": int(info.get("gt", 0)),
            "pairs": pairs,
        }
    return class_data


def write_shard_result(
    shard_output_dir: Path,
    *,
    split: str,
    class_names: dict[int, str],
    num_images: int,
    processed_images: int,
    infer_time_sum_ms: float,
    metrics_meta: dict[str, int],
    legacy_class_data: dict[int, dict[str, Any]],
    total_gt: int,
    total_pred: int,
    total_tp: int,
    metric_entries: list[dict[str, Any]] | None = None,
    metric_entries_path: Path | None = None,
) -> Path:
    has_metric_entries = bool(metric_entries) or (
        metric_entries_path is not None and metric_entries_path.is_file()
    )
    payload = {
        "split": split,
        "class_names": {str(class_id): name for class_id, name in class_names.items()},
        "num_images": num_images,
        "processed_images": processed_images,
        "infer_time_sum_ms": infer_time_sum_ms,
        "metrics_meta": metrics_meta,
        "legacy_class_data": serialize_legacy_class_data(legacy_class_data),
        "total_gt": total_gt,
        "total_pred": total_pred,
        "total_tp": total_tp,
        "metric_entries_file": "metric_entries.jsonl" if has_metric_entries else None,
    }
    result_path = shard_output_dir / "shard_result.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if metric_entries:
        entries_path = shard_output_dir / "metric_entries.jsonl"
        with entries_path.open("w", encoding="utf-8") as stream:
            for entry in metric_entries:
                stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    elif metric_entries_path is not None and metric_entries_path.resolve() != (shard_output_dir / "metric_entries.jsonl").resolve():
        shutil.copy2(metric_entries_path, shard_output_dir / "metric_entries.jsonl")
    return result_path


def copy_tree_contents(
    source_dir: Path,
    target_dir: Path,
    *,
    progress_label: str | None = None,
) -> None:
    rt.copy_tree_contents(source_dir, target_dir, progress_label=progress_label)


def aggregate_metric_entries(metric, label_mapping: dict[int, int], metric_entries: list[dict[str, Any]]) -> None:
    for entry in metric_entries:
        pred_boxes = rt.torch.as_tensor(entry.get("pred_boxes", []), dtype=rt.torch.float32).reshape(-1, 4)
        pred_scores = rt.torch.as_tensor(entry.get("pred_scores", []), dtype=rt.torch.float32)
        pred_labels = rt.torch.as_tensor(entry.get("pred_labels", []), dtype=rt.torch.int64)
        gt_boxes = rt.torch.as_tensor(entry.get("gt_boxes", []), dtype=rt.torch.float32).reshape(-1, 4)
        gt_labels = rt.torch.as_tensor(entry.get("gt_labels", []), dtype=rt.torch.int64)
        metric.update_with_predictions(
            preds=[{"boxes": pred_boxes, "scores": pred_scores, "labels": rt.remap_labels(pred_labels, label_mapping)}],
            target=[{"boxes": gt_boxes, "labels": rt.remap_labels(gt_labels, label_mapping)}],
        )


def merge_shard_results(
    *,
    shard_output_dirs: list[Path],
    final_output_dir: Path,
    checkpoint_path: Path,
    data_cfg: dict[str, Any] | None,
    split: str,
    args,
) -> None:
    merged_class_names: dict[int, str] = {}
    metric_entry_paths: list[Path] = []
    merged_legacy_class_data: dict[int, dict[str, Any]] = defaultdict(lambda: {"gt": 0, "pairs": []})
    num_images = 0
    processed_images = 0
    infer_time_sum_ms = 0.0
    total_gt = 0
    total_pred = 0
    total_tp = 0
    metrics_meta = {"images_with_labels": 0, "images_without_labels": 0}

    for shard_number, shard_output_dir in enumerate(track(
        shard_output_dirs,
        label=f"{det_label(args)} 合并分片产物",
        total=len(shard_output_dirs),
        unit="shard",
    ), start=1):
        shard_result_path = shard_output_dir / "shard_result.json"
        shard_payload = json.loads(shard_result_path.read_text(encoding="utf-8"))
        class_names_payload = shard_payload.get("class_names", {})
        if isinstance(class_names_payload, dict):
            for class_id_raw, name in class_names_payload.items():
                try:
                    merged_class_names[int(class_id_raw)] = str(name)
                except ValueError:
                    continue
        entries_file = shard_payload.get("metric_entries_file")
        if entries_file:
            metric_entry_paths.append(shard_output_dir / str(entries_file))
        for class_id, info in deserialize_legacy_class_data(shard_payload.get("legacy_class_data", {})).items():
            merged_legacy_class_data[class_id]["gt"] += int(info.get("gt", 0))
            merged_legacy_class_data[class_id]["pairs"].extend(info.get("pairs", []))
        num_images += int(shard_payload.get("num_images", 0))
        processed_images += int(shard_payload.get("processed_images", 0))
        infer_time_sum_ms += float(shard_payload.get("infer_time_sum_ms", 0.0))
        total_gt += int(shard_payload.get("total_gt", 0))
        total_pred += int(shard_payload.get("total_pred", 0))
        total_tp += int(shard_payload.get("total_tp", 0))
        shard_metrics_meta = shard_payload.get("metrics_meta", {})
        metrics_meta["images_with_labels"] += int(shard_metrics_meta.get("images_with_labels", 0))
        metrics_meta["images_without_labels"] += int(shard_metrics_meta.get("images_without_labels", 0))
        final_temp_dir = rt.infer_temp_dir(final_output_dir)
        merge_prefix = f"{det_label(args)} 合并 {shard_number}/{len(shard_output_dirs)}"
        if args.save_visualization:
            copy_tree_contents(
                shard_output_dir / "images",
                final_output_dir / "images",
                progress_label=f"{merge_prefix} images",
            )
            copy_tree_contents(
                shard_output_dir / "compare",
                final_output_dir / "compare",
                progress_label=f"{merge_prefix} compare",
            )
        if args.save_json or args.save_txt:
            shard_temp_dir = rt.infer_temp_dir(shard_output_dir)
        if args.save_json:
            copy_tree_contents(
                shard_temp_dir / "json",
                final_temp_dir / "json",
                progress_label=f"{merge_prefix} json",
            )
        if args.save_txt:
            copy_tree_contents(
                shard_temp_dir / "txt",
                final_temp_dir / "txt",
                progress_label=f"{merge_prefix} txt",
            )

    aggregated_metric_values: dict[str, float] | None = None
    compute_full_metrics = data_cfg is not None and (args.compute_metrics or args.save_test_report)
    if compute_full_metrics and merged_class_names and metrics_meta["images_with_labels"] > 0:
        metric, label_mapping = maybe_create_metric(True, merged_class_names, args.metric_classwise)

        def _metric_entries():
            for entries_path in metric_entry_paths:
                with entries_path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            yield json.loads(line)

        for entry in track(
            _metric_entries(),
            label=f"{det_label(args)} 聚合评估指标",
            total=metrics_meta["images_with_labels"],
            unit="img",
        ):
            aggregate_metric_entries(metric, label_mapping, [entry])
        print(f"[{det_label(args)}] 阶段: 计算聚合指标", flush=True)
        aggregated_metric_values = metric.compute_aggregated_values().metric_values
        print(f"[{det_label(args)}] 阶段完成: 聚合指标", flush=True)
        metrics_payload = {
            "checkpoint": str(checkpoint_path),
            "input_mode": "dataset",
            "num_images": num_images,
            **metrics_meta,
            "metrics": aggregated_metric_values,
            "avg_infer_time_ms": infer_time_sum_ms / max(num_images, 1),
        }
        metrics_path = final_output_dir / "metrics_summary.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Metrics saved to: {metrics_path}")

    if data_cfg is not None and args.save_test_report:
        report_path = build_split_report_path(
            args=args,
            output_dir=final_output_dir,
            split=split,
            checkpoint_path=checkpoint_path,
        )
        report_payload = build_legacy_report(
            checkpoint_path=checkpoint_path,
            data_cfg=data_cfg,
            split=split,
            score_threshold=args.score_threshold,
            iou_threshold=args.report_iou_threshold,
            class_names=merged_class_names,
            class_data=dict(merged_legacy_class_data),
            num_images=num_images,
            processed_images=processed_images,
            total_gt=total_gt,
            total_pred=total_pred,
            total_tp=total_tp,
            avg_infer_time_ms=infer_time_sum_ms / max(num_images, 1),
            metric_values=aggregated_metric_values,
            metrics_meta=metrics_meta,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"test_report saved to: {report_path}")
        if det_action(args) == "infer" and args.save_visualization:
            export_bad_class_images(
                args=args,
                output_dir=final_output_dir,
                report_path=report_path,
                data_cfg=data_cfg,
                split=split,
            )
    write_confusion_inputs_index(
        output_dir=final_output_dir,
        report_path=build_split_report_path(
            args=args,
            output_dir=final_output_dir,
            split=split,
            checkpoint_path=checkpoint_path,
        ),
        args=args,
        data_cfg=data_cfg,
        split=split,
        num_images=num_images,
    )


def run_multi_split_shard_infer(args) -> None:
    action = det_action(args)
    selected_splits_raw = str(getattr(args, "selected_splits", "") or "")
    splits = [split.strip() for split in selected_splits_raw.split(",") if split.strip()]
    if not splits:
        raise ValueError("Multi-split shard infer requires selected_splits.")

    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    args._resolved_checkpoint_path = checkpoint_path
    split_samples, input_class_names, data_cfg = get_multi_split_input_samples(args, splits)
    rt.ensure_image_samples(split_samples)

    shard_workspace_dir = Path(args.output_dir).expanduser().resolve()
    final_output_root = Path(getattr(args, "multi_output_root", "")).expanduser().resolve()
    output_dirs = {split: final_output_root / split for split in splits}

    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    ensure_object_detection_model(model)
    class_names = rt.merge_class_names(rt.get_model_class_names(model), input_class_names)

    split_states: dict[str, dict[str, Any]] = {}
    for split in splits:
        split_states[split] = {
            "num_images": 0,
            "processed_images": 0,
            "infer_time_sum_ms": 0.0,
            "metrics_meta": {"images_with_labels": 0, "images_without_labels": 0},
            "legacy_class_data": {},
            "total_gt": 0,
            "total_pred": 0,
            "total_tp": 0,
            "metric_entries_path": shard_workspace_dir / split / "metric_entries.jsonl",
        }

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Input mode: dataset")
    print(f"Images to process: {len(split_samples)}")
    print(f"Selected splits: {', '.join(splits)}")
    for split in splits:
        split_count = sum(1 for item in split_samples if item.split == split)
        print(f"Split images: {split}={split_count}")

    split_vis_paths: set[tuple[str, Path]] = set()
    for split in splits:
        local_samples = [item for item in split_samples if item.split == split]
        selected = visualization_indices(len(local_samples), args)
        split_vis_paths.update(
            (split, local_samples[index].sample.image_path)
            for index in selected
        )

    with tempfile.TemporaryDirectory(prefix="lt_rgb_") as _rgb_scratch_str:
        _rgb_scratch = Path(_rgb_scratch_str)
        for idx, split_sample in enumerate(
            track(
                split_samples,
                label=f"{det_label(args)} 推理",
                total=len(split_samples),
                unit="img",
                shard_scope="inference",
            ),
            start=1,
        ):
            split = split_sample.split
            sample = split_sample.sample
            state = split_states[split]
            state["num_images"] += 1

            output_dir = output_dirs[split]
            temp_dir = rt.infer_temp_dir(output_dir)
            images_dir = output_dir / "images"
            json_dir = temp_dir / "json"
            txt_dir = temp_dir / "txt"

            with rt.Image.open(sample.image_path) as image:
                image_size = image.size
                predict_path = sample.image_path
                if image.mode != "RGB":
                    predict_path = _prepare_rgb_predict_path(image, sample, _rgb_scratch)

            infer_start = time.perf_counter()
            # 全量预测（供 mAP 用）；score 阈值过滤只作用于可视化/JSON/TXT/legacy 报表。
            prediction = predict_for_infer(model, predict_path, args)
            state["infer_time_sum_ms"] += (time.perf_counter() - infer_start) * 1000.0
            records = filter_records_by_score(
                prediction_records(prediction, class_names, image_size),
                args.score_threshold,
            )

            gt_boxes, gt_labels, has_label = load_ground_truth(sample.label_path, image_size)

            if (split, sample.image_path) in split_vis_paths:
                gt_items_vis = gt_records(gt_boxes, gt_labels) if has_label else []
                if action == "infer":
                    vis_path = visualization_output_path(
                        images_dir,
                        sample,
                        rt.visualization_suffix(sample.image_path),
                        records,
                    )
                    draw_predictions(sample.image_path, vis_path, records, gt_items_vis, class_names)
                if has_label:
                    compare_path = comparison_output_path(
                        output_dir, sample, rt.visualization_suffix(sample.image_path)
                    )
                    draw_comparison(sample.image_path, compare_path, records, gt_items_vis, class_names)
            if args.save_json:
                save_prediction_json(json_dir / relative_output_path(sample, ".json"), sample, image_size, records)
            if args.save_txt:
                save_prediction_txt(txt_dir / relative_output_path(sample, ".txt"), records)

            if has_label:
                state["metrics_meta"]["images_with_labels"] += 1
            else:
                state["metrics_meta"]["images_without_labels"] += 1
            metric_entry = build_metric_entry(prediction, gt_boxes, gt_labels)
            entries_path = state["metric_entries_path"]
            entries_path.parent.mkdir(parents=True, exist_ok=True)
            with entries_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(metric_entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
            if args.save_test_report:
                gt_items = gt_records(gt_boxes, gt_labels)
                image_total_gt, image_total_pred, image_total_tp = update_legacy_report_state(
                    state["legacy_class_data"],
                    gt_items,
                    records,
                    args.report_iou_threshold,
                )
                state["total_gt"] += image_total_gt
                state["total_pred"] += image_total_pred
                state["total_tp"] += image_total_tp

            state["processed_images"] += 1

    for split in splits:
        state = split_states[split]
        if state["num_images"] == 0:
            continue
        shard_split_dir = shard_workspace_dir / split
        shard_split_dir.mkdir(parents=True, exist_ok=True)
        shard_result_path = write_shard_result(
            shard_split_dir,
            split=split,
            class_names=class_names,
            num_images=state["num_images"],
            processed_images=state["processed_images"],
            infer_time_sum_ms=state["infer_time_sum_ms"],
            metrics_meta=state["metrics_meta"],
            legacy_class_data=state["legacy_class_data"],
            total_gt=state["total_gt"],
            total_pred=state["total_pred"],
            total_tp=state["total_tp"],
            metric_entries_path=state["metric_entries_path"],
        )
        print(f"shard_result saved to: {shard_result_path}")


def finalize_parallel_infer_output(
    *,
    args,
    checkpoint_path: Path,
    data_cfg: dict[str, Any] | None,
    final_output_dir: Path,
    split: str,
    input_mode: str,
    num_images: int,
) -> None:
    dashboard_path = None
    if not getattr(args, "skip_important_artifacts", False):
        print(f"[{det_label(args)}] 阶段: 同步训练仪表盘")
        dashboard_path = sync_important_artifacts(checkpoint_path)
    if dashboard_path is not None:
        print(f"training_dashboard copied to: {dashboard_path}")

    report_path = build_split_report_path(
        args=args,
        output_dir=final_output_dir,
        split=split,
        checkpoint_path=checkpoint_path,
    )
    run_meta_path = final_output_dir / "run_meta.json"
    args.output_dir = final_output_dir
    args.report_path = report_path
    write_run_meta(
        run_meta_path,
        checkpoint_path=checkpoint_path,
        output_dir=final_output_dir,
        report_path=report_path,
        input_mode=input_mode,
        args=args,
        data_cfg=data_cfg,
        num_images=num_images,
        dashboard_path=dashboard_path,
    )
    print(f"run_meta saved to: {run_meta_path}")


def finalize_published_infer_reports(
    *,
    args,
    checkpoint_path: Path,
    output_dir: Path,
) -> list[Path]:
    """在 staging 发布完成后生成 Markdown 报告并同步 important 产物。"""
    if (
        not args.save_test_report
        or getattr(args, "skip_important_artifacts", False)
        or getattr(args, "_defer_report_generation", False)
    ):
        return []

    output_dir = Path(output_dir).expanduser().resolve()
    run_meta_path = output_dir / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    report_path_value = run_meta.get("paths", {}).get("report_path")
    report_path = (
        Path(report_path_value).expanduser().resolve()
        if isinstance(report_path_value, str) and report_path_value.strip()
        else build_split_report_path(
            args=SimpleNamespace(**{**vars(args), "report_path": None}),
            output_dir=output_dir,
            split=getattr(args, "split", "test"),
            checkpoint_path=checkpoint_path,
        )
    )

    print(f"[{det_label(args)}] 阶段: 生成最终报告")
    report_markdown_paths = generate_report_for_infer_output(
        experiment_dir=rt.experiment_dir_from_checkpoint_path(checkpoint_path),
        output_dir=output_dir,
    )
    for report_markdown_path in report_markdown_paths:
        print(f"single_report saved to: {report_markdown_path}")
    for copied_path in copy_infer_artifacts_to_important(
        checkpoint_path, [report_path, *report_markdown_paths]
    ):
        print(f"{det_label(args)} artifact copied to important: {copied_path}")
    return report_markdown_paths


def finalize_published_multi_split_reports(
    *,
    args,
    checkpoint_path: Path,
    output_root: Path,
    splits: list[str],
) -> None:
    """刷新联合任务的每个 split 报告，确保报告能同时读取全部正式结果。"""
    for split in splits:
        output_dir = Path(output_root) / split
        run_meta_path = output_dir / "run_meta.json"
        if not run_meta_path.exists():
            raise FileNotFoundError(f"split={split} 缺少已发布的 run_meta.json: {run_meta_path}")
        split_args = SimpleNamespace(**vars(args))
        split_args.requested_split = args.split
        split_args.split = split
        split_args.output_dir = output_dir
        split_args._defer_report_generation = False
        finalize_published_infer_reports(
            args=split_args,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
        )


def run_parallel_split_infer(args) -> bool:
    label = det_label(args)
    if getattr(args, "dry_run", False):
        print(f"[{label}] dry-run 保持顺序模式。")
        return False
    if is_shard_child(args):
        return False
    if args.data is None or args.split in MULTI_SPLIT_MODES:
        return False
    if args.device != "auto":
        return False

    all_gpus, inventory_message = query_gpu_inventory()
    if inventory_message:
        print(f"[{label}] {inventory_message}")
        return False
    eligible_gpus = filter_high_memory_gpus(all_gpus)
    if len(eligible_gpus) < 2:
        print(
            f"[{label}] 当前 split={args.split} 保留 GPU 数量为 {len(eligible_gpus)}，进入单卡顺序模式。"
        )
        return False

    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    args._resolved_checkpoint_path = checkpoint_path
    data_cfg = rt.load_data_config(args.data)
    final_output_dir = build_split_output_dir(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        split=args.split,
    )
    if _det_reuse_precheck(args, final_output_dir, checkpoint_path) == run_reuse.REUSE:
        print(f"[{label}] 已复用上次结果：{final_output_dir}")
        args.output_dir = final_output_dir
        args.report_path = build_split_report_path(
            args=args,
            output_dir=final_output_dir,
            split=args.split,
            checkpoint_path=checkpoint_path,
        )
        finalize_published_infer_reports(
            args=args,
            checkpoint_path=checkpoint_path,
            output_dir=final_output_dir,
        )
        print_published_eval_artifacts(final_output_dir, args)
        return True

    base_samples, _, input_mode = get_input_samples(SimpleNamespace(**{**vars(args), "shard_index": None, "num_shards": 1}))
    rt.ensure_image_samples(base_samples)
    num_shards = min(len(eligible_gpus), len(base_samples))
    if num_shards < 2:
        print(f"[{label}] split={args.split} 可分配 shard 数量为 {num_shards}，进入单卡顺序模式。")
        return False
    published_output_dir = final_output_dir
    published_report_path = build_split_report_path(
        args=args,
        output_dir=published_output_dir,
        split=args.split,
        checkpoint_path=checkpoint_path,
    )
    final_output_dir = create_stage(published_output_dir, overwrite=args.overwrite)
    args._published_output_dir = published_output_dir
    args.report_path = rebase_output_artifact_path(
        published_report_path,
        source_output_dir=published_output_dir,
        target_output_dir=final_output_dir,
    )
    rt.prepare_output_dir(final_output_dir, args.overwrite, clean=True)

    shard_output_dirs: list[Path] = []
    jobs: list[tuple[int, int | str, list[str]]] = []
    report_path = build_split_report_path(
        args=args,
        output_dir=final_output_dir,
        split=args.split,
        checkpoint_path=checkpoint_path,
    )
    for shard_index, gpu in enumerate(eligible_gpus[:num_shards]):
        shard_output_dir = rt.infer_temp_dir(final_output_dir) / "_shards" / f"shard_{shard_index:02d}"
        shard_output_dirs.append(shard_output_dir)
        command = build_parallel_child_command(
            args=args,
            split=args.split,
            device="auto",
            output_dir=shard_output_dir,
            report_path=report_path.with_name(f"{report_path.stem}-shard-{shard_index:02d}{report_path.suffix}"),
            shard_index=shard_index,
            num_shards=num_shards,
        )
        jobs.append((shard_index, gpu.get("device_token", int(gpu["index"])), command))

    print(f"[{label}] split={args.split} 已进入 shard 多卡推理模式。")
    selected_gpus = eligible_gpus[:num_shards]
    for gpu in selected_gpus:
        print(f"[{label}] 保留 {format_gpu_summary(gpu)}")
    for shard_index, gpu_index, _ in jobs:
        print(f"[{label}] shard={shard_index}/{num_shards} -> CUDA_VISIBLE_DEVICES={gpu_index}")

    shard_totals = gpu_parallel.shard_item_totals(
        len(base_samples),
        num_shards=num_shards,
        weights=[_sample_weight(sample) for sample in base_samples],
    )
    failed_shards = gpu_parallel.run_sharded_subprocesses(
        jobs,
        cwd=rt.ROOT_DIR,
        shard_totals=shard_totals,
    )
    if failed_shards:
        raise RuntimeError(f"{label} shard 失败: {', '.join(failed_shards)}")

    merge_shard_results(
        shard_output_dirs=shard_output_dirs,
        final_output_dir=final_output_dir,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        split=args.split,
        args=args,
    )
    finalize_parallel_infer_output(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        final_output_dir=final_output_dir,
        split=args.split,
        input_mode=input_mode,
        num_images=len(base_samples),
    )
    shutil.rmtree(rt.infer_temp_dir(final_output_dir) / "_shards", ignore_errors=False)
    publish_stage(final_output_dir, published_output_dir, overwrite=args.overwrite)
    args.output_dir = published_output_dir
    args.report_path = published_report_path
    finalize_published_infer_reports(
        args=args,
        checkpoint_path=checkpoint_path,
        output_dir=published_output_dir,
    )
    print_published_eval_artifacts(published_output_dir, args)
    delattr(args, "_published_output_dir")
    return True


def run_parallel_all_infer(args, splits: list[str]) -> bool:
    label = det_label(args)
    if getattr(args, "dry_run", False):
        print(f"[{label}] dry-run 保持顺序模式。")
        return False
    if args.device != "auto":
        print(f"[{label}] 当前使用指定 device={args.device}，进入单卡顺序模式。")
        return False
    if len(splits) < 2:
        print(f"[{label}] 当前可执行 split 数量为 {len(splits)}，进入单卡顺序模式。")
        return False

    all_gpus, inventory_message = query_gpu_inventory()
    if inventory_message:
        print(f"[{label}] {inventory_message}")
        return False
    eligible_gpus = filter_high_memory_gpus(all_gpus)
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    data_cfg = rt.load_data_config(args.data) if args.data is not None else None
    if data_cfg is None:
        raise ValueError("Multi-split infer requires dataset configuration.")
    requested_splits = list(splits)
    published_output_root = build_multi_split_output_root(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
    )

    # 逐 split 初步复用检测：可复用的直接跳过，只并行重跑缺数据/不一致的 split。
    rerun_splits: list[str] = []
    for split in splits:
        split_args = SimpleNamespace(**vars(args))
        split_args.requested_split = args.split
        split_args.split = split
        out_dir = build_split_output_dir(
            args=split_args, checkpoint_path=checkpoint_path, data_cfg=data_cfg, split=split,
        )
        if _det_reuse_precheck(split_args, out_dir, checkpoint_path) == run_reuse.REUSE:
            print(f"[{label}] split={split} 已复用，跳过。")
        else:
            rerun_splits.append(split)
    if not rerun_splits:
        print(f"[{label}] 所有 split 均已复用。")
        args.output_dir = published_output_root
        write_multi_split_confusion_inputs_index(
            output_root=published_output_root,
            args=args,
            data_cfg=data_cfg,
            splits=requested_splits,
        )
        finalize_published_multi_split_reports(
            args=args,
            checkpoint_path=checkpoint_path,
            output_root=published_output_root,
            splits=requested_splits,
        )
        print_published_eval_artifacts(published_output_root, args)
        return True
    splits = rerun_splits
    args.overwrite = True  # 需重跑的 split 走干净覆盖

    base_samples, _, _ = get_multi_split_input_samples(
        SimpleNamespace(**{**vars(args), "shard_index": None, "num_shards": 1}),
        splits,
    )
    rt.ensure_image_samples(base_samples)
    split_num_images = {
        split: sum(1 for item in base_samples if item.split == split)
        for split in splits
    }
    empty_splits = [split for split, count in split_num_images.items() if count == 0]
    if empty_splits:
        raise ValueError(f"以下 split 未找到图片: {', '.join(empty_splits)}")
    num_shards = min(len(eligible_gpus), len(base_samples))
    if num_shards < 2:
        print(
            f"[{label}] 可用于并行的 GPU 数量为 {len(eligible_gpus)}，样本数为 {len(base_samples)}，进入单卡顺序模式。"
        )
        return False

    work_output_root = create_stage(published_output_root, overwrite=True)
    if published_output_root.exists():
        shutil.copytree(published_output_root, work_output_root, dirs_exist_ok=True)
    final_output_root = work_output_root
    for split in splits:
        split_args = SimpleNamespace(**vars(args))
        split_args.requested_split = args.split
        split_args.split = split
        output_dir = final_output_root / split
        rt.prepare_output_dir(output_dir, args.overwrite, clean=True)
    print(
        f"[{label}] split 图片数: "
        + ", ".join(f"{split}={split_num_images[split]}" for split in splits)
    )

    shard_root = rt.infer_temp_dir(final_output_root) / "_multi_shards" / str(args.split)
    shard_output_dirs: list[Path] = []
    jobs: list[tuple[int, int | str, list[str]]] = []
    for shard_index, gpu in enumerate(eligible_gpus[:num_shards]):
        shard_output_dir = shard_root / f"shard_{shard_index:02d}"
        shard_output_dirs.append(shard_output_dir)
        child_args = SimpleNamespace(**vars(args))
        child_args.selected_splits = ",".join(splits)
        child_args.multi_output_root = final_output_root
        command = build_parallel_child_command(
            args=child_args,
            split=args.split,
            device="auto",
            output_dir=shard_output_dir,
            report_path=shard_output_dir / "shard_report.json",
            shard_index=shard_index,
            num_shards=num_shards,
        )
        jobs.append((shard_index, gpu.get("device_token", int(gpu["index"])), command))

    print(f"[{label}] 已进入自动多卡并行模式。")
    print(
        f"[{label}] 共检测到 {len(all_gpus)} 张 GPU，剔除高显存占用后保留 {len(eligible_gpus)} 张，按样本平铺到所有卡。"
    )
    for gpu in eligible_gpus:
        print(f"[{label}] 保留 {format_gpu_summary(gpu)}")
    for shard_index, gpu_index, _ in jobs:
        print(f"[{label}] shard={shard_index}/{num_shards} -> CUDA_VISIBLE_DEVICES={gpu_index}")

    shard_totals = gpu_parallel.shard_item_totals(
        len(base_samples),
        num_shards=num_shards,
        weights=[_sample_weight(sample) for sample in base_samples],
    )
    failed_shards = gpu_parallel.run_sharded_subprocesses(
        jobs,
        cwd=rt.ROOT_DIR,
        shard_totals=shard_totals,
    )
    if failed_shards:
        raise RuntimeError(f"并行 {label} 失败: {', '.join(failed_shards)}")

    for split in splits:
        shard_split_dirs = [shard_output_dir / split for shard_output_dir in shard_output_dirs if (shard_output_dir / split / "shard_result.json").exists()]
        if not shard_split_dirs:
            raise RuntimeError(f"并行 {label} 缺少 split={split} 的分片结果。")
        split_args = SimpleNamespace(**vars(args))
        split_args.requested_split = args.split
        split_args.split = split
        final_output_dir = final_output_root / split
        published_split_output_dir = published_output_root / split
        split_args._published_output_dir = published_split_output_dir
        published_report_path = build_split_report_path(
            args=split_args,
            output_dir=published_split_output_dir,
            split=split,
            checkpoint_path=checkpoint_path,
        )
        split_args.report_path = rebase_output_artifact_path(
            published_report_path,
            source_output_dir=published_split_output_dir,
            target_output_dir=final_output_dir,
        )
        merge_shard_results(
            shard_output_dirs=shard_split_dirs,
            final_output_dir=final_output_dir,
            checkpoint_path=checkpoint_path,
            data_cfg=data_cfg,
            split=split,
            args=split_args,
        )
        finalize_parallel_infer_output(
            args=split_args,
            checkpoint_path=checkpoint_path,
            data_cfg=data_cfg,
            final_output_dir=final_output_dir,
            split=split,
            input_mode="dataset",
            num_images=split_num_images.get(split, 0),
        )
    write_multi_split_confusion_inputs_index(
        output_root=final_output_root,
        args=args,
        data_cfg=data_cfg,
        splits=requested_splits,
        published_output_root=published_output_root,
    )
    shutil.rmtree(shard_root, ignore_errors=False)
    publish_stage(final_output_root, published_output_root, overwrite=True)
    args.output_dir = published_output_root
    finalize_published_multi_split_reports(
        args=args,
        checkpoint_path=checkpoint_path,
        output_root=published_output_root,
        splits=requested_splits,
    )
    print_published_eval_artifacts(published_output_root, args)
    return True



def _det_fingerprint_options(args) -> dict[str, Any]:
    return {
        "save_visualization": bool(getattr(args, "save_visualization", False)),
        "save_json": bool(getattr(args, "save_json", False)),
        "save_txt": bool(getattr(args, "save_txt", False)),
        "save_test_report": bool(getattr(args, "save_test_report", False)),
        "compute_metrics": bool(getattr(args, "compute_metrics", False)),
        "metric_classwise": bool(getattr(args, "metric_classwise", False)),
        "vis_max_images": getattr(args, "vis_max_images", None),
        "report_iou_threshold": getattr(args, "report_iou_threshold", None),
        "sahi": bool(getattr(args, "sahi", False)),
        "sahi_overlap": getattr(args, "sahi_overlap", rt.INFER_DEFAULT_SAHI_OVERLAP),
        "sahi_nms_iou": getattr(args, "sahi_nms_iou", rt.INFER_DEFAULT_SAHI_NMS_IOU),
        "sahi_global_local_iou": getattr(args, "sahi_global_local_iou", rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU),
        "sahi_skip_small": getattr(args, "sahi_skip_small", rt.INFER_DEFAULT_SAHI_SKIP_SMALL),
    }


def _det_reuse_precheck(args, output_dir: Path, checkpoint_path: Path) -> str:
    """构造 det 指纹并做复用前置检查，返回 run_reuse 的决策常量。"""
    data = getattr(args, "data", None)
    if data is not None:
        input_mode = "dataset"
    elif getattr(args, "image", None) is not None:
        input_mode = "image"
    else:
        input_mode = "image_dir"
    action = det_action(args)
    fingerprint = run_reuse.make_fingerprint(
        task="det", action=action, checkpoint=checkpoint_path,
        data=data, split=getattr(args, "split", None),
        threshold=getattr(args, "score_threshold", None), input_mode=input_mode,
        image=getattr(args, "image", None), image_dir=getattr(args, "image_dir", None),
        options=_det_fingerprint_options(args),
    )
    required = ["run_meta.json"]
    if action == "infer" and getattr(args, "save_visualization", False):
        required.append("images")
    if getattr(args, "save_json", False):
        required.append(f"{rt.INFER_TEMP_DIRNAME}/json")
    if getattr(args, "save_txt", False):
        required.append(f"{rt.INFER_TEMP_DIRNAME}/txt")
    if action == "eval":
        required.append("metrics_summary.json")
        report_path = build_split_report_path(
            args=args,
            output_dir=output_dir,
            split=getattr(args, "split", "test"),
            checkpoint_path=checkpoint_path,
        )
        try:
            required.append(str(report_path.relative_to(output_dir)))
        except ValueError:
            pass
    return run_reuse.precheck(
        args, output_dir, fingerprint, action_label=f"det/{action}", required=required
    )


def run_single_infer(args) -> None:
    action = det_action(args)
    use_dataset = args.data is not None
    compute_full_metrics = use_dataset and (args.compute_metrics or args.save_test_report)
    shard_child = is_shard_child(args)
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    args._resolved_checkpoint_path = checkpoint_path
    data_cfg = rt.load_data_config(args.data) if use_dataset else None
    output_dir = build_split_output_dir(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        split=args.split,
    )
    report_path = build_split_report_path(
        args=args,
        output_dir=output_dir,
        split=args.split,
        checkpoint_path=checkpoint_path,
    )
    run_meta_path = output_dir / "run_meta.json"
    temp_dir = rt.infer_temp_dir(output_dir)
    images_dir = output_dir / "images"
    json_dir = temp_dir / "json"
    txt_dir = temp_dir / "txt"

    args.output_dir = output_dir
    args.report_path = report_path

    samples, input_class_names, input_mode = get_input_samples(args)
    rt.ensure_image_samples(samples)
    if getattr(args, "dry_run", False):
        print_dry_run_summary(
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            report_path=report_path,
            run_meta_path=run_meta_path,
            input_mode=input_mode,
            args=args,
            data_cfg=data_cfg,
            num_images=len(samples),
        )
        return

    if not shard_child:
        if _det_reuse_precheck(args, args.output_dir, checkpoint_path) == run_reuse.REUSE:
            print(f"[{det_label(args)}] 已复用上次结果：{args.output_dir}")
            finalize_published_infer_reports(
                args=args,
                checkpoint_path=checkpoint_path,
                output_dir=args.output_dir,
            )
            print_published_eval_artifacts(args.output_dir, args)
            return
        published_output_dir = output_dir
        published_report_path = report_path
        output_dir = create_stage(published_output_dir, overwrite=args.overwrite)
        args._published_output_dir = published_output_dir
        args.output_dir = output_dir
        report_path = rebase_output_artifact_path(
            published_report_path,
            source_output_dir=published_output_dir,
            target_output_dir=output_dir,
        )
        args.report_path = report_path
        run_meta_path = output_dir / "run_meta.json"
        temp_dir = rt.infer_temp_dir(output_dir)
        images_dir = output_dir / "images"
        json_dir = temp_dir / "json"
        txt_dir = temp_dir / "txt"
    rt.prepare_output_dir(args.output_dir, args.overwrite, clean=True)
    dashboard_path: Path | None = None
    if not shard_child and not getattr(args, "skip_important_artifacts", False):
        dashboard_path = sync_important_artifacts(checkpoint_path)

    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    ensure_object_detection_model(model)
    class_names = rt.merge_class_names(rt.get_model_class_names(model), input_class_names)

    metric, label_mapping = None, {}
    if use_dataset:
        metric, label_mapping = maybe_create_metric(compute_full_metrics, class_names, args.metric_classwise)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Input mode: {input_mode}")
    print(f"Images to process: {len(samples)}")
    if dashboard_path is not None:
        print(f"training_dashboard copied to: {dashboard_path}")

    infer_time_sum_ms = 0.0
    metrics_meta = {"images_with_labels": 0, "images_without_labels": 0}
    legacy_class_data: dict[int, dict[str, Any]] = {}
    legacy_total_gt = 0
    legacy_total_pred = 0
    legacy_total_tp = 0
    metric_entries_path = args.output_dir / "metric_entries.jsonl" if shard_child and use_dataset else None
    vis_indices = visualization_indices(len(samples), args)
    with tempfile.TemporaryDirectory(prefix="lt_rgb_") as _rgb_scratch_str:
        _rgb_scratch = Path(_rgb_scratch_str)
        for sample_index, sample in enumerate(
            track(
                samples,
                label=f"{det_label(args)} 推理",
                total=len(samples),
                unit="img",
                shard_scope="inference",
            )
        ):
            with rt.Image.open(sample.image_path) as image:
                image_size = image.size
                predict_path = sample.image_path
                if image.mode != "RGB":
                    predict_path = _prepare_rgb_predict_path(image, sample, _rgb_scratch)

            infer_start = time.perf_counter()
            # 全量预测（供 mAP 用）；score 阈值过滤只作用于可视化/JSON/TXT/legacy 报表。
            prediction = predict_for_infer(model, predict_path, args)
            infer_time_sum_ms += (time.perf_counter() - infer_start) * 1000.0
            records = filter_records_by_score(
                prediction_records(prediction, class_names, image_size),
                args.score_threshold,
            )

            if sample.label_path is not None:
                label_path_cur = sample.label_path
            elif args.image_dir is not None:
                labels_dir = args.image_dir.expanduser().resolve().parent.parent / "labels" / args.image_dir.name
                label_path_cur = (labels_dir / sample.relative_path).with_suffix(".txt")
            else:
                label_path_cur = None
            gt_boxes, gt_labels, has_label = load_ground_truth(label_path_cur, image_size)

            if sample_index in vis_indices:
                gt_items_vis = gt_records(gt_boxes, gt_labels) if has_label else []
                if action == "infer":
                    vis_path = visualization_output_path(
                        images_dir,
                        sample,
                        rt.visualization_suffix(sample.image_path),
                        records,
                    )
                    draw_predictions(sample.image_path, vis_path, records, gt_items_vis, class_names)
                if has_label:
                    compare_path = comparison_output_path(
                        output_dir, sample, rt.visualization_suffix(sample.image_path)
                    )
                    draw_comparison(sample.image_path, compare_path, records, gt_items_vis, class_names)
            if args.save_json:
                save_prediction_json(json_dir / relative_output_path(sample, ".json"), sample, image_size, records)
            if args.save_txt:
                save_prediction_txt(txt_dir / relative_output_path(sample, ".txt"), records)

            if use_dataset:
                if has_label:
                    metrics_meta["images_with_labels"] += 1
                else:
                    metrics_meta["images_without_labels"] += 1
                update_metric(metric, label_mapping, prediction, gt_boxes, gt_labels)
                if shard_child:
                    metric_entry = build_metric_entry(prediction, gt_boxes, gt_labels)
                    assert metric_entries_path is not None
                    with metric_entries_path.open("a", encoding="utf-8") as stream:
                        stream.write(
                            json.dumps(metric_entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                        )
                if args.save_test_report:
                    gt_items = gt_records(gt_boxes, gt_labels)
                    image_total_gt, image_total_pred, image_total_tp = update_legacy_report_state(
                        legacy_class_data,
                        gt_items,
                        records,
                        args.report_iou_threshold,
                    )
                    legacy_total_gt += image_total_gt
                    legacy_total_pred += image_total_pred
                    legacy_total_tp += image_total_tp

    if shard_child:
        shard_result_path = write_shard_result(
            args.output_dir,
            split=args.split,
            class_names=class_names,
            num_images=len(samples),
            processed_images=len(samples),
            infer_time_sum_ms=infer_time_sum_ms,
            metrics_meta=metrics_meta,
            legacy_class_data=legacy_class_data,
            total_gt=legacy_total_gt,
            total_pred=legacy_total_pred,
            total_tp=legacy_total_tp,
            metric_entries_path=metric_entries_path,
        )
        print(f"shard_result saved to: {shard_result_path}")
        return

    aggregated_metric_values: dict[str, float] | None = None
    if use_dataset and metric is not None and metrics_meta["images_with_labels"] > 0:
        print(f"[{det_label(args)}] 阶段: 计算聚合指标", flush=True)
        aggregated_metric_values = metric.compute_aggregated_values().metric_values
        print(f"[{det_label(args)}] 阶段完成: 聚合指标", flush=True)
        metrics_payload = {
            "checkpoint": str(checkpoint_path),
            "input_mode": input_mode,
            "num_images": len(samples),
            **metrics_meta,
            "metrics": aggregated_metric_values,
            "avg_infer_time_ms": infer_time_sum_ms / max(len(samples), 1),
        }
        metrics_path = args.output_dir / "metrics_summary.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Metrics saved to: {metrics_path}")
    if use_dataset and args.save_test_report:
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
            metric_values=aggregated_metric_values,
            metrics_meta=metrics_meta,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"test_report saved to: {report_path}")
        if action == "infer" and args.save_visualization:
            export_bad_class_images(
                args=args,
                output_dir=args.output_dir,
                report_path=report_path,
                data_cfg=data_cfg,
                split=args.split,
                samples=samples,
            )
    if use_dataset and not shard_child:
        write_confusion_inputs_index(
            output_dir=args.output_dir,
            report_path=report_path,
            args=args,
            data_cfg=data_cfg,
            split=args.split,
            num_images=len(samples),
        )
    write_run_meta(
        run_meta_path,
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        report_path=report_path,
        input_mode=input_mode,
        args=args,
        data_cfg=data_cfg,
        num_images=len(samples),
        dashboard_path=dashboard_path,
    )
    print(f"run_meta saved to: {run_meta_path}")
    publish_stage(output_dir, published_output_dir, overwrite=args.overwrite)
    args.output_dir = published_output_dir
    args.report_path = published_report_path
    if use_dataset:
        finalize_published_infer_reports(
            args=args,
            checkpoint_path=checkpoint_path,
            output_dir=published_output_dir,
        )
    print_published_eval_artifacts(published_output_dir, args)
    delattr(args, "_published_output_dir")


def run_infer(args) -> None:
    if is_shard_child(args) and getattr(args, "selected_splits", None):
        run_multi_split_shard_infer(args)
        return

    if is_shard_child(args):
        run_single_infer(args)
        return

    if args.data is None:
        run_single_infer(args)
        return

    if getattr(args, "split", None) not in MULTI_SPLIT_MODES:
        if run_parallel_split_infer(args):
            return
        run_single_infer(args)
        return

    data_cfg = rt.load_data_config(args.data)
    splits = resolve_dataset_infer_splits(data_cfg, args.split)
    if run_parallel_all_infer(args, splits):
        return
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    output_root = build_multi_split_output_root(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
    )

    fallback_device: str | None = None
    if args.device == "auto":
        all_gpus, inventory_message = query_gpu_inventory()
        if inventory_message:
            print(f"[{det_label(args)}] {inventory_message}")
        eligible_gpus = filter_high_memory_gpus(all_gpus)
        fallback_device, fallback_message = select_fallback_single_gpu(all_gpus, eligible_gpus)
        print(fallback_message)

    total = len(splits)
    for index, split in enumerate(splits, start=1):
        single_args = SimpleNamespace(**vars(args))
        single_args.requested_split = args.split
        single_args.split = split
        single_args._defer_report_generation = True
        if fallback_device is not None:
            single_args.device = fallback_device
        print(f"\n=== [{index}/{total}] 开始 {det_action(args)} split: {split} ===")
        if run_parallel_split_infer(single_args):
            continue
        run_single_infer(single_args)
    if getattr(args, "dry_run", False):
        print(f"[dry-run] multi-split output root: {output_root}")
        return
    write_multi_split_confusion_inputs_index(
        output_root=output_root,
        args=args,
        data_cfg=data_cfg,
        splits=splits,
    )
    args.output_dir = output_root
    finalize_published_multi_split_reports(
        args=args,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        splits=splits,
    )
    print_published_eval_artifacts(output_root, args)


def run_eval(args) -> None:
    """运行检测评估，完整累计指标并限额保存抽样对比图。"""
    if getattr(args, "data", None) is None:
        raise ValueError("det eval requires --data.")
    args.tool_action = "eval"
    args.command = "eval"
    args.image = None
    args.image_dir = None
    args.save_json = False
    args.save_txt = False
    args.compute_metrics = True
    args.save_test_report = True
    args.metric_classwise = bool(
        getattr(args, "metric_classwise", getattr(args, "classwise", False))
    )
    args.vis_max_images = int(
        getattr(args, "vis_max_images", rt.DET_EVAL_VIS_MAX_IMAGES)
        or 0
    )
    if args.vis_max_images < 0:
        raise ValueError("det eval --vis-max-images 必须大于或等于 0。")
    run_infer(args)
