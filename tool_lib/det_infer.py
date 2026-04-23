"""检测任务推理入口。

这里聚合 det infer 相关的入口函数和只被 infer 使用的辅助函数。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import common as rt
from .det_shared import (
    build_legacy_report,
    draw_predictions,
    gt_records,
    load_ground_truth,
    prediction_records,
    relative_output_path,
    update_legacy_report_state,
)
from .det_report import generate_report_for_infer_output


def get_input_samples(args) -> tuple[list[rt.ImageSample], dict[int, str], str]:
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image does not exist: {image_path}")
        return [rt.ImageSample(image_path=image_path, relative_path=Path(image_path.name))], {}, "image"
    if args.image_dir is not None:
        return rt.list_directory_samples(args.image_dir), {}, "dir"
    data_cfg = rt.load_data_config(args.data)
    samples, class_names = rt.list_dataset_samples(data_cfg=data_cfg, split=args.split)
    return samples, class_names, "dataset"

def ensure_object_detection_model(model: Any) -> None:
    class_name = model.__class__.__name__.lower()
    if "objectdetection" not in class_name:
        raise ValueError(f"Expected an object detection model, got '{model.__class__.__name__}'.")

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
    copied_curve_paths: list[Path],
) -> None:
    experiment_dir = rt.experiment_dir_from_checkpoint_path(checkpoint_path)
    payload = {
        "task": "det",
        "action": "infer",
        "created_at": rt.timestamp_now_iso(),
        "run_name": output_dir.name,
        "input_mode": input_mode,
        "split": getattr(args, "split", None),
        "num_images": num_images,
        "paths": {
            "output_dir": str(output_dir),
            "report_path": str(report_path),
            "checkpoint_path": str(checkpoint_path),
            "experiment_dir": str(experiment_dir),
            "data_yaml": str(args.data.expanduser().resolve()) if args.data is not None else None,
            "data_root": str(data_cfg["_root_dir"]) if data_cfg is not None else None,
            "image": str(args.image.expanduser().resolve()) if args.image is not None else None,
            "image_dir": str(args.image_dir.expanduser().resolve()) if args.image_dir is not None else None,
        },
        "settings": {
            "device": args.device,
            "score_threshold": args.score_threshold,
            "report_iou_threshold": args.report_iou_threshold,
            "save_visualization": args.save_visualization,
            "save_json": args.save_json,
            "save_txt": args.save_txt,
            "save_test_report": args.save_test_report,
            "compute_metrics": args.compute_metrics,
            "metric_classwise": args.metric_classwise,
            "overwrite": args.overwrite,
        },
        "artifacts": {
            "metrics_summary": str(output_dir / "metrics_summary.json") if args.data is not None else None,
            "training_curves": [str(path) for path in copied_curve_paths],
        },
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
    planned_artifacts = [
        run_meta_path,
        output_dir / "training_curve_loss.png",
        output_dir / "training_curve_map.png",
        output_dir / "training_curve_lr.png",
        output_dir / "training_dashboard.png",
    ]
    if args.save_visualization:
        planned_artifacts.append(output_dir / "images")
    if args.save_json:
        planned_artifacts.append(output_dir / "json")
    if args.save_txt:
        planned_artifacts.append(output_dir / "txt")
    if args.data is not None:
        planned_artifacts.append(output_dir / "metrics_summary.json")
    if args.save_test_report:
        planned_artifacts.append(report_path)
        planned_artifacts.append(rt.build_report_archive_path(report_path))

    print("[dry-run] detection infer plan")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  experiment_dir: {experiment_dir}")
    print(f"  input_mode: {input_mode}")
    print(f"  split: {getattr(args, 'split', None)}")
    print(f"  num_images: {num_images}")
    if data_cfg is not None:
        print(f"  data_yaml: {args.data.expanduser().resolve()}")
        print(f"  data_root: {Path(data_cfg['_root_dir'])}")
    if args.image is not None:
        print(f"  image: {args.image.expanduser().resolve()}")
    if args.image_dir is not None:
        print(f"  image_dir: {args.image_dir.expanduser().resolve()}")
    print(f"  output_dir: {output_dir}")
    print(f"  report_path: {report_path}")
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


def resolve_dataset_infer_splits(data_cfg: dict[str, Any], requested_split: str) -> list[str]:
    if requested_split != "all":
        return [requested_split]

    splits = [split for split in ("test", "val") if data_cfg.get(split)]
    if not splits:
        raise ValueError("data.yaml 中未找到可用于 infer 的 test 或 val 配置。")
    return splits


def build_split_output_dir(
    *,
    args,
    checkpoint_path: Path,
    data_cfg: dict[str, Any] | None,
    split: str,
) -> Path:
    split_mode = getattr(args, "requested_split", getattr(args, "split", None))
    if args.output_dir is not None:
        return args.output_dir / split if split_mode == "all" else args.output_dir

    dataset_dir = rt.DATASET_DIR if data_cfg is None else data_cfg["_root_dir"]
    return rt.derive_det_run_dir(
        checkpoint_path=checkpoint_path,
        dataset_dir=Path(dataset_dir),
        split=split,
    )


def build_split_report_path(*, args, output_dir: Path, split: str) -> Path:
    split_mode = getattr(args, "requested_split", getattr(args, "split", None))
    if args.report_path is None:
        return rt.build_det_report_path(output_dir)
    if split_mode != "all":
        return args.report_path
    stem = args.report_path.stem
    suffix = args.report_path.suffix or ".json"
    return args.report_path.with_name(f"{stem}-{split}{suffix}")


def run_single_infer(args) -> None:
    use_dataset = args.data is not None
    compute_full_metrics = use_dataset and (args.compute_metrics or args.save_test_report)
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    data_cfg = rt.load_data_config(args.data) if use_dataset else None
    output_dir = build_split_output_dir(
        args=args,
        checkpoint_path=checkpoint_path,
        data_cfg=data_cfg,
        split=args.split,
    )
    report_path = build_split_report_path(args=args, output_dir=output_dir, split=args.split)
    run_meta_path = output_dir / "run_meta.json"
    images_dir = output_dir / "images"
    json_dir = output_dir / "json"
    txt_dir = output_dir / "txt"

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

    rt.prepare_output_dir(args.output_dir, args.overwrite)
    copied_curve_paths = rt.copy_training_curve_artifacts(
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
    )

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
    for curve_path in copied_curve_paths:
        print(f"training_curve copied to: {curve_path}")

    write_run_meta(
        run_meta_path,
        checkpoint_path=checkpoint_path,
        output_dir=args.output_dir,
        report_path=report_path,
        input_mode=input_mode,
        args=args,
        data_cfg=data_cfg,
        num_images=len(samples),
        copied_curve_paths=copied_curve_paths,
    )
    print(f"run_meta saved to: {run_meta_path}")

    infer_time_sum_ms = 0.0
    metrics_meta = {"images_with_labels": 0, "images_without_labels": 0}
    legacy_class_data: dict[int, dict[str, Any]] = {}
    legacy_total_gt = 0
    legacy_total_pred = 0
    legacy_total_tp = 0
    for idx, sample in enumerate(samples, start=1):
        with rt.Image.open(sample.image_path) as image:
            image_size = image.size
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
        records = prediction_records(prediction, class_names, image_size)

        if sample.label_path is not None:
            label_path_cur = sample.label_path
        elif args.image_dir is not None:
            labels_dir = args.image_dir.expanduser().resolve().parent.parent / "labels" / args.image_dir.name
            label_path_cur = (labels_dir / sample.relative_path).with_suffix(".txt")
        else:
            label_path_cur = None
        gt_boxes, gt_labels, has_label = load_ground_truth(label_path_cur, image_size)

        if args.save_visualization:
            vis_path = images_dir / relative_output_path(sample, rt.visualization_suffix(sample.image_path))
            gt_items_vis = gt_records(gt_boxes, gt_labels) if has_label else []
            draw_predictions(sample.image_path, vis_path, records, gt_items_vis, class_names)
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

        if idx == 1 or idx % 20 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] processed: {sample.image_path}")

    aggregated_metric_values: dict[str, float] | None = None
    if use_dataset and metric is not None and metrics_meta["images_with_labels"] > 0:
        aggregated_metric_values = metric.compute_aggregated_values().metric_values
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
        archive_path = rt.archive_report_copy(report_path)
        print(f"test_report saved to: {report_path}")
        print(f"test_report archived to: {archive_path}")
        report_markdown_paths = generate_report_for_infer_output(
            experiment_dir=rt.experiment_dir_from_checkpoint_path(checkpoint_path),
            output_dir=args.output_dir,
        )
        for report_markdown_path in report_markdown_paths:
            print(f"single_report saved to: {report_markdown_path}")


def run_infer(args) -> None:
    if args.data is None or getattr(args, "split", None) != "all":
        run_single_infer(args)
        return

    data_cfg = rt.load_data_config(args.data)
    splits = resolve_dataset_infer_splits(data_cfg, args.split)
    total = len(splits)
    for index, split in enumerate(splits, start=1):
        single_args = SimpleNamespace(**vars(args))
        single_args.requested_split = args.split
        single_args.split = split
        print(f"\n=== [{index}/{total}] 开始推理 split: {split} ===")
        run_single_infer(single_args)
