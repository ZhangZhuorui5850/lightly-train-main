"""分割任务工具。

当前这个文件负责 seg 的非训练功能，主要包括：
- infer: 对单张图或目录做分割推理，并保存可视化结果
- eval: 在带标注的数据集上做分割评估

运行流程大致是：
1. 解析 checkpoint
2. 加载分割模型
3. 读取图片或数据集
4. 执行 predict
5. 保存可视化或评估结果

这里同时兼容语义分割和实例分割，但 eval 目前主要按实例分割流程实现。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import common as rt
from . import train_tools
from .seg_export import run_export  # noqa: F401  re-export to keep dispatch wiring simple


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (231, 76, 60),
        (52, 152, 219),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
    ]
    return palette[index % len(palette)]


def predict_model(model: Any, image_path: Path, threshold: float) -> Any:
    try:
        return model.predict(str(image_path), threshold=threshold)
    except TypeError:
        return model.predict(str(image_path))


def _normalize_seg_type(args: Any) -> str:
    value = str(getattr(args, "seg_train_type", "instance") or "instance").lower()
    if value not in {"instance", "semantic"}:
        raise ValueError("seg_train_type must be either 'instance' or 'semantic'.")
    return value


def _semantic_class_names(classes: dict[int, Any], ignore_classes: set[int]) -> dict[int, str]:
    names: dict[int, str] = {}
    for internal_id, class_id in enumerate(sorted(set(classes) - ignore_classes)):
        info = classes[class_id]
        names[internal_id] = str(info.get("name", class_id) if isinstance(info, dict) else info)
    return names


def _semantic_image_mask_samples(data_path: Path, split: str) -> tuple[list[tuple[Path, Path]], dict[int, str], int]:
    data_cfg = train_tools.load_semantic_segmentation_split_config(data_path, split)
    split_cfg = data_cfg[split]
    image_dir = Path(split_cfg["images"])
    mask_dir_or_file = str(split_cfg["masks"])
    mask_dir = Path(mask_dir_or_file)
    is_mask_dir = mask_dir.is_dir()
    classes = data_cfg["classes"]
    ignore_classes = {int(item) for item in data_cfg.get("ignore_classes", set()) or set()}
    class_names = _semantic_class_names(classes, ignore_classes)

    samples: list[tuple[Path, Path]] = []
    for rel_image in rt.file_helpers.list_image_filenames_from_dir(image_dir=image_dir):
        image_path = image_dir / Path(rel_image)
        if is_mask_dir:
            mask_path = (mask_dir / Path(rel_image)).with_suffix(".png")
        else:
            mask_path = Path(mask_dir_or_file.format(image_path=image_path))
        if mask_path.exists():
            samples.append((image_path, mask_path))
    return samples, class_names, len(classes)


def _seg_infer_image_paths(args: Any) -> list[Path]:
    data_path = getattr(args, "data", None)
    if data_path is not None:
        if _normalize_seg_type(args) == "semantic":
            split_cfg = train_tools.load_semantic_segmentation_split_config(Path(data_path), args.split)
            image_dir = Path(split_cfg[args.split]["images"])
            return [image_dir / Path(rel) for rel in rt.file_helpers.list_image_filenames_from_dir(image_dir=image_dir)]
        data_cfg = rt.load_data_config(Path(data_path))
        samples, _ = rt.list_dataset_samples(data_cfg=data_cfg, split=args.split)
        return [sample.image_path for sample in samples]
    return rt.list_image_files(args.image if args.image is not None else args.image_dir)


def _class_labels(classes: dict[int, Any], class_id: int) -> set[Any]:
    info = classes[class_id]
    if isinstance(info, str):
        return {class_id}
    if isinstance(info, dict):
        labels = info.get("labels", info.get("values"))
        if labels is None:
            return {class_id}
        normalized = set()
        for label in labels:
            normalized.add(tuple(label) if isinstance(label, list) else label)
        return normalized
    return {class_id}


def _load_semantic_mask(mask_path: Path, classes: dict[int, Any], ignore_classes: set[int]) -> Any:
    with rt.Image.open(mask_path) as mask_image:
        mask_np = rt.np.array(mask_image)
    compare_np = mask_np if mask_np.ndim == 3 else mask_np[:, :, None]
    target = rt.np.full(mask_np.shape[:2], -100, dtype=rt.np.int64)
    original_to_internal = {
        class_id: internal_id
        for internal_id, class_id in enumerate(sorted(set(classes) - ignore_classes))
    }
    for class_id, internal_id in original_to_internal.items():
        for label in _class_labels(classes, class_id):
            label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            target[rt.np.all(compare_np == rt.np.array(label_tuple), axis=2)] = internal_id
    return target


def _prediction_to_numpy(prediction: Any) -> Any:
    if isinstance(prediction, dict):
        if len(prediction) != 1:
            raise ValueError("Multihead semantic segmentation eval requires selecting one output head first.")
        prediction = next(iter(prediction.values()))
    return prediction.detach().cpu().numpy().astype(rt.np.int64)


def _resize_prediction_if_needed(pred_np: Any, target_shape: tuple[int, int]) -> Any:
    if tuple(pred_np.shape[-2:]) == target_shape:
        return pred_np
    image = rt.Image.fromarray(pred_np.astype(rt.np.int32), mode="I")
    resampling = getattr(rt.Image, "Resampling", rt.Image)
    resized = image.resize((target_shape[1], target_shape[0]), resample=resampling.NEAREST)
    return rt.np.array(resized).astype(rt.np.int64)


def _compute_semantic_iou(confusion: Any) -> tuple[dict[str, float], dict[str, dict[str, float | int]]]:
    per_class: dict[str, dict[str, float | int]] = {}
    ious: list[float] = []
    for class_id in range(confusion.shape[0]):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0
        if union > 0:
            ious.append(iou)
        per_class[str(class_id)] = {"tp": tp, "fp": fp, "fn": fn, "iou": iou}
    metrics = {
        "miou": float(sum(ious) / len(ious)) if ious else 0.0,
        "pixel_accuracy": float(confusion.diagonal().sum() / max(confusion.sum(), 1)),
    }
    return metrics, per_class


def save_semantic_visualization(image_path: Path, output_path: Path, mask_tensor: Any, class_names: dict[int, str]) -> None:
    with rt.Image.open(image_path) as image:
        image = image.convert("RGB")
        image_np = rt.np.array(image)
    if isinstance(mask_tensor, dict):
        if len(mask_tensor) != 1:
            raise ValueError("Multihead semantic segmentation visualization requires selecting one output head first.")
        mask_tensor = next(iter(mask_tensor.values()))
    mask_np = mask_tensor.detach().cpu().numpy()
    overlay = image_np.copy()
    for class_id in sorted(set(mask_np.reshape(-1).tolist())):
        color = color_for_index(int(class_id))
        overlay[mask_np == class_id] = (0.55 * overlay[mask_np == class_id] + 0.45 * rt.np.array(color)).astype(rt.np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rt.Image.fromarray(overlay).save(output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps({"classes": {str(k): class_names.get(k, str(k)) for k in sorted(class_names)}}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_instance_visualization(image_path: Path, output_path: Path, prediction: dict[str, Any], class_names: dict[int, str]) -> None:
    with rt.Image.open(image_path) as image:
        image = image.convert("RGB")
        image_np = rt.np.array(image)

    labels = prediction["labels"].detach().cpu().tolist()
    masks = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().tolist()
    overlay = image_np.copy()
    meta: list[dict[str, Any]] = []
    for idx, (label, mask, score) in enumerate(zip(labels, masks, scores)):
        color = rt.np.array(color_for_index(idx), dtype=rt.np.uint8)
        mask_bool = mask.astype(bool)
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(rt.np.uint8)
        meta.append({"class_id": int(label), "class_name": class_names.get(int(label), str(label)), "score": round(float(score), 6)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rt.Image.fromarray(overlay).save(output_path)
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_infer(args) -> None:
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    rt.prepare_output_dir(args.output_dir, args.overwrite)
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    class_names = rt.get_model_class_names(model)
    image_paths = _seg_infer_image_paths(args)
    if not image_paths:
        raise ValueError("没有找到可推理的图片。")
    for idx, image_path in enumerate(image_paths, start=1):
        prediction = predict_model(model, image_path, args.threshold)
        out_path = args.output_dir / image_path.name
        if isinstance(prediction, dict) and "masks" in prediction:
            save_instance_visualization(image_path, out_path, prediction, class_names)
        else:
            save_semantic_visualization(image_path, out_path, prediction, class_names)
        if idx == 1 or idx % 20 == 0 or idx == len(image_paths):
            print(f"[{idx}/{len(image_paths)}] processed: {image_path}")


def load_instance_ground_truth(label_path: Path | None, image_size: tuple[int, int]) -> tuple[dict[str, Any], bool]:
    width, height = image_size
    if label_path is None or not label_path.exists():
        return {"labels": rt.torch.zeros((0,), dtype=rt.torch.int64), "masks": rt.torch.zeros((0, height, width), dtype=rt.torch.bool)}, False
    polygons_np, _, class_labels_np = rt.file_helpers.open_yolo_instance_segmentation_label_numpy(label_path)
    binary_masks_np = rt.yolo_helpers.binary_masks_from_polygons(polygons_np, height=height, width=width)
    return {
        "labels": rt.torch.as_tensor(class_labels_np, dtype=rt.torch.int64),
        "masks": rt.torch.as_tensor(binary_masks_np, dtype=rt.torch.bool),
    }, True


def create_metric(class_names: dict[int, str], classwise: bool):
    if not class_names:
        raise ValueError("Instance segmentation metrics require class names from data.yaml.")
    mapping, metric_class_names = rt.build_metric_label_mapping(class_names)
    metric = rt.InstanceSegmentationTaskMetric(
        task_metric_args=rt.InstanceSegmentationTaskMetricArgs(
            watch_metric="eval_metric/map",
            classwise=classwise,
            train=False,
        ),
        split="eval",
        class_names=metric_class_names,
        loss_names=[],
        init_metrics=True,
    )
    return metric, mapping


def update_metric(metric, label_mapping: dict[int, int], prediction: dict[str, Any], target: dict[str, Any]) -> None:
    pred_labels = rt.remap_labels(prediction["labels"].detach().cpu().to(rt.torch.int64), label_mapping)
    pred_masks = prediction["masks"].detach().cpu().to(rt.torch.bool)
    pred_scores = prediction["scores"].detach().cpu().to(rt.torch.float32)
    gt_labels = rt.remap_labels(target["labels"].to(rt.torch.int64), label_mapping)
    gt_masks = target["masks"].to(rt.torch.bool)
    metric.update_with_predictions(
        preds=[{"labels": pred_labels, "masks": pred_masks, "scores": pred_scores}],
        target=[{"labels": gt_labels, "masks": gt_masks}],
    )


def _normalize_splits(split: Any) -> list[str]:
    """把 --split 统一成列表，兼容 CLI 多值 (list) 与交互式单值 (str)。"""
    if isinstance(split, (list, tuple)):
        return [str(item) for item in split]
    return [str(split)]


def _print_semantic_metrics(
    split: str,
    metrics: dict[str, float],
    per_class: dict[str, dict[str, float | int]],
    class_names: dict[int, str],
) -> None:
    """把单个 split 的 mIoU / Pixel Accuracy / 逐类 IoU 打印到控制台。"""
    print(f"\n  {'=' * 50}")
    print(f"    [{split}] 评估结果")
    print(f"  {'=' * 50}")
    print(f"    mIoU          : {metrics['miou']:.4f}")
    print(f"    Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  {'=' * 50}")
    print(f"    逐类 IoU:")
    for class_id_str, info in per_class.items():
        class_name = class_names.get(int(class_id_str), f"class_{class_id_str}")
        print(f"      {class_name:20s}  IoU={info['iou']:.4f}  TP={info['tp']}  FP={info['fp']}  FN={info['fn']}")


def _evaluate_semantic_split(
    model: Any,
    data_path: Path,
    split: str,
    output_dir: Path,
    threshold: float,
    overwrite: bool,
    checkpoint_path: Path,
) -> dict[str, Any] | None:
    """对单个 split 执行语义分割评估，保存 summary/CSV 并打印指标。"""
    # 先查 yaml 里是否定义了该 split；未定义就跳过（多 split 评估时常见，
    # 不能让 load_semantic_segmentation_split_config 直接抛异常中断整轮评估）。
    if train_tools._load_yaml_dict(data_path).get(split) is None:
        print(f"  ⚠ 数据配置中没有 '{split}' split，跳过。")
        return None
    data_cfg = train_tools.load_semantic_segmentation_split_config(data_path, split)
    samples, data_class_names, num_classes = _semantic_image_mask_samples(data_path, split)
    if not samples:
        print(f"  ⚠ {split} split 中没有找到图片/掩码对，跳过。")
        return None

    classes = data_cfg["classes"]
    ignore_classes = {int(item) for item in data_cfg.get("ignore_classes", set()) or set()}
    class_names = rt.merge_class_names(rt.get_model_class_names(model), data_class_names)
    confusion = rt.np.zeros((len(class_names), len(class_names)), dtype=rt.np.int64)
    rows: list[dict[str, Any]] = []

    print(f"\n  [{split}] 找到 {len(samples)} 个图片/掩码对")
    for idx, (image_path, mask_path) in enumerate(samples, start=1):
        prediction = predict_model(model, image_path, threshold)
        pred_np = _prediction_to_numpy(prediction)
        target_np = _load_semantic_mask(mask_path, classes, ignore_classes)
        pred_np = _resize_prediction_if_needed(pred_np, target_np.shape)
        valid = target_np != -100
        valid &= pred_np >= 0
        valid &= pred_np < len(class_names)
        if valid.any():
            bincount = rt.np.bincount(
                len(class_names) * target_np[valid].astype(rt.np.int64) + pred_np[valid].astype(rt.np.int64),
                minlength=len(class_names) ** 2,
            )
            confusion += bincount.reshape((len(class_names), len(class_names)))
        rows.append({"image_path": str(image_path), "mask_path": str(mask_path), "valid_pixels": int(valid.sum())})
        if idx == 1 or idx % 20 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] processed: {image_path}")

    metrics, per_class = _compute_semantic_iou(confusion)
    rt.prepare_output_dir(output_dir, overwrite)
    summary_path = output_dir / "seg_semantic_eval_summary.json"
    csv_path = output_dir / "seg_semantic_eval_samples.csv"
    rt.save_records_csv(csv_path, rows, ["image_path", "mask_path", "valid_pixels"])
    summary = {
        "task": "semantic_segmentation",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "split": split,
        "num_images": len(samples),
        "num_classes": num_classes,
        "class_names": class_names,
        "metrics": metrics,
        "per_class": per_class,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_semantic_metrics(split, metrics, per_class, class_names)
    print(f"Summary saved to: {summary_path}")
    print(f"CSV saved to: {csv_path}")
    return summary


def run_semantic_eval(args) -> None:
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    data_path = Path(args.data).expanduser().resolve()
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()

    splits = _normalize_splits(args.split)
    summaries: dict[str, Any] = {}
    for split in splits:
        # 多 split 时各自写入子目录；单 split 时保持原有的直接写入输出目录。
        split_output_dir = args.output_dir / split if len(splits) > 1 else args.output_dir
        # 语义分割是逐像素 argmax，没有可过滤的"检测列表"；阈值过滤只会人为
        # 砍掉低分像素、压低 mIoU，与训练内部验证不一致。故恒用 0.0（不过滤），
        # 与实例分割 eval 强制 0.0 的做法一致。args.threshold 对语义分割无效。
        summary = _evaluate_semantic_split(
            model=model,
            data_path=data_path,
            split=split,
            output_dir=split_output_dir,
            threshold=0.0,
            overwrite=args.overwrite,
            checkpoint_path=checkpoint_path,
        )
        if summary is not None:
            summaries[split] = summary

    if not summaries:
        raise ValueError("No image/mask pairs found for semantic segmentation eval.")


def run_eval(args) -> None:
    if _normalize_seg_type(args) == "semantic":
        run_semantic_eval(args)
        return

    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    rt.prepare_output_dir(args.output_dir, args.overwrite)
    # --split 现在可能是多值列表（语义分割需要），实例评估只取第一个。
    split = _normalize_splits(args.split)[0]
    data_cfg = rt.load_data_config(args.data)
    samples, data_class_names = rt.list_dataset_samples(data_cfg=data_cfg, split=split)
    rt.ensure_image_samples(samples)
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    class_names = rt.merge_class_names(rt.get_model_class_names(model), data_class_names)
    metric, label_mapping = create_metric(class_names, args.classwise)

    images_with_labels = 0
    for idx, sample in enumerate(samples, start=1):
        with rt.Image.open(sample.image_path) as image:
            image_size = image.size
        # mAP 必须吃全量带分预测：阈值过滤会截断 PR 曲线、人为压低指标，
        # 与训练内部验证（不做阈值过滤）不一致。故评估恒用 0.0。
        prediction = predict_model(model, sample.image_path, 0.0)
        if not isinstance(prediction, dict) or "masks" not in prediction:
            raise ValueError("当前 seg eval 只支持实例分割模型。")
        target, has_label = load_instance_ground_truth(sample.label_path, image_size)
        if has_label:
            images_with_labels += 1
        update_metric(metric, label_mapping, prediction, target)
        if idx == 1 or idx % 20 == 0 or idx == len(samples):
            print(f"[{idx}/{len(samples)}] processed: {sample.image_path}")

    result = metric.compute_aggregated_values().metric_values
    summary_path = args.output_dir / "seg_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "data": str(args.data),
                "split": split,
                "num_images": len(samples),
                "images_with_labels": images_with_labels,
                "metrics": result,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary saved to: {summary_path}")
