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


def save_semantic_visualization(image_path: Path, output_path: Path, mask_tensor: Any, class_names: dict[int, str]) -> None:
    with rt.Image.open(image_path) as image:
        image = image.convert("RGB")
        image_np = rt.np.array(image)
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
    image_paths = rt.list_image_files(args.image if args.image is not None else args.image_dir)
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


def run_eval(args) -> None:
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    rt.prepare_output_dir(args.output_dir, args.overwrite)
    data_cfg = rt.load_data_config(args.data)
    samples, data_class_names = rt.list_dataset_samples(data_cfg=data_cfg, split=args.split)
    rt.ensure_image_samples(samples)
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    class_names = rt.merge_class_names(rt.get_model_class_names(model), data_class_names)
    metric, label_mapping = create_metric(class_names, args.classwise)

    images_with_labels = 0
    for idx, sample in enumerate(samples, start=1):
        with rt.Image.open(sample.image_path) as image:
            image_size = image.size
        prediction = predict_model(model, sample.image_path, args.threshold)
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
                "split": args.split,
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
