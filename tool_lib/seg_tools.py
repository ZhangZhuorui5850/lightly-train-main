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
import sys
import time
from pathlib import Path
from typing import Any

from . import common as rt
from . import gpu_parallel
from . import train_tools
from .seg_export import run_export  # noqa: F401  re-export to keep dispatch wiring simple
from .seg_shared import rle_decode, rle_encode


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
    original_to_internal = {
        class_id: internal_id
        for internal_id, class_id in enumerate(sorted(set(classes) - ignore_classes))
    }

    # 收集 (label_tuple, internal_id)；单通道标签长度 1，RGB 长度 3。
    single_pairs: list[tuple[int, int]] = []
    rgb_pairs: list[tuple[tuple[int, int, int], int]] = []
    for class_id, internal_id in original_to_internal.items():
        for label in _class_labels(classes, class_id):
            values = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            if len(values) == 1:
                single_pairs.append((values[0], internal_id))
            else:
                rgb_pairs.append((tuple(int(v) for v in values[:3]), internal_id))

    if mask_np.ndim == 2 and single_pairs and not rgb_pairs:
        max_label = max(label for label, _ in single_pairs)
        lut = rt.np.full(max_label + 1, -100, dtype=rt.np.int64)
        for label, internal_id in single_pairs:
            lut[label] = internal_id
        clipped = rt.np.clip(mask_np, 0, max_label)
        target = rt.np.where(mask_np <= max_label, lut[clipped], -100).astype(rt.np.int64)
        return target

    # RGB（或混合）：打包成 int 后用排序键映射。
    compare_np = mask_np if mask_np.ndim == 3 else rt.np.repeat(mask_np[:, :, None], 3, axis=2)
    packed = (
        compare_np[:, :, 0].astype(rt.np.int64) << 16
    ) | (compare_np[:, :, 1].astype(rt.np.int64) << 8) | compare_np[:, :, 2].astype(rt.np.int64)
    keys: list[int] = []
    vals: list[int] = []
    for (r, g, b), internal_id in rgb_pairs:
        keys.append((r << 16) | (g << 8) | b)
        vals.append(internal_id)
    for label, internal_id in single_pairs:
        keys.append((label << 16) | (label << 8) | label)
        vals.append(internal_id)
    target = rt.np.full(mask_np.shape[:2], -100, dtype=rt.np.int64)
    if keys:
        keys_arr = rt.np.array(keys, dtype=rt.np.int64)
        vals_arr = rt.np.array(vals, dtype=rt.np.int64)
        order = rt.np.argsort(keys_arr)
        keys_sorted = keys_arr[order]
        vals_sorted = vals_arr[order]
        idx = rt.np.searchsorted(keys_sorted, packed)
        idx_clipped = rt.np.clip(idx, 0, len(keys_sorted) - 1)
        match = keys_sorted[idx_clipped] == packed
        target = rt.np.where(match, vals_sorted[idx_clipped], -100).astype(rt.np.int64)
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


def _serialize_instance_entry(prediction: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    pred_masks = prediction["masks"].detach().cpu().numpy().astype(bool)
    gt_masks = target["masks"].detach().cpu().numpy().astype(bool)
    return {
        "pred_labels": prediction["labels"].detach().cpu().to(rt.torch.int64).tolist(),
        "pred_scores": prediction["scores"].detach().cpu().to(rt.torch.float32).tolist(),
        "pred_masks_rle": [rle_encode(m) for m in pred_masks],
        "gt_labels": target["labels"].detach().cpu().to(rt.torch.int64).tolist(),
        "gt_masks_rle": [rle_encode(m) for m in gt_masks],
    }


def _stack_rle(rle_list: list[dict], *, height_width: tuple[int, int] | None) -> Any:
    if rle_list:
        masks = rt.np.stack([rle_decode(rle) for rle in rle_list])
        return rt.torch.as_tensor(masks, dtype=rt.torch.bool)
    h, w = height_width if height_width is not None else (1, 1)
    return rt.torch.zeros((0, h, w), dtype=rt.torch.bool)


def _deserialize_instance_entry(entry: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    all_rle = entry.get("pred_masks_rle", []) + entry.get("gt_masks_rle", [])
    hw = (all_rle[0]["size"][0], all_rle[0]["size"][1]) if all_rle else None
    prediction = {
        "labels": rt.torch.as_tensor(entry["pred_labels"], dtype=rt.torch.int64),
        "scores": rt.torch.as_tensor(entry["pred_scores"], dtype=rt.torch.float32),
        "masks": _stack_rle(entry["pred_masks_rle"], height_width=hw),
    }
    target = {
        "labels": rt.torch.as_tensor(entry["gt_labels"], dtype=rt.torch.int64),
        "masks": _stack_rle(entry["gt_masks_rle"], height_width=hw),
    }
    return prediction, target


def _is_seg_shard_child(args: Any) -> bool:
    """子进程分片模式判定：显式给了 shard_index 且 num_shards>1 才算分片子进程。"""
    return getattr(args, "shard_index", None) is not None and int(getattr(args, "num_shards", 1) or 1) > 1


def _write_instance_shard_result(
    shard_dir: Path,
    *,
    split: str,
    class_names: dict[int, str],
    entries: list[dict[str, Any]],
    images_with_labels: int,
    num_images: int,
    infer_time_sum_ms: float,
    failed: int,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "class_names": {str(k): v for k, v in class_names.items()},
        "entries": entries,
        "images_with_labels": int(images_with_labels),
        "num_images": int(num_images),
        "infer_time_sum_ms": float(infer_time_sum_ms),
        "failed": int(failed),
    }
    path = shard_dir / "seg_instance_shard_result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _merge_instance_shard_results(shard_dirs: list[Path], *, classwise: bool) -> dict[str, Any]:
    class_names: dict[int, str] = {}
    entries: list[dict[str, Any]] = []
    num_images = 0
    images_with_labels = 0
    infer_time_sum_ms = 0.0
    failed = 0
    for shard_dir in shard_dirs:
        payload = json.loads((shard_dir / "seg_instance_shard_result.json").read_text(encoding="utf-8"))
        for cid_raw, name in payload.get("class_names", {}).items():
            class_names[int(cid_raw)] = str(name)
        entries.extend(payload.get("entries", []))
        num_images += int(payload.get("num_images", 0))
        images_with_labels += int(payload.get("images_with_labels", 0))
        infer_time_sum_ms += float(payload.get("infer_time_sum_ms", 0.0))
        failed += int(payload.get("failed", 0))
    metric, label_mapping = create_metric(class_names, classwise)
    for entry in entries:
        prediction, target = _deserialize_instance_entry(entry)
        update_metric(metric, label_mapping, prediction, target)
    metric_values = metric.compute_aggregated_values().metric_values
    return {
        "metric_values": metric_values,
        "class_names": class_names,
        "num_images": num_images,
        "images_with_labels": images_with_labels,
        "infer_time_sum_ms": infer_time_sum_ms,
        "failed": failed,
    }


def _accumulate_instance_eval(
    model: Any,
    data_cfg: Any,
    split: str,
    classwise: bool,
    *,
    shard_index: int | None = None,
    num_shards: int = 1,
) -> dict[str, Any]:
    """加载样本并累积实例分割评估，可分片、带容错与计时。

    返回 metric/label_mapping（供非分片路径直接 compute）以及每图序列化 entry
    （供分片路径写盘后由父进程合并）。
    """
    samples, data_class_names = rt.list_dataset_samples(data_cfg=data_cfg, split=split)
    rt.ensure_image_samples(samples)
    class_names = rt.merge_class_names(rt.get_model_class_names(model), data_class_names)
    metric, label_mapping = create_metric(class_names, classwise)

    shard_indices = gpu_parallel.filter_indices_for_shard(
        len(samples), shard_index=shard_index, num_shards=num_shards
    )
    shard_samples = [samples[i] for i in shard_indices]

    entries: list[dict[str, Any]] = []
    images_with_labels = 0
    infer_time_sum_ms = 0.0
    failed = 0
    for idx, sample in enumerate(shard_samples, start=1):
        with rt.Image.open(sample.image_path) as image:
            image_size = image.size
        try:
            start = time.perf_counter()
            # mAP 必须吃全量带分预测：阈值过滤会截断 PR 曲线、人为压低指标，
            # 与训练内部验证（不做阈值过滤）不一致。故评估恒用 0.0。
            prediction = predict_model(model, sample.image_path, 0.0)
            infer_time_sum_ms += (time.perf_counter() - start) * 1000.0
        except Exception as exc:  # noqa: BLE001 单图失败不应中断整轮评估
            failed += 1
            print(f"  ⚠ 跳过 {sample.image_path}: {exc}")
            continue
        if not isinstance(prediction, dict) or "masks" not in prediction:
            raise ValueError("当前 seg eval 只支持实例分割模型。")
        target, has_label = load_instance_ground_truth(sample.label_path, image_size)
        if has_label:
            images_with_labels += 1
        update_metric(metric, label_mapping, prediction, target)
        entries.append(_serialize_instance_entry(prediction, target))
        if idx == 1 or idx % 20 == 0 or idx == len(shard_samples):
            print(f"[{idx}/{len(shard_samples)}] processed: {sample.image_path}")

    return {
        "class_names": class_names,
        "entries": entries,
        "metric": metric,
        "label_mapping": label_mapping,
        "num_images": len(shard_samples),
        "images_with_labels": images_with_labels,
        "infer_time_sum_ms": infer_time_sum_ms,
        "failed": failed,
    }


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


def _write_semantic_shard_result(
    shard_dir: Path,
    *,
    split: str,
    confusion: Any,
    rows: list[dict[str, Any]],
    class_names: dict[int, str],
    num_samples: int,
    infer_time_sum_ms: float,
    failed: int,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "confusion": confusion.astype(int).tolist(),
        "rows": rows,
        "class_names": {str(k): v for k, v in class_names.items()},
        "num_samples": int(num_samples),
        "infer_time_sum_ms": float(infer_time_sum_ms),
        "failed": int(failed),
    }
    path = shard_dir / "seg_semantic_shard_result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _merge_semantic_shard_results(shard_dirs: list[Path], *, split: str) -> dict[str, Any]:
    confusion = None
    rows: list[dict[str, Any]] = []
    class_names: dict[int, str] = {}
    num_samples = 0
    infer_time_sum_ms = 0.0
    failed = 0
    for shard_dir in shard_dirs:
        payload = json.loads((shard_dir / "seg_semantic_shard_result.json").read_text(encoding="utf-8"))
        mat = rt.np.array(payload["confusion"], dtype=rt.np.int64)
        confusion = mat if confusion is None else confusion + mat
        rows.extend(payload.get("rows", []))
        for cid_raw, name in payload.get("class_names", {}).items():
            class_names[int(cid_raw)] = str(name)
        num_samples += int(payload.get("num_samples", 0))
        infer_time_sum_ms += float(payload.get("infer_time_sum_ms", 0.0))
        failed += int(payload.get("failed", 0))
    return {
        "confusion": confusion,
        "rows": rows,
        "class_names": class_names,
        "num_samples": num_samples,
        "infer_time_sum_ms": infer_time_sum_ms,
        "failed": failed,
    }


def _accumulate_semantic_confusion(
    model: Any,
    data_path: Path,
    split: str,
    threshold: float,
    *,
    shard_index: int | None = None,
    num_shards: int = 1,
) -> dict[str, Any] | None:
    """加载样本并累积混淆矩阵，可分片、带容错与计时。无样本/无该 split 时返回 None。"""
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
    infer_time_sum_ms = 0.0
    failed = 0

    shard_indices = gpu_parallel.filter_indices_for_shard(
        len(samples), shard_index=shard_index, num_shards=num_shards
    )
    shard_samples = [samples[i] for i in shard_indices]

    print(f"\n  [{split}] 找到 {len(samples)} 个图片/掩码对" + (
        f"（本分片 {len(shard_samples)} 个）" if len(shard_samples) != len(samples) else ""
    ))
    for idx, (image_path, mask_path) in enumerate(shard_samples, start=1):
        try:
            start = time.perf_counter()
            prediction = predict_model(model, image_path, threshold)
            infer_time_sum_ms += (time.perf_counter() - start) * 1000.0
        except Exception as exc:  # noqa: BLE001 单图失败不应中断整轮评估
            failed += 1
            print(f"  ⚠ 跳过 {image_path}: {exc}")
            continue
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
        if idx == 1 or idx % 20 == 0 or idx == len(shard_samples):
            print(f"[{idx}/{len(shard_samples)}] processed: {image_path}")

    return {
        "confusion": confusion,
        "rows": rows,
        "class_names": class_names,
        "num_classes": num_classes,
        "num_samples": len(rows),
        "infer_time_sum_ms": infer_time_sum_ms,
        "failed": failed,
    }


def _write_semantic_summary(
    output_dir: Path,
    *,
    split: str,
    confusion: Any,
    rows: list[dict[str, Any]],
    class_names: dict[int, str],
    num_classes: int,
    num_samples: int,
    infer_time_sum_ms: float,
    failed: int,
    overwrite: bool,
    checkpoint_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    """计算 IoU、写 summary/CSV 并打印指标。顺序路径与并行合并路径共用，保证产物字节一致。"""
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
        "num_images": num_samples,
        "num_classes": num_classes,
        "class_names": class_names,
        "metrics": metrics,
        "per_class": per_class,
        "avg_infer_time_ms": infer_time_sum_ms / max(num_samples, 1),
        "failed_images": failed,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _print_semantic_metrics(split, metrics, per_class, class_names)
    print(f"Summary saved to: {summary_path}")
    print(f"CSV saved to: {csv_path}")
    return summary


def _evaluate_semantic_split(
    model: Any,
    data_path: Path,
    split: str,
    output_dir: Path,
    threshold: float,
    overwrite: bool,
    checkpoint_path: Path,
    *,
    shard_index: int | None = None,
    num_shards: int = 1,
) -> dict[str, Any] | None:
    """对单个 split 执行语义分割评估，保存 summary/CSV 并打印指标。

    分片子进程模式（shard_index 给定且 num_shards>1）下不写最终 summary，
    而是把本分片的混淆矩阵/rows 写成 shard result，留给父进程合并。
    """
    accumulated = _accumulate_semantic_confusion(
        model, data_path, split, threshold, shard_index=shard_index, num_shards=num_shards
    )
    if accumulated is None:
        return None

    confusion = accumulated["confusion"]
    rows = accumulated["rows"]
    class_names = accumulated["class_names"]
    num_classes = accumulated["num_classes"]
    num_samples = accumulated["num_samples"]
    infer_time_sum_ms = accumulated["infer_time_sum_ms"]
    failed = accumulated["failed"]

    if shard_index is not None and int(num_shards or 1) > 1:
        # 分片子进程：每个 split 写到 output_dir/<split>/，父进程按 split 合并。
        _write_semantic_shard_result(
            output_dir / split,
            split=split,
            confusion=confusion,
            rows=rows,
            class_names=class_names,
            num_samples=num_samples,
            infer_time_sum_ms=infer_time_sum_ms,
            failed=failed,
        )
        return None

    return _write_semantic_summary(
        output_dir,
        split=split,
        confusion=confusion,
        rows=rows,
        class_names=class_names,
        num_classes=num_classes,
        num_samples=num_samples,
        infer_time_sum_ms=infer_time_sum_ms,
        failed=failed,
        overwrite=overwrite,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
    )


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
        # 分片子进程：每个 split 写到 args.output_dir/<split>/（无论单/多 split），
        # 父进程统一按 split 在该目录下合并。故 shard 模式不走多 split 子目录分支。
        if _is_seg_shard_child(args):
            summary = _evaluate_semantic_split(
                model=model,
                data_path=data_path,
                split=split,
                output_dir=args.output_dir,
                threshold=0.0,
                overwrite=args.overwrite,
                checkpoint_path=checkpoint_path,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
            )
            continue
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

    if _is_seg_shard_child(args):
        return
    if not summaries:
        raise ValueError("No image/mask pairs found for semantic segmentation eval.")


def _build_seg_eval_child_command(args, *, shard_index, num_shards, output_dir, device="auto") -> list[str]:
    command = [sys.executable, str(rt.ROOT_DIR / "launcher.py"), "eval", "--task", "seg"]
    command += ["--seg-train-type", str(getattr(args, "seg_train_type", "instance"))]
    if getattr(args, "experiment_dir", None) is not None:
        command += ["--experiment-dir", str(args.experiment_dir)]
    if getattr(args, "checkpoint", None) is not None:
        command += ["--checkpoint", str(args.checkpoint)]
    command += ["--data", str(args.data)]
    # --split 是 nargs="+"：必须用单个多值标志（--split val test），
    # 重复 --split 会被 argparse 覆盖、只保留最后一个，导致多 split 静默丢失。
    command += ["--split", *_normalize_splits(args.split)]
    command += ["--output-dir", str(output_dir), "--device", device]
    command += ["--shard-index", str(shard_index), "--num-shards", str(num_shards)]
    command += ["--skip-important-artifacts"]
    if getattr(args, "classwise", False):
        command += ["--classwise"]
    if getattr(args, "overwrite", False):
        command += ["--overwrite"]
    return command


def run_parallel_seg_eval(args) -> bool:
    if getattr(args, "device", "auto") != "auto":
        return False
    if _is_seg_shard_child(args) or getattr(args, "dry_run", False):
        return False
    all_gpus, message = gpu_parallel.query_gpu_inventory()
    if message:
        print(f"[seg/eval] {message}")
        return False
    eligible = gpu_parallel.filter_high_memory_gpus(all_gpus)
    if len(eligible) < 2:
        print(f"[seg/eval] 可用 GPU 数量为 {len(eligible)}，进入单卡顺序模式。")
        return False
    return _run_parallel_seg_eval_impl(args, eligible)


def _run_parallel_seg_eval_impl(args, eligible) -> bool:
    semantic = _normalize_seg_type(args) == "semantic"
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    data_path = Path(args.data).expanduser().resolve()
    # args.output_dir 在 parse_cli_args 阶段已默认成 <experiment_dir>/eval，
    # 与顺序路径一致；这里直接沿用，保证并行/顺序写到同一处。
    final_output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else rt.experiment_dir_from_checkpoint_path(checkpoint_path) / "eval"
    )
    splits = _normalize_splits(args.split)
    rt.prepare_output_dir(final_output_dir, args.overwrite)

    num_shards = len(eligible)
    shard_root = final_output_dir / "_shards"
    shard_dirs = [shard_root / f"shard_{i:02d}" for i in range(num_shards)]

    jobs: list[tuple[int, int, list[str]]] = []
    for i, gpu in enumerate(eligible):
        command = _build_seg_eval_child_command(
            args, shard_index=i, num_shards=num_shards, output_dir=shard_dirs[i], device="auto"
        )
        jobs.append((i, int(gpu["index"]), command))

    print("[seg/eval] 已进入自动多卡并行模式。")
    for gpu in eligible:
        print(f"[seg/eval] 保留 {gpu_parallel.format_gpu_summary(gpu)}")
    for shard_index, gpu_index, _ in jobs:
        print(f"[seg/eval] shard={shard_index}/{num_shards} -> CUDA_VISIBLE_DEVICES={gpu_index}")

    failed = gpu_parallel.run_sharded_subprocesses(jobs, cwd=rt.ROOT_DIR)
    if failed:
        raise RuntimeError(f"seg eval shard 失败: {', '.join(failed)}")

    if semantic:
        _merge_parallel_semantic(
            args, shard_dirs, splits=splits, final_output_dir=final_output_dir,
            checkpoint_path=checkpoint_path, data_path=data_path,
        )
    else:
        _merge_parallel_instance(
            args, shard_dirs, split=splits[0], final_output_dir=final_output_dir,
            checkpoint_path=checkpoint_path,
        )
    return True


def _merge_parallel_semantic(
    args, shard_dirs, *, splits, final_output_dir, checkpoint_path, data_path
) -> None:
    """按 split 合并各分片的语义混淆矩阵，写出与顺序路径字节一致的 summary/CSV。"""
    wrote_any = False
    for split in splits:
        # 每个子进程把该 split 写到 shard_dir/<split>/seg_semantic_shard_result.json。
        split_shard_dirs = [
            sd / split for sd in shard_dirs
            if (sd / split / "seg_semantic_shard_result.json").exists()
        ]
        if not split_shard_dirs:
            continue
        merged = _merge_semantic_shard_results(split_shard_dirs, split=split)
        # num_classes 不在 shard result 里，从数据配置取（与顺序路径一致：len(classes)）。
        _, _, num_classes = _semantic_image_mask_samples(data_path, split)
        split_output_dir = final_output_dir / split if len(splits) > 1 else final_output_dir
        _write_semantic_summary(
            split_output_dir,
            split=split,
            confusion=merged["confusion"],
            rows=merged["rows"],
            class_names=merged["class_names"],
            num_classes=num_classes,
            num_samples=merged["num_samples"],
            infer_time_sum_ms=merged["infer_time_sum_ms"],
            failed=merged["failed"],
            overwrite=args.overwrite,
            checkpoint_path=checkpoint_path,
            data_path=data_path,
        )
        wrote_any = True
    if not wrote_any:
        raise ValueError("No image/mask pairs found for semantic segmentation eval.")


def _merge_parallel_instance(args, shard_dirs, *, split, final_output_dir, checkpoint_path) -> None:
    """合并各分片的实例分割 entry，重算全局 mAP，写出与顺序路径一致的 summary。"""
    split_shard_dirs = [
        sd for sd in shard_dirs if (sd / "seg_instance_shard_result.json").exists()
    ]
    merged = _merge_instance_shard_results(split_shard_dirs, classwise=args.classwise)
    summary_path = final_output_dir / "seg_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "data": str(args.data),
                "split": split,
                "num_images": merged["num_images"],
                "images_with_labels": merged["images_with_labels"],
                "metrics": merged["metric_values"],
                "avg_infer_time_ms": merged["infer_time_sum_ms"] / max(merged["num_images"], 1),
                "failed_images": merged["failed"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary saved to: {summary_path}")


def run_eval(args) -> None:
    seg_type = _normalize_seg_type(args)
    # 自动多卡：device=auto 且 ≥2 张空闲卡时分片到子进程，否则回退顺序模式（返回 False）。
    # 子进程通过 launcher.py eval 重入本函数，_is_seg_shard_child 为真而跳过并行、直走分片写盘。
    if not _is_seg_shard_child(args) and run_parallel_seg_eval(args):
        return

    if seg_type == "semantic":
        run_semantic_eval(args)
        return

    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    rt.prepare_output_dir(args.output_dir, args.overwrite)
    # --split 现在可能是多值列表（语义分割需要），实例评估只取第一个。
    split = _normalize_splits(args.split)[0]
    data_cfg = rt.load_data_config(args.data)
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()

    # 分片子进程：只跑本分片、把序列化预测/标注写盘，由父进程合并出全局 mAP。
    # mAP 是全局指标，逐分片各算各的再平均是错的，所以分片模式绝不在此计算最终指标。
    if _is_seg_shard_child(args):
        accumulated = _accumulate_instance_eval(
            model, data_cfg, split, args.classwise,
            shard_index=args.shard_index, num_shards=args.num_shards,
        )
        _write_instance_shard_result(
            args.output_dir,
            split=split,
            class_names=accumulated["class_names"],
            entries=accumulated["entries"],
            images_with_labels=accumulated["images_with_labels"],
            num_images=accumulated["num_images"],
            infer_time_sum_ms=accumulated["infer_time_sum_ms"],
            failed=accumulated["failed"],
        )
        return

    accumulated = _accumulate_instance_eval(model, data_cfg, split, args.classwise)
    metric = accumulated["metric"]
    num_images = accumulated["num_images"]
    images_with_labels = accumulated["images_with_labels"]
    infer_time_sum_ms = accumulated["infer_time_sum_ms"]
    failed = accumulated["failed"]

    result = metric.compute_aggregated_values().metric_values
    summary_path = args.output_dir / "seg_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "data": str(args.data),
                "split": split,
                "num_images": num_images,
                "images_with_labels": images_with_labels,
                "metrics": result,
                "avg_infer_time_ms": infer_time_sum_ms / max(num_images, 1),
                "failed_images": failed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary saved to: {summary_path}")
