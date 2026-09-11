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
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import common as rt
from . import gpu_parallel
from . import run_reuse
from .artifact_transaction import staged_directory
from . import train_tools
from .progress import track
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


def _prefetch_iter(items: list[Any], load_fn):
    """单 worker 预取：消费当前项时后台解码下一项。产出 (item, loaded) 保序。"""
    from concurrent.futures import ThreadPoolExecutor

    if not items:
        return
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_fn, items[0])
        for index in range(len(items)):
            loaded = future.result()
            if index + 1 < len(items):
                future = executor.submit(load_fn, items[index + 1])
            yield items[index], loaded


def predict_model(model: Any, image: Any, threshold: float) -> Any:
    arg = image if not isinstance(image, (str, Path)) else str(image)
    try:
        return model.predict(arg, threshold=threshold)
    except TypeError:
        return model.predict(arg)


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


def _seg_infer_relative_paths(args: Any, image_paths: list[Path]) -> dict[Path, Path]:
    """Build collision-free output paths while preserving input subdirectories."""
    if not image_paths:
        return {}
    if getattr(args, "image", None) is not None:
        return {image_paths[0]: Path(image_paths[0].name)}
    explicit_dir = getattr(args, "image_dir", None)
    if explicit_dir is not None:
        root = Path(explicit_dir).expanduser().resolve()
    else:
        common = Path(os.path.commonpath([str(path.resolve()) for path in image_paths]))
        root = common if common.is_dir() else common.parent
    result: dict[Path, Path] = {}
    used: set[str] = set()
    for image_path in image_paths:
        try:
            relative = image_path.resolve().relative_to(root)
        except ValueError:
            relative = Path(image_path.name)
        key = relative.as_posix().casefold()
        if key in used:
            raise ValueError(f"分割推理输出路径冲突: {relative}（来源 {image_path}）")
        used.add(key)
        result[image_path] = relative
    return result


def _seg_infer_gt_lookup(args: Any) -> dict[Path, Path | None]:
    """实例分割 + 数据集模式下返回 {image_path: label_path}，用于产出 GT 对比图。

    语义分割、单图/目录模式（无标注）一律返回空 dict。
    """
    if _normalize_seg_type(args) == "semantic":
        return {}
    data_path = getattr(args, "data", None)
    if data_path is None:
        return {}
    try:
        data_cfg = rt.load_data_config(Path(data_path))
        samples, _ = rt.list_dataset_samples(data_cfg=data_cfg, split=args.split)
    except Exception:  # noqa: BLE001 拿不到真值就退回不画对比图
        return {}
    return {sample.image_path: sample.label_path for sample in samples}


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


def _semantic_class_mapping(model: Any, data_path: Path, split: str) -> dict[str, Any]:
    """构造语义评估用的类别映射与 LUT（与 _accumulate_semantic_confusion 内一致）。

    抽成独立函数，让"评估后挑图渲染"能重建同一套映射来复算显示用的类别索引，
    避免在渲染处重复一大段易漂移的 LUT 代码。指标仍只由 _accumulate 产出，不受此影响。
    """
    data_cfg = train_tools.load_semantic_segmentation_split_config(data_path, split)
    classes = data_cfg["classes"]
    ignore_classes = {int(item) for item in data_cfg.get("ignore_classes", set()) or set()}
    class_names = model if isinstance(model, dict) else rt.get_model_class_names(model)
    model_class_ids = sorted(class_names)
    model_class_to_idx = {cid: i for i, cid in enumerate(model_class_ids)}
    data_class_ids_sorted = sorted(set(classes) - ignore_classes)
    internal_to_model_idx = rt.np.full(len(data_class_ids_sorted) + 1, -1, dtype=rt.np.int64)
    for internal_id, orig_id in enumerate(data_class_ids_sorted):
        if orig_id in model_class_to_idx:
            internal_to_model_idx[internal_id] = model_class_to_idx[orig_id]
    pred_lut_size = (max(model_class_ids) + 1) if model_class_ids else 1
    pred_remap = rt.np.full(pred_lut_size, -1, dtype=rt.np.int64)
    for cid, idx in model_class_to_idx.items():
        pred_remap[cid] = idx
    return {
        "classes": classes,
        "ignore_classes": ignore_classes,
        "internal_to_model_idx": internal_to_model_idx,
        "pred_remap": pred_remap,
        "pred_lut_size": pred_lut_size,
        "idx_to_name": {i: class_names[cid] for i, cid in enumerate(model_class_ids)},
    }


def _remap_semantic_arrays(pred_np: Any, target_np: Any, mapping: dict[str, Any]) -> tuple[Any, Any]:
    """把单张图的原始 GT / 预测数组映射到统一类别索引空间（<0 表示忽略）。"""
    itmi = mapping["internal_to_model_idx"]
    target_clipped = rt.np.clip(target_np, 0, len(itmi) - 1)
    target_remapped = rt.np.where(
        target_np == -100, -100,
        rt.np.where(
            (target_np >= 0) & (target_np < len(itmi)),
            itmi[target_clipped], -100,
        ),
    )
    pred_remap = mapping["pred_remap"]
    pred_lut_size = mapping["pred_lut_size"]
    pred_clipped = rt.np.clip(pred_np, 0, pred_lut_size - 1)
    pred_remapped = rt.np.where(
        (pred_np >= 0) & (pred_np < pred_lut_size),
        pred_remap[pred_clipped], -1,
    )
    return target_remapped, pred_remapped


def _semantic_image_quality(target_remapped: Any, pred_remapped: Any, valid: Any, num_classes: int) -> float:
    """单张图的 mIoU，用于挑图排序（不参与全局指标）。异常一律返回 0.0。"""
    try:
        t = target_remapped[valid]
        p = pred_remapped[valid]
        if t.size == 0:
            return 0.0
        conf = rt.np.bincount(
            num_classes * t + p, minlength=num_classes ** 2
        ).reshape((num_classes, num_classes))
        metrics, _ = _compute_semantic_iou(conf)
        return float(metrics["miou"])
    except Exception:  # noqa: BLE001 选图评分绝不能中断评估
        return 0.0


def _semantic_dominant_class(target_remapped: Any, valid: Any) -> int:
    """一张图的主类别：GT 里占像素最多的统一索引类；无有效 GT 记 -1。"""
    gt = target_remapped[valid]
    if gt.size == 0:
        return -1
    return int(rt.np.bincount(gt).argmax())


def _select_diverse_entries(items: list[dict[str, Any]], count: int, *, best: bool, score_of, class_of) -> list[dict[str, Any]]:
    """按主类别轮询挑样本，让稀有类别也能进入样本，最多返回 count 个。

    best=True 取高分（表现好），best=False 取低分（表现差）。返回顺序即最终排名。
    """
    if count <= 0 or not items:
        return []
    groups: dict[Any, list[dict[str, Any]]] = {}
    for it in items:
        groups.setdefault(class_of(it), []).append(it)
    for lst in groups.values():
        lst.sort(key=score_of, reverse=best)
    class_order = sorted(groups.keys())
    selected: list[dict[str, Any]] = []
    cursors = {c: 0 for c in class_order}
    while len(selected) < count:
        progressed = False
        for c in class_order:
            lst = groups[c]
            idx = cursors[c]
            if idx < len(lst):
                selected.append(lst[idx])
                cursors[c] = idx + 1
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break
    return selected


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


def _render_instance_overlay(
    base_image: Any,
    labels: list[int],
    masks: Any,
    class_names: dict[int, str],
    *,
    scores: list[float] | None = None,
    color_by: str = "instance",
    with_labels: bool = False,
    font: Any = None,
) -> Any:
    """把实例掩码半透明叠加到 base_image 的副本上并返回新图（不改入参）。

    color_by="instance" 每个实例不同色（便于区分重叠实例）；
    color_by="class" 按类别上色（便于 GT 与预测同类对照）。
    with_labels=True 时在每个实例左上角画 类名(+分数) 文字标签。
    """
    overlay = rt.np.array(base_image.convert("RGB"))
    for idx, (label, mask) in enumerate(zip(labels, masks)):
        key = int(label) if color_by == "class" else idx
        color = rt.np.array(color_for_index(key), dtype=rt.np.uint8)
        mask_bool = rt.np.asarray(mask).astype(bool)
        overlay[mask_bool] = (0.55 * overlay[mask_bool] + 0.45 * color).astype(rt.np.uint8)
    image = rt.Image.fromarray(overlay)
    if with_labels:
        draw = rt.ImageDraw.Draw(image)
        if font is None:
            font = rt.load_cjk_font(15)
        for idx, (label, mask) in enumerate(zip(labels, masks)):
            ys, xs = rt.np.where(rt.np.asarray(mask).astype(bool))
            if xs.size == 0:
                continue
            key = int(label) if color_by == "class" else idx
            color = color_for_index(key)
            name = class_names.get(int(label), str(int(label)))
            text = f"{name} {scores[idx]:.2f}" if scores is not None else name
            x0, y0 = int(xs.min()), int(ys.min())
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_w, text_h = right - left, bottom - top
            rect_y0 = max(0, y0 - text_h - 4)
            draw.rectangle((x0, rect_y0, x0 + text_w + 4, rect_y0 + text_h + 4), fill=color)
            draw.text((x0 + 2, rect_y0 + 2), text, fill=(255, 255, 255), font=font)
    return image


def save_instance_visualization(image_path: Path, output_path: Path, prediction: dict[str, Any], class_names: dict[int, str]) -> None:
    with rt.Image.open(image_path) as image:
        base = image.convert("RGB")

    labels = prediction["labels"].detach().cpu().tolist()
    masks = prediction["masks"].detach().cpu().numpy()
    scores = prediction["scores"].detach().cpu().tolist()
    rendered = _render_instance_overlay(base, labels, masks, class_names, color_by="instance")
    meta = [
        {"class_id": int(label), "class_name": class_names.get(int(label), str(label)), "score": round(float(score), 6)}
        for label, score in zip(labels, scores)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(output_path)
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_instance_comparison(image_path: Path, output_path: Path, prediction: dict[str, Any], target: dict[str, Any], class_names: dict[int, str]) -> None:
    """生成 [原图 | 真值 GT | 预测 Pred] 三联对比图（实例分割），便于汇报展示。

    GT 与预测都按类别上色、带类名标签，便于直接比对同类区域。
    """
    with rt.Image.open(image_path) as image:
        base = image.convert("RGB")
    original = base.copy()
    font = rt.load_cjk_font(15)
    gt_panel = _render_instance_overlay(
        base,
        target["labels"].detach().cpu().tolist(),
        target["masks"].detach().cpu().numpy(),
        class_names,
        color_by="class",
        with_labels=True,
        font=font,
    )
    pred_panel = _render_instance_overlay(
        base,
        prediction["labels"].detach().cpu().tolist(),
        prediction["masks"].detach().cpu().numpy(),
        class_names,
        scores=prediction["scores"].detach().cpu().tolist(),
        color_by="class",
        with_labels=True,
        font=font,
    )
    rt.make_comparison_panel(
        [("原图", original), ("真值 GT", gt_panel), ("预测 Pred", pred_panel)],
        output_path,
    )


def _eval_compare_dir(
    output_like: Path,
    *,
    split: str | None = None,
    multi_split: bool = False,
) -> Path:
    """对比图的稳定落盘目录。

    分片子进程的 output_dir 形如 <final>/_shards/shard_xx，该目录评估后会被整体删除，
    所以对比图要写到 <final>/compare；非分片场景直接写 <output_dir>/compare。
    """
    path = Path(output_like)
    if path.parent.name == "_shards":
        path = path.parent.parent
    if multi_split:
        if split is None:
            raise ValueError("split is required when multi_split is enabled.")
        path = path / split
    return path / "compare"


def _filter_instance_prediction(prediction: dict[str, Any], threshold: float) -> dict[str, Any]:
    """按分数阈值过滤实例预测，用于展示（eval 的指标仍用全量预测，不受影响）。"""
    scores = prediction["scores"]
    keep = scores >= threshold
    return {"labels": prediction["labels"][keep], "masks": prediction["masks"][keep], "scores": scores[keep]}


def _render_semantic_index_overlay(base_image: Any, label_idx: Any) -> Any:
    """把统一索引空间的语义标签图半透明叠加到 base_image（按索引上色，GT 与预测同色）。"""
    h, w = label_idx.shape
    base = base_image.convert("RGB").resize((w, h))
    overlay = rt.np.array(base)
    for value in sorted(int(v) for v in set(label_idx.reshape(-1).tolist()) if v >= 0):
        color = rt.np.array(color_for_index(value), dtype=rt.np.uint8)
        sel = label_idx == value
        overlay[sel] = (0.55 * overlay[sel] + 0.45 * color).astype(rt.np.uint8)
    return rt.Image.fromarray(overlay)


def _render_class_legend(items: list[tuple[int, str]], height: int, *, font: Any = None) -> Any:
    """生成类别图例图（色块+类名），高度对齐到 height，便于和对比图等高拼接。"""
    if font is None:
        font = rt.load_cjk_font(16)
    swatch, pad = 18, 8
    probe = rt.ImageDraw.Draw(rt.Image.new("RGB", (8, 8)))
    text_w = max((probe.textbbox((0, 0), name, font=font)[2] for _, name in items), default=40)
    width = swatch + pad * 3 + int(text_w)
    legend = rt.Image.new("RGB", (max(width, 60), max(height, 1)), (255, 255, 255))
    draw = rt.ImageDraw.Draw(legend)
    y = pad
    for idx, name in items:
        draw.rectangle((pad, y, pad + swatch, y + swatch), fill=color_for_index(idx))
        draw.text((pad * 2 + swatch, y), name, fill=(20, 20, 20), font=font)
        y += swatch + pad
    return legend


def save_semantic_comparison(image_path: Path, output_path: Path, gt_idx: Any, pred_idx: Any, idx_to_name: dict[int, str]) -> None:
    """生成 [原图 | 真值 GT | 预测 Pred | 图例] 语义分割对比图。

    gt_idx / pred_idx 均为统一类别索引空间的 2D 数组（<0 表示忽略/背景），保证同类同色。
    """
    with rt.Image.open(image_path) as image:
        base = image.convert("RGB")
    h, w = pred_idx.shape
    original = base.resize((w, h))
    gt_panel = _render_semantic_index_overlay(base, gt_idx)
    pred_panel = _render_semantic_index_overlay(base, pred_idx)
    present = sorted({int(v) for v in set(gt_idx.reshape(-1).tolist()) | set(pred_idx.reshape(-1).tolist()) if v >= 0})
    panels = [("原图", original), ("真值 GT", gt_panel), ("预测 Pred", pred_panel)]
    if present:
        legend = _render_class_legend([(v, idx_to_name.get(v, str(v))) for v in present], h)
        panels.append(("类别图例", legend))
    rt.make_comparison_panel(panels, output_path)


def _render_semantic_visualizations(
    model: Any | None,
    data_path: Path,
    split: str,
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    max_images: int,
) -> None:
    """评估后统一挑图渲染语义对比图：好/差各半、类别尽量全，最多 max_images 张。

    max_images<=0 表示不限制、渲染全部（旧行为）。挑中的图按 per-image mIoU 排序，
    重新推理一次以拿到预测掩码后渲染，写入 compare/good 与 compare/bad，并汇总 manifest。
    重新推理只针对挑中的这一小批图，成本很低，也避免为全量图缓存逐像素预测。
    """
    scored = [
        r for r in rows
        if r.get("image_path") and r.get("mask_path") and r.get("vis_miou") is not None
    ]
    if not scored:
        return

    def _score_of(r: dict[str, Any]) -> float:
        return float(r.get("vis_miou", 0.0))

    def _class_of(r: dict[str, Any]) -> int:
        return int(r.get("vis_class", -1))

    if max_images and max_images > 0:
        n_good = max_images // 2
        n_bad = max_images - n_good
        good = _select_diverse_entries(scored, n_good, best=True, score_of=_score_of, class_of=_class_of)
        picked = {id(r) for r in good}
        remaining = [r for r in scored if id(r) not in picked]
        bad = _select_diverse_entries(remaining, n_bad, best=False, score_of=_score_of, class_of=_class_of)
    else:
        good = sorted(scored, key=_score_of, reverse=True)
        bad = []

    mapping = _semantic_class_mapping(model, data_path, split)
    idx_to_name = mapping["idx_to_name"]
    compare_root = output_dir / "compare"
    manifest: list[dict[str, Any]] = []

    def _render_bucket(bucket: list[dict[str, Any]], subdir: str | None) -> None:
        target_dir = compare_root / subdir if subdir else compare_root
        for rank, row in enumerate(
            track(
                bucket,
                label=f"seg/eval 出图 {subdir or 'all'}",
                total=len(bucket),
                unit="img",
                aggregate="extra",
            ),
            start=1,
        ):
            image_path = Path(row["image_path"])
            mask_path = Path(row["mask_path"])
            score = float(row.get("vis_miou", 0.0))
            try:
                prediction_path = row.get("prediction_path")
                if prediction_path:
                    with rt.Image.open(Path(prediction_path)) as prediction_image:
                        pred_np = rt.np.array(prediction_image)
                else:
                    if model is None:
                        raise ValueError("缺少缓存预测，且父进程没有加载模型")
                    prediction = predict_model(model, image_path, 0.0)
                    pred_np = _prediction_to_numpy(prediction)
                target_np = _load_semantic_mask(mask_path, mapping["classes"], mapping["ignore_classes"])
                pred_np = _resize_prediction_if_needed(pred_np, target_np.shape)
                target_remapped, pred_remapped = _remap_semantic_arrays(pred_np, target_np, mapping)
                name = f"{rank:03d}_miou{score:.2f}_{image_path.stem}.png"
                out_path = target_dir / name
                save_semantic_comparison(image_path, out_path, target_remapped, pred_remapped, idx_to_name)
            except Exception as exc:  # noqa: BLE001 单图出图失败不影响其余
                print(f"  ⚠ 对比图生成失败 {image_path}: {exc}")
                continue
            manifest.append({
                "bucket": subdir or "all",
                "rank": rank,
                "miou": round(score, 6),
                "dominant_class": idx_to_name.get(_class_of(row), str(_class_of(row))),
                "image": str(image_path),
                "mask": str(mask_path),
                "output": str(out_path.relative_to(output_dir)),
            })

    if bad:
        _render_bucket(good, "good")
        _render_bucket(bad, "bad")
    else:
        _render_bucket(good, None)

    compare_root.mkdir(parents=True, exist_ok=True)
    (compare_root / "manifest.json").write_text(
        json.dumps(
            {
                "split": split,
                "total_scored": len(scored),
                "rendered": len(manifest),
                "max_images": max_images,
                "items": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    # 人类可读的映射文档：说明命名规则，并把每张输出对比图映射回原图 + 原 mask，
    # 方便从原数据集里定位原始数据。字段与 manifest.json 一致，只是排版成表格。
    lines = [
        f"# seg eval 对比图映射（split: {split}）",
        "",
        "## 命名规则",
        "",
        "输出文件名格式：`<排名 3 位>_miou<该图 mIoU>_<原图文件名>.png`",
        "",
        "- `good/` = 表现最好的一批（mIoU 高），`bad/` = 表现最差的一批（mIoU 低）。",
        "- 文件名里的 `原图文件名` 就是原图去掉扩展名后的 stem；下表给出原图与原 mask 的完整路径。",
        "- 挑图规则：好/差各半、按主类别轮询以尽量覆盖更多类别；`max_images` 控制总数（0=不限制）。",
        f"- 本次：共评分 {len(scored)} 张，实际出图 {len(manifest)} 张，max_images={max_images}。",
        "",
        "## 逐图映射",
        "",
        "| 输出对比图 | mIoU | 主类别 | 原图 | 原 mask |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in manifest:
        lines.append(
            f"| {item['output']} | {item['miou']:.4f} | {item['dominant_class']} "
            f"| {item['image']} | {item['mask']} |"
        )
    (compare_root / "mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{split}] 对比图已保存到: {compare_root}（共 {len(manifest)} 张）")
    print(f"[{split}] 文件名↔原图/原mask 映射: {compare_root / 'mapping.md'}（另有 manifest.json）")


def _seg_reuse_precheck(args, action: str) -> str:
    """构造 seg 指纹 + 所需产物清单，做初步复用检测，返回 run_reuse 的决策常量。"""
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    data = getattr(args, "data", None)
    if data is not None:
        input_mode = "dataset"
    elif getattr(args, "image", None) is not None:
        input_mode = "image"
    else:
        input_mode = "image_dir"
    seg_type = _normalize_seg_type(args)
    fingerprint = run_reuse.make_fingerprint(
        task="seg", action=action, checkpoint=checkpoint_path,
        data=data, split=getattr(args, "split", None),
        threshold=getattr(args, "threshold", None), input_mode=input_mode,
        image=getattr(args, "image", None), image_dir=getattr(args, "image_dir", None),
        seg_train_type=seg_type,
        options={
            "save_visualization": bool(getattr(args, "save_visualization", True)),
            "classwise": bool(getattr(args, "classwise", False)),
            "vis_max_images": getattr(args, "vis_max_images", None),
        },
    )
    # run_meta 只有成功完成后才原子写入。对比图可能因数据没有有效标注而合法为空，
    # 因此不把非空 compare/ 当作完成条件。
    if action == "eval":
        summary = "seg_semantic_eval_summary.json" if seg_type == "semantic" else "seg_eval_summary.json"
        splits = _normalize_splits(getattr(args, "split", None))
        prefixes = [f"{split}/" for split in splits] if len(splits) > 1 else [""]
        required = [f"{prefix}{summary}" for prefix in prefixes]
        required.extend(f"{prefix}run_meta.json" for prefix in prefixes)
    else:
        required = ["run_meta.json"]
    return run_reuse.precheck(args, args.output_dir, fingerprint, action_label=f"seg/{action}", required=required)


def run_infer(args) -> None:
    if _is_seg_shard_child(args) or getattr(args, "dry_run", False):
        _run_infer_impl(args)
        return
    if _seg_reuse_precheck(args, "infer") == run_reuse.REUSE:
        print(f"[seg/infer] 已复用上次结果，未重新推理：{args.output_dir}")
        return
    final_output_dir = Path(args.output_dir).expanduser().resolve()
    original_output_dir = args.output_dir
    original_overwrite = args.overwrite
    try:
        with staged_directory(final_output_dir, overwrite=args.overwrite) as stage:
            args.output_dir = stage
            args.overwrite = True
            args._published_output_dir = final_output_dir
            _run_infer_impl(args)
    finally:
        args.output_dir = original_output_dir
        args.overwrite = original_overwrite
        if hasattr(args, "_published_output_dir"):
            delattr(args, "_published_output_dir")


def _run_infer_impl(args) -> None:
    action = "infer"
    if not _is_seg_shard_child(args) and run_parallel_seg_infer(args):
        return
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    output_dir = args.output_dir
    if getattr(args, "dry_run", False):
        print(f"[seg/{action}] dry-run 计划")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  seg_train_type: {_normalize_seg_type(args)}")
        print(f"  split: {getattr(args, 'split', None)}")
        print(f"  output_dir: {output_dir}")
        try:
            print(f"  num_images: {len(_seg_infer_image_paths(args))}")
        except Exception:  # noqa: BLE001 dry-run 仅尽力打印，拿不到数量就跳过
            pass
        return
    rt.prepare_output_dir(output_dir, args.overwrite, clean=True)

    device_mode = args.device
    resolved_device_arg = args.device
    if args.device == "auto" and not _is_seg_shard_child(args):
        all_gpus, message = gpu_parallel.query_gpu_inventory()
        if message:
            print(f"[seg/{action}] {message}")
        else:
            eligible = gpu_parallel.filter_high_memory_gpus(all_gpus)
            chosen, choose_msg = gpu_parallel.select_fallback_single_gpu(
                all_gpus, eligible, component=f"seg/{action}"
            )
            print(f"[seg/{action}] {choose_msg}")
            if chosen is not None:
                resolved_device_arg = chosen
                device_mode = chosen

    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(resolved_device_arg))
    model.eval()
    class_names = rt.get_model_class_names(model)
    image_paths = _seg_infer_image_paths(args)
    if not image_paths:
        raise ValueError("没有找到可推理的图片。")
    relative_outputs = _seg_infer_relative_paths(args, image_paths)
    if _is_seg_shard_child(args):
        subset = gpu_parallel.filter_indices_for_shard(
            len(image_paths), shard_index=args.shard_index, num_shards=args.num_shards,
            weights=[float(path.stat().st_size) if path.exists() else 1.0 for path in image_paths],
        )
        image_paths = [image_paths[i] for i in subset]
    # 数据集模式（实例分割）下能拿到真值，额外产出 [原图|GT|预测] 三联对比图。
    gt_lookup = _seg_infer_gt_lookup(args)
    # 统一目录结构：普通可视化在 images/，三联对比图在 compare/。
    images_dir = output_dir / "images"
    compare_dir = output_dir / "compare"
    for idx, image_path in enumerate(
        track(
            image_paths,
            label="seg/infer 推理",
            total=len(image_paths),
            unit="img",
            shard_scope="inference",
        ),
        start=1,
    ):
        prediction = predict_model(model, image_path, args.threshold)
        out_path = images_dir / relative_outputs[image_path]
        if isinstance(prediction, dict) and "masks" in prediction:
            save_instance_visualization(image_path, out_path, prediction, class_names)
            label_path = gt_lookup.get(image_path)
            if label_path is not None and Path(label_path).exists():
                masks = prediction["masks"]
                image_size = (int(masks.shape[-1]), int(masks.shape[-2]))  # (width, height)
                target, has_label = load_instance_ground_truth(label_path, image_size)
                if has_label:
                    relative_out = out_path.relative_to(images_dir)
                    compare_path = (compare_dir / relative_out).with_name(
                        f"{relative_out.stem}_compare.png"
                    )
                    save_instance_comparison(image_path, compare_path, prediction, target, class_names)
        else:
            save_semantic_visualization(image_path, out_path, prediction, class_names)
        # 进度由 track() 单行进度条展示

    if not _is_seg_shard_child(args):
        _write_seg_run_meta(
            output_dir / "run_meta.json",
            action=action, checkpoint_path=checkpoint_path,
            output_dir=output_dir, args=args, num_images=len(image_paths), device_mode=device_mode,
        )


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


def _write_seg_run_meta(meta_path, *, action, checkpoint_path, output_dir, args, num_images, device_mode):
    split_value = getattr(args, "split", None)
    recorded_output_dir = Path(output_dir)
    published_root = getattr(args, "_published_output_dir", None)
    stage_root = getattr(args, "output_dir", None)
    if published_root is not None and stage_root is not None:
        try:
            recorded_output_dir = Path(published_root) / Path(output_dir).relative_to(Path(stage_root))
        except ValueError:
            pass
    payload = {
        "task": "seg",
        "action": action,
        "complete": True,
        "created_at": rt.timestamp_now_iso(),
        "seg_train_type": str(getattr(args, "seg_train_type", "instance")),
        "split": split_value if isinstance(split_value, str) else _normalize_splits(split_value or []),
        "num_images": int(num_images),
        "input_mode": (
            "dataset" if getattr(args, "data", None) is not None
            else ("image" if getattr(args, "image", None) is not None else "image_dir")
        ),
        "paths": {
            "output_dir": str(recorded_output_dir),
            "checkpoint_path": str(checkpoint_path),
            "data": str(getattr(args, "data", None)) if getattr(args, "data", None) is not None else None,
            "image": str(getattr(args, "image", None)) if getattr(args, "image", None) is not None else None,
            "image_dir": str(getattr(args, "image_dir", None)) if getattr(args, "image_dir", None) is not None else None,
        },
        "settings": {
            "device": getattr(args, "device", None),
            "device_mode": device_mode,
            "threshold": getattr(args, "threshold", None),
            "overwrite": getattr(args, "overwrite", None),
        },
    }
    payload["fingerprint"] = run_reuse.make_fingerprint(
        task="seg", action=action, checkpoint=checkpoint_path,
        data=getattr(args, "data", None), split=getattr(args, "split", None),
        threshold=getattr(args, "threshold", None),
        input_mode=payload["input_mode"], image=getattr(args, "image", None),
        image_dir=getattr(args, "image_dir", None),
        seg_train_type=_normalize_seg_type(args),
        options={
            "save_visualization": bool(getattr(args, "save_visualization", True)),
            "classwise": bool(getattr(args, "classwise", False)),
            "vis_max_images": getattr(args, "vis_max_images", None),
        },
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temp_meta = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.tmp")
    temp_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_meta.replace(meta_path)


def _is_seg_shard_child(args: Any) -> bool:
    """子进程分片模式判定：显式给了 shard_index 且 num_shards>1 才算分片子进程。"""
    return getattr(args, "shard_index", None) is not None and int(getattr(args, "num_shards", 1) or 1) > 1


def _write_instance_shard_result(
    shard_dir: Path,
    *,
    split: str,
    class_names: dict[int, str],
    entries: list[dict[str, Any]] | None,
    images_with_labels: int,
    num_images: int,
    infer_time_sum_ms: float,
    failed: int,
    entries_path: Path | None = None,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "class_names": {str(k): v for k, v in class_names.items()},
        "entries_file": "seg_instance_entries.jsonl" if entries or (entries_path is not None and entries_path.is_file()) else None,
        "images_with_labels": int(images_with_labels),
        "num_images": int(num_images),
        "infer_time_sum_ms": float(infer_time_sum_ms),
        "failed": int(failed),
    }
    path = shard_dir / "seg_instance_shard_result.json"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    if entries:
        entries_path = shard_dir / "seg_instance_entries.jsonl"
        with entries_path.open("w", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    elif entries_path is not None and entries_path.resolve() != (shard_dir / "seg_instance_entries.jsonl").resolve():
        shutil.copy2(entries_path, shard_dir / "seg_instance_entries.jsonl")
    return path


def _merge_instance_shard_results(shard_dirs: list[Path], *, classwise: bool) -> dict[str, Any]:
    class_names: dict[int, str] = {}
    entry_paths: list[Path] = []
    num_images = 0
    images_with_labels = 0
    infer_time_sum_ms = 0.0
    failed = 0
    for shard_dir in shard_dirs:
        payload = json.loads((shard_dir / "seg_instance_shard_result.json").read_text(encoding="utf-8"))
        for cid_raw, name in payload.get("class_names", {}).items():
            class_names[int(cid_raw)] = str(name)
        entries_file = payload.get("entries_file")
        if entries_file:
            entry_paths.append(shard_dir / str(entries_file))
        num_images += int(payload.get("num_images", 0))
        images_with_labels += int(payload.get("images_with_labels", 0))
        infer_time_sum_ms += float(payload.get("infer_time_sum_ms", 0.0))
        failed += int(payload.get("failed", 0))
    metric, label_mapping = create_metric(class_names, classwise)

    def _entries():
        for entries_path in entry_paths:
            with entries_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)

    for entry in track(
        _entries(),
        label="seg/eval 聚合实例指标",
        total=images_with_labels,
        unit="img",
    ):
        prediction, target = _deserialize_instance_entry(entry)
        update_metric(metric, label_mapping, prediction, target)
    print("[seg/eval] 阶段: 计算聚合指标", flush=True)
    metric_values = metric.compute_aggregated_values().metric_values
    print("[seg/eval] 阶段完成: 聚合指标", flush=True)
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
    vis_dir: Path | None = None,
    vis_threshold: float = 0.0,
    entries_path: Path | None = None,
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
        len(samples), shard_index=shard_index, num_shards=num_shards,
        weights=[float(sample.image_path.stat().st_size) if sample.image_path.exists() else 1.0 for sample in samples],
    )
    shard_samples = [samples[i] for i in shard_indices]

    entries: list[dict[str, Any]] = []
    images_with_labels = 0
    infer_time_sum_ms = 0.0
    failed = 0
    # 后台解码下一张图，与当前图的 GPU 推理重叠。不强制 .convert("RGB")：
    # 上游 model.predict 对 path/PIL 都不会强制转 RGB；image_size 取自 PIL 的
    # .size (W, H)，与原先单独打开取 size 等价。解码异常延后到 try 里抛出，
    # 使坏图计入 failed 并跳过。
    def _load_image(sample):
        try:
            return rt.Image.open(sample.image_path), None
        except Exception as exc:  # noqa: BLE001 解码失败延后到主线程处理
            return None, exc

    for idx, (sample, (image, load_error)) in enumerate(
        track(
            _prefetch_iter(shard_samples, _load_image),
            label="seg/eval 评估",
            total=len(shard_samples),
            unit="img",
            shard_scope="inference",
        ),
        start=1,
    ):
        try:
            if load_error is not None:
                raise load_error
            image_size = image.size
            start = time.perf_counter()
            # mAP 必须吃全量带分预测：阈值过滤会截断 PR 曲线、人为压低指标，
            # 与训练内部验证（不做阈值过滤）不一致。故评估恒用 0.0。
            prediction = predict_model(model, image, 0.0)
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
        serialized_entry = _serialize_instance_entry(prediction, target)
        if entries_path is not None:
            entries_path.parent.mkdir(parents=True, exist_ok=True)
            with entries_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(serialized_entry, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
        else:
            entries.append(serialized_entry)
        # 指标用全量预测（0.0）；对比图按展示阈值过滤，避免低分实例糊满画面。
        if vis_dir is not None and has_label:
            try:
                relative = Path(getattr(sample, "relative_path", Path(sample.image_path).name))
                compare_path = (vis_dir / relative).with_name(
                    f"{relative.stem}_compare.png"
                )
                shown = _filter_instance_prediction(prediction, vis_threshold)
                save_instance_comparison(sample.image_path, compare_path, shown, target, class_names)
            except Exception as exc:  # noqa: BLE001 可视化失败绝不能中断评估
                print(f"  ⚠ 对比图生成失败 {sample.image_path}: {exc}")
        # 进度由 track() 单行进度条展示

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
    # per_class 的 key 是混淆矩阵连续下标（0..K-1），class_names 的 key 是
    # 模型原始类别 ID；用排序后的 ID 列表做下标→名称的映射。
    sorted_ids = sorted(class_names)
    print(f"\n  {'=' * 50}")
    print(f"    [{split}] 评估结果")
    print(f"  {'=' * 50}")
    print(f"    mIoU          : {metrics['miou']:.4f}")
    print(f"    Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
    print(f"  {'=' * 50}")
    print(f"    逐类 IoU:")
    for class_id_str, info in per_class.items():
        idx = int(class_id_str)
        class_name = class_names[sorted_ids[idx]] if idx < len(sorted_ids) else f"class_{class_id_str}"
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
    prediction_cache_dir: Path | None = None,
) -> dict[str, Any] | None:
    """加载样本并累积混淆矩阵，可分片、带容错与计时。无样本/无该 split 时返回 None。

    每图会额外记录用于挑图的 mIoU 与主类别；对比图的挑选与渲染统一在评估结束后进行。
    """
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

    # 以模型类别 ID 为唯一权威类表，构造 → 混淆矩阵下标的映射。
    # _load_semantic_mask 内部会把 GT 压成 sorted(classes-ignore) 的连续 ID，
    # 而 model.predict 返回原始类别 ID；两套编号在第一个缺口后错位。
    # 统一映射确保 GT 和 predictions 都落在同一张混淆矩阵里。
    class_names = rt.get_model_class_names(model)
    model_class_ids = sorted(class_names)
    num_eval_classes = len(model_class_ids)
    model_class_to_idx = {cid: i for i, cid in enumerate(model_class_ids)}

    # _load_semantic_mask 输出的 internal ID 对应 sorted(data_classes - ignore)，
    # 需要先还原到原始 ID 再映射到统一索引。
    data_class_ids_sorted = sorted(set(classes) - ignore_classes)
    internal_to_model_idx = rt.np.full(len(data_class_ids_sorted) + 1, -1, dtype=rt.np.int64)
    for internal_id, orig_id in enumerate(data_class_ids_sorted):
        if orig_id in model_class_to_idx:
            internal_to_model_idx[internal_id] = model_class_to_idx[orig_id]

    # 预测的原始 ID → 统一索引（不在模型类表中的 ID 得 -1）。
    pred_lut_size = (max(model_class_ids) + 1) if model_class_ids else 1
    pred_remap = rt.np.full(pred_lut_size, -1, dtype=rt.np.int64)
    for cid, idx in model_class_to_idx.items():
        pred_remap[cid] = idx

    confusion = rt.np.zeros((num_eval_classes, num_eval_classes), dtype=rt.np.int64)
    rows: list[dict[str, Any]] = []
    infer_time_sum_ms = 0.0
    failed = 0

    shard_indices = gpu_parallel.filter_indices_for_shard(
        len(samples), shard_index=shard_index, num_shards=num_shards,
        weights=[float(sample[0].stat().st_size) if sample[0].exists() else 1.0 for sample in samples],
    )
    shard_samples = [samples[i] for i in shard_indices]

    print(f"\n  [{split}] 找到 {len(samples)} 个图片/掩码对" + (
        f"（本分片 {len(shard_samples)} 个）" if len(shard_samples) != len(samples) else ""
    ))
    # 后台解码下一张图，与当前图的 GPU 推理重叠。不强制 .convert("RGB")：
    # 上游 model.predict 对 path/PIL 都不会强制转 RGB，强制转换会改变非 RGB
    # 图的输入张量、进而改变数值结果。保持与上游一致。
    # 解码异常在 worker 里捕获后随返回值带出，留到循环体的 try 里再抛，
    # 这样坏图仍被计入 failed 并跳过（保持原有逐图容错语义）。
    def _load_image(sample):
        image_path, _mask_path = sample
        try:
            return rt.Image.open(image_path), None
        except Exception as exc:  # noqa: BLE001 解码失败延后到主线程处理
            return None, exc

    for idx, ((image_path, mask_path), (image, load_error)) in enumerate(
        track(
            _prefetch_iter(shard_samples, _load_image),
            label="seg/eval 语义评估",
            total=len(shard_samples),
            unit="img",
            shard_scope="inference",
        ),
        start=1,
    ):
        try:
            if load_error is not None:
                raise load_error
            start = time.perf_counter()
            prediction = predict_model(model, image, threshold)
            infer_time_sum_ms += (time.perf_counter() - start) * 1000.0
        except Exception as exc:  # noqa: BLE001 单图失败不应中断整轮评估
            failed += 1
            print(f"  ⚠ 跳过 {image_path}: {exc}")
            continue
        pred_np = _prediction_to_numpy(prediction)
        target_np = _load_semantic_mask(mask_path, classes, ignore_classes)
        pred_np = _resize_prediction_if_needed(pred_np, target_np.shape)

        # GT: internal ID → 统一索引（-100 保持不变用于忽略）
        target_clipped = rt.np.clip(target_np, 0, len(internal_to_model_idx) - 1)
        target_remapped = rt.np.where(
            target_np == -100, -100,
            rt.np.where(
                (target_np >= 0) & (target_np < len(internal_to_model_idx)),
                internal_to_model_idx[target_clipped], -100,
            ),
        )
        # predictions: 原始 ID → 统一索引（越界或不在模型类表中的 → -1）
        pred_clipped = rt.np.clip(pred_np, 0, pred_lut_size - 1)
        pred_remapped = rt.np.where(
            (pred_np >= 0) & (pred_np < pred_lut_size),
            pred_remap[pred_clipped], -1,
        )

        valid = target_remapped >= 0
        valid &= pred_remapped >= 0
        if valid.any():
            bincount = rt.np.bincount(
                num_eval_classes * target_remapped[valid] + pred_remapped[valid],
                minlength=num_eval_classes ** 2,
            )
            confusion += bincount.reshape((num_eval_classes, num_eval_classes))
        # 记录每图 mIoU 与主类别，供评估后统一挑图（好/差各半、类别尽量全）。
        # 对比图的渲染不再逐图内联进行——那样会给全部图都出图；挑选是全局的，
        # 必须看完所有图（多卡时还要汇总各分片）后再渲染选中的一小批。
        vis_miou = _semantic_image_quality(target_remapped, pred_remapped, valid, num_eval_classes)
        vis_class = _semantic_dominant_class(target_remapped, valid)
        prediction_path = None
        if prediction_cache_dir is not None:
            prediction_cache_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = prediction_cache_dir / f"prediction_{len(rows):08d}.png"
            prediction_dtype = rt.np.uint8 if int(pred_np.max()) <= 255 else rt.np.uint16
            rt.Image.fromarray(pred_np.astype(prediction_dtype)).save(prediction_path)
        rows.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "valid_pixels": int(valid.sum()),
            "vis_miou": round(float(vis_miou), 6),
            "vis_class": vis_class,
            "prediction_path": str(prediction_path) if prediction_path is not None else None,
        })
        # 进度由 track() 单行进度条展示

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
    checkpoint_path: Path,
    data_path: Path,
) -> dict[str, Any]:
    """计算 IoU、写 summary/CSV 并打印指标。顺序路径与并行合并路径共用，保证产物字节一致。"""
    metrics, per_class = _compute_semantic_iou(confusion)
    # 目录清理由调用方在运行边界负责；多卡汇总时目录内仍有待读取的分片缓存。
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "seg_semantic_eval_summary.json"
    csv_path = output_dir / "seg_semantic_eval_samples.csv"
    rt.save_records_csv(csv_path, rows, ["image_path", "mask_path", "valid_pixels", "vis_miou", "vis_class"])
    summary = {
        "task": "semantic_segmentation",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "split": split,
        "num_images": num_samples,
        "attempted_images": num_samples + failed,
        "succeeded_images": num_samples,
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
    save_visualization: bool = True,
    vis_max_images: int = 0,
) -> dict[str, Any] | None:
    """对单个 split 执行语义分割评估，保存 summary/CSV 并打印指标。

    分片子进程模式（shard_index 给定且 num_shards>1）下不写最终 summary，
    而是把本分片的混淆矩阵/rows 写成 shard result，留给父进程合并。
    非分片模式下，写完 summary 再挑图渲染对比图（好/差各半、类别尽量全）。
    """
    accumulated = _accumulate_semantic_confusion(
        model, data_path, split, threshold,
        shard_index=shard_index, num_shards=num_shards,
        prediction_cache_dir=(
            output_dir / split / "_prediction_cache"
            if shard_index is not None and int(num_shards or 1) > 1 and save_visualization
            else None
        ),
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

    rt.prepare_output_dir(output_dir, overwrite, clean=True)
    summary = _write_semantic_summary(
        output_dir,
        split=split,
        confusion=confusion,
        rows=rows,
        class_names=class_names,
        num_classes=num_classes,
        num_samples=num_samples,
        infer_time_sum_ms=infer_time_sum_ms,
        failed=failed,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
    )
    # 输出目录已准备完毕，随后生成对比图。
    if save_visualization:
        _render_semantic_visualizations(
            model, data_path, split, rows,
            output_dir=output_dir, max_images=vis_max_images,
        )
    return summary


def run_semantic_eval(args) -> None:
    action = "eval"
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    data_path = Path(args.data).expanduser().resolve()
    splits = _normalize_splits(args.split)
    if getattr(args, "dry_run", False):
        print(f"[seg/{action}] dry-run 计划")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  seg_train_type: {_normalize_seg_type(args)}")
        print(f"  split: {getattr(args, 'split', None)}")
        print(f"  output_dir: {args.output_dir}")
        for split in splits:
            try:
                samples, _, _ = _semantic_image_mask_samples(data_path, split)
                print(f"  num_images[{split}]: {len(samples)}")
            except Exception:  # noqa: BLE001 dry-run 仅尽力打印，拿不到数量就跳过
                pass
        return

    if not _is_seg_shard_child(args) and len(splits) > 1:
        rt.prepare_output_dir(args.output_dir, args.overwrite, clean=True)

    device_mode = args.device
    resolved_device_arg = args.device
    if args.device == "auto" and not _is_seg_shard_child(args):
        all_gpus, message = gpu_parallel.query_gpu_inventory()
        if message:
            print(f"[seg/{action}] {message}")
        else:
            eligible = gpu_parallel.filter_high_memory_gpus(all_gpus)
            chosen, choose_msg = gpu_parallel.select_fallback_single_gpu(
                all_gpus, eligible, component=f"seg/{action}"
            )
            print(f"[seg/{action}] {choose_msg}")
            if chosen is not None:
                resolved_device_arg = chosen
                device_mode = chosen

    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(resolved_device_arg))
    model.eval()

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
                save_visualization=getattr(args, "save_visualization", True),
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
            save_visualization=getattr(args, "save_visualization", True),
            vis_max_images=int(getattr(args, "vis_max_images", rt.SEG_EVAL_VIS_MAX_IMAGES)),
        )
        if summary is not None:
            summaries[split] = summary
            _write_seg_run_meta(
                split_output_dir / "run_meta.json",
                action=action, checkpoint_path=checkpoint_path,
                output_dir=split_output_dir, args=args,
                num_images=summary.get("num_images", 0), device_mode=device_mode,
            )

    if _is_seg_shard_child(args):
        return
    if not summaries:
        raise ValueError("No image/mask pairs found for semantic segmentation eval.")
    if len(splits) > 1:
        _write_seg_run_meta(
            args.output_dir / "run_meta.json",
            action=action,
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            args=args,
            num_images=sum(int(summary.get("num_images", 0)) for summary in summaries.values()),
            device_mode=device_mode,
        )


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
    command += ["--save-visualization"] if getattr(args, "save_visualization", True) else ["--no-save-visualization"]
    if getattr(args, "threshold", None) is not None:
        command += ["--threshold", str(args.threshold)]
    command += ["--vis-max-images", str(getattr(args, "vis_max_images", rt.SEG_EVAL_VIS_MAX_IMAGES))]
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
    split_weights: list[list[float]] = []
    if semantic:
        for split in splits:
            samples = _semantic_image_mask_samples(data_path, split)[0]
            split_weights.append(
                [float(image_path.stat().st_size) if image_path.exists() else 1.0 for image_path, _ in samples]
            )
    else:
        data_cfg = rt.load_data_config(data_path)
        for split in splits:
            samples = rt.list_dataset_samples(data_cfg=data_cfg, split=split)[0]
            split_weights.append(
                [
                    float(sample.image_path.stat().st_size) if sample.image_path.exists() else 1.0
                    for sample in samples
                ]
            )
    sample_count = sum(len(weights) for weights in split_weights)
    if sample_count < 2:
        print(f"[seg/eval] 待评估样本数量为 {sample_count}，进入单卡顺序模式。")
        return False
    rt.prepare_output_dir(final_output_dir, args.overwrite, clean=True)

    # 子进程在每个 split 内独立分片，因此 shard 数由最大 split 决定，
    # 保证每个启动的子进程至少获得一个样本。
    num_shards = min(len(eligible), max((len(weights) for weights in split_weights), default=0))
    if num_shards < 2:
        print("[seg/eval] 单个 split 可分配的 shard 数小于 2，进入单卡顺序模式。")
        return False
    shard_totals = {shard: 0 for shard in range(num_shards)}
    for weights in split_weights:
        per_split = gpu_parallel.shard_item_totals(
            len(weights), num_shards=num_shards, weights=weights,
        )
        for shard, count in per_split.items():
            shard_totals[shard] += count
    shard_root = final_output_dir / "_shards"
    shard_dirs = [shard_root / f"shard_{i:02d}" for i in range(num_shards)]

    jobs: list[tuple[int, int | str, list[str]]] = []
    for i, gpu in enumerate(eligible[:num_shards]):
        command = _build_seg_eval_child_command(
            args, shard_index=i, num_shards=num_shards, output_dir=shard_dirs[i], device="auto"
        )
        jobs.append((i, gpu.get("device_token", int(gpu["index"])), command))

    print("[seg/eval] 已进入自动多卡并行模式。")
    for gpu in eligible[:num_shards]:
        print(f"[seg/eval] 保留 {gpu_parallel.format_gpu_summary(gpu)}")
    for shard_index, gpu_index, _ in jobs:
        print(f"[seg/eval] shard={shard_index}/{num_shards} -> CUDA_VISIBLE_DEVICES={gpu_index}")

    failed = gpu_parallel.run_sharded_subprocesses(
        jobs, cwd=rt.ROOT_DIR, shard_totals=shard_totals,
    )
    if failed:
        raise RuntimeError(f"seg eval shard 失败: {', '.join(failed)}")

    if semantic:
        _merge_parallel_semantic(
            args, shard_dirs, splits=splits, final_output_dir=final_output_dir,
            checkpoint_path=checkpoint_path, data_path=data_path,
        )
    else:
        _merge_parallel_instance(
            args, shard_dirs, splits=splits, final_output_dir=final_output_dir,
            checkpoint_path=checkpoint_path,
        )
    shutil.rmtree(shard_root, ignore_errors=True)
    return True


def _build_seg_infer_child_command(args, *, shard_index, num_shards, output_dir, device="auto") -> list[str]:
    command = [sys.executable, str(rt.ROOT_DIR / "launcher.py"), "infer", "--task", "seg"]
    command += ["--seg-train-type", str(getattr(args, "seg_train_type", "instance"))]
    if getattr(args, "experiment_dir", None) is not None:
        command += ["--experiment-dir", str(args.experiment_dir)]
    if getattr(args, "checkpoint", None) is not None:
        command += ["--checkpoint", str(args.checkpoint)]
    # 复刻 _seg_infer_image_paths 的输入解析优先级：data > image_dir > image，
    # 子进程才能重建同一份图片列表再分片。
    if getattr(args, "data", None) is not None:
        command += ["--data", str(args.data)]
        # infer 的 --split 是单值 choices（非 nargs），传单个值即可。
        command += ["--split", str(args.split)]
    elif getattr(args, "image_dir", None) is not None:
        command += ["--image-dir", str(args.image_dir)]
    elif getattr(args, "image", None) is not None:
        command += ["--image", str(args.image)]
    command += ["--output-dir", str(output_dir), "--device", device]
    command += ["--shard-index", str(shard_index), "--num-shards", str(num_shards)]
    command += ["--skip-important-artifacts"]
    if getattr(args, "threshold", None) is not None:
        command += ["--threshold", str(args.threshold)]
    if getattr(args, "overwrite", False):
        command += ["--overwrite"]
    return command


def run_parallel_seg_infer(args) -> bool:
    if getattr(args, "device", "auto") != "auto":
        return False
    if _is_seg_shard_child(args) or getattr(args, "dry_run", False):
        return False
    all_gpus, message = gpu_parallel.query_gpu_inventory()
    if message:
        print(f"[seg/infer] {message}")
        return False
    eligible = gpu_parallel.filter_high_memory_gpus(all_gpus)
    if len(eligible) < 2:
        print(f"[seg/infer] 可用 GPU 数量为 {len(eligible)}，进入单卡顺序模式。")
        return False
    return _run_parallel_seg_infer_impl(args, eligible)


def _run_parallel_seg_infer_impl(args, eligible) -> bool:
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    # args.output_dir 在 parse_cli_args 阶段已默认成 <experiment_dir>/infer，
    # 与顺序路径 run_infer 一致；这里直接沿用，保证并行/顺序写到同一处。
    final_output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else rt.experiment_dir_from_checkpoint_path(checkpoint_path) / "infer"
    )
    rt.prepare_output_dir(final_output_dir, args.overwrite, clean=True)

    image_paths = _seg_infer_image_paths(args)
    num_shards = min(len(eligible), len(image_paths))
    if num_shards < 2:
        print(f"[seg/infer] 待推理图片数量为 {len(image_paths)}，进入单卡顺序模式。")
        return False
    shard_totals = gpu_parallel.shard_item_totals(
        len(image_paths),
        num_shards=num_shards,
        weights=[float(path.stat().st_size) if path.exists() else 1.0 for path in image_paths],
    )

    shard_root = final_output_dir / "_shards"
    shard_dirs = [shard_root / f"shard_{i:02d}" for i in range(num_shards)]

    jobs: list[tuple[int, int | str, list[str]]] = []
    for i in range(num_shards):
        command = _build_seg_infer_child_command(
            args, shard_index=i, num_shards=num_shards, output_dir=shard_dirs[i], device="auto"
        )
        gpu = eligible[i]
        jobs.append((i, gpu.get("device_token", int(gpu["index"])), command))

    print("[seg/infer] 已进入自动多卡并行模式。")
    for gpu in eligible[:num_shards]:
        print(f"[seg/infer] 保留 {gpu_parallel.format_gpu_summary(gpu)}")
    for shard_index, gpu_index, _ in jobs:
        print(f"[seg/infer] shard={shard_index}/{num_shards} -> CUDA_VISIBLE_DEVICES={gpu_index}")

    failed = gpu_parallel.run_sharded_subprocesses(
        jobs, cwd=rt.ROOT_DIR, shard_totals=shard_totals,
    )
    if failed:
        raise RuntimeError(f"seg infer shard 失败: {', '.join(failed)}")

    for shard_number, shard_dir in enumerate(track(
        shard_dirs,
        label="seg/infer 合并分片产物",
        total=len(shard_dirs),
        unit="shard",
    ), start=1):
        rt.copy_tree_contents(
            shard_dir,
            final_output_dir,
            progress_label=f"seg/infer 合并 {shard_number}/{len(shard_dirs)}",
        )
    shutil.rmtree(shard_root, ignore_errors=True)
    _write_seg_run_meta(
        final_output_dir / "run_meta.json",
        action="infer",
        checkpoint_path=checkpoint_path,
        output_dir=final_output_dir,
        args=args,
        num_images=len(image_paths),
        device_mode="sharded",
    )
    return True


def _merge_parallel_semantic(
    args, shard_dirs, *, splits, final_output_dir, checkpoint_path, data_path
) -> None:
    """按 split 合并各分片的语义混淆矩阵，写出与顺序路径字节一致的 summary/CSV。"""
    wrote_any = False
    total_images = 0
    want_vis = getattr(args, "save_visualization", True)
    vis_max = int(getattr(args, "vis_max_images", rt.SEG_EVAL_VIS_MAX_IMAGES))
    for split in track(
        splits,
        label="seg/eval 合并语义分片",
        total=len(splits),
        unit="split",
    ):
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
            checkpoint_path=checkpoint_path,
            data_path=data_path,
        )
        # shard 已缓存逐像素预测，父进程只挑图和渲染，不再占用 GPU 重跑模型。
        if want_vis:
            _render_semantic_visualizations(
                merged["class_names"], data_path, split, merged["rows"],
                output_dir=split_output_dir, max_images=vis_max,
            )
        _write_seg_run_meta(
            split_output_dir / "run_meta.json",
            action="eval",
            checkpoint_path=checkpoint_path,
            output_dir=split_output_dir,
            args=args,
            num_images=merged["num_samples"],
            device_mode="sharded",
        )
        total_images += merged["num_samples"]
        wrote_any = True
    if not wrote_any:
        raise ValueError("No image/mask pairs found for semantic segmentation eval.")
    if len(splits) > 1:
        _write_seg_run_meta(
            final_output_dir / "run_meta.json",
            action="eval",
            checkpoint_path=checkpoint_path,
            output_dir=final_output_dir,
            args=args,
            num_images=total_images,
            device_mode="sharded",
        )


def _write_instance_summary(
    output_dir: Path,
    *,
    checkpoint_path: Path,
    data_path: Path,
    split: str,
    metric_values: dict[str, Any],
    num_images: int,
    images_with_labels: int,
    infer_time_sum_ms: float,
    failed: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "seg_eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "data": str(data_path),
                "split": split,
                "num_images": num_images,
                "attempted_images": num_images,
                "succeeded_images": max(0, num_images - failed),
                "images_with_labels": images_with_labels,
                "metrics": metric_values,
                "avg_infer_time_ms": infer_time_sum_ms / max(num_images - failed, 1),
                "failed_images": failed,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Summary saved to: {summary_path}")
    return summary_path


def _merge_parallel_instance(args, shard_dirs, *, splits, final_output_dir, checkpoint_path) -> None:
    """按 split 合并实例分割分片，分别重算全局 mAP 并落盘。"""
    total_images = 0
    wrote_any = False
    for split in track(
        splits,
        label="seg/eval 合并实例分片",
        total=len(splits),
        unit="split",
    ):
        split_shard_dirs = [
            sd / split
            for sd in shard_dirs
            if (sd / split / "seg_instance_shard_result.json").exists()
        ]
        if not split_shard_dirs:
            continue
        merged = _merge_instance_shard_results(split_shard_dirs, classwise=args.classwise)
        split_output_dir = final_output_dir / split if len(splits) > 1 else final_output_dir
        _write_instance_summary(
            split_output_dir,
            checkpoint_path=checkpoint_path,
            data_path=Path(args.data),
            split=split,
            metric_values=merged["metric_values"],
            num_images=merged["num_images"],
            images_with_labels=merged["images_with_labels"],
            infer_time_sum_ms=merged["infer_time_sum_ms"],
            failed=merged["failed"],
        )
        _write_seg_run_meta(
            split_output_dir / "run_meta.json",
            action="eval",
            checkpoint_path=checkpoint_path,
            output_dir=split_output_dir,
            args=args,
            num_images=merged["num_images"],
            device_mode="sharded",
        )
        total_images += merged["num_images"]
        wrote_any = True
    if not wrote_any:
        raise ValueError("No image/label pairs found for instance segmentation eval.")
    if len(splits) > 1:
        _write_seg_run_meta(
            final_output_dir / "run_meta.json",
            action="eval",
            checkpoint_path=checkpoint_path,
            output_dir=final_output_dir,
            args=args,
            num_images=total_images,
            device_mode="sharded",
        )


def run_eval(args) -> None:
    if _is_seg_shard_child(args) or getattr(args, "dry_run", False):
        _run_eval_impl(args)
        return
    if _seg_reuse_precheck(args, "eval") == run_reuse.REUSE:
        print(f"[seg/eval] 已复用上次评估结果，未重新评估：{args.output_dir}")
        return
    final_output_dir = Path(args.output_dir).expanduser().resolve()
    original_output_dir = args.output_dir
    original_overwrite = args.overwrite
    try:
        with staged_directory(final_output_dir, overwrite=args.overwrite) as stage:
            args.output_dir = stage
            args.overwrite = True
            args._published_output_dir = final_output_dir
            _run_eval_impl(args)
    finally:
        args.output_dir = original_output_dir
        args.overwrite = original_overwrite
        if hasattr(args, "_published_output_dir"):
            delattr(args, "_published_output_dir")


def _run_eval_impl(args) -> None:
    seg_type = _normalize_seg_type(args)
    # 自动多卡：device=auto 且 ≥2 张空闲卡时分片到子进程，否则回退顺序模式（返回 False）。
    # 子进程通过 launcher.py eval 重入本函数，_is_seg_shard_child 为真而跳过并行、直走分片写盘。
    if not _is_seg_shard_child(args) and run_parallel_seg_eval(args):
        return

    if seg_type == "semantic":
        run_semantic_eval(args)
        return

    action = "eval"
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    output_dir = args.output_dir
    splits = _normalize_splits(args.split)
    if getattr(args, "dry_run", False):
        print(f"[seg/{action}] dry-run 计划")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  seg_train_type: {seg_type}")
        print(f"  split: {getattr(args, 'split', None)}")
        print(f"  output_dir: {output_dir}")
        try:
            data_cfg = rt.load_data_config(args.data)
            for split in splits:
                samples, _ = rt.list_dataset_samples(data_cfg=data_cfg, split=split)
                print(f"  num_images[{split}]: {len(samples)}")
        except Exception:  # noqa: BLE001 dry-run 仅尽力打印，拿不到数量就跳过
            pass
        return
    if len(splits) > 1:
        rt.prepare_output_dir(output_dir, args.overwrite, clean=True)
    data_cfg = rt.load_data_config(args.data)

    device_mode = args.device
    resolved_device_arg = args.device
    if args.device == "auto" and not _is_seg_shard_child(args):
        all_gpus, message = gpu_parallel.query_gpu_inventory()
        if message:
            print(f"[seg/{action}] {message}")
        else:
            eligible = gpu_parallel.filter_high_memory_gpus(all_gpus)
            chosen, choose_msg = gpu_parallel.select_fallback_single_gpu(
                all_gpus, eligible, component=f"seg/{action}"
            )
            print(f"[seg/{action}] {choose_msg}")
            if chosen is not None:
                resolved_device_arg = chosen
                device_mode = chosen

    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(resolved_device_arg))
    model.eval()

    # 分片子进程逐 split 跑本分片，把序列化预测/标注写盘，由父进程分别合并全局 mAP。
    want_vis = getattr(args, "save_visualization", True)
    vis_threshold = float(getattr(args, "threshold", None) or rt.DEFAULT_SEG_THRESHOLD)
    if _is_seg_shard_child(args):
        for split in splits:
            accumulated = _accumulate_instance_eval(
                model, data_cfg, split, args.classwise,
                shard_index=args.shard_index, num_shards=args.num_shards,
                vis_dir=(
                    _eval_compare_dir(
                        args.output_dir,
                        split=split,
                        multi_split=len(splits) > 1,
                    )
                    if want_vis
                    else None
                ),
                vis_threshold=vis_threshold,
                entries_path=args.output_dir / split / "seg_instance_entries.jsonl",
            )
            _write_instance_shard_result(
                args.output_dir / split,
                split=split,
                class_names=accumulated["class_names"],
                entries=accumulated["entries"],
                entries_path=args.output_dir / split / "seg_instance_entries.jsonl",
                images_with_labels=accumulated["images_with_labels"],
                num_images=accumulated["num_images"],
                infer_time_sum_ms=accumulated["infer_time_sum_ms"],
                failed=accumulated["failed"],
            )
        return

    total_images = 0
    for split in splits:
        split_output_dir = output_dir / split if len(splits) > 1 else output_dir
        rt.prepare_output_dir(split_output_dir, args.overwrite, clean=True)
        accumulated = _accumulate_instance_eval(
            model, data_cfg, split, args.classwise,
            vis_dir=_eval_compare_dir(split_output_dir) if want_vis else None,
            vis_threshold=vis_threshold,
        )
        num_images = accumulated["num_images"]
        print(f"[seg/eval] 阶段: 计算 {split} 聚合指标", flush=True)
        result = accumulated["metric"].compute_aggregated_values().metric_values
        print(f"[seg/eval] 阶段完成: {split} 聚合指标", flush=True)
        _write_instance_summary(
            split_output_dir,
            checkpoint_path=checkpoint_path,
            data_path=Path(args.data),
            split=split,
            metric_values=result,
            num_images=num_images,
            images_with_labels=accumulated["images_with_labels"],
            infer_time_sum_ms=accumulated["infer_time_sum_ms"],
            failed=accumulated["failed"],
        )
        _write_seg_run_meta(
            split_output_dir / "run_meta.json",
            action=action, checkpoint_path=checkpoint_path,
            output_dir=split_output_dir, args=args, num_images=num_images, device_mode=device_mode,
        )
        total_images += num_images

    if len(splits) > 1:
        _write_seg_run_meta(
            output_dir / "run_meta.json",
            action=action,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            args=args,
            num_images=total_images,
            device_mode=device_mode,
        )
