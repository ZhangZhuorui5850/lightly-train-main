"""检测任务分析与导出筛选逻辑。

这里放 report 分析、类别筛选、候选样本选择、重划分等逻辑。
"""

from __future__ import annotations

import dataclasses
import heapq
from collections import Counter, defaultdict
from math import ceil, log1p, sqrt
from pathlib import Path
from typing import Any, Callable

from .det_shared import (
    ExportImageCandidate,
    SourceImageInfo,
    safe_class_name,
)


def collect_class_split_stats(source_infos_by_split: dict[str, list[SourceImageInfo]]) -> dict[int, dict[str, dict[str, int]]]:
    stats: dict[int, dict[str, dict[str, int]]] = defaultdict(lambda: {"all": {"images": 0, "boxes": 0}})
    for split_name, infos in source_infos_by_split.items():
        for info in infos:
            for class_id, box_count in info.class_box_counts.items():
                split_stats = stats[class_id].setdefault(split_name, {"images": 0, "boxes": 0})
                split_stats["images"] += 1
                split_stats["boxes"] += box_count
                stats[class_id]["all"]["images"] += 1
                stats[class_id]["all"]["boxes"] += box_count
    return {class_id: dict(split_stats) for class_id, split_stats in stats.items()}


def collect_candidate_split_stats(candidates_by_split: dict[str, list[ExportImageCandidate]]) -> dict[int, dict[str, dict[str, int]]]:
    stats: dict[int, dict[str, dict[str, int]]] = defaultdict(lambda: {"all": {"images": 0, "boxes": 0}})
    for split_name, candidates in candidates_by_split.items():
        for candidate in candidates:
            for class_id, box_count in candidate.class_box_counts.items():
                split_stats = stats[class_id].setdefault(split_name, {"images": 0, "boxes": 0})
                split_stats["images"] += 1
                split_stats["boxes"] += box_count
                stats[class_id]["all"]["images"] += 1
                stats[class_id]["all"]["boxes"] += box_count
    return {class_id: dict(split_stats) for class_id, split_stats in stats.items()}

def analyze_report_signal(per_class_ap: dict[str, Any]) -> dict[str, Any]:
    class_entries: list[dict[str, float]] = []
    for class_id, info in per_class_ap.items():
        if not isinstance(info, dict):
            continue
        gt_count = int(info.get("gt", 0) or 0)
        ap_value = float(info.get("ap", 0.0) or 0.0)
        class_entries.append({"class_id": float(class_id), "gt": gt_count, "ap": ap_value})
    positive_gt_entries = [entry for entry in class_entries if entry["gt"] > 0]
    positive_ap_classes = sum(1 for entry in positive_gt_entries if entry["ap"] > 0.0)
    max_ap = max((entry["ap"] for entry in positive_gt_entries), default=0.0)
    mean_ap = (
        sum(entry["ap"] for entry in positive_gt_entries) / len(positive_gt_entries)
        if positive_gt_entries
        else 0.0
    )
    signal = "usable" if positive_ap_classes >= 2 or max_ap >= 0.05 or mean_ap >= 0.02 else "weak"
    return {
        "signal": signal,
        "classes_with_gt": len(positive_gt_entries),
        "classes_with_positive_ap": positive_ap_classes,
        "max_ap": max_ap,
        "mean_ap": mean_ap,
    }


def percentile_float(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    clamped_quantile = min(max(quantile, 0.0), 1.0)
    index = min(len(ordered) - 1, max(0, int(ceil((len(ordered) - 1) * clamped_quantile))))
    return float(ordered[index])


def derive_selection_pressure(
    *,
    target_total_images: int,
    available_total_images: int,
) -> tuple[int, float, float, dict[str, int]]:
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=available_total_images,
    )
    keep_fraction = (
        min(1.0, effective_target_total_images / available_total_images)
        if available_total_images > 0 and effective_target_total_images > 0
        else 1.0
    )
    selection_pressure = max(0.0, 1.0 - sqrt(keep_fraction))
    return effective_target_total_images, keep_fraction, selection_pressure, target_total_images_summary

def derive_effective_class_threshold(
    *,
    requested_threshold: float,
    per_class_ap: dict[str, Any],
    auto_relax_class_threshold: bool,
    target_total_images: int,
    available_total_images: int,
) -> tuple[float, dict[str, Any]]:
    report_signal = analyze_report_signal(per_class_ap)
    positive_ap_values: list[float] = []
    for info in per_class_ap.values():
        if not isinstance(info, dict):
            continue
        gt_count = int(info.get("gt", 0) or 0)
        if gt_count <= 0:
            continue
        positive_ap_values.append(float(info.get("ap", 0.0) or 0.0))

    adaptive_threshold = 0.0
    adaptive_quantile = 0.0
    effective_target_total_images = 0
    keep_fraction = 1.0
    selection_pressure = 0.0
    target_total_images_summary = {
        "requested_target_total_images": max(target_total_images, 0),
        "effective_target_total_images": 0,
        "available_total_images": available_total_images,
    }
    if auto_relax_class_threshold and report_signal["signal"] == "usable" and positive_ap_values:
        (
            effective_target_total_images,
            keep_fraction,
            selection_pressure,
            target_total_images_summary,
        ) = derive_selection_pressure(
            target_total_images=target_total_images,
            available_total_images=available_total_images,
        )
        adaptive_quantile = 0.2 + (0.55 * selection_pressure)
        adaptive_threshold = percentile_float(positive_ap_values, adaptive_quantile)

    if requested_threshold > 0.0:
        effective_threshold = max(requested_threshold, adaptive_threshold) if adaptive_threshold > 0.0 else requested_threshold
        report_filter_mode = "manual_adaptive_floor" if adaptive_threshold > requested_threshold else "manual"
    elif adaptive_threshold > 0.0:
        effective_threshold = adaptive_threshold
        report_filter_mode = "adaptive"
    else:
        effective_threshold = 0.0
        report_filter_mode = "advisory"
    relaxed = auto_relax_class_threshold and adaptive_threshold > requested_threshold
    report_signal["requested_threshold"] = requested_threshold
    report_signal["effective_threshold"] = effective_threshold
    report_signal["threshold_relaxed"] = relaxed
    report_signal["report_filter_mode"] = report_filter_mode
    report_signal["adaptive_threshold"] = adaptive_threshold
    report_signal["adaptive_threshold_quantile"] = adaptive_quantile
    report_signal["effective_target_total_images"] = effective_target_total_images
    report_signal["keep_fraction"] = keep_fraction
    report_signal["selection_pressure"] = selection_pressure
    report_signal["target_total_images_summary"] = target_total_images_summary
    return effective_threshold, report_signal


def derive_effective_balance_ratio(
    *,
    available_images_by_class: dict[int, int],
    available_boxes_by_class: dict[int, int],
    requested_balance_ratio: float,
    auto_balance: bool,
    target_total_images: int,
    available_total_images: int,
) -> tuple[float, dict[str, Any]]:
    requested = max(float(requested_balance_ratio), 0.0)
    positive_image_counts = [count for count in available_images_by_class.values() if count > 0]
    positive_box_counts = [count for count in available_boxes_by_class.values() if count > 0]
    if requested > 0.0 or not auto_balance:
        effective_ratio = requested
        return effective_ratio, {
            "requested_balance_ratio": requested,
            "effective_balance_ratio": effective_ratio,
            "auto_balance_ratio": 0.0,
            "auto_balance": auto_balance,
            "available_total_images": available_total_images,
            "positive_class_count": len(positive_image_counts),
            "mode": "manual" if requested > 0.0 else "disabled",
        }

    (
        effective_target_total_images,
        keep_fraction,
        selection_pressure,
        target_total_images_summary,
    ) = derive_selection_pressure(
        target_total_images=target_total_images,
        available_total_images=available_total_images,
    )
    positive_class_count = max(len(positive_image_counts), 1)
    budget_images_per_class = (
        effective_target_total_images / positive_class_count
        if effective_target_total_images > 0
        else float(percentile_int(positive_image_counts, 0.5))
    )
    image_spread = (
        percentile_int(positive_image_counts, 0.75) / max(percentile_int(positive_image_counts, 0.25), 1)
        if positive_image_counts
        else 1.0
    )
    box_spread = (
        percentile_int(positive_box_counts, 0.75) / max(percentile_int(positive_box_counts, 0.25), 1)
        if positive_box_counts
        else 1.0
    )
    spread_ratio = max(image_spread, box_spread, 1.0)
    base_ratio = 2.0 + (0.9 * log1p(max(budget_images_per_class, 1.0))) + (1.1 * log1p(spread_ratio))
    pressure_scale = max(0.45, 1.0 - (0.55 * selection_pressure))
    # 上限收紧到 4.0：以前允许多数类最多是少数类的 12 倍，对"类别间尽量均衡"
    # 来说太松。改成 4× 后，多数类会被降采样向尾部类靠拢，而不是放任其膨胀。
    auto_ratio = min(4.0, max(1.5, round(base_ratio * pressure_scale, 4)))
    return auto_ratio, {
        "requested_balance_ratio": requested,
        "effective_balance_ratio": auto_ratio,
        "auto_balance_ratio": auto_ratio,
        "auto_balance": auto_balance,
        "available_total_images": available_total_images,
        "positive_class_count": positive_class_count,
        "budget_images_per_class": budget_images_per_class,
        "image_spread_ratio": image_spread,
        "box_spread_ratio": box_spread,
        "spread_ratio": spread_ratio,
        "effective_target_total_images": effective_target_total_images,
        "keep_fraction": keep_fraction,
        "selection_pressure": selection_pressure,
        "target_total_images_summary": target_total_images_summary,
        "mode": "adaptive",
    }

def derive_effective_min_class_boxes(
    *,
    available_boxes_by_class: dict[int, int],
    available_images_by_class: dict[int, int],
    requested_min_class_boxes: int,
    auto_balance: bool,
    balance_ratio: float,
    target_total_images: int,
    estimated_class_count: int,
) -> tuple[int, dict[str, Any]]:
    positive_box_counts = [count for count in available_boxes_by_class.values() if count > 0]
    positive_image_counts = [count for count in available_images_by_class.values() if count > 0]
    max_available_boxes = max(positive_box_counts, default=0)
    reference_available_boxes = percentile_int(positive_box_counts, 0.75) if positive_box_counts else 0
    available_total_boxes = sum(positive_box_counts)
    available_total_images = sum(positive_image_counts)
    average_boxes_per_image = (
        available_total_boxes / available_total_images
        if available_total_boxes > 0 and available_total_images > 0
        else 0.0
    )
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=available_total_images,
    )
    effective_class_count = max(estimated_class_count, len(positive_box_counts), 1)
    budget_boxes_per_class = (
        (effective_target_total_images * average_boxes_per_image) / effective_class_count
        if effective_target_total_images > 0 and average_boxes_per_image > 0.0
        else 0.0
    )
    # 低分位(p10)锚点：删类的目标不再是"向 p75 多数类看齐"，而是只剔除连均衡
    # 窗口下沿都够不到的极端少数类。剩余类别由选图阶段把多数类降采样向尾部
    # 靠拢来实现均衡，而不是把尾部类删掉。同时该下沿 = p10/ratio 也保证保留类
    # 的最小框数不会把 min_available*ratio 的窗口上限压垮（防坍缩）。
    low_anchor_boxes = percentile_int(positive_box_counts, 0.10) if positive_box_counts else 0
    auto_min_class_boxes = 0
    if auto_balance and balance_ratio > 0 and low_anchor_boxes > 0:
        auto_min_class_boxes = max(1, int(ceil(low_anchor_boxes / balance_ratio)))
    effective_min_class_boxes = max(max(requested_min_class_boxes, 0), auto_min_class_boxes)
    box_tolerance = (
        max(
            3,
            int(
                ceil(
                    min(
                        max(average_boxes_per_image, 1.0),
                        max(effective_min_class_boxes * 0.1, 3.0),
                    )
                )
            ),
        )
        if effective_min_class_boxes > 0
        else 0
    )
    soft_min_class_boxes = max(1, effective_min_class_boxes - box_tolerance) if effective_min_class_boxes > 0 else 0
    return effective_min_class_boxes, {
        "requested_min_class_boxes": max(requested_min_class_boxes, 0),
        "auto_min_class_boxes": auto_min_class_boxes,
        "effective_min_class_boxes": effective_min_class_boxes,
        "soft_min_class_boxes": soft_min_class_boxes,
        "box_tolerance": box_tolerance,
        "max_available_boxes": max_available_boxes,
        "reference_available_boxes": reference_available_boxes,
        "reference_quantile": 0.75,
        "low_anchor_boxes_p10": low_anchor_boxes,
        "available_total_boxes": available_total_boxes,
        "available_total_images": available_total_images,
        "average_boxes_per_image": average_boxes_per_image,
        "effective_target_total_images": effective_target_total_images,
        "budget_boxes_per_class": budget_boxes_per_class,
        "estimated_class_count": effective_class_count,
        "target_total_images_summary": target_total_images_summary,
        "balance_ratio": balance_ratio,
        "auto_balance": auto_balance,
    }

def derive_effective_target_total_images(
    *,
    requested_target_total_images: int,
    available_total_images: int,
) -> tuple[int, dict[str, int]]:
    requested = max(requested_target_total_images, 0)
    if requested <= 0:
        return 0, {
            "requested_target_total_images": requested,
            "effective_target_total_images": 0,
            "available_total_images": available_total_images,
        }
    return min(requested, available_total_images), {
        "requested_target_total_images": requested,
        "effective_target_total_images": min(requested, available_total_images),
        "available_total_images": available_total_images,
    }

def derive_effective_min_class_images(
    *,
    available_images_by_class: dict[int, int],
    requested_min_class_images: int,
    auto_balance: bool,
    balance_ratio: float,
    target_total_images: int,
    split_ratio: str,
    estimated_class_count: int,
) -> tuple[int, dict[str, Any]]:
    requested = max(requested_min_class_images, 0)
    positive_image_counts = [count for count in available_images_by_class.values() if count > 0]
    available_total_images = sum(positive_image_counts)
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=available_total_images,
    )
    ratio_values = parse_split_ratio(split_ratio)
    if effective_target_total_images > 0:
        split_targets = allocate_split_targets_from_source(
            total_selected_images=effective_target_total_images,
            split_ratio=split_ratio,
        )
        active_splits = {split_name: split_targets.get(split_name, 0) > 0 for split_name in ("train", "val", "test")}
    else:
        split_targets = {split_name: 0 for split_name in ("train", "val", "test")}
        active_splits = {split_name: ratio_values.get(split_name, 0.0) > 0 for split_name in ("train", "val", "test")}
    active_split_count = sum(1 for is_active in active_splits.values() if is_active)
    # split 地板降到"只需在 train 里出现 1 张"。以前的 train=3/val=1/test=1（合计 5）
    # 是把"能填满每个 split"误当成"必须保留"的判据，导致只有几张图的尾部类被直接
    # 删掉。val/test 缺图只是该类在那里不被评测，不构成删类理由。
    split_floor = {
        "train": 1 if active_splits.get("train", False) else 0,
        "val": 0,
        "test": 0,
    }
    auto_min_images_from_split = sum(split_floor.values())
    effective_class_count = max(estimated_class_count, len(positive_image_counts), 1)
    # 低分位(p10)锚点：与框数下沿同理，只剔除连均衡窗口下沿都够不到的极端少数类，
    # 而不是按 平均目标/ratio 把大量中小类一并删掉。
    low_anchor_images = percentile_int(positive_image_counts, 0.10) if positive_image_counts else 0
    auto_min_images_from_budget = 0
    if auto_balance and balance_ratio > 0 and low_anchor_images > 0:
        auto_min_images_from_budget = max(1, int(ceil(low_anchor_images / balance_ratio)))
    auto_min_class_images = (
        max(auto_min_images_from_split, auto_min_images_from_budget)
        if auto_balance
        else requested
    )
    effective_min_class_images = max(requested, auto_min_class_images)
    soft_min_class_images = (
        max(1, effective_min_class_images - max(2, int(ceil(effective_min_class_images * 0.15))))
        if effective_min_class_images > 0
        else 0
    )
    return effective_min_class_images, {
        "requested_min_class_images": requested,
        "auto_min_images_from_split": auto_min_images_from_split,
        "auto_min_images_from_budget": auto_min_images_from_budget,
        "low_anchor_images_p10": low_anchor_images,
        "effective_min_class_images": effective_min_class_images,
        "soft_min_class_images": soft_min_class_images,
        "estimated_class_count": effective_class_count,
        "available_images_by_class": available_images_by_class,
        "available_total_images": available_total_images,
        "active_split_count": active_split_count,
        "split_floor": split_floor,
        "split_targets_for_budget": split_targets,
        "target_total_images_summary": target_total_images_summary,
        "balance_ratio": balance_ratio,
        "auto_balance": auto_balance,
    }


def derive_effective_target_images_per_class(
    *,
    available_images_by_class: dict[int, int],
    requested_target_images_per_class: int,
    auto_balance: bool,
    target_total_images: int,
    estimated_class_count: int,
    balance_ratio: float,
) -> tuple[int, dict[str, Any]]:
    requested = max(requested_target_images_per_class, 0)
    positive_counts = [count for count in available_images_by_class.values() if count > 0]
    available_total_images = sum(positive_counts)
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=available_total_images,
    )
    average_target_images_per_class = (
        effective_target_total_images / max(estimated_class_count, 1)
        if effective_target_total_images > 0 and estimated_class_count > 0
        else 0.0
    )
    auto_target_images_per_class = 0
    if auto_balance and average_target_images_per_class > 0.0:
        slack_factor = 1.0 + min(0.35, 1.0 / max(balance_ratio, 1.0))
        auto_target_images_per_class = max(1, int(ceil(average_target_images_per_class * slack_factor)))
    effective_target_images_per_class = requested if requested > 0 else auto_target_images_per_class
    if effective_target_images_per_class > 0 and positive_counts:
        effective_target_images_per_class = min(effective_target_images_per_class, max(positive_counts))
    return effective_target_images_per_class, {
        "requested_target_images_per_class": requested,
        "auto_target_images_per_class": auto_target_images_per_class,
        "effective_target_images_per_class": effective_target_images_per_class,
        "estimated_class_count": max(estimated_class_count, 1),
        "average_target_images_per_class": average_target_images_per_class,
        "available_total_images": available_total_images,
        "balance_ratio": balance_ratio,
        "target_total_images_summary": target_total_images_summary,
        "auto_balance": auto_balance,
    }

def percentile_int(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    clamped_quantile = min(max(quantile, 0.0), 1.0)
    index = min(len(ordered) - 1, max(0, int(ceil((len(ordered) - 1) * clamped_quantile))))
    return int(ordered[index])

def derive_effective_density_controls(
    *,
    candidates: list[ExportImageCandidate],
    requested_max_boxes_per_image: int,
    requested_max_boxes_per_class_per_image: int,
    requested_box_density_penalty: float,
    auto_balance: bool,
    target_total_images: int,
) -> tuple[int, int, float, dict[str, Any]]:
    requested_box_density_penalty_value = max(float(requested_box_density_penalty), 0.0)
    available_total_images = len(candidates)
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=available_total_images,
    )
    keep_fraction = (
        min(1.0, effective_target_total_images / available_total_images)
        if available_total_images > 0 and effective_target_total_images > 0
        else 1.0
    )
    selection_pressure = max(0.0, 1.0 - keep_fraction)
    total_boxes_values = [candidate.total_boxes for candidate in candidates]
    max_class_boxes_values = [
        max(candidate.class_box_counts.values(), default=0)
        for candidate in candidates
    ]
    total_boxes_stats = {
        "min": min(total_boxes_values, default=0),
        "median": percentile_int(total_boxes_values, 0.5),
        "p90": percentile_int(total_boxes_values, 0.9),
        "p95": percentile_int(total_boxes_values, 0.95),
        "p98": percentile_int(total_boxes_values, 0.98),
        "max": max(total_boxes_values, default=0),
    }
    max_class_boxes_stats = {
        "min": min(max_class_boxes_values, default=0),
        "median": percentile_int(max_class_boxes_values, 0.5),
        "p90": percentile_int(max_class_boxes_values, 0.9),
        "p95": percentile_int(max_class_boxes_values, 0.95),
        "p98": percentile_int(max_class_boxes_values, 0.98),
        "max": max(max_class_boxes_values, default=0),
    }

    density_quantile = 0.95
    if keep_fraction < 0.5:
        density_quantile = 0.98
    elif keep_fraction < 0.8:
        density_quantile = 0.97

    spread_boxes = max(total_boxes_stats["p90"] - total_boxes_stats["median"], 0)
    spread_class_boxes = max(max_class_boxes_stats["p90"] - max_class_boxes_stats["median"], 0)
    auto_max_boxes_per_image = max(
        percentile_int(total_boxes_values, density_quantile),
        total_boxes_stats["p95"],
        total_boxes_stats["median"] + max(2, int(ceil(spread_boxes * 2.0))),
    )
    auto_max_boxes_per_class_per_image = max(
        percentile_int(max_class_boxes_values, density_quantile),
        max_class_boxes_stats["p95"],
        max_class_boxes_stats["median"] + max(1, int(ceil(spread_class_boxes * 2.0))),
    )
    use_auto_density_limits = (
        auto_balance
        and selection_pressure > 0.0
        and total_boxes_stats["max"] > total_boxes_stats["p95"]
    )
    effective_max_boxes_per_image = max(requested_max_boxes_per_image, 0)
    effective_max_boxes_per_class_per_image = max(requested_max_boxes_per_class_per_image, 0)
    if use_auto_density_limits and effective_max_boxes_per_image <= 0 and auto_max_boxes_per_image < total_boxes_stats["max"]:
        effective_max_boxes_per_image = max(auto_max_boxes_per_image, 1)
    if (
        use_auto_density_limits
        and effective_max_boxes_per_class_per_image <= 0
        and auto_max_boxes_per_class_per_image < max_class_boxes_stats["max"]
    ):
        effective_max_boxes_per_class_per_image = max(auto_max_boxes_per_class_per_image, 1)
    spread_ratio = (
        (total_boxes_stats["p90"] - total_boxes_stats["median"]) / max(total_boxes_stats["median"], 1)
        if total_boxes_stats["p90"] > 0
        else 0.0
    )
    auto_box_density_penalty = min(
        0.95,
        0.25 + (0.40 * selection_pressure) + (0.20 * min(spread_ratio, 3.0) / 3.0),
    )
    effective_box_density_penalty = (
        max(requested_box_density_penalty_value, auto_box_density_penalty)
        if auto_balance
        else requested_box_density_penalty_value
    )
    return (
        effective_max_boxes_per_image,
        effective_max_boxes_per_class_per_image,
        effective_box_density_penalty,
        {
            "requested_max_boxes_per_image": max(requested_max_boxes_per_image, 0),
            "requested_max_boxes_per_class_per_image": max(requested_max_boxes_per_class_per_image, 0),
            "requested_box_density_penalty": requested_box_density_penalty_value,
            "effective_max_boxes_per_image": effective_max_boxes_per_image,
            "effective_max_boxes_per_class_per_image": effective_max_boxes_per_class_per_image,
            "effective_box_density_penalty": effective_box_density_penalty,
            "auto_max_boxes_per_image": auto_max_boxes_per_image,
            "auto_max_boxes_per_class_per_image": auto_max_boxes_per_class_per_image,
            "auto_box_density_penalty": auto_box_density_penalty,
            "available_total_images": available_total_images,
            "effective_target_total_images": effective_target_total_images,
            "target_total_images_summary": target_total_images_summary,
            "keep_fraction": keep_fraction,
            "selection_pressure": selection_pressure,
            "density_quantile": density_quantile,
            "total_boxes_per_image_stats": total_boxes_stats,
            "max_class_boxes_per_image_stats": max_class_boxes_stats,
        },
    )

def choose_kept_class_ids(
    *,
    class_ids: list[int],
    class_names: dict[int, str],
    class_stats: dict[int, dict[str, dict[str, int]]],
    per_class_ap: dict[str, Any],
    class_threshold: float,
    requested_class_threshold: float,
    min_class_images: int,
    min_class_boxes: int,
    soft_min_class_images: int | None,
    soft_min_class_boxes: int | None,
    reference_split: str,
    report_filter_mode: str,
) -> tuple[list[int], dict[int, dict[str, Any]]]:
    decisions: dict[int, dict[str, Any]] = {}
    kept_class_ids: list[int] = []
    effective_soft_min_class_images = (
        soft_min_class_images
        if soft_min_class_images is not None
        else (
            max(1, min_class_images - max(2, int(ceil(min_class_images * 0.15))))
            if min_class_images > 0
            else 0
        )
    )
    effective_soft_min_class_boxes = (
        soft_min_class_boxes
        if soft_min_class_boxes is not None
        else (
            max(1, min_class_boxes - max(3, int(ceil(min_class_boxes * 0.1))))
            if min_class_boxes > 0
            else 0
        )
    )
    for class_id in class_ids:
        stats = class_stats.get(class_id, {})
        reference_stats = stats.get(reference_split, {"images": 0, "boxes": 0})
        all_stats = stats.get("all", {"images": 0, "boxes": 0})
        ap_info = per_class_ap.get(str(class_id), {})
        ap_value = float(ap_info.get("ap", 0.0)) if isinstance(ap_info, dict) else 0.0
        reasons: list[str] = []
        advisory_notes: list[str] = []
        if report_filter_mode == "strict" and ap_value < class_threshold:
            reasons.append("low_ap")
        elif requested_class_threshold > 0.0 and ap_value < requested_class_threshold:
            advisory_notes.append("low_ap_reference")
        if min_class_images > 0 and reference_stats["images"] < effective_soft_min_class_images:
            reasons.append(f"too_few_{reference_split}_images")
        elif min_class_images > 0 and reference_stats["images"] < min_class_images:
            advisory_notes.append("low_image_volume_reference")
        if min_class_boxes > 0 and reference_stats["boxes"] < effective_soft_min_class_boxes:
            reasons.append(f"too_few_{reference_split}_boxes")
        elif min_class_boxes > 0 and reference_stats["boxes"] < min_class_boxes:
            advisory_notes.append("low_box_volume_reference")
        if all_stats["boxes"] <= 0:
            reasons.append("no_boxes")
        decisions[class_id] = {
            "name": safe_class_name(class_names, class_id),
            "ap": ap_value,
            "reference_split": reference_split,
            "reference_images": reference_stats["images"],
            "reference_boxes": reference_stats["boxes"],
            "soft_min_class_images": effective_soft_min_class_images,
            "soft_min_class_boxes": effective_soft_min_class_boxes,
            "all_images": all_stats["images"],
            "all_boxes": all_stats["boxes"],
            "advisory_notes": advisory_notes,
            "drop_reasons": reasons,
        }
        if not reasons:
            kept_class_ids.append(class_id)
    return kept_class_ids, decisions

def primary_drop_reason(drop_reasons: list[str]) -> str:
    priority = {
        "no_boxes": 0,
        "low_ap": 1,
        "too_few_all_images": 2,
        "too_few_train_images": 2,
        "too_few_val_images": 2,
        "too_few_test_images": 2,
        "too_few_all_boxes": 3,
        "too_few_train_boxes": 3,
        "too_few_val_boxes": 3,
        "too_few_test_boxes": 3,
    }
    if not drop_reasons:
        return ""
    return min(drop_reasons, key=lambda reason: priority.get(reason, 99))

def format_reference_split_label(reference_split: str) -> str:
    return "全量" if reference_split == "all" else reference_split

def build_drop_reason_details(
    *,
    decision: dict[str, Any],
    effective_min_class_images: int,
    effective_min_class_boxes: int,
    requested_class_threshold: float,
    effective_class_threshold: float,
    report_filter_mode: str,
    reference_split: str,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    reference_images = int(decision.get("reference_images", 0) or 0)
    reference_boxes = int(decision.get("reference_boxes", 0) or 0)
    soft_min_class_images = int(decision.get("soft_min_class_images", 0) or 0)
    soft_min_class_boxes = int(decision.get("soft_min_class_boxes", 0) or 0)
    ap_value = float(decision.get("ap", 0.0) or 0.0)
    for reason in decision.get("drop_reasons", []):
        if reason == "low_ap":
            details.append(
                {
                    "reason": reason,
                    "category": "report_quality",
                    "metric": "ap",
                    "current": ap_value,
                    "required": effective_class_threshold,
                    "gap": max(effective_class_threshold - ap_value, 0.0),
                    "report_filter_mode": report_filter_mode,
                    "requested_threshold": requested_class_threshold,
                }
            )
            continue
        if reason == "no_boxes":
            details.append(
                {
                    "reason": reason,
                    "category": "label_volume",
                    "metric": "boxes",
                    "current": 0,
                    "required": max(effective_min_class_boxes, 1),
                    "gap": max(max(effective_min_class_boxes, 1), 1),
                }
            )
            continue
        if reason.endswith("_images"):
            required_images = soft_min_class_images if soft_min_class_images > 0 else effective_min_class_images
            details.append(
                {
                    "reason": reason,
                    "category": "image_volume",
                    "metric": "images",
                    "reference_split": reference_split,
                    "current": reference_images,
                    "required": required_images,
                    "effective_reference_floor": effective_min_class_images,
                    "gap": max(required_images - reference_images, 0),
                }
            )
            continue
        if reason.endswith("_boxes"):
            required_boxes = soft_min_class_boxes if soft_min_class_boxes > 0 else effective_min_class_boxes
            details.append(
                {
                    "reason": reason,
                    "category": "box_volume",
                    "metric": "boxes",
                    "reference_split": reference_split,
                    "current": reference_boxes,
                    "required": required_boxes,
                    "effective_reference_floor": effective_min_class_boxes,
                    "gap": max(required_boxes - reference_boxes, 0),
                }
            )
    return details

def build_dropped_class_analysis(
    *,
    class_decisions: dict[int, dict[str, Any]],
    kept_class_ids: list[int],
    class_names: dict[int, str],
    reference_split: str,
    balance_ratio: float,
    effective_min_class_images: int,
    effective_min_class_boxes: int,
    requested_class_threshold: float,
    effective_class_threshold: float,
    report_filter_mode: str,
    min_images_summary: dict[str, Any],
    min_boxes_summary: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reference_split_label = format_reference_split_label(reference_split)
    kept_reference_images = [
        int(class_decisions[class_id].get("reference_images", 0) or 0)
        for class_id in kept_class_ids
        if class_id in class_decisions
    ]
    kept_reference_boxes = [
        int(class_decisions[class_id].get("reference_boxes", 0) or 0)
        for class_id in kept_class_ids
        if class_id in class_decisions
    ]
    max_kept_images = max(kept_reference_images, default=0)
    max_kept_boxes = max(kept_reference_boxes, default=0)
    active_split_count = int(min_images_summary.get("active_split_count", 0) or 0)
    reason_histogram: Counter[str] = Counter()
    dropped_analysis: dict[str, Any] = {}
    borderline_kept: dict[str, Any] = {}

    for class_id, decision in class_decisions.items():
        reference_images = int(decision.get("reference_images", 0) or 0)
        reference_boxes = int(decision.get("reference_boxes", 0) or 0)
        soft_min_class_images = int(decision.get("soft_min_class_images", 0) or 0)
        soft_min_class_boxes = int(decision.get("soft_min_class_boxes", 0) or 0)
        drop_reasons = list(decision.get("drop_reasons", []))
        advisory_notes = list(decision.get("advisory_notes", []))
        for reason in drop_reasons:
            reason_histogram[reason] += 1

        image_drop_threshold = (
            soft_min_class_images
            if any(reason.endswith("_images") for reason in drop_reasons) and soft_min_class_images > 0
            else effective_min_class_images
        )
        image_gap = max(image_drop_threshold - reference_images, 0)
        box_drop_threshold = (
            soft_min_class_boxes
            if any(reason.endswith("_boxes") for reason in drop_reasons) and soft_min_class_boxes > 0
            else effective_min_class_boxes
        )
        box_gap = max(box_drop_threshold - reference_boxes, 0)
        image_ratio_to_largest_kept = (
            round(max_kept_images / reference_images, 4)
            if reference_images > 0 and max_kept_images > 0
            else None
        )
        box_ratio_to_largest_kept = (
            round(max_kept_boxes / reference_boxes, 4)
            if reference_boxes > 0 and max_kept_boxes > 0
            else None
        )
        can_cover_active_splits = reference_images >= active_split_count if active_split_count > 0 else reference_images > 0
        can_support_current_floor = reference_images >= effective_min_class_images and reference_boxes >= effective_min_class_boxes
        detail_payload = {
            "name": safe_class_name(class_names, class_id),
            "reference_split": reference_split,
            "current": {
                "images": reference_images,
                "boxes": reference_boxes,
                "ap": float(decision.get("ap", 0.0) or 0.0),
            },
            "thresholds": {
                "effective_min_class_images": effective_min_class_images,
                "effective_min_class_boxes": effective_min_class_boxes,
                "soft_min_class_images": soft_min_class_images,
                "soft_min_class_boxes": soft_min_class_boxes,
                "requested_class_threshold": requested_class_threshold,
                "effective_class_threshold": effective_class_threshold,
                "report_filter_mode": report_filter_mode,
            },
            "gaps": {
                "images_to_floor": image_gap,
                "boxes_to_floor": box_gap,
            },
            "balance_view": {
                "balance_ratio_target": balance_ratio,
                "image_ratio_to_largest_kept_class": image_ratio_to_largest_kept,
                "box_ratio_to_largest_kept_class": box_ratio_to_largest_kept,
                "estimated_box_capacity_under_ratio": (
                    int(reference_boxes * balance_ratio) if balance_ratio > 0 and reference_boxes > 0 else reference_boxes
                ),
            },
            "coverage_view": {
                "active_split_count": active_split_count,
                "can_cover_active_splits": can_cover_active_splits,
                "can_support_current_floor": can_support_current_floor,
            },
            "advisory_notes": advisory_notes,
        }
        if drop_reasons:
            analysis_lines: list[str] = []
            if image_gap > 0:
                analysis_lines.append(
                    f"{reference_split_label}图数 {reference_images}，当前删类门槛 {image_drop_threshold}，还需要补充 {image_gap} 张图。"
                )
            if box_gap > 0:
                analysis_lines.append(
                    f"{reference_split_label}框数 {reference_boxes}，当前删类门槛 {box_drop_threshold}，还需要补充 {box_gap} 个框。"
                )
            if box_ratio_to_largest_kept is not None:
                analysis_lines.append(
                    f"当前类别与保留大类的框数比约为 1:{box_ratio_to_largest_kept}。"
                )
            if "low_ap" in drop_reasons:
                analysis_lines.append(
                    f"AP 为 {float(decision.get('ap', 0.0) or 0.0):.4f}，严格阈值为 {effective_class_threshold:.4f}。"
                )
            if not analysis_lines:
                analysis_lines.append("当前类别未达到导出保留门槛。")
            dropped_analysis[str(class_id)] = {
                **detail_payload,
                "primary_reason": primary_drop_reason(drop_reasons),
                "drop_reasons": drop_reasons,
                "reason_details": build_drop_reason_details(
                    decision=decision,
                    effective_min_class_images=effective_min_class_images,
                    effective_min_class_boxes=effective_min_class_boxes,
                    requested_class_threshold=requested_class_threshold,
                    effective_class_threshold=effective_class_threshold,
                    report_filter_mode=report_filter_mode,
                    reference_split=reference_split,
                ),
                "analysis": analysis_lines,
                "suggested_action": (
                    (
                        f"优先补充至少 {max(image_gap, 1)} 张图，并同步补齐 {box_gap} 个有效框后再重新参与导出。"
                        if box_gap > 0
                        else f"优先补充至少 {max(image_gap, 1)} 张图后再重新参与导出。"
                    )
                    if image_gap > 0 or box_gap > 0
                    else "当前类别已经接近门槛，可以结合业务价值决定是否单独补数。"
                ),
            }
            continue

        image_margin = reference_images - effective_min_class_images
        box_margin = reference_boxes - effective_min_class_boxes
        if image_margin <= 2 or box_margin <= max(2, effective_min_class_boxes):
            borderline_kept[str(class_id)] = {
                **detail_payload,
                "margin": {
                    "images_over_floor": image_margin,
                    "boxes_over_floor": box_margin,
                },
                "analysis": [
                    f"{reference_split_label}图数 {reference_images}，距离生效图数门槛还有 {image_margin} 张余量。",
                    f"{reference_split_label}框数 {reference_boxes}，距离生效框数门槛还有 {box_margin} 个框余量。",
                ],
            }

    filter_summary = {
        "reference_split": reference_split,
        "total_classes": len(class_decisions),
        "kept_classes": len(kept_class_ids),
        "dropped_classes": len(dropped_analysis),
        "drop_reason_histogram": dict(reason_histogram),
        "effective_min_class_images": effective_min_class_images,
        "effective_min_class_boxes": effective_min_class_boxes,
        "active_split_count": active_split_count,
        "max_kept_images": max_kept_images,
        "max_kept_boxes": max_kept_boxes,
        "balance_ratio": balance_ratio,
        "report_filter_mode": report_filter_mode,
        "min_images_summary": min_images_summary,
        "min_boxes_summary": min_boxes_summary,
    }
    return filter_summary, dropped_analysis, borderline_kept

def build_class_filter_highlights(
    *,
    class_filter_summary: dict[str, Any],
    dropped_class_analysis: dict[str, Any],
    borderline_kept_class_analysis: dict[str, Any],
) -> list[str]:
    highlights = [
        f"本次类别筛选共检查 {class_filter_summary['total_classes']} 个类别，保留 {class_filter_summary['kept_classes']} 个，筛掉 {class_filter_summary['dropped_classes']} 个。",
        f"生效门槛为最少 {class_filter_summary['effective_min_class_images']} 张图、最少 {class_filter_summary['effective_min_class_boxes']} 个框，目标类别比例为 {class_filter_summary['balance_ratio']}:1。",
    ]
    drop_reason_histogram = class_filter_summary.get("drop_reason_histogram", {})
    if drop_reason_histogram:
        major_reason = max(drop_reason_histogram, key=drop_reason_histogram.get)
        highlights.append(
            f"主导删类原因是 {major_reason}，涉及 {drop_reason_histogram[major_reason]} 个类别。"
        )
    if dropped_class_analysis:
        tightest_class = min(
            dropped_class_analysis.values(),
            key=lambda item: (
                item["gaps"]["images_to_floor"] + item["gaps"]["boxes_to_floor"],
                item["name"],
            ),
        )
        highlights.append(
            f"最接近保留门槛的筛掉类别是 {tightest_class['name']}，补齐 {tightest_class['gaps']['images_to_floor']} 张图和 {tightest_class['gaps']['boxes_to_floor']} 个框后最容易回到候选池。"
        )
    if borderline_kept_class_analysis:
        borderline_class = min(
            borderline_kept_class_analysis.values(),
            key=lambda item: (
                item["margin"]["images_over_floor"] + item["margin"]["boxes_over_floor"],
                item["name"],
            ),
        )
        highlights.append(
            f"边缘保留类别 {borderline_class['name']} 需要重点关注，当前图数余量 {borderline_class['margin']['images_over_floor']}，框数余量 {borderline_class['margin']['boxes_over_floor']}。"
        )
    return highlights


def build_dataset_eda(
    *,
    class_ids: list[int],
    class_names: dict[int, str],
    class_stats: dict[int, dict[str, dict[str, int]]],
    per_class_ap: dict[str, Any],
    reference_split: str,
    source_total_images: int,
    target_total_images: int,
) -> dict[str, Any]:
    class_entries: list[dict[str, Any]] = []
    image_counts: list[int] = []
    box_counts: list[int] = []
    boxes_per_image_values: list[float] = []
    ap_values: list[float] = []
    for class_id in class_ids:
        stats = class_stats.get(class_id, {})
        reference_stats = stats.get(reference_split, {"images": 0, "boxes": 0})
        images = int(reference_stats.get("images", 0) or 0)
        boxes = int(reference_stats.get("boxes", 0) or 0)
        ap_info = per_class_ap.get(str(class_id), {})
        ap_value = float(ap_info.get("ap", 0.0) or 0.0) if isinstance(ap_info, dict) else 0.0
        gt_count = int(ap_info.get("gt", 0) or 0) if isinstance(ap_info, dict) else 0
        boxes_per_image = (boxes / images) if images > 0 else 0.0
        image_share = (images / source_total_images) if source_total_images > 0 else 0.0
        target_share = (target_total_images / source_total_images) if source_total_images > 0 and target_total_images > 0 else 1.0
        class_entries.append(
            {
                "class_id": class_id,
                "name": safe_class_name(class_names, class_id),
                "images": images,
                "boxes": boxes,
                "boxes_per_image": round(boxes_per_image, 4),
                "image_share": round(image_share, 6),
                "target_share": round(target_share, 6),
                "ap": ap_value,
                "gt": gt_count,
                "splits": stats,
            }
        )
        if images > 0:
            image_counts.append(images)
            boxes_per_image_values.append(boxes_per_image)
        if boxes > 0:
            box_counts.append(boxes)
        if gt_count > 0:
            ap_values.append(ap_value)
    ranked_by_images = sorted(class_entries, key=lambda item: (-item["images"], item["class_id"]))
    ranked_by_boxes = sorted(class_entries, key=lambda item: (-item["boxes"], item["class_id"]))
    return {
        "overview": {
            "reference_split": reference_split,
            "source_total_images": source_total_images,
            "target_total_images": max(target_total_images, 0),
            "class_count": len(class_entries),
            "classes_with_images": sum(1 for value in image_counts if value > 0),
            "classes_with_boxes": sum(1 for value in box_counts if value > 0),
            "image_count_stats": {
                "min": min(image_counts, default=0),
                "median": percentile_int(image_counts, 0.5),
                "p75": percentile_int(image_counts, 0.75),
                "p90": percentile_int(image_counts, 0.9),
                "max": max(image_counts, default=0),
            },
            "box_count_stats": {
                "min": min(box_counts, default=0),
                "median": percentile_int(box_counts, 0.5),
                "p75": percentile_int(box_counts, 0.75),
                "p90": percentile_int(box_counts, 0.9),
                "max": max(box_counts, default=0),
            },
            "boxes_per_image_stats": {
                "min": round(min(boxes_per_image_values, default=0.0), 4),
                "median": round(percentile_float(boxes_per_image_values, 0.5), 4),
                "p75": round(percentile_float(boxes_per_image_values, 0.75), 4),
                "p90": round(percentile_float(boxes_per_image_values, 0.9), 4),
                "max": round(max(boxes_per_image_values, default=0.0), 4),
            },
            "ap_stats": {
                "min": round(min(ap_values, default=0.0), 6),
                "median": round(percentile_float(ap_values, 0.5), 6),
                "p75": round(percentile_float(ap_values, 0.75), 6),
                "p90": round(percentile_float(ap_values, 0.9), 6),
                "max": round(max(ap_values, default=0.0), 6),
            },
        },
        "largest_classes_by_images": ranked_by_images[: min(10, len(ranked_by_images))],
        "largest_classes_by_boxes": ranked_by_boxes[: min(10, len(ranked_by_boxes))],
        "classes": {
            str(entry["class_id"]): {
                key: value
                for key, value in entry.items()
                if key != "class_id"
            }
            for entry in class_entries
        },
    }

def build_export_candidates(
    *,
    source_infos_by_split: dict[str, list[SourceImageInfo]],
    kept_class_ids: list[int],
    max_boxes_per_image: int,
    max_boxes_per_class_per_image: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, list[ExportImageCandidate]], dict[str, dict[str, int]]]:
    kept_class_id_set = set(kept_class_ids)
    candidates_by_split: dict[str, list[ExportImageCandidate]] = {}
    skipped_summary: dict[str, dict[str, int]] = {}
    total_infos = sum(len(infos) for infos in source_infos_by_split.values())
    processed_infos = 0
    for split_name, infos in source_infos_by_split.items():
        split_candidates: list[ExportImageCandidate] = []
        skipped_reasons: Counter[str] = Counter()
        split_total_infos = len(infos)
        split_processed_infos = 0

        def _build_progress_detail() -> str:
            kept_count = len(split_candidates)
            skipped_count = sum(skipped_reasons.values())
            if skipped_reasons:
                top_reason, top_count = max(
                    skipped_reasons.items(),
                    key=lambda item: (item[1], item[0]),
                )
                return (
                    f"split={split_name} local={split_processed_infos}/{split_total_infos} "
                    f"kept={kept_count} skipped={skipped_count} top_skip={top_reason}:{top_count}"
                )
            return (
                f"split={split_name} local={split_processed_infos}/{split_total_infos} "
                f"kept={kept_count} skipped={skipped_count}"
            )

        def _emit_progress() -> None:
            if progress_callback is not None and (
                processed_infos == 1
                or processed_infos == total_infos
                or split_processed_infos == split_total_infos
                or processed_infos % 100 == 0
            ):
                progress_callback(processed_infos, total_infos, _build_progress_detail())

        for info in infos:
            processed_infos += 1
            split_processed_infos += 1

            filtered_lines = tuple(
                line
                for line in info.label_lines
                if int(float(line.split()[0])) in kept_class_id_set
            )
            class_box_counts = {
                class_id: box_count
                for class_id, box_count in info.class_box_counts.items()
                if class_id in kept_class_id_set
            }
            total_boxes = sum(class_box_counts.values())
            if not filtered_lines:
                skipped_reasons["no_kept_labels"] += 1
                _emit_progress()
                continue
            if max_boxes_per_image > 0 and total_boxes > max_boxes_per_image:
                skipped_reasons["too_many_boxes_in_image"] += 1
                _emit_progress()
                continue
            if max_boxes_per_class_per_image > 0 and max(class_box_counts.values()) > max_boxes_per_class_per_image:
                skipped_reasons["too_many_boxes_for_single_class"] += 1
                _emit_progress()
                continue
            split_candidates.append(
                ExportImageCandidate(
                    split_name=info.split_name,
                    rel_split_image_dir=info.rel_split_image_dir,
                    rel_split_label_dir=info.rel_split_label_dir,
                    rel_path=info.rel_path,
                    src_image_path=info.src_image_path,
                    src_label_path=info.src_label_path,
                    filtered_lines=filtered_lines,
                    class_box_counts=class_box_counts,
                )
            )
            _emit_progress()
        candidates_by_split[split_name] = split_candidates
        skipped_summary[split_name] = dict(skipped_reasons)
    return candidates_by_split, skipped_summary

def count_candidate_images_per_class(candidates: list[ExportImageCandidate], class_ids: list[int]) -> dict[int, int]:
    counts = {class_id: 0 for class_id in class_ids}
    for candidate in candidates:
        for class_id in candidate.class_box_counts:
            counts[class_id] += 1
    return counts

def count_candidate_boxes_per_class(candidates: list[ExportImageCandidate], class_ids: list[int]) -> dict[int, int]:
    counts = {class_id: 0 for class_id in class_ids}
    for candidate in candidates:
        for class_id, box_count in candidate.class_box_counts.items():
            counts[class_id] += box_count
    return counts

def derive_target_boxes_per_class(
    *,
    available_boxes_per_class: dict[int, int],
    requested_target_boxes_per_class: int,
    balance_ratio: float,
    effective_target_total_images: int,
    average_boxes_per_image: float,
) -> tuple[dict[int, int], dict[str, Any]]:
    positive_counts = [count for count in available_boxes_per_class.values() if count > 0]
    if not positive_counts:
        return {}, {
            "requested_target_boxes_per_class": requested_target_boxes_per_class,
            "effective_target_boxes_per_class": 0,
            "available_boxes_per_class": available_boxes_per_class,
        }
    min_available_boxes = min(positive_counts)
    positive_class_count = len(positive_counts)
    estimated_total_box_budget = (
        int(round(effective_target_total_images * average_boxes_per_image))
        if effective_target_total_images > 0 and average_boxes_per_image > 0.0
        else 0
    )
    ratio_cap_boxes_per_class = (
        max(1, int(min_available_boxes * balance_ratio))
        if balance_ratio > 0
        else max(positive_counts)
    )
    auto_target_boxes_per_class = (
        max(1, int(estimated_total_box_budget / positive_class_count))
        if estimated_total_box_budget > 0 and positive_class_count > 0
        else ratio_cap_boxes_per_class
    )
    auto_target_boxes_per_class = min(auto_target_boxes_per_class, ratio_cap_boxes_per_class)
    effective_target_boxes_per_class = (
        max(requested_target_boxes_per_class, 0)
        if requested_target_boxes_per_class > 0
        else auto_target_boxes_per_class
    )
    target_boxes_per_class = {
        class_id: min(box_count, effective_target_boxes_per_class)
        for class_id, box_count in available_boxes_per_class.items()
    }
    return target_boxes_per_class, {
        "requested_target_boxes_per_class": max(requested_target_boxes_per_class, 0),
        "auto_target_boxes_per_class": auto_target_boxes_per_class,
        "effective_target_boxes_per_class": effective_target_boxes_per_class,
        "min_available_boxes_per_class": min_available_boxes,
        "ratio_cap_boxes_per_class": ratio_cap_boxes_per_class,
        "estimated_total_box_budget": estimated_total_box_budget,
        "average_boxes_per_image": average_boxes_per_image,
        "positive_class_count": positive_class_count,
        "available_boxes_per_class": available_boxes_per_class,
    }

def score_export_candidate(
    *,
    candidate: ExportImageCandidate,
    desired_box_counts: dict[int, int],
    selected_box_counts: dict[int, int],
    available_box_counts: dict[int, int],
    box_density_penalty: float,
) -> float:
    """凹覆盖(concave-over-modular)子模目标的边际增益。

    目标 F(S)=Σ_c w_c·g(min(n_c(S), cap_c))，其中 n_c 为已选集合里 c 类的框数，
    g=√(凹)，cap_c=desired_box_counts(均衡窗口上限)，w_c=1/√availability(逆频
    权重)。F 单调子模(凹∘截断模函数)，边际增益单调不增——与 CELF 懒贪心假设一致。

    与旧线性打分(usable/availability)的本质区别：g 的边际递减让"给已较满的类
    加框"几乎不涨分，因此密集共现里的主导类(如油漆剥落)作为"乘客"不再被奖励，
    贪心转而优先挑能抬升覆盖不足类、且主导类杂框更少的图。这就是"只选图、不改
    标注地惩罚超配额"。cap_c 仍作硬窗口防止主导类被无限追逐。
    """
    gain = 0.0
    for class_id in sorted(candidate.class_box_counts):
        cap = desired_box_counts.get(class_id, 0)
        current = selected_box_counts.get(class_id, 0)
        if current >= cap:
            # 已满配额：零增益(惩罚超配额)。注意这是抑制"主动追逐"，realized 框数
            # 仍可能因共现被动超出——那是只选图方案的物理上限，需框级裁剪才能消除。
            continue
        effective_new = min(current + candidate.class_box_counts[class_id], cap)
        marginal = sqrt(effective_new) - sqrt(current)
        if marginal <= 0.0:
            continue
        weight = 1.0 / sqrt(max(available_box_counts.get(class_id, 0), 1))
        gain += weight * marginal
    if gain <= 0.0:
        return 0.0
    density_weight = 1.0 / (1.0 + max(candidate.total_boxes - 1, 0) * max(box_density_penalty, 0.0))
    return gain * density_weight


RATIO_BUCKET_NAMES = ("small", "medium", "large")


def ratio_bucket_props(buckets: dict[str, int]) -> dict[str, float]:
    """small/medium/large 占比（分母排除 tiny）。"""
    total = sum(int(buckets.get(name, 0)) for name in RATIO_BUCKET_NAMES)
    if total <= 0:
        return {name: 0.0 for name in RATIO_BUCKET_NAMES}
    return {name: buckets.get(name, 0) / total for name in RATIO_BUCKET_NAMES}


def size_deficit_score(
    cand_buckets: dict[str, int],
    current_buckets: dict[str, int],
    target_ratio: dict[str, float],
    *,
    over_penalty: float = 1.0,
) -> float:
    """候选图对"当前亏空尺寸桶"的贡献减去对超标桶的惩罚（赤字驱动）。"""
    props = ratio_bucket_props(current_buckets)
    score = 0.0
    for name in RATIO_BUCKET_NAMES:
        deficit = max(0.0, target_ratio.get(name, 0.0) - props[name])
        over = max(0.0, props[name] - target_ratio.get(name, 0.0))
        boxes = cand_buckets.get(name, 0)
        score += deficit * boxes - over_penalty * over * boxes
    return score


def density_steer_term(
    *,
    total_boxes: int,
    current_avg: float,
    lo: float,
    hi: float,
) -> float:
    """平均框数软导向：低于 lo 奖励多框图，高于 hi 奖励少框图，带内为 0。

    返回带符号的方向项，量纲为"框数"，由调用方乘以内置权重后并入总分。
    """
    if lo <= 0.0 and hi <= 0.0:
        return 0.0
    if current_avg < lo:
        return float(total_boxes)
    if hi > 0.0 and current_avg > hi:
        return -float(total_boxes)
    return 0.0


def fallback_fill_score(
    *,
    candidate: ExportImageCandidate,
    selected_box_counts: dict[int, int],
    available_box_counts: dict[int, int],
    box_density_penalty: float,
) -> float:
    if not candidate.class_box_counts:
        return 0.0
    max_selected_boxes = max(selected_box_counts.values(), default=0)
    balance_score = 0.0
    for class_id, box_count in candidate.class_box_counts.items():
        availability = max(available_box_counts.get(class_id, 0), 1)
        scarcity_weight = 1.0 / availability
        balance_weight = (max_selected_boxes + 1.0) / (selected_box_counts.get(class_id, 0) + 1.0)
        balance_score += box_count * scarcity_weight * balance_weight
    density_weight = 1.0 / (1.0 + max(candidate.total_boxes - 1, 0) * max(box_density_penalty, 0.0))
    return balance_score * density_weight

def _select_balanced_train_candidates_legacy(
    *,
    candidates: list[ExportImageCandidate],
    kept_class_ids: list[int],
    target_total_images: int,
    target_boxes_per_class: int,
    balance_ratio: float,
    target_images_per_class: int,
    box_density_penalty: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[ExportImageCandidate], dict[str, Any]]:
    """旧版 O(N*K) 平衡选图。保留作 reference / 回退。

    新流水线默认走 select_balanced_train_candidates（CELF 懒贪心，等价语义但
    复杂度降到 O((N+K) log N)）。如需逐字节复现旧行为可手动调用本函数。
    """
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=len(candidates),
    )
    available_counts = count_candidate_images_per_class(candidates, kept_class_ids)
    available_box_counts = count_candidate_boxes_per_class(candidates, kept_class_ids)
    average_boxes_per_image = (
        sum(candidate.total_boxes for candidate in candidates) / len(candidates)
        if candidates
        else 0.0
    )
    desired_box_counts, box_target_summary = derive_target_boxes_per_class(
        available_boxes_per_class=available_box_counts,
        requested_target_boxes_per_class=target_boxes_per_class,
        balance_ratio=balance_ratio,
        effective_target_total_images=effective_target_total_images,
        average_boxes_per_image=average_boxes_per_image,
    )
    effective_target_images = max(target_images_per_class, 0)
    desired_image_counts = {
        class_id: min(effective_target_images, available_counts[class_id])
        for class_id in kept_class_ids
    }
    selected_box_counts = {class_id: 0 for class_id in kept_class_ids}
    selected_image_counts = {class_id: 0 for class_id in kept_class_ids}
    remaining = sorted(
        candidates,
        key=lambda item: (item.rel_path.as_posix(), item.total_boxes),
    )
    selected: list[ExportImageCandidate] = []
    planned_rounds = (
        min(effective_target_total_images, len(candidates))
        if effective_target_total_images > 0
        else len(candidates)
    )
    estimated_total_evaluations = (
        int(planned_rounds * (2 * len(candidates) - planned_rounds + 1) / 2)
        if planned_rounds > 0
        else 0
    )
    evaluation_count = 0
    display_resolution = 10
    selection_progress_total = max(planned_rounds * display_resolution, 1)

    while remaining:
        if effective_target_total_images > 0 and len(selected) >= effective_target_total_images:
            break
        if (
            effective_target_total_images <= 0
            and all(selected_box_counts[class_id] >= desired_box_counts.get(class_id, 0) for class_id in kept_class_ids)
        ):
            break
        best_idx = -1
        best_score = 0.0
        best_total_boxes = 0
        best_class_count = 0
        best_path = ""
        for idx, candidate in enumerate(remaining):
            evaluation_count += 1
            image_target_saturated = effective_target_images > 0 and all(
                selected_image_counts[class_id] >= desired_image_counts[class_id]
                for class_id in candidate.class_box_counts
            )
            score = score_export_candidate(
                candidate=candidate,
                desired_box_counts=desired_box_counts,
                selected_box_counts=selected_box_counts,
                available_box_counts=available_box_counts,
                box_density_penalty=box_density_penalty,
            )
            if score <= 0.0 and effective_target_total_images > 0:
                score = fallback_fill_score(
                    candidate=candidate,
                    selected_box_counts=selected_box_counts,
                    available_box_counts=available_box_counts,
                    box_density_penalty=box_density_penalty,
                )
            if image_target_saturated:
                score *= 0.2
            path_key = candidate.rel_path.as_posix()
            should_take = False
            if best_idx < 0 or score > best_score + 1e-12:
                should_take = True
            elif abs(score - best_score) <= 1e-12 and candidate.total_boxes < best_total_boxes:
                should_take = True
            elif (
                abs(score - best_score) <= 1e-12
                and candidate.total_boxes == best_total_boxes
                and len(candidate.class_box_counts) > best_class_count
            ):
                should_take = True
            elif (
                abs(score - best_score) <= 1e-12
                and candidate.total_boxes == best_total_boxes
                and len(candidate.class_box_counts) == best_class_count
                and (best_path == "" or path_key < best_path)
            ):
                should_take = True
            if should_take:
                best_idx = idx
                best_score = score
                best_total_boxes = candidate.total_boxes
                best_class_count = len(candidate.class_box_counts)
                best_path = path_key
            if progress_callback is not None and estimated_total_evaluations > 0 and (
                evaluation_count == 1
                or evaluation_count == estimated_total_evaluations
                or evaluation_count % 2000 == 0
            ):
                scan_progress = idx + 1
                display_current = min(
                    selection_progress_total,
                    len(selected) * display_resolution
                    + int(display_resolution * scan_progress / max(len(remaining), 1)),
                )
                progress_callback(
                    display_current,
                    selection_progress_total,
                    (
                        f"pick={len(selected)}/{planned_rounds} "
                        f"scan={scan_progress}/{len(remaining)} "
                        f"eval={evaluation_count}/{estimated_total_evaluations}"
                    ),
                )
        if best_idx < 0 or best_score <= 0.0:
            break
        candidate = remaining.pop(best_idx)
        selected.append(candidate)
        for class_id, box_count in candidate.class_box_counts.items():
            selected_box_counts[class_id] += box_count
            selected_image_counts[class_id] += 1
        if progress_callback is not None and estimated_total_evaluations <= 0:
            progress_callback(
                len(selected),
                max(len(candidates), 1),
                f"pick={len(selected)}/{max(planned_rounds, len(candidates), 1)} remaining={len(remaining)}",
            )

    selected.sort(key=lambda item: item.rel_path.as_posix())
    return selected, {
        "target_total_images": target_total_images,
        "effective_target_total_images": effective_target_total_images,
        "available_total_images": len(candidates),
        "available_images_per_class": available_counts,
        "available_boxes_per_class": available_box_counts,
        "target_images_per_class": target_images_per_class,
        "effective_target_images_per_class": effective_target_images,
        "desired_images_per_class": desired_image_counts,
        "selected_images_per_class": selected_image_counts,
        "target_boxes_per_class": box_target_summary["requested_target_boxes_per_class"],
        "auto_target_boxes_per_class": box_target_summary.get("auto_target_boxes_per_class", 0),
        "effective_target_boxes_per_class": box_target_summary.get("effective_target_boxes_per_class", 0),
        "desired_boxes_per_class": desired_box_counts,
        "selected_boxes_per_class": selected_box_counts,
        "average_boxes_per_image": average_boxes_per_image,
        "balance_ratio": balance_ratio,
        "target_total_images_summary": target_total_images_summary,
        "estimated_total_evaluations": estimated_total_evaluations,
        "actual_total_evaluations": evaluation_count,
        "selection_algorithm": "legacy_full_rescan",
    }


def select_balanced_train_candidates(
    *,
    candidates: list[ExportImageCandidate],
    kept_class_ids: list[int],
    target_total_images: int,
    target_boxes_per_class: int,
    balance_ratio: float,
    target_images_per_class: int,
    box_density_penalty: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
    size_buckets_by_candidate: dict[str, dict[str, int]] | None = None,
    target_size_ratio: dict[str, float] | None = None,
    size_balance_weight: float = 0.0,
    avg_boxes_per_image_min: float = 0.0,
    avg_boxes_per_image_max: float = 0.0,
    prioritize_balance: bool = False,
) -> tuple[list[ExportImageCandidate], dict[str, Any]]:
    """CELF (Cost-Effective Lazy Forward) 懒贪心平衡选图。

    prioritize_balance=True 时把 target_total_images 当作**上限**而非"必须填满"的
    目标：Phase 1 按每类框数配额（min_available*ratio 的均衡窗口）选完即停，跳过
    Phase 2 的兜底补齐。否则继续会用多数类的图把数据集填到 target_total_images，
    重新引入类别不均衡。换言之：均衡优先，数量让步——少数类不再被牺牲。

    旧实现每轮 O(N) 扫描所有候选重新算分，整体 O(N*K)。在 7 万张数据集上
    K=5000 时 ~3.4 亿次内层评估，纯 Python 跑十几到几十分钟；K=0（不限）
    时则膨胀到 N²/2 ≈ 24 亿次。

    本实现利用 score_export_candidate 的单调不增性（desired_box_counts 是
    常数、selected_box_counts 只增不减、density_weight 是常数；凹覆盖目标
    g=√ 的边际随 selected_box_counts 增大而递减，单调不增依然成立）——这是
    经典的(单调)子模最大化结构，标准解法即 CELF：
      - 维护一个 max-heap，每次 pop 堆顶；
      - 用 class_epoch 记录每个类别的"最后一次被影响的时刻"；候选只要其类别
        集合里有 epoch 比记录值新的，就 stale，需 pop 后重算 + 推回；
      - 否则当前堆顶就是真正的全局最高分，直接 accept。
    Phase 1 与原贪心**结果完全一致**，但每次 pick 只触发 O(log N) 次堆操作
    + O(C_picked) 次 epoch 自增；总评估数从 K*N 降到 O((N+K) log N)。

    Phase 2 (fallback_fill_score) 因 balance_weight 含 (max_selected+1)/
    (selected[c]+1)，*可能上升*，不严格单调。Phase 2 仅在 target_total_images
    > 0 且 Phase 1 已经把所有主分>0 的候选用光（即 per-class 目标已满足）后
    才会出现，剩余指标对最终数据集分布影响有限；这里一次性按当前状态打分
    + 排序取 top-K，O(R log R)，把潜在的 O(K_phase2 * R) 退化压成单次排序。

    复杂度：Phase 1 ≈ O((N+K) log N)；Phase 2 ≤ O(R log R)。
    内存：O(N + 已推回的过期堆条目)。
    """
    if target_size_ratio:
        return _select_size_aware_candidates(
            candidates=candidates,
            kept_class_ids=kept_class_ids,
            target_total_images=target_total_images,
            target_boxes_per_class=target_boxes_per_class,
            balance_ratio=balance_ratio,
            target_images_per_class=target_images_per_class,
            box_density_penalty=box_density_penalty,
            size_buckets_by_candidate=size_buckets_by_candidate or {},
            target_size_ratio=target_size_ratio,
            size_balance_weight=size_balance_weight,
            avg_boxes_per_image_min=avg_boxes_per_image_min,
            avg_boxes_per_image_max=avg_boxes_per_image_max,
            prioritize_balance=prioritize_balance,
            progress_callback=progress_callback,
        )
    effective_target_total_images, target_total_images_summary = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=len(candidates),
    )
    available_counts = count_candidate_images_per_class(candidates, kept_class_ids)
    available_box_counts = count_candidate_boxes_per_class(candidates, kept_class_ids)
    average_boxes_per_image = (
        sum(candidate.total_boxes for candidate in candidates) / len(candidates)
        if candidates
        else 0.0
    )
    desired_box_counts, box_target_summary = derive_target_boxes_per_class(
        available_boxes_per_class=available_box_counts,
        requested_target_boxes_per_class=target_boxes_per_class,
        balance_ratio=balance_ratio,
        effective_target_total_images=effective_target_total_images,
        average_boxes_per_image=average_boxes_per_image,
    )
    effective_target_images = max(target_images_per_class, 0)
    desired_image_counts = {
        class_id: min(effective_target_images, available_counts[class_id])
        for class_id in kept_class_ids
    }
    selected_box_counts = {class_id: 0 for class_id in kept_class_ids}
    selected_image_counts = {class_id: 0 for class_id in kept_class_ids}

    summary_template = {
        "target_total_images": target_total_images,
        "effective_target_total_images": effective_target_total_images,
        "available_total_images": len(candidates),
        "available_images_per_class": available_counts,
        "available_boxes_per_class": available_box_counts,
        "target_images_per_class": target_images_per_class,
        "effective_target_images_per_class": effective_target_images,
        "desired_images_per_class": desired_image_counts,
        "selected_images_per_class": selected_image_counts,
        "target_boxes_per_class": box_target_summary["requested_target_boxes_per_class"],
        "auto_target_boxes_per_class": box_target_summary.get("auto_target_boxes_per_class", 0),
        "effective_target_boxes_per_class": box_target_summary.get("effective_target_boxes_per_class", 0),
        "desired_boxes_per_class": desired_box_counts,
        "selected_boxes_per_class": selected_box_counts,
        "average_boxes_per_image": average_boxes_per_image,
        "balance_ratio": balance_ratio,
        "target_total_images_summary": target_total_images_summary,
        "selection_algorithm": "celf_lazy_greedy",
        "phase1_picks": 0,
        "phase2_picks": 0,
        "phase1_heap_pops": 0,
        "phase1_stale_pops": 0,
    }

    n_candidates = len(candidates)
    if n_candidates == 0:
        return [], summary_template

    picked = [False] * n_candidates
    class_epoch: dict[int, int] = {class_id: 0 for class_id in kept_class_ids}
    cand_seen_epoch = [0] * n_candidates

    def _candidate_max_epoch(idx: int) -> int:
        c = candidates[idx]
        max_e = 0
        for class_id in c.class_box_counts:
            e = class_epoch.get(class_id, 0)
            if e > max_e:
                max_e = e
        return max_e

    def _saturation_multiplier(c: ExportImageCandidate) -> float:
        if effective_target_images <= 0 or not c.class_box_counts:
            return 1.0
        for class_id in c.class_box_counts:
            if selected_image_counts.get(class_id, 0) < desired_image_counts.get(class_id, 0):
                return 1.0
        return 0.2

    def _primary_score(idx: int) -> float:
        c = candidates[idx]
        sc = score_export_candidate(
            candidate=c,
            desired_box_counts=desired_box_counts,
            selected_box_counts=selected_box_counts,
            available_box_counts=available_box_counts,
            box_density_penalty=box_density_penalty,
        )
        return sc * _saturation_multiplier(c)

    def _heap_key(idx: int, score: float) -> tuple:
        c = candidates[idx]
        # tie-break 完全对齐 legacy: score desc, total_boxes asc, class_count desc, path asc
        return (
            -score,
            c.total_boxes,
            -len(c.class_box_counts),
            c.rel_path.as_posix(),
            idx,
        )

    planned_rounds = (
        min(effective_target_total_images, n_candidates)
        if effective_target_total_images > 0
        else n_candidates
    )
    progress_denom = max(planned_rounds * 10, 1)
    selected: list[ExportImageCandidate] = []
    last_progress_emit = -1
    heap_pops = 0
    stale_pops = 0

    def _emit_progress(phase: str, extra: str = "") -> None:
        nonlocal last_progress_emit
        if progress_callback is None:
            return
        if last_progress_emit == len(selected):
            return
        last_progress_emit = len(selected)
        current = min(progress_denom, len(selected) * 10)
        msg = f"phase={phase} pick={len(selected)}/{planned_rounds}"
        if extra:
            msg = f"{msg} {extra}"
        progress_callback(current, progress_denom, msg)

    def _needs_met() -> bool:
        for class_id in kept_class_ids:
            if selected_box_counts[class_id] < desired_box_counts.get(class_id, 0):
                return False
        return True

    # ----- Phase 1: 主分（单调不增）上的 CELF 懒贪心 -----
    # 初始按 rel_path 排序入堆，使初始堆构造在 score 相等时退化到 legacy 一致顺序。
    initial_order = sorted(range(n_candidates), key=lambda i: candidates[i].rel_path.as_posix())
    heap: list[tuple] = []
    for idx in initial_order:
        sc = _primary_score(idx)
        if sc > 0.0:
            heap.append(_heap_key(idx, sc))
    heapq.heapify(heap)

    if progress_callback is not None:
        _emit_progress("primary")

    while heap:
        if effective_target_total_images > 0 and len(selected) >= effective_target_total_images:
            break
        if effective_target_total_images <= 0 and _needs_met():
            break

        entry = heapq.heappop(heap)
        heap_pops += 1
        neg_score = entry[0]
        idx = entry[-1]
        if picked[idx]:
            continue
        cur_epoch = _candidate_max_epoch(idx)
        if cur_epoch > cand_seen_epoch[idx]:
            # stale —— 主分单调不增，重算后再入堆
            stale_pops += 1
            cand_seen_epoch[idx] = cur_epoch
            new_score = _primary_score(idx)
            if new_score > 0.0:
                heapq.heappush(heap, _heap_key(idx, new_score))
            continue

        current_score = -neg_score
        if current_score <= 0.0:
            break  # 主分耗尽

        # 通过 fresh 检查，即全局最高 —— accept
        picked[idx] = True
        c = candidates[idx]
        selected.append(c)
        for class_id, box_count in c.class_box_counts.items():
            selected_box_counts[class_id] += box_count
            selected_image_counts[class_id] += 1
            class_epoch[class_id] = class_epoch.get(class_id, 0) + 1

        if progress_callback is not None and (
            len(selected) == 1
            or len(selected) >= planned_rounds
            or len(selected) - max(last_progress_emit, 0) >= 200
        ):
            _emit_progress("primary", f"heap={len(heap)}")

    phase1_picks = len(selected)
    if progress_callback is not None:
        _emit_progress("primary", f"heap={len(heap)} done")

    # ----- Phase 2: fallback 一次性批量补齐 -----
    phase2_picks = 0
    if prioritize_balance and phase1_picks < effective_target_total_images:
        # 均衡优先：每类配额已满，剩余只能靠多数类的图补齐，会破坏均衡 —— 跳过。
        # target_total_images 在此模式下是上限，实际可能更少但类间均衡。
        summary_template["target_treated_as_cap"] = True
    if not prioritize_balance and effective_target_total_images > 0 and len(selected) < effective_target_total_images:
        need = effective_target_total_images - len(selected)
        fallback_entries: list[tuple[tuple, int]] = []
        for idx in range(n_candidates):
            if picked[idx]:
                continue
            c = candidates[idx]
            sc = fallback_fill_score(
                candidate=c,
                selected_box_counts=selected_box_counts,
                available_box_counts=available_box_counts,
                box_density_penalty=box_density_penalty,
            )
            sc *= _saturation_multiplier(c)
            if sc <= 0.0:
                continue
            fallback_entries.append((_heap_key(idx, sc), idx))
        fallback_entries.sort(key=lambda item: item[0])
        for _, idx in fallback_entries[:need]:
            c = candidates[idx]
            picked[idx] = True
            selected.append(c)
            for class_id, box_count in c.class_box_counts.items():
                selected_box_counts[class_id] += box_count
                selected_image_counts[class_id] += 1
            phase2_picks += 1
        if progress_callback is not None and phase2_picks > 0:
            _emit_progress("fallback", f"added={phase2_picks}")

    selected.sort(key=lambda item: item.rel_path.as_posix())

    summary = dict(summary_template)
    summary["selected_images_per_class"] = selected_image_counts
    summary["selected_boxes_per_class"] = selected_box_counts
    summary["phase1_picks"] = phase1_picks
    summary["phase2_picks"] = phase2_picks
    summary["phase1_heap_pops"] = heap_pops
    summary["phase1_stale_pops"] = stale_pops
    return selected, summary


DENSITY_STEER_WEIGHT = 0.5

# 划分阶段次级打分权重。类别覆盖(×1000)、每类图数覆盖(×100)严格优先于这两项；
# 在它们之下，"每类框数均衡"与"尺寸桶均衡"以归一化分数(各 0~1)同量级竞争，
# 避免原始 deficit_gain 奖励高框密度、把密集小目标图独占给 train。
SPLIT_BOX_FILL_WEIGHT = 10.0
SPLIT_SIZE_FILL_WEIGHT = 10.0


def _candidate_size_key(candidate: ExportImageCandidate) -> str:
    return candidate.rel_path.as_posix() + "|" + candidate.split_name


# 尺寸感知选图里"类别均衡"相对"尺寸均衡 / 密度导向"的主导权重。类别覆盖增益
# (凹、带 cap、覆盖感知，量纲 ~Σ 1/√avail·Δ√) 数值本就远小于按框数计的尺寸 /
# 密度项；归一化后再乘上该权重，保证"补齐欠覆盖类(尤其稀有类)"严格优先于"凑尺寸
# 占比"，尺寸与密度只在类别增益相近的候选之间做次级塑形。这把均衡选图的"类别优先、
# 数量与尺寸让步"落到了尺寸感知路径上。
SIZE_AWARE_CLASS_WEIGHT = 8.0


def _select_size_aware_candidates(
    *,
    candidates: list[ExportImageCandidate],
    kept_class_ids: list[int],
    target_total_images: int,
    target_boxes_per_class: int,
    balance_ratio: float,
    target_images_per_class: int,
    box_density_penalty: float,
    size_buckets_by_candidate: dict[str, dict[str, int]],
    target_size_ratio: dict[str, float],
    size_balance_weight: float,
    avg_boxes_per_image_min: float,
    avg_boxes_per_image_max: float,
    prioritize_balance: bool = False,
    progress_callback: Callable[[int, int, str], None] | None = None,
    batch_size: int = 200,
) -> tuple[list[ExportImageCandidate], dict[str, Any]]:
    """类别优先的尺寸感知贪心选图。

    评分 = SIZE_AWARE_CLASS_WEIGHT·类别覆盖增益
            + size_balance_weight·尺寸赤字(归一)
            + DENSITY_STEER_WEIGHT·密度软导向(归一)

    类别项复用 ``score_export_candidate`` 的"凹覆盖(concave-over-modular)子模"边际
    增益：**覆盖感知**(读 ``selected_box_counts``)、带**均衡窗口上限 cap**
    (``derive_target_boxes_per_class``，与框级裁剪同口径)、**逆频权重** 1/√avail。
    这修复了旧 ``_class_deficit_score`` 的三处缺陷——静态(从不看已选)、无上限(超配额
    零惩罚)、线性(奖励高框密度图)——它们会让多数类(共现乘客，如油漆剥落)反复中选、
    把尾部类挤到 0 张。尺寸 / 密度项各自按候选框数归一到 ~[-1,1] 后做次级塑形。

    ``prioritize_balance=True`` 且 ``target_total_images>0`` 时把 target 当**上限**：
    每类框数配额已满(``_needs_met``)即停，不再用多数类的图把数据集填到 target(那会
    重新打破均衡)——均衡优先、数量让步。类别赤字阶段**逐张重排**(覆盖反馈要精确)，
    配额满后的尺寸填充阶段按 ``batch_size`` 批量推进。
    """
    available_box_counts = count_candidate_boxes_per_class(candidates, kept_class_ids)
    n = len(candidates)
    average_boxes_per_image = (
        sum(c.total_boxes for c in candidates) / n if n else 0.0
    )
    effective_target_total_images, _ = derive_effective_target_total_images(
        requested_target_total_images=target_total_images,
        available_total_images=n,
    )
    desired_box_counts, box_target_summary = derive_target_boxes_per_class(
        available_boxes_per_class=available_box_counts,
        requested_target_boxes_per_class=max(target_boxes_per_class, 0),
        balance_ratio=balance_ratio,
        effective_target_total_images=effective_target_total_images,
        average_boxes_per_image=average_boxes_per_image,
    )
    target_n = effective_target_total_images if effective_target_total_images > 0 else n
    has_caps = any(cap > 0 for cap in desired_box_counts.values())

    selected: list[ExportImageCandidate] = []
    selected_box_counts = {class_id: 0 for class_id in kept_class_ids}
    current_buckets = {name: 0 for name in ("tiny", "small", "medium", "large")}
    total_selected_boxes = 0
    remaining = list(candidates)
    chosen_keys: set[str] = set()

    def _needs_met() -> bool:
        if not has_caps:
            return True
        for class_id in kept_class_ids:
            if selected_box_counts[class_id] < desired_box_counts.get(class_id, 0):
                return False
        return True

    def _score(c: ExportImageCandidate) -> float:
        key = _candidate_size_key(c)
        buckets = size_buckets_by_candidate.get(key, {})
        class_gain = score_export_candidate(
            candidate=c,
            desired_box_counts=desired_box_counts,
            selected_box_counts=selected_box_counts,
            available_box_counts=available_box_counts,
            box_density_penalty=box_density_penalty,
        )
        size_raw = size_deficit_score(buckets, current_buckets, target_size_ratio)
        size_norm = size_raw / max(c.total_boxes, 1)  # ~[-1, 1]
        cur_avg = (total_selected_boxes / len(selected)) if selected else 0.0
        density_raw = density_steer_term(
            total_boxes=c.total_boxes,
            current_avg=cur_avg,
            lo=avg_boxes_per_image_min,
            hi=avg_boxes_per_image_max,
        )
        density_norm = density_raw / max(c.total_boxes, 1)  # ~{-1, 0, 1}
        return (
            SIZE_AWARE_CLASS_WEIGHT * class_gain
            + size_balance_weight * size_norm
            + DENSITY_STEER_WEIGHT * density_norm
        )

    def _accept(c: ExportImageCandidate) -> None:
        nonlocal total_selected_boxes
        key = _candidate_size_key(c)
        buckets = size_buckets_by_candidate.get(key, {})
        for name in ("tiny", "small", "medium", "large"):
            current_buckets[name] += buckets.get(name, 0)
        for class_id, cnt in c.class_box_counts.items():
            if class_id in selected_box_counts:
                selected_box_counts[class_id] += cnt
        total_selected_boxes += c.total_boxes
        selected.append(c)
        chosen_keys.add(key)

    while remaining and len(selected) < target_n:
        needs_met = _needs_met()
        if prioritize_balance and effective_target_total_images > 0 and needs_met:
            break  # 均衡优先：配额已满，数量让步，不用多数类把数据集填满
        ranked = sorted(
            remaining,
            key=lambda c: (-_score(c), c.total_boxes, c.rel_path.as_posix()),
        )
        # 类别赤字阶段逐张重排(覆盖反馈要精确)；尺寸填充阶段批量推进。
        max_picks = batch_size if needs_met else 1
        picks_this_round = 0
        for c in ranked:
            if len(selected) >= target_n or picks_this_round >= max_picks:
                break
            if _candidate_size_key(c) in chosen_keys:
                continue
            _accept(c)
            picks_this_round += 1
        remaining = [c for c in remaining if _candidate_size_key(c) not in chosen_keys]
        if picks_this_round == 0:
            break
        if progress_callback is not None:
            progress_callback(min(len(selected), target_n), max(target_n, 1),
                              f"size-aware pick={len(selected)}/{target_n}")

    selected.sort(key=lambda item: item.rel_path.as_posix())
    achieved = ratio_bucket_props(current_buckets)
    summary = {
        "selection_algorithm": "size_aware_greedy",
        "available_total_images": n,
        "effective_target_total_images": target_n,
        "desired_boxes_per_class": desired_box_counts,
        "selected_boxes_per_class": selected_box_counts,
        "effective_target_boxes_per_class": box_target_summary.get("effective_target_boxes_per_class", 0),
        "class_weight": SIZE_AWARE_CLASS_WEIGHT,
        "balance_ratio": balance_ratio,
        "prioritize_balance": prioritize_balance,
        "target_treated_as_cap": bool(
            prioritize_balance and effective_target_total_images > 0 and len(selected) < target_n
        ),
        "achieved_size_buckets": dict(current_buckets),
        "achieved_size_ratio": achieved,
        "target_size_ratio": dict(target_size_ratio),
        "size_balance_weight": size_balance_weight,
        "avg_boxes_per_image_achieved": (total_selected_boxes / len(selected)) if selected else 0.0,
        "avg_boxes_per_image_min": avg_boxes_per_image_min,
        "avg_boxes_per_image_max": avg_boxes_per_image_max,
    }
    return selected, summary


def trim_selected_candidates_to_quota(
    selected_candidates: list[ExportImageCandidate],
    kept_class_ids: list[int],
    box_caps: dict[int, int],
) -> tuple[list[ExportImageCandidate], dict[str, Any]]:
    """框级裁剪：把超出均衡窗口(box_caps)的类的多余框从标签里删掉。

    这是"只选图"无法突破共现物理底之后的硬手段：对每个超配额类 c，需删除
    (realized_c - cap_c) 个框；优先从"当前含 c 框最多的图"里删(用 max-heap 反复
    削最高)，从而把 c 的框在各图间摊平、优先清掉密集乘客，最大程度保留每张图里
    其它类的信息。被删的框只是从 .txt 标签里消失(图片照常导出)——代价是这些主导
    类实例变成无标签(训练时的潜在漏标),换来各类框数真正落入窗口。
    """
    realized: dict[int, int] = {class_id: 0 for class_id in kept_class_ids}
    for candidate in selected_candidates:
        for class_id, box_count in candidate.class_box_counts.items():
            realized[class_id] = realized.get(class_id, 0) + box_count
    remove_target = {
        class_id: max(realized.get(class_id, 0) - box_caps.get(class_id, realized.get(class_id, 0)), 0)
        for class_id in kept_class_ids
    }
    if not any(remove_target.values()):
        return selected_candidates, {
            "trimmed_boxes_per_class": {},
            "trimmed_total": 0,
            "realized_before": realized,
            "box_caps": dict(box_caps),
        }

    # 每张图的标签行可变副本；删除即置 None。
    cand_lines: list[list[str | None]] = [list(c.filtered_lines) for c in selected_candidates]

    def _line_class(line: str | None) -> int | None:
        if line is None:
            return None
        try:
            return int(float(line.split()[0]))
        except (ValueError, IndexError):
            return None

    trimmed_per_class: dict[int, int] = {}
    for class_id in kept_class_ids:
        need = remove_target.get(class_id, 0)
        if need <= 0:
            continue
        heap: list[tuple[int, str, int]] = []
        for idx, lines in enumerate(cand_lines):
            count = sum(1 for line in lines if _line_class(line) == class_id)
            if count > 0:
                heap.append((-count, selected_candidates[idx].rel_path.as_posix(), idx))
        heapq.heapify(heap)
        removed = 0
        while removed < need and heap:
            neg_count, path_key, idx = heapq.heappop(heap)
            lines = cand_lines[idx]
            for pos in range(len(lines)):
                if _line_class(lines[pos]) == class_id:
                    lines[pos] = None
                    removed += 1
                    break
            remaining = (-neg_count) - 1
            if remaining > 0:
                heapq.heappush(heap, (-remaining, path_key, idx))
        trimmed_per_class[class_id] = removed

    trimmed_candidates: list[ExportImageCandidate] = []
    for idx, candidate in enumerate(selected_candidates):
        new_lines = tuple(line for line in cand_lines[idx] if line is not None)
        if new_lines == candidate.filtered_lines:
            trimmed_candidates.append(candidate)
            continue
        new_class_box_counts: dict[int, int] = {}
        for line in new_lines:
            cid = _line_class(line)
            if cid is not None:
                new_class_box_counts[cid] = new_class_box_counts.get(cid, 0) + 1
        trimmed_candidates.append(
            dataclasses.replace(
                candidate,
                filtered_lines=new_lines,
                class_box_counts=new_class_box_counts,
            )
        )
    realized_after = {class_id: 0 for class_id in kept_class_ids}
    for candidate in trimmed_candidates:
        for class_id, box_count in candidate.class_box_counts.items():
            realized_after[class_id] = realized_after.get(class_id, 0) + box_count
    return trimmed_candidates, {
        "trimmed_boxes_per_class": trimmed_per_class,
        "trimmed_total": sum(trimmed_per_class.values()),
        "realized_before": realized,
        "realized_after": realized_after,
        "box_caps": dict(box_caps),
    }


def pool_candidates(candidates_by_split: dict[str, list[ExportImageCandidate]]) -> list[ExportImageCandidate]:
    pooled: list[ExportImageCandidate] = []
    for split_name in ("train", "val", "test"):
        pooled.extend(candidates_by_split.get(split_name, []))
    pooled.sort(key=lambda item: (item.rel_path.as_posix(), item.split_name, item.total_boxes))
    return pooled

def parse_split_ratio(split_ratio: str) -> dict[str, float]:
    ratio_text = split_ratio.strip()
    if not ratio_text:
        raise ValueError("split_ratio cannot be empty.")
    parts = [part.strip() for part in ratio_text.split(":")]
    if len(parts) != 3:
        raise ValueError("split_ratio must use the format train:val:test, for example 8:1:1.")
    keys = ("train", "val", "test")
    values: dict[str, float] = {}
    for key, part in zip(keys, parts):
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(
                "split_ratio must contain numeric values, for example 8:1:1."
            ) from exc
        if value < 0:
            raise ValueError("split_ratio values must be >= 0.")
        values[key] = value
    if sum(values.values()) <= 0:
        raise ValueError("split_ratio must have a positive total.")
    return values

def allocate_split_targets_from_source(
    *,
    total_selected_images: int,
    split_ratio: str,
) -> dict[str, int]:
    ratio_values = parse_split_ratio(split_ratio)
    ratio_total = sum(ratio_values.values())
    if total_selected_images <= 0:
        return {split_name: 0 for split_name in ("train", "val", "test")}
    raw_targets = {
        split_name: (total_selected_images * ratio_values.get(split_name, 0.0) / ratio_total)
        for split_name in ("train", "val", "test")
    }
    targets = {split_name: int(raw_targets[split_name]) for split_name in raw_targets}
    assigned = sum(targets.values())
    remainders = sorted(
        (
            raw_targets[split_name] - targets[split_name],
            split_name,
        )
        for split_name in ("train", "val", "test")
    )
    while assigned < total_selected_images:
        remainder, split_name = remainders.pop()
        targets[split_name] += 1
        assigned += 1
        remainders.append((remainder, split_name))
        remainders.sort()
    return targets

def allocate_box_targets_per_split(
    *,
    selected_candidates: list[ExportImageCandidate],
    split_image_targets: dict[str, int],
    kept_class_ids: list[int],
) -> dict[str, dict[int, int]]:
    total_selected_images = len(selected_candidates)
    if total_selected_images <= 0:
        return {split_name: {class_id: 0 for class_id in kept_class_ids} for split_name in ("train", "val", "test")}

    total_boxes_per_class = count_candidate_boxes_per_class(selected_candidates, kept_class_ids)
    desired: dict[str, dict[int, int]] = {
        split_name: {class_id: 0 for class_id in kept_class_ids}
        for split_name in ("train", "val", "test")
    }
    for class_id in kept_class_ids:
        total_class_boxes = total_boxes_per_class[class_id]
        raw_targets = {
            split_name: (
                total_class_boxes * split_image_targets.get(split_name, 0) / total_selected_images
            )
            for split_name in ("train", "val", "test")
        }
        class_targets = {split_name: int(raw_targets[split_name]) for split_name in raw_targets}
        assigned = sum(class_targets.values())
        remainders = sorted(
            (
                raw_targets[split_name] - class_targets[split_name],
                split_name,
            )
            for split_name in ("train", "val", "test")
        )
        while assigned < total_class_boxes:
            remainder, split_name = remainders.pop()
            class_targets[split_name] += 1
            assigned += 1
            remainders.append((remainder, split_name))
            remainders.sort()
        for split_name in ("train", "val", "test"):
            desired[split_name][class_id] = class_targets[split_name]
    return desired


def allocate_size_bucket_targets_per_split(
    *,
    selected_candidates: list[ExportImageCandidate],
    size_buckets_by_candidate: dict[str, dict[str, int]],
    split_image_targets: dict[str, int],
) -> dict[str, dict[str, int]]:
    """按 split 图数比例给每 split 分配 small/medium/large 目标框数（排除 tiny）。"""
    desired: dict[str, dict[str, int]] = {
        split_name: {name: 0 for name in RATIO_BUCKET_NAMES}
        for split_name in ("train", "val", "test")
    }
    total_images = len(selected_candidates)
    if total_images <= 0:
        return desired
    bucket_totals = {name: 0 for name in RATIO_BUCKET_NAMES}
    for c in selected_candidates:
        b = size_buckets_by_candidate.get(_candidate_size_key(c), {})
        for name in RATIO_BUCKET_NAMES:
            bucket_totals[name] += int(b.get(name, 0))
    for name in RATIO_BUCKET_NAMES:
        total = bucket_totals[name]
        raw = {
            s: total * split_image_targets.get(s, 0) / total_images
            for s in ("train", "val", "test")
        }
        alloc = {s: int(raw[s]) for s in raw}
        assigned = sum(alloc.values())
        rema = sorted(((raw[s] - alloc[s], s) for s in ("train", "val", "test")))
        while assigned < total:
            _, s = rema.pop()
            alloc[s] += 1
            assigned += 1
            rema.append((0.0, s))
            rema.sort()
        for s in ("train", "val", "test"):
            desired[s][name] = alloc[s]
    return desired


def allocate_image_coverage_targets_per_split(
    *,
    selected_candidates: list[ExportImageCandidate],
    split_image_targets: dict[str, int],
    kept_class_ids: list[int],
) -> dict[str, dict[int, int]]:
    desired: dict[str, dict[int, int]] = {
        split_name: {class_id: 0 for class_id in kept_class_ids}
        for split_name in ("train", "val", "test")
    }
    active_splits = [
        split_name
        for split_name in ("train", "val", "test")
        if split_image_targets.get(split_name, 0) > 0
    ]
    if not active_splits:
        return desired

    available_images_per_class = count_candidate_images_per_class(selected_candidates, kept_class_ids)
    eval_splits = sorted(
        (split_name for split_name in active_splits if split_name != "train"),
        key=lambda split_name: (split_image_targets.get(split_name, 0), split_name),
    )
    coverage_priority = (["train"] if "train" in active_splits else []) + eval_splits
    for class_id in kept_class_ids:
        remaining_images = available_images_per_class.get(class_id, 0)
        if remaining_images <= 0:
            continue
        for split_name in coverage_priority:
            if remaining_images <= 0:
                break
            desired[split_name][class_id] = 1
            remaining_images -= 1
    return desired


def allocate_image_targets_per_split(
    *,
    selected_candidates: list[ExportImageCandidate],
    split_image_targets: dict[str, int],
    kept_class_ids: list[int],
) -> dict[str, dict[int, int]]:
    desired: dict[str, dict[int, int]] = {
        split_name: {class_id: 0 for class_id in kept_class_ids}
        for split_name in ("train", "val", "test")
    }
    active_splits = [
        split_name
        for split_name in ("train", "val", "test")
        if split_image_targets.get(split_name, 0) > 0
    ]
    if not active_splits:
        return desired

    ratio_total = sum(split_image_targets.get(split_name, 0) for split_name in active_splits)
    available_images_per_class = count_candidate_images_per_class(selected_candidates, kept_class_ids)
    for class_id in kept_class_ids:
        total_class_images = available_images_per_class.get(class_id, 0)
        if total_class_images <= 0 or ratio_total <= 0:
            continue
        raw_targets = {
            split_name: (
                total_class_images * split_image_targets.get(split_name, 0) / ratio_total
            )
            for split_name in active_splits
        }
        class_targets = {split_name: int(raw_targets[split_name]) for split_name in active_splits}
        assigned = sum(class_targets.values())
        remainders = sorted(
            (
                raw_targets[split_name] - class_targets[split_name],
                split_name,
            )
            for split_name in active_splits
        )
        while assigned < total_class_images:
            _, split_name = remainders.pop()
            class_targets[split_name] += 1
            assigned += 1
            remainders.append((raw_targets[split_name] - int(raw_targets[split_name]), split_name))
            remainders.sort()

        if total_class_images >= len(active_splits):
            zero_target_splits = [
                split_name
                for split_name in active_splits
                if class_targets[split_name] <= 0
            ]
            for split_name in zero_target_splits:
                donor_split = max(
                    (
                        donor_name
                        for donor_name in active_splits
                        if class_targets[donor_name] > 1
                    ),
                    key=lambda donor_name: (class_targets[donor_name], split_image_targets.get(donor_name, 0), donor_name),
                    default="",
                )
                if donor_split:
                    class_targets[donor_split] -= 1
                    class_targets[split_name] += 1
        for split_name in active_splits:
            desired[split_name][class_id] = class_targets[split_name]
    return desired


def assign_candidates_to_new_splits(
    *,
    selected_candidates: list[ExportImageCandidate],
    split_image_targets: dict[str, int],
    desired_box_targets_by_split: dict[str, dict[int, int]],
    kept_class_ids: list[int],
    size_buckets_by_candidate: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, list[ExportImageCandidate]], dict[str, Any]]:
    assigned_candidates: dict[str, list[ExportImageCandidate]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    assigned_box_counts = {
        "train": {class_id: 0 for class_id in kept_class_ids},
        "val": {class_id: 0 for class_id in kept_class_ids},
        "test": {class_id: 0 for class_id in kept_class_ids},
    }
    desired_image_coverage_by_split = allocate_image_coverage_targets_per_split(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )
    desired_image_targets_by_split = allocate_image_targets_per_split(
        selected_candidates=selected_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )
    assigned_image_counts = {
        "train": {class_id: 0 for class_id in kept_class_ids},
        "val": {class_id: 0 for class_id in kept_class_ids},
        "test": {class_id: 0 for class_id in kept_class_ids},
    }
    remaining_image_targets = dict(split_image_targets)
    size_targets_by_split = (
        allocate_size_bucket_targets_per_split(
            selected_candidates=selected_candidates,
            size_buckets_by_candidate=size_buckets_by_candidate,
            split_image_targets=split_image_targets,
        )
        if size_buckets_by_candidate
        else None
    )
    assigned_size_counts = {
        s: {name: 0 for name in RATIO_BUCKET_NAMES} for s in ("train", "val", "test")
    }
    ordered_candidates = sorted(
        selected_candidates,
        key=lambda item: (-item.total_boxes, -len(item.class_box_counts), item.rel_path.as_posix(), item.split_name),
    )
    remaining_candidates = list(ordered_candidates)

    active_splits = [
        split_name
        for split_name in ("train", "val", "test")
        if split_image_targets.get(split_name, 0) > 0
    ]
    limited_coverage_classes = {
        class_id: sum(1 for candidate in selected_candidates if class_id in candidate.class_box_counts)
        for class_id in kept_class_ids
        if sum(1 for candidate in selected_candidates if class_id in candidate.class_box_counts) < len(active_splits)
    }

    def _box_fill_fraction(cand: ExportImageCandidate, split_name: str) -> float:
        """该候选有多大比例的框，落进该 split 仍亏空的每类框数配额（0~1，密度无关）。"""
        total = cand.total_boxes
        if total <= 0:
            return 0.0
        filled = 0.0
        for class_id, box_count in cand.class_box_counts.items():
            need = (
                desired_box_targets_by_split[split_name][class_id]
                - assigned_box_counts[split_name][class_id]
            )
            if need > 0:
                filled += min(box_count, need)
        return filled / total

    def _size_fill_fraction(cand: ExportImageCandidate, split_name: str) -> float:
        """该候选有多大比例的(小/中/大)框，落进该 split 仍亏空的尺寸桶配额（0~1）。"""
        if not size_targets_by_split or not size_buckets_by_candidate:
            return 0.0
        b = size_buckets_by_candidate.get(_candidate_size_key(cand), {})
        sized_total = sum(int(b.get(name, 0)) for name in RATIO_BUCKET_NAMES)
        if sized_total <= 0:
            return 0.0
        filled = 0.0
        for name in RATIO_BUCKET_NAMES:
            need = size_targets_by_split[split_name][name] - assigned_size_counts[split_name][name]
            if need > 0:
                filled += min(int(b.get(name, 0)), need)
        return filled / sized_total

    while remaining_candidates:
        best_candidate_idx = -1
        best_split = ""
        best_score = float("-inf")
        best_total_boxes = 0
        best_class_count = 0
        best_path = ""
        for idx, candidate in enumerate(remaining_candidates):
            path_key = candidate.rel_path.as_posix()
            for split_name in ("train", "val", "test"):
                if remaining_image_targets.get(split_name, 0) <= 0:
                    continue
                coverage_gain = sum(
                    1
                    for class_id in candidate.class_box_counts
                    if assigned_image_counts[split_name][class_id] < desired_image_coverage_by_split[split_name][class_id]
                )
                if coverage_gain <= 0:
                    continue
                image_deficit_gain = sum(
                    1
                    for class_id in candidate.class_box_counts
                    if assigned_image_counts[split_name][class_id] < desired_image_targets_by_split[split_name][class_id]
                )
                smaller_split_bonus = 1.0 / max(split_image_targets.get(split_name, 0), 1)
                box_fill = _box_fill_fraction(candidate, split_name)
                size_fill = _size_fill_fraction(candidate, split_name)
                score = (
                    (coverage_gain * 1000.0)
                    + (image_deficit_gain * 100.0)
                    + (SPLIT_BOX_FILL_WEIGHT * box_fill)
                    + (SPLIT_SIZE_FILL_WEIGHT * size_fill)
                    + smaller_split_bonus
                )
                should_take = False
                if best_split == "" or score > best_score + 1e-12:
                    should_take = True
                elif abs(score - best_score) <= 1e-12 and coverage_gain > best_class_count:
                    should_take = True
                elif abs(score - best_score) <= 1e-12 and coverage_gain == best_class_count and candidate.total_boxes < best_total_boxes:
                    should_take = True
                elif (
                    abs(score - best_score) <= 1e-12
                    and coverage_gain == best_class_count
                    and candidate.total_boxes == best_total_boxes
                    and (best_path == "" or path_key < best_path)
                ):
                    should_take = True
                if should_take:
                    best_candidate_idx = idx
                    best_split = split_name
                    best_score = score
                    best_total_boxes = candidate.total_boxes
                    best_class_count = coverage_gain
                    best_path = path_key
        if best_candidate_idx < 0 or best_split == "":
            break
        candidate = remaining_candidates.pop(best_candidate_idx)
        assigned_candidates[best_split].append(candidate)
        remaining_image_targets[best_split] = max(remaining_image_targets.get(best_split, 0) - 1, 0)
        for class_id, box_count in candidate.class_box_counts.items():
            assigned_box_counts[best_split][class_id] += box_count
            assigned_image_counts[best_split][class_id] += 1
        if size_buckets_by_candidate:
            b = size_buckets_by_candidate.get(_candidate_size_key(candidate), {})
            for name in RATIO_BUCKET_NAMES:
                assigned_size_counts[best_split][name] += int(b.get(name, 0))

    for candidate in remaining_candidates:
        best_split = ""
        best_score = float("-inf")
        for split_name in ("train", "val", "test"):
            if remaining_image_targets.get(split_name, 0) <= 0:
                continue
            coverage_gain = sum(
                1
                for class_id in candidate.class_box_counts
                if assigned_image_counts[split_name][class_id] < desired_image_coverage_by_split[split_name][class_id]
            )
            image_deficit_gain = sum(
                1
                for class_id in candidate.class_box_counts
                if assigned_image_counts[split_name][class_id] < desired_image_targets_by_split[split_name][class_id]
            )
            box_fill = _box_fill_fraction(candidate, split_name)
            size_fill = _size_fill_fraction(candidate, split_name)
            score = (
                (coverage_gain * 1000.0)
                + (image_deficit_gain * 100.0)
                + (SPLIT_BOX_FILL_WEIGHT * box_fill)
                + (SPLIT_SIZE_FILL_WEIGHT * size_fill)
                + (0.01 * remaining_image_targets[split_name])
            )
            if best_split == "" or score > best_score:
                best_split = split_name
                best_score = score
        if best_split == "":
            best_split = max(
                ("train", "val", "test"),
                key=lambda split_name: remaining_image_targets.get(split_name, 0),
            )
        assigned_candidates[best_split].append(candidate)
        remaining_image_targets[best_split] = max(remaining_image_targets.get(best_split, 0) - 1, 0)
        for class_id, box_count in candidate.class_box_counts.items():
            assigned_box_counts[best_split][class_id] += box_count
            assigned_image_counts[best_split][class_id] += 1
        if size_buckets_by_candidate:
            b = size_buckets_by_candidate.get(_candidate_size_key(candidate), {})
            for name in RATIO_BUCKET_NAMES:
                assigned_size_counts[best_split][name] += int(b.get(name, 0))

    for split_name in assigned_candidates:
        assigned_candidates[split_name].sort(key=lambda item: (item.rel_path.as_posix(), item.split_name))

    missing_image_coverage_by_split = {
        split_name: {
            class_id: max(
                desired_image_coverage_by_split[split_name][class_id] - assigned_image_counts[split_name][class_id],
                0,
            )
            for class_id in kept_class_ids
            if desired_image_coverage_by_split[split_name][class_id] > assigned_image_counts[split_name][class_id]
        }
        for split_name in ("train", "val", "test")
    }
    return assigned_candidates, {
        "target_images_per_split": split_image_targets,
        "assigned_images_per_split": {
            split_name: len(candidates)
            for split_name, candidates in assigned_candidates.items()
        },
        "desired_boxes_per_split": desired_box_targets_by_split,
        "assigned_boxes_per_split": assigned_box_counts,
        "desired_image_coverage_by_split": desired_image_coverage_by_split,
        "desired_images_per_split_by_class": desired_image_targets_by_split,
        "assigned_image_coverage_by_split": assigned_image_counts,
        "assigned_images_per_split_by_class": assigned_image_counts,
        "missing_image_coverage_by_split": missing_image_coverage_by_split,
        "missing_image_coverage_counts": {
            split_name: len(missing_image_coverage_by_split[split_name])
            for split_name in ("train", "val", "test")
        },
        "limited_coverage_classes": limited_coverage_classes,
    }

def resolve_destination_rel_path(
    *,
    candidate: ExportImageCandidate,
    used_paths: set[str],
) -> Path:
    base_path = candidate.rel_path
    path_text = base_path.as_posix()
    if path_text not in used_paths:
        used_paths.add(path_text)
        return base_path

    split_prefixed = Path(f"from_{candidate.split_name}") / candidate.rel_path
    split_text = split_prefixed.as_posix()
    if split_text not in used_paths:
        used_paths.add(split_text)
        return split_prefixed

    stem = candidate.rel_path.stem
    suffix = candidate.rel_path.suffix
    parent = candidate.rel_path.parent
    index = 1
    while True:
        candidate_path = parent / f"{stem}_{index}{suffix}"
        candidate_text = candidate_path.as_posix()
        if candidate_text not in used_paths:
            used_paths.add(candidate_text)
            return candidate_path
        index += 1
