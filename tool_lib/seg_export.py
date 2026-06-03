"""分割任务导出入口。

复用 det_analysis 的阈值自适应 / 候选筛选 / 平衡选图 / 重划分等逻辑
（seg 的 SegSourceImageInfo / SegExportImageCandidate 字段名与 det 版
对齐，duck typing 即可），主流程、EDA 报告、polygon 标签处理是 seg 专属。

入口：run_export(args)。
"""

from __future__ import annotations

import inspect
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from . import common as rt
from .det_analysis import (
    allocate_box_targets_per_split,
    allocate_split_targets_from_source,
    assign_candidates_to_new_splits,
    build_class_filter_highlights,
    build_dataset_eda,
    build_dropped_class_analysis,
    build_export_candidates,
    choose_kept_class_ids,
    collect_candidate_split_stats,
    collect_class_split_stats,
    derive_effective_balance_ratio,
    derive_effective_class_threshold,
    derive_effective_density_controls,
    derive_effective_min_class_boxes,
    derive_effective_min_class_images,
    derive_effective_target_images_per_class,
    pool_candidates,
    resolve_destination_rel_path,
    select_balanced_train_candidates,
)
from .det_shared import collect_candidate_class_summary
from .seg_shared import (
    SegExportImageCandidate,
    SegSourceImageInfo,
    collect_source_image_infos,
    parse_polygon_line,
    polygon_area_normalized,
    polygon_bbox_normalized,
    remap_yolo_seg_label_lines,
    safe_class_name,
)


TINY_OBJECT_AREA_THRESHOLD = 16.0 * 16.0
SMALL_OBJECT_AREA_THRESHOLD = 32.0 * 32.0
MEDIUM_OBJECT_AREA_THRESHOLD = 96.0 * 96.0
SEG_DATASET_TAG_PREFIX = "seg"


def _progress_bar(current: int, total: int, width: int = 28) -> str:
    safe_total = max(total, 1)
    clamped_current = min(max(current, 0), safe_total)
    filled = int(round(width * clamped_current / safe_total))
    return f"[{'#' * filled}{'.' * (width - filled)}] {clamped_current}/{safe_total}"


def _print_progress_line(label: str, current: int, total: int, detail: str = "", *, end: str = "\n") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[seg/export] {label} {_progress_bar(current, total)}{suffix}", end=end, flush=True)


def _make_live_progress_callback(label: str, *, single_line: bool = False):
    last_print_at = 0.0

    def _callback(current: int, total: int, detail: str) -> None:
        nonlocal last_print_at
        now = time.monotonic()
        should_commit_line = current >= total or ((now - last_print_at) >= 0.8 and not single_line)
        _print_progress_line(
            label,
            current,
            total,
            detail,
            end="\n" if should_commit_line else "\r",
        )
        if should_commit_line:
            last_print_at = now
    return _callback


def _supports_keyword_arg(func: Any, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _build_export_candidates_compat(
    *,
    source_infos_by_split,
    kept_class_ids,
    max_boxes_per_image,
    max_boxes_per_class_per_image,
    progress_callback,
):
    kwargs = {
        "source_infos_by_split": source_infos_by_split,
        "kept_class_ids": kept_class_ids,
        "max_boxes_per_image": max_boxes_per_image,
        "max_boxes_per_class_per_image": max_boxes_per_class_per_image,
    }
    if _supports_keyword_arg(build_export_candidates, "progress_callback"):
        kwargs["progress_callback"] = progress_callback
    return build_export_candidates(**kwargs)


def _select_balanced_train_candidates_compat(
    *,
    candidates,
    kept_class_ids,
    target_total_images,
    target_boxes_per_class,
    balance_ratio,
    target_images_per_class,
    box_density_penalty,
    progress_callback,
):
    kwargs = {
        "candidates": candidates,
        "kept_class_ids": kept_class_ids,
        "target_total_images": target_total_images,
        "target_boxes_per_class": target_boxes_per_class,
        "balance_ratio": balance_ratio,
        "target_images_per_class": target_images_per_class,
        "box_density_penalty": box_density_penalty,
    }
    if _supports_keyword_arg(select_balanced_train_candidates, "progress_callback"):
        kwargs["progress_callback"] = progress_callback
    return select_balanced_train_candidates(**kwargs)


def _convert_seg_candidates(
    candidates_by_split: dict[str, list[Any]],
) -> dict[str, list[SegExportImageCandidate]]:
    """把 det_analysis.build_export_candidates 产出的 ExportImageCandidate（det 类）
    转成 SegExportImageCandidate（seg 类），仅为名义类型一致；字段内容完全相同。"""
    converted: dict[str, list[SegExportImageCandidate]] = {}
    for split_name, items in candidates_by_split.items():
        converted[split_name] = [
            SegExportImageCandidate(
                split_name=item.split_name,
                rel_split_image_dir=item.rel_split_image_dir,
                rel_split_label_dir=item.rel_split_label_dir,
                rel_path=item.rel_path,
                src_image_path=item.src_image_path,
                src_label_path=item.src_label_path,
                filtered_lines=item.filtered_lines,
                class_box_counts=item.class_box_counts,
            )
            for item in items
        ]
    return converted


def _log_export_stage(title: str, *lines: str, current: int | None = None, total: int | None = None) -> None:
    if current is not None and total is not None:
        _print_progress_line("Stage", current, total, title)
    print(f"\n[seg/export] {title}")
    for line in lines:
        print(f"  - {line}")


def _format_top_class_names(class_ids: list[int], class_names: dict[int, str], limit: int = 5) -> str:
    if not class_ids:
        return "(none)"
    names = [safe_class_name(class_names, class_id) for class_id in class_ids[:limit]]
    if len(class_ids) > limit:
        names.append("...")
    return ", ".join(names)


def _safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _empty_size_bucket_counts() -> dict[str, int]:
    return {name: 0 for name in ("tiny", "small", "medium", "large")}


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _bucket_mask_area(area_pixels: float) -> str:
    if area_pixels < TINY_OBJECT_AREA_THRESHOLD:
        return "tiny"
    if area_pixels < SMALL_OBJECT_AREA_THRESHOLD:
        return "small"
    if area_pixels < MEDIUM_OBJECT_AREA_THRESHOLD:
        return "medium"
    return "large"


def _open_image_size(image_path: Path) -> tuple[int, int]:
    if rt.Image is None and not getattr(rt, "ensure_plot_dependencies", lambda: False)():
        rt.import_runtime_dependencies()
    image_module = rt.Image
    if image_module is None:
        raise ModuleNotFoundError("Missing runtime dependency Pillow.")
    with image_module.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _analyze_candidate_polygons(
    candidate: SegExportImageCandidate,
    class_id_mapping: dict[int, int],
) -> dict[str, Any]:
    """对一张候选图片做 seg 维度分析：实例数、按类掩码面积桶、polygon 顶点数等。"""
    width, height = _open_image_size(candidate.src_image_path)
    image_area_pixels = max(width * height, 1)
    per_class_label_counts: dict[int, int] = defaultdict(int)
    per_class_size_buckets: dict[int, dict[str, int]] = defaultdict(_empty_size_bucket_counts)
    per_class_mask_pixels: dict[int, float] = defaultdict(float)
    per_class_vertex_total: dict[int, int] = defaultdict(int)
    size_buckets = _empty_size_bucket_counts()
    total_mask_pixels = 0.0
    total_vertex_count = 0
    total_instance_count = 0

    for line in candidate.filtered_lines:
        parsed = parse_polygon_line(line)
        if parsed is None:
            continue
        old_class_id, coords = parsed
        new_class_id = class_id_mapping[old_class_id]
        norm_area = polygon_area_normalized(coords)
        pixel_area = norm_area * image_area_pixels
        bucket = _bucket_mask_area(pixel_area)
        vertex_count = len(coords) // 2

        per_class_label_counts[new_class_id] += 1
        per_class_size_buckets[new_class_id][bucket] += 1
        per_class_mask_pixels[new_class_id] += pixel_area
        per_class_vertex_total[new_class_id] += vertex_count
        size_buckets[bucket] += 1
        total_mask_pixels += pixel_area
        total_vertex_count += vertex_count
        total_instance_count += 1

    total_labels = sum(per_class_label_counts.values())
    return {
        "image_width": width,
        "image_height": height,
        "image_area_pixels": image_area_pixels,
        "total_labels": total_labels,
        "total_instances": total_instance_count,
        "total_mask_pixels": total_mask_pixels,
        "total_vertex_count": total_vertex_count,
        "size_buckets": size_buckets,
        "per_class_label_counts": dict(per_class_label_counts),
        "per_class_size_buckets": {
            class_id: dict(bucket_counts)
            for class_id, bucket_counts in per_class_size_buckets.items()
        },
        "per_class_mask_pixels": dict(per_class_mask_pixels),
        "per_class_vertex_total": dict(per_class_vertex_total),
    }


def _finalize_split_eda_entry(entry: dict[str, Any]) -> dict[str, Any]:
    images = int(entry.get("images", 0) or 0)
    labeled_images = int(entry.get("labeled_images", 0) or 0)
    labels = int(entry.get("labels", 0) or 0)
    size_buckets = entry.get("size_buckets", _empty_size_bucket_counts())
    entry["instances"] = labels
    entry["avg_instances_per_image"] = round(labels / images, 4) if images > 0 else 0.0
    entry["avg_instances_per_labeled_image"] = round(labels / labeled_images, 4) if labeled_images > 0 else 0.0
    entry["labeled_image_ratio"] = _safe_ratio(labeled_images, images)
    entry["size_bucket_ratio"] = {
        bucket_name: _safe_ratio(bucket_value, labels)
        for bucket_name, bucket_value in size_buckets.items()
    }
    image_area_total = float(entry.get("image_area_total_pixels", 0.0) or 0.0)
    mask_area_total = float(entry.get("total_mask_pixels", 0.0) or 0.0)
    entry["mask_coverage_ratio"] = round(mask_area_total / image_area_total, 6) if image_area_total > 0 else 0.0
    total_vertices = int(entry.get("total_vertex_count", 0) or 0)
    entry["avg_vertices_per_instance"] = round(total_vertices / labels, 4) if labels > 0 else 0.0

    classes = list(entry.get("classes", {}).values())
    classes_by_labels = sorted(
        classes,
        key=lambda item: (-item["labels"], -item["images"], item["class_id"]),
    )
    classes_by_images = sorted(
        classes,
        key=lambda item: (-item["images"], -item["labels"], item["class_id"]),
    )
    entry["class_count"] = len(classes)
    entry["top_classes_by_labels"] = classes_by_labels[: min(10, len(classes_by_labels))]
    entry["top_classes_by_images"] = classes_by_images[: min(10, len(classes_by_images))]
    dense_images = sorted(
        entry.get("top_dense_images", []),
        key=lambda item: (-item["labels"], -item["class_count"], item["image"]),
    )
    entry["top_dense_images"] = dense_images[: min(10, len(dense_images))]
    entry["classes"] = {
        str(item["class_id"]): item
        for item in classes_by_labels
    }
    return entry


def _build_split_summary_view(split_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "images": split_entry["images"],
        "labeled_images": split_entry["labeled_images"],
        "empty_images": split_entry["empty_images"],
        "labels": split_entry["labels"],
        "instances": split_entry["instances"],
        "avg_instances_per_image": split_entry["avg_instances_per_image"],
        "avg_instances_per_labeled_image": split_entry["avg_instances_per_labeled_image"],
        "size_buckets": dict(split_entry["size_buckets"]),
        "size_bucket_ratio": dict(split_entry["size_bucket_ratio"]),
        "class_count": split_entry["class_count"],
        "mask_coverage_ratio": split_entry["mask_coverage_ratio"],
        "avg_vertices_per_instance": split_entry["avg_vertices_per_instance"],
    }


def build_export_dataset_eda(
    *,
    selected_candidates_by_split: dict[str, list[SegExportImageCandidate]],
    kept_names: dict[int, str],
    class_id_mapping: dict[int, int],
) -> dict[str, Any]:
    split_order = ["train", "val", "test"]
    split_entries: dict[str, dict[str, Any]] = {}
    all_entry = {
        "split": "all",
        "images": 0,
        "labeled_images": 0,
        "empty_images": 0,
        "labels": 0,
        "image_area_total_pixels": 0.0,
        "total_mask_pixels": 0.0,
        "total_vertex_count": 0,
        "size_buckets": _empty_size_bucket_counts(),
        "classes": {},
        "top_dense_images": [],
    }
    class_entries = {
        str(class_id): {
            "class_id": class_id,
            "name": class_name,
            "all": {
                "images": 0,
                "labels": 0,
                "instances": 0,
                "mask_pixels": 0.0,
                "size_buckets": _empty_size_bucket_counts(),
            },
            "splits": {},
        }
        for class_id, class_name in sorted(kept_names.items())
    }

    for split_name in split_order:
        candidates = selected_candidates_by_split.get(split_name, [])
        split_entry = {
            "split": split_name,
            "images": len(candidates),
            "labeled_images": 0,
            "empty_images": 0,
            "labels": 0,
            "image_area_total_pixels": 0.0,
            "total_mask_pixels": 0.0,
            "total_vertex_count": 0,
            "size_buckets": _empty_size_bucket_counts(),
            "classes": {},
            "top_dense_images": [],
        }
        all_entry["images"] += len(candidates)

        for candidate in candidates:
            analysis = _analyze_candidate_polygons(candidate, class_id_mapping)
            total_labels = int(analysis["total_labels"])
            size_buckets = cast(dict[str, int], analysis["size_buckets"])
            per_class_label_counts = cast(dict[int, int], analysis["per_class_label_counts"])
            per_class_size_buckets = cast(dict[int, dict[str, int]], analysis["per_class_size_buckets"])
            per_class_mask_pixels = cast(dict[int, float], analysis["per_class_mask_pixels"])
            image_area = float(analysis["image_area_pixels"])
            total_mask_pixels = float(analysis["total_mask_pixels"])
            total_vertex_count = int(analysis["total_vertex_count"])

            if total_labels > 0:
                split_entry["labeled_images"] += 1
                all_entry["labeled_images"] += 1
            else:
                split_entry["empty_images"] += 1
                all_entry["empty_images"] += 1
            split_entry["labels"] += total_labels
            split_entry["image_area_total_pixels"] += image_area
            split_entry["total_mask_pixels"] += total_mask_pixels
            split_entry["total_vertex_count"] += total_vertex_count
            all_entry["labels"] += total_labels
            all_entry["image_area_total_pixels"] += image_area
            all_entry["total_mask_pixels"] += total_mask_pixels
            all_entry["total_vertex_count"] += total_vertex_count

            for bucket_name, bucket_value in size_buckets.items():
                split_entry["size_buckets"][bucket_name] += bucket_value
                all_entry["size_buckets"][bucket_name] += bucket_value

            dense_image_record = {
                "split": split_name,
                "image": candidate.rel_path.as_posix(),
                "labels": total_labels,
                "class_count": len(per_class_label_counts),
                "image_width": analysis["image_width"],
                "image_height": analysis["image_height"],
                "mask_coverage_ratio": round(total_mask_pixels / image_area, 6) if image_area > 0 else 0.0,
            }
            split_entry["top_dense_images"].append(dense_image_record)
            all_entry["top_dense_images"].append(dense_image_record)

            for class_id, label_count in per_class_label_counts.items():
                class_name = kept_names[class_id]
                split_class_entry = split_entry["classes"].setdefault(
                    str(class_id),
                    {
                        "class_id": class_id,
                        "name": class_name,
                        "images": 0,
                        "labels": 0,
                        "instances": 0,
                        "mask_pixels": 0.0,
                        "size_buckets": _empty_size_bucket_counts(),
                    },
                )
                split_class_entry["images"] += 1
                split_class_entry["labels"] += label_count
                split_class_entry["instances"] += label_count
                split_class_entry["mask_pixels"] += per_class_mask_pixels.get(class_id, 0.0)

                all_class_entry = all_entry["classes"].setdefault(
                    str(class_id),
                    {
                        "class_id": class_id,
                        "name": class_name,
                        "images": 0,
                        "labels": 0,
                        "instances": 0,
                        "mask_pixels": 0.0,
                        "size_buckets": _empty_size_bucket_counts(),
                    },
                )
                all_class_entry["images"] += 1
                all_class_entry["labels"] += label_count
                all_class_entry["instances"] += label_count
                all_class_entry["mask_pixels"] += per_class_mask_pixels.get(class_id, 0.0)

                class_split_entry = class_entries[str(class_id)]["splits"].setdefault(
                    split_name,
                    {
                        "images": 0,
                        "labels": 0,
                        "instances": 0,
                        "mask_pixels": 0.0,
                        "size_buckets": _empty_size_bucket_counts(),
                    },
                )
                class_split_entry["images"] += 1
                class_split_entry["labels"] += label_count
                class_split_entry["instances"] += label_count
                class_split_entry["mask_pixels"] += per_class_mask_pixels.get(class_id, 0.0)

                class_entries[str(class_id)]["all"]["images"] += 1
                class_entries[str(class_id)]["all"]["labels"] += label_count
                class_entries[str(class_id)]["all"]["instances"] += label_count
                class_entries[str(class_id)]["all"]["mask_pixels"] += per_class_mask_pixels.get(class_id, 0.0)

                class_size_buckets = per_class_size_buckets.get(class_id, _empty_size_bucket_counts())
                for bucket_name, bucket_value in class_size_buckets.items():
                    split_class_entry["size_buckets"][bucket_name] += bucket_value
                    all_class_entry["size_buckets"][bucket_name] += bucket_value
                    class_split_entry["size_buckets"][bucket_name] += bucket_value
                    class_entries[str(class_id)]["all"]["size_buckets"][bucket_name] += bucket_value

        split_entries[split_name] = _finalize_split_eda_entry(split_entry)

    all_entry = _finalize_split_eda_entry(all_entry)
    all_classes_sorted = sorted(
        class_entries.values(),
        key=lambda item: (-item["all"]["labels"], -item["all"]["images"], item["class_id"]),
    )
    class_label_counts = [item["all"]["labels"] for item in all_classes_sorted if item["all"]["labels"] > 0]
    split_summaries = {
        split_name: _build_split_summary_view(split_entry)
        for split_name, split_entry in split_entries.items()
    }

    overview = {
        "split_order": split_order,
        "class_count": len(kept_names),
        "images": all_entry["images"],
        "labeled_images": all_entry["labeled_images"],
        "empty_images": all_entry["empty_images"],
        "labels": all_entry["labels"],
        "instances": all_entry["instances"],
        "avg_instances_per_image": all_entry["avg_instances_per_image"],
        "avg_instances_per_labeled_image": all_entry["avg_instances_per_labeled_image"],
        "size_buckets": dict(all_entry["size_buckets"]),
        "size_bucket_ratio": dict(all_entry["size_bucket_ratio"]),
        "mask_coverage_ratio": all_entry["mask_coverage_ratio"],
        "avg_vertices_per_instance": all_entry["avg_vertices_per_instance"],
        "class_label_count_stats": {
            "min": min(class_label_counts, default=0),
            "max": max(class_label_counts, default=0),
            "imbalance_ratio": round(max(class_label_counts) / max(min(class_label_counts), 1), 4)
            if class_label_counts
            else 0.0,
        },
        "object_size_rule": {
            "small": "mask_area < 32^2 px",
            "medium": "32^2 px <= mask_area < 96^2 px",
            "large": "mask_area >= 96^2 px",
        },
    }

    insights: list[str] = []
    if all_classes_sorted:
        head_class = all_classes_sorted[0]
        tail_class = min(
            all_classes_sorted,
            key=lambda item: (item["all"]["labels"], item["all"]["images"], item["class_id"]),
        )
        insights.append(
            f"实例数最高的类别是 {head_class['name']}，共 {head_class['all']['labels']} 个实例，覆盖 {head_class['all']['images']} 张图。"
        )
        insights.append(
            f"实例数最低的类别是 {tail_class['name']}，共 {tail_class['all']['labels']} 个实例，覆盖 {tail_class['all']['images']} 张图。"
        )
    dominant_bucket = max(
        overview["size_buckets"],
        key=overview["size_buckets"].get,
        default="small",
    )
    insights.append(
        f"导出数据集的掩码尺寸以 {dominant_bucket} 为主，对应 {overview['size_buckets'][dominant_bucket]} 个实例。"
    )
    insights.append(
        f"整体掩码像素覆盖率约 {overview['mask_coverage_ratio'] * 100:.2f}%。"
    )
    if split_summaries:
        densest_split_name = max(
            split_summaries,
            key=lambda name: split_summaries[name]["avg_instances_per_image"],
        )
        densest_split = split_summaries[densest_split_name]
        insights.append(
            f"{densest_split_name} split 的平均每图实例数最高，为 {densest_split['avg_instances_per_image']:.4f}。"
        )

    return {
        "overview": overview,
        "split_summaries": split_summaries,
        "splits": {
            **split_entries,
            "all": all_entry,
        },
        "classes": {
            str(entry["class_id"]): entry
            for entry in all_classes_sorted
        },
        "insights": insights,
    }


def render_export_dataset_eda_markdown(
    *,
    export_dataset_eda: dict[str, Any],
    export_root: Path,
    source_data_path: Path,
    report_path: Path | None,
) -> str:
    overview = cast(dict[str, Any], export_dataset_eda["overview"])
    split_order = cast(list[str], overview.get("split_order", ["train", "val", "test"]))
    split_summaries = cast(dict[str, dict[str, Any]], export_dataset_eda["split_summaries"])
    insights = cast(list[str], export_dataset_eda.get("insights", []))

    lines = [
        "# 导出数据集 EDA 报告（seg）",
        "",
        "## 1. 基本信息",
        f"- 导出目录：`{export_root}`",
        f"- 源数据配置：`{source_data_path}`",
        f"- 评估报告：`{report_path}`" if report_path is not None else "- 评估报告：dataset-only 模式",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 类别数：{overview['class_count']}",
        f"- 图片数：{overview['images']}",
        f"- 实例数：{overview['labels']}",
        f"- 掩码像素覆盖率：{overview['mask_coverage_ratio'] * 100:.2f}%",
        f"- 平均每实例顶点数：{overview['avg_vertices_per_instance']:.2f}",
        f"- 掩码尺寸统计规则：small < 32^2 px，medium < 96^2 px，large >= 96^2 px",
        "",
        "## 2. Split 总览",
        "",
        "| split | 图片数 | 实例数 | 平均每图实例 | 小目标 | 中目标 | 大目标 | 类别数 | 掩码占比 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in split_order:
        split_entry = split_summaries.get(split_name, {})
        if not split_entry:
            continue
        size_buckets = cast(dict[str, int], split_entry["size_buckets"])
        lines.append(
            f"| {split_name} | {split_entry['images']} | {split_entry['instances']} | "
            f"{split_entry['avg_instances_per_image']:.4f} | {size_buckets['small']} | "
            f"{size_buckets['medium']} | {size_buckets['large']} | {split_entry['class_count']} | "
            f"{split_entry['mask_coverage_ratio'] * 100:.2f}% |"
        )

    all_entry = cast(dict[str, Any], export_dataset_eda["splits"]["all"])
    all_size_buckets = cast(dict[str, int], all_entry["size_buckets"])
    lines.extend(
        [
            f"| all | {all_entry['images']} | {all_entry['instances']} | {all_entry['avg_instances_per_image']:.4f} | "
            f"{all_size_buckets['small']} | {all_size_buckets['medium']} | {all_size_buckets['large']} | "
            f"{all_entry['class_count']} | {all_entry['mask_coverage_ratio'] * 100:.2f}% |",
            "",
            "## 3. 全量类别分布",
            "",
            "| 类别 | 图片数 | 实例数 | 小目标 | 中目标 | 大目标 | 平均每图实例 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for class_entry in cast(dict[str, Any], export_dataset_eda["classes"]).values():
        class_all = cast(dict[str, Any], class_entry["all"])
        size_buckets = cast(dict[str, int], class_all["size_buckets"])
        avg_instances_per_image = round(
            class_all["labels"] / class_all["images"], 4
        ) if class_all["images"] > 0 else 0.0
        lines.append(
            f"| {class_entry['name']} | {class_all['images']} | {class_all['labels']} | "
            f"{size_buckets['small']} | {size_buckets['medium']} | {size_buckets['large']} | {avg_instances_per_image:.4f} |"
        )

    for split_name in split_order:
        split_entry = cast(dict[str, Any], export_dataset_eda["splits"].get(split_name, {}))
        split_classes = cast(dict[str, Any], split_entry.get("classes", {}))
        lines.extend(
            [
                "",
                f"## 4. {split_name} 类别明细",
                "",
                "| 类别 | 图片数 | 实例数 | 小目标 | 中目标 | 大目标 | 平均每图实例 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for class_entry in split_classes.values():
            size_buckets = cast(dict[str, int], class_entry["size_buckets"])
            avg_instances_per_image = round(
                class_entry["labels"] / class_entry["images"], 4
            ) if class_entry["images"] > 0 else 0.0
            lines.append(
                f"| {class_entry['name']} | {class_entry['images']} | {class_entry['labels']} | "
                f"{size_buckets['small']} | {size_buckets['medium']} | {size_buckets['large']} | {avg_instances_per_image:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 5. EDA 观察",
        ]
    )
    for insight in insights:
        lines.append(f"- {insight}")

    for split_name in split_order:
        split_entry = cast(dict[str, Any], export_dataset_eda["splits"].get(split_name, {}))
        dense_images = cast(list[dict[str, Any]], split_entry.get("top_dense_images", []))
        if not dense_images:
            continue
        lines.extend(
            [
                "",
                f"## 6. {split_name} 高密度样本 Top 10",
                "",
                "| 图片 | 实例数 | 类别数 | 尺寸 | 掩码占比 |",
                "|---|---:|---:|---|---:|",
            ]
        )
        for item in dense_images:
            mask_ratio = item.get("mask_coverage_ratio", 0.0)
            lines.append(
                f"| {item['image']} | {item['labels']} | {item['class_count']} | "
                f"{item['image_width']}x{item['image_height']} | {mask_ratio * 100:.2f}% |"
            )

    lines.append("")
    return "\n".join(lines)


def _report_matches_source_data(report_payload: dict[str, Any], source_data_path: Path) -> bool:
    source_cfg = rt.load_data_config(source_data_path)
    source_root = Path(source_cfg["_root_dir"]).resolve()
    source_yaml = source_data_path.expanduser().resolve()
    candidate_paths: list[Path] = []

    def _append_candidate(raw_value: Any) -> None:
        if not isinstance(raw_value, str) or not raw_value.strip():
            return
        try:
            candidate_paths.append(Path(raw_value).expanduser().resolve())
        except Exception:
            return

    config_raw = report_payload.get("config")
    paths_raw = report_payload.get("paths")
    config: dict[str, Any] = config_raw if isinstance(config_raw, dict) else {}
    paths: dict[str, Any] = paths_raw if isinstance(paths_raw, dict) else {}
    _append_candidate(config.get("data_root"))
    _append_candidate(paths.get("data_root"))
    _append_candidate(paths.get("data_yaml"))
    _append_candidate(config.get("data"))
    eval_data = config.get("test_images")
    if isinstance(eval_data, str) and eval_data.strip():
        try:
            test_image_dir = Path(eval_data).expanduser().resolve()
            candidate_paths.append(test_image_dir)
            if test_image_dir.parent.name == "images":
                candidate_paths.append(test_image_dir.parent.parent)
            elif len(test_image_dir.parents) >= 2:
                candidate_paths.append(test_image_dir.parents[1])
        except Exception:
            pass

    for candidate in candidate_paths:
        if candidate == source_yaml or candidate == source_root:
            return True
        if candidate.parent == source_root or source_root in candidate.parents:
            return True
    return False


def _iter_seg_report_candidates() -> list[Path]:
    patterns = ["*seg_eval_summary.json", "seg_eval_summary.json"]
    roots = [rt.EXPERIMENT_ROOT_DIR, rt.TEST_OUTPUT_ROOT_DIR]
    results: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file():
                    results[str(path.resolve())] = path.resolve()
    ordered = sorted(
        results.values(),
        key=lambda item: (item.stat().st_mtime, str(item)),
        reverse=True,
    )
    return ordered


def _normalize_seg_report_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """把 seg_eval_summary.json（实例分割评估格式）的 metrics 转换成 per_class_ap 形式，
    供 det_analysis.derive_effective_class_threshold / choose_kept_class_ids 复用。

    seg eval 当前由 InstanceSegmentationTaskMetric 产出，每类 AP 字段名形如
    `eval_metric/map_<class_name>` 或 `eval_metric/map_per_class/<class_idx>`。
    无法解析时按 dataset-only 处理（per_class_ap 为空）。
    """
    if not isinstance(payload, dict):
        return {"per_class_ap": {}, "summary": {}, "config": {}}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    per_class_ap: dict[str, Any] = {}
    if isinstance(metrics, dict):
        for raw_key, raw_value in metrics.items():
            if not isinstance(raw_key, str):
                continue
            ap_value: float | None = None
            class_token: str | None = None
            key_lower = raw_key.lower()
            if "map_per_class" in key_lower:
                class_token = raw_key.split("/")[-1]
                try:
                    ap_value = float(raw_value)
                except (TypeError, ValueError):
                    ap_value = None
            elif key_lower.startswith("eval_metric/map_") and not key_lower.endswith(("_small", "_medium", "_large", "_50", "_75", "_1", "_10", "_100")):
                tail = raw_key.split("/", 1)[1]
                class_token = tail[len("map_"):]
                try:
                    ap_value = float(raw_value)
                except (TypeError, ValueError):
                    ap_value = None
            if ap_value is None or class_token is None:
                continue
            try:
                class_id = int(class_token)
            except ValueError:
                class_id_lookup: dict[str, int] = {}
                payload_class_names = payload.get("class_names") if isinstance(payload.get("class_names"), dict) else {}
                for raw_id, raw_name in payload_class_names.items():
                    try:
                        class_id_lookup[str(raw_name).strip().lower()] = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                normalized = class_token.strip().lower()
                if normalized not in class_id_lookup:
                    continue
                class_id = class_id_lookup[normalized]
            per_class_ap[str(class_id)] = {
                "ap": ap_value,
                "gt": 1,
                "name": class_token,
            }

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    if "data" in payload and "data_root" not in config:
        config = dict(config)
        config["data_root"] = str(Path(payload["data"]).expanduser().resolve().parent) if isinstance(payload.get("data"), str) else config.get("data_root")
    return {
        "per_class_ap": per_class_ap,
        "summary": summary,
        "config": config,
    }


def resolve_export_report(
    requested_report_path: Path | None,
    source_data_path: Path,
) -> tuple[Path | None, dict[str, Any], dict[str, Any]]:
    checked_paths: list[str] = []
    if requested_report_path is not None:
        explicit_path = requested_report_path.expanduser().resolve()
        checked_paths.append(str(explicit_path))
        if explicit_path.exists():
            payload = _safe_load_json(explicit_path)
            if payload is not None:
                normalized = _normalize_seg_report_payload(payload)
                return explicit_path, normalized, {
                    "mode": "explicit",
                    "used_report_json": str(explicit_path),
                    "matched_source_data": _report_matches_source_data(payload, source_data_path),
                    "checked_paths": checked_paths,
                }

    for candidate in _iter_seg_report_candidates():
        checked_paths.append(str(candidate))
        payload = _safe_load_json(candidate)
        if payload is None:
            continue
        if _report_matches_source_data(payload, source_data_path):
            normalized = _normalize_seg_report_payload(payload)
            return candidate, normalized, {
                "mode": "auto_matched",
                "used_report_json": str(candidate),
                "matched_source_data": True,
                "checked_paths": checked_paths[:20],
            }

    if requested_report_path is not None:
        explicit_path = requested_report_path.expanduser().resolve()
        if explicit_path.exists():
            payload = _safe_load_json(explicit_path)
            if payload is not None:
                normalized = _normalize_seg_report_payload(payload)
                return explicit_path, normalized, {
                    "mode": "explicit_unmatched",
                    "used_report_json": str(explicit_path),
                    "matched_source_data": False,
                    "checked_paths": checked_paths[:20],
                }

    neutral_payload: dict[str, Any] = {
        "config": {
            "data_root": str(rt.load_data_config(source_data_path)["_root_dir"]),
        },
        "summary": {},
        "per_class_ap": {},
    }
    return None, neutral_payload, {
        "mode": "dataset_only",
        "used_report_json": None,
        "matched_source_data": False,
        "checked_paths": checked_paths[:20],
    }


def export_filtered_dataset(
    source_data_path: Path,
    report: dict[str, Any],
    class_threshold: float,
    export_suffix: str,
    *,
    report_path: Path | None = None,
    report_resolution: dict[str, Any] | None = None,
    auto_balance: bool,
    auto_relax_class_threshold: bool,
    balance_ratio: float,
    min_class_images: int,
    min_class_instances: int,
    target_images_per_class: int,
    target_total_images: int,
    split_ratio: str,
    target_instances_per_class: int,
    max_instances_per_image: int,
    max_instances_per_class_per_image: int,
    instance_density_penalty: float,
) -> Path:
    """异常安全的对外入口：rename 之前任何步骤失败都会清理 __export_tmp。"""
    state: dict[str, Any] = {"temp_dir": None, "renamed": False}
    try:
        return _export_filtered_dataset_impl(
            source_data_path=source_data_path,
            report=report,
            class_threshold=class_threshold,
            export_suffix=export_suffix,
            report_path=report_path,
            report_resolution=report_resolution,
            auto_balance=auto_balance,
            auto_relax_class_threshold=auto_relax_class_threshold,
            balance_ratio=balance_ratio,
            min_class_images=min_class_images,
            min_class_instances=min_class_instances,
            target_images_per_class=target_images_per_class,
            target_total_images=target_total_images,
            split_ratio=split_ratio,
            target_instances_per_class=target_instances_per_class,
            max_instances_per_image=max_instances_per_image,
            max_instances_per_class_per_image=max_instances_per_class_per_image,
            instance_density_penalty=instance_density_penalty,
            _state=state,
        )
    except BaseException:
        td = state.get("temp_dir")
        if td and not state["renamed"] and td.exists():
            shutil.rmtree(td, ignore_errors=True)
        raise


def _export_filtered_dataset_impl(
    *,
    source_data_path: Path,
    report: dict[str, Any],
    class_threshold: float,
    export_suffix: str,
    report_path: Path | None,
    report_resolution: dict[str, Any] | None,
    auto_balance: bool,
    auto_relax_class_threshold: bool,
    balance_ratio: float,
    min_class_images: int,
    min_class_instances: int,
    target_images_per_class: int,
    target_total_images: int,
    split_ratio: str,
    target_instances_per_class: int,
    max_instances_per_image: int,
    max_instances_per_class_per_image: int,
    instance_density_penalty: float,
    _state: dict[str, Any],
) -> Path:
    """seg 数据集筛选导出主流程。参数命名以"实例"为主，内部复用 det_analysis 的"box" 命名。"""
    total_stage_count = 11
    stage_idx = 0
    source_cfg = rt.load_data_config(source_data_path)
    source_root = Path(source_cfg["_root_dir"])
    temp_export_root = rt.deduplicate_path(source_root.parent / f"{source_root.name}__export_tmp")
    export_root = temp_export_root
    export_root.mkdir(parents=True, exist_ok=True)
    _state["temp_dir"] = temp_export_root
    stage_idx += 1
    _log_export_stage(
        "Start",
        f"source_data={source_data_path}",
        f"export_root={export_root}",
        f"target_total_images={max(target_total_images, 0)}",
        f"split_ratio={split_ratio}",
        current=stage_idx,
        total=total_stage_count,
    )

    per_class_ap = report.get("per_class_ap", {})
    if not isinstance(per_class_ap, dict):
        per_class_ap = {}

    source_infos_by_split, _ = collect_source_image_infos(source_cfg, source_root)
    class_stats = collect_class_split_stats(source_infos_by_split)
    class_names = rt.normalize_names(source_cfg.get("names"))
    source_total_images = sum(len(infos) for infos in source_infos_by_split.values())
    all_class_ids = sorted(
        set(class_names)
        | set(class_stats)
        | {int(class_id) for class_id in per_class_ap}
    )
    reference_split = "all"
    dataset_eda = build_dataset_eda(
        class_ids=all_class_ids,
        class_names=class_names,
        class_stats=class_stats,
        per_class_ap=per_class_ap,
        reference_split=reference_split,
        source_total_images=source_total_images,
        target_total_images=max(target_total_images, 0),
    )
    eda_overview_raw = dataset_eda.get("overview")
    eda_overview: dict[str, Any] = eda_overview_raw if isinstance(eda_overview_raw, dict) else {}
    stage_idx += 1
    _log_export_stage(
        "EDA Ready",
        f"source_images={source_total_images}",
        f"class_count={eda_overview.get('class_count', len(all_class_ids))}",
        f"classes_with_instances={eda_overview.get('classes_with_boxes', 0)}",
        f"classes_with_images={eda_overview.get('classes_with_images', 0)}",
        current=stage_idx,
        total=total_stage_count,
    )

    effective_class_threshold, report_signal = derive_effective_class_threshold(
        requested_threshold=class_threshold,
        per_class_ap=per_class_ap,
        auto_relax_class_threshold=auto_relax_class_threshold,
        target_total_images=max(target_total_images, 0),
        available_total_images=source_total_images,
    )
    report_filter_mode = cast(str, report_signal["report_filter_mode"])
    reference_instances_by_class = {
        class_id: class_stats.get(class_id, {}).get(reference_split, {"boxes": 0})["boxes"]
        for class_id in all_class_ids
    }
    reference_images_by_class = {
        class_id: class_stats.get(class_id, {}).get(reference_split, {"images": 0})["images"]
        for class_id in all_class_ids
    }
    effective_balance_ratio, balance_ratio_summary = derive_effective_balance_ratio(
        available_images_by_class=reference_images_by_class,
        available_boxes_by_class=reference_instances_by_class,
        requested_balance_ratio=balance_ratio,
        auto_balance=auto_balance,
        target_total_images=max(target_total_images, 0),
        available_total_images=source_total_images,
    )
    effective_min_class_instances, min_instances_summary = derive_effective_min_class_boxes(
        available_boxes_by_class=reference_instances_by_class,
        available_images_by_class=reference_images_by_class,
        requested_min_class_boxes=min_class_instances,
        auto_balance=auto_balance,
        balance_ratio=effective_balance_ratio,
        target_total_images=max(target_total_images, 0),
        estimated_class_count=len(all_class_ids),
    )
    preliminary_kept_class_ids, _ = choose_kept_class_ids(
        class_ids=all_class_ids,
        class_names=class_names,
        class_stats=class_stats,
        per_class_ap=per_class_ap,
        class_threshold=effective_class_threshold,
        requested_class_threshold=class_threshold,
        min_class_images=max(min_class_images, 0),
        min_class_boxes=effective_min_class_instances,
        soft_min_class_images=None,
        soft_min_class_boxes=min_instances_summary.get("soft_min_class_boxes"),
        reference_split=reference_split,
        report_filter_mode=report_filter_mode,
    )
    effective_min_class_images, min_images_summary = derive_effective_min_class_images(
        available_images_by_class={
            class_id: reference_images_by_class.get(class_id, 0)
            for class_id in preliminary_kept_class_ids
        },
        requested_min_class_images=min_class_images,
        auto_balance=auto_balance,
        balance_ratio=effective_balance_ratio,
        target_total_images=max(target_total_images, 0),
        split_ratio=split_ratio,
        estimated_class_count=len(preliminary_kept_class_ids),
    )
    stage_idx += 1
    _log_export_stage(
        "Dynamic Thresholds",
        f"effective_report_threshold={effective_class_threshold:.4f}",
        f"effective_balance_ratio={effective_balance_ratio:.4f}",
        f"effective_min_class_images={effective_min_class_images}",
        f"soft_min_class_images={min_images_summary.get('soft_min_class_images', 0)}",
        f"effective_min_class_instances={effective_min_class_instances}",
        f"soft_min_class_instances={min_instances_summary.get('soft_min_class_boxes', 0)}",
        f"instance_tolerance={min_instances_summary.get('box_tolerance', 0)}",
        f"report_filter_mode={report_filter_mode}",
        current=stage_idx,
        total=total_stage_count,
    )
    kept_class_ids, class_decisions = choose_kept_class_ids(
        class_ids=all_class_ids,
        class_names=class_names,
        class_stats=class_stats,
        per_class_ap=per_class_ap,
        class_threshold=effective_class_threshold,
        requested_class_threshold=class_threshold,
        min_class_images=effective_min_class_images,
        min_class_boxes=effective_min_class_instances,
        soft_min_class_images=min_images_summary.get("soft_min_class_images"),
        soft_min_class_boxes=min_instances_summary.get("soft_min_class_boxes"),
        reference_split=reference_split,
        report_filter_mode=report_filter_mode,
    )
    effective_target_images_per_class, target_images_summary = derive_effective_target_images_per_class(
        available_images_by_class={
            class_id: reference_images_by_class.get(class_id, 0)
            for class_id in kept_class_ids
        },
        requested_target_images_per_class=target_images_per_class,
        auto_balance=auto_balance,
        target_total_images=max(target_total_images, 0),
        estimated_class_count=len(kept_class_ids),
        balance_ratio=effective_balance_ratio,
    )
    if not kept_class_ids:
        raise ValueError(
            f"No classes satisfy seg export rules. class_threshold={effective_class_threshold:.4f}, "
            f"min_{reference_split}_images={effective_min_class_images}, "
            f"min_{reference_split}_instances={effective_min_class_instances}."
        )
    dropped_class_count = len(all_class_ids) - len(kept_class_ids)
    stage_idx += 1
    _log_export_stage(
        "Class Filter",
        f"kept_classes={len(kept_class_ids)}",
        f"dropped_classes={dropped_class_count}",
        f"effective_min_class_images={effective_min_class_images}",
        f"effective_target_images_per_class={effective_target_images_per_class}",
        f"kept_head={_format_top_class_names(kept_class_ids, class_names)}",
        current=stage_idx,
        total=total_stage_count,
    )
    class_filter_summary, dropped_class_analysis, borderline_kept_class_analysis = build_dropped_class_analysis(
        class_decisions=class_decisions,
        kept_class_ids=kept_class_ids,
        class_names=class_names,
        reference_split=reference_split,
        balance_ratio=effective_balance_ratio,
        effective_min_class_images=effective_min_class_images,
        effective_min_class_boxes=effective_min_class_instances,
        requested_class_threshold=class_threshold,
        effective_class_threshold=effective_class_threshold,
        report_filter_mode=report_filter_mode,
        min_images_summary=min_images_summary,
        min_boxes_summary=min_instances_summary,
    )
    class_filter_highlights = build_class_filter_highlights(
        class_filter_summary=class_filter_summary,
        dropped_class_analysis=dropped_class_analysis,
        borderline_kept_class_analysis=borderline_kept_class_analysis,
    )

    raw_candidates_by_split, _ = _build_export_candidates_compat(
        source_infos_by_split=source_infos_by_split,
        kept_class_ids=kept_class_ids,
        max_boxes_per_image=0,
        max_boxes_per_class_per_image=0,
        progress_callback=_make_live_progress_callback("Raw Candidate Scan"),
    )
    raw_pooled_candidates = pool_candidates(raw_candidates_by_split)
    (
        effective_max_instances_per_image,
        effective_max_instances_per_class_per_image,
        effective_instance_density_penalty,
        density_controls_summary,
    ) = derive_effective_density_controls(
        candidates=raw_pooled_candidates,
        requested_max_boxes_per_image=max(max_instances_per_image, 0),
        requested_max_boxes_per_class_per_image=max(max_instances_per_class_per_image, 0),
        requested_box_density_penalty=instance_density_penalty,
        auto_balance=auto_balance,
        target_total_images=max(target_total_images, 0),
    )
    stage_idx += 1
    _log_export_stage(
        "Density Controls",
        f"raw_candidates={len(raw_pooled_candidates)}",
        f"effective_max_instances_per_image={effective_max_instances_per_image}",
        f"effective_max_instances_per_class_per_image={effective_max_instances_per_class_per_image}",
        f"effective_instance_density_penalty={effective_instance_density_penalty:.4f}",
        current=stage_idx,
        total=total_stage_count,
    )

    filtered_candidates_by_split, filter_summary = _build_export_candidates_compat(
        source_infos_by_split=source_infos_by_split,
        kept_class_ids=kept_class_ids,
        max_boxes_per_image=effective_max_instances_per_image,
        max_boxes_per_class_per_image=effective_max_instances_per_class_per_image,
        progress_callback=_make_live_progress_callback("Candidate Filter"),
    )

    post_density_class_stats = collect_candidate_split_stats(filtered_candidates_by_split)
    post_density_kept_class_ids, post_density_class_decisions = choose_kept_class_ids(
        class_ids=kept_class_ids,
        class_names=class_names,
        class_stats=post_density_class_stats,
        per_class_ap=per_class_ap,
        class_threshold=effective_class_threshold,
        requested_class_threshold=class_threshold,
        min_class_images=effective_min_class_images,
        min_class_boxes=effective_min_class_instances,
        soft_min_class_images=min_images_summary.get("soft_min_class_images"),
        soft_min_class_boxes=min_instances_summary.get("soft_min_class_boxes"),
        reference_split=reference_split,
        report_filter_mode=report_filter_mode,
    )
    post_density_removed_class_ids = [
        class_id for class_id in kept_class_ids
        if class_id not in set(post_density_kept_class_ids)
    ]
    stage_idx += 1
    _log_export_stage(
        "Candidate Filter",
        f"filtered_candidates={sum(len(items) for items in filtered_candidates_by_split.values())}",
        f"post_density_removed_classes={len(post_density_removed_class_ids)}",
        f"skip_summary={filter_summary}",
        current=stage_idx,
        total=total_stage_count,
    )
    if post_density_removed_class_ids:
        for class_id in post_density_removed_class_ids:
            class_decisions[class_id] = post_density_class_decisions[class_id]
        kept_class_ids = post_density_kept_class_ids
        class_filter_summary, dropped_class_analysis, borderline_kept_class_analysis = build_dropped_class_analysis(
            class_decisions=class_decisions,
            kept_class_ids=kept_class_ids,
            class_names=class_names,
            reference_split=reference_split,
            balance_ratio=effective_balance_ratio,
            effective_min_class_images=effective_min_class_images,
            effective_min_class_boxes=effective_min_class_instances,
            requested_class_threshold=class_threshold,
            effective_class_threshold=effective_class_threshold,
            report_filter_mode=report_filter_mode,
            min_images_summary=min_images_summary,
            min_boxes_summary=min_instances_summary,
        )
        class_filter_highlights = build_class_filter_highlights(
            class_filter_summary=class_filter_summary,
            dropped_class_analysis=dropped_class_analysis,
            borderline_kept_class_analysis=borderline_kept_class_analysis,
        )
        filtered_candidates_by_split, filter_summary = _build_export_candidates_compat(
            source_infos_by_split=source_infos_by_split,
            kept_class_ids=kept_class_ids,
            max_boxes_per_image=effective_max_instances_per_image,
            max_boxes_per_class_per_image=effective_max_instances_per_class_per_image,
            progress_callback=_make_live_progress_callback("Candidate Filter"),
        )
        post_density_class_stats = collect_candidate_split_stats(filtered_candidates_by_split)

    class_id_mapping = {class_id: idx for idx, class_id in enumerate(kept_class_ids)}
    kept_names = {idx: safe_class_name(class_names, class_id) for class_id, idx in class_id_mapping.items()}

    pooled_candidates = pool_candidates(filtered_candidates_by_split)
    selected_pooled_candidates, selection_summary = _select_balanced_train_candidates_compat(
        candidates=pooled_candidates,
        kept_class_ids=kept_class_ids,
        target_total_images=max(target_total_images, 0),
        target_boxes_per_class=max(target_instances_per_class, 0),
        balance_ratio=effective_balance_ratio,
        target_images_per_class=effective_target_images_per_class,
        box_density_penalty=effective_instance_density_penalty,
        progress_callback=_make_live_progress_callback("Balanced Selection", single_line=True),
    )
    stage_idx += 1
    _log_export_stage(
        "Balanced Selection",
        f"selected_images={len(selected_pooled_candidates)}",
        f"selected_train_head={_format_top_class_names(kept_class_ids, class_names)}",
        f"effective_target_total_images={selection_summary.get('effective_target_total_images', 0)}",
        f"effective_target_instances_per_class={selection_summary.get('effective_target_boxes_per_class', 0)}",
        current=stage_idx,
        total=total_stage_count,
    )

    split_image_targets = allocate_split_targets_from_source(
        total_selected_images=len(selected_pooled_candidates),
        split_ratio=split_ratio,
    )
    desired_box_targets_by_split = allocate_box_targets_per_split(
        selected_candidates=selected_pooled_candidates,
        split_image_targets=split_image_targets,
        kept_class_ids=kept_class_ids,
    )
    selected_candidates_by_split, repartition_summary = assign_candidates_to_new_splits(
        selected_candidates=selected_pooled_candidates,
        split_image_targets=split_image_targets,
        desired_box_targets_by_split=desired_box_targets_by_split,
        kept_class_ids=kept_class_ids,
    )
    selected_candidates_by_split = _convert_seg_candidates(selected_candidates_by_split)
    stage_idx += 1
    missing_coverage_counts = cast(dict[str, int], repartition_summary.get("missing_image_coverage_counts", {}))
    limited_coverage_classes = cast(dict[int, int] | dict[str, int], repartition_summary.get("limited_coverage_classes", {}))
    _log_export_stage(
        "Repartition Ready",
        f"split_image_targets={split_image_targets}",
        f"assigned_train={len(selected_candidates_by_split.get('train', []))}",
        f"assigned_val={len(selected_candidates_by_split.get('val', []))}",
        f"assigned_test={len(selected_candidates_by_split.get('test', []))}",
        f"missing_class_coverage={missing_coverage_counts}",
        f"limited_coverage_classes={len(limited_coverage_classes)}",
        current=stage_idx,
        total=total_stage_count,
    )

    copied_images = 0
    copied_labels = 0
    total_copy_items = sum(len(candidates) for candidates in selected_candidates_by_split.values())
    used_output_paths_by_split = {
        "train": set(),
        "val": set(),
        "test": set(),
    }
    stage_idx += 1
    _log_export_stage(
        "Writing Files",
        f"total_images_to_copy={total_copy_items}",
        f"target_splits={', '.join(split_name for split_name, items in selected_candidates_by_split.items() if items)}",
        current=stage_idx,
        total=total_stage_count,
    )
    for split_name, candidates in selected_candidates_by_split.items():
        if not candidates:
            continue
        rel_split_image_dir = Path("images") / split_name
        rel_split_label_dir = Path("labels") / split_name
        dst_image_dir = export_root / rel_split_image_dir
        dst_label_dir = export_root / rel_split_label_dir
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            dst_rel_path = resolve_destination_rel_path(
                candidate=candidate,
                used_paths=used_output_paths_by_split[split_name],
            )
            dst_image_path = dst_image_dir / dst_rel_path
            dst_label_path = dst_label_dir / dst_rel_path.with_suffix(".txt")
            dst_image_path.parent.mkdir(parents=True, exist_ok=True)
            dst_label_path.parent.mkdir(parents=True, exist_ok=True)
            remapped_lines = remap_yolo_seg_label_lines(candidate.filtered_lines, class_id_mapping)
            shutil.copy2(candidate.src_image_path, dst_image_path)
            dst_label_path.write_text("\n".join(remapped_lines) + "\n", encoding="utf-8")
            copied_images += 1
            copied_labels += 1
            _print_progress_line(
                f"Copy {split_name}",
                copied_images,
                total_copy_items,
                f"latest={candidate.rel_path.as_posix()}",
                end="\r" if copied_images < total_copy_items else "\n",
            )

    resolved_export_suffix = rt.resolve_seg_export_dir_suffix(
        export_suffix=export_suffix,
        image_count=copied_images,
    )
    final_export_root = rt.deduplicate_path(source_root.parent / f"{source_root.name}{resolved_export_suffix}")
    if final_export_root != export_root:
        export_root.rename(final_export_root)
        export_root = final_export_root
    _state["renamed"] = True

    source_class_summary = collect_candidate_class_summary(source_infos_by_split, kept_class_ids)
    exported_class_summary = collect_candidate_class_summary(selected_candidates_by_split, kept_class_ids)
    export_dataset_eda = build_export_dataset_eda(
        selected_candidates_by_split=selected_candidates_by_split,
        kept_names=kept_names,
        class_id_mapping=class_id_mapping,
    )
    split_image_targets_compact = {
        split_name: split_image_targets.get(split_name, 0)
        for split_name in ("train", "val", "test")
    }
    exported_split_stats = {
        split_name: export_dataset_eda["split_summaries"].get(split_name, {})
        for split_name in ("train", "val", "test")
    }

    export_cfg = {
        "path": str(export_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": source_cfg.get("task", "segment"),
        "nc": len(kept_names),
        "names": kept_names,
    }
    rt.dump_yaml(export_root / "data.yaml", export_cfg)
    (export_root / "classes.txt").write_text(
        "\n".join(kept_names[idx] for idx in range(len(kept_names))) + "\n",
        encoding="utf-8",
    )
    export_dataset_tag = rt.dataset_tag_from_dir(source_root)
    export_eda_json_path = export_root / f"export_dataset_eda_{export_dataset_tag}.json"
    export_eda_md_path = export_root / f"export_dataset_eda_{export_dataset_tag}.md"
    export_eda_markdown = render_export_dataset_eda_markdown(
        export_dataset_eda=export_dataset_eda,
        export_root=export_root,
        source_data_path=source_data_path,
        report_path=report_path,
    )
    export_eda_json_path.write_text(
        json.dumps(export_dataset_eda, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    export_eda_md_path.write_text(export_eda_markdown + "\n", encoding="utf-8")
    stage_idx += 1
    _log_export_stage(
        "EDA Report",
        f"eda_json={export_eda_json_path}",
        f"eda_markdown={export_eda_md_path}",
        f"overview={export_dataset_eda['overview']['images']} images / {export_dataset_eda['overview']['labels']} instances",
        current=stage_idx,
        total=total_stage_count,
    )

    resolved_report_resolution = report_resolution if report_resolution is not None else {}
    (export_root / "export_summary.json").write_text(
        json.dumps(
            {
                "task": "segment",
                "source_data": str(source_data_path),
                "source_root": str(source_root),
                "report_json": str(report_path) if report_path is not None else None,
                "report_resolution": resolved_report_resolution,
                "report_threshold": class_threshold,
                "effective_report_threshold": effective_class_threshold,
                "auto_balance": auto_balance,
                "auto_relax_class_threshold": auto_relax_class_threshold,
                "balance_ratio": effective_balance_ratio,
                "requested_balance_ratio": balance_ratio,
                "min_class_images": max(min_class_images, 0),
                "effective_min_class_images": effective_min_class_images,
                "min_class_instances": max(min_class_instances, 0),
                "effective_min_class_instances": effective_min_class_instances,
                "target_images_per_class": effective_target_images_per_class,
                "requested_target_images_per_class": max(target_images_per_class, 0),
                "target_total_images": max(target_total_images, 0),
                "split_ratio": split_ratio,
                "target_instances_per_class": max(target_instances_per_class, 0),
                "max_instances_per_image": effective_max_instances_per_image,
                "max_instances_per_class_per_image": effective_max_instances_per_class_per_image,
                "instance_density_penalty": effective_instance_density_penalty,
                "reference_split": reference_split,
                "selection_mode": "pool_filter_resplit",
                "report_signal": report_signal,
                "class_filter_summary": class_filter_summary,
                "class_filter_highlights": class_filter_highlights,
                "balance_ratio_summary": balance_ratio_summary,
                "min_class_images_summary": min_images_summary,
                "min_class_instances_summary": min_instances_summary,
                "target_images_per_class_summary": target_images_summary,
                "density_controls_summary": density_controls_summary,
                "post_density_class_stats": {
                    str(class_id): {
                        "name": safe_class_name(class_names, class_id),
                        "splits": post_density_class_stats.get(class_id, {}),
                    }
                    for class_id in kept_class_ids
                },
                "post_density_removed_class_ids": post_density_removed_class_ids,
                "post_density_removed_class_names": [
                    safe_class_name(class_names, class_id)
                    for class_id in post_density_removed_class_ids
                ],
                "report_advisory": {
                    str(class_id): {
                        "name": decision["name"],
                        "ap": decision["ap"],
                        "advisory_notes": decision["advisory_notes"],
                    }
                    for class_id, decision in class_decisions.items()
                    if decision["advisory_notes"]
                },
                "dataset_eda": dataset_eda,
                "selected_class_ids": kept_class_ids,
                "selected_class_names": [safe_class_name(class_names, class_id) for class_id in kept_class_ids],
                "dropped_classes": {
                    str(class_id): decision
                    for class_id, decision in class_decisions.items()
                    if decision["drop_reasons"]
                },
                "dropped_class_analysis": dropped_class_analysis,
                "borderline_kept_class_analysis": borderline_kept_class_analysis,
                "filter_summary": filter_summary,
                "selection_summary": selection_summary,
                "repartition_summary": repartition_summary,
                "split_image_targets": split_image_targets_compact,
                "selected_split_stats": exported_split_stats,
                "exported_split_stats": exported_split_stats,
                "source_class_summary": {
                    str(class_id): {
                        "name": safe_class_name(class_names, class_id),
                        "ap": class_decisions[class_id]["ap"],
                        **source_class_summary.get(class_id, {"images": 0, "boxes": 0, "splits": {}}),
                    }
                    for class_id in kept_class_ids
                },
                "exported_class_summary": {
                    str(class_id): {
                        "name": safe_class_name(class_names, class_id),
                        "ap": class_decisions[class_id]["ap"],
                        **exported_class_summary.get(class_id, {"images": 0, "boxes": 0, "splits": {}}),
                    }
                    for class_id in kept_class_ids
                },
                "selected_images_by_split": {
                    split_name: len(candidates)
                    for split_name, candidates in selected_candidates_by_split.items()
                },
                "export_dataset_eda_json": str(export_eda_json_path),
                "export_dataset_eda_markdown": str(export_eda_md_path),
                "export_dataset_eda": export_dataset_eda,
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
    stage_idx += 1
    _log_export_stage(
        "Done",
        f"copied_images={copied_images}",
        f"copied_labels={copied_labels}",
        f"export_root={export_root}",
        f"summary_json={export_root / 'export_summary.json'}",
        f"eda_markdown={export_eda_md_path}",
        current=stage_idx,
        total=total_stage_count,
    )
    return export_root


def run_export(args) -> None:
    if str(getattr(args, "seg_train_type", "instance") or "instance").lower() == "semantic":
        raise NotImplementedError(
            "semantic segmentation export is not supported by this launcher yet. "
            "The existing seg-export pipeline is for YOLO polygon instance segmentation datasets."
        )
    report_path, report_payload, report_resolution = resolve_export_report(
        args.report_json,
        args.export_source_data,
    )
    if report_path is not None:
        print(f"Using seg eval report: {report_path}")
    else:
        print("No matching seg eval report found. Export will use dataset distribution only.")
    export_root = export_filtered_dataset(
        args.export_source_data,
        report_payload,
        args.good_class_threshold,
        args.export_suffix,
        report_path=report_path,
        report_resolution=report_resolution,
        auto_balance=args.auto_balance,
        auto_relax_class_threshold=args.auto_relax_class_threshold,
        balance_ratio=args.balance_ratio,
        min_class_images=args.min_class_images,
        min_class_instances=args.min_class_instances,
        target_images_per_class=args.target_images_per_class,
        target_total_images=args.target_total_images,
        split_ratio=args.split_ratio,
        target_instances_per_class=args.target_instances_per_class,
        max_instances_per_image=args.max_instances_per_image,
        max_instances_per_class_per_image=args.max_instances_per_class_per_image,
        instance_density_penalty=args.instance_density_penalty,
    )
    print(f"Filtered seg dataset exported to: {export_root}")
