"""语义分割数据集 EDA。

面向 PNG mask 标签的语义分割数据集，产出按 split 对照的详细数据分析结果。
关键中间产物：逐图类别清单 CSV（image_class_inventory），供 seg_semantic_curate 消费。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from . import common as rt
from . import train_tools
from . import seg_tools
from .det_eda import _distribution_metrics, _gini, _entropy_evenness, _distribution, _js_divergence
from .det_analysis import percentile_int

SPLIT_ORDER = ("train", "val", "test")


def _progress_bar(current: int, total: int, width: int = 20) -> str:
    safe_total = max(total, 1)
    clamped_current = min(max(current, 0), safe_total)
    filled = int(round(width * clamped_current / safe_total))
    bar = "█" * filled + "░" * (width - filled)
    pct = int(round(100 * clamped_current / safe_total))
    return f"{bar} {pct:3d}% {clamped_current}/{safe_total}"


def _print_progress(label: str, current: int, total: int, detail: str = "", *, newline: bool = False) -> None:
    suffix = f" {detail}" if detail else ""
    end = "\n" if newline else "\r"
    print(f"  {label} {_progress_bar(current, total)}{suffix}  ", end=end, flush=True)


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------

def _empty_class_entry(class_id: int, class_name: str) -> dict[str, Any]:
    return {
        "class_id": class_id,
        "name": class_name,
        "image_count_all": 0,
        "pixel_count_all": 0,
        "image_count_by_split": {},
        "pixel_count_by_split": {},
        "present_in_splits": [],
        "missing_in_splits": [],
    }


def _empty_split_entry(split_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "total_images": 0,
        "total_labeled_pixels": 0,
        "total_ignore_pixels": 0,
        "class_count": 0,
    }


# ---------------------------------------------------------------------------
# mask 扫描核心
# ---------------------------------------------------------------------------

def _scan_mask_pixels(
    mask_path: Path,
    classes: dict[int, Any],
    ignore_classes: set[int],
) -> tuple[set[int], dict[int, int], int, int, int, int]:
    """扫描单张 mask，返回 (class_ids, per_class_pixels, total_labeled, ignore_pixels, width, height)。"""
    with rt.Image.open(mask_path) as mask_image:
        mask_np = rt.np.array(mask_image)

    height, width = mask_np.shape[:2]
    compare_np = mask_np if mask_np.ndim == 3 else mask_np[:, :, None]

    # 构建 label → class_id 的反向映射
    label_to_class: dict[tuple, int] = {}
    for class_id in classes:
        if class_id in ignore_classes:
            continue
        for label in seg_tools._class_labels(classes, class_id):
            label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            label_to_class[label_tuple] = class_id

    # 统计每个像素
    class_ids: set[int] = set()
    per_class_pixels: Counter = Counter()
    total_labeled = 0
    ignore_pixels = 0

    # 扁平化处理
    flat = compare_np.reshape(-1, compare_np.shape[-1])
    for i in range(flat.shape[0]):
        pixel_tuple = tuple(int(v) for v in flat[i])
        if pixel_tuple in label_to_class:
            cid = label_to_class[pixel_tuple]
            class_ids.add(cid)
            per_class_pixels[cid] += 1
            total_labeled += 1
        else:
            # 检查是否属于 ignore_classes 的 label
            is_ignore = False
            for ignore_cid in ignore_classes:
                for label in seg_tools._class_labels(classes, ignore_cid):
                    label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
                    if pixel_tuple == label_tuple:
                        is_ignore = True
                        break
                if is_ignore:
                    break
            if is_ignore:
                ignore_pixels += 1
            # 其他未知像素不计入（等同于背景/未标注）

    return class_ids, dict(per_class_pixels), total_labeled, ignore_pixels, width, height


def _scan_mask_pixels_fast(
    mask_path: Path,
    classes: dict[int, Any],
    ignore_classes: set[int],
) -> tuple[set[int], dict[int, int], int, int, int, int]:
    """快速扫描单张 mask（numpy 向量化版本）。"""
    with rt.Image.open(mask_path) as mask_image:
        mask_np = rt.np.array(mask_image)

    height, width = mask_np.shape[:2]
    compare_np = mask_np if mask_np.ndim == 3 else mask_np[:, :, None]

    # 构建 label → class_id 的反向映射
    label_to_class: dict[tuple, int] = {}
    all_ignore_labels: set[tuple] = set()
    for class_id in classes:
        if class_id in ignore_classes:
            for label in seg_tools._class_labels(classes, class_id):
                label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
                all_ignore_labels.add(label_tuple)
            continue
        for label in seg_tools._class_labels(classes, class_id):
            label_tuple = tuple(int(v) for v in label) if isinstance(label, tuple) else (int(label),)
            label_to_class[label_tuple] = class_id

    # 用 numpy 向量化统计
    class_ids: set[int] = set()
    per_class_pixels: dict[int, int] = {}
    total_labeled = 0
    ignore_pixels = 0

    # 对每个已知 label 做匹配
    matched = rt.np.zeros(mask_np.shape[:2], dtype=bool)
    for label_tuple, cid in label_to_class.items():
        mask_match = rt.np.all(compare_np == rt.np.array(label_tuple), axis=2)
        count = int(mask_match.sum())
        if count > 0:
            class_ids.add(cid)
            per_class_pixels[cid] = per_class_pixels.get(cid, 0) + count
            total_labeled += count
            matched |= mask_match

    # 统计 ignore 像素
    for label_tuple in all_ignore_labels:
        mask_match = rt.np.all(compare_np == rt.np.array(label_tuple), axis=2)
        ignore_pixels += int(mask_match.sum())

    return class_ids, per_class_pixels, total_labeled, ignore_pixels, width, height


# ---------------------------------------------------------------------------
# EDA 报告生成
# ---------------------------------------------------------------------------

def _collect_semantic_eda(
    source_data_path: Path,
    *,
    min_class_images: int = 10,
    threshold_percentile: float = 0.9,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """扫描所有 split 的 mask，聚合 EDA 统计。返回 (report, inventory_rows)。"""
    source_data_path = source_data_path.expanduser().resolve()

    # 加载 data config（用 train split 的 config 获取 classes 等全局信息）
    data_cfg = train_tools.load_semantic_segmentation_data_config(source_data_path)
    classes = data_cfg["classes"]
    ignore_classes = {int(item) for item in data_cfg.get("ignore_classes", set()) or set()}

    # 类名映射（内部连续 id → 名称）
    class_names = seg_tools._semantic_class_names(classes, ignore_classes)
    # 原始 class_id → 名称
    original_class_names: dict[int, str] = {}
    for class_id, info in classes.items():
        if isinstance(info, dict):
            original_class_names[class_id] = str(info.get("name", class_id))
        else:
            original_class_names[class_id] = str(info)

    # 初始化聚合结构
    all_class_entries: dict[int, dict[str, Any]] = {}
    for class_id in sorted(classes.keys()):
        if class_id not in ignore_classes:
            all_class_entries[class_id] = _empty_class_entry(class_id, original_class_names.get(class_id, str(class_id)))

    split_entries: dict[str, dict[str, Any]] = {}
    inventory_rows: list[dict[str, Any]] = []

    # 逐 split 扫描
    for split in SPLIT_ORDER:
        try:
            samples, _, _ = seg_tools._semantic_image_mask_samples(source_data_path, split)
        except Exception:
            continue

        split_entry = _empty_split_entry(split)
        split_entry["total_images"] = len(samples)
        split_entries[split] = split_entry

        total_samples = len(samples)
        for sample_idx, (image_path, mask_path) in enumerate(samples):
            _print_progress(f"[semantic-eda] 扫描 {split}", sample_idx + 1, total_samples, detail=mask_path.name, newline=(sample_idx + 1 == total_samples))
            try:
                class_ids, per_class_pixels, total_labeled, ignore_pixels, width, height = \
                    _scan_mask_pixels_fast(mask_path, classes, ignore_classes)
            except Exception as e:
                print(f"  [WARN] 跳过 {mask_path}: {e}")
                continue

            # 更新 split 统计
            split_entry["total_labeled_pixels"] += total_labeled
            split_entry["total_ignore_pixels"] += ignore_pixels

            # 更新 class 统计
            for cid in class_ids:
                if cid in all_class_entries:
                    entry = all_class_entries[cid]
                    entry["image_count_all"] += 1
                    entry["image_count_by_split"][split] = entry["image_count_by_split"].get(split, 0) + 1
                    entry["pixel_count_all"] += per_class_pixels.get(cid, 0)
                    entry["pixel_count_by_split"][split] = entry["pixel_count_by_split"].get(split, 0) + per_class_pixels.get(cid, 0)

            # 记录 inventory 行
            rel_image = str(image_path.relative_to(Path(data_cfg.get(split, {}).get("images", image_path.parent))))
            inventory_rows.append({
                "split": split,
                "image": rel_image,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "width": width,
                "height": height,
                "class_ids": "|".join(str(cid) for cid in sorted(class_ids)),
                "per_class_pixels": json.dumps({str(k): v for k, v in per_class_pixels.items()}, ensure_ascii=False),
                "total_labeled_pixels": total_labeled,
            })

    # 补充 present_in_splits / missing_in_splits
    for class_id, entry in all_class_entries.items():
        present = sorted(entry["image_count_by_split"].keys())
        entry["present_in_splits"] = present
        entry["missing_in_splits"] = [s for s in SPLIT_ORDER if s in split_entries and s not in present]

    # split 统计：class_count
    for split, entry in split_entries.items():
        entry["class_count"] = sum(
            1 for ce in all_class_entries.values()
            if ce["image_count_by_split"].get(split, 0) > 0
        )

    # --- 不平衡指标（按图片数口径）---
    active_class_ids = sorted(all_class_entries.keys())
    image_counts_all = [all_class_entries[cid]["image_count_all"] for cid in active_class_ids]
    image_counts_by_split: dict[str, list[int]] = {}
    for split in split_entries:
        image_counts_by_split[split] = [all_class_entries[cid]["image_count_by_split"].get(split, 0) for cid in active_class_ids]

    pixel_counts_all = [all_class_entries[cid]["pixel_count_all"] for cid in active_class_ids]

    imbalance_image = _distribution_metrics(image_counts_all) if image_counts_all else {}
    imbalance_pixel = _distribution_metrics(pixel_counts_all) if pixel_counts_all else {}

    # 跨 split JS 漂移（按图片数）
    js_divergence_by_split: dict[str, float] = {}
    if len(split_entries) >= 2:
        splits_present = [s for s in SPLIT_ORDER if s in split_entries]
        dist_a = _distribution(image_counts_by_split.get(splits_present[0], []))
        for other_split in splits_present[1:]:
            dist_b = _distribution(image_counts_by_split.get(other_split, []))
            js_divergence_by_split[f"{splits_present[0]}_vs_{other_split}"] = _js_divergence(dist_a, dist_b)

    # --- 推荐 ---
    # 推荐删除：全局图片数 < min_class_images
    recommend_delete: list[int] = []
    recommend_delete_val_test_missing: list[int] = []
    for class_id, entry in all_class_entries.items():
        if entry["image_count_all"] < min_class_images:
            recommend_delete.append(class_id)
        if not entry["present_in_splits"] or all(s in entry["missing_in_splits"] for s in ("val", "test") if s in split_entries):
            recommend_delete_val_test_missing.append(class_id)

    # 推荐压缩阈值：train p90
    train_image_counts = image_counts_by_split.get("train", [])
    recommend_threshold = 0
    if train_image_counts and any(c > 0 for c in train_image_counts):
        p90_val = percentile_int(train_image_counts, threshold_percentile)
        recommend_threshold = int(math.ceil(p90_val / 10.0) * 10) if p90_val > 0 else 0

    # 压缩候选：train 图片数 > 阈值的类
    recommend_compress: list[int] = []
    if recommend_threshold > 0:
        for class_id in active_class_ids:
            train_count = all_class_entries[class_id]["image_count_by_split"].get("train", 0)
            if train_count > recommend_threshold:
                recommend_compress.append(class_id)

    # 组装报告
    report: dict[str, Any] = {
        "meta": {
            "source_data": str(source_data_path),
            "split_order": [s for s in SPLIT_ORDER if s in split_entries],
            "total_classes": len(classes),
            "active_classes": len(active_class_ids),
            "ignored_classes": sorted(ignore_classes),
        },
        "overview": {
            "total_images": sum(e["total_images"] for e in split_entries.values()),
            "total_labeled_pixels": sum(e["total_labeled_pixels"] for e in split_entries.values()),
            "total_ignore_pixels": sum(e["total_ignore_pixels"] for e in split_entries.values()),
        },
        "splits": {
            split: {
                "total_images": entry["total_images"],
                "total_labeled_pixels": entry["total_labeled_pixels"],
                "total_ignore_pixels": entry["total_ignore_pixels"],
                "class_count": entry["class_count"],
            }
            for split, entry in split_entries.items()
        },
        "classes": {
            str(class_id): {
                "name": entry["name"],
                "image_count_all": entry["image_count_all"],
                "pixel_count_all": entry["pixel_count_all"],
                "image_count_by_split": entry["image_count_by_split"],
                "pixel_count_by_split": entry["pixel_count_by_split"],
                "present_in_splits": entry["present_in_splits"],
                "missing_in_splits": entry["missing_in_splits"],
            }
            for class_id, entry in all_class_entries.items()
        },
        "imbalance": {
            "image_count": imbalance_image,
            "pixel_count": imbalance_pixel,
            "cross_split_js_divergence": js_divergence_by_split,
        },
        "recommendations": {
            "min_class_images_threshold": min_class_images,
            "threshold_percentile": threshold_percentile,
            "recommend_delete": recommend_delete,
            "recommend_delete_val_test_missing": recommend_delete_val_test_missing,
            "recommend_threshold": recommend_threshold,
            "recommend_compress": recommend_compress,
        },
    }

    return report, inventory_rows


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------

def _render_markdown(report: dict[str, Any], output_dir: Path) -> str:
    lines: list[str] = []
    meta = report["meta"]
    overview = report["overview"]
    classes = report["classes"]
    rec = report["recommendations"]

    lines.append("# 语义分割 EDA 报告")
    lines.append("")
    lines.append(f"- 数据源: `{meta['source_data']}`")
    lines.append(f"- 分析时间: {rt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 活跃类别: {meta['active_classes']} / {meta['total_classes']}")
    if meta["ignored_classes"]:
        lines.append(f"- 忽略类别: {meta['ignored_classes']}")
    lines.append("")

    # 概览
    lines.append("## 概览")
    lines.append("")
    lines.append(f"- 总图片数: {overview['total_images']}")
    lines.append(f"- 总标注像素: {overview['total_labeled_pixels']:,}")
    lines.append(f"- 总忽略像素: {overview['total_ignore_pixels']:,}")
    lines.append("")

    # Split 概览
    lines.append("## Split 概览")
    lines.append("")
    lines.append("| Split | 图片数 | 类别数 | 标注像素 | 忽略像素 |")
    lines.append("|-------|--------|--------|----------|----------|")
    for split_name, split_data in report["splits"].items():
        lines.append(
            f"| {split_name} | {split_data['total_images']} | {split_data['class_count']} | "
            f"{split_data['total_labeled_pixels']:,} | {split_data['total_ignore_pixels']:,} |"
        )
    lines.append("")

    # 类别表（带编号）
    lines.append("## 类别详情")
    lines.append("")
    lines.append("| # | ID | 名称 | Train 图数 | All 图数 | 图片占比 | 像素占比 | 出现 Split | 标记 |")
    lines.append("|---|----|------|-----------|---------|---------|---------|-----------|------|")

    image_counts = [int(c["image_count_all"]) for c in classes.values()]
    total_images = overview["total_images"]
    pixel_counts = [int(c["pixel_count_all"]) for c in classes.values()]
    total_pixels = sum(pixel_counts) if pixel_counts else 1

    for idx, (class_id_str, class_data) in enumerate(classes.items()):
        class_id = int(class_id_str)
        train_count = class_data["image_count_by_split"].get("train", 0)
        all_count = class_data["image_count_all"]
        img_share = f"{all_count / total_images * 100:.1f}%" if total_images > 0 else "0.0%"
        px_share = f"{class_data['pixel_count_all'] / total_pixels * 100:.1f}%" if total_pixels > 0 else "0.0%"
        splits_str = ", ".join(class_data["present_in_splits"]) if class_data["present_in_splits"] else "-"

        marks: list[str] = []
        if class_id in rec.get("recommend_delete", []):
            marks.append("[建议删除]")
        if class_id in rec.get("recommend_delete_val_test_missing", []):
            marks.append("[val/test缺失]")
        if class_id in rec.get("recommend_compress", []):
            marks.append("[建议压缩]")
        mark_str = " ".join(marks) if marks else ""

        lines.append(
            f"| {idx + 1} | {class_id} | {class_data['name']} | {train_count} | {all_count} | "
            f"{img_share} | {px_share} | {splits_str} | {mark_str} |"
        )
    lines.append("")

    # 不平衡指标
    lines.append("## 不平衡指标")
    lines.append("")
    imbalance = report.get("imbalance", {})
    for metric_name, metric_label in [("image_count", "按图片数"), ("pixel_count", "按像素数")]:
        metrics = imbalance.get(metric_name, {})
        if not metrics:
            continue
        lines.append(f"### {metric_label}")
        lines.append("")
        lines.append(f"- Gini 系数: {metrics.get('gini', 0):.4f}")
        lines.append(f"- 熵均衡度: {metrics.get('entropy_evenness', 0):.4f}")
        lines.append(f"- 变异系数 (CV): {metrics.get('coefficient_of_variation', 0):.4f}")
        lines.append(f"- Max/Min 比: {metrics.get('max_min_ratio', 0):.2f}")
        lines.append(f"- P90/P50 比: {metrics.get('p90_p50_ratio', 0):.2f}")
        lines.append("")

    # 跨 split 漂移
    js_div = imbalance.get("cross_split_js_divergence", {})
    if js_div:
        lines.append("### 跨 Split JS 漂移（按图片数）")
        lines.append("")
        for pair, value in js_div.items():
            lines.append(f"- {pair}: {value:.6f} bits")
        lines.append("")

    # 推荐
    lines.append("## 推荐")
    lines.append("")
    if rec.get("recommend_delete"):
        names = [f"{cid}({classes[str(cid)]['name']})" for cid in rec["recommend_delete"]]
        lines.append(f"**推荐删除**（全局图片数 < {rec['min_class_images_threshold']}）: {', '.join(names)}")
    else:
        lines.append("**推荐删除**: 无")
    lines.append("")
    if rec.get("recommend_delete_val_test_missing"):
        names = [f"{cid}({classes[str(cid)]['name']})" for cid in rec["recommend_delete_val_test_missing"]]
        lines.append(f"**val/test 缺失类**: {', '.join(names)}")
        lines.append("")
    if rec.get("recommend_threshold"):
        lines.append(f"**推荐压缩阈值**: {rec['recommend_threshold']}（train 每类最多保留图片数）")
    else:
        lines.append("**推荐压缩阈值**: 无需压缩")
    lines.append("")
    if rec.get("recommend_compress"):
        names = [f"{cid}({classes[str(cid)]['name']})" for cid in rec["recommend_compress"]]
        lines.append(f"**压缩候选类**: {', '.join(names)}")
    else:
        lines.append("**压缩候选类**: 无")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 辅助：生成 summary 行
# ---------------------------------------------------------------------------

def _class_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    classes = report["classes"]
    rec = report["recommendations"]
    overview = report["overview"]
    total_images = overview["total_images"]
    total_pixels = sum(int(c["pixel_count_all"]) for c in classes.values()) or 1
    rows: list[dict[str, Any]] = []
    for class_id_str, class_data in classes.items():
        class_id = int(class_id_str)
        row: dict[str, Any] = {
            "class_id": class_id,
            "name": class_data["name"],
            "all_images": class_data["image_count_all"],
            "all_pixels": class_data["pixel_count_all"],
            "all_image_share": round(class_data["image_count_all"] / total_images, 6) if total_images > 0 else 0.0,
            "all_pixel_share": round(class_data["pixel_count_all"] / total_pixels, 6) if total_pixels > 0 else 0.0,
            "present_in_splits": ",".join(class_data["present_in_splits"]),
            "missing_in_splits": ",".join(class_data["missing_in_splits"]),
            "recommend_delete": class_id in rec.get("recommend_delete", []),
            "recommend_compress": class_id in rec.get("recommend_compress", []),
        }
        # 按 split 的图片数/像素数
        for split in report["meta"]["split_order"]:
            row[f"{split}_images"] = class_data["image_count_by_split"].get(split, 0)
            row[f"{split}_pixels"] = class_data["pixel_count_by_split"].get(split, 0)
        rows.append(row)
    return rows


def _split_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_data in report["splits"].items():
        rows.append({
            "split": split_name,
            "total_images": split_data["total_images"],
            "class_count": split_data["class_count"],
            "total_labeled_pixels": split_data["total_labeled_pixels"],
            "total_ignore_pixels": split_data["total_ignore_pixels"],
        })
    return rows


# ---------------------------------------------------------------------------
# 输出目录
# ---------------------------------------------------------------------------

def _default_eda_output_dir(source_data_path: Path) -> Path:
    source_cfg = train_tools.load_semantic_segmentation_data_config(source_data_path)
    # 用 train 的 images 目录作为数据集根
    train_images = source_cfg.get("train", {}).get("images", "")
    if train_images:
        source_root = Path(train_images).resolve().parent.parent
    else:
        source_root = source_data_path.resolve().parent
    dataset_tag = rt.dataset_tag_from_dir(source_root)
    base_dir = rt.EDA_OUTPUT_ROOT_DIR / f"{dataset_tag}-semantic-eda"
    return rt.deduplicate_path(base_dir)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def generate_semantic_eda_report(
    *,
    source_data_path: Path,
    output_dir: Path | None,
    overwrite: bool,
    min_class_images: int = 10,
    threshold_percentile: float = 0.9,
) -> Path:
    """生成语义分割 EDA 报告，返回输出目录。"""
    source_data_path = source_data_path.expanduser().resolve()
    final_output_dir = output_dir.expanduser().resolve() if output_dir is not None else _default_eda_output_dir(source_data_path)
    rt.prepare_output_dir(final_output_dir, overwrite=overwrite)

    print(f"[semantic-eda] 扫描数据集: {source_data_path}")
    report, inventory_rows = _collect_semantic_eda(
        source_data_path,
        min_class_images=min_class_images,
        threshold_percentile=threshold_percentile,
    )

    # 生成 markdown
    markdown = _render_markdown(report, final_output_dir)

    # 写文件
    dataset_tag = "semantic"
    json_path = final_output_dir / f"semantic_eda_{dataset_tag}.json"
    markdown_path = final_output_dir / f"semantic_eda_{dataset_tag}.md"
    split_csv_path = final_output_dir / f"split_summary_{dataset_tag}.csv"
    class_csv_path = final_output_dir / f"class_summary_{dataset_tag}.csv"
    inventory_csv_path = final_output_dir / f"image_class_inventory_{dataset_tag}.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown + "\n", encoding="utf-8")

    # split summary CSV
    split_rows = _split_summary_rows(report)
    rt.save_records_csv(
        split_csv_path,
        split_rows,
        ["split", "total_images", "class_count", "total_labeled_pixels", "total_ignore_pixels"],
    )

    # class summary CSV
    class_rows = _class_summary_rows(report)
    class_fieldnames = ["class_id", "name", "all_images", "all_pixels", "all_image_share", "all_pixel_share",
                        "present_in_splits", "missing_in_splits", "recommend_delete", "recommend_compress"]
    for split in report["meta"]["split_order"]:
        class_fieldnames.extend([f"{split}_images", f"{split}_pixels"])
    rt.save_records_csv(class_csv_path, class_rows, class_fieldnames)

    # inventory CSV（curate 消费的关键中间产物）
    rt.save_records_csv(
        inventory_csv_path,
        inventory_rows,
        ["split", "image", "image_path", "mask_path", "width", "height", "class_ids", "per_class_pixels", "total_labeled_pixels"],
    )

    # 控制台输出
    print(f"\n{'=' * 63}")
    print(f"  语义分割 EDA 报告已生成")
    print(f"{'=' * 63}")
    print(f"\n  输出目录: {final_output_dir}")
    print(f"  - {markdown_path.name}")
    print(f"  - {json_path.name}")
    print(f"  - {split_csv_path.name}")
    print(f"  - {class_csv_path.name}")
    print(f"  - {inventory_csv_path.name}")

    # 打印类别表摘要
    classes = report["classes"]
    rec = report["recommendations"]
    print(f"\n{'─' * 63}")
    print(f"  {'#':>3} {'ID':>4} {'名称':<20} {'Train':>6} {'All':>6} {'标记'}")
    print(f"{'─' * 63}")
    for idx, (cid_str, cdata) in enumerate(classes.items()):
        cid = int(cid_str)
        train_n = cdata["image_count_by_split"].get("train", 0)
        all_n = cdata["image_count_all"]
        marks: list[str] = []
        if cid in rec.get("recommend_delete", []):
            marks.append("[删]")
        if cid in rec.get("recommend_compress", []):
            marks.append("[压]")
        mark_str = " ".join(marks)
        print(f"  {idx + 1:>3} {cid:>4} {cdata['name']:<20} {train_n:>6} {all_n:>6} {mark_str}")
    print(f"{'─' * 63}")

    # 推荐
    if rec.get("recommend_delete"):
        names = [f"{cid}({classes[str(cid)]['name']})" for cid in rec["recommend_delete"]]
        print(f"\n  推荐删除: {', '.join(names)}")
    if rec.get("recommend_threshold"):
        print(f"  推荐压缩阈值: {rec['recommend_threshold']}")
    if rec.get("recommend_compress"):
        names = [f"{cid}({classes[str(cid)]['name']})" for cid in rec["recommend_compress"]]
        print(f"  压缩候选: {', '.join(names)}")

    print(f"\n  ── 下一步 ──")
    print(f"  运行 seg curate 进行交互式类别整理:")
    print(f"    python launcher.py seg curate --eda-dir {final_output_dir}")
    print()

    return final_output_dir


def run_semantic_eda(args: argparse.Namespace) -> None:
    """dispatch 入口。"""
    generate_semantic_eda_report(
        source_data_path=Path(args.data),
        output_dir=getattr(args, "output_dir", None),
        overwrite=bool(getattr(args, "overwrite", False)),
        min_class_images=int(getattr(args, "min_class_images", 10)),
        threshold_percentile=float(getattr(args, "threshold_percentile", 0.9)),
    )
