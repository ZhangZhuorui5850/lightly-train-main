"""EDA for YOLO polygon instance-segmentation datasets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

from . import common as rt
from .det_analysis import percentile_int
from .seg_export import (
    _analyze_candidate_polygons,
    build_export_dataset_eda,
    render_export_dataset_eda_markdown,
)
from .seg_shared import (
    SegExportImageCandidate,
    collect_source_image_infos,
    parse_polygon_line,
    polygon_area_normalized,
)


def _to_candidates(
    source_infos_by_split: dict[str, list[Any]],
) -> dict[str, list[SegExportImageCandidate]]:
    return {
        split_name: [
            SegExportImageCandidate(
                split_name=info.split_name,
                rel_split_image_dir=info.rel_split_image_dir,
                rel_split_label_dir=info.rel_split_label_dir,
                rel_path=info.rel_path,
                src_image_path=info.src_image_path,
                src_label_path=info.src_label_path,
                filtered_lines=info.label_lines,
                class_box_counts=info.class_box_counts,
            )
            for info in infos
        ]
        for split_name, infos in source_infos_by_split.items()
    }


def _default_eda_output_dir(source_root: Path) -> Path:
    dataset_tag = rt.dataset_tag_from_dir(source_root)
    return rt.EDA_OUTPUT_ROOT_DIR / f"{dataset_tag}-instance-eda"


def _build_recommendations(
    report: dict[str, Any],
    *,
    min_class_images: int,
    threshold_percentile: float,
) -> dict[str, Any]:
    classes = cast(dict[str, dict[str, Any]], report["classes"])
    recommend_delete = [
        int(class_id)
        for class_id, entry in classes.items()
        if int(entry["all"]["images"]) < min_class_images
    ]
    train_counts = [
        int(entry.get("splits", {}).get("train", {}).get("images", 0))
        for entry in classes.values()
    ]
    threshold = 0
    if any(train_counts):
        raw_threshold = percentile_int(train_counts, threshold_percentile)
        threshold = int(math.ceil(raw_threshold / 10.0) * 10) if raw_threshold > 0 else 0
    recommend_compress = [
        int(class_id)
        for class_id, entry in classes.items()
        if threshold > 0
        and int(entry.get("splits", {}).get("train", {}).get("images", 0)) > threshold
    ]
    return {
        "min_class_images_threshold": min_class_images,
        "threshold_percentile": threshold_percentile,
        "recommend_delete": recommend_delete,
        "recommend_threshold": threshold,
        "recommend_compress": recommend_compress,
    }


def _inventory_rows(
    candidates_by_split: dict[str, list[SegExportImageCandidate]],
    class_id_mapping: dict[int, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        for candidate in candidates_by_split.get(split_name, []):
            analysis = _analyze_candidate_polygons(candidate, class_id_mapping)
            per_class_counts = cast(dict[int, int], analysis["per_class_label_counts"])
            rows.append(
                {
                    "split": split_name,
                    "image": candidate.rel_path.as_posix(),
                    "image_path": str(candidate.src_image_path),
                    "label_path": str(candidate.src_label_path),
                    "width": analysis["image_width"],
                    "height": analysis["image_height"],
                    "class_ids": "|".join(str(value) for value in sorted(per_class_counts)),
                    "per_class_instances": json.dumps(
                        {str(key): value for key, value in sorted(per_class_counts.items())},
                        ensure_ascii=False,
                    ),
                    "total_instances": analysis["total_instances"],
                    "mask_coverage_ratio": round(
                        float(analysis["total_mask_pixels"])
                        / max(float(analysis["image_area_pixels"]), 1.0),
                        6,
                    ),
                    "total_vertices": analysis["total_vertex_count"],
                }
            )
    return rows


def _class_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    recommendations = report["recommendations"]
    delete_ids = set(recommendations["recommend_delete"])
    compress_ids = set(recommendations["recommend_compress"])
    rows: list[dict[str, Any]] = []
    for class_id_text, entry in cast(dict[str, dict[str, Any]], report["classes"]).items():
        class_id = int(class_id_text)
        all_stats = entry["all"]
        row: dict[str, Any] = {
            "class_id": class_id,
            "name": entry["name"],
            "all_images": all_stats["images"],
            "all_instances": all_stats["instances"],
            "tiny_instances": all_stats["size_buckets"]["tiny"],
            "small_instances": all_stats["size_buckets"]["small"],
            "medium_instances": all_stats["size_buckets"]["medium"],
            "large_instances": all_stats["size_buckets"]["large"],
            "recommend_delete": class_id in delete_ids,
            "recommend_compress": class_id in compress_ids,
        }
        for split_name in ("train", "val", "test"):
            split_stats = entry.get("splits", {}).get(split_name, {})
            row[f"{split_name}_images"] = split_stats.get("images", 0)
            row[f"{split_name}_instances"] = split_stats.get("instances", 0)
        rows.append(row)
    return rows


def _annotation_quality(
    candidates_by_split: dict[str, list[SegExportImageCandidate]],
    configured_class_ids: set[int],
) -> dict[str, Any]:
    quality = {
        "missing_label_files": 0,
        "empty_label_files": 0,
        "valid_polygon_lines": 0,
        "invalid_polygon_lines": 0,
        "out_of_range_polygons": 0,
        "zero_area_polygons": 0,
        "unknown_class_ids": [],
    }
    unknown_class_ids: set[int] = set()
    for candidates in candidates_by_split.values():
        for candidate in candidates:
            if not candidate.src_label_path.exists():
                quality["missing_label_files"] += 1
                continue
            raw_lines = [
                line.strip()
                for line in candidate.src_label_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines()
                if line.strip()
            ]
            if not raw_lines:
                quality["empty_label_files"] += 1
            quality["valid_polygon_lines"] += len(candidate.filtered_lines)
            quality["invalid_polygon_lines"] += max(
                len(raw_lines) - len(candidate.filtered_lines), 0
            )
            for line in candidate.filtered_lines:
                parsed = parse_polygon_line(line)
                if parsed is None:
                    continue
                class_id, coords = parsed
                if class_id not in configured_class_ids:
                    unknown_class_ids.add(class_id)
                if any(value < 0.0 or value > 1.0 for value in coords):
                    quality["out_of_range_polygons"] += 1
                if polygon_area_normalized(coords) <= 0.0:
                    quality["zero_area_polygons"] += 1
    quality["unknown_class_ids"] = sorted(unknown_class_ids)
    return quality


def generate_instance_eda_report(
    *,
    source_data_path: Path,
    output_dir: Path | None,
    overwrite: bool,
    min_class_images: int = 10,
    threshold_percentile: float = 0.9,
) -> Path:
    source_data_path = source_data_path.expanduser().resolve()
    source_cfg = rt.load_data_config(source_data_path)
    source_root = Path(source_cfg["_root_dir"])
    final_output_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else (
            _default_eda_output_dir(source_root)
            if overwrite
            else rt.deduplicate_path(_default_eda_output_dir(source_root))
        )
    )
    rt.prepare_output_dir(final_output_dir, overwrite=overwrite)
    print(f"[instance-eda] 扫描数据集: {source_data_path}")

    class_names = rt.normalize_names(source_cfg.get("names"))
    configured_class_ids = set(class_names)
    source_infos_by_split, _ = collect_source_image_infos(source_cfg, source_root)
    if not source_infos_by_split:
        raise ValueError(f"data.yaml 中没有可扫描的实例分割 split: {source_data_path}")

    discovered_class_ids = {
        class_id
        for infos in source_infos_by_split.values()
        for info in infos
        for class_id in info.class_box_counts
    }
    for class_id in sorted(discovered_class_ids):
        class_names.setdefault(class_id, f"class_{class_id}")

    candidates_by_split = _to_candidates(source_infos_by_split)
    annotation_quality = _annotation_quality(candidates_by_split, configured_class_ids)
    class_id_mapping = {class_id: class_id for class_id in class_names}
    report = build_export_dataset_eda(
        selected_candidates_by_split=candidates_by_split,
        kept_names=class_names,
        class_id_mapping=class_id_mapping,
    )
    report["meta"] = {
        "segmentation_type": "instance",
        "annotation_format": "yolo_polygon",
        "source_data": str(source_data_path),
        "source_root": str(source_root),
    }
    report["recommendations"] = _build_recommendations(
        report,
        min_class_images=min_class_images,
        threshold_percentile=threshold_percentile,
    )
    report["annotation_quality"] = annotation_quality
    report["insights"] = [
        str(item).replace("导出数据集", "数据集")
        for item in report.get("insights", [])
    ]

    tag = "instance"
    json_path = final_output_dir / f"instance_eda_{tag}.json"
    markdown_path = final_output_dir / f"instance_eda_{tag}.md"
    split_csv_path = final_output_dir / f"split_summary_{tag}.csv"
    class_csv_path = final_output_dir / f"class_summary_{tag}.csv"
    inventory_csv_path = final_output_dir / f"image_instance_inventory_{tag}.csv"

    markdown = render_export_dataset_eda_markdown(
        export_dataset_eda=report,
        export_root=source_root,
        source_data_path=source_data_path,
        report_path=None,
    )
    markdown = markdown.replace("# 导出数据集 EDA 报告（seg）", "# 实例分割 EDA 报告", 1)
    markdown = markdown.replace("- 导出目录：", "- 数据集目录：", 1)
    markdown = markdown.replace("- 评估报告：dataset-only 模式\n", "", 1)
    quality = report["annotation_quality"]
    markdown += (
        "\n## 7. 标注质量\n\n"
        f"- 缺失标签文件：{quality['missing_label_files']}\n"
        f"- 空标签文件：{quality['empty_label_files']}\n"
        f"- 有效 polygon：{quality['valid_polygon_lines']}\n"
        f"- 无效标签行：{quality['invalid_polygon_lines']}\n"
        f"- 坐标越界 polygon：{quality['out_of_range_polygons']}\n"
        f"- 退化 polygon：{quality['zero_area_polygons']}\n"
        f"- YAML 类别表外 ID：{quality['unknown_class_ids']}\n"
    )
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown + "\n", encoding="utf-8")

    split_rows = [
        {"split": split_name, **summary}
        for split_name, summary in report["split_summaries"].items()
    ]
    rt.save_records_csv(
        split_csv_path,
        split_rows,
        [
            "split",
            "images",
            "labeled_images",
            "empty_images",
            "instances",
            "avg_instances_per_image",
            "avg_instances_per_labeled_image",
            "class_count",
            "mask_coverage_ratio",
            "avg_vertices_per_instance",
            "size_buckets",
            "size_bucket_ratio",
            "labels",
        ],
    )
    class_rows = _class_summary_rows(report)
    class_fields = [
        "class_id",
        "name",
        "all_images",
        "all_instances",
        "tiny_instances",
        "small_instances",
        "medium_instances",
        "large_instances",
        "recommend_delete",
        "recommend_compress",
    ]
    for split_name in ("train", "val", "test"):
        class_fields.extend([f"{split_name}_images", f"{split_name}_instances"])
    rt.save_records_csv(class_csv_path, class_rows, class_fields)
    inventory_rows = _inventory_rows(candidates_by_split, class_id_mapping)
    rt.save_records_csv(
        inventory_csv_path,
        inventory_rows,
        [
            "split",
            "image",
            "image_path",
            "label_path",
            "width",
            "height",
            "class_ids",
            "per_class_instances",
            "total_instances",
            "mask_coverage_ratio",
            "total_vertices",
        ],
    )

    overview = report["overview"]
    print("\n  实例分割 EDA 报告已生成")
    print(f"  输出目录: {final_output_dir}")
    print(
        f"  图片: {overview['images']}，实例: {overview['instances']}，"
        f"类别: {overview['class_count']}"
    )
    for path in (
        markdown_path,
        json_path,
        split_csv_path,
        class_csv_path,
        inventory_csv_path,
    ):
        print(f"  - {path.name}")
    return final_output_dir


__all__ = ["generate_instance_eda_report"]
