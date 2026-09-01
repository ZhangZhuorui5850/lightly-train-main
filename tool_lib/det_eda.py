"""检测数据集 EDA。

面向 YOLO 检测数据集，输出按 split 对照的详细数据分析结果。
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

from . import common as rt
from .det_analysis import percentile_float, percentile_int
from .det_shared import safe_class_name
from .progress import track

SMALL_OBJECT_AREA_THRESHOLD = 32.0 * 32.0
MEDIUM_OBJECT_AREA_THRESHOLD = 96.0 * 96.0
SPLIT_ORDER = ("train", "val", "test")
BAD_IMAGE_EXPORT_PER_SPLIT_LIMIT = 24
BAD_IMAGE_EXPORT_PER_ISSUE_LIMIT = 8
BAD_IMAGE_ISSUE_ORDER = (
    "image_open_error",
    "missing_label_file",
    "invalid_label_lines",
    "empty_label_file",
    "coord_anomaly_lines",
)


@dataclass(frozen=True)
class _LightSourceImageInfo:
    split_name: str
    rel_path: Path
    src_image_path: Path
    src_label_path: Path


def _summarize_bad_image_issues(row: dict[str, Any]) -> tuple[list[str], str, float]:
    issue_tags: list[str] = []
    score = 0.0
    if str(row.get("image_open_error") or "").strip():
        issue_tags.append("image_open_error")
        score += 1000.0
    if not bool(row.get("label_file_exists", False)):
        issue_tags.append("missing_label_file")
        score += 800.0
    if int(row.get("invalid_label_lines", 0)) > 0:
        issue_tags.append("invalid_label_lines")
        score += 200.0 + (int(row.get("invalid_label_lines", 0)) * 100.0)
    if bool(row.get("label_file_exists", False)) and int(row.get("raw_nonempty_label_lines", 0)) == 0:
        issue_tags.append("empty_label_file")
        score += 180.0
    if int(row.get("coord_anomaly_lines", 0)) > 0:
        issue_tags.append("coord_anomaly_lines")
        score += 80.0 + (int(row.get("coord_anomaly_lines", 0)) * 20.0)
    primary_issue = next((name for name in BAD_IMAGE_ISSUE_ORDER if name in issue_tags), "")
    return issue_tags, primary_issue, round(score, 2)


def _sort_key_for_bad_image(row: dict[str, Any]) -> tuple[float, int, int, int, str]:
    return (
        -float(row.get("bad_image_score", 0.0)),
        -int(row.get("invalid_label_lines", 0)),
        -int(row.get("coord_anomaly_lines", 0)),
        -int(row.get("valid_boxes", 0)),
        str(row.get("image", "")),
    )


def _attach_bad_image_flags(row: dict[str, Any]) -> dict[str, Any]:
    issue_tags, primary_issue, bad_image_score = _summarize_bad_image_issues(row)
    row["bad_image_candidate"] = bool(issue_tags)
    row["bad_image_issue_tags"] = ",".join(issue_tags)
    row["bad_image_primary_issue"] = primary_issue
    row["bad_image_score"] = bad_image_score
    row["bad_image_issue_count"] = len(issue_tags)
    return row


def _select_representative_bad_images(per_image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_rows: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    candidate_rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for row in per_image_rows:
        if not bool(row.get("bad_image_candidate", False)):
            continue
        candidate_rows_by_split.setdefault(str(row.get("split", "")), []).append(row)

    for split_name in SPLIT_ORDER:
        split_rows = candidate_rows_by_split.get(split_name, [])
        if not split_rows:
            continue
        split_selected: list[dict[str, Any]] = []
        for issue_name in BAD_IMAGE_ISSUE_ORDER:
            issue_rows = [row for row in split_rows if issue_name in str(row.get("bad_image_issue_tags", "")).split(",")]
            issue_rows.sort(key=_sort_key_for_bad_image)
            for row in issue_rows[:BAD_IMAGE_EXPORT_PER_ISSUE_LIMIT]:
                row_key = (str(row.get("split", "")), str(row.get("image", "")))
                if row_key in selected_keys or len(split_selected) >= BAD_IMAGE_EXPORT_PER_SPLIT_LIMIT:
                    continue
                split_selected.append(row)
                selected_keys.add(row_key)
        if len(split_selected) < BAD_IMAGE_EXPORT_PER_SPLIT_LIMIT:
            remaining_rows = sorted(split_rows, key=_sort_key_for_bad_image)
            for row in remaining_rows:
                row_key = (str(row.get("split", "")), str(row.get("image", "")))
                if row_key in selected_keys:
                    continue
                split_selected.append(row)
                selected_keys.add(row_key)
                if len(split_selected) >= BAD_IMAGE_EXPORT_PER_SPLIT_LIMIT:
                    break
        selected_rows.extend(split_selected)
    return selected_rows


def _build_bad_image_export_summary(
    *,
    bad_images_dir: Path,
    per_image_rows: list[dict[str, Any]],
    exported_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate_rows = [row for row in per_image_rows if bool(row.get("bad_image_candidate", False))]
    summary: dict[str, Any] = {
        "directory": str(bad_images_dir),
        "per_split_limit": BAD_IMAGE_EXPORT_PER_SPLIT_LIMIT,
        "per_issue_limit": BAD_IMAGE_EXPORT_PER_ISSUE_LIMIT,
        "issue_order": list(BAD_IMAGE_ISSUE_ORDER),
        "candidate_count": len(candidate_rows),
        "exported_count": len(exported_rows),
        "splits": {},
    }
    for split_name in sorted({str(row.get("split", "")) for row in candidate_rows + exported_rows}):
        split_candidate_rows = [row for row in candidate_rows if str(row.get("split", "")) == split_name]
        split_exported_rows = [row for row in exported_rows if str(row.get("split", "")) == split_name]
        issue_counts = {
            issue_name: sum(
                1 for row in split_candidate_rows if issue_name in str(row.get("bad_image_issue_tags", "")).split(",")
            )
            for issue_name in BAD_IMAGE_ISSUE_ORDER
        }
        exported_issue_counts = {
            issue_name: sum(
                1 for row in split_exported_rows if issue_name in str(row.get("bad_image_issue_tags", "")).split(",")
            )
            for issue_name in BAD_IMAGE_ISSUE_ORDER
        }
        summary["splits"][split_name] = {
            "candidate_count": len(split_candidate_rows),
            "exported_count": len(split_exported_rows),
            "issue_counts": issue_counts,
            "exported_issue_counts": exported_issue_counts,
        }
    return summary


def _export_bad_image_bundle(
    *,
    output_dir: Path,
    per_image_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    bad_images_dir = output_dir / "bad_images"
    selected_rows = _select_representative_bad_images(per_image_rows)
    exported_rows: list[dict[str, Any]] = []

    for row in selected_rows:
        split_name = str(row.get("split", "unknown"))
        image_rel_path = Path(str(row.get("image", "")))
        image_export_path = bad_images_dir / split_name / "images" / image_rel_path
        label_export_path = bad_images_dir / split_name / "labels" / image_rel_path.with_suffix(".txt")
        meta_export_path = bad_images_dir / split_name / "meta" / image_rel_path.with_suffix(".json")

        copied_image_path = rt.copy_file_if_exists(Path(str(row.get("image_path", ""))), image_export_path)
        copied_label_path = rt.copy_file_if_exists(Path(str(row.get("label_path", ""))), label_export_path)
        export_row = dict(row)
        export_row["export_image_path"] = str(copied_image_path) if copied_image_path is not None else ""
        export_row["export_label_path"] = str(copied_label_path) if copied_label_path is not None else ""
        export_row["export_meta_path"] = str(meta_export_path)

        meta_export_path.parent.mkdir(parents=True, exist_ok=True)
        meta_export_path.write_text(json.dumps(export_row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        exported_rows.append(export_row)

    manifest_csv_path = bad_images_dir / "manifest.csv"
    manifest_json_path = bad_images_dir / "manifest.json"
    summary = _build_bad_image_export_summary(
        bad_images_dir=bad_images_dir,
        per_image_rows=per_image_rows,
        exported_rows=exported_rows,
    )
    summary["manifest_csv"] = str(manifest_csv_path)
    summary["manifest_json"] = str(manifest_json_path)

    manifest_rows = [
        {
            "split": row.get("split", ""),
            "image": row.get("image", ""),
            "image_path": row.get("image_path", ""),
            "label_path": row.get("label_path", ""),
            "export_image_path": row.get("export_image_path", ""),
            "export_label_path": row.get("export_label_path", ""),
            "export_meta_path": row.get("export_meta_path", ""),
            "label_file_exists": row.get("label_file_exists", False),
            "raw_nonempty_label_lines": row.get("raw_nonempty_label_lines", 0),
            "valid_boxes": row.get("valid_boxes", 0),
            "invalid_label_lines": row.get("invalid_label_lines", 0),
            "coord_anomaly_lines": row.get("coord_anomaly_lines", 0),
            "bad_image_candidate": row.get("bad_image_candidate", False),
            "bad_image_issue_tags": row.get("bad_image_issue_tags", ""),
            "bad_image_primary_issue": row.get("bad_image_primary_issue", ""),
            "bad_image_score": row.get("bad_image_score", 0.0),
            "image_open_error": row.get("image_open_error", ""),
        }
        for row in exported_rows
    ]

    rt.save_records_csv(
        manifest_csv_path,
        manifest_rows,
        [
            "split",
            "image",
            "image_path",
            "label_path",
            "export_image_path",
            "export_label_path",
            "export_meta_path",
            "label_file_exists",
            "raw_nonempty_label_lines",
            "valid_boxes",
            "invalid_label_lines",
            "coord_anomaly_lines",
            "bad_image_candidate",
            "bad_image_issue_tags",
            "bad_image_primary_issue",
            "bad_image_score",
            "image_open_error",
        ],
    )
    manifest_json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "samples": exported_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _strip_yaml_comment(text: str) -> str:
    in_single_quote = False
    in_double_quote = False
    escaped = False
    result: list[str] = []
    for char in text:
        if char == "\\" and in_double_quote and not escaped:
            escaped = True
            result.append(char)
            continue
        if char == "'" and not in_double_quote and not escaped:
            in_single_quote = not in_single_quote
        elif char == '"' and not in_single_quote and not escaped:
            in_double_quote = not in_double_quote
        elif char == "#" and not in_single_quote and not in_double_quote:
            break
        result.append(char)
        escaped = False
    return "".join(result).rstrip()


def _parse_yaml_scalar(text: str) -> Any:
    stripped = text.strip()
    if stripped == "":
        return ""
    lowered = stripped.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return stripped[1:-1]
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    current_container: dict[str, Any] | list[Any] | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0:
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current_key = key
                current_container = {}
                data[key] = current_container
            else:
                data[key] = _parse_yaml_scalar(value)
                current_key = None
                current_container = None
            continue

        if current_key is None or current_container is None:
            continue
        if stripped.startswith("- "):
            if not isinstance(current_container, list):
                current_container = []
                data[current_key] = current_container
            current_container.append(_parse_yaml_scalar(stripped[2:].strip()))
            continue
        if ":" in stripped:
            if not isinstance(current_container, dict):
                current_container = {}
                data[current_key] = current_container
            child_key, child_value = stripped.split(":", 1)
            current_container[child_key.strip()] = _parse_yaml_scalar(child_value.strip())
    return data


def _load_data_config_light(data_path: Path) -> dict[str, Any]:
    resolved_path = data_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Data config does not exist: {resolved_path}")
    raw_text = resolved_path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(raw_text)
    else:
        payload = _parse_simple_yaml(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid data config: {resolved_path}")

    base_dir = resolved_path.parent
    payload["_data_yaml_path"] = resolved_path
    payload["_base_dir"] = base_dir
    payload["_root_dir"] = rt.dataset_adapter.resolve_dataset_root(
        resolved_path, payload
    )
    return payload


def _collect_source_image_infos_light(
    source_cfg: dict[str, Any],
) -> tuple[dict[str, list[_LightSourceImageInfo]], dict[str, str]]:
    source_infos_by_split: dict[str, list[_LightSourceImageInfo]] = {}
    export_split_paths: dict[str, str] = {}
    source_root = Path(source_cfg["_root_dir"]).resolve()
    for split_name in SPLIT_ORDER:
        split_value = source_cfg.get(split_name)
        if not split_value:
            continue
        try:
            samples, _ = rt.list_dataset_samples(source_cfg, split_name)
        except ValueError:
            continue
        source_infos_by_split[split_name] = [
            _LightSourceImageInfo(
                split_name=split_name,
                rel_path=sample.relative_path,
                src_image_path=sample.image_path,
                src_label_path=sample.label_path
                or (source_root / ".missing_labels" / split_name / sample.relative_path).with_suffix(".txt"),
            )
            for sample in samples
        ]
        complex_source = isinstance(split_value, (list, dict)) or Path(
            str(split_value)
        ).suffix.casefold() in ({".txt"} | rt.VISUALIZATION_SUFFIXES)
        if complex_source:
            export_split_paths[split_name] = (Path("images") / split_name).as_posix()
        else:
            split_image_dir, _, _ = rt.resolve_dataset_split_paths(source_cfg, split_name)
            try:
                export_split_paths[split_name] = split_image_dir.resolve().relative_to(source_root).as_posix()
            except ValueError:
                export_split_paths[split_name] = (Path("images") / split_name).as_posix()
    return source_infos_by_split, export_split_paths


def _empty_size_buckets() -> dict[str, int]:
    return {"small": 0, "medium": 0, "large": 0}


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if float(denominator) <= 0.0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _round_or_zero(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value), digits)


def _bucket_box_area(area_pixels: float) -> str:
    if area_pixels < SMALL_OBJECT_AREA_THRESHOLD:
        return "small"
    if area_pixels < MEDIUM_OBJECT_AREA_THRESHOLD:
        return "medium"
    return "large"


def _distribution(values: list[int]) -> list[float]:
    total = sum(max(int(value), 0) for value in values)
    if total <= 0:
        return []
    return [max(int(value), 0) / total for value in values]


def _gini(values: list[int]) -> float:
    positives = sorted(max(int(value), 0) for value in values)
    n = len(positives)
    total = sum(positives)
    if n == 0 or total <= 0:
        return 0.0
    weighted_sum = sum((index + 1) * value for index, value in enumerate(positives))
    gini = ((2 * weighted_sum) / (n * total)) - ((n + 1) / n)
    return round(max(gini, 0.0), 6)


def _entropy_evenness(values: list[int]) -> float:
    distribution = _distribution(values)
    if len(distribution) <= 1:
        return 1.0 if distribution else 0.0
    entropy = 0.0
    for prob in distribution:
        if prob > 0.0:
            entropy -= prob * math.log(prob)
    max_entropy = math.log(len(distribution))
    if max_entropy <= 0.0:
        return 0.0
    return round(entropy / max_entropy, 6)


def _js_divergence(p: list[float], q: list[float]) -> float:
    if not p or not q or len(p) != len(q):
        return 0.0
    midpoint = [(left + right) / 2.0 for left, right in zip(p, q)]

    def _kl_divergence(a: list[float], b: list[float]) -> float:
        value = 0.0
        for prob_a, prob_b in zip(a, b):
            if prob_a <= 0.0 or prob_b <= 0.0:
                continue
            value += prob_a * math.log2(prob_a / prob_b)
        return value

    return round((_kl_divergence(p, midpoint) + _kl_divergence(q, midpoint)) / 2.0, 6)


def _distribution_metrics(counts: list[int]) -> dict[str, Any]:
    positive_counts = [int(value) for value in counts if int(value) > 0]
    total = sum(int(value) for value in counts)
    ordered_desc = sorted((int(value) for value in counts), reverse=True)
    top1 = ordered_desc[0] if ordered_desc else 0
    top3 = sum(ordered_desc[:3]) if ordered_desc else 0
    mean_value = (total / len(counts)) if counts else 0.0
    variance = (
        sum((int(value) - mean_value) ** 2 for value in counts) / len(counts)
        if counts
        else 0.0
    )
    std_value = math.sqrt(variance)
    return {
        "total": total,
        "class_count": len(counts),
        "active_class_count": len(positive_counts),
        "min": min(positive_counts, default=0),
        "median": percentile_int(positive_counts, 0.5),
        "p75": percentile_int(positive_counts, 0.75),
        "p90": percentile_int(positive_counts, 0.9),
        "max": max(positive_counts, default=0),
        "max_min_ratio": round(max(positive_counts) / min(positive_counts), 4)
        if len(positive_counts) >= 2 and min(positive_counts) > 0
        else 0.0,
        "max_median_ratio": round(max(positive_counts) / max(percentile_int(positive_counts, 0.5), 1), 4)
        if positive_counts
        else 0.0,
        "p90_p50_ratio": round(percentile_int(positive_counts, 0.9) / max(percentile_int(positive_counts, 0.5), 1), 4)
        if positive_counts
        else 0.0,
        "top1_share": _safe_ratio(top1, total),
        "top3_share": _safe_ratio(top3, total),
        "coefficient_of_variation": _round_or_zero(std_value / mean_value, 6) if mean_value > 0 else 0.0,
        "gini": _gini(counts),
        "entropy_evenness": _entropy_evenness(counts),
    }


def _parse_label_file(label_path: Path) -> dict[str, Any]:
    payload = {
        "exists": label_path.exists(),
        "raw_nonempty_lines": 0,
        "valid_label_lines": 0,
        "invalid_label_lines": 0,
        "coord_anomaly_lines": 0,
        "boxes": [],
    }
    if not label_path.exists():
        return payload

    for raw_line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload["raw_nonempty_lines"] += 1
        parts = line.split()
        if len(parts) != 5:
            payload["invalid_label_lines"] += 1
            continue
        try:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
        except ValueError:
            payload["invalid_label_lines"] += 1
            continue

        payload["valid_label_lines"] += 1
        x1 = x_center - (box_width / 2.0)
        y1 = y_center - (box_height / 2.0)
        x2 = x_center + (box_width / 2.0)
        y2 = y_center + (box_height / 2.0)
        has_coord_anomaly = (
            x_center < 0.0
            or x_center > 1.0
            or y_center < 0.0
            or y_center > 1.0
            or box_width <= 0.0
            or box_width > 1.0
            or box_height <= 0.0
            or box_height > 1.0
            or x1 < 0.0
            or y1 < 0.0
            or x2 > 1.0
            or y2 > 1.0
        )
        if has_coord_anomaly:
            payload["coord_anomaly_lines"] += 1
        payload["boxes"].append(
            {
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": box_width,
                "height": box_height,
                "area_ratio": max(box_width * box_height, 0.0),
                "aspect_ratio": (box_width / box_height) if box_height > 0.0 else 0.0,
                "coord_anomaly": has_coord_anomaly,
            }
        )
    return payload


def _open_image_size(image_path: Path) -> tuple[int, int]:
    if rt.Image is None and not rt.ensure_plot_dependencies():
        raise ModuleNotFoundError("Missing runtime dependency Pillow. Please install pillow in the current environment.")
    image_module = rt.Image
    if image_module is None:
        raise ModuleNotFoundError("Missing runtime dependency Pillow. Please install pillow in the current environment.")
    with image_module.open(image_path) as image:
        width, height = image.size
    return int(width), int(height)


def _build_empty_split_entry(split_name: str) -> dict[str, Any]:
    return {
        "split": split_name,
        "images": 0,
        "labeled_images": 0,
        "empty_images": 0,
        "missing_label_files": 0,
        "empty_label_files": 0,
        "invalid_label_lines": 0,
        "coord_anomaly_lines": 0,
        "image_open_failures": 0,
        "boxes": 0,
        "size_buckets": _empty_size_buckets(),
        "classes": {},
        "image_width_values": [],
        "image_height_values": [],
        "image_area_values": [],
        "image_aspect_ratio_values": [],
        "boxes_per_image_values": [],
        "box_area_ratio_values": [],
        "box_aspect_ratio_values": [],
        "top_dense_images": [],
    }


def _build_class_split_stub(class_id: int, name: str) -> dict[str, Any]:
    return {
        "class_id": class_id,
        "name": name,
        "images": 0,
        "boxes": 0,
        "size_buckets": _empty_size_buckets(),
        "box_area_ratio_values": [],
        "box_aspect_ratio_values": [],
    }


def _push_top_dense_image(split_entry: dict[str, Any], record: dict[str, Any]) -> None:
    dense_images = split_entry["top_dense_images"]
    dense_images.append(record)
    dense_images.sort(
        key=lambda item: (
            -int(item["boxes"]),
            -int(item["class_count"]),
            item["image"],
        )
    )
    del dense_images[10:]


def _finalize_class_entry(class_entry: dict[str, Any], total_images: int, total_boxes: int) -> dict[str, Any]:
    image_values = class_entry.pop("box_area_ratio_values")
    aspect_values = class_entry.pop("box_aspect_ratio_values")
    class_entry["image_share"] = _safe_ratio(class_entry["images"], total_images)
    class_entry["box_share"] = _safe_ratio(class_entry["boxes"], total_boxes)
    class_entry["avg_boxes_per_image"] = _round_or_zero(class_entry["boxes"] / class_entry["images"], 4) if class_entry["images"] > 0 else 0.0
    class_entry["avg_box_area_ratio"] = _round_or_zero(sum(image_values) / len(image_values), 6) if image_values else 0.0
    class_entry["median_box_area_ratio"] = _round_or_zero(percentile_float(image_values, 0.5), 6) if image_values else 0.0
    class_entry["avg_box_aspect_ratio"] = _round_or_zero(sum(aspect_values) / len(aspect_values), 4) if aspect_values else 0.0
    class_entry["median_box_aspect_ratio"] = _round_or_zero(percentile_float(aspect_values, 0.5), 4) if aspect_values else 0.0
    class_entry["size_bucket_ratio"] = {
        bucket_name: _safe_ratio(bucket_value, class_entry["boxes"])
        for bucket_name, bucket_value in class_entry["size_buckets"].items()
    }
    return class_entry


def _finalize_split_entry(split_entry: dict[str, Any]) -> dict[str, Any]:
    total_images = int(split_entry["images"])
    total_boxes = int(split_entry["boxes"])
    labeled_images = int(split_entry["labeled_images"])

    image_width_values = split_entry.pop("image_width_values")
    image_height_values = split_entry.pop("image_height_values")
    image_area_values = split_entry.pop("image_area_values")
    image_aspect_ratio_values = split_entry.pop("image_aspect_ratio_values")
    boxes_per_image_values = split_entry.pop("boxes_per_image_values")
    box_area_ratio_values = split_entry.pop("box_area_ratio_values")
    box_aspect_ratio_values = split_entry.pop("box_aspect_ratio_values")

    split_entry["empty_image_ratio"] = _safe_ratio(split_entry["empty_images"], total_images)
    split_entry["labeled_image_ratio"] = _safe_ratio(labeled_images, total_images)
    split_entry["missing_label_file_ratio"] = _safe_ratio(split_entry["missing_label_files"], total_images)
    split_entry["empty_label_file_ratio"] = _safe_ratio(split_entry["empty_label_files"], total_images)
    split_entry["avg_boxes_per_image"] = _round_or_zero(total_boxes / total_images, 4) if total_images > 0 else 0.0
    split_entry["avg_boxes_per_labeled_image"] = _round_or_zero(total_boxes / labeled_images, 4) if labeled_images > 0 else 0.0
    split_entry["avg_invalid_label_lines_per_image"] = _round_or_zero(split_entry["invalid_label_lines"] / total_images, 4) if total_images > 0 else 0.0
    split_entry["avg_coord_anomaly_lines_per_image"] = _round_or_zero(split_entry["coord_anomaly_lines"] / total_images, 4) if total_images > 0 else 0.0
    split_entry["size_bucket_ratio"] = {
        bucket_name: _safe_ratio(bucket_value, total_boxes)
        for bucket_name, bucket_value in split_entry["size_buckets"].items()
    }
    split_entry["resolution_stats"] = {
        "width": {
            "min": min(image_width_values, default=0),
            "median": percentile_int(image_width_values, 0.5),
            "p90": percentile_int(image_width_values, 0.9),
            "max": max(image_width_values, default=0),
        },
        "height": {
            "min": min(image_height_values, default=0),
            "median": percentile_int(image_height_values, 0.5),
            "p90": percentile_int(image_height_values, 0.9),
            "max": max(image_height_values, default=0),
        },
        "area_pixels": {
            "min": min(image_area_values, default=0),
            "median": percentile_int(image_area_values, 0.5),
            "p90": percentile_int(image_area_values, 0.9),
            "max": max(image_area_values, default=0),
        },
        "aspect_ratio": {
            "min": _round_or_zero(min(image_aspect_ratio_values, default=0.0), 4),
            "median": _round_or_zero(percentile_float(image_aspect_ratio_values, 0.5), 4),
            "p90": _round_or_zero(percentile_float(image_aspect_ratio_values, 0.9), 4),
            "max": _round_or_zero(max(image_aspect_ratio_values, default=0.0), 4),
        },
    }
    split_entry["density_stats"] = {
        "boxes_per_image": {
            "min": min(boxes_per_image_values, default=0),
            "median": percentile_int(boxes_per_image_values, 0.5),
            "p90": percentile_int(boxes_per_image_values, 0.9),
            "max": max(boxes_per_image_values, default=0),
        },
        "box_area_ratio": {
            "min": _round_or_zero(min(box_area_ratio_values, default=0.0), 6),
            "median": _round_or_zero(percentile_float(box_area_ratio_values, 0.5), 6),
            "p90": _round_or_zero(percentile_float(box_area_ratio_values, 0.9), 6),
            "max": _round_or_zero(max(box_area_ratio_values, default=0.0), 6),
        },
        "box_aspect_ratio": {
            "min": _round_or_zero(min(box_aspect_ratio_values, default=0.0), 4),
            "median": _round_or_zero(percentile_float(box_aspect_ratio_values, 0.5), 4),
            "p90": _round_or_zero(percentile_float(box_aspect_ratio_values, 0.9), 4),
            "max": _round_or_zero(max(box_aspect_ratio_values, default=0.0), 4),
        },
    }

    class_entries = list(split_entry["classes"].values())
    class_entries.sort(key=lambda item: (-item["boxes"], -item["images"], item["class_id"]))
    split_entry["class_count"] = len(class_entries)
    split_entry["imbalance_images"] = _distribution_metrics([entry["images"] for entry in class_entries])
    split_entry["imbalance_boxes"] = _distribution_metrics([entry["boxes"] for entry in class_entries])
    split_entry["top_classes_by_boxes"] = class_entries[:10]
    split_entry["top_classes_by_images"] = sorted(
        class_entries,
        key=lambda item: (-item["images"], -item["boxes"], item["class_id"]),
    )[:10]
    split_entry["tail_classes_by_boxes"] = sorted(
        class_entries,
        key=lambda item: (item["boxes"], item["images"], item["class_id"]),
    )[:10]
    split_entry["classes"] = {
        str(entry["class_id"]): _finalize_class_entry(entry, total_images, total_boxes)
        for entry in class_entries
    }
    split_entry["top_dense_images"] = sorted(
        split_entry["top_dense_images"],
        key=lambda item: (-item["boxes"], -item["class_count"], item["image"]),
    )
    return split_entry


def _collect_dataset_eda(source_data_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_cfg = _load_data_config_light(source_data_path)
    source_root = Path(source_cfg["_root_dir"]).resolve()
    source_infos_by_split, export_split_paths = _collect_source_image_infos_light(source_cfg)
    class_names = rt.normalize_names(source_cfg.get("names"))

    split_entries = {split_name: _build_empty_split_entry(split_name) for split_name in SPLIT_ORDER}
    all_entry = _build_empty_split_entry("all")
    per_image_rows: list[dict[str, Any]] = []
    all_class_entries: dict[int, dict[str, Any]] = {}

    for split_name in SPLIT_ORDER:
        infos = source_infos_by_split.get(split_name, [])
        split_entry = split_entries[split_name]
        for info in track(infos, label=f"det/eda 扫描 {split_name}", total=len(infos), unit="img"):
            split_entry["images"] += 1
            all_entry["images"] += 1

            image_width = 0
            image_height = 0
            image_open_error = ""
            try:
                image_width, image_height = _open_image_size(info.src_image_path)
            except Exception as exc:
                split_entry["image_open_failures"] += 1
                all_entry["image_open_failures"] += 1
                image_open_error = str(exc)

            if image_width > 0 and image_height > 0:
                image_area = image_width * image_height
                image_aspect_ratio = image_width / image_height if image_height > 0 else 0.0
                for target_entry in (split_entry, all_entry):
                    target_entry["image_width_values"].append(image_width)
                    target_entry["image_height_values"].append(image_height)
                    target_entry["image_area_values"].append(image_area)
                    target_entry["image_aspect_ratio_values"].append(image_aspect_ratio)
            else:
                image_area = 0
                image_aspect_ratio = 0.0

            label_payload = _parse_label_file(info.src_label_path)
            valid_boxes = list(label_payload["boxes"])
            class_box_counts: Counter[int] = Counter()
            class_size_buckets: dict[int, dict[str, int]] = {}

            for box in valid_boxes:
                class_id = int(box["class_id"])
                class_box_counts[class_id] += 1
                size_bucket = "small"
                if image_width > 0 and image_height > 0:
                    area_pixels = float(box["width"]) * image_width * float(box["height"]) * image_height
                    size_bucket = _bucket_box_area(area_pixels)
                if class_id not in class_size_buckets:
                    class_size_buckets[class_id] = _empty_size_buckets()
                class_size_buckets[class_id][size_bucket] += 1
                for target_entry in (split_entry, all_entry):
                    target_entry["size_buckets"][size_bucket] += 1
                    target_entry["box_area_ratio_values"].append(float(box["area_ratio"]))
                    target_entry["box_aspect_ratio_values"].append(float(box["aspect_ratio"]))

            valid_box_count = len(valid_boxes)
            if valid_box_count > 0:
                split_entry["labeled_images"] += 1
                all_entry["labeled_images"] += 1
            else:
                split_entry["empty_images"] += 1
                all_entry["empty_images"] += 1

            if not label_payload["exists"]:
                split_entry["missing_label_files"] += 1
                all_entry["missing_label_files"] += 1
            elif int(label_payload["raw_nonempty_lines"]) == 0:
                split_entry["empty_label_files"] += 1
                all_entry["empty_label_files"] += 1

            split_entry["invalid_label_lines"] += int(label_payload["invalid_label_lines"])
            all_entry["invalid_label_lines"] += int(label_payload["invalid_label_lines"])
            split_entry["coord_anomaly_lines"] += int(label_payload["coord_anomaly_lines"])
            all_entry["coord_anomaly_lines"] += int(label_payload["coord_anomaly_lines"])
            split_entry["boxes"] += valid_box_count
            all_entry["boxes"] += valid_box_count
            split_entry["boxes_per_image_values"].append(valid_box_count)
            all_entry["boxes_per_image_values"].append(valid_box_count)

            if valid_box_count > 0:
                _push_top_dense_image(
                    split_entry,
                    {
                        "image": info.rel_path.as_posix(),
                        "boxes": valid_box_count,
                        "class_count": len(class_box_counts),
                        "image_width": image_width,
                        "image_height": image_height,
                    },
                )

            for class_id, box_count in class_box_counts.items():
                class_name = safe_class_name(class_names, class_id)
                split_class_entry = split_entry["classes"].setdefault(
                    class_id,
                    _build_class_split_stub(class_id, class_name),
                )
                all_class_entry = all_entry["classes"].setdefault(
                    class_id,
                    _build_class_split_stub(class_id, class_name),
                )
                dataset_class_entry = all_class_entries.setdefault(
                    class_id,
                    {
                        "class_id": class_id,
                        "name": class_name,
                        "all": _build_class_split_stub(class_id, class_name),
                        "splits": {},
                    },
                )
                dataset_split_entry = dataset_class_entry["splits"].setdefault(
                    split_name,
                    _build_class_split_stub(class_id, class_name),
                )

                for target_entry in (
                    split_class_entry,
                    all_class_entry,
                    dataset_class_entry["all"],
                    dataset_split_entry,
                ):
                    target_entry["images"] += 1
                    target_entry["boxes"] += int(box_count)
                    for bucket_name, bucket_value in class_size_buckets.get(class_id, {}).items():
                        target_entry["size_buckets"][bucket_name] += int(bucket_value)
                    target_entry["box_area_ratio_values"].extend(
                        float(box["area_ratio"]) for box in valid_boxes if int(box["class_id"]) == class_id
                    )
                    target_entry["box_aspect_ratio_values"].extend(
                        float(box["aspect_ratio"]) for box in valid_boxes if int(box["class_id"]) == class_id
                    )

            per_image_rows.append(
                _attach_bad_image_flags(
                    {
                        "split": split_name,
                        "image": info.rel_path.as_posix(),
                        "image_path": str(info.src_image_path),
                        "label_path": str(info.src_label_path),
                        "image_width": image_width,
                        "image_height": image_height,
                        "image_area_pixels": image_area,
                        "image_aspect_ratio": _round_or_zero(image_aspect_ratio, 4),
                        "label_file_exists": bool(label_payload["exists"]),
                        "raw_nonempty_label_lines": int(label_payload["raw_nonempty_lines"]),
                        "valid_boxes": valid_box_count,
                        "invalid_label_lines": int(label_payload["invalid_label_lines"]),
                        "coord_anomaly_lines": int(label_payload["coord_anomaly_lines"]),
                        "labeled": valid_box_count > 0,
                        "empty_image": valid_box_count == 0,
                        "class_count": len(class_box_counts),
                        "dominant_class": safe_class_name(class_names, class_box_counts.most_common(1)[0][0]) if class_box_counts else "",
                        "dominant_class_boxes": int(class_box_counts.most_common(1)[0][1]) if class_box_counts else 0,
                        "small_boxes": int(sum(1 for box in valid_boxes if image_width > 0 and image_height > 0 and _bucket_box_area(float(box["width"]) * image_width * float(box["height"]) * image_height) == "small")),
                        "medium_boxes": int(sum(1 for box in valid_boxes if image_width > 0 and image_height > 0 and _bucket_box_area(float(box["width"]) * image_width * float(box["height"]) * image_height) == "medium")),
                        "large_boxes": int(sum(1 for box in valid_boxes if image_width > 0 and image_height > 0 and _bucket_box_area(float(box["width"]) * image_width * float(box["height"]) * image_height) == "large")),
                        "mean_box_area_ratio": _round_or_zero(
                            sum(float(box["area_ratio"]) for box in valid_boxes) / valid_box_count,
                            6,
                        ) if valid_box_count > 0 else 0.0,
                        "mean_box_aspect_ratio": _round_or_zero(
                            sum(float(box["aspect_ratio"]) for box in valid_boxes) / valid_box_count,
                            4,
                        ) if valid_box_count > 0 else 0.0,
                        "image_open_error": image_open_error,
                    }
                )
            )

    finalized_splits = {
        split_name: _finalize_split_entry(split_entries[split_name])
        for split_name in SPLIT_ORDER
        if split_entries[split_name]["images"] > 0
    }
    finalized_all = _finalize_split_entry(all_entry)
    finalized_splits["all"] = finalized_all

    class_summary = {}
    total_images = finalized_all["images"]
    total_boxes = finalized_all["boxes"]
    overall_image_distribution = _distribution([finalized_splits.get(split_name, {}).get("images", 0) for split_name in SPLIT_ORDER])
    overall_box_distribution = _distribution([finalized_splits.get(split_name, {}).get("boxes", 0) for split_name in SPLIT_ORDER])

    for class_id in sorted(all_class_entries):
        class_name = safe_class_name(class_names, class_id)
        class_record = all_class_entries[class_id]
        split_presence = [split_name for split_name in SPLIT_ORDER if split_name in class_record["splits"]]
        image_distribution = _distribution([class_record["splits"].get(split_name, {}).get("images", 0) for split_name in SPLIT_ORDER])
        box_distribution = _distribution([class_record["splits"].get(split_name, {}).get("boxes", 0) for split_name in SPLIT_ORDER])
        class_summary[str(class_id)] = {
            "class_id": class_id,
            "name": class_name,
            "all": _finalize_class_entry(class_record["all"], total_images, total_boxes),
            "splits": {
                split_name: _finalize_class_entry(
                    class_record["splits"].get(split_name, _build_class_split_stub(class_id, class_name)),
                    finalized_splits.get(split_name, {}).get("images", 0),
                    finalized_splits.get(split_name, {}).get("boxes", 0),
                )
                for split_name in SPLIT_ORDER
                if split_name in finalized_splits
            },
            "present_in_splits": split_presence,
            "missing_in_splits": [split_name for split_name in SPLIT_ORDER if split_name in finalized_splits and split_name not in split_presence],
            "coverage_split_count": len(split_presence),
            "image_split_distribution": {
                split_name: _safe_ratio(class_record["splits"].get(split_name, {}).get("images", 0), class_record["all"]["images"])
                for split_name in SPLIT_ORDER
                if split_name in finalized_splits
            },
            "box_split_distribution": {
                split_name: _safe_ratio(class_record["splits"].get(split_name, {}).get("boxes", 0), class_record["all"]["boxes"])
                for split_name in SPLIT_ORDER
                if split_name in finalized_splits
            },
            "image_split_js_divergence": _js_divergence(image_distribution, overall_image_distribution),
            "box_split_js_divergence": _js_divergence(box_distribution, overall_box_distribution),
        }

    class_entries_sorted = sorted(
        class_summary.values(),
        key=lambda item: (-item["all"]["boxes"], -item["all"]["images"], item["class_id"]),
    )

    imbalance_analysis = {
        scope: {
            "images": finalized_splits[scope]["imbalance_images"],
            "boxes": finalized_splits[scope]["imbalance_boxes"],
        }
        for scope in finalized_splits
    }

    class_drift_ranking = sorted(
        (
            {
                "class_id": item["class_id"],
                "name": item["name"],
                "image_split_js_divergence": item["image_split_js_divergence"],
                "box_split_js_divergence": item["box_split_js_divergence"],
                "missing_in_splits": item["missing_in_splits"],
            }
            for item in class_entries_sorted
        ),
        key=lambda item: (-item["box_split_js_divergence"], -item["image_split_js_divergence"], item["class_id"]),
    )

    cross_split_analysis = {
        "overall_split_distribution": {
            "images": {
                split_name: _safe_ratio(finalized_splits.get(split_name, {}).get("images", 0), total_images)
                for split_name in SPLIT_ORDER
                if split_name in finalized_splits
            },
            "boxes": {
                split_name: _safe_ratio(finalized_splits.get(split_name, {}).get("boxes", 0), total_boxes)
                for split_name in SPLIT_ORDER
                if split_name in finalized_splits
            },
        },
        "classes_missing_by_split": {
            split_name: [
                item["name"]
                for item in class_entries_sorted
                if split_name in item["missing_in_splits"]
            ]
            for split_name in SPLIT_ORDER
            if split_name in finalized_splits
        },
        "top_class_drift_by_boxes": class_drift_ranking[:10],
        "top_class_drift_by_images": sorted(
            class_drift_ranking,
            key=lambda item: (-item["image_split_js_divergence"], -item["box_split_js_divergence"], item["class_id"]),
        )[:10],
    }

    insights: list[str] = []
    all_box_imbalance = imbalance_analysis["all"]["boxes"]
    all_image_imbalance = imbalance_analysis["all"]["images"]
    insights.append(
        f"全量共有 {total_images} 张图、{total_boxes} 个框、{len(class_summary)} 个有效类别。"
    )
    insights.append(
        f"类别框分布的最大最小比为 {all_box_imbalance['max_min_ratio']}，Gini 为 {all_box_imbalance['gini']}，熵均衡度为 {all_box_imbalance['entropy_evenness']}。"
    )
    if all_box_imbalance["max_min_ratio"] >= 5.0 or all_box_imbalance["gini"] >= 0.4:
        insights.append("类别框分布已经进入明显长尾区间，训练与评估都需要关注少样本类。")
    missing_by_split = cross_split_analysis["classes_missing_by_split"]
    for split_name in SPLIT_ORDER:
        missing_names = missing_by_split.get(split_name, [])
        if missing_names:
            insights.append(f"{split_name} 缺少 {len(missing_names)} 个类别，split 间类别覆盖存在偏移。")
    all_size_ratio = finalized_all["size_bucket_ratio"]
    if all_size_ratio["small"] >= 0.5:
        insights.append("小目标占比超过一半，检测阈值、输入分辨率和标注质量会直接影响效果。")
    if finalized_all["empty_image_ratio"] >= 0.2:
        insights.append("空标注图占比较高，建议结合业务目标确认负样本比例。")
    if finalized_all["coord_anomaly_lines"] > 0:
        insights.append(f"检测到 {finalized_all['coord_anomaly_lines']} 条坐标异常标注，建议回查标注质量。")
    top_drift = cross_split_analysis["top_class_drift_by_boxes"][:1]
    if top_drift and top_drift[0]["box_split_js_divergence"] > 0.1:
        insights.append(
            f"类别 {top_drift[0]['name']} 的 box split 偏移最大，JS divergence 为 {top_drift[0]['box_split_js_divergence']}。"
        )

    report = {
        "meta": {
            "task": "det",
            "generated_at": rt.timestamp_now_iso(),
            "data_yaml": str(source_data_path.expanduser().resolve()),
            "dataset_root": str(source_root),
            "dataset_tag": rt.dataset_tag_from_dir(source_root),
            "split_order": [split_name for split_name in SPLIT_ORDER if split_name in finalized_splits],
            "export_split_paths": export_split_paths,
            "size_bucket_rule": {
                "small_max_area_px": SMALL_OBJECT_AREA_THRESHOLD,
                "medium_max_area_px": MEDIUM_OBJECT_AREA_THRESHOLD,
            },
        },
        "overview": {
            "images": total_images,
            "labeled_images": finalized_all["labeled_images"],
            "empty_images": finalized_all["empty_images"],
            "boxes": total_boxes,
            "class_count": len(class_summary),
            "empty_image_ratio": finalized_all["empty_image_ratio"],
            "labeled_image_ratio": finalized_all["labeled_image_ratio"],
        },
        "splits": finalized_splits,
        "classes": class_summary,
        "imbalance_analysis": imbalance_analysis,
        "cross_split_analysis": cross_split_analysis,
        "insights": insights,
    }
    return report, per_image_rows


def _split_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, split_entry in report["splits"].items():
        if split_name == "all":
            continue
        rows.append(
            {
                "split": split_name,
                "images": split_entry["images"],
                "labeled_images": split_entry["labeled_images"],
                "empty_images": split_entry["empty_images"],
                "empty_image_ratio": split_entry["empty_image_ratio"],
                "missing_label_files": split_entry["missing_label_files"],
                "empty_label_files": split_entry["empty_label_files"],
                "invalid_label_lines": split_entry["invalid_label_lines"],
                "coord_anomaly_lines": split_entry["coord_anomaly_lines"],
                "boxes": split_entry["boxes"],
                "avg_boxes_per_image": split_entry["avg_boxes_per_image"],
                "avg_boxes_per_labeled_image": split_entry["avg_boxes_per_labeled_image"],
                "class_count": split_entry["class_count"],
                "small_ratio": split_entry["size_bucket_ratio"]["small"],
                "medium_ratio": split_entry["size_bucket_ratio"]["medium"],
                "large_ratio": split_entry["size_bucket_ratio"]["large"],
                "box_max_min_ratio": split_entry["imbalance_boxes"]["max_min_ratio"],
                "box_gini": split_entry["imbalance_boxes"]["gini"],
                "box_entropy_evenness": split_entry["imbalance_boxes"]["entropy_evenness"],
            }
        )
    return rows


def _class_summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_order = report["meta"]["split_order"]
    for class_entry in sorted(
        report["classes"].values(),
        key=lambda item: (-item["all"]["boxes"], -item["all"]["images"], item["class_id"]),
    ):
        row = {
            "class_id": class_entry["class_id"],
            "name": class_entry["name"],
            "all_images": class_entry["all"]["images"],
            "all_boxes": class_entry["all"]["boxes"],
            "all_image_share": class_entry["all"]["image_share"],
            "all_box_share": class_entry["all"]["box_share"],
            "avg_boxes_per_image": class_entry["all"]["avg_boxes_per_image"],
            "avg_box_area_ratio": class_entry["all"]["avg_box_area_ratio"],
            "median_box_area_ratio": class_entry["all"]["median_box_area_ratio"],
            "coverage_split_count": class_entry["coverage_split_count"],
            "missing_in_splits": ",".join(class_entry["missing_in_splits"]),
            "image_split_js_divergence": class_entry["image_split_js_divergence"],
            "box_split_js_divergence": class_entry["box_split_js_divergence"],
        }
        for split_name in split_order:
            split_entry = class_entry["splits"].get(split_name, {})
            row[f"{split_name}_images"] = split_entry.get("images", 0)
            row[f"{split_name}_boxes"] = split_entry.get("boxes", 0)
            row[f"{split_name}_image_share"] = split_entry.get("image_share", 0.0)
            row[f"{split_name}_box_share"] = split_entry.get("box_share", 0.0)
        rows.append(row)
    return rows


def _render_markdown(report: dict[str, Any], output_dir: Path) -> str:
    overview = report["overview"]
    split_order = report["meta"]["split_order"]
    lines = [
        "# 检测数据集 EDA 报告",
        "",
        "## 1. 基本信息",
        f"- 数据配置：`{report['meta']['data_yaml']}`",
        f"- 数据根目录：`{report['meta']['dataset_root']}`",
        f"- 输出目录：`{output_dir}`",
        f"- 生成时间：{report['meta']['generated_at']}",
        f"- 总图片数：{overview['images']}",
        f"- 有标注图片数：{overview['labeled_images']}",
        f"- 空标注图片数：{overview['empty_images']}",
        f"- 总框数：{overview['boxes']}",
        f"- 类别数：{overview['class_count']}",
        "- 目标尺寸规则：small < 32^2 px，medium < 96^2 px，large >= 96^2 px",
        "",
        "## 2. Split 总览",
        "",
        "| split | 图片数 | 有标注图 | 空标注图 | 框数 | 平均每图框数 | 小目标占比 | 中目标占比 | 大目标占比 | 类别数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split_name in split_order:
        split_entry = report["splits"][split_name]
        size_ratio = split_entry["size_bucket_ratio"]
        lines.append(
            f"| {split_name} | {split_entry['images']} | {split_entry['labeled_images']} | {split_entry['empty_images']} | "
            f"{split_entry['boxes']} | {split_entry['avg_boxes_per_image']:.4f} | {size_ratio['small']:.4f} | "
            f"{size_ratio['medium']:.4f} | {size_ratio['large']:.4f} | {split_entry['class_count']} |"
        )
    all_entry = report["splits"]["all"]
    all_ratio = all_entry["size_bucket_ratio"]
    lines.append(
        f"| all | {all_entry['images']} | {all_entry['labeled_images']} | {all_entry['empty_images']} | "
        f"{all_entry['boxes']} | {all_entry['avg_boxes_per_image']:.4f} | {all_ratio['small']:.4f} | "
        f"{all_ratio['medium']:.4f} | {all_ratio['large']:.4f} | {all_entry['class_count']} |"
    )

    lines.extend(
        [
            "",
            "## 3. 不平衡性分析",
            "",
            "| scope | 口径 | max/min | max/median | top1 share | top3 share | CV | Gini | 熵均衡度 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scope_name, imbalance_entry in report["imbalance_analysis"].items():
        for metric_name in ("images", "boxes"):
            metric = imbalance_entry[metric_name]
            lines.append(
                f"| {scope_name} | {metric_name} | {metric['max_min_ratio']:.4f} | {metric['max_median_ratio']:.4f} | "
                f"{metric['top1_share']:.4f} | {metric['top3_share']:.4f} | {metric['coefficient_of_variation']:.4f} | "
                f"{metric['gini']:.4f} | {metric['entropy_evenness']:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 4. 类别全量对照",
            "",
            "| 类别 | all 图数 | all 框数 | train 框数 | val 框数 | test 框数 | 缺失 split | box JS drift |",
            "|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for class_entry in sorted(
        report["classes"].values(),
        key=lambda item: (-item["all"]["boxes"], -item["all"]["images"], item["class_id"]),
    ):
        lines.append(
            f"| {class_entry['name']} | {class_entry['all']['images']} | {class_entry['all']['boxes']} | "
            f"{class_entry['splits'].get('train', {}).get('boxes', 0)} | "
            f"{class_entry['splits'].get('val', {}).get('boxes', 0)} | "
            f"{class_entry['splits'].get('test', {}).get('boxes', 0)} | "
            f"{','.join(class_entry['missing_in_splits']) or '-'} | {class_entry['box_split_js_divergence']:.4f} |"
        )

    lines.extend(["", "## 5. 关键观察", ""])
    for insight in report["insights"]:
        lines.append(f"- {insight}")

    drift_items = report["cross_split_analysis"]["top_class_drift_by_boxes"]
    if drift_items:
        lines.extend(
            [
                "",
                "## 6. 跨 Split 偏移 Top 10",
                "",
                "| 类别 | image JS | box JS | 缺失 split |",
                "|---|---:|---:|---|",
            ]
        )
        for item in drift_items[:10]:
            lines.append(
                f"| {item['name']} | {item['image_split_js_divergence']:.4f} | "
                f"{item['box_split_js_divergence']:.4f} | {','.join(item['missing_in_splits']) or '-'} |"
            )

    for split_name in split_order:
        dense_images = report["splits"][split_name]["top_dense_images"]
        if not dense_images:
            continue
        lines.extend(
            [
                "",
                f"## 7. {split_name} 高密度样本 Top 10",
                "",
                "| 图片 | 框数 | 类别数 | 尺寸 |",
                "|---|---:|---:|---|",
            ]
        )
        for item in dense_images:
            lines.append(
                f"| {item['image']} | {item['boxes']} | {item['class_count']} | "
                f"{item['image_width']}x{item['image_height']} |"
            )
    bad_image_export = report.get("bad_image_export", {})
    if bad_image_export:
        lines.extend(
            [
                "",
                "## 8. 问题样本导出",
                "",
                f"- 导出目录：`{bad_image_export.get('directory', output_dir / 'bad_images')}`",
                f"- 候选问题样本数：{bad_image_export.get('candidate_count', 0)}",
                f"- 已导出代表性样本数：{bad_image_export.get('exported_count', 0)}",
                f"- 清单文件：`{bad_image_export.get('manifest_csv', '')}`",
                "",
                "| split | 候选数 | 导出数 | 读图失败 | 缺失标注 | 非法标注行 | 空标注文件 | 坐标异常 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split_name in split_order:
            split_export = bad_image_export.get("splits", {}).get(split_name)
            if not split_export:
                continue
            issue_counts = split_export.get("issue_counts", {})
            lines.append(
                f"| {split_name} | {split_export.get('candidate_count', 0)} | {split_export.get('exported_count', 0)} | "
                f"{issue_counts.get('image_open_error', 0)} | {issue_counts.get('missing_label_file', 0)} | "
                f"{issue_counts.get('invalid_label_lines', 0)} | {issue_counts.get('empty_label_file', 0)} | "
                f"{issue_counts.get('coord_anomaly_lines', 0)} |"
            )
    lines.append("")
    return "\n".join(lines)


def _default_eda_output_dir(source_data_path: Path) -> Path:
    source_cfg = _load_data_config_light(source_data_path)
    source_root = Path(source_cfg["_root_dir"]).resolve()
    dataset_tag = rt.dataset_tag_from_dir(source_root)
    base_dir = rt.EDA_OUTPUT_ROOT_DIR / f"{dataset_tag}-eda"
    return base_dir


def generate_eda_report(*, source_data_path: Path, output_dir: Path | None, overwrite: bool) -> Path:
    source_data_path = source_data_path.expanduser().resolve()
    if output_dir is not None:
        final_output_dir = output_dir.expanduser().resolve()
    else:
        base_dir = _default_eda_output_dir(source_data_path)
        final_output_dir = base_dir if overwrite else rt.deduplicate_path(base_dir)
    rt.prepare_output_dir(final_output_dir, overwrite=overwrite)

    report, per_image_rows = _collect_dataset_eda(source_data_path)
    report["bad_image_export"] = _export_bad_image_bundle(
        output_dir=final_output_dir,
        per_image_rows=per_image_rows,
    )
    markdown = _render_markdown(report, final_output_dir)
    dataset_tag = str(report.get("overview", {}).get("dataset_tag") or "dataset")

    split_rows = _split_summary_rows(report)
    class_rows = _class_summary_rows(report)

    json_path = final_output_dir / f"dataset_eda_{dataset_tag}.json"
    markdown_path = final_output_dir / f"dataset_eda_{dataset_tag}.md"
    split_csv_path = final_output_dir / f"split_summary_{dataset_tag}.csv"
    class_csv_path = final_output_dir / f"class_summary_{dataset_tag}.csv"
    image_csv_path = final_output_dir / f"image_inventory_{dataset_tag}.csv"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown + "\n", encoding="utf-8")
    rt.save_records_csv(
        split_csv_path,
        split_rows,
        [
            "split",
            "images",
            "labeled_images",
            "empty_images",
            "empty_image_ratio",
            "missing_label_files",
            "empty_label_files",
            "invalid_label_lines",
            "coord_anomaly_lines",
            "boxes",
            "avg_boxes_per_image",
            "avg_boxes_per_labeled_image",
            "class_count",
            "small_ratio",
            "medium_ratio",
            "large_ratio",
            "box_max_min_ratio",
            "box_gini",
            "box_entropy_evenness",
        ],
    )
    class_fieldnames = [
        "class_id",
        "name",
        "all_images",
        "all_boxes",
        "all_image_share",
        "all_box_share",
        "avg_boxes_per_image",
        "avg_box_area_ratio",
        "median_box_area_ratio",
        "coverage_split_count",
        "missing_in_splits",
        "image_split_js_divergence",
        "box_split_js_divergence",
    ]
    for split_name in report["meta"]["split_order"]:
        class_fieldnames.extend(
            [
                f"{split_name}_images",
                f"{split_name}_boxes",
                f"{split_name}_image_share",
                f"{split_name}_box_share",
            ]
        )
    rt.save_records_csv(class_csv_path, class_rows, class_fieldnames)
    rt.save_records_csv(
        image_csv_path,
        per_image_rows,
        [
            "split",
            "image",
            "image_path",
            "label_path",
            "image_width",
            "image_height",
            "image_area_pixels",
            "image_aspect_ratio",
            "label_file_exists",
            "raw_nonempty_label_lines",
            "valid_boxes",
            "invalid_label_lines",
            "coord_anomaly_lines",
            "labeled",
            "empty_image",
            "class_count",
            "dominant_class",
            "dominant_class_boxes",
            "small_boxes",
            "medium_boxes",
            "large_boxes",
            "mean_box_area_ratio",
            "mean_box_aspect_ratio",
            "image_open_error",
            "bad_image_candidate",
            "bad_image_issue_tags",
            "bad_image_primary_issue",
            "bad_image_score",
            "bad_image_issue_count",
        ],
    )

    print("EDA generated:")
    print(f"  - {markdown_path}")
    print(f"  - {json_path}")
    print(f"  - {split_csv_path}")
    print(f"  - {class_csv_path}")
    print(f"  - {image_csv_path}")
    print(f"  - {final_output_dir / 'bad_images'}")
    return final_output_dir


def run_eda(args: argparse.Namespace) -> None:
    generate_eda_report(
        source_data_path=Path(args.data),
        output_dir=getattr(args, "output_dir", None),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
