"""检测实验报告生成。

这里负责两类场景：
- det infer 在数据集模式跑完后，自动把报告写到当前 infer 输出目录
- 用户单独执行 report 命令时，按实验目录聚合已有 val/test infer 结果生成报告
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from . import common as rt

REPORT_BASENAME = "single_report"
TRAIN_SINGLE_REPORT_ROOT = rt.OUT_DIR / "train_sigle_report"
SMALL_OBJECT_AREA_THRESHOLD = 32.0 * 32.0
MEDIUM_OBJECT_AREA_THRESHOLD = 96.0 * 96.0
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class InferRunRecord:
    output_dir: Path
    split: str
    created_at: str | None
    report_path: Path | None
    metrics_path: Path | None
    run_meta_path: Path
    report_payload: dict[str, Any] | None
    metrics_payload: dict[str, Any] | None
    run_meta: dict[str, Any]
    mtime: float


def _display_path(path: Path | None) -> str:
    if path is None:
        return "-"
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(rt.ROOT_DIR))
    except ValueError:
        return str(resolved)


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _resolve_optional_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _extract_json_block(text: str, marker: str) -> dict[str, Any]:
    marker_index = text.find(marker)
    if marker_index < 0:
        return {}
    start = text.find("{", marker_index)
    if start < 0:
        return {}
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                block = text[start : index + 1]
                try:
                    payload = json.loads(block)
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


def _extract_first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None


def _extract_last_match(text: str, pattern: str) -> str | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    return value.strip() if isinstance(value, str) else str(value).strip()


def _format_metric(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "-"


def _format_count(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return "-"


def _format_class_list(class_names: dict[int, str], limit: int = 5) -> str:
    if not class_names:
        return "待补充"
    ordered_names = [class_names[index] for index in sorted(class_names)]
    if len(ordered_names) <= limit:
        return "、".join(ordered_names)
    return "、".join(ordered_names[:limit]) + "..."


def _sanitize_report_name_fragment(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-._")
    return cleaned or "dataset"


def _dataset_name_from_root(data_root: Path | None) -> str | None:
    if data_root is None:
        return None
    leaf_name = data_root.name.strip()
    parent_name = data_root.parent.name.strip() if data_root.parent != data_root else ""
    if not leaf_name:
        return None
    if leaf_name.startswith("dataset_") and parent_name:
        return _sanitize_report_name_fragment(f"{parent_name}-{leaf_name}")
    return _sanitize_report_name_fragment(leaf_name)


def _infer_data_root_from_run(primary_run: InferRunRecord, train_info: dict[str, Any]) -> Path | None:
    dataset_info = _dataset_summary(train_info)
    data_root = dataset_info.get("data_root")
    if isinstance(data_root, Path):
        return data_root

    run_meta_data_root = _resolve_optional_path(primary_run.run_meta.get("paths", {}).get("data_root"))
    if run_meta_data_root is not None:
        return run_meta_data_root

    report_config = primary_run.report_payload.get("config") if isinstance(primary_run.report_payload, dict) else None
    if isinstance(report_config, dict):
        report_data_root = _resolve_optional_path(report_config.get("data_root"))
        if report_data_root is not None:
            return report_data_root
    return None


def _build_report_filename(primary_run: InferRunRecord, train_info: dict[str, Any]) -> str:
    data_root = _infer_data_root_from_run(primary_run, train_info)
    dataset_name = _dataset_name_from_root(data_root)
    if dataset_name:
        return f"{REPORT_BASENAME}_{dataset_name}.md"
    return f"{REPORT_BASENAME}.md"


def _experiment_archive_dir(experiment_dir: Path) -> Path:
    return TRAIN_SINGLE_REPORT_ROOT / experiment_dir.name


def _build_lr_text(args_payload: dict[str, Any]) -> str:
    lr = args_payload.get("lr")
    warmup_steps = args_payload.get("lr_warmup_steps")
    backbone_lr_factor = args_payload.get("backbone_lr_factor")
    scheduler_start_factor = args_payload.get("scheduler_start_factor")
    parts: list[str] = []
    if isinstance(lr, (int, float)):
        parts.append(f"lr={float(lr):g}")
    if isinstance(warmup_steps, int):
        parts.append(f"warmup_steps={warmup_steps}")
    if isinstance(backbone_lr_factor, (int, float)):
        parts.append(f"backbone_lr_factor={float(backbone_lr_factor):g}")
    if isinstance(scheduler_start_factor, (int, float)):
        parts.append(f"scheduler_start_factor={float(scheduler_start_factor):g}")
    if parts:
        return ", ".join(parts)
    return "见 `training_curve_lr.png`"


def _build_lr_text_from_log(text: str) -> str | None:
    patterns = [
        ("lr", r'"lr"\s*:\s*([0-9.eE+-]+)'),
        ("warmup_steps", r'"lr_warmup_steps"\s*:\s*(\d+)'),
        ("backbone_lr_factor", r'"backbone_lr_factor"\s*:\s*([0-9.eE+-]+)'),
        ("scheduler_start_factor", r'"scheduler_start_factor"\s*:\s*([0-9.eE+-]+)'),
    ]
    parts: list[str] = []
    for label, pattern in patterns:
        value = _extract_first_match(text, pattern)
        if value is not None:
            parts.append(f"{label}={value}")
    if parts:
        return ", ".join(parts)
    return None


def _metric_value(run: InferRunRecord, *keys: str) -> float | None:
    containers = []
    if run.metrics_payload:
        containers.append(run.metrics_payload.get("metrics"))
    if run.report_payload:
        containers.append(run.report_payload.get("metrics"))
        containers.append(run.report_payload.get("summary"))
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in keys:
            value = container.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _summary_value(run: InferRunRecord, key: str) -> Any:
    if run.report_payload and isinstance(run.report_payload.get("summary"), dict):
        summary = run.report_payload["summary"]
        if key in summary:
            return summary[key]
    if run.metrics_payload and key == "avg_infer_time_ms":
        return run.metrics_payload.get("avg_infer_time_ms")
    return None


def _bucket_box_area(area_pixels: float) -> str:
    if area_pixels < SMALL_OBJECT_AREA_THRESHOLD:
        return "small"
    if area_pixels < MEDIUM_OBJECT_AREA_THRESHOLD:
        return "medium"
    return "large"


def _parse_train_log(experiment_dir: Path) -> dict[str, Any]:
    train_log_path = experiment_dir / "train.log"
    if not train_log_path.exists():
        return {
            "path": train_log_path,
            "args": {},
            "start_time": None,
            "end_time": None,
            "total_time": None,
            "train_time": None,
            "val_time": None,
            "train_throughput": None,
            "val_throughput": None,
            "hardware_summary": None,
            "final_val_line": None,
            "best_result": None,
            "lr_text": None,
        }

    text = train_log_path.read_text(encoding="utf-8", errors="ignore")
    args_payload = _extract_json_block(text, "Args:")
    timestamps = re.findall(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]", text)
    gpu_count = _extract_first_match(text, r"\]\[DEBUG\] GPUs:\s+(\d+)")
    gpu_lines = re.findall(r"\]\[DEBUG\]\s+-\s+(.+?\(\d+\))$", text, flags=re.MULTILINE)
    hardware_summary = None
    if gpu_lines:
        gpu_label = gpu_lines[0].split("(")[0].strip()
        hardware_summary = f"{gpu_count or len(gpu_lines)} * {gpu_label}"

    final_val_line = _extract_last_match(
        text,
        r"(\[\d{4}-\d{2}-\d{2} .*?Val Step\s+\d+/\d+ \| Val Loss: .*?)$",
    )
    best_result = _extract_last_match(text, r"\]\[INFO\] Best result:\s+(.+)$")
    if best_result is None:
        best_result = _extract_last_match(text, r"\]\[INFO\] The best validation metric\s+(.+?)\s+was reached\.$")

    return {
        "path": train_log_path,
        "args": args_payload,
        "start_time": timestamps[0] if timestamps else None,
        "end_time": timestamps[-1] if timestamps else None,
        "total_time": _extract_first_match(text, r"\]\[INFO\]\s+Total Time\s+:\s+(.+)$"),
        "train_time": _extract_first_match(text, r"\]\[INFO\]\s+Train Time\s+:\s+(.+)$"),
        "val_time": _extract_first_match(text, r"\]\[INFO\]\s+Val Time\s+:\s+(.+)$"),
        "train_throughput": _extract_first_match(text, r"\]\[INFO\]\s+Train Throughput\s+:\s+(.+)$"),
        "val_throughput": _extract_first_match(text, r"\]\[INFO\]\s+Val Throughput\s+:\s+(.+)$"),
        "hardware_summary": hardware_summary,
        "final_val_line": final_val_line,
        "best_result": best_result,
        "lr_text": _build_lr_text_from_log(text),
    }


def _resolve_split_dir(root_dir: Path, split_value: str | None) -> Path | None:
    if not split_value:
        return None
    raw = Path(split_value)
    if raw.is_absolute():
        return raw.resolve()
    return (root_dir / raw).resolve()


def _count_images(image_dir: Path | None) -> int:
    if image_dir is None or not image_dir.exists():
        return 0
    return sum(1 for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _dataset_summary(train_info: dict[str, Any]) -> dict[str, Any]:
    args_payload = train_info.get("args", {})
    data_payload = args_payload.get("data")
    if not isinstance(data_payload, dict):
        return {
            "data_root": None,
            "class_names": {},
            "counts": {},
        }

    data_root_raw = data_payload.get("path")
    data_root = _resolve_optional_path(data_root_raw)
    if data_root is None and isinstance(data_root_raw, str):
        data_root = (rt.ROOT_DIR / data_root_raw).resolve()

    names_raw = data_payload.get("names")
    class_names = {}
    if isinstance(names_raw, dict):
        class_names = {int(key): str(value) for key, value in names_raw.items()}
    elif isinstance(names_raw, list):
        class_names = {index: str(value) for index, value in enumerate(names_raw)}

    counts: dict[str, int] = {}
    if data_root is not None:
        for split_name in ("train", "val", "test"):
            split_dir = _resolve_split_dir(data_root, data_payload.get(split_name))
            counts[split_name] = _count_images(split_dir)

    return {
        "data_root": data_root,
        "class_names": class_names,
        "counts": counts,
    }


def _find_report_path(output_dir: Path, run_meta: dict[str, Any]) -> Path | None:
    report_path = _resolve_optional_path(run_meta.get("paths", {}).get("report_path"))
    if report_path is not None and report_path.exists():
        return report_path
    candidates = sorted(output_dir.glob("*-test_report.json"))
    if candidates:
        return candidates[0]
    fallback = output_dir / "test_report.json"
    if fallback.exists():
        return fallback
    return None


def _discover_infer_runs(experiment_dir: Path) -> list[InferRunRecord]:
    runs: list[InferRunRecord] = []
    infer_root = rt.TEST_OUTPUT_ROOT_DIR
    if not infer_root.exists():
        return runs

    expected_experiment_dir = experiment_dir.expanduser().resolve()
    for run_meta_path in infer_root.rglob("run_meta.json"):
        run_meta = _load_json(run_meta_path)
        if run_meta is None:
            continue
        if run_meta.get("task") != "det" or run_meta.get("action") != "infer":
            continue
        paths_payload = run_meta.get("paths", {})
        recorded_experiment_dir = _resolve_optional_path(paths_payload.get("experiment_dir"))
        if recorded_experiment_dir != expected_experiment_dir:
            continue

        output_dir = _resolve_optional_path(paths_payload.get("output_dir"))
        if output_dir is None:
            output_dir = run_meta_path.parent
        report_path = _find_report_path(output_dir, run_meta)
        metrics_path = _resolve_optional_path(run_meta.get("artifacts", {}).get("metrics_summary"))
        if metrics_path is None:
            candidate = output_dir / "metrics_summary.json"
            metrics_path = candidate if candidate.exists() else None

        report_payload = _load_json(report_path)
        metrics_payload = _load_json(metrics_path)
        report_config = report_payload.get("config", {}) if isinstance(report_payload, dict) else {}
        split = str(run_meta.get("split") or report_config.get("split") or "unknown")
        runs.append(
            InferRunRecord(
                output_dir=output_dir,
                split=split,
                created_at=str(run_meta.get("created_at")) if run_meta.get("created_at") else None,
                report_path=report_path,
                metrics_path=metrics_path,
                run_meta_path=run_meta_path,
                report_payload=report_payload,
                metrics_payload=metrics_payload,
                run_meta=run_meta,
                mtime=run_meta_path.stat().st_mtime,
            )
        )

    runs.sort(
        key=lambda item: (
            item.created_at or "",
            item.mtime,
            item.output_dir.name,
        ),
        reverse=True,
    )
    return runs


def _latest_runs_by_split(runs: list[InferRunRecord]) -> list[InferRunRecord]:
    latest: dict[str, InferRunRecord] = {}
    for run in runs:
        latest.setdefault(run.split, run)
    ordered: list[InferRunRecord] = []
    for split_name in ("test", "val"):
        if split_name in latest:
            ordered.append(latest[split_name])
    for run in latest.values():
        if run.split not in {"test", "val"}:
            ordered.append(run)
    return ordered


def _pick_primary_run(runs: list[InferRunRecord]) -> InferRunRecord:
    latest_runs = _latest_runs_by_split(runs)
    for split_name in ("test", "val"):
        for run in latest_runs:
            if run.split == split_name:
                return run
    return runs[0]


def _list_images_by_stem(image_dir: Path) -> dict[Path, Path]:
    mapping: dict[Path, Path] = {}
    for image_path in image_dir.rglob("*"):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
            rel_path = image_path.relative_to(image_dir)
            mapping[rel_path.with_suffix("")] = image_path
    return mapping


def _compute_size_buckets(report_payload: dict[str, Any] | None) -> dict[int, dict[str, int]]:
    if not isinstance(report_payload, dict):
        return {}
    config = report_payload.get("config", {})
    class_names = report_payload.get("class_names", {})
    if not isinstance(config, dict) or not isinstance(class_names, dict):
        return {}

    image_dir = _resolve_optional_path(config.get("test_images"))
    label_dir = _resolve_optional_path(config.get("test_labels"))
    if image_dir is None or label_dir is None or not image_dir.exists() or not label_dir.exists():
        return {}

    image_map = _list_images_by_stem(image_dir)
    buckets: dict[int, dict[str, int]] = defaultdict(lambda: {"small": 0, "medium": 0, "large": 0})
    for label_path in label_dir.rglob("*.txt"):
        rel_stem = label_path.relative_to(label_dir).with_suffix("")
        image_path = image_map.get(rel_stem)
        if image_path is None:
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception:
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(float(parts[0]))
                box_width = max(float(parts[3]), 0.0) * float(width)
                box_height = max(float(parts[4]), 0.0) * float(height)
            except ValueError:
                continue
            bucket_name = _bucket_box_area(box_width * box_height)
            buckets[class_id][bucket_name] += 1
    return {class_id: dict(values) for class_id, values in buckets.items()}


def _classwise_map_lookup(run: InferRunRecord) -> dict[str, float]:
    if not run.metrics_payload:
        return {}
    metrics = run.metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    lookup: dict[str, float] = {}
    prefix = "eval_metric_classwise/map_"
    for key, value in metrics.items():
        if key.startswith(prefix) and isinstance(value, (int, float)):
            lookup[key[len(prefix) :]] = float(value)
    return lookup


def _classwise_size_map_lookup(run: InferRunRecord, size_name: str) -> dict[str, float]:
    if not run.metrics_payload:
        return {}
    metrics = run.metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        return {}

    prefixes = [
        f"eval_metric_classwise/map_{size_name}_",
        f"eval_metric_classwise/map_{size_name}/",
    ]
    suffixes = [
        f"_{size_name}",
        f"/{size_name}",
    ]

    lookup: dict[str, float] = {}
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        for prefix in prefixes:
            if key.startswith(prefix):
                lookup[key[len(prefix) :]] = float(value)
        for suffix in suffixes:
            if key.startswith("eval_metric_classwise/map_") and key.endswith(suffix):
                class_name = key[len("eval_metric_classwise/map_") : -len(suffix)]
                if class_name:
                    lookup[class_name] = float(value)
    return lookup


def _dominant_bucket(bucket_counts: dict[str, int]) -> str:
    if not bucket_counts:
        return "-"
    return max(bucket_counts, key=bucket_counts.get)


def _build_overall_table(run: InferRunRecord) -> str:
    rows = [
        ("整体准确度 (mAP@0.5:0.95)", _metric_value(run, "eval_metric/map", "map")),
        ("mAP@0.5", _metric_value(run, "eval_metric/map_50", "map_50", "map_05")),
        ("mAP@0.75", _metric_value(run, "eval_metric/map_75", "map_75")),
        ("小目标 (Small)", _metric_value(run, "eval_metric/map_small", "map_small")),
        ("中目标 (Medium)", _metric_value(run, "eval_metric/map_medium", "map_medium")),
        ("大目标 (Large)", _metric_value(run, "eval_metric/map_large", "map_large")),
        ("平均推理耗时 (ms)", _summary_value(run, "avg_infer_time_ms")),
    ]
    lines = ["| 指标 | 数值 |", "|---|---:|"]
    for label, value in rows:
        lines.append(f"| {label} | {_format_metric(value)} |")
    return "\n".join(lines)


def _build_per_class_table(run: InferRunRecord) -> str:
    report_payload = run.report_payload or {}
    per_class_ap = report_payload.get("per_class_ap")
    if not isinstance(per_class_ap, dict) or not per_class_ap:
        return "| 类别名称 (Label) | GT | Pred | AP@0.5 | mAP@0.5:0.95 | mAP(S) | mAP(M) | mAP(L) | 备注 |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n| 当前 split 缺少按类结果 | - | - | - | - | - | - | - | - |"

    classwise_lookup = _classwise_map_lookup(run)
    small_lookup = _classwise_size_map_lookup(run, "small")
    medium_lookup = _classwise_size_map_lookup(run, "medium")
    large_lookup = _classwise_size_map_lookup(run, "large")
    entries: list[tuple[int, dict[str, Any]]] = []
    for class_id_raw, info in per_class_ap.items():
        try:
            class_id = int(class_id_raw)
        except ValueError:
            continue
        if isinstance(info, dict):
            entries.append((class_id, info))
    entries.sort(key=lambda item: item[0])

    lines = [
        "| 类别名称 (Label) | GT | Pred | AP@0.5 | mAP@0.5:0.95 | mAP(S) | mAP(M) | mAP(L) | 备注 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for class_id, info in entries:
        class_name = str(info.get("name") or f"class_{class_id}")
        remark = "来源: test_report.per_class_ap"
        if class_name in small_lookup or class_name in medium_lookup or class_name in large_lookup:
            remark = "来源: test_report + classwise size mAP"
        lines.append(
            "| "
            + " | ".join(
                [
                    class_name,
                    _format_count(info.get("gt")),
                    _format_count(info.get("pred")),
                    _format_metric(info.get("ap")),
                    _format_metric(classwise_lookup.get(class_name)),
                    _format_metric(small_lookup.get(class_name)),
                    _format_metric(medium_lookup.get(class_name)),
                    _format_metric(large_lookup.get(class_name)),
                    remark,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_split_table(runs: list[InferRunRecord]) -> str:
    lines = [
        "| Split | 输出目录 | 图像数 | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | Small | Medium | Large | Avg Infer (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            "| "
            + " | ".join(
                [
                    run.split,
                    f"`{_display_path(run.output_dir)}`",
                    _format_count(_summary_value(run, "num_images")),
                    _format_metric(_metric_value(run, "eval_metric/map", "map")),
                    _format_metric(_metric_value(run, "eval_metric/map_50", "map_50", "map_05")),
                    _format_metric(_metric_value(run, "eval_metric/map_75", "map_75")),
                    _format_metric(_metric_value(run, "eval_metric/map_small", "map_small")),
                    _format_metric(_metric_value(run, "eval_metric/map_medium", "map_medium")),
                    _format_metric(_metric_value(run, "eval_metric/map_large", "map_large")),
                    _format_metric(_summary_value(run, "avg_infer_time_ms")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _build_conclusion_lines(primary_run: InferRunRecord, train_info: dict[str, Any]) -> list[str]:
    report_payload = primary_run.report_payload or {}
    per_class_ap = report_payload.get("per_class_ap") if isinstance(report_payload.get("per_class_ap"), dict) else {}
    ap_entries: list[tuple[str, float]] = []
    for info in per_class_ap.values():
        if isinstance(info, dict) and isinstance(info.get("ap"), (int, float)):
            ap_entries.append((str(info.get("name") or "unknown"), float(info["ap"])))
    ap_entries.sort(key=lambda item: item[1], reverse=True)

    lines = []
    lines.append(
        f"本次主报告以 `{primary_run.split}` 结果为准，mAP@0.5:0.95 为 {_format_metric(_metric_value(primary_run, 'eval_metric/map', 'map'))}，mAP@0.5 为 {_format_metric(_metric_value(primary_run, 'eval_metric/map_50', 'map_50', 'map_05'))}。"
    )
    large_value = _metric_value(primary_run, "eval_metric/map_large", "map_large") or 0.0
    medium_value = _metric_value(primary_run, "eval_metric/map_medium", "map_medium") or 0.0
    small_value = _metric_value(primary_run, "eval_metric/map_small", "map_small") or 0.0
    if large_value >= medium_value and large_value >= small_value:
        lines.append("当前结果里大目标表现更突出，下一步建议优先增强 small / medium 样本覆盖、输入尺度和召回策略。")
    else:
        lines.append("当前结果里各尺寸目标已经开始同步收敛，下一步建议继续围绕低值尺寸桶补齐样本和阈值调优。")
    if ap_entries and ap_entries[0][1] > 0.0:
        best_name, best_ap = ap_entries[0]
        focus_names = [name for name, ap in ap_entries if ap <= 0.01][:3]
        lines.append(f"类别层面 `{best_name}` 当前 AP@0.5 相对更高，为 {best_ap:.4f}。")
        if focus_names:
            lines.append(f"后续优化建议优先覆盖 {', '.join(focus_names)} 这几类的样本质量、正样本数量和标签一致性。")
    elif ap_entries:
        lines.append("当前类别 AP 仍处于统一起点，下一步建议优先围绕高频类别做样本质量复核和召回提升。")
    if train_info.get("best_result"):
        lines.append(f"训练过程记录的最佳指标为 `{train_info['best_result']}`。")
    return lines


def _render_report(
    *,
    experiment_dir: Path,
    primary_run: InferRunRecord,
    split_runs: list[InferRunRecord],
    train_info: dict[str, Any],
) -> str:
    dataset_info = _dataset_summary(train_info)
    args_payload = train_info.get("args", {})
    data_root = dataset_info.get("data_root")
    class_names = dataset_info.get("class_names", {})
    counts = dataset_info.get("counts", {})

    experiment_date = "-"
    if train_info.get("start_time"):
        experiment_date = str(train_info["start_time"]).split(" ")[0]
    elif primary_run.created_at:
        try:
            experiment_date = datetime.fromisoformat(primary_run.created_at).date().isoformat()
        except ValueError:
            experiment_date = primary_run.created_at[:10]

    model_name = str(args_payload.get("model") or "待补充")
    batch_size = args_payload.get("batch_size")
    steps = args_payload.get("steps")
    class_count = len(class_names) if class_names else "-"
    data_root_text = f"`{_display_path(data_root)}`" if data_root is not None else "待补充"
    class_list_text = _format_class_list(class_names, limit=5)
    lr_text = train_info.get("lr_text") or _build_lr_text(args_payload)
    hardware_text = train_info.get("hardware_summary") or "待补充"
    total_time_text = train_info.get("total_time") or "待补充"
    train_time_text = train_info.get("train_time") or "待补充"
    val_time_text = train_info.get("val_time") or "待补充"
    conclusion_lines = _build_conclusion_lines(primary_run, train_info)

    lines = [
        "# 算法实验验证报告",
        "",
        "## 一、实验基本信息",
        "",
        "- **实验目的**：目标检测实验结果汇总与推理结果报告生成",
        f"- **实验日期**：{experiment_date}",
        "- **实验人员**：待补充",
        f"- **实验目录**：`{_display_path(experiment_dir)}`",
        "",
        "## 二、模型结构与训练设置",
        "",
        "### 1. 模型结构",
        f"- Lightly-train + {model_name}",
        "",
        "### 2. 训练数据情况",
        f"- 数据集根目录：{data_root_text}",
        f"- 数据集划分：train {counts.get('train', 0)} 张，val {counts.get('val', 0)} 张，test {counts.get('test', 0)} 张",
        f"- 类别列表：{class_list_text}",
        "",
        "### 3. 训练时长",
        f"- 总耗时：{total_time_text}",
        f"- 训练耗时：{train_time_text}",
        f"- 验证耗时：{val_time_text}",
        "",
        "### 4. 基本参数设置",
        "",
        "| 参数项 | 内容 |",
        "|---|---|",
        f"| 骨干网 (Backbone) | {model_name} |",
        f"| Epoch / Iters | {steps if isinstance(steps, int) else '待补充'} steps |",
        f"| Batch Size | {batch_size if isinstance(batch_size, int) else '待补充'} |",
        f"| 学习率 (LR) | {lr_text} |",
        f"| 硬件环境 | {hardware_text} |",
        f"| 类别数量 | {class_count} |",
        "",
        "## 三、验证结果分析",
        "",
        f"### 1. 整体及不同尺寸目标准确度（主结果：{primary_run.split})",
        "",
        _build_overall_table(primary_run),
        "",
        f"### 2. 具体类别（Label）准确度（主结果：{primary_run.split})",
        "> 每个类别的 GT / Pred / AP@0.5 来自 `test_report.per_class_ap`；按类 mAP(S/M/L) 有值时会直接写入。",
        "",
        _build_per_class_table(primary_run),
        "",
        "### 3. 对应 infer 结果汇总",
        "",
        _build_split_table(split_runs),
        "",
        "### 4. 训练收敛摘要",
        "",
        f"- 最终验证日志：{train_info.get('final_val_line') or '待补充'}",
        f"- 最佳指标：{train_info.get('best_result') or '待补充'}",
        "",
        "## 四、结论与后续建议",
        "",
    ]
    for item in conclusion_lines:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 五、附件",
            "",
            "- 训练曲线文件：`training_curve_loss.png`、`training_curve_map.png`、`training_curve_lr.png`、`training_dashboard.png`",
            f"- 当前主结果目录：`{_display_path(primary_run.output_dir)}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def generate_experiment_report(
    *,
    experiment_dir: Path,
    output_dir: Path | None = None,
    write_all_infer_dirs: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    resolved_experiment_dir = experiment_dir.expanduser().resolve()
    if not resolved_experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {resolved_experiment_dir}")

    infer_runs = _discover_infer_runs(resolved_experiment_dir)
    if not infer_runs:
        raise FileNotFoundError(
            f"No det infer results with run_meta.json were found for experiment: {resolved_experiment_dir}"
        )

    primary_run = _pick_primary_run(infer_runs)
    split_runs = _latest_runs_by_split(infer_runs)
    train_info = _parse_train_log(resolved_experiment_dir)
    report_text = _render_report(
        experiment_dir=resolved_experiment_dir,
        primary_run=primary_run,
        split_runs=split_runs,
        train_info=train_info,
    )
    report_filename = _build_report_filename(primary_run, train_info)

    if output_dir is not None:
        target_dirs = [output_dir.expanduser().resolve()]
    elif write_all_infer_dirs:
        target_dirs = list(dict.fromkeys(run.output_dir for run in infer_runs))
    else:
        target_dirs = [primary_run.output_dir]
    archive_dir = _experiment_archive_dir(resolved_experiment_dir)
    if archive_dir not in target_dirs:
        target_dirs.append(archive_dir)

    output_paths: list[Path] = []
    for target_dir in target_dirs:
        report_path = target_dir / report_filename
        output_paths.append(report_path)
        if dry_run:
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
    return output_paths


def generate_report_for_infer_output(*, experiment_dir: Path, output_dir: Path) -> list[Path]:
    return generate_experiment_report(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        write_all_infer_dirs=False,
        dry_run=False,
    )


def run_report(args: argparse.Namespace) -> None:
    experiment_dir = getattr(args, "experiment_dir", None)
    search_keyword = getattr(args, "search", None)
    if experiment_dir is None:
        from .interactive import prompt_experiment_dir

        experiment_dir = prompt_experiment_dir(
            "det",
            rt.INFER_DEFAULT_EXPERIMENT_DIR,
            initial_keyword=search_keyword,
        )

    report_paths = generate_experiment_report(
        experiment_dir=Path(experiment_dir),
        output_dir=getattr(args, "output_dir", None),
        write_all_infer_dirs=getattr(args, "output_dir", None) is None,
        dry_run=getattr(args, "dry_run", False),
    )
    if getattr(args, "dry_run", False):
        print("Planned report outputs:")
        for report_path in report_paths:
            print(f"  - {report_path}")
        return

    print("Report generated:")
    for report_path in report_paths:
        print(f"  - {report_path}")
