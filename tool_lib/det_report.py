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
from .file_index import find_files

REPORT_BASENAME = "single_report"
SMALL_OBJECT_AREA_THRESHOLD = 32.0 * 32.0
MEDIUM_OBJECT_AREA_THRESHOLD = 96.0 * 96.0
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
RUN_RECORD_CANDIDATES = (
    Path("daily_report/run_records.jsonl"),
    Path("out/daily_reports/run_records.jsonl"),
)


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


def _parse_duration_to_minutes(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(h|hr|hrs|hour|hours|min|m|sec|s)\b", text.lower())
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return value * 60.0
    if unit in {"min", "m"}:
        return value
    return value / 60.0


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_ratio_percent(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"\(([0-9]+(?:\.[0-9]+)?)%\)", text)
    if match is None:
        return None
    return float(match.group(1))


def _parse_seconds_per_step(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"\(([0-9]+(?:\.[0-9]+)?)\s*s/step\)", text)
    if match is None:
        return None
    return float(match.group(1))


def _format_minutes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f} min"


def _classify_model_scale(model_name: str | None) -> str:
    text = (model_name or "").lower()
    if any(token in text for token in ("vitl", "large")):
        return "L"
    if any(token in text for token in ("vitb", "vitm", "base", "medium")):
        return "M"
    if any(token in text for token in ("vits", "small", "tiny")):
        return "S"
    return "UNKNOWN"


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


def _infer_date_tag_from_run(primary_run: InferRunRecord) -> str:
    if primary_run.created_at:
        try:
            return datetime.fromisoformat(primary_run.created_at).strftime("%m%d")
        except ValueError:
            return _sanitize_report_name_fragment(primary_run.created_at[:10]).replace("-", "")[-4:]
    return datetime.fromtimestamp(primary_run.mtime).strftime("%m%d")


def _build_report_filename(primary_run: InferRunRecord, train_info: dict[str, Any], *, split_name: str | None = None) -> str:
    data_root = _infer_data_root_from_run(primary_run, train_info)
    dataset_name = _dataset_name_from_root(data_root)
    date_tag = _infer_date_tag_from_run(primary_run)
    split_suffix = f"_{_sanitize_report_name_fragment(split_name)}" if split_name else ""
    if dataset_name:
        return f"{REPORT_BASENAME}_{date_tag}{split_suffix}_{dataset_name}.md"
    return f"{REPORT_BASENAME}_{date_tag}{split_suffix}.md"


def _extract_backbone_name(args_payload: dict[str, Any]) -> str:
    model_args = args_payload.get("model_args")
    if isinstance(model_args, dict):
        backbone_weights = model_args.get("backbone_weights")
        if isinstance(backbone_weights, str) and backbone_weights.strip():
            weight_name = Path(backbone_weights).stem
            match = re.search(r"(dinov\d+[_-][A-Za-z0-9]+)", weight_name)
            if match:
                return match.group(1).replace("_", "/")
            return weight_name
        backbone_args = model_args.get("backbone_args")
        if isinstance(backbone_args, dict):
            weights = backbone_args.get("weights")
            if isinstance(weights, str) and weights.strip():
                weight_name = Path(weights).stem
                match = re.search(r"(dinov\d+[_-][A-Za-z0-9]+)", weight_name)
                if match:
                    return match.group(1).replace("_", "/")
                return weight_name

    model_name = str(args_payload.get("model") or "").strip()
    if "/" in model_name and "-" in model_name:
        return model_name.rsplit("-", 1)[0]
    if model_name:
        return model_name
    return "待补充"


def _build_epoch_text(steps: Any, batch_size: Any, train_images: int) -> str:
    if not isinstance(steps, int):
        return "待补充"
    if isinstance(batch_size, int) and train_images > 0:
        epochs = (steps * batch_size) / float(train_images)
        return f"{epochs:.2f} epoch / {steps} steps"
    return f"{steps} steps"


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
    return "见 `training_dashboard.png`"


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
            "total_minutes": None,
            "train_minutes": None,
            "val_minutes": None,
            "train_ratio_percent": None,
            "val_ratio_percent": None,
            "train_seconds_per_step": None,
            "val_seconds_per_step": None,
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

    total_time_text = _extract_last_match(text, r"\]\[INFO\]\s+Total Time\s+:\s+(.+)$")
    train_time_text = _extract_last_match(text, r"\]\[INFO\]\s+Train Time\s+:\s+(.+)$")
    val_time_text = _extract_last_match(text, r"\]\[INFO\]\s+Val Time\s+:\s+(.+)$")

    return {
        "path": train_log_path,
        "args": args_payload,
        "start_time": timestamps[0] if timestamps else None,
        "end_time": timestamps[-1] if timestamps else None,
        "total_time": total_time_text,
        "train_time": train_time_text,
        "val_time": val_time_text,
        "train_throughput": _extract_first_match(text, r"\]\[INFO\]\s+Train Throughput\s+:\s+(.+)$"),
        "val_throughput": _extract_first_match(text, r"\]\[INFO\]\s+Val Throughput\s+:\s+(.+)$"),
        "hardware_summary": hardware_summary,
        "final_val_line": final_val_line,
        "best_result": best_result,
        "lr_text": _build_lr_text_from_log(text),
        "total_minutes": _parse_duration_to_minutes(total_time_text),
        "train_minutes": _parse_duration_to_minutes(train_time_text),
        "val_minutes": _parse_duration_to_minutes(val_time_text),
        "train_ratio_percent": _parse_ratio_percent(train_time_text),
        "val_ratio_percent": _parse_ratio_percent(val_time_text),
        "train_seconds_per_step": _parse_seconds_per_step(train_time_text),
        "val_seconds_per_step": _parse_seconds_per_step(val_time_text),
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


def _load_training_benchmark_summary(experiment_dir: Path, train_info: dict[str, Any]) -> dict[str, Any]:
    expected_dir = experiment_dir.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    current_row: dict[str, Any] | None = None

    for record_path in RUN_RECORD_CANDIDATES:
        resolved_path = (rt.ROOT_DIR / record_path).resolve()
        if not resolved_path.exists():
            continue
        for line in resolved_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("record_kind") != "train_run":
                continue
            if payload.get("task") != "object_detection":
                continue
            train_payload = payload.get("train") or {}
            model_name = str(train_payload.get("model") or "")
            row = {
                "scale": _classify_model_scale(model_name),
                "name": payload.get("name"),
                "model": model_name,
                "source_dir": str(payload.get("source_dir") or train_payload.get("output_dir") or ""),
                "total_minutes": _parse_duration_to_minutes(train_payload.get("total_time")),
                "train_minutes": _parse_duration_to_minutes(train_payload.get("train_time")),
                "val_minutes": _parse_duration_to_minutes(train_payload.get("val_time")),
                "start_time": payload.get("start_time"),
                "end_time": payload.get("end_time"),
            }
            rows.append(row)
            source_dir = Path(row["source_dir"]).expanduser().resolve() if row["source_dir"] else None
            if source_dir == expected_dir:
                current_row = row

    if current_row is None:
        model_name = str(train_info.get("args", {}).get("model") or "")
        current_row = {
            "scale": _classify_model_scale(model_name),
            "name": expected_dir.name,
            "model": model_name,
            "source_dir": str(expected_dir),
            "total_minutes": train_info.get("total_minutes"),
            "train_minutes": train_info.get("train_minutes"),
            "val_minutes": train_info.get("val_minutes"),
            "start_time": train_info.get("start_time"),
            "end_time": train_info.get("end_time"),
        }

    current_scale = current_row.get("scale") or "UNKNOWN"
    same_scale_rows = [row for row in rows if row.get("scale") == current_scale]
    total_values = [float(row["total_minutes"]) for row in same_scale_rows if isinstance(row.get("total_minutes"), (int, float))]
    train_values = [float(row["train_minutes"]) for row in same_scale_rows if isinstance(row.get("train_minutes"), (int, float))]
    val_values = [float(row["val_minutes"]) for row in same_scale_rows if isinstance(row.get("val_minutes"), (int, float))]

    latest_same_scale = None
    if same_scale_rows:
        latest_same_scale = max(same_scale_rows, key=lambda item: str(item.get("start_time") or ""))

    return {
        "current": current_row,
        "same_scale": {
            "scale": current_scale,
            "count": len(same_scale_rows),
            "latest": latest_same_scale,
            "avg_total_minutes": sum(total_values) / len(total_values) if total_values else None,
            "avg_train_minutes": sum(train_values) / len(train_values) if train_values else None,
            "avg_val_minutes": sum(val_values) / len(val_values) if val_values else None,
        },
    }


def _build_time_occupancy_section(train_info: dict[str, Any], benchmark_summary: dict[str, Any]) -> list[str]:
    current = benchmark_summary.get("current") or {}
    same_scale = benchmark_summary.get("same_scale") or {}
    train_step_text = "-"
    if isinstance(train_info.get("train_seconds_per_step"), (int, float)):
        train_step_text = f"{float(train_info['train_seconds_per_step']):.2f}"
    val_step_text = "-"
    if isinstance(train_info.get("val_seconds_per_step"), (int, float)):
        val_step_text = f"{float(train_info['val_seconds_per_step']):.2f}"
    lines = [
        "### 3. 训练耗时与占用分析",
        f"- 起止时间：{train_info.get('start_time') or '-'} -> {train_info.get('end_time') or '-'}",
        f"- 总耗时：{train_info.get('total_time') or _format_minutes(current.get('total_minutes'))}",
        f"- 训练阶段：{train_info.get('train_time') or _format_minutes(current.get('train_minutes'))}",
        f"- 验证阶段：{train_info.get('val_time') or _format_minutes(current.get('val_minutes'))}",
        f"- 训练吞吐：{train_info.get('train_throughput') or '-'}",
        f"- 验证吞吐：{train_info.get('val_throughput') or '-'}",
        "",
        "| 维度 | 当前实验 | 同尺度参考 |",
        "|---|---|---|",
        f"| 模型尺度 | {current.get('scale') or '-'} | {same_scale.get('scale') or '-'} |",
        f"| 总耗时 | {_format_minutes(current.get('total_minutes'))} | {_format_minutes(same_scale.get('avg_total_minutes'))} |",
        f"| 训练耗时 | {_format_minutes(current.get('train_minutes'))} | {_format_minutes(same_scale.get('avg_train_minutes'))} |",
        f"| 验证耗时 | {_format_minutes(current.get('val_minutes'))} | {_format_minutes(same_scale.get('avg_val_minutes'))} |",
        f"| 训练占比 | {str(train_info.get('train_ratio_percent')) + '%' if train_info.get('train_ratio_percent') is not None else '-'} | - |",
        f"| 验证占比 | {str(train_info.get('val_ratio_percent')) + '%' if train_info.get('val_ratio_percent') is not None else '-'} | - |",
        f"| Train s/step | {train_step_text} | - |",
        f"| Val s/step | {val_step_text} | - |",
        "",
        f"- 同尺度样本数：{same_scale.get('count', 0)}",
    ]
    latest = same_scale.get("latest") or {}
    if latest:
        lines.append(
            f"- 同尺度最近记录：{latest.get('name') or '-'}，总耗时 {_format_minutes(latest.get('total_minutes'))}，训练耗时 {_format_minutes(latest.get('train_minutes'))}。"
        )
    lines.append("")
    return lines


def _find_report_path(output_dir: Path, run_meta: dict[str, Any]) -> Path | None:
    report_path = _resolve_optional_path(run_meta.get("paths", {}).get("report_path"))
    if report_path is not None and report_path.exists():
        return report_path
    split_name = str(run_meta.get("split") or "").strip().lower()
    candidates = sorted(output_dir.glob("*_report.json"))
    if candidates:
        prioritized = []
        for candidate in candidates:
            score = 0
            name = candidate.name.lower()
            if name.endswith("_report.json"):
                score += 10
            if split_name and f"-{split_name}-" in name:
                score += 6
            if split_name and name.endswith(f"-{split_name}_report.json"):
                score += 8
            if split_name and split_name in name:
                score += 2
            prioritized.append((score, candidate.stat().st_mtime, candidate))
        prioritized.sort(reverse=True)
        return prioritized[0][2]
    fallback = output_dir / "test_report.json"
    if fallback.exists():
        return fallback
    return None


def _discover_infer_runs(experiment_dir: Path) -> list[InferRunRecord]:
    runs: list[InferRunRecord] = []
    expected_experiment_dir = experiment_dir.expanduser().resolve()
    search_roots: list[Path] = [expected_experiment_dir]
    if rt.TEST_OUTPUT_ROOT_DIR.exists() and rt.TEST_OUTPUT_ROOT_DIR != expected_experiment_dir:
        search_roots.append(rt.TEST_OUTPUT_ROOT_DIR)

    seen_run_meta_paths: set[Path] = set()
    for run_meta_path in find_files(
        search_roots,
        label="索引 infer/eval 记录",
        filenames={"run_meta.json"},
    ):
        resolved_run_meta_path = run_meta_path.resolve()
        if resolved_run_meta_path in seen_run_meta_paths:
            continue
        seen_run_meta_paths.add(resolved_run_meta_path)
        run_meta = _load_json(run_meta_path)
        if run_meta is None:
            continue
        if run_meta.get("task") != "det" or run_meta.get("action") not in {"eval", "infer"}:
            continue
        paths_payload = run_meta.get("paths", {})
        recorded_experiment_dir = _resolve_optional_path(paths_payload.get("experiment_dir"))
        try:
            physically_inside = resolved_run_meta_path.is_relative_to(expected_experiment_dir)
        except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
            physically_inside = expected_experiment_dir in resolved_run_meta_path.parents
        if not physically_inside and recorded_experiment_dir != expected_experiment_dir:
            continue

        output_dir = run_meta_path.parent.resolve()
        report_path = _find_report_path(output_dir, run_meta)
        metrics_path = _resolve_optional_path(run_meta.get("artifacts", {}).get("metrics_summary"))
        if metrics_path is None or not metrics_path.exists():
            candidate = output_dir / "metrics_summary.json"
            metrics_path = candidate if candidate.exists() else None

        report_payload = _load_json(report_path)
        metrics_payload = _load_json(metrics_path)
        if report_payload is None and metrics_payload is None:
            continue
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
            item.run_meta.get("action") == "eval",
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
    default_headers = ["类别名称 (Label)", "GT", "Pred", "Precision@0.5", "Recall@0.5", "AP@0.5"]
    default_align = ["---", "---:", "---:", "---:", "---:", "---:"]
    if not isinstance(per_class_ap, dict) or not per_class_ap:
        return (
            f"| {' | '.join(default_headers)} |\n"
            f"| {' | '.join(default_align)} |\n"
            "| 当前 split 缺少按类结果 | - | - | - |"
        )

    column_specs = [
        ("name", "类别名称 (Label)", lambda value, class_id: str(value) if isinstance(value, str) and value.strip() else f"class_{class_id}", "---"),
        ("gt", "GT", lambda value, _class_id: _format_count(value), "---:"),
        ("pred", "Pred", lambda value, _class_id: _format_count(value), "---:"),
        ("precision", "Precision@0.5", lambda value, _class_id: _format_metric(value), "---:"),
        ("recall", "Recall@0.5", lambda value, _class_id: _format_metric(value), "---:"),
        ("ap", "AP@0.5", lambda value, _class_id: _format_metric(value), "---:"),
    ]

    entries: list[tuple[int, dict[str, Any]]] = []
    for class_id_raw, info in per_class_ap.items():
        try:
            class_id = int(class_id_raw)
        except ValueError:
            continue
        if isinstance(info, dict):
            entries.append((class_id, info))

    present_columns = []
    for key, header, formatter, align in column_specs:
        if any(key in info for _, info in entries):
            present_columns.append((key, header, formatter, align))
    if not present_columns:
        present_columns = column_specs

    entries.sort(
        key=lambda item: (
            float(item[1]["ap"]) if isinstance(item[1].get("ap"), (int, float)) else float("inf"),
            item[0],
        )
    )

    headers = [header for _, header, _, _ in present_columns]
    aligns = [align for _, _, _, align in present_columns]
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(aligns)} |",
    ]
    for class_id, info in entries:
        row = []
        for key, _header, formatter, _align in present_columns:
            row.append(formatter(info.get(key), class_id))
        lines.append(
            "| " + " | ".join(row) + " |"
        )
    return "\n".join(lines)


def _build_per_class_sections(runs: list[InferRunRecord]) -> list[str]:
    sections: list[str] = []
    for run in runs:
        sections.extend(
            [
                f"#### {run.split}",
                "",
                _build_per_class_table(run),
                "",
            ]
        )
    if not sections:
        sections.extend(
            [
                "#### test",
                "",
                "| 类别名称 (Label) | GT | Pred | AP@0.5 |",
                "|---|---:|---:|---:|",
                "| 当前没有可用的按类结果 | - | - | - |",
                "",
            ]
        )
    return sections


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
    backbone_name = _extract_backbone_name(args_payload)
    batch_size = args_payload.get("batch_size")
    steps = args_payload.get("steps")
    class_count = len(class_names) if class_names else "-"
    epoch_text = _build_epoch_text(steps, batch_size, counts.get("train", 0))
    data_root_text = f"`{_display_path(data_root)}`" if data_root is not None else "待补充"
    class_list_text = _format_class_list(class_names, limit=5)
    lr_text = train_info.get("lr_text") or _build_lr_text(args_payload)
    hardware_text = train_info.get("hardware_summary") or "待补充"
    total_time_text = train_info.get("total_time") or "待补充"
    train_time_text = train_info.get("train_time") or "待补充"
    val_time_text = train_info.get("val_time") or "待补充"
    conclusion_lines = _build_conclusion_lines(primary_run, train_info)
    benchmark_summary = _load_training_benchmark_summary(experiment_dir, train_info)

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
    ]
    lines.extend(_build_time_occupancy_section(train_info, benchmark_summary))
    lines.extend(
        [
            "### 4. 基本参数设置",
            "",
            "| 参数项 | 内容 |",
            "|---|---|",
            f"| 模型结构 (Model) | {model_name} |",
            f"| 骨干网 (Backbone) | {backbone_name} |",
            f"| Epoch / Iters | {epoch_text} |",
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
            "### 2. 具体类别（Label）准确度",
            "> 每个类别的指标直接来自对应 split 的 `test_report.per_class_ap`，其中 Precision@0.5 和 Recall@0.5 由该类别的 TP、Pred、GT 计算，并按 AP@0.5 从低到高排序。",
            "",
        ]
    )
    lines.extend(_build_per_class_sections(split_runs))
    lines.extend(
        [
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
    )
    for item in conclusion_lines:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## 五、附件",
            "",
            "- 训练总览图：`training_dashboard.png`",
            f"- 当前主结果目录：`{_display_path(primary_run.output_dir)}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def generate_experiment_report(
    *,
    experiment_dir: Path,
    output_dir: Path | None = None,
    only_output_dir: Path | None = None,
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

    split_runs = _latest_runs_by_split(infer_runs)
    train_info = _parse_train_log(resolved_experiment_dir)

    if output_dir is not None:
        target_dirs = [output_dir.expanduser().resolve()]
    else:
        target_dirs = [rt.experiment_important_dir(resolved_experiment_dir)]

    if only_output_dir is not None:
        resolved_only_output_dir = only_output_dir.expanduser().resolve()
        report_runs = [run for run in infer_runs if run.output_dir.expanduser().resolve() == resolved_only_output_dir]
    else:
        report_runs = [run for run in split_runs if run.split in {"test", "val"}]
    if not report_runs:
        report_runs = [_pick_primary_run(infer_runs)]

    output_paths: list[Path] = []
    for target_dir in target_dirs:
        for primary_run in report_runs:
            report_text = _render_report(
                experiment_dir=resolved_experiment_dir,
                primary_run=primary_run,
                split_runs=split_runs,
                train_info=train_info,
            )
            report_filename = _build_report_filename(
                primary_run,
                train_info,
                split_name=primary_run.split,
            )
            report_path = target_dir / report_filename
            output_paths.append(report_path)
            if dry_run:
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report_text, encoding="utf-8")
    if not dry_run and any(target_dir == rt.experiment_important_dir(resolved_experiment_dir) for target_dir in target_dirs):
        rt.sync_important_to_all_report(resolved_experiment_dir)
    return output_paths


def generate_report_for_infer_output(*, experiment_dir: Path, output_dir: Path) -> list[Path]:
    return generate_experiment_report(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        only_output_dir=output_dir,
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
