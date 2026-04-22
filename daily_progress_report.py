from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent

TIMESTAMP_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]")
TRAIN_STEP_RE = re.compile(
    r"Train Step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+(?:train_loss|Train Loss):\s+(?P<loss>[-+]?\d*\.?\d+)(?:\s+\|\s+lr:\s+(?P<lr>[-+]?\d*\.?\d+))?"
)
CLS_VAL_STEP_RE = re.compile(
    r"Val Step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+val_loss:\s+(?P<loss>[-+]?\d*\.?\d+)"
)
DET_VAL_STEP_RE = re.compile(
    r"Val Step\s+(?P<step>\d+)/(?P<total>\d+)\s+\|\s+Val Loss:\s+(?P<loss>[-+]?\d*\.?\d+)\s+\|\s+Val mAP@0\.5:0\.95:\s+(?P<map>[-+]?\d*\.?\d+)\s+\|\s+Val mAP@0\.5:\s+(?P<map50>[-+]?\d*\.?\d+)"
)
BEST_METRIC_RE = re.compile(
    r"(?:The best validation metric|Best result:)\s+(?P<metric>[\w/@.]+)=(?P<value>[-+]?\d*\.?\d+)"
)
TRAIN_IMAGES_RE = re.compile(r"Train images:\s*(?P<train>\d+),\s*Val images:\s*(?P<val>\d+)")
PROFILE_TABLE_RE = re.compile(r"\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|")
KV_RE = re.compile(r"^\s*(?P<key>[^:|]+?)\s*:\s*(?P<value>.+?)\s*$")


@dataclass
class Record:
    record_kind: str
    report_date: str
    task: str
    name: str
    source_dir: str
    source_file: str
    start_time: str | None = None
    end_time: str | None = None
    status: str = "unknown"
    dataset: dict[str, Any] = field(default_factory=dict)
    train: dict[str, Any] = field(default_factory=dict)
    tests: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总训练日志和测试结果，生成日报与结构化记录。")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="训练与测试产物根目录，默认 out/")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT_DIR / "daily_report",
        help="日报输出根目录，默认 <repo>/daily_report/",
    )
    parser.add_argument("--date", default="all", help="仅导出指定日期，格式 YYYY-MM-DD，也支持 today；默认 all")
    return parser.parse_args()


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def extract_last_args_block(lines: list[str]) -> dict[str, Any]:
    start_index = None
    for idx, line in enumerate(lines):
        if "Args:" in line:
            start_index = idx
    if start_index is None:
        return {}

    json_lines: list[str] = []
    brace_count = 0
    started = False
    for line in lines[start_index:]:
        payload = line.split("Args:", 1)[1] if not started and "Args:" in line else line.split("] ", 1)[-1]
        if not started:
            started = True
        json_lines.append(payload)
        brace_count += payload.count("{") - payload.count("}")
        if brace_count <= 0 and "}" in payload:
            break

    raw = "\n".join(json_lines).strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def normalize_task(task: str | None, name: str) -> str:
    lowered = (task or "").lower()
    if lowered:
        return lowered
    name_lower = name.lower()
    if "cls" in name_lower:
        return "image_classification"
    if "seg" in name_lower:
        return "segmentation"
    if "det" in name_lower:
        return "object_detection"
    return "unknown"


def report_date_from(dt: datetime | None, fallback_path: Path) -> str:
    if dt is not None:
        return dt.strftime("%Y-%m-%d")
    return datetime.fromtimestamp(fallback_path.stat().st_mtime).strftime("%Y-%m-%d")


def extract_dataset_summary(args_block: dict[str, Any], task: str) -> dict[str, Any]:
    data_cfg = args_block.get("data")
    if isinstance(data_cfg, dict):
        summary: dict[str, Any] = {}
        for key in ("train", "val", "test", "path"):
            if key in data_cfg and data_cfg[key] is not None:
                summary[key] = data_cfg[key]
        if "classes" in data_cfg and isinstance(data_cfg["classes"], dict):
            summary["num_classes"] = len(data_cfg["classes"])
        if "names" in data_cfg and isinstance(data_cfg["names"], dict):
            summary["num_classes"] = len(data_cfg["names"])
        if task == "image_classification":
            summary["task"] = "cls"
        elif task == "object_detection":
            summary["task"] = "det"
        return summary
    return {}


def pick_keys(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: data[key] for key in keys if key in data and data[key] is not None}


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def build_train_params(args_block: dict[str, Any]) -> dict[str, Any]:
    if not args_block:
        return {}
    key_params = pick_keys(
        args_block,
        [
            "model",
            "batch_size",
            "steps",
            "devices",
            "accelerator",
            "precision",
            "strategy",
            "num_workers",
            "num_nodes",
            "seed",
            "checkpoint",
            "overwrite",
            "resume_interrupted",
            "reuse_class_head",
        ],
    )
    optional_keys = pick_keys(
        args_block,
        [
            "gradient_accumulation_steps",
            "float32_matmul_precision",
            "checkpoint",
        ],
    )
    model_args = args_block.get("model_args") or {}
    logger_args = args_block.get("logger_args") or {}
    save_checkpoint_args = args_block.get("save_checkpoint_args") or {}
    optional_nested = {
        key: value
        for key, value in {
            "backbone_weights": model_args.get("backbone_weights"),
            "lr": model_args.get("lr"),
            "weight_decay": model_args.get("weight_decay"),
            "ema_momentum": model_args.get("ema_momentum"),
            "use_ema_model": model_args.get("use_ema_model"),
            "loss_alpha": model_args.get("loss_alpha"),
            "loss_gamma": model_args.get("loss_gamma"),
            "val_every_num_steps": logger_args.get("val_every_num_steps"),
            "log_every_num_steps": logger_args.get("log_every_num_steps"),
            "val_log_every_num_steps": logger_args.get("val_log_every_num_steps"),
            "save_best": save_checkpoint_args.get("save_best"),
            "save_last": save_checkpoint_args.get("save_last"),
            "save_every_num_steps": save_checkpoint_args.get("save_every_num_steps"),
            "watch_metric": save_checkpoint_args.get("watch_metric"),
        }.items()
        if value is not None
    }
    return {
        "key": key_params,
        "extra": {**optional_keys, **optional_nested},
    }


def build_test_params_from_meta(meta: dict[str, Any], report_data: dict[str, Any] | None) -> dict[str, Any]:
    key_params: dict[str, Any] = {}
    extra_params: dict[str, Any] = {}
    settings = meta.get("settings", {})
    paths = meta.get("paths", {})
    config = (report_data or {}).get("config", {})
    key_params.update(
        {
            "input_mode": meta.get("input_mode"),
            "split": meta.get("split"),
            "score_threshold": settings.get("score_threshold", config.get("conf_threshold")),
            "report_iou_threshold": settings.get("report_iou_threshold", config.get("iou_threshold")),
            "checkpoint_path": paths.get("checkpoint_path", config.get("model_path")),
            "data_yaml": paths.get("data_yaml"),
        }
    )
    extra_params.update(
        {
            "device": settings.get("device"),
            "save_visualization": settings.get("save_visualization"),
            "save_json": settings.get("save_json"),
            "save_txt": settings.get("save_txt"),
            "compute_metrics": settings.get("compute_metrics"),
            "metric_classwise": settings.get("metric_classwise"),
            "data_root": paths.get("data_root", config.get("data_root")),
        }
    )
    return {
        "key": {key: value for key, value in key_params.items() if value is not None},
        "extra": {key: value for key, value in extra_params.items() if value is not None},
    }


def build_test_params_from_report(report_data: dict[str, Any] | None) -> dict[str, Any]:
    config = (report_data or {}).get("config", {})
    return {
        "key": {
            key: value
            for key, value in {
                "checkpoint_path": config.get("model_path"),
                "split": config.get("split"),
                "score_threshold": config.get("conf_threshold"),
                "report_iou_threshold": config.get("iou_threshold"),
            }.items()
            if value is not None
        },
        "extra": {
            key: value
            for key, value in {
                "data_root": config.get("data_root"),
                "test_images": config.get("test_images"),
                "test_labels": config.get("test_labels"),
            }.items()
            if value is not None
        },
    }


def compact_json_lines(payload: dict[str, Any], indent: int = 2) -> list[str]:
    if not payload:
        return []
    return json.dumps(payload, ensure_ascii=False, indent=indent).splitlines()


def format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    if not rows:
        return []
    lines = [
        f"| {' | '.join(headers)} |",
        f"| {' | '.join(['---'] * len(headers))} |",
    ]
    for row in rows:
        lines.append(f"| {' | '.join(format_scalar(cell) for cell in row)} |")
    return lines


def append_test(record: Record, test_summary: dict[str, Any]) -> None:
    record.tests.append(test_summary)


def dataset_name_from_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    parts = list(path.parts)
    if "datasets" in parts:
        idx = parts.index("datasets")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name or path.name


def classify_test_target(test_summary: dict[str, Any]) -> str:
    params = test_summary.get("params", {})
    key_params = params.get("key", {}) if isinstance(params, dict) else {}
    split = key_params.get("split") or "unknown"
    data_yaml = key_params.get("data_yaml")
    extra_params = params.get("extra", {}) if isinstance(params, dict) else {}
    data_root = extra_params.get("data_root")
    dataset_name = dataset_name_from_path(data_yaml or data_root) or "unknown_dataset"
    return f"{dataset_name} / {split}"


def auto_notes_for_record(record: Record) -> list[str]:
    notes = list(record.notes)
    if record.record_kind == "train_run" and not record.tests:
        notes.append("该训练已发现，但暂未关联到测试结果。")
    for test in record.tests:
        if test.get("kind") == "classification_csv":
            acc = test.get("accuracy")
            if isinstance(acc, (int, float)) and acc < 0.6:
                notes.append("分类准确率低于 0.6000，建议复查数据划分、类别平衡和阈值设置。")
        if test.get("kind") == "detection_report":
            map50 = test.get("map_50")
            map05 = test.get("map_05")
            if isinstance(map50, (int, float)) and map50 == 0.0:
                notes.append("检测 mAP@0.5 为 0，建议优先检查标签类别映射、阈值、权重是否匹配当前数据集。")
            elif isinstance(map05, (int, float)) and map05 < 0.05:
                notes.append("检测指标偏低，建议结合可视化结果排查漏检和类别混淆。")
    return notes


def parse_profile_value(lines: list[str], target: str) -> str | None:
    for line in reversed(lines):
        match = PROFILE_TABLE_RE.search(line)
        if not match:
            kv_match = KV_RE.search(line.split("] ", 1)[-1])
            if not kv_match:
                continue
            name = kv_match.group("key").strip().lower()
            if name == target.lower():
                return kv_match.group("value").strip()
            continue
        name = match.group("name").strip().lower()
        if name == target.lower():
            return match.group("value").strip()
    return None


def parse_train_log(log_path: Path) -> Record:
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    args_block = extract_last_args_block(lines)
    task = normalize_task(args_block.get("task"), log_path.parent.name)

    timestamps = [ts for line in lines if (ts := parse_timestamp(line)) is not None]
    start_time = timestamps[0].isoformat(sep=" ") if timestamps else None
    end_time = timestamps[-1].isoformat(sep=" ") if timestamps else None

    dataset = extract_dataset_summary(args_block, task)
    train_summary: dict[str, Any] = {
        "log_path": str(log_path),
        "output_dir": args_block.get("out"),
        "model": args_block.get("model"),
        "steps": args_block.get("steps"),
        "batch_size": args_block.get("batch_size"),
        "devices": args_block.get("devices"),
        "accelerator": args_block.get("accelerator"),
        "params": build_train_params(args_block),
    }

    for line in lines:
        if (match := TRAIN_IMAGES_RE.search(line)) is not None:
            dataset["train_images"] = int(match.group("train"))
            dataset["val_images"] = int(match.group("val"))
        if (match := TRAIN_STEP_RE.search(line)) is not None:
            train_summary["last_train_step"] = int(match.group("step"))
            train_summary["train_total_steps"] = int(match.group("total"))
            train_summary["last_train_loss"] = float(match.group("loss"))
            lr_value = match.group("lr")
            if lr_value is not None:
                train_summary["last_lr"] = float(lr_value)
        if (match := CLS_VAL_STEP_RE.search(line)) is not None:
            train_summary["last_val_step"] = int(match.group("step"))
            train_summary["val_total_steps"] = int(match.group("total"))
            train_summary["last_val_loss"] = float(match.group("loss"))
        if (match := DET_VAL_STEP_RE.search(line)) is not None:
            train_summary["last_val_step"] = int(match.group("step"))
            train_summary["val_total_steps"] = int(match.group("total"))
            train_summary["last_val_loss"] = float(match.group("loss"))
            train_summary["last_val_map"] = float(match.group("map"))
            train_summary["last_val_map50"] = float(match.group("map50"))
        if (match := BEST_METRIC_RE.search(line)) is not None:
            train_summary["best_metric_name"] = match.group("metric")
            train_summary["best_metric_value"] = float(match.group("value"))

    for line in lines:
        if not ("[INFO]" in line and "| " in line):
            continue
        table_match = PROFILE_TABLE_RE.search(line)
        if not table_match:
            continue
        metric_name = table_match.group("name").strip()
        metric_value = table_match.group("value").strip()
        if metric_name.startswith("val_metric/") or metric_name == "val_loss":
            train_summary[metric_name.replace("/", "_")] = metric_value

    total_time = parse_profile_value(lines, "Total Time")
    train_time = parse_profile_value(lines, "Train Time")
    val_time = parse_profile_value(lines, "Val Time")
    if total_time:
        train_summary["total_time"] = total_time
    if train_time:
        train_summary["train_time"] = train_time
    if val_time:
        train_summary["val_time"] = val_time

    status = "completed" if any("Training completed." in line for line in lines) else "in_progress"
    if any("Exporting the best model" in line or "Exporting the last model" in line for line in lines):
        status = "completed"

    record = Record(
        record_kind="train_run",
        report_date=report_date_from(timestamps[-1] if timestamps else None, log_path),
        task=task,
        name=log_path.parent.name,
        source_dir=str(log_path.parent),
        source_file=str(log_path),
        start_time=start_time,
        end_time=end_time,
        status=status,
        dataset=dataset,
        train=train_summary,
    )
    attach_related_test_results(record, log_path.parent)
    return record


def attach_related_test_results(record: Record, run_dir: Path) -> None:
    cls_csv = run_dir / "test_results.csv"
    det_report = run_dir / "test_report.json"
    det_metrics = run_dir / "metrics_summary.json"

    if cls_csv.exists():
        append_test(record, parse_cls_test_csv(cls_csv))
        return

    report_data = load_json(det_report)
    metrics_data = load_json(det_metrics)
    if report_data or metrics_data:
        append_test(record, parse_det_results(det_report, report_data, metrics_data))


def attach_det_runs_from_meta(records: list[Record], out_dir: Path) -> list[Record]:
    record_by_source_dir = {Path(record.source_dir).resolve(): record for record in records if record.record_kind == "train_run"}
    consumed_meta_dirs: set[Path] = set()

    for meta_path in sorted(out_dir.rglob("run_meta.json")):
        meta = load_json(meta_path)
        if not meta:
            continue
        meta_dir = meta_path.parent.resolve()
        consumed_meta_dirs.add(meta_dir)
        paths = meta.get("paths", {})
        experiment_dir_raw = paths.get("experiment_dir")
        report_path_raw = paths.get("report_path")
        metrics_path_raw = (meta.get("artifacts") or {}).get("metrics_summary")
        if not report_path_raw and not metrics_path_raw:
            continue

        report_path = Path(report_path_raw) if report_path_raw else meta_dir / f"{meta_dir.name}-test_report.json"
        metrics_path = Path(metrics_path_raw) if metrics_path_raw else meta_dir / "metrics_summary.json"
        report_data = load_json(report_path)
        metrics_data = load_json(metrics_path)
        test_summary = parse_det_results(report_path, report_data, metrics_data)
        test_summary["run_meta"] = str(meta_path)
        test_summary["output_dir"] = str(meta_dir)
        test_summary["input_mode"] = meta.get("input_mode")
        test_summary["run_name"] = meta.get("run_name")
        test_summary["params"] = build_test_params_from_meta(meta, report_data)

        linked = False
        if experiment_dir_raw:
            experiment_dir = Path(experiment_dir_raw).resolve()
            record = record_by_source_dir.get(experiment_dir)
            if record is not None:
                append_test(record, test_summary)
                record.notes.append(f"linked infer run: {meta_dir}")
                linked = True

        if linked:
            continue

        created_at = meta.get("created_at")
        dt = None
        if isinstance(created_at, str):
            try:
                dt = datetime.fromisoformat(created_at)
            except ValueError:
                dt = None
        if dt is None:
            dt = datetime.fromtimestamp(meta_path.stat().st_mtime)
        records.append(
            Record(
                record_kind="test_only",
                report_date=dt.strftime("%Y-%m-%d"),
                task="object_detection",
                name=meta.get("run_name") or meta_dir.name,
                source_dir=str(meta_dir),
                source_file=str(meta_path),
                start_time=dt.isoformat(sep=" "),
                end_time=dt.isoformat(sep=" "),
                status="completed",
                dataset={
                    "data_yaml": paths.get("data_yaml"),
                    "data_root": paths.get("data_root"),
                    "split": meta.get("split"),
                    "test_images": (report_data or {}).get("config", {}).get("test_images"),
                },
                test=test_summary,
                notes=[f"discovered from run_meta: {meta_path.name}"],
            )
        )

    for path in sorted(out_dir.rglob("*-test_report.json")):
        if "test_reports" in path.parts:
            continue
        if path.parent.resolve() in consumed_meta_dirs:
            continue
        record = parse_standalone_test_result(path)
        if record is not None:
            records.append(record)

    return records


def parse_cls_test_csv(csv_path: Path) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)

    total = len(rows)
    correct = sum(int(row.get("is_correct", "0") or 0) for row in rows)
    per_class: dict[str, dict[str, int | float]] = {}
    for row in rows:
        label = row.get("true_label_name", "UNKNOWN")
        stats = per_class.setdefault(label, {"total": 0, "correct": 0, "accuracy": 0.0})
        stats["total"] = int(stats["total"]) + 1
        stats["correct"] = int(stats["correct"]) + int(row.get("is_correct", "0") or 0)

    for label, stats in per_class.items():
        total_count = int(stats["total"])
        correct_count = int(stats["correct"])
        stats["accuracy"] = round(correct_count / total_count, 4) if total_count else 0.0
        per_class[label] = stats

    return {
        "kind": "classification_csv",
        "source_file": str(csv_path),
        "evaluated_images": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "per_class": per_class,
    }


def parse_det_results(
    report_path: Path,
    report_data: dict[str, Any] | None,
    metrics_data: dict[str, Any] | None,
) -> dict[str, Any]:
    summary = (report_data or {}).get("summary", {})
    metrics = (metrics_data or {}).get("metrics", {})
    per_class_ap = (report_data or {}).get("per_class_ap", {})

    top_classes = sorted(
        (
            {
                "class_id": class_id,
                "name": item.get("name"),
                "ap": item.get("ap"),
                "gt": item.get("gt"),
                "pred": item.get("pred"),
            }
            for class_id, item in per_class_ap.items()
            if isinstance(item, dict)
        ),
        key=lambda item: item.get("ap") if isinstance(item.get("ap"), (int, float)) else -1.0,
        reverse=True,
    )[:5]

    return {
        "kind": "detection_report",
        "source_file": str(report_path),
        "num_images": summary.get("num_images"),
        "processed_images": summary.get("processed_images"),
        "precision_05": summary.get("precision_05"),
        "recall_05": summary.get("recall_05"),
        "map_05": summary.get("map_05"),
        "map": metrics.get("eval_metric/map"),
        "map_50": metrics.get("eval_metric/map_50"),
        "map_75": metrics.get("eval_metric/map_75"),
        "avg_infer_time_ms": summary.get("avg_infer_time_ms") or (metrics_data or {}).get("avg_infer_time_ms"),
        "top_classes": top_classes,
    }


def parse_standalone_test_result(path: Path) -> Record | None:
    parent = path.parent
    if path.name == "test_results.csv":
        test_summary = parse_cls_test_csv(path)
        dt = datetime.fromtimestamp(path.stat().st_mtime)
        return Record(
            record_kind="test_only",
            report_date=dt.strftime("%Y-%m-%d"),
            task="image_classification",
            name=parent.name,
            source_dir=str(parent),
            source_file=str(path),
            start_time=dt.isoformat(sep=" "),
            end_time=dt.isoformat(sep=" "),
            status="completed",
            tests=[test_summary],
        )

    if path.name == "test_report.json" or path.name.endswith("-test_report.json"):
        report_data = load_json(path)
        if not report_data:
            return None
        metrics_path = parent / "metrics_summary.json"
        test_summary = parse_det_results(path, report_data, load_json(metrics_path))
        test_summary["params"] = build_test_params_from_report(report_data)
        dt = datetime.fromtimestamp(path.stat().st_mtime)
        return Record(
            record_kind="test_only",
            report_date=dt.strftime("%Y-%m-%d"),
            task="object_detection",
            name=parent.name,
            source_dir=str(parent),
            source_file=str(path),
            start_time=dt.isoformat(sep=" "),
            end_time=dt.isoformat(sep=" "),
            status="completed",
            dataset={"test_images": report_data.get("config", {}).get("test_images")},
            tests=[test_summary],
        )
    return None


def discover_records(out_dir: Path) -> list[Record]:
    records: list[Record] = []
    seen_test_dirs: set[Path] = set()

    for log_path in sorted(out_dir.rglob("train.log")):
        try:
            record = parse_train_log(log_path)
        except Exception as exc:
            record = Record(
                record_kind="train_run",
                report_date=report_date_from(None, log_path),
                task="unknown",
                name=log_path.parent.name,
                source_dir=str(log_path.parent),
                source_file=str(log_path),
                status="parse_failed",
                notes=[f"failed to parse train log: {exc}"],
            )
        records.append(record)
        seen_test_dirs.add(log_path.parent.resolve())

    for path in sorted(out_dir.rglob("test_results.csv")):
        if path.parent.resolve() in seen_test_dirs:
            continue
        record = parse_standalone_test_result(path)
        if record is not None:
            records.append(record)

    records = attach_det_runs_from_meta(records, out_dir)

    for path in sorted(out_dir.rglob("test_report.json")):
        if "test_reports" in path.parts:
            continue
        if path.parent.resolve() in seen_test_dirs:
            continue
        if any(Path(record.source_dir).resolve() == path.parent.resolve() and record.task == "object_detection" for record in records):
            continue
        record = parse_standalone_test_result(path)
        if record is not None:
            records.append(record)

    records.sort(key=lambda item: (item.report_date, item.end_time or "", item.name))
    return records


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[Record]) -> None:
    lines = [json.dumps(asdict(record), ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_notes_template(path: Path, report_date: str, items: list[Record]) -> None:
    if path.exists():
        return
    task_names = sorted({item.task for item in items})
    lines = [
        f"# 工作补充笔记 - {report_date}",
        "",
        "## 今日结论",
        "",
        "- ",
        "",
        "## 问题记录",
        "",
        "- ",
        "",
        "## 明日计划",
        "",
        "- ",
        "",
        "## 备注",
        "",
        f"- 今日涉及任务: {', '.join(task_names)}",
        f"- 今日记录数: {len(items)}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def report_day_dir(report_root: Path, report_date: str) -> Path:
    return report_root / report_date


def build_train_group_payload(record: Record) -> dict[str, Any]:
    params = record.train.get("params", {})
    return {
        "name": record.name,
        "task": record.task,
        "status": record.status,
        "time_range": {
            "start": record.start_time,
            "end": record.end_time,
        },
        "dataset": record.dataset,
        "train_summary": {
            key: value
            for key, value in record.train.items()
            if key
            not in {
                "params",
                "log_path",
                "output_dir",
            }
        },
        "hyper_params": {
            "key": params.get("key", {}) if isinstance(params, dict) else {},
            "extra": params.get("extra", {}) if isinstance(params, dict) else {},
        },
        "tests": record.tests,
        "notes": auto_notes_for_record(record),
    }


def format_train_section(record: Record) -> list[str]:
    lines: list[str] = []
    train = record.train
    dataset = record.dataset
    lines.append("### 训练信息")
    lines.append(f"- 任务: `{record.task}`")
    if dataset:
        pieces = []
        for key in ("train", "val", "test", "path"):
            if dataset.get(key):
                pieces.append(f"{key}=`{dataset[key]}`")
        if dataset.get("train_images") is not None or dataset.get("val_images") is not None:
            pieces.append(f"images(train/val)={dataset.get('train_images', '-')}/{dataset.get('val_images', '-')}")
        if dataset.get("num_classes") is not None:
            pieces.append(f"classes={dataset['num_classes']}")
        if pieces:
            lines.append(f"- 数据集: {', '.join(pieces)}")
    summary_bits = []
    if train.get("model"):
        summary_bits.append(f"model=`{train['model']}`")
    if train.get("batch_size") is not None:
        summary_bits.append(f"batch_size={train['batch_size']}")
    if train.get("steps") is not None:
        summary_bits.append(f"steps={train['steps']}")
    if train.get("devices") is not None:
        summary_bits.append(f"devices={train['devices']}")
    if summary_bits:
        lines.append(f"- 训练概览: {', '.join(summary_bits)}")

    train_table_rows = [[
        train.get("model", "-"),
        train.get("steps", "-"),
        train.get("batch_size", "-"),
        train.get("last_train_loss", "-"),
        train.get("last_val_loss", "-"),
        train.get("last_val_map", train.get("val_metric_top1_acc_micro", "-")),
        train.get("best_metric_value", "-"),
    ]]
    lines.append("")
    lines.append("### 训练结果表")
    lines.extend(
        markdown_table(
            ["模型", "steps", "batch", "train_loss", "val_loss", "val主指标", "best"],
            train_table_rows,
        )
    )

    metric_bits = []
    if train.get("last_train_loss") is not None:
        metric_bits.append(f"last_train_loss={train['last_train_loss']:.4f}")
    if train.get("last_val_loss") is not None:
        metric_bits.append(f"last_val_loss={train['last_val_loss']:.4f}")
    if train.get("last_val_map") is not None:
        metric_bits.append(f"last_val_map={train['last_val_map']:.4f}")
    if train.get("last_val_map50") is not None:
        metric_bits.append(f"last_val_map50={train['last_val_map50']:.4f}")
    if train.get("val_metric_top1_acc_micro") is not None:
        metric_bits.append(f"val_top1_acc={train['val_metric_top1_acc_micro']}")
    if train.get("best_metric_name") and train.get("best_metric_value") is not None:
        metric_bits.append(f"best={train['best_metric_name']}={train['best_metric_value']:.4f}")
    if metric_bits:
        lines.append(f"- 训练结果: {', '.join(metric_bits)}")
    time_bits = []
    for key in ("total_time", "train_time", "val_time"):
        if train.get(key):
            time_bits.append(f"{key}={train[key]}")
    if time_bits:
        lines.append(f"- 耗时: {', '.join(time_bits)}")
    return lines


def format_params_group(title: str, payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if not payload:
        return lines
    lines.append(f"### {title}")
    lines.append("```json")
    lines.extend(compact_json_lines(payload))
    lines.append("```")
    return lines


def format_single_test(test: dict[str, Any], index: int) -> list[str]:
    lines: list[str] = []
    target = classify_test_target(test)
    lines.append(f"#### 测试 {index}")
    lines.append(f"- 测试对象: `{target}`")
    if test.get("run_name"):
        lines.append(f"- 测试运行名: `{test['run_name']}`")
    if test.get("kind") == "classification_csv":
        lines.append(
            f"- 测试结果: accuracy={test.get('accuracy', 0.0):.4f}, correct={test.get('correct', 0)}/{test.get('evaluated_images', 0)}"
        )
        per_class = test.get("per_class", {})
        if per_class:
            pieces = [
                f"{label}={stats['accuracy']:.4f}({stats['correct']}/{stats['total']})"
                for label, stats in per_class.items()
            ]
            lines.append(f"- 分类明细: {', '.join(pieces)}")
    elif test.get("kind") == "detection_report":
        metric_bits = []
        for key in ("map", "map_50", "map_75", "map_05", "precision_05", "recall_05"):
            value = test.get(key)
            if isinstance(value, (int, float)):
                metric_bits.append(f"{key}={value:.4f}")
        if test.get("num_images") is not None:
            metric_bits.append(f"images={test['num_images']}")
        if test.get("avg_infer_time_ms") is not None:
            metric_bits.append(f"avg_infer_time_ms={test['avg_infer_time_ms']:.2f}")
        lines.append(f"- 测试结果: {', '.join(metric_bits) if metric_bits else '已读取检测报告'}")
        top_classes = test.get("top_classes") or []
        if top_classes:
            pieces = []
            for item in top_classes:
                ap = item.get("ap")
                if isinstance(ap, (int, float)):
                    pieces.append(f"{item.get('name', item.get('class_id'))}={ap:.4f}")
            if pieces:
                lines.append(f"- Top AP 类别: {', '.join(pieces)}")
    params = test.get("params") or {}
    key_params = params.get("key", {}) if isinstance(params, dict) else {}
    extra_params = params.get("extra", {}) if isinstance(params, dict) else {}
    lines.extend(format_params_group("测试关键参数", key_params))
    lines.extend(format_params_group("测试次要参数", extra_params))
    return lines


def format_tests_section(record: Record) -> list[str]:
    if not record.tests:
        return ["### 测试信息", "- 暂未发现与该训练关联的测试输出"]
    lines = ["### 测试信息", f"- 已关联测试数: {len(record.tests)}", ""]
    test_table_rows: list[list[Any]] = []
    for idx, test in enumerate(record.tests, start=1):
        params = test.get("params") or {}
        key_params = params.get("key", {}) if isinstance(params, dict) else {}
        test_table_rows.append(
            [
                idx,
                dataset_name_from_path(key_params.get("data_yaml")) or dataset_name_from_path((params.get("extra", {}) if isinstance(params, dict) else {}).get("data_root")) or "-",
                key_params.get("split", "-"),
                test.get("kind", "-"),
                test.get("map", test.get("accuracy", "-")),
                test.get("map_50", "-"),
                test.get("recall_05", test.get("correct", "-")),
                test.get("num_images", test.get("evaluated_images", "-")),
            ]
        )
    lines.append("#### 测试结果表")
    lines.extend(
        markdown_table(
            ["序号", "数据集", "split", "类型", "主指标", "map_50", "recall/正确数", "样本数"],
            test_table_rows,
        )
    )
    for idx, test in enumerate(record.tests, start=1):
        lines.append("")
        lines.extend(format_single_test(test, idx))
    return lines


def write_daily_reports(report_dir: Path, records: list[Record], date_filter: str) -> list[Path]:
    ensure_dir(report_dir)
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if date_filter != "all" and record.report_date != date_filter:
            continue
        grouped[record.report_date].append(record)

    generated: list[Path] = []
    for report_date, items in sorted(grouped.items()):
        day_dir = report_day_dir(report_dir, report_date)
        ensure_dir(day_dir)
        train_items = [item for item in items if item.record_kind == "train_run"]
        test_items = [item for item in items if item.record_kind == "test_only"]
        task_counter: dict[str, int] = defaultdict(int)
        for item in items:
            task_counter[item.task] += 1

        md_lines = [
            f"# 每日训练测试汇报 - {report_date}",
            "",
            "## 今日总览",
            "",
            f"- 记录数量: {len(items)}",
            f"- 训练记录: {len(train_items)}",
            f"- 独立测试记录: {len(test_items)}",
            f"- 任务分布: {', '.join(f'{task}={count}' for task, count in sorted(task_counter.items()))}",
            "",
            "## 今日工作摘要",
            "",
        ]
        for item in items:
            summary_bits = [f"`{item.name}`", f"类型=`{item.record_kind}`", f"状态=`{item.status}`"]
            if item.record_kind == "train_run" and item.train.get("model"):
                summary_bits.append(f"model=`{item.train['model']}`")
            if item.tests:
                summary_bits.append(f"tests={len(item.tests)}")
                first_test = item.tests[0]
                if first_test.get("kind") == "classification_csv":
                    summary_bits.append(f"accuracy={format_scalar(first_test.get('accuracy', 0.0))}")
                elif first_test.get("kind") == "detection_report":
                    for key in ("map", "map_50", "map_05"):
                        value = first_test.get(key)
                        if isinstance(value, (int, float)):
                            summary_bits.append(f"{key}={format_scalar(value)}")
                            break
            md_lines.append(f"- {', '.join(summary_bits)}")
        md_lines.extend(["", "## 实验明细", ""])
        for item in items:
            md_lines.append(f"## {item.name}")
            md_lines.append("")
            md_lines.append(f"- 类型: `{item.record_kind}`")
            md_lines.append(f"- 状态: `{item.status}`")
            if item.start_time:
                md_lines.append(f"- 开始时间: `{item.start_time}`")
            if item.end_time:
                md_lines.append(f"- 结束时间: `{item.end_time}`")
            md_lines.append(f"- 来源目录: `{item.source_dir}`")
            md_lines.extend(format_train_section(item) if item.record_kind == "train_run" else [])
            if item.record_kind == "train_run":
                params = item.train.get("params") or {}
                md_lines.extend(format_params_group("训练关键参数", params.get("key", {}) if isinstance(params, dict) else {}))
                md_lines.extend(format_params_group("训练次要参数", params.get("extra", {}) if isinstance(params, dict) else {}))
            md_lines.extend(format_tests_section(item))
            notes = auto_notes_for_record(item)
            if notes:
                md_lines.append("")
                md_lines.append("### 工作笔记")
                for note in notes:
                    md_lines.append(f"- {note}")
            md_lines.append("")

        md_path = day_dir / "report.md"
        md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
        generated.append(md_path)

        json_path = day_dir / "report.json"
        dump_json(
            json_path,
            {
                "report_date": report_date,
                "summary": {
                    "record_count": len(items),
                    "train_count": len(train_items),
                    "test_only_count": len(test_items),
                    "task_distribution": dict(sorted(task_counter.items())),
                },
                "train_groups": [build_train_group_payload(item) for item in train_items],
                "test_only_groups": [asdict(item) for item in test_items],
            },
        )
        generated.append(json_path)

        notes_path = day_dir / "notes.md"
        write_notes_template(notes_path, report_date, items)
        generated.append(notes_path)
    return generated


def main() -> None:
    args = parse_args()
    if args.date == "today":
        args.date = datetime.now().strftime("%Y-%m-%d")
    out_dir = args.out_dir.resolve()
    report_dir = args.report_dir.resolve()
    ensure_dir(report_dir)

    records = discover_records(out_dir)
    write_jsonl(report_dir / "run_records.jsonl", records)
    generated = write_daily_reports(report_dir, records, args.date)

    summary = {
        "out_dir": str(out_dir),
        "report_dir": str(report_dir),
        "record_count": len(records),
        "generated_files": [str(path) for path in generated],
    }
    dump_json(report_dir / "summary.json", summary)

    print(f"records: {len(records)}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
