from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import launcher
from tool_lib import common as rt
from tool_lib.det_infer import ensure_object_detection_model, get_input_samples
from tool_lib.dispatch import dispatch
from tool_lib.interactive import (
    compact_display_path,
    parse_cli_args,
    prompt_choice,
    prompt_dataset_yaml,
    prompt_experiment_dir,
    prompt_float,
    prompt_int,
    prompt_required_path,
    prompt_text,
    prompt_yes_no,
)


RUN_RECORD_CANDIDATES = (
    Path("daily_report/run_records.jsonl"),
    Path("out/daily_reports/run_records.jsonl"),
)

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SampleInfo:
    sample: rt.ImageSample
    width: int
    height: int
    mode: str

    @property
    def resolution_key(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def pixels(self) -> int:
        return self.width * self.height


def print_step(index: int, total: int, title: str, detail: str | None = None) -> None:
    message = f"\n[Step {index}/{total}] {title}"
    if detail:
        message += f"\n  {detail}"
    print(message)


def print_kv(label: str, value: Any) -> None:
    print(f"  - {label}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用 launcher 的 det infer 配置，并输出分辨率推理耗时与 L/M/S 训练耗时汇总。",
    )
    parser.add_argument("--wizard", action="store_true", default=False)
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)

    infer_input = parser.add_mutually_exclusive_group(required=False)
    infer_input.add_argument("--data", type=Path, default=None)
    infer_input.add_argument("--image-dir", type=Path, default=None)
    infer_input.add_argument("--image", type=Path, default=None)

    parser.add_argument("--split", choices=("val", "test"), default=None)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--run-launcher-infer", action="store_true", default=False)
    parser.add_argument("--launcher-output-dir", type=Path, default=None)
    parser.add_argument("--launcher-overwrite", action="store_true", default=False)
    parser.add_argument("--launcher-save-visualization", action="store_true", default=False)
    parser.add_argument("--launcher-save-json", action="store_true", default=False)
    parser.add_argument("--launcher-save-txt", action="store_true", default=False)

    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-images-per-resolution", type=int, default=0)

    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    return parser


def build_guided_args() -> argparse.Namespace | None:
    print("\n检测 benchmark 引导模式")
    print("这个流程会先收集参数，再统一执行。")

    experiment_dir = prompt_experiment_dir("det", rt.INFER_DEFAULT_EXPERIMENT_DIR)
    input_mode = prompt_choice(
        "请选择输入方式",
        [
            ("dataset", "dataset 数据集评测"),
            ("image_dir", "image_dir 文件夹批量推理"),
            ("image", "image 单张图片"),
        ],
    )

    args = argparse.Namespace(
        wizard=True,
        experiment_dir=experiment_dir,
        checkpoint=None,
        data=None,
        image_dir=None,
        image=None,
        split=rt.INFER_DEFAULT_SPLIT,
        score_threshold=rt.INFER_DEFAULT_SCORE_THRESHOLD,
        device=rt.INFER_DEFAULT_DEVICE,
        run_launcher_infer=True,
        launcher_output_dir=None,
        launcher_overwrite=False,
        launcher_save_visualization=True,
        launcher_save_json=False,
        launcher_save_txt=False,
        warmup=3,
        repeat=3,
        max_images=0,
        max_images_per_resolution=20,
        output_root=None,
        dry_run=False,
    )

    if input_mode == "dataset":
        args.data = prompt_dataset_yaml("det", Path(rt.INFER_DEFAULT_DATA))
        args.split = prompt_choice("请选择数据集划分", [("test", "test"), ("val", "val")])
    elif input_mode == "image_dir":
        args.image_dir = prompt_required_path("图片目录 --image-dir", str(rt.INFER_DEFAULT_IMAGE_DIR))
    else:
        args.image = prompt_required_path("图片路径 --image")

    print("\n推理参数")
    args.score_threshold = prompt_float("置信度阈值 --score-threshold", rt.INFER_DEFAULT_SCORE_THRESHOLD)
    args.device = prompt_text("推理设备 --device", rt.INFER_DEFAULT_DEVICE) or rt.INFER_DEFAULT_DEVICE

    print("\nbenchmark 参数")
    args.warmup = prompt_int("预热张数 --warmup", 3)
    args.repeat = prompt_int("每张重复次数 --repeat", 3)
    args.max_images_per_resolution = prompt_int("每个分辨率最多取多少张 --max-images-per-resolution", 20)
    args.max_images = prompt_int("总图数上限，0 表示不限制 --max-images", 0)

    print("\nlauncher infer 行为")
    args.run_launcher_infer = prompt_yes_no("最后是否先执行 launcher infer", True)
    if args.run_launcher_infer:
        args.launcher_save_visualization = prompt_yes_no("是否保存可视化结果", True)
        args.launcher_save_json = prompt_yes_no("是否保存 JSON 预测", False)
        args.launcher_save_txt = prompt_yes_no("是否保存 TXT 预测", False)
        args.launcher_overwrite = prompt_yes_no("launcher 输出目录非空时允许覆盖", False)
        launcher_output_raw = prompt_text("launcher 输出目录，回车自动生成", None)
        args.launcher_output_dir = Path(launcher_output_raw) if launcher_output_raw else None

    print("\n结果输出")
    output_root_raw = prompt_text("benchmark 输出目录，回车自动生成", None)
    args.output_root = Path(output_root_raw) if output_root_raw else None
    args.dry_run = prompt_yes_no("是否先做 dry-run 预览", False)

    print("\n执行确认")
    print(f"  experiment_dir: {compact_display_path(experiment_dir)}")
    if args.data is not None:
        print(f"  data: {compact_display_path(args.data)}")
        print(f"  split: {args.split}")
    if args.image_dir is not None:
        print(f"  image_dir: {compact_display_path(args.image_dir)}")
    if args.image is not None:
        print(f"  image: {compact_display_path(args.image)}")
    print(f"  score_threshold: {args.score_threshold}")
    print(f"  device: {args.device}")
    print(f"  warmup: {args.warmup}")
    print(f"  repeat: {args.repeat}")
    print(f"  max_images_per_resolution: {args.max_images_per_resolution}")
    print(f"  max_images: {args.max_images}")
    print(f"  run_launcher_infer: {args.run_launcher_infer}")
    print(f"  dry_run: {args.dry_run}")
    if not prompt_yes_no("确认按以上配置执行吗", True):
        print("已取消执行。")
        return None
    return args


def timestamp_tag() -> str:
    return datetime.now().strftime("%m%d-%H%M%S")


def resolve_output_root(output_root: Path | None) -> Path:
    if output_root is None:
        return (rt.OUT_DIR / "benchmarks" / "det_launcher" / timestamp_tag()).resolve()
    if output_root.is_absolute():
        return output_root.expanduser().resolve()
    return (rt.ROOT_DIR / output_root).resolve()


def build_launcher_cli_args(args: argparse.Namespace, launcher_output_dir: Path | None) -> list[str]:
    cli_args = ["infer"]
    if args.experiment_dir is not None:
        cli_args.extend(["--experiment-dir", str(args.experiment_dir)])
    if args.checkpoint is not None:
        cli_args.extend(["--checkpoint", str(args.checkpoint)])
    if args.data is not None:
        cli_args.extend(["--data", str(args.data)])
    if args.image_dir is not None:
        cli_args.extend(["--image-dir", str(args.image_dir)])
    if args.image is not None:
        cli_args.extend(["--image", str(args.image)])
    if args.split is not None:
        cli_args.extend(["--split", args.split])
    if args.score_threshold is not None:
        cli_args.extend(["--score-threshold", str(args.score_threshold)])
    if args.device is not None:
        cli_args.extend(["--device", args.device])
    if launcher_output_dir is not None:
        cli_args.extend(["--output-dir", str(launcher_output_dir)])
    if args.launcher_overwrite:
        cli_args.append("--overwrite")
    if args.launcher_save_visualization:
        cli_args.append("--save-visualization")
    else:
        cli_args.append("--skip-visualization")
    if args.launcher_save_json:
        cli_args.append("--save-json")
    if args.launcher_save_txt:
        cli_args.append("--save-txt")
    return cli_args


def namespace_to_dict(ns: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in vars(ns).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def collect_sample_infos(samples: list[rt.ImageSample]) -> list[SampleInfo]:
    infos: list[SampleInfo] = []
    for sample in samples:
        with rt.Image.open(sample.image_path) as image:
            width, height = image.size
            infos.append(
                SampleInfo(
                    sample=sample,
                    width=width,
                    height=height,
                    mode=image.mode,
                )
            )
    infos.sort(key=lambda item: (item.pixels, item.width, item.height, str(item.sample.relative_path)))
    return infos


def guess_image_dir_from_data_yaml(data_yaml: Path, split: str) -> Path | None:
    candidates = [
        data_yaml.parent / "images" / split,
        data_yaml.parent.parent / "images" / split,
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.exists():
            return resolved
    return None


def list_simple_samples_from_dir(image_dir: Path) -> list[Any]:
    image_dir = image_dir.expanduser().resolve()
    samples: list[Any] = []
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        samples.append(
            SimpleNamespace(
                image_path=path,
                relative_path=path.relative_to(image_dir),
            )
        )
    return samples


def collect_sample_infos_fallback(args: argparse.Namespace) -> list[SampleInfo]:
    if args.image is not None:
        image_path = args.image.expanduser().resolve()
        with rt.Image.open(image_path) as image:
            return [
                SampleInfo(
                    sample=SimpleNamespace(image_path=image_path, relative_path=Path(image_path.name)),
                    width=image.size[0],
                    height=image.size[1],
                    mode=image.mode,
                )
            ]

    if args.image_dir is not None:
        raw_samples = list_simple_samples_from_dir(args.image_dir)
    elif args.data is not None:
        guessed_dir = guess_image_dir_from_data_yaml(args.data.expanduser().resolve(), args.split)
        raw_samples = [] if guessed_dir is None else list_simple_samples_from_dir(guessed_dir)
    else:
        raw_samples = []

    infos: list[SampleInfo] = []
    for sample in raw_samples:
        with rt.Image.open(sample.image_path) as image:
            infos.append(
                SampleInfo(
                    sample=sample,
                    width=image.size[0],
                    height=image.size[1],
                    mode=image.mode,
                )
            )
    infos.sort(key=lambda item: (item.pixels, item.width, item.height, str(item.sample.relative_path)))
    return infos


def select_sample_infos(
    infos: list[SampleInfo],
    *,
    max_images: int,
    max_images_per_resolution: int,
) -> list[SampleInfo]:
    if max_images_per_resolution > 0:
        grouped: dict[str, list[SampleInfo]] = defaultdict(list)
        for info in infos:
            grouped[info.resolution_key].append(info)
        selected: list[SampleInfo] = []
        for resolution_key in sorted(grouped.keys(), key=lambda key: _resolution_sort_tuple(key)):
            selected.extend(grouped[resolution_key][:max_images_per_resolution])
    else:
        selected = list(infos)

    if max_images > 0:
        return selected[:max_images]
    return selected


def _resolution_sort_tuple(resolution_key: str) -> tuple[int, int]:
    width_text, height_text = resolution_key.split("x", maxsplit=1)
    return int(width_text) * int(height_text), int(width_text)


def build_predict_path(tmp_root: Path, info: SampleInfo) -> Path:
    if info.mode == "RGB":
        return info.sample.image_path
    predict_path = tmp_root / info.sample.relative_path
    predict_path = predict_path.with_suffix(".jpg")
    predict_path.parent.mkdir(parents=True, exist_ok=True)
    with rt.Image.open(info.sample.image_path) as image:
        image.convert("RGB").save(predict_path)
    return predict_path


def should_sync_cuda(model: Any) -> bool:
    if rt.torch is None or not rt.torch.cuda.is_available():
        return False
    try:
        parameter = next(model.parameters())
    except (StopIteration, AttributeError, TypeError):
        return False
    return getattr(parameter, "device", None) is not None and parameter.device.type == "cuda"


def sync_cuda_if_needed(enabled: bool) -> None:
    if enabled:
        rt.torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "avg_ms": 0.0,
            "median_ms": 0.0,
            "p95_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }
    avg_ms = statistics.fmean(values)
    return {
        "avg_ms": avg_ms,
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "fps": 1000.0 / avg_ms if avg_ms > 0 else 0.0,
    }


def parse_duration_to_minutes(text: str | None) -> float | None:
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


def parse_datetime(value: str | None) -> datetime | None:
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


def classify_model_scale(model_name: str | None) -> str:
    text = (model_name or "").lower()
    if any(token in text for token in ("vitl", "large")):
        return "L"
    if any(token in text for token in ("vitb", "vitm", "base", "medium")):
        return "M"
    if any(token in text for token in ("vits", "small", "tiny")):
        return "S"
    return "UNKNOWN"


def load_training_time_summary() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for record_path in RUN_RECORD_CANDIDATES:
        resolved_path = (rt.ROOT_DIR / record_path).resolve()
        if not resolved_path.exists():
            continue
        with resolved_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("record_kind") != "train_run":
                    continue
                if payload.get("task") != "object_detection":
                    continue
                train_info = payload.get("train") or {}
                source_dir = str(payload.get("source_dir") or train_info.get("output_dir") or "")
                start_time = str(payload.get("start_time") or "")
                model_name = str(train_info.get("model") or "")
                dedupe_key = (source_dir, start_time, model_name)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                records.append(payload)

    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for payload in records:
        train_info = payload.get("train") or {}
        model_name = str(train_info.get("model") or "")
        scale = classify_model_scale(model_name)
        total_minutes = parse_duration_to_minutes(train_info.get("total_time"))
        train_minutes = parse_duration_to_minutes(train_info.get("train_time"))
        val_minutes = parse_duration_to_minutes(train_info.get("val_time"))

        if total_minutes is None:
            start_dt = parse_datetime(payload.get("start_time"))
            end_dt = parse_datetime(payload.get("end_time"))
            if start_dt is not None and end_dt is not None:
                total_minutes = (end_dt - start_dt).total_seconds() / 60.0

        row = {
            "scale": scale,
            "name": payload.get("name"),
            "model": model_name,
            "status": payload.get("status"),
            "source_dir": payload.get("source_dir"),
            "steps": train_info.get("steps"),
            "batch_size": train_info.get("batch_size"),
            "train_total_steps": train_info.get("train_total_steps"),
            "total_minutes": total_minutes,
            "train_minutes": train_minutes,
            "val_minutes": val_minutes,
            "start_time": payload.get("start_time"),
            "end_time": payload.get("end_time"),
        }
        rows.append(row)
        grouped[scale].append(row)

    size_summary: dict[str, Any] = {}
    for scale in ("L", "M", "S", "UNKNOWN"):
        size_rows = grouped.get(scale, [])
        latest_row = None
        if size_rows:
            latest_row = max(size_rows, key=lambda item: str(item.get("start_time") or ""))

        total_values = [row["total_minutes"] for row in size_rows if row["total_minutes"] is not None]
        train_values = [row["train_minutes"] for row in size_rows if row["train_minutes"] is not None]
        val_values = [row["val_minutes"] for row in size_rows if row["val_minutes"] is not None]
        size_summary[scale] = {
            "count": len(size_rows),
            "latest": latest_row,
            "avg_total_minutes": statistics.fmean(total_values) if total_values else None,
            "avg_train_minutes": statistics.fmean(train_values) if train_values else None,
            "avg_val_minutes": statistics.fmean(val_values) if val_values else None,
        }

    return {
        "record_paths": [
            str((rt.ROOT_DIR / candidate).resolve())
            for candidate in RUN_RECORD_CANDIDATES
            if (rt.ROOT_DIR / candidate).resolve().exists()
        ],
        "rows": rows,
        "by_scale": size_summary,
    }


def round_payload(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: round_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_payload(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(round_payload(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_minutes(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f} min"


def build_markdown_report(
    *,
    launcher_args: argparse.Namespace,
    checkpoint_path: Path,
    input_mode: str,
    sample_infos: list[SampleInfo],
    selected_infos: list[SampleInfo],
    overall_stats: dict[str, Any] | None,
    resolution_rows: list[dict[str, Any]],
    training_summary: dict[str, Any],
    launcher_output_dir: Path | None,
    launcher_was_run: bool,
) -> str:
    lines: list[str] = []
    lines.append("# Detection Launcher Benchmark")
    lines.append("")
    lines.append("## Infer Config")
    lines.append("")
    lines.append(f"- checkpoint: `{checkpoint_path}`")
    lines.append(f"- input_mode: `{input_mode}`")
    lines.append(f"- split: `{launcher_args.split}`")
    lines.append(f"- score_threshold: `{launcher_args.score_threshold}`")
    lines.append(f"- device: `{launcher_args.device}`")
    lines.append(f"- total_images_found: `{len(sample_infos)}`")
    lines.append(f"- selected_images: `{len(selected_infos)}`")
    lines.append(f"- launcher_infer_run: `{launcher_was_run}`")
    if launcher_output_dir is not None:
        lines.append(f"- launcher_output_dir: `{launcher_output_dir}`")
    lines.append("")

    if overall_stats is not None:
        lines.append("## Overall Timing")
        lines.append("")
        lines.append(f"- avg_ms: `{overall_stats['avg_ms']:.3f}`")
        lines.append(f"- median_ms: `{overall_stats['median_ms']:.3f}`")
        lines.append(f"- p95_ms: `{overall_stats['p95_ms']:.3f}`")
        lines.append(f"- fps: `{overall_stats['fps']:.3f}`")
        lines.append("")

    lines.append("## Resolution Timing")
    lines.append("")
    lines.append("| resolution | images | avg_ms | median_ms | p95_ms | fps |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in resolution_rows:
        lines.append(
            f"| {row['resolution']} | {row['images']} | {row['avg_ms']:.3f} | "
            f"{row['median_ms']:.3f} | {row['p95_ms']:.3f} | {row['fps']:.3f} |"
        )
    lines.append("")

    lines.append("## Training Time By Scale")
    lines.append("")
    lines.append("| scale | records | latest_model | latest_total | latest_train | avg_total | avg_train |")
    lines.append("| --- | ---: | --- | ---: | ---: | ---: | ---: |")
    for scale in ("L", "M", "S"):
        scale_info = training_summary["by_scale"].get(scale, {})
        latest = scale_info.get("latest") or {}
        lines.append(
            f"| {scale} | {scale_info.get('count', 0)} | {latest.get('model', '-')} | "
            f"{format_minutes(latest.get('total_minutes'))} | {format_minutes(latest.get('train_minutes'))} | "
            f"{format_minutes(scale_info.get('avg_total_minutes'))} | {format_minutes(scale_info.get('avg_train_minutes'))} |"
        )
    return "\n".join(lines) + "\n"


def print_resolution_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("未产生分辨率计时结果。")
        return
    print("\n分辨率推理耗时:")
    print("resolution      images     avg_ms   median_ms      p95_ms        fps")
    for row in rows:
        print(
            f"{row['resolution']:<14}{row['images']:>6}  "
            f"{row['avg_ms']:>10.3f}{row['median_ms']:>12.3f}{row['p95_ms']:>12.3f}{row['fps']:>11.3f}"
        )


def print_training_table(training_summary: dict[str, Any]) -> None:
    print("\nL/M/S 训练耗时:")
    print("scale   records   latest_model                    latest_total   latest_train   avg_total   avg_train")
    for scale in ("L", "M", "S"):
        scale_info = training_summary["by_scale"].get(scale, {})
        latest = scale_info.get("latest") or {}
        latest_model = str(latest.get("model") or "-")
        print(
            f"{scale:<5}{scale_info.get('count', 0):>7}   "
            f"{latest_model:<30}{format_minutes(latest.get('total_minutes')):>13}   "
            f"{format_minutes(latest.get('train_minutes')):>12}   "
            f"{format_minutes(scale_info.get('avg_total_minutes')):>9}   "
            f"{format_minutes(scale_info.get('avg_train_minutes')):>9}"
        )


def benchmark_predict(
    *,
    model: Any,
    selected_infos: list[SampleInfo],
    score_threshold: float,
    output_root: Path,
    warmup: int,
    repeat: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    tmp_root = output_root / "_tmp_rgb"
    use_cuda_sync = should_sync_cuda(model)

    warmup_infos = selected_infos[: min(max(warmup, 0), len(selected_infos))]
    if warmup_infos:
        print(f"\n预热推理: {len(warmup_infos)} 张")
    for info in warmup_infos:
        predict_path = build_predict_path(tmp_root, info)
        sync_cuda_if_needed(use_cuda_sync)
        model.predict(predict_path, threshold=score_threshold)
        sync_cuda_if_needed(use_cuda_sync)

    image_rows: list[dict[str, Any]] = []
    grouped_times: dict[str, list[float]] = defaultdict(list)
    grouped_pixels: dict[str, int] = {}
    grouped_shapes: dict[str, tuple[int, int]] = {}

    print(f"\n正式计时: {len(selected_infos)} 张, 每张重复 {max(repeat, 1)} 次")
    for index, info in enumerate(selected_infos, start=1):
        predict_path = build_predict_path(tmp_root, info)
        timings_ms: list[float] = []
        prediction_count = 0
        for _ in range(max(repeat, 1)):
            sync_cuda_if_needed(use_cuda_sync)
            start_time = time.perf_counter()
            prediction = model.predict(predict_path, threshold=score_threshold)
            sync_cuda_if_needed(use_cuda_sync)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            timings_ms.append(elapsed_ms)
            scores = prediction.get("scores") if isinstance(prediction, dict) else None
            if scores is not None:
                prediction_count = int(scores.shape[0])

        avg_ms = statistics.fmean(timings_ms)
        image_row = {
            "image_path": str(info.sample.image_path),
            "relative_path": str(info.sample.relative_path),
            "resolution": info.resolution_key,
            "width": info.width,
            "height": info.height,
            "pixels": info.pixels,
            "repeat": max(repeat, 1),
            "avg_ms": avg_ms,
            "min_ms": min(timings_ms),
            "max_ms": max(timings_ms),
            "predictions": prediction_count,
        }
        image_rows.append(image_row)
        grouped_times[info.resolution_key].append(avg_ms)
        grouped_pixels[info.resolution_key] = info.pixels
        grouped_shapes[info.resolution_key] = (info.width, info.height)

        if index == 1 or index % 20 == 0 or index == len(selected_infos):
            print(f"[{index}/{len(selected_infos)}] benchmarked: {info.sample.image_path}")

    resolution_rows: list[dict[str, Any]] = []
    for resolution_key in sorted(grouped_times.keys(), key=_resolution_sort_tuple):
        stats = summarize_ms(grouped_times[resolution_key])
        width, height = grouped_shapes[resolution_key]
        resolution_rows.append(
            {
                "resolution": resolution_key,
                "width": width,
                "height": height,
                "pixels": grouped_pixels[resolution_key],
                "images": len(grouped_times[resolution_key]),
                **stats,
            }
        )

    overall_stats = summarize_ms([row["avg_ms"] for row in image_rows]) if image_rows else None
    return image_rows, resolution_rows, overall_stats


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        guided_args = build_guided_args()
        if guided_args is None:
            return
        args = guided_args
    else:
        args = parser.parse_args()
        if args.wizard:
            guided_args = build_guided_args()
            if guided_args is None:
                return
            args = guided_args
    total_steps = 8

    print_step(1, total_steps, "加载 launcher 配置")
    rt.apply_user_settings(launcher.build_user_settings())
    runtime_error: ModuleNotFoundError | None = None
    try:
        rt.import_runtime_dependencies()
        print_kv("runtime", "完整运行时可用")
    except ModuleNotFoundError as exc:
        runtime_error = exc
        print_kv("runtime", f"静态预览模式: {exc}")
        if not args.dry_run:
            raise

    print_step(2, total_steps, "生成 benchmark 输出目录")
    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    print_kv("output_root", output_root)

    launcher_output_dir = args.launcher_output_dir
    if launcher_output_dir is not None and not launcher_output_dir.is_absolute():
        launcher_output_dir = (rt.ROOT_DIR / launcher_output_dir).resolve()
    if args.run_launcher_infer and launcher_output_dir is None:
        launcher_output_dir = output_root / "launcher_infer"

    print_step(3, total_steps, "解析 launcher infer 参数")
    launcher_cli_args = build_launcher_cli_args(args, launcher_output_dir)
    launcher_args = parse_cli_args(launcher_cli_args)
    if launcher_args.data is not None:
        launcher_args.compute_metrics = True
        launcher_args.save_test_report = True
    print_kv("launcher_cli", " ".join(str(item) for item in launcher_cli_args))

    print_step(4, total_steps, "解析 checkpoint 与历史训练时间")
    checkpoint_path = rt.resolve_checkpoint_path(launcher_args.checkpoint, launcher_args.experiment_dir)
    training_summary = load_training_time_summary()
    print_kv("checkpoint", checkpoint_path)
    print_kv("train_records", len(training_summary.get("rows", [])))

    print_step(5, total_steps, "收集图片与分辨率")
    if runtime_error is None:
        samples, _, input_mode = get_input_samples(launcher_args)
        rt.ensure_image_samples(samples)
        sample_infos = collect_sample_infos(samples)
        selected_infos = select_sample_infos(
            sample_infos,
            max_images=args.max_images,
            max_images_per_resolution=args.max_images_per_resolution,
        )
    else:
        input_mode = "fallback_preview"
        sample_infos = collect_sample_infos_fallback(launcher_args)
        selected_infos = select_sample_infos(
            sample_infos,
            max_images=args.max_images,
            max_images_per_resolution=args.max_images_per_resolution,
        )
    print_kv("input_mode", input_mode)
    print_kv("images_found", len(sample_infos))
    print_kv("images_selected", len(selected_infos))
    print_kv("resolutions_found", len({info.resolution_key for info in sample_infos}))

    write_json(output_root / "launcher_args.json", namespace_to_dict(launcher_args))
    write_json(
        output_root / "input_summary.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "input_mode": input_mode,
            "images_found": len(sample_infos),
            "images_selected": len(selected_infos),
            "resolutions_found": sorted({info.resolution_key for info in sample_infos}, key=_resolution_sort_tuple),
        },
    )
    write_json(output_root / "training_time_summary.json", training_summary)

    print_step(6, total_steps, "执行 launcher infer 或跳过")
    if args.run_launcher_infer:
        print_kv("launcher_infer", "开始执行")
        dispatch(launcher_args)
        print_kv("launcher_infer", "执行完成")
    else:
        print_kv("launcher_infer", "当前命令未开启 --run-launcher-infer，本步跳过")

    if args.dry_run:
        print_step(7, total_steps, "输出 dry-run 预览")
        grouped_count: dict[str, int] = defaultdict(int)
        for info in selected_infos:
            grouped_count[info.resolution_key] += 1
        preview_rows = [
            {
                "resolution": resolution_key,
                "images": grouped_count[resolution_key],
            }
            for resolution_key in sorted(grouped_count.keys(), key=_resolution_sort_tuple)
        ]
        print("\nDry run 预览:")
        if runtime_error is not None:
            print(f"当前环境缺少完整运行时，dry-run 走静态预览: {runtime_error}")
        print(f"checkpoint: {checkpoint_path}")
        print(f"input_mode: {input_mode}")
        print(f"images_found: {len(sample_infos)}")
        print(f"images_selected: {len(selected_infos)}")
        if launcher_args.data is not None and runtime_error is not None and not sample_infos:
            print(f"data_yaml: {launcher_args.data}")
            print("当前 dry-run 没有解析出图片目录，建议进入训练环境后执行正式 benchmark。")
        print_resolution_table(
            [
                {
                    "resolution": row["resolution"],
                    "images": row["images"],
                    "avg_ms": 0.0,
                    "median_ms": 0.0,
                    "p95_ms": 0.0,
                    "fps": 0.0,
                }
                for row in preview_rows
            ]
        )
        print_training_table(training_summary)
        print(f"\n输出目录: {output_root}")
        return

    print_step(7, total_steps, "加载模型并执行逐图计时")
    model = rt.lightly_train.load_model(
        model=checkpoint_path,
        device=rt.resolve_device(launcher_args.device),
    )
    model.eval()
    ensure_object_detection_model(model)

    image_rows, resolution_rows, overall_stats = benchmark_predict(
        model=model,
        selected_infos=selected_infos,
        score_threshold=launcher_args.score_threshold,
        output_root=output_root,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    print_step(8, total_steps, "写出统计结果")
    benchmark_summary = {
        "checkpoint_path": str(checkpoint_path),
        "input_mode": input_mode,
        "images_found": len(sample_infos),
        "images_selected": len(selected_infos),
        "repeat": max(args.repeat, 1),
        "warmup": max(args.warmup, 0),
        "launcher_infer_run": args.run_launcher_infer,
        "launcher_output_dir": str(launcher_output_dir) if launcher_output_dir is not None else None,
        "overall": overall_stats,
        "by_resolution": resolution_rows,
    }

    write_json(output_root / "benchmark_summary.json", benchmark_summary)
    write_json(output_root / "image_timings.json", image_rows)
    write_csv(
        output_root / "image_timings.csv",
        image_rows,
        [
            "image_path",
            "relative_path",
            "resolution",
            "width",
            "height",
            "pixels",
            "repeat",
            "avg_ms",
            "min_ms",
            "max_ms",
            "predictions",
        ],
    )
    write_csv(
        output_root / "resolution_stats.csv",
        resolution_rows,
        [
            "resolution",
            "width",
            "height",
            "pixels",
            "images",
            "avg_ms",
            "median_ms",
            "p95_ms",
            "min_ms",
            "max_ms",
            "fps",
        ],
    )

    report_markdown = build_markdown_report(
        launcher_args=launcher_args,
        checkpoint_path=checkpoint_path,
        input_mode=input_mode,
        sample_infos=sample_infos,
        selected_infos=selected_infos,
        overall_stats=overall_stats,
        resolution_rows=resolution_rows,
        training_summary=training_summary,
        launcher_output_dir=launcher_output_dir,
        launcher_was_run=args.run_launcher_infer,
    )
    (output_root / "report.md").write_text(report_markdown, encoding="utf-8")

    print_resolution_table(resolution_rows)
    print_training_table(training_summary)
    if overall_stats is not None:
        print(
            "\n整体平均耗时: "
            f"{overall_stats['avg_ms']:.3f} ms, "
            f"median {overall_stats['median_ms']:.3f} ms, "
            f"p95 {overall_stats['p95_ms']:.3f} ms, "
            f"fps {overall_stats['fps']:.3f}"
        )
    print(f"\n结果输出目录: {output_root}")


if __name__ == "__main__":
    main()
