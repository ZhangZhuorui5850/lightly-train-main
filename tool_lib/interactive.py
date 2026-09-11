"""交互入口与 CLI 参数解析。

这个文件负责两种入口形式：
- 交互模式：python launcher.py 后进入菜单
- CLI 模式：python launcher.py infer/export/eda ...

主要内容包括：
- 读取用户输入
- 打印菜单和配置确认页
- 生成 dispatch 需要的 argparse.Namespace

它只负责“收集用户意图”，不负责真正执行训练、推理或评估。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from . import common as rt
from . import dataset_adapter
from .file_index import find_files


class InputCancelled(Exception):
    """Raised when an interactive input stream closes before confirmation."""


COMMAND_HELP = {
    "train": "训练 cls、det 或 seg 模型",
    "infer": "对图片、目录或数据集执行推理",
    "eval": "评估 cls、det 或 seg 模型",
    "export": "筛选并重划分检测数据集",
    "seg-export": "筛选并重划分实例分割数据集",
    "seg-eda": "自动识别并分析实例或语义分割数据集",
    "seg-curate": "根据 EDA 结果整理语义分割类别",
    "eda": "分析检测数据集",
    "review-sample": "生成检测数据集质检抽样",
    "report": "生成单实验或汇总报告",
    "clean": "预览或执行实验产物清理",
    "optimize": "分析类别质量与混淆并交互优化数据集",
}
_PENDING_OPTIMIZE_TEMP_DIRS: set[Path] = set()
ARGUMENT_HELP = {
    "task": "任务类型。",
    "seg_train_type": "分割训练类型；省略时优先从实验记录推断。",
    "data_yaml": "训练数据集目录或 data.yaml。",
    "model": "LightlyTrain 模型名称。",
    "backbone_weights": "backbone 预训练权重路径。",
    "checkpoint": "checkpoint 路径；省略时从实验目录解析最佳权重。",
    "out_dir": "训练或质检输出目录。",
    "steps": "训练步数，支持正整数或 auto。",
    "batch_size": "batch size，支持正整数或 auto。",
    "num_workers": "数据加载进程数，支持非负整数或 auto。",
    "devices": "GPU 编号、逗号分隔编号或 auto。",
    "overwrite": "允许安全替换已有输出。",
    "resume_interrupted": "从指定已有实验目录继续中断训练。",
    "experiment_dir": "实验目录；省略时自动选择最近的同任务实验。",
    "image": "单张输入图片。",
    "image_dir": "批量输入图片目录。",
    "data": "数据集目录或 data.yaml。",
    "split": "数据集划分；部分命令支持一次提供多个 split。",
    "output_dir": "输出目录；省略时按实验、数据集和 split 自动生成。",
    "score_threshold": "检测置信度阈值，范围 [0,1]。",
    "threshold": "分类或分割阈值，范围 [0,1]。",
    "topk": "分类返回数量，必须大于 0。",
    "device": "推理设备，例如 auto、cpu、cuda。",
    "save_visualization": "保存可视化结果。",
    "save_json": "保存 JSON 预测结果。",
    "save_txt": "保存 TXT 预测结果。",
    "report_iou_threshold": "报告 TP/FP/FN 匹配 IoU，范围 [0,1]。",
    "bad_class_map50_threshold": "劣质类别 mAP50 参考阈值。",
    "compute_metrics": "计算完整评估指标。",
    "metric_classwise": "输出按类别指标。",
    "save_test_report": "保存评估报告 JSON。",
    "report_path": "评估报告输出路径。",
    "sahi": "启用 SAHI 切片推理。",
    "sahi_overlap": "相邻切片重叠比例，范围 [0,1)。",
    "sahi_nms_iou": "切片结果 NMS IoU，范围 [0,1]。",
    "sahi_global_local_iou": "全局与局部预测匹配 IoU，范围 [0,1]。",
    "sahi_skip_small": "小图跳过 SAHI。",
    "dry_run": "只分析并显示执行计划，保持零写入。",
    "test_dir": "按类别子目录组织的分类测试集。",
    "classwise": "输出按类别指标。",
    "vis_max_images": "每个 split 的对比图上限；0 表示全量。",
    "report_json": "评估报告 JSON；可按命令语义自动匹配。",
    "export_source_data": "待筛选的数据集目录或 data.yaml。",
    "good_class_threshold": "类别质量参考阈值，范围 [0,1]。",
    "balance_ratio": "类别最大/最小数量比例；0 表示自动，人工值需大于等于 1。",
    "split_ratio": "train:val:test 三段比例，例如 8:1:1。",
    "size_ratio": "small:medium:large 三段目标比例；空字符串关闭尺寸均衡。",
    "size_balance_weight": "尺寸均衡权重，必须为非负有限数字。",
    "export_suffix": "导出数据集目录后缀。",
    "seg_type": "分割数据类型；auto 会读取标注内容判断。",
    "min_class_images": "每类最少图片数或 EDA 推荐门槛。",
    "threshold_percentile": "EDA 推荐阈值百分位，范围 [0,1]。",
    "eda_dir": "已有 EDA 输出目录。",
    "drop_classes": "逗号分隔的待删除类别 ID。",
    "image_threshold": "train 每类图片上限；0 表示保留全量。",
    "contiguous_ids": "将保留类别重映射为连续 ID。",
    "search": "实验目录关键词。",
    "source_data_yaml": "待优化的检测数据集目录或 data.yaml。",
    "infer_output_dir": "包含预测 JSON 的 infer 输出目录。",
    "optimize_experiment_dir": "优化报告归档实验目录。",
    "confusion_threshold": "类别混淆候选阈值，范围 [0,1]。",
}

OPTION_HELP = {
    "--skip-visualization": "跳过可视化结果。",
    "--no-save-visualization": "跳过评估对比图。",
    "--skip-json": "跳过 JSON 预测结果。",
    "--no-sahi-skip-small": "所有图片均按已启用的 SAHI 策略处理。",
    "--no-auto-balance": "使用显式导出阈值和数量参数。",
    "--strict-class-threshold": "将类别质量阈值作为严格筛选条件。",
    "--no-trim-boxes": "保留入选图片中的全部原始框。",
    "--disable-geometry": "关闭几何标注问题分析。",
    "--disable-model-analysis": "关闭模型预测问题分析。",
    "--disable-outlier-class": "关闭异常类别分析。",
    "--disable-visualization": "关闭质检样本可视化。",
    "--no-embed-images": "报告引用外部图片文件。",
}


class CommandHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show value defaults while keeping boolean toggle descriptions unambiguous."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            return action.help or ""
        return super()._get_help_string(action)


def _add_command_parser(subparsers: Any, command: str) -> argparse.ArgumentParser:
    description = COMMAND_HELP[command]
    return subparsers.add_parser(
        command,
        help=description,
        description=description,
        formatter_class=CommandHelpFormatter,
    )


def _complete_parser_help(subparsers: Any) -> None:
    for command_parser in subparsers.choices.values():
        for action in command_parser._actions:
            if action.help is None:
                action.help = next(
                    (
                        OPTION_HELP[option]
                        for option in action.option_strings
                        if option in OPTION_HELP
                    ),
                    ARGUMENT_HELP.get(
                        action.dest,
                        f"{action.dest.replace('_', ' ')} 参数。",
                    ),
                )


def read_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise InputCancelled("输入结束，已取消执行。")


def prompt_choice(title: str, options: list[tuple[str, str]]) -> str:
    print(f"\n{title}")
    for idx, (_, label) in enumerate(options, start=1):
        print(f"  {idx}. {label}")
    while True:
        raw = read_input("请输入编号或名称: ").strip().lower()
        if not raw:
            print("输入不能为空，请重新选择。")
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(options):
                return options[index][0]
        for key, label in options:
            if raw in {key.lower(), label.lower()}:
                return key
        print("无效选择，请重新输入。")


def prompt_text(prompt: str, default: str | None = None) -> str | None:
    suffix = f" [{default}]" if default else ""
    value = read_input(f"{prompt}{suffix}: ").strip()
    return value or default


def prompt_required_path(prompt: str, default: str | None = None) -> Path:
    while True:
        value = prompt_text(prompt, default)
        if value and str(value).strip():
            return Path(value)
        print("该路径不能为空，请重新输入。")


def prompt_float(prompt: str, default: float) -> float:
    while True:
        value = read_input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("请输入数字，例如 0.6。")


def prompt_int(prompt: str, default: int) -> int:
    while True:
        value = read_input(f"{prompt} [{default}]: ").strip()
        if not value:
            return default
        try:
            return int(value)
        except ValueError:
            print("请输入整数，例如 2。")


def prompt_yes_no(prompt: str, default: bool) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        value = read_input(f"{prompt} [{default_hint}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "1"}:
            return True
        if value in {"n", "no", "0"}:
            return False
        print("请输入 y 或 n。")


def list_recent_experiment_dirs(task: str, limit: int = 5) -> list[Path]:
    return rt.discover_recent_experiment_dirs(task, limit=limit, require_checkpoint=False)


def list_experiment_dirs(task: str | None = None) -> list[Path]:
    return rt.discover_recent_experiment_dirs(
        task,
        limit=None,
        require_checkpoint=False,
    )


def compact_display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    parts = resolved.parts

    for anchor in ("out", "datasets"):
        if anchor in parts:
            index = parts.index(anchor)
            tail_parts = parts[index + 1 :]
            if tail_parts:
                return str(Path(*tail_parts))
            return resolved.name

    try:
        return str(resolved.relative_to(rt.ROOT_DIR))
    except ValueError:
        return resolved.name


def compact_display_value(value: object) -> object:
    if isinstance(value, Path):
        return compact_display_path(value)
    return value


def filter_dirs_by_keyword(dirs: list[Path], keyword: str) -> list[Path]:
    raw = keyword.strip().casefold()
    if not raw:
        return dirs
    normalized = re.sub(r"[^\w]+", " ", raw, flags=re.UNICODE)
    terms = [term for term in normalized.split() if term]
    if not terms:
        return dirs
    return [
        path
        for path in dirs
        if all(
            term in re.sub(r"[^\w]+", " ", str(path).casefold(), flags=re.UNICODE)
            for term in terms
        )
    ]


def _normalize_search_text(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", text.lower())


def _tokenize_path_text(text: str) -> set[str]:
    stop_words = {
        "all",
        "best",
        "checkpoint",
        "checkpoints",
        "data",
        "dataset",
        "datasets",
        "det",
        "eval",
        "export",
        "exported",
        "image",
        "images",
        "important",
        "infer",
        "label",
        "labels",
        "last",
        "manual",
        "models",
        "out",
        "report",
        "results",
        "run",
        "runs",
        "split",
        "test",
        "train",
        "val",
    }
    normalized = _normalize_search_text(text)
    return {
        token
        for token in normalized.split()
        if token and token not in stop_words and (len(token) >= 3 or token.isdigit())
    }


def _load_json_dict(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_json_block(text: str, marker: str) -> dict[str, object]:
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
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return {}
                return payload if isinstance(payload, dict) else {}
    return {}


def _resolve_existing_path(path_text: object) -> Path | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (rt.ROOT_DIR / path).resolve()
    else:
        path = path.resolve()
    return path if path.exists() else None


def _dataset_preferences_from_experiment(experiment_dir: Path | None) -> tuple[set[Path], set[Path], set[str]]:
    preferred_yaml_paths: set[Path] = set()
    preferred_root_dirs: set[Path] = set()
    preferred_tokens: set[str] = set()

    if experiment_dir is None:
        return preferred_yaml_paths, preferred_root_dirs, preferred_tokens

    resolved_experiment_dir = experiment_dir.expanduser().resolve()
    preferred_tokens |= _tokenize_path_text(resolved_experiment_dir.name)
    try:
        experiment_text = str(resolved_experiment_dir.relative_to(rt.ROOT_DIR))
    except ValueError:
        experiment_text = str(resolved_experiment_dir)
    preferred_tokens |= _tokenize_path_text(experiment_text)

    train_log_path = resolved_experiment_dir / "train.log"
    if train_log_path.exists():
        text = train_log_path.read_text(encoding="utf-8", errors="ignore")
        args_payload = _extract_json_block(text, "Args:")
        data_payload = args_payload.get("data")
        if isinstance(data_payload, dict):
            data_root = _resolve_existing_path(data_payload.get("path"))
            if data_root is not None:
                preferred_root_dirs.add(data_root)
                preferred_yaml_paths.add((data_root / "data.yaml").resolve())
                preferred_tokens |= _tokenize_path_text(data_root.name)
                preferred_tokens |= _tokenize_path_text(data_root.parent.name)

    search_roots: list[Path] = [resolved_experiment_dir]
    if rt.TEST_OUTPUT_ROOT_DIR.exists() and rt.TEST_OUTPUT_ROOT_DIR != resolved_experiment_dir:
        search_roots.append(rt.TEST_OUTPUT_ROOT_DIR)

    seen_run_meta_paths: set[Path] = set()
    for run_meta_path in find_files(
        search_roots,
        label="索引数据集关联记录",
        filenames={"run_meta.json"},
    ):
        resolved_run_meta_path = run_meta_path.resolve()
        if resolved_run_meta_path in seen_run_meta_paths:
            continue
        seen_run_meta_paths.add(resolved_run_meta_path)
        payload = _load_json_dict(run_meta_path)
        if payload.get("task") != "det" or payload.get("action") != "infer":
            continue
        paths_payload = payload.get("paths")
        if not isinstance(paths_payload, dict):
            continue
        recorded_experiment_dir = _resolve_existing_path(paths_payload.get("experiment_dir"))
        physically_inside = resolved_experiment_dir in resolved_run_meta_path.parents
        if not physically_inside and recorded_experiment_dir != resolved_experiment_dir:
            continue
        data_yaml_path = _resolve_existing_path(paths_payload.get("data_yaml"))
        if data_yaml_path is not None:
            preferred_yaml_paths.add(data_yaml_path)
            preferred_root_dirs.add(data_yaml_path.parent.resolve())
        data_root = _resolve_existing_path(paths_payload.get("data_root"))
        if data_root is not None:
            preferred_root_dirs.add(data_root)
            preferred_yaml_paths.add((data_root / "data.yaml").resolve())
            preferred_tokens |= _tokenize_path_text(data_root.name)
            preferred_tokens |= _tokenize_path_text(data_root.parent.name)

    return preferred_yaml_paths, preferred_root_dirs, preferred_tokens


def _dataset_candidate_sort_key(
    candidate: Path,
    *,
    preferred_yaml_paths: set[Path],
    preferred_root_dirs: set[Path],
    preferred_tokens: set[str],
) -> tuple[int, int, str]:
    resolved_candidate = candidate.resolve()
    candidate_root = resolved_candidate.parent
    try:
        candidate_text = str(candidate_root.relative_to(rt.ROOT_DIR))
    except ValueError:
        candidate_text = str(candidate_root)
    candidate_tokens = _tokenize_path_text(candidate_text)
    token_overlap = len(candidate_tokens & preferred_tokens)

    relevance_score = 0
    if resolved_candidate in preferred_yaml_paths:
        relevance_score += 1000
    if candidate_root in preferred_root_dirs:
        relevance_score += 800
    if token_overlap:
        relevance_score += token_overlap * 20

    return (relevance_score, int(resolved_candidate.stat().st_mtime), str(resolved_candidate).lower())


def list_dataset_yaml_candidates(
    task: str | None = None,
    *,
    preferred_experiment_dir: Path | None = None,
) -> list[Path]:
    search_roots = list(rt.DATASET_SEARCH_ROOTS)
    default_root = (rt.ROOT_DIR / "datasets").resolve()
    if default_root not in search_roots:
        search_roots.insert(0, default_root)
    # Batch jobs and mounted datasets can extend discovery without editing the
    # launcher file.  The value follows the platform path separator syntax.
    for raw_root in os.environ.get("LIGHTLY_DATASET_SEARCH_ROOTS", "").split(os.pathsep):
        if not raw_root.strip():
            continue
        env_root = Path(raw_root).expanduser().resolve()
        if env_root not in search_roots:
            search_roots.append(env_root)
    indexed_roots = set(search_roots)
    for configured in (
        rt.DATASET_DIR,
        rt.INFER_DEFAULT_DATA,
        rt.EXPORT_DEFAULT_SOURCE_DATA,
        rt.SEG_DATASET_DIR,
        rt.SEG_EXPORT_DEFAULT_SOURCE_DATA,
        rt.SEMANTIC_SEG_DATASET_DIR,
        rt.SEMANTIC_SEG_DEFAULT_DATA,
    ):
        configured = Path(configured).expanduser().resolve()
        candidate_root = configured.parent if configured.suffix.casefold() in {".yaml", ".yml"} else configured
        if candidate_root.exists() and not any(
            _path_is_inside(candidate_root, root) for root in search_roots
        ):
            search_roots.append(candidate_root)

    candidates: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            candidates.extend(
                dataset_adapter.discover_dataset_configs(
                    search_root,
                    task=task,
                    include_empty=search_root in indexed_roots,
                    include_unknown=(
                        search_root in indexed_roots
                        and task in {"det", "seg", "instance"}
                    ),
                    show_progress=True,
                )
            )
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))

    preferred_yaml_paths, preferred_root_dirs, preferred_tokens = _dataset_preferences_from_experiment(
        preferred_experiment_dir
    )
    candidates.sort(
        key=lambda path: _dataset_candidate_sort_key(
            path,
            preferred_yaml_paths=preferred_yaml_paths,
            preferred_root_dirs=preferred_root_dirs,
            preferred_tokens=preferred_tokens,
        ),
        reverse=True,
    )
    return candidates


def prompt_directory_name(prompt: str, default: str) -> str:
    while True:
        value = prompt_text(prompt, default)
        if value is None:
            print("目录名不能为空，请重新输入。")
            continue
        name = value.strip()
        if not name:
            print("目录名不能为空，请重新输入。")
            continue
        if Path(name).name != name or name in {".", ".."}:
            print("请输入单层目录名，例如 military_dataset。")
            continue
        return name


def prompt_dataset_yaml(task: str, default: Path, *, preferred_experiment_dir: Path | None = None) -> Path:
    candidates = list_dataset_yaml_candidates(task=task, preferred_experiment_dir=preferred_experiment_dir)
    if not candidates:
        return Path(prompt_text("数据配置 --data", str(default)) or str(default))

    visible_limit = 20
    active_candidates = candidates
    visible_candidates = candidates[:visible_limit]

    while True:
        print(f"\n自动发现的 {task} 数据集:")
        for idx, path in enumerate(visible_candidates, start=1):
            print(f"  {idx}. {compact_display_path(path)}")
        if len(visible_candidates) < len(active_candidates):
            print(
                f"  ... 当前仅显示前 {len(visible_candidates)} 个，"
                f"当前筛选共 {len(active_candidates)} 个"
            )
        print("  输入编号直接选择")
        print("  输入关键词筛选")
        print("  输入 all 恢复并查看全部")
        custom_label = "data.yaml 或 ImageFolder 根目录" if task == "cls" else "data.yaml"
        print(f"  输入 custom 手动输入 {custom_label}")

        raw = read_input("请选择数据集: ").strip()
        lowered = raw.lower()
        if not raw:
            resolved_default = default.expanduser().resolve()
            return resolved_default if resolved_default in candidates else candidates[0]
        if lowered == "custom":
            return Path(prompt_text("数据配置 --data", str(default)) or str(default))
        if lowered == "all":
            active_candidates = candidates
            visible_candidates = candidates
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(visible_candidates):
                return visible_candidates[index]
        matched = filter_dirs_by_keyword(candidates, raw)
        if matched:
            active_candidates = matched
            visible_candidates = matched[:visible_limit]
            continue
        print("无效选择，请重新输入。")


def prompt_experiment_dir(task: str, default: Path, *, initial_keyword: str | None = None) -> Path:
    all_dirs = list_experiment_dirs(task=task)
    if not all_dirs:
        all_dirs = list_experiment_dirs()
    if not all_dirs:
        return Path(prompt_text("实验目录", str(default)) or str(default))

    visible_dirs = filter_dirs_by_keyword(all_dirs, initial_keyword or "")
    if not visible_dirs:
        visible_dirs = all_dirs
    while True:
        print(f"\nout/ 下实验目录列表（递归扫描，当前任务: {task}）:")
        for idx, path in enumerate(visible_dirs, start=1):
            modified_text = rt.format_experiment_modified_time(path)
            print(f"  {idx}. [{modified_text}] {compact_display_path(path)}")
        print("  输入编号直接选择")
        print("  输入关键字筛选，例如: 0408 / military_dataset")
        print("  输入 all 查看全部")
        print("  输入 custom 手动输入路径")

        raw = read_input("请选择实验目录: ").strip()
        lowered = raw.lower()

        if not raw:
            return default
        if lowered == "custom":
            return Path(prompt_text("实验目录", str(default)) or str(default))
        if lowered == "all":
            visible_dirs = all_dirs
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(visible_dirs):
                return visible_dirs[index]

        filtered_dirs = filter_dirs_by_keyword(all_dirs, raw)
        if not filtered_dirs:
            print(f"没有匹配关键字 '{raw}' 的目录，请重新输入。")
            continue
        visible_dirs = filtered_dirs


def print_config_preview(title: str, args: argparse.Namespace) -> None:
    print(f"\n{title} 配置确认")
    for key in sorted(vars(args)):
        value = getattr(args, key)
        if value is None and key in {"output_dir", "report_path", "report_json"}:
            value = "(auto)"
        value = compact_display_value(value)
        print(f"  {key}: {value}")


def _det_export_strategy_text(label: str, value, *, enabled: bool = True) -> str:
    if not enabled:
        return f"  {label}: 关闭"
    if isinstance(value, bool):
        return f"  {label}: {'开启' if value else '关闭'}"
    if isinstance(value, (int, float)) and float(value) <= 0.0:
        return f"  {label}: 自动推导"
    if value is None:
        return f"  {label}: 自动匹配"
    return f"  {label}: 手动覆盖为 {compact_display_value(value)}"


def print_det_export_preview(args: argparse.Namespace) -> None:
    print("\ndet/export 输入确认")
    report_value = "(auto)" if args.report_json is None else compact_display_value(args.report_json)
    if args.export_suffix == rt.EXPORT_DEFAULT_EXPORT_SUFFIX:
        if int(args.target_total_images) > 0:
            export_name_preview = rt.default_det_export_dir_suffix(image_count=int(args.target_total_images))
        else:
            export_name_preview = f"{rt.EXPORT_DEFAULT_EXPORT_SUFFIX}_<实际导出图数>"
    else:
        export_name_preview = str(args.export_suffix)
    print(f"  export_source_data: {compact_display_value(args.export_source_data)}")
    print(f"  report_json: {report_value}")
    print(f"  target_total_images: {compact_display_value(args.target_total_images)}")
    print(f"  split_ratio: {compact_display_value(args.split_ratio)}")
    print(f"  export_suffix: {compact_display_value(args.export_suffix)}")
    print(f"  export_dir_name: <dataset>{export_name_preview}")

    print("\n自动分析策略")
    print(f"  auto_balance: {'开启' if args.auto_balance else '关闭'}")
    print(f"  auto_relax_class_threshold: {'开启' if args.auto_relax_class_threshold else '关闭'}")
    print(_det_export_strategy_text("good_class_threshold", args.good_class_threshold))
    print(_det_export_strategy_text("balance_ratio", args.balance_ratio, enabled=args.auto_balance))
    print(_det_export_strategy_text("min_class_images", args.min_class_images, enabled=args.auto_balance))
    print(_det_export_strategy_text("min_class_boxes", args.min_class_boxes, enabled=args.auto_balance))
    print(_det_export_strategy_text("target_images_per_class", args.target_images_per_class, enabled=args.auto_balance))
    print(_det_export_strategy_text("target_boxes_per_class", args.target_boxes_per_class, enabled=args.auto_balance))
    print(_det_export_strategy_text("max_boxes_per_image", args.max_boxes_per_image, enabled=args.auto_balance))
    print(_det_export_strategy_text("max_boxes_per_class_per_image", args.max_boxes_per_class_per_image, enabled=args.auto_balance))
    print(_det_export_strategy_text("box_density_penalty", args.box_density_penalty, enabled=args.auto_balance))
    print(f"  size_ratio: {compact_display_value(args.size_ratio)}")
    size_enabled = bool(str(args.size_ratio).strip())
    print(_det_export_strategy_text("size_balance_weight", args.size_balance_weight, enabled=size_enabled))
    print(_det_export_strategy_text("avg_boxes_per_image_min", args.avg_boxes_per_image_min, enabled=size_enabled))
    print(_det_export_strategy_text("avg_boxes_per_image_max", args.avg_boxes_per_image_max, enabled=size_enabled))


def _seg_export_strategy_text(label: str, value, *, enabled: bool = True) -> str:
    if not enabled:
        return f"  {label}: 关闭"
    if isinstance(value, bool):
        return f"  {label}: {'开启' if value else '关闭'}"
    if isinstance(value, (int, float)) and float(value) <= 0.0:
        return f"  {label}: 自动推导"
    if value is None:
        return f"  {label}: 自动匹配"
    return f"  {label}: 手动覆盖为 {compact_display_value(value)}"


def print_seg_export_preview(args: argparse.Namespace) -> None:
    print("\nseg/export 输入确认")
    report_value = "(auto)" if args.report_json is None else compact_display_value(args.report_json)
    if args.export_suffix == rt.SEG_EXPORT_DEFAULT_EXPORT_SUFFIX:
        if int(args.target_total_images) > 0:
            export_name_preview = rt.default_seg_export_dir_suffix(image_count=int(args.target_total_images))
        else:
            export_name_preview = f"{rt.SEG_EXPORT_DEFAULT_EXPORT_SUFFIX}_<实际导出图数>"
    else:
        export_name_preview = str(args.export_suffix)
    print(f"  export_source_data: {compact_display_value(args.export_source_data)}")
    print(f"  report_json: {report_value}")
    print(f"  target_total_images: {compact_display_value(args.target_total_images)}")
    print(f"  split_ratio: {compact_display_value(args.split_ratio)}")
    print(f"  export_suffix: {compact_display_value(args.export_suffix)}")
    print(f"  export_dir_name: <dataset>{export_name_preview}")

    print("\n自动分析策略")
    print(f"  auto_balance: {'开启' if args.auto_balance else '关闭'}")
    print(f"  auto_relax_class_threshold: {'开启' if args.auto_relax_class_threshold else '关闭'}")
    print(_seg_export_strategy_text("good_class_threshold", args.good_class_threshold))
    print(_seg_export_strategy_text("balance_ratio", args.balance_ratio, enabled=args.auto_balance))
    print(_seg_export_strategy_text("min_class_images", args.min_class_images, enabled=args.auto_balance))
    print(_seg_export_strategy_text("min_class_instances", args.min_class_instances, enabled=args.auto_balance))
    print(_seg_export_strategy_text("target_images_per_class", args.target_images_per_class, enabled=args.auto_balance))
    print(_seg_export_strategy_text("target_instances_per_class", args.target_instances_per_class, enabled=args.auto_balance))
    print(_seg_export_strategy_text("max_instances_per_image", args.max_instances_per_image, enabled=args.auto_balance))
    print(_seg_export_strategy_text("max_instances_per_class_per_image", args.max_instances_per_class_per_image, enabled=args.auto_balance))
    print(_seg_export_strategy_text("instance_density_penalty", args.instance_density_penalty, enabled=args.auto_balance))
    print(f"  size_ratio: {compact_display_value(args.size_ratio)}")
    size_enabled = bool(str(args.size_ratio).strip())
    print(_seg_export_strategy_text("size_balance_weight", args.size_balance_weight, enabled=size_enabled))
    print(_seg_export_strategy_text("avg_instances_per_image_min", args.avg_instances_per_image_min, enabled=size_enabled))
    print(_seg_export_strategy_text("avg_instances_per_image_max", args.avg_instances_per_image_max, enabled=size_enabled))


def confirm_args(title: str, args: argparse.Namespace) -> argparse.Namespace | None:
    if title == "det/export":
        print_det_export_preview(args)
        confirm_text = "确认按以上输入开始分析并导出吗"
    elif title == "seg/export":
        print_seg_export_preview(args)
        confirm_text = "确认按以上输入开始分析并导出吗"
    else:
        print_config_preview(title, args)
        confirm_text = "确认执行以上配置吗"
    if prompt_yes_no(confirm_text, True):
        return args
    print("已取消本次执行。")
    return None


def print_default_det_infer_summary(*, mode: str, data_path: Path | None = None) -> None:
    print("\n默认参数摘要")
    print(f"  input_mode: {mode}")
    print(f"  score_threshold: {rt.INFER_DEFAULT_SCORE_THRESHOLD}")
    print(f"  device: {rt.INFER_DEFAULT_DEVICE}")
    print("  save_visualization: True")
    print("  save_json: True")
    print("  save_txt: False")
    split_text = "交互选择 train / test / val / test+val / all" if mode == "dataset" else "(not used)"
    print(f"  split: {split_text}")
    selected_data = data_path if data_path is not None else rt.INFER_DEFAULT_DATA
    default_data = compact_display_path(selected_data) if mode == "dataset" else "(not used)"
    print(f"  data: {default_data}")
    print(f"  output_dir: (auto)")
    print("  report_path: (not used)")
    print("  compute_metrics: False")
    print("  save_test_report: False")
    print("  overwrite: False")


def print_default_det_eval_summary(*, data_path: Path, split: str) -> None:
    print("\n默认参数摘要")
    print(f"  data: {compact_display_path(data_path)}")
    print(f"  split: {split}")
    print(f"  score_threshold: {rt.INFER_DEFAULT_SCORE_THRESHOLD}")
    print(f"  report_iou_threshold: {rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD}")
    print("  compute_metrics: True")
    print("  metric_classwise: False")
    print("  save_test_report: True")
    print("  save_visualization: True")
    print(f"  vis_max_images: {rt.DET_EVAL_VIS_MAX_IMAGES}（每个 split）")
    print("  save_json: False")
    print("  save_txt: False")
    print(f"  device: {rt.INFER_DEFAULT_DEVICE}")
    print("  output_dir: (auto)")
    print("  overwrite: False")


def print_default_seg_eval_summary(
    *, seg_train_type: str, data_path: Path | None, splits: list[str]
) -> None:
    print("\n默认参数摘要")
    print(f"  seg_train_type: {seg_train_type}")
    print(f"  split: {' '.join(splits)}")
    selected_data = compact_display_path(data_path) if data_path is not None else "(auto)"
    print(f"  data: {selected_data}")
    if seg_train_type == "semantic":
        print("  threshold: 0.0（语义分割不过滤）")
    else:
        print(f"  threshold: {rt.DEFAULT_SEG_THRESHOLD}")
    if len(splits) > 1:
        print("  output_dir: <experiment_dir>/eval/<split>")
    else:
        print("  output_dir: <experiment_dir>/eval")
    print("  classwise: False")
    if seg_train_type == "semantic":
        max_vis = rt.SEG_EVAL_VIS_MAX_IMAGES
        print(f"  vis_max_images: {max_vis}（{'不限制' if max_vis <= 0 else '好/差各半，类别尽量全'}）")
    print(f"  device: {rt.DEFAULT_DEVICE}")
    print("  overwrite: False")


def build_seg_eval_cli_preview(args: argparse.Namespace) -> str:
    parts = ["python", "launcher.py", "eval", "--task", "seg"]
    parts.extend(["--seg-train-type", str(args.seg_train_type)])
    parts.extend(["--experiment-dir", str(args.experiment_dir)])
    if getattr(args, "checkpoint", None) is not None:
        parts.extend(["--checkpoint", str(args.checkpoint)])
    if getattr(args, "data", None) is not None:
        parts.extend(["--data", str(args.data)])
    splits = args.split if isinstance(args.split, (list, tuple)) else [args.split]
    parts.extend(["--split", *[str(item) for item in splits]])
    if getattr(args, "output_dir", None) is not None:
        parts.extend(["--output-dir", str(args.output_dir)])
    parts.extend(["--threshold", str(args.threshold)])
    if getattr(args, "classwise", False):
        parts.append("--classwise")
    parts.extend(["--vis-max-images", str(getattr(args, "vis_max_images", rt.SEG_EVAL_VIS_MAX_IMAGES))])
    parts.extend(["--device", str(args.device)])
    if getattr(args, "overwrite", False):
        parts.append("--overwrite")
    return " ".join(parts)


def _collect_existing_det_infer_outputs(
    *,
    experiment_dir: Path,
    data_path: Path,
    splits: tuple[str, ...] = ("train", "test", "val"),
) -> dict[str, Path]:
    try:
        from .det_problem_export import discover_infer_runs

        requested_data = data_path.expanduser().resolve()
        requested_root = requested_data.parent
        existing: dict[str, Path] = {}
        for run in discover_infer_runs(experiment_dir):
            split = str(run.get("split", ""))
            if split not in splits or split in existing:
                continue
            meta = run.get("run_meta", {})
            paths = meta.get("paths", {}) if isinstance(meta, dict) else {}
            if not isinstance(paths, dict):
                paths = {}
            recorded_data = paths.get("data_yaml")
            recorded_root = paths.get("data_root")
            data_matches = recorded_data is None and recorded_root is None
            recorded_candidates: list[Path] = []
            if recorded_data:
                recorded_data_path = Path(recorded_data).expanduser().resolve()
                recorded_candidates.append(recorded_data_path)
                data_matches = recorded_data_path == requested_data
            if recorded_root:
                recorded_root_path = Path(recorded_root).expanduser().resolve()
                recorded_candidates.append(recorded_root_path)
                data_matches = data_matches or recorded_root_path == requested_root
            if not data_matches and recorded_candidates and not any(path.exists() for path in recorded_candidates):
                requested_tokens = _tokenize_path_text(
                    f"{requested_root.parent.name} {requested_root.name}"
                )
                recorded_tokens = {
                    token
                    for path in recorded_candidates
                    for token in _tokenize_path_text(
                        f"{(path.parent if path.suffix else path).parent.name} "
                        f"{(path.parent if path.suffix else path).name}"
                    )
                }
                data_matches = bool(requested_tokens & recorded_tokens)
            output_dir = run.get("output_dir")
            if data_matches and isinstance(output_dir, Path) and output_dir.is_dir():
                existing[split] = output_dir
        return existing
    except Exception:
        return {}


def prompt_det_infer_split(*, experiment_dir: Path, data_path: Path) -> str:
    existing_outputs = _collect_existing_det_infer_outputs(
        experiment_dir=experiment_dir,
        data_path=data_path,
    )
    if existing_outputs:
        print("\n默认输出目录已有推理结果:")
        for split, output_dir in existing_outputs.items():
            print(f"  {split}: {compact_display_path(output_dir)}")
    return prompt_choice(
        "请选择数据集划分 --split",
        [
            ("train", "train"),
            ("test", "test"),
            ("val", "val"),
            ("test+val", "test+val"),
            ("all", "all (train + test + val)"),
        ],
    )


def build_det_cli_preview(args: argparse.Namespace) -> str:
    parts = ["python", "launcher.py", args.command]
    if args.command in {"infer", "eval"}:
        if args.command == "eval":
            parts.extend(["--task", "det"])
        parts.extend(["--experiment-dir", str(args.experiment_dir)])
        if args.checkpoint is not None:
            parts.extend(["--checkpoint", str(args.checkpoint)])
        if getattr(args, "image", None) is not None:
            parts.extend(["--image", str(args.image)])
        elif getattr(args, "image_dir", None) is not None:
            parts.extend(["--image-dir", str(args.image_dir)])
        elif getattr(args, "data", None) is not None:
            parts.extend(["--data", str(args.data), "--split", str(args.split)])
        if args.output_dir is not None:
            parts.extend(["--output-dir", str(args.output_dir)])
        parts.extend(["--score-threshold", str(args.score_threshold), "--device", str(args.device)])
        parts.extend(
            [
                "--bad-class-map50-threshold",
                str(getattr(args, "bad_class_map50_threshold", rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD)),
            ]
        )
        if args.command == "eval":
            parts.extend(["--report-iou-threshold", str(args.report_iou_threshold)])
            parts.extend(["--vis-max-images", str(args.vis_max_images)])
            if getattr(args, "metric_classwise", False):
                parts.append("--classwise")
            if not getattr(args, "save_visualization", True):
                parts.append("--skip-visualization")
            # eval 的 save_json 命令行默认是 False，只有打开时才需要在预览里体现。
            if getattr(args, "save_json", False):
                parts.append("--save-json")
        if getattr(args, "sahi", False):
            parts.append("--sahi")
    elif args.command == "export":
        if args.report_json is not None:
            parts.extend(["--report-json", str(args.report_json)])
        parts.extend(["--export-source-data", str(args.export_source_data)])
        parts.extend(["--target-total-images", str(args.target_total_images)])
        parts.extend(["--split-ratio", str(args.split_ratio)])
        parts.extend(["--good-class-threshold", str(args.good_class_threshold)])
        if args.auto_balance:
            parts.append("--auto-balance")
        else:
            parts.append("--no-auto-balance")
        if args.auto_relax_class_threshold:
            parts.append("--auto-relax-class-threshold")
        else:
            parts.append("--strict-class-threshold")
        parts.extend(["--balance-ratio", str(args.balance_ratio)])
        parts.extend(["--min-class-images", str(args.min_class_images)])
        parts.extend(["--min-class-boxes", str(args.min_class_boxes)])
        parts.extend(["--target-images-per-class", str(args.target_images_per_class)])
        parts.extend(["--target-boxes-per-class", str(args.target_boxes_per_class)])
        parts.extend(["--max-boxes-per-image", str(args.max_boxes_per_image)])
        parts.extend(["--max-boxes-per-class-per-image", str(args.max_boxes_per_class_per_image)])
        parts.extend(["--box-density-penalty", str(args.box_density_penalty)])
        parts.extend(["--size-ratio", str(args.size_ratio)])
        parts.extend(["--size-balance-weight", str(args.size_balance_weight)])
        parts.extend(["--avg-boxes-per-image-min", str(args.avg_boxes_per_image_min)])
        parts.extend(["--avg-boxes-per-image-max", str(args.avg_boxes_per_image_max)])
        parts.append("--trim-boxes" if getattr(args, "trim_boxes", False) else "--no-trim-boxes")
        parts.extend(["--export-suffix", str(args.export_suffix)])
    elif args.command == "eda":
        parts.extend(["--data", str(args.data)])
        if args.output_dir is not None:
            parts.extend(["--output-dir", str(args.output_dir)])
        if getattr(args, "overwrite", False):
            parts.append("--overwrite")
    return " ".join(parts)


def build_seg_export_cli_preview(args: argparse.Namespace) -> str:
    parts = ["python", "launcher.py", "seg-export"]
    if args.report_json is not None:
        parts.extend(["--report-json", str(args.report_json)])
    parts.extend(["--export-source-data", str(args.export_source_data)])
    parts.extend(["--target-total-images", str(args.target_total_images)])
    parts.extend(["--split-ratio", str(args.split_ratio)])
    parts.extend(["--good-class-threshold", str(args.good_class_threshold)])
    if args.auto_balance:
        parts.append("--auto-balance")
    else:
        parts.append("--no-auto-balance")
    if args.auto_relax_class_threshold:
        parts.append("--auto-relax-class-threshold")
    else:
        parts.append("--strict-class-threshold")
    parts.extend(["--balance-ratio", str(args.balance_ratio)])
    parts.extend(["--min-class-images", str(args.min_class_images)])
    parts.extend(["--min-class-instances", str(args.min_class_instances)])
    parts.extend(["--target-images-per-class", str(args.target_images_per_class)])
    parts.extend(["--target-instances-per-class", str(args.target_instances_per_class)])
    parts.extend(["--max-instances-per-image", str(args.max_instances_per_image)])
    parts.extend(["--max-instances-per-class-per-image", str(args.max_instances_per_class_per_image)])
    parts.extend(["--instance-density-penalty", str(args.instance_density_penalty)])
    parts.extend(["--size-ratio", str(args.size_ratio)])
    parts.extend(["--size-balance-weight", str(args.size_balance_weight)])
    parts.extend(["--avg-instances-per-image-min", str(args.avg_instances_per_image_min)])
    parts.extend(["--avg-instances-per-image-max", str(args.avg_instances_per_image_max)])
    parts.extend(["--export-suffix", str(args.export_suffix)])
    return " ".join(parts)


def _find_recent_test_reports(
    source_data_yaml: Path | None = None,
) -> list[Path]:
    """在 out/ 下搜索 test_report.json，按数据集关系排序。"""
    patterns = ["*test_report.json", "*val_report.json", "*train_report.json"]
    roots = [rt.EXPERIMENT_ROOT_DIR, rt.TEST_OUTPUT_ROOT_DIR]
    all_report_root = rt.ALL_REPORT_ROOT_DIR.resolve()
    seen: dict[str, Path] = {}
    for path in find_files(
        roots,
        label="索引 Det 评估报告",
        patterns=patterns,
        skip_dir_names={
            all_report_root.name,
            rt.IMPORTANT_ARTIFACT_DIRNAME,
            "old",
        },
    ):
        resolved = path.resolve()
        seen[str(resolved)] = resolved

    candidates = list(seen.values())
    if source_data_yaml is not None:
        candidates.sort(
            key=lambda p: (
                -_report_dataset_relation_score(p, source_data_yaml),
                -int(p.stat().st_mtime),
            )
        )
    else:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.expanduser().resolve().relative_to(parent.expanduser().resolve())
        return True
    except ValueError:
        return False


def _read_report_data_paths(report_json: Path) -> tuple[Path | None, Path | None]:
    payload = _load_json_dict(report_json)
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}
    data_root = _resolve_existing_path(config.get("data_root"))
    split_image_dir = _resolve_existing_path(config.get("test_images"))
    if data_root is not None:
        data_yaml = data_root / "data.yaml"
        return (data_yaml.resolve() if data_yaml.exists() else None), data_root.resolve()
    if split_image_dir is not None:
        for parent in split_image_dir.parents:
            data_yaml = parent / "data.yaml"
            if data_yaml.exists():
                return data_yaml.resolve(), parent.resolve()
    return None, None


def _metadata_data_paths_for_output_dir(output_dir: Path) -> tuple[Path | None, Path | None]:
    run_meta_path = output_dir.expanduser().resolve() / "run_meta.json"
    payload = _load_json_dict(run_meta_path) if run_meta_path.exists() else {}
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        return None, None
    data_yaml = _resolve_existing_path(paths_payload.get("data_yaml"))
    data_root = _resolve_existing_path(paths_payload.get("data_root"))
    if data_yaml is not None:
        data_yaml = data_yaml.resolve()
    if data_root is not None:
        data_root = data_root.resolve()
    return data_yaml, data_root


def _confusion_index_data_paths(index_path: Path) -> tuple[Path | None, Path | None]:
    payload = _load_json_dict(index_path)
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        return None, None
    data_yaml = _resolve_existing_path(paths_payload.get("data_yaml"))
    data_root = _resolve_existing_path(paths_payload.get("data_root"))
    if data_yaml is not None:
        data_yaml = data_yaml.resolve()
    if data_root is not None:
        data_root = data_root.resolve()
    return data_yaml, data_root


def _add_dataset_path_score(
    *,
    data_yaml: Path | None,
    data_root: Path | None,
    source_yaml: Path,
    source_root: Path,
) -> int:
    score = 0
    if data_yaml == source_yaml:
        score += 5000
    if data_root == source_root:
        score += 3000
    if data_root is not None and (_path_is_inside(source_root, data_root) or _path_is_inside(data_root, source_root)):
        score += 1200
    return score


def _report_dataset_relation_score(report_json: Path, source_data_yaml: Path) -> int:
    source_yaml = source_data_yaml.expanduser().resolve()
    source_root = source_yaml.parent
    source_tokens = _tokenize_path_text(source_root.name) | _tokenize_path_text(source_root.parent.name)
    score = 0

    report_data_yaml, report_data_root = _read_report_data_paths(report_json)
    score += _add_dataset_path_score(
        data_yaml=report_data_yaml,
        data_root=report_data_root,
        source_yaml=source_yaml,
        source_root=source_root,
    )

    output_dir = report_json.expanduser().resolve().parent
    meta_data_yaml, meta_data_root = _metadata_data_paths_for_output_dir(output_dir)
    score += _add_dataset_path_score(
        data_yaml=meta_data_yaml,
        data_root=meta_data_root,
        source_yaml=source_yaml,
        source_root=source_root,
    )

    for index_path in (output_dir / "confusion_inputs.json", output_dir.parent / "confusion_inputs.json"):
        if not index_path.exists():
            continue
        index_data_yaml, index_data_root = _confusion_index_data_paths(index_path)
        score += _add_dataset_path_score(
            data_yaml=index_data_yaml,
            data_root=index_data_root,
            source_yaml=source_yaml,
            source_root=source_root,
        )

    path_tokens = _tokenize_path_text(str(report_json))
    score += len(path_tokens & source_tokens) * 40
    return score


def _optimize_candidate_display_name(candidate: dict[str, Any]) -> str:
    display_name = candidate.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        return display_name
    report_path = candidate.get("report_json")
    if isinstance(report_path, Path):
        return compact_display_path(report_path)
    infer_output_dir = candidate.get("infer_output_dir")
    if isinstance(infer_output_dir, Path):
        return compact_display_path(infer_output_dir)
    return "unknown"


def _build_combined_optimize_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[Path, list[dict[str, Any]]] = {}
    for candidate in candidates:
        infer_output_dir = candidate.get("infer_output_dir")
        experiment_dir = candidate.get("experiment_dir")
        report_json = candidate.get("report_json")
        if not isinstance(infer_output_dir, Path) or not isinstance(experiment_dir, Path) or not isinstance(report_json, Path):
            continue
        split_name = str(candidate.get("split_name") or infer_output_dir.name)
        if split_name not in {"train", "test", "val"}:
            continue
        grouped.setdefault(experiment_dir, []).append(candidate)

    combined_candidates: list[dict[str, Any]] = []
    supplemental_candidates: list[dict[str, Any]] = []
    preferred_report_order = {"test": 0, "val": 1, "train": 2}
    for experiment_dir, group in grouped.items():
        infer_root = group[0]["infer_output_dir"].parent
        if not isinstance(infer_root, Path) or not infer_root.exists():
            continue
        original_group_len = len(group)
        _append_available_split_candidates(
            group,
            experiment_dir=experiment_dir,
            infer_root=infer_root,
        )
        supplemental_candidates.extend(group[original_group_len:])
        if len(group) < 2:
            continue
        if not _infer_output_has_prediction_json(infer_root):
            continue
        split_names = sorted(
            {
                str(item.get("split_name") or item["infer_output_dir"].name)
                for item in group
                if isinstance(item.get("infer_output_dir"), Path)
            },
            key=lambda name: preferred_report_order.get(name, 99),
        )
        best_report_candidate = sorted(
            group,
            key=lambda item: (
                preferred_report_order.get(str(item.get("split_name") or item["infer_output_dir"].name), 99)
                if isinstance(item.get("infer_output_dir"), Path)
                else 99,
                -int(item["report_json"].stat().st_mtime) if isinstance(item.get("report_json"), Path) else 0,
            ),
        )[0]
        dataset_tag = compact_display_path(experiment_dir)
        all_report_jsons = [
            item["report_json"] for item in group
            if isinstance(item.get("report_json"), Path)
        ]
        combined_candidates.append(
            {
                "report_json": best_report_candidate["report_json"],
                "report_jsons": all_report_jsons if len(all_report_jsons) > 1 else None,
                "infer_output_dir": infer_root,
                "experiment_dir": experiment_dir,
                "has_prediction_json": True,
                "display_name": f"{dataset_tag}  [联合: {' + '.join(split_names)}]",
                "candidate_kind": "combined",
                "split_names": split_names,
                "relation_score": max(int(item.get("relation_score") or 0) for item in group),
                "mtime": max(int(item.get("mtime") or 0) for item in group),
            }
        )
    return [*combined_candidates, *supplemental_candidates]


def _split_name_from_output_dir(output_dir: Path) -> str | None:
    run_meta_path = output_dir.expanduser().resolve() / "run_meta.json"
    payload = _load_json_dict(run_meta_path) if run_meta_path.exists() else {}
    split_value = payload.get("split")
    if isinstance(split_value, str) and split_value in {"train", "test", "val"}:
        return split_value
    if output_dir.name in {"train", "test", "val"}:
        return output_dir.name
    lowered = output_dir.name.lower()
    for split in ("train", "test", "val"):
        if lowered.endswith(f"-{split}") or lowered.endswith(f"_{split}"):
            return split
    return None


def _append_available_split_candidates(
    group: list[dict[str, Any]],
    *,
    experiment_dir: Path,
    infer_root: Path,
) -> None:
    existing_splits = {
        str(item.get("split_name"))
        for item in group
        if item.get("split_name") in {"train", "test", "val"}
    }
    relation_score = max((int(item.get("relation_score") or 0) for item in group), default=0)
    mtime = max((int(item.get("mtime") or 0) for item in group), default=0)
    for split in ("test", "val", "train"):
        if split in existing_splits:
            continue
        output_dir = infer_root / split
        if not output_dir.exists() or not _infer_output_has_prediction_json(output_dir):
            continue
        group.append(
            {
                "report_json": None,
                "infer_output_dir": output_dir,
                "experiment_dir": experiment_dir,
                "has_prediction_json": True,
                "display_name": f"{compact_display_path(output_dir)}  [单 split: {split}]",
                "candidate_kind": "single",
                "split_name": split,
                "relation_score": relation_score,
                "mtime": mtime,
            }
        )


def _dedupe_single_optimize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[Path | None, str, Path], dict[str, Any]] = {}
    for candidate in candidates:
        infer_output_dir = candidate.get("infer_output_dir")
        report_json = candidate.get("report_json")
        if not isinstance(infer_output_dir, Path):
            if isinstance(report_json, Path):
                infer_output_dir = report_json.parent
            else:
                continue
        split_name = str(candidate.get("split_name") or _split_name_from_output_dir(infer_output_dir) or infer_output_dir.name)
        experiment_dir = candidate.get("experiment_dir")
        key = (
            experiment_dir if isinstance(experiment_dir, Path) else None,
            split_name,
            infer_output_dir.expanduser().resolve(),
        )
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = candidate
            continue
        current_mtime = int(current.get("mtime") or 0)
        candidate_mtime = int(candidate.get("mtime") or 0)
        if candidate_mtime > current_mtime:
            best_by_key[key] = candidate
    return list(best_by_key.values())


def _sort_optimize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_order = {"test": 0, "val": 1, "train": 2}
    kind_order = {"combined": 0, "single": 1, "custom": 2}

    def _key(candidate: dict[str, Any]) -> tuple[int, int, str, int, int, str]:
        experiment_dir = candidate.get("experiment_dir")
        experiment_text = str(experiment_dir) if isinstance(experiment_dir, Path) else ""
        split_name = str(candidate.get("split_name") or "")
        split_count = len(candidate.get("split_names") or [])
        return (
            -int(candidate.get("relation_score") or 0),
            -int(candidate.get("mtime") or 0),
            experiment_text.lower(),
            kind_order.get(str(candidate.get("candidate_kind")), 9),
            split_order.get(split_name, 9) if split_count <= 1 else -split_count,
            _optimize_candidate_display_name(candidate).lower(),
        )

    return sorted(candidates, key=_key)


def _collect_optimize_analysis_candidates(source_data_yaml: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for report_path in _find_recent_test_reports(source_data_yaml):
        infer_output_dir = _infer_output_dir_from_report(report_path)
        output_dir = infer_output_dir if infer_output_dir is not None else report_path.parent
        split_name = _split_name_from_output_dir(report_path.parent) or _split_name_from_output_dir(output_dir)
        relation_score = _report_dataset_relation_score(report_path, source_data_yaml)
        mtime = int(report_path.stat().st_mtime)
        candidates.append(
            {
                "report_json": report_path,
                "infer_output_dir": infer_output_dir,
                "experiment_dir": _experiment_dir_from_report(report_path),
                "has_prediction_json": infer_output_dir is not None,
                "display_name": f"{compact_display_path(report_path)}  [单 split: {split_name or '?'}]",
                "candidate_kind": "single",
                "split_name": split_name,
                "relation_score": relation_score,
                "mtime": mtime,
            }
        )
    candidates = _dedupe_single_optimize_candidates(candidates)
    combined_candidates = _build_combined_optimize_candidates(candidates)
    return _sort_optimize_candidates([*combined_candidates, *candidates])


def _pick_best_optimize_analysis_candidate(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for candidate in candidates:
        if candidate["has_prediction_json"] and int(candidate.get("relation_score") or 0) > 0:
            return candidate
    return None


def _prompt_optimize_report_json(
    source_data_yaml: Path,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """让用户选择一个 optimize 分析候选，或跳过。"""
    if candidates is None:
        candidate_items = _collect_optimize_analysis_candidates(source_data_yaml)
    else:
        candidate_items = candidates
    print("\n[可选] 选择已有分析候选以复用 report 和混淆矩阵输入。")
    if not candidate_items:
        print("  当前列表为空，直接回车将自动运行 infer。")
        return None

    print("  可用候选（按数据集关系排序）：")
    for idx, candidate in enumerate(candidate_items, start=1):
        report_status = "含 report" if isinstance(candidate.get("report_json"), Path) else "仅预测 JSON"
        cm_status = "可直接构建混淆矩阵" if candidate["has_prediction_json"] else "需要补预测 JSON"
        print(f"    {idx}. {_optimize_candidate_display_name(candidate)}  [{cm_status}]")
        print(f"       {report_status}")
    print("  直接回车自动运行 infer，输入编号选择，输入 custom 手动输入 report")

    while True:
        raw = read_input("请选择: ").strip()
        if not raw:
            return None
        if raw.lower() == "custom":
            val = prompt_text("test_report.json 路径", None)
            if val:
                p = Path(val).expanduser().resolve()
                if p.exists():
                    infer_output_dir = _infer_output_dir_from_report(p)
                    return {
                        "report_json": p,
                        "infer_output_dir": infer_output_dir,
                        "experiment_dir": _experiment_dir_from_report(p),
                        "has_prediction_json": infer_output_dir is not None,
                        "display_name": compact_display_path(p),
                        "candidate_kind": "custom",
                    }
                print(f"  文件不存在: {p}")
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(candidate_items):
                return candidate_items[idx]
        print("  无效选择，请重新输入。")


def _prompt_optimize_infer_output_dir() -> Path | None:
    """让用户选择 infer 输出目录用于混淆矩阵分析，或跳过。"""
    print("\n[可选] 选择推理输出目录以构建混淆矩阵（需要推理时开启了 --save-json）。")
    print("  直接回车跳过（跳过后仅根据 AP/F1 分析劣质类别，不分析合并候选）")

    raw = read_input("infer 输出目录路径（回车跳过）: ").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (rt.ROOT_DIR / p).resolve()
    else:
        p = p.resolve()
    if not p.exists():
        print(f"  目录不存在，已跳过: {p}")
        return None
    return p


def _is_prediction_json(path: Path) -> bool:
    payload = _load_json_dict(path)
    return isinstance(payload.get("predictions"), list)


def _infer_output_has_prediction_json(output_dir: Path) -> bool:
    output_dir = output_dir.expanduser().resolve()
    search_dirs = [output_dir / rt.INFER_TEMP_DIRNAME / "json", output_dir]
    seen: set[Path] = set()
    for json_path in find_files(
        search_dirs,
        label="索引预测 JSON",
        patterns=["*.json"],
    ):
        resolved = json_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if json_path.name in {"run_meta.json", "metrics_summary.json"}:
            continue
        if "report" in json_path.name or "summary" in json_path.name or "manifest" in json_path.name:
            continue
        if _is_prediction_json(json_path):
            return True
    return False


def _confusion_input_dirs_from_index(index_path: Path) -> list[Path]:
    payload = _load_json_dict(index_path)
    paths_payload = payload.get("paths")
    dirs: list[Path] = []
    if not isinstance(paths_payload, dict):
        paths_payload = {}
    for key in ("output_dir", "prediction_json_dir"):
        path = _resolve_existing_path(paths_payload.get(key))
        if path is not None:
            dirs.append(path)
    split_outputs = payload.get("split_outputs")
    if isinstance(split_outputs, list):
        for item in split_outputs:
            if not isinstance(item, dict):
                continue
            for key in ("output_dir", "prediction_json_dir"):
                path = _resolve_existing_path(item.get(key))
                if path is not None:
                    dirs.append(path)
    return dirs


def _infer_output_dir_from_report(report_json: Path) -> Path | None:
    """从 test_report 附近自动推断可用于混淆矩阵的 infer 输出目录。"""
    report_json = report_json.expanduser().resolve()
    candidates: list[Path] = []

    confusion_inputs_path = report_json.parent / "confusion_inputs.json"
    if confusion_inputs_path.exists():
        candidates.extend(_confusion_input_dirs_from_index(confusion_inputs_path))

    run_meta_path = report_json.parent / "run_meta.json"
    run_meta = _load_json_dict(run_meta_path) if run_meta_path.exists() else {}
    artifacts_payload = run_meta.get("artifacts")
    if isinstance(artifacts_payload, dict):
        confusion_inputs = _resolve_existing_path(artifacts_payload.get("confusion_inputs"))
        if confusion_inputs is not None:
            candidates.extend(_confusion_input_dirs_from_index(confusion_inputs))
    paths_payload = run_meta.get("paths")
    if isinstance(paths_payload, dict):
        for key in ("output_dir", "temp_dir"):
            path = _resolve_existing_path(paths_payload.get(key))
            if path is None:
                continue
            candidates.append(path.parent if path.name == rt.INFER_TEMP_DIRNAME else path)

    candidates.append(report_json.parent)

    root_confusion_inputs_path = report_json.parent.parent / "confusion_inputs.json"
    if root_confusion_inputs_path.exists():
        candidates.extend(_confusion_input_dirs_from_index(root_confusion_inputs_path))

    candidates.append(report_json.parent.parent)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _infer_output_has_prediction_json(resolved):
            return resolved
    return None


def _experiment_dir_from_report(report_json: Path) -> Path | None:
    run_meta_path = report_json.expanduser().resolve().parent / "run_meta.json"
    run_meta = _load_json_dict(run_meta_path) if run_meta_path.exists() else {}
    paths_payload = run_meta.get("paths")
    if not isinstance(paths_payload, dict):
        return None
    recorded = _resolve_existing_path(paths_payload.get("experiment_dir"))
    if recorded is not None:
        return recorded
    for parent in report_json.expanduser().resolve().parents:
        if rt.is_experiment_dir(parent):
            return parent
    return None


def _cleanup_optimize_temp_dir(temp_dir: Path | None) -> None:
    if temp_dir is not None and temp_dir.exists():
        shutil.rmtree(temp_dir)
    if temp_dir is not None:
        _PENDING_OPTIMIZE_TEMP_DIRS.discard(temp_dir)


def cleanup_pending_optimize_temp_dirs() -> None:
    """Remove auto-infer workspaces left by cancellation or argument-build errors."""
    for temp_dir in list(_PENDING_OPTIMIZE_TEMP_DIRS):
        _cleanup_optimize_temp_dir(temp_dir)


def _prompt_optimize_infer_split() -> str:
    return prompt_choice(
        "请选择用于混淆矩阵的推理范围",
        [
            ("train", "train"),
            ("test", "test"),
            ("val", "val"),
            ("test+val", "test+val"),
            ("all", "all (train + test + val)"),
        ],
    )


def _find_auto_infer_report(temp_dir: Path, splits: list[str]) -> Path | None:
    preferred_splits = [split for split in ("test", "val", "train") if split in splits]
    for split in preferred_splits:
        split_dir = temp_dir / split
        for pattern in ("*test_report.json", "*val_report.json", "*train_report.json"):
            candidates = sorted(
                split_dir.rglob(pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
    for pattern in ("*test_report.json", "*val_report.json", "*train_report.json"):
        candidates = sorted(
            temp_dir.rglob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def _run_auto_infer_for_optimize(
    source_data_yaml: Path,
    *,
    report_json_for_quality: Path | None = None,
    experiment_dir: Path | None = None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    """为 optimize 自动运行推理，返回 (test_report_path, infer_output_dir, experiment_dir, temp_dir)。

    infer 输出统一写入系统临时目录，optimize 完成后由调用方负责清理。
    """
    from . import det_infer  # 延迟导入，避免循环

    print("\n[auto-infer] 自动运行推理以生成 test_report.json 和预测 JSON（用于混淆矩阵分析）。")
    if experiment_dir is None:
        experiment_dir = prompt_experiment_dir("det", rt.INFER_DEFAULT_EXPERIMENT_DIR)

    split_choice = _prompt_optimize_infer_split()
    rt.import_runtime_dependencies()
    data_cfg = rt.load_data_config(source_data_yaml)
    splits = det_infer.resolve_dataset_infer_splits(data_cfg, split_choice)

    temp_dir = Path(tempfile.mkdtemp(prefix="lightly-train-optimize-", dir="/tmp"))
    _PENDING_OPTIMIZE_TEMP_DIRS.add(temp_dir)
    save_test_report = report_json_for_quality is None

    print("\n[auto-infer] 配置摘要")
    print(f"  experiment_dir : {compact_display_path(experiment_dir)}")
    print(f"  data           : {compact_display_path(source_data_yaml)}")
    print(f"  split          : {split_choice} -> {', '.join(splits)}")
    print(f"  output_dir     : {compact_display_path(temp_dir)}  （临时目录，分析后自动清理）")
    print(f"  score_threshold: {rt.INFER_DEFAULT_SCORE_THRESHOLD}")
    print(f"  save_json      : True （用于混淆矩阵分析）")
    print(f"  save_test_report: {save_test_report}")

    if not prompt_yes_no("确认运行推理", True):
        _cleanup_optimize_temp_dir(temp_dir)
        return None, None, None, None

    infer_args = argparse.Namespace(
        tool_task="det",
        tool_action="infer",
        command="infer",
        experiment_dir=experiment_dir,
        checkpoint=None,
        image=None,
        image_dir=None,
        data=source_data_yaml,
        split=split_choice,
        output_dir=temp_dir,
        score_threshold=rt.INFER_DEFAULT_SCORE_THRESHOLD,
        device=rt.INFER_DEFAULT_DEVICE,
        save_visualization=False,
        save_json=True,
        save_txt=False,
        compute_metrics=save_test_report,
        metric_classwise=False,
        report_iou_threshold=rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD,
        bad_class_map50_threshold=rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD,
        save_test_report=save_test_report,
        report_path=None,
        overwrite=True,
        infer_config_mode="default",
        shard_index=None,
        num_shards=1,
        dry_run=False,
        skip_important_artifacts=True,
        selected_splits=None,
        multi_output_root=None,
    )
    try:
        det_infer.run_infer(infer_args)
    except Exception:
        _cleanup_optimize_temp_dir(temp_dir)
        raise

    # 在 temp_dir 下找最新生成的 test_report
    report_json: Path | None = report_json_for_quality
    infer_output_dir: Path | None = None
    if save_test_report and temp_dir.exists():
        report_json = _find_auto_infer_report(temp_dir, splits)
        if report_json is not None:
            infer_output_dir = temp_dir
    elif _infer_output_has_prediction_json(temp_dir):
        infer_output_dir = temp_dir

    if report_json is None and save_test_report:
        print("\n[auto-infer] 警告：未找到 test_report.json，请手动指定路径。")
    if infer_output_dir is None:
        print("\n[auto-infer] 警告：未找到预测 JSON。")
    else:
        print(f"\n[auto-infer] 完成！")
        if report_json is not None:
            print(f"  test_report    : {compact_display_path(report_json)}")
        print(f"  infer_output_dir: {compact_display_path(infer_output_dir)}")
        print("  临时目录将在 optimize 完成后自动清理")

    return report_json, infer_output_dir, experiment_dir, temp_dir


def _prompt_gpu_selection(available_gpus: list[tuple[int, str]]) -> list[int] | str:
    """让用户选择使用哪几张 GPU，返回 list[int] 或 "auto"。"""
    if not available_gpus:
        print("\n未检测到可用 GPU，将使用 CPU 训练。")
        return "auto"

    print("\n检测到以下 GPU：")
    for idx, name in available_gpus:
        print(f"  {idx}: {name}")

    if len(available_gpus) == 1:
        use_it = prompt_yes_no(f"使用 GPU 0 ({available_gpus[0][1]}) 训练", True)
        return [0] if use_it else "auto"

    print("  输入编号（逗号分隔，如 0,1）选择指定 GPU，直接回车使用全部 GPU")
    while True:
        raw = read_input("请选择 GPU: ").strip()
        if not raw:
            return "auto"
        parts = [p.strip() for p in raw.split(",")]
        try:
            indices = [int(p) for p in parts if p]
        except ValueError:
            print("  请输入数字编号，例如 0 或 0,1")
            continue
        valid_indices = {idx for idx, _ in available_gpus}
        invalid = [i for i in indices if i not in valid_indices]
        if invalid:
            print(f"  无效编号: {invalid}，请重新输入。")
            continue
        if not indices:
            return "auto"
        return indices


def _prompt_train_duration(
    data_yaml: Path,
) -> tuple[int | str, int | str]:
    """询问训练时长（epoch 模式或 step 模式），返回 (steps, batch_size)。

    epoch 模式：由用户输入 epochs + batch_size → 自动换算 steps。
    step 模式 ：由用户直接输入 steps（可为 "auto"）。

    epoch 模式下若数据集图片数读取失败，会显式降级到 step 输入而不是错误地把
    epochs 当 steps 用（之前的兜底会让 10 epochs 跑成 10 steps）。
    """
    from . import train_tools

    def _prompt_step_input(prompt: str = "训练步数 --steps [auto]: ") -> int | str:
        raw = read_input(prompt).strip()
        if not raw or raw.lower() == "auto":
            return "auto"
        try:
            return int(raw)
        except ValueError:
            print("  请输入整数或 auto，默认改为 auto。")
            return "auto"

    mode = prompt_choice(
        "请选择训练时长输入方式",
        [
            ("epoch", "epoch  输入轮数，自动换算 steps"),
            ("step",  "step   直接输入 steps（可输入 auto）"),
        ],
    )

    if mode == "step":
        return _prompt_step_input(), "auto"

    # epoch 模式
    epochs = prompt_int("训练轮数 --epochs", 10)
    batch_size = prompt_int("批大小 --batch-size（epoch 换算需要，不可为 auto）", 32)

    try:
        steps_calc, num_train = train_tools.estimate_steps_from_epochs(data_yaml, epochs, batch_size)
    except Exception as exc:
        steps_calc, num_train = None, 0
        print(f"  ⚠ 换算出错：{exc}")

    if steps_calc is None:
        print(f"  ⚠ 无法从 {compact_display_path(data_yaml)} 读取 train 图片数，epoch→steps 失败。")
        print( "    请确认 data.yaml 的 train 路径正确，或直接输入步数。")
        return _prompt_step_input("退回 step 模式，请输入 --steps [auto]: "), batch_size

    print(
        f"  换算结果: {epochs} epochs × ⌈{num_train} 图 ÷ batch {batch_size}⌉"
        f" = {steps_calc} steps"
    )
    return steps_calc, batch_size


def _prompt_train_model(recent_models: list[str], default: str) -> str:
    """让用户选择或输入模型字符串。"""
    if recent_models:
        print("\n最近使用过的模型：")
        for idx, m in enumerate(recent_models, start=1):
            print(f"  {idx}. {m}")
        print("  输入编号选择，回车使用默认，或直接输入新模型名称")
    else:
        print(f"\n模型名称示例: dinov3/vits16-ltdetr, dinov3/vitl16-ltdetr")

    while True:
        raw = read_input(f"模型 [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(recent_models):
                return recent_models[idx]
            print("  编号超出范围，请重新输入。")
            continue
        return raw


def _prompt_seg_train_type(default: str | None = None) -> str:
    default = (default or rt.SEG_TRAIN_TYPE or "instance").lower()
    options = [
        ("semantic", "semantic 语义分割（PNG mask）"),
        ("instance", "instance 实例分割（YOLO polygon txt）"),
    ]
    if default == "instance":
        options = [options[1], options[0]]
    return prompt_choice("请选择 seg 训练类型", options)


def _default_train_data_yaml(
    task: str,
    seg_train_type: str | None = None,
    *,
    preferred_experiment_dir: Path | None = None,
) -> Path:
    if task == "seg" and seg_train_type == "semantic":
        configured = Path(rt.SEMANTIC_SEG_DEFAULT_DATA)
        candidate_task = "semantic"
    elif task == "seg":
        configured = Path(rt.SEG_EXPORT_DEFAULT_SOURCE_DATA)
        candidate_task = seg_train_type or "instance"
    else:
        configured = Path(rt.INFER_DEFAULT_DATA)
        candidate_task = task
    if configured.exists():
        return configured
    candidates = list_dataset_yaml_candidates(
        candidate_task, preferred_experiment_dir=preferred_experiment_dir,
    )
    if candidates:
        return candidates[0]
    return configured


def _default_train_model(task: str, seg_train_type: str | None = None) -> str:
    if task == "det":
        return "dinov3/vits16-ltdetr"
    if task == "seg" and seg_train_type == "semantic":
        return "dinov3/vits16-eomt"
    if task == "seg":
        return "dinov3/vits16-eomt"
    return "dinov3/vits16"


def _seg_train_type_from_logged_task(task_name: object) -> str:
    text = str(task_name or "").lower()
    if "semantic" in text:
        return "semantic"
    if "instance" in text:
        return "instance"
    return rt.SEG_TRAIN_TYPE


def _prompt_backbone_weights(weight_files: list[Path]) -> Path | None:
    """让用户选择骨干预训练权重文件，或跳过。"""
    print("\n[可选] 选择骨干预训练权重文件（回车跳过）：")
    if not weight_files:
        print("  未在 weights/ 目录下发现权重文件。")
    else:
        for idx, p in enumerate(weight_files, start=1):
            print(f"  {idx}. {compact_display_path(p)}")
    print("  输入编号选择，回车跳过，输入 custom 手动输入路径")

    while True:
        raw = read_input("请选择: ").strip()
        if not raw:
            return None
        if raw.lower() == "custom":
            path_raw = read_input("权重文件路径: ").strip()
            if not path_raw:
                return None
            p = Path(path_raw).expanduser()
            return p.resolve() if p.is_absolute() else (rt.ROOT_DIR / p).resolve()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(weight_files):
                return weight_files[idx]
        print("  无效选择，请重新输入。")


def _build_train_args(task: str) -> argparse.Namespace | None:
    """交互收集训练参数，返回 Namespace 或 None（用户取消）。"""
    from . import train_tools

    print(f"\n{task}/train 训练参数配置")

    # 检测 GPU（供后续选择使用）
    available_gpus = train_tools.detect_available_gpus()

    # 先选模式：新训练 or 续跑
    train_mode = prompt_choice(
        "请选择训练模式",
        [
            ("new",    "new    新建训练"),
            ("resume", "resume 续跑（从已有实验继续）"),
        ],
    )

    # ------------------------------------------------------------------ #
    # 路径 A：新训练
    # ------------------------------------------------------------------ #
    if train_mode == "new":
        seg_train_type = _prompt_seg_train_type() if task == "seg" else None
        data_yaml = prompt_dataset_yaml(
            seg_train_type if task == "seg" else task,
            _default_train_data_yaml(task, seg_train_type),
        )
        # A1. 数据集

        # A2. 模型
        recent_models = train_tools.discover_recent_models(task)
        default_model = recent_models[0] if recent_models else _default_train_model(task, seg_train_type)
        model = _prompt_train_model(recent_models, default_model)

        # A3. 骨干权重（可选）
        weight_files = train_tools.discover_weight_files()
        backbone_weights = _prompt_backbone_weights(weight_files)

        # A4. 训练时长
        steps, batch_size = _prompt_train_duration(data_yaml)

        # A5. GPU 选择
        devices = _prompt_gpu_selection(available_gpus)

        # A6. 输出目录（循环：若存在且非空，允许覆盖 / 输入新路径 / 取消）
        default_out = train_tools.build_default_out_dir(data_yaml, model)
        overwrite = False
        out_dir: Path | None = None
        while out_dir is None:
            try:
                default_out_display = str(default_out.relative_to(rt.ROOT_DIR))
            except ValueError:
                default_out_display = str(default_out)
            out_raw = read_input(f"\n训练输出目录 [{default_out_display}]: ").strip()
            if not out_raw:
                candidate = default_out
            else:
                p = Path(out_raw).expanduser()
                candidate = p.resolve() if p.is_absolute() else (rt.ROOT_DIR / p).resolve()

            if candidate.exists() and any(candidate.iterdir()):
                choice = prompt_choice(
                    f"输出目录已存在且非空：{compact_display_path(candidate)}",
                    [
                        ("overwrite", "overwrite 覆盖（清空后重新训练）"),
                        ("rename",    "rename    输入另一个目录名"),
                        ("cancel",    "cancel    取消本次训练"),
                    ],
                )
                if choice == "overwrite":
                    overwrite = True
                    out_dir = candidate
                elif choice == "cancel":
                    print("已取消。")
                    return None
                else:
                    default_out = candidate  # 把当前选择作为下一次默认，方便手动加后缀
                    continue
            else:
                out_dir = candidate

        args = argparse.Namespace(
            tool_task=task,
            tool_action="train",
            seg_train_type=seg_train_type,
            data_yaml=data_yaml,
            model=model,
            backbone_weights=backbone_weights,
            out_dir=out_dir,
            steps=steps,
            batch_size=batch_size,
            num_workers="auto",
            devices=devices,
            checkpoint=None,
            overwrite=overwrite,
            resume_interrupted=False,
        )

        # A7. 确认
        gpu_display = (
            ", ".join(f"{i}:{name}" for i, name in available_gpus if i in devices)
            if isinstance(devices, list) else "auto (全部)"
        )
        print(f"\n{task}/train 新训练配置确认")
        if task == "seg":
            print(f"  seg_train_type  : {seg_train_type}")
        print(f"  data_yaml       : {compact_display_path(data_yaml)}")
        print(f"  model           : {model}")
        print(f"  backbone_weights: {compact_display_path(backbone_weights) if backbone_weights else '(无)'}")
        print(f"  steps           : {steps}")
        print(f"  batch_size      : {batch_size}")
        print(f"  devices         : {devices}  ({gpu_display})")
        print(f"  out_dir         : {compact_display_path(out_dir)}")
        if overwrite:
            print(f"  overwrite       : True")

        if prompt_yes_no("确认开始训练吗", True):
            return args
        print("已取消。")
        return None

    # ------------------------------------------------------------------ #
    # 路径 B：续跑
    # ------------------------------------------------------------------ #
    # B1. 选择实验目录（要求有 checkpoint）
    all_dirs = list_experiment_dirs(task=task)
    checkpoint_dirs = [d for d in all_dirs if rt.is_experiment_dir(d, require_checkpoint=True)]
    if not checkpoint_dirs:
        checkpoint_dirs = [d for d in list_experiment_dirs() if rt.is_experiment_dir(d, require_checkpoint=True)]

    if not checkpoint_dirs:
        print("未找到含 checkpoint 的实验目录，请先完成至少一次训练。")
        return None

    src_experiment_dir = prompt_experiment_dir(task, checkpoint_dirs[0])
    orig_params = train_tools.read_original_train_params(src_experiment_dir)
    seg_train_type = (
        _seg_train_type_from_logged_task(orig_params.get("task"))
        if task == "seg"
        else None
    )

    # 读取原始训练参数并回显
    if orig_params:
        print("\n原始训练参数摘要：")
        for k in ("model", "steps", "batch_size", "devices", "data"):
            if k in orig_params:
                print(f"  {k}: {orig_params[k]}")
    else:
        print("\n⚠ 未能从 train.log 解析到原始训练参数；续跑可能缺少 model/data 字段。")

    # 解析原始 data 路径：可能是相对路径，针对 ROOT_DIR 还原
    orig_data_raw = orig_params.get("data")
    orig_data_path: Path | None = None
    if isinstance(orig_data_raw, str) and orig_data_raw.strip():
        cand = Path(orig_data_raw).expanduser()
        orig_data_path = cand.resolve() if cand.is_absolute() else (rt.ROOT_DIR / cand).resolve()

    # B2. 是否修改参数
    keep_params = prompt_yes_no("是否保持原参数不变（resume_interrupted 模式）", True)

    if keep_params:
        if not orig_params.get("model"):
            print("无法续跑：原 train.log 没有 model 字段，请改用「修改参数」从 checkpoint 微调。")
            return None
        data_config = orig_params.get("data_config")
        if (orig_data_path is None or not orig_data_path.exists()) and data_config is None:
            print(f"无法续跑：原数据集路径无效或不存在 ({orig_data_raw})。请改用「修改参数」并重新指定数据集。")
            return None

        # B2a. 仅选 GPU
        devices = _prompt_gpu_selection(available_gpus)

        args = argparse.Namespace(
            tool_task=task,
            tool_action="train",
            seg_train_type=seg_train_type,
            data_yaml=orig_data_path,
            data_config=data_config if orig_data_path is None else None,
            model=orig_params["model"],
            backbone_weights=None,
            out_dir=src_experiment_dir,
            steps=orig_params.get("steps", "auto"),
            batch_size=orig_params.get("batch_size", "auto"),
            num_workers="auto",
            devices=devices,
            checkpoint=None,
            overwrite=False,
            resume_interrupted=True,
        )

        gpu_display = (
            ", ".join(f"{i}:{name}" for i, name in available_gpus if i in devices)
            if isinstance(devices, list) else "auto (全部)"
        )
        print(f"\n{task}/train 续跑（保持原参数）配置确认")
        print(f"  mode            : resume_interrupted")
        if task == "seg":
            print(f"  seg_train_type  : {seg_train_type}")
        print(f"  out_dir         : {compact_display_path(src_experiment_dir)}")
        print(f"  model           : {args.model}")
        print(f"  steps           : {args.steps}")
        print(f"  devices         : {devices}  ({gpu_display})")

        if prompt_yes_no("确认开始训练吗", True):
            return args
        print("已取消。")
        return None

    # 修改参数：fine-tune from checkpoint
    # 找到 last.ckpt
    ckpt_candidates = rt.experiment_checkpoint_candidates(src_experiment_dir)
    # 优先 last.ckpt，然后 last.pt，然后 best
    last_ckpt: Path | None = None
    for cand in ckpt_candidates:
        if "last" in cand.name and cand.exists():
            last_ckpt = cand
            break
    if last_ckpt is None:
        for cand in ckpt_candidates:
            if cand.exists():
                last_ckpt = cand
                break
    if last_ckpt is None:
        print("未找到可用 checkpoint 文件，无法续跑。")
        return None

    # 数据集（默认复用原实验的；若原路径无效则强制重选）
    if orig_data_path is None:
        orig_data_path = _default_train_data_yaml(task, seg_train_type)
        if not orig_data_path.is_absolute():
            orig_data_path = (rt.ROOT_DIR / orig_data_path).resolve()
    if not orig_data_path.exists():
        print(f"原数据集不存在: {compact_display_path(orig_data_path)}，请重新指定。")
        data_yaml = prompt_dataset_yaml(seg_train_type if task == "seg" else task, orig_data_path)
    else:
        reuse_data = prompt_yes_no(f"复用原数据集 [{compact_display_path(orig_data_path)}]", True)
        data_yaml = orig_data_path if reuse_data else prompt_dataset_yaml(
            seg_train_type if task == "seg" else task, orig_data_path
        )

    # B2c. 训练时长
    steps, batch_size = _prompt_train_duration(data_yaml)

    # B2d. GPU 选择
    devices = _prompt_gpu_selection(available_gpus)

    # B2e. 模型字符串（原 train.log 缺失时让用户补）
    orig_model = (orig_params.get("model") or "").strip()
    if not orig_model:
        print("\n原 train.log 没有 model 字段，请手动输入要使用的模型。")
        recent_models = train_tools.discover_recent_models(task)
        default_model = recent_models[0] if recent_models else _default_train_model(task, seg_train_type)
        orig_model = _prompt_train_model(recent_models, default_model)

    # B2f. 新输出目录（与 path A 一致：存在且非空时给覆盖 / 改名 / 取消三选项）
    default_out = train_tools.build_default_out_dir(data_yaml, orig_model or "ft")
    overwrite = False
    out_dir: Path | None = None
    while out_dir is None:
        try:
            default_out_display = str(default_out.relative_to(rt.ROOT_DIR))
        except ValueError:
            default_out_display = str(default_out)
        out_raw = read_input(f"\n新训练输出目录 [{default_out_display}]: ").strip()
        if not out_raw:
            candidate = default_out
        else:
            p = Path(out_raw).expanduser()
            candidate = p.resolve() if p.is_absolute() else (rt.ROOT_DIR / p).resolve()

        if candidate.exists() and any(candidate.iterdir()):
            choice = prompt_choice(
                f"输出目录已存在且非空：{compact_display_path(candidate)}",
                [
                    ("overwrite", "overwrite 覆盖（清空后重新训练）"),
                    ("rename",    "rename    输入另一个目录名"),
                    ("cancel",    "cancel    取消本次训练"),
                ],
            )
            if choice == "overwrite":
                overwrite = True
                out_dir = candidate
            elif choice == "cancel":
                print("已取消。")
                return None
            else:
                default_out = candidate
                continue
        else:
            out_dir = candidate

    args = argparse.Namespace(
        tool_task=task,
        tool_action="train",
        seg_train_type=seg_train_type,
        data_yaml=data_yaml,
        model=orig_model,
        backbone_weights=None,
        out_dir=out_dir,
        steps=steps,
        batch_size=batch_size,
        num_workers="auto",
        devices=devices,
        checkpoint=last_ckpt,
        overwrite=overwrite,
        resume_interrupted=False,
    )

    gpu_display = (
        ", ".join(f"{i}:{name}" for i, name in available_gpus if i in devices)
        if isinstance(devices, list) else "auto (全部)"
    )
    print(f"\n{task}/train 续跑（修改参数）配置确认")
    print(f"  mode            : finetune from checkpoint")
    if task == "seg":
        print(f"  seg_train_type  : {seg_train_type}")
    print(f"  checkpoint      : {compact_display_path(last_ckpt)}")
    print(f"  data_yaml       : {compact_display_path(data_yaml)}")
    print(f"  model           : {orig_model}")
    print(f"  steps           : {steps}")
    print(f"  batch_size      : {batch_size}")
    print(f"  devices         : {devices}  ({gpu_display})")
    print(f"  out_dir         : {compact_display_path(out_dir)}")
    if overwrite:
        print(f"  overwrite       : True")

    if prompt_yes_no("确认开始训练吗", True):
        return args
    print("已取消。")
    return None


def _build_clean_args() -> argparse.Namespace | None:
    """收集实验清理计划；实际删除由 dispatch 调用 exp_cleaner 执行。"""
    from . import exp_cleaner

    # Step 1: 扫描并生成报告
    print("\n正在扫描实验目录...")
    analyses = exp_cleaner.scan_experiments()
    if not analyses:
        print("未扫描到任何实验目录，无需清理。")
        return None

    exp_cleaner.print_clean_report(analyses)

    # Step 2: 用户标记重要实验
    important_indices: set[int] = set()
    while True:
        print("请输入要保留完整文件的重要实验编号（逗号分隔，如 1,4；留空表示全部进入清理候选）:")
        raw = read_input("> ").strip()
        if not raw:
            break
        parts = [part.strip() for part in raw.split(",")]
        invalid = [
            part
            for part in parts
            if not part.isdigit() or not 1 <= int(part) <= len(analyses)
        ]
        if invalid:
            print(f"输入包含无效编号: {', '.join(repr(part) for part in invalid)}，请重新输入。")
            continue
        important_indices = {int(part) - 1 for part in parts}
        break

    if important_indices:
        print("\n已标记为重要（跳过清理）:")
        for idx in sorted(important_indices):
            print(f"  ✅ {exp_cleaner.compact_display(analyses[idx].exp_dir)}")
        print()

    # 过滤出待清理的实验
    to_clean = [a for i, a in enumerate(analyses) if i not in important_indices]
    if not to_clean:
        print("所有实验都已标记为重要，无需清理。")
        return None

    # Step 3: 展示清理预览
    exp_cleaner.print_clean_preview(to_clean)

    total_cleanable = sum(a.cleanable_size for a in to_clean)
    print(f"总计释放: {exp_cleaner.format_size(total_cleanable)}")
    print()

    # Step 4: 确认
    confirm = prompt_yes_no("确认执行清理吗", False)
    if not confirm:
        print("已取消清理。")
        return None

    return argparse.Namespace(
        tool_task="clean",
        tool_action="clean",
        analyses=to_clean,
        dry_run=False,
    )


def build_interactive_args() -> argparse.Namespace | None:
    task = prompt_choice(
        "请选择任务类型",
        [("cls", "cls 分类"), ("det", "det 检测"), ("seg", "seg 分割"), ("clean", "clean 实验清理（释放磁盘空间）")],
    )
    if task == "clean":
        return _build_clean_args()
    action_options = [("train", "train 训练"), ("infer", "infer 推理")]
    if task in {"cls", "det", "seg"}:
        action_options.append(("eval", "eval 评估"))
    if task == "seg":
        action_options.append(("eda", "EDA 分割数据分析（自动识别语义/实例）"))
        action_options.append(("curate", "curate 语义分割交互式整理"))
        action_options.append(("export", "export 数据集筛选"))
    if task == "det":
        action_options.append(("eda", "EDA 数据集分析"))
        action_options.append(("export", "export 数据集筛选"))
        action_options.append(("optimize", "optimize 训练后数据集优化（合并/删除类别）"))
        action_options.append(("review-sample", "review-sample 数据集质检抽样"))
        action_options.append(("report", "report 生成实验报告"))
    action = prompt_choice(f"请选择 {task} 功能", action_options)

    if action == "train":
        return _build_train_args(task)
    if task == "det" and action == "review-sample":
        return build_interactive_review_sample_args()

    if task == "cls" and action == "eval":
        experiment_dir = prompt_experiment_dir(
            "cls", rt.EXPERIMENT_ROOT_DIR / "my_experiment_cls"
        )
        test_dir = prompt_required_path("测试图片目录（按类别子目录组织） --test-dir")
        default_output = experiment_dir / "eval"
        output_raw = prompt_text("输出目录 --output-dir", str(default_output))
        args = argparse.Namespace(
            tool_task="cls",
            tool_action="eval",
            command="eval",
            experiment_dir=experiment_dir,
            checkpoint=None,
            test_dir=test_dir,
            output_dir=Path(output_raw or str(default_output)).expanduser(),
            threshold=prompt_float("分类阈值 --threshold", rt.DEFAULT_CLS_THRESHOLD),
            topk=prompt_int("topk", 1),
            device=prompt_text("推理设备 --device", rt.DEFAULT_DEVICE) or rt.DEFAULT_DEVICE,
            overwrite=False,
        )
        return confirm_args("cls/eval", args)
    if task == "cls" and action == "infer":
        experiment_dir = prompt_experiment_dir("cls", rt.EXPERIMENT_ROOT_DIR / "my_experiment_cls")
        mode = prompt_choice("请选择 cls 推理输入方式", [("image", "image 单张图片"), ("image_dir", "image_dir 文件夹批量推理")])
        args = argparse.Namespace(
            tool_task="cls",
            tool_action="infer",
            experiment_dir=experiment_dir,
            checkpoint=None,
            image=None,
            image_dir=None,
            output_dir=Path(prompt_text("输出目录", str(experiment_dir / "infer")) or str(experiment_dir / "infer")),
            threshold=prompt_float("分类阈值 threshold", rt.DEFAULT_CLS_THRESHOLD),
            topk=prompt_int("topk", 1),
            device=rt.DEFAULT_DEVICE,
        )
        if mode == "image":
            args.image = prompt_required_path("图片路径")
        else:
            args.image_dir = prompt_required_path("图片目录")
        return confirm_args("cls/infer", args)

    if task == "seg" and action == "eda":
        data_path = prompt_dataset_yaml("seg", _default_train_data_yaml("seg", rt.SEG_TRAIN_TYPE))
        output_dir_raw = prompt_text("输出目录 --output-dir（留空自动生成）", None)
        min_class_images = prompt_int("推荐删除阈值 --min-class-images（全局图片数低于此值的类推荐删除）", 10)
        threshold_percentile = prompt_float("压缩阈值百分位 --threshold-percentile", 0.9)
        args = argparse.Namespace(
            tool_task="seg",
            tool_action="eda",
            seg_type="auto",
            data=data_path,
            output_dir=Path(output_dir_raw).expanduser() if output_dir_raw else None,
            overwrite=prompt_yes_no("输出目录非空时是否允许覆盖 --overwrite", False),
            min_class_images=min_class_images,
            threshold_percentile=threshold_percentile,
        )
        return confirm_args("seg/eda", args)

    if task == "seg" and action == "curate":
        data_path = prompt_dataset_yaml("semantic", _default_train_data_yaml("seg", "semantic"))
        eda_dir_raw = prompt_text("EDA 输出目录 --eda-dir（留空自动查找最近的 EDA）", None)
        args = argparse.Namespace(
            tool_task="seg",
            tool_action="curate",
            data=data_path,
            eda_dir=Path(eda_dir_raw).expanduser() if eda_dir_raw else None,
            drop_classes=None,
            image_threshold=None,
            export_suffix="__curated",
        )
        return confirm_args("seg/curate", args)

    if task == "seg" and action in {"infer", "eval"}:
        seg_train_type = _prompt_seg_train_type()
        experiment_dir = prompt_experiment_dir("seg", rt.SEG_DEFAULT_EXPERIMENT_DIR)
        if action == "infer":
            mode = prompt_choice(
                "请选择 seg 推理输入方式",
                [
                    ("dataset", "dataset 数据集模式（自动发现 data.yaml 并选择）"),
                    ("image", "image 单张图片"),
                    ("image_dir", "image_dir 文件夹批量推理"),
                ],
            )
            args = argparse.Namespace(
                tool_task="seg",
                tool_action="infer",
                seg_train_type=seg_train_type,
                experiment_dir=experiment_dir,
                checkpoint=None,
                image=None,
                image_dir=None,
                data=None,
                split="test",
                output_dir=None,
                threshold=prompt_float("分割阈值 threshold", rt.DEFAULT_SEG_THRESHOLD),
                overwrite=prompt_yes_no("输出目录非空时是否允许覆盖", False),
                device=rt.DEFAULT_DEVICE,
            )
            if mode == "dataset":
                args.data = prompt_dataset_yaml(
                    seg_train_type,
                    _default_train_data_yaml("seg", seg_train_type),
                    preferred_experiment_dir=experiment_dir,
                )
                args.split = prompt_choice(
                    "请选择数据集划分 --split",
                    [("train", "train"), ("test", "test"), ("val", "val")],
                )
            elif mode == "image":
                args.image = prompt_required_path("图片路径")
            else:
                args.image_dir = prompt_required_path("图片目录")
            input_path = args.data or args.image_dir or args.image
            default_output = rt.build_action_output_dir(
                experiment_dir,
                "infer",
                input_path=input_path,
                split=args.split if args.data is not None else None,
            )
            output_raw = prompt_text("输出目录", str(default_output))
            args.output_dir = Path(output_raw or str(default_output)).expanduser()
            return confirm_args("seg/infer", args)
        data_path = prompt_dataset_yaml(
            seg_train_type,
            _default_train_data_yaml("seg", seg_train_type),
            preferred_experiment_dir=experiment_dir,
        )
        split_value = prompt_choice(
            "请选择数据集划分 --split",
            [("test", "test"), ("val", "val"), ("val test", "val + test（两个都评估）")],
        ).split()
        config_mode = prompt_choice(
            "请选择 eval 配置方式",
            [
                ("default", "default 默认配置"),
                ("custom", "custom 自定义配置"),
            ],
        )
        use_custom = config_mode == "custom"
        if not use_custom:
            print_default_seg_eval_summary(
                seg_train_type=seg_train_type, data_path=data_path, splits=split_value
            )
        default_eval_output = rt.build_action_output_dir(
            experiment_dir, "eval", input_path=data_path, split=split_value,
        )
        output_dir_raw = (
            prompt_text("输出目录 --output-dir", str(default_eval_output))
            if use_custom
            else None
        )
        args = argparse.Namespace(
            tool_task="seg",
            tool_action="eval",
            command="eval",
            seg_train_type=seg_train_type,
            experiment_dir=experiment_dir,
            checkpoint=None,
            data=data_path,
            split=split_value,
            output_dir=Path(output_dir_raw).expanduser() if output_dir_raw else default_eval_output,
            threshold=0.0
            if seg_train_type == "semantic"  # 语义分割逐像素 argmax，不过滤
            else (
                prompt_float("分割阈值 --threshold", rt.DEFAULT_SEG_THRESHOLD)
                if use_custom
                else rt.DEFAULT_SEG_THRESHOLD
            ),
            overwrite=prompt_yes_no("输出目录非空时是否允许覆盖 --overwrite", False)
            if use_custom
            else False,
            classwise=prompt_yes_no("是否输出按类指标 --classwise", False)
            if use_custom
            else False,
            save_visualization=prompt_yes_no("是否保存 GT/预测对比图 --save-visualization", True)
            if use_custom
            else True,
            vis_max_images=prompt_int(
                "对比图最多出多少张（好/差各半，0=不限制）--vis-max-images",
                rt.SEG_EVAL_VIS_MAX_IMAGES,
            )
            if use_custom
            else rt.SEG_EVAL_VIS_MAX_IMAGES,
            device=(prompt_text("推理设备 --device", rt.DEFAULT_DEVICE) or rt.DEFAULT_DEVICE)
            if use_custom
            else rt.DEFAULT_DEVICE,
            infer_config_mode=config_mode,
            shard_index=None,
            num_shards=1,
            dry_run=False,
            skip_important_artifacts=False,
            selected_splits=None,
            multi_output_root=None,
        )
        print(f"\n等价命令预览:\n  {build_seg_eval_cli_preview(args)}")
        return confirm_args("seg/eval", args)

    if task == "det" and action == "eda":
        output_dir_raw = prompt_text("输出目录 --output-dir，直接回车写入 out/EDA 自动目录", None)
        args = argparse.Namespace(
            tool_task="det",
            tool_action="eda",
            command="eda",
            data=prompt_dataset_yaml("det", Path(rt.INFER_DEFAULT_DATA)),
            output_dir=Path(output_dir_raw).expanduser() if output_dir_raw else None,
            overwrite=prompt_yes_no("输出目录非空时是否允许覆盖 --overwrite", False) if output_dir_raw else False,
        )
        print("\nEDA 内容: split 对照、类别分布、不平衡分析、目标尺寸、框密度、分辨率和逐图清单。")
        print(f"\n等价命令预览:\n  {build_det_cli_preview(args)}")
        return confirm_args("det/eda", args)

    if task == "seg" and action == "export":
        seg_train_type = _prompt_seg_train_type()
        if seg_train_type == "semantic":
            print("semantic segmentation export is not supported yet; current seg/export only supports YOLO polygon instance segmentation datasets.")
            return None
        export_source_data = prompt_dataset_yaml("instance", Path(rt.SEG_EXPORT_DEFAULT_SOURCE_DATA))
        target_total_images = prompt_int(
            "导出总图数 --target-total-images，0 表示按数据自动决定",
            rt.SEG_EXPORT_DEFAULT_TARGET_TOTAL_IMAGES,
        )
        args = argparse.Namespace(
            tool_task="seg",
            tool_action="export",
            seg_train_type=seg_train_type,
            command="seg-export",
            report_json=None,
            export_source_data=export_source_data,
            target_total_images=target_total_images,
            split_ratio=rt.SEG_EXPORT_DEFAULT_SPLIT_RATIO,
            good_class_threshold=rt.SEG_EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD,
            auto_balance=True,
            auto_relax_class_threshold=True,
            balance_ratio=rt.SEG_EXPORT_DEFAULT_BALANCE_RATIO,
            min_class_images=rt.SEG_EXPORT_DEFAULT_MIN_CLASS_IMAGES,
            min_class_instances=rt.SEG_EXPORT_DEFAULT_MIN_CLASS_INSTANCES,
            target_images_per_class=rt.SEG_EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS,
            target_instances_per_class=rt.SEG_EXPORT_DEFAULT_TARGET_INSTANCES_PER_CLASS,
            max_instances_per_image=rt.SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_IMAGE,
            max_instances_per_class_per_image=rt.SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_CLASS_PER_IMAGE,
            instance_density_penalty=rt.SEG_EXPORT_DEFAULT_INSTANCE_DENSITY_PENALTY,
            size_ratio=rt.SEG_EXPORT_DEFAULT_SIZE_RATIO,
            size_balance_weight=rt.SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT,
            avg_instances_per_image_min=rt.SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN,
            avg_instances_per_image_max=rt.SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX,
            export_suffix=rt.SEG_EXPORT_DEFAULT_EXPORT_SUFFIX,
        )
        print("\n导出策略: 只询问总图数，其余阈值基于 EDA 和目标图数自动联合推导。")
        print(f"重划分比例固定为 train:val:test = {rt.SEG_EXPORT_DEFAULT_SPLIT_RATIO}")
        print("report_json 将自动优先匹配当前数据集最近的 seg_eval_summary.json；找不到时按纯数据分布导出。")
        print(f"\n等价命令预览:\n  {build_seg_export_cli_preview(args)}")
        return confirm_args("seg/export", args)

    if task == "det" and action == "export":
        export_source_data = prompt_dataset_yaml("det", Path(rt.EXPORT_DEFAULT_SOURCE_DATA))
        target_total_images = prompt_int(
            "导出总图数 --target-total-images，0 表示按数据自动决定",
            rt.EXPORT_DEFAULT_TARGET_TOTAL_IMAGES,
        )
        args = argparse.Namespace(
            tool_task="det",
            tool_action="export",
            command="export",
            report_json=None,
            export_source_data=export_source_data,
            target_total_images=target_total_images,
            split_ratio=rt.EXPORT_DEFAULT_SPLIT_RATIO,
            good_class_threshold=rt.EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD,
            auto_balance=True,
            auto_relax_class_threshold=True,
            balance_ratio=rt.EXPORT_DEFAULT_BALANCE_RATIO,
            min_class_images=rt.EXPORT_DEFAULT_MIN_CLASS_IMAGES,
            min_class_boxes=rt.EXPORT_DEFAULT_MIN_CLASS_BOXES,
            target_images_per_class=rt.EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS,
            target_boxes_per_class=rt.EXPORT_DEFAULT_TARGET_BOXES_PER_CLASS,
            max_boxes_per_image=rt.EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE,
            max_boxes_per_class_per_image=rt.EXPORT_DEFAULT_MAX_BOXES_PER_CLASS_PER_IMAGE,
            box_density_penalty=rt.EXPORT_DEFAULT_BOX_DENSITY_PENALTY,
            size_ratio=rt.EXPORT_DEFAULT_SIZE_RATIO,
            size_balance_weight=rt.EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT,
            avg_boxes_per_image_min=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN,
            avg_boxes_per_image_max=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX,
            trim_boxes=rt.EXPORT_DEFAULT_TRIM_BOXES,
            export_suffix=rt.EXPORT_DEFAULT_EXPORT_SUFFIX,
        )
        print("\n导出策略: 只询问总图数，其余阈值基于 EDA 和目标图数自动联合推导。")
        print(f"重划分比例固定为 train:val:test = {rt.EXPORT_DEFAULT_SPLIT_RATIO}")
        print("report_json 将自动优先匹配当前数据集最近的 test_report.json；找不到时按纯数据分布导出。")
        print(f"\n等价命令预览:\n  {build_det_cli_preview(args)}")
        return confirm_args("det/export", args)

    if task == "det" and action == "eval":
        experiment_dir = prompt_experiment_dir("det", rt.INFER_DEFAULT_EXPERIMENT_DIR)
        data_path = prompt_dataset_yaml(
            "det",
            Path(rt.EVAL_DEFAULT_DATA),
            preferred_experiment_dir=experiment_dir,
        )
        split = prompt_choice(
            "请选择数据集划分 --split",
            [
                ("test", "test"),
                ("val", "val"),
                ("test+val", "test + val"),
                ("all", "all (train + test + val)"),
                ("train", "train"),
            ],
        )
        config_mode = prompt_choice(
            "请选择 eval 配置方式",
            [("default", "default 默认配置"), ("custom", "custom 自定义配置")],
        )
        use_custom = config_mode == "custom"
        if not use_custom:
            print_default_det_eval_summary(data_path=data_path, split=split)
        output_dir_raw = prompt_text("输出目录 --output-dir，直接回车自动生成", None) if use_custom else None
        args = argparse.Namespace(
            tool_task="det",
            tool_action="eval",
            command="eval",
            experiment_dir=experiment_dir,
            checkpoint=None,
            image=None,
            image_dir=None,
            data=data_path,
            split=split,
            output_dir=(
                Path(output_dir_raw).expanduser()
                if output_dir_raw
                else (Path(rt.EVAL_DEFAULT_OUTPUT_DIR) if rt.EVAL_OUTPUT_DIR_CONFIGURED else None)
            ),
            score_threshold=(
                prompt_float("置信度阈值 --score-threshold", rt.INFER_DEFAULT_SCORE_THRESHOLD)
                if use_custom else rt.INFER_DEFAULT_SCORE_THRESHOLD
            ),
            report_iou_threshold=(
                prompt_float("报告 IoU 阈值 --report-iou-threshold", rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD)
                if use_custom else rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD
            ),
            bad_class_map50_threshold=rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD,
            metric_classwise=(
                prompt_yes_no("是否输出按类指标 --classwise", False) if use_custom else False
            ),
            classwise=False,
            save_visualization=(
                prompt_yes_no("是否保存抽样 GT/预测对比图", True) if use_custom else True
            ),
            vis_max_images=(
                prompt_int("每个 split 最多输出多少张对比图 --vis-max-images", rt.DET_EVAL_VIS_MAX_IMAGES)
                if use_custom else rt.DET_EVAL_VIS_MAX_IMAGES
            ),
            save_json=(
                prompt_yes_no("是否保存每图预测 JSON --save-json", rt.DET_EVAL_SAVE_JSON)
                if use_custom else rt.DET_EVAL_SAVE_JSON
            ),
            save_txt=False,
            compute_metrics=True,
            save_test_report=True,
            report_path=(
                Path(rt.EVAL_DEFAULT_REPORT_PATH)
                if rt.EVAL_REPORT_PATH_CONFIGURED
                else None
            ),
            overwrite=(
                prompt_yes_no("输出目录非空时是否允许覆盖 --overwrite", False)
                if use_custom else False
            ),
            device=(prompt_text("推理设备 --device", rt.INFER_DEFAULT_DEVICE) or rt.INFER_DEFAULT_DEVICE)
            if use_custom else rt.INFER_DEFAULT_DEVICE,
            infer_config_mode=config_mode,
            sahi=rt.INFER_DEFAULT_SAHI,
            sahi_overlap=rt.INFER_DEFAULT_SAHI_OVERLAP,
            sahi_nms_iou=rt.INFER_DEFAULT_SAHI_NMS_IOU,
            sahi_global_local_iou=rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU,
            sahi_skip_small=rt.INFER_DEFAULT_SAHI_SKIP_SMALL,
            shard_index=None,
            num_shards=1,
            dry_run=False,
            skip_important_artifacts=False,
            selected_splits=None,
            multi_output_root=None,
        )
        print(f"\n等价命令预览:\n  {build_det_cli_preview(args)}")
        return confirm_args("det/eval", args)

    if task == "det" and action == "infer":
        experiment_dir = prompt_experiment_dir("det", rt.INFER_DEFAULT_EXPERIMENT_DIR)
        mode = prompt_choice("请选择输入方式", [("dataset", "dataset 数据集批量推理"), ("image", "image 单张图片"), ("image_dir", "image_dir 文件夹批量推理")])
        is_dataset_mode = mode == "dataset"
        data_path = (
            prompt_dataset_yaml(
                "det",
                Path(rt.INFER_DEFAULT_DATA),
                preferred_experiment_dir=experiment_dir,
            )
            if is_dataset_mode
            else None
        )
        config_mode = prompt_choice(
            "请选择 infer 配置方式",
            [
                ("default", "default 默认配置"),
                ("custom", "custom 自定义配置"),
            ],
        )
        use_custom = config_mode == "custom"
        if not use_custom:
            print_default_det_infer_summary(mode=mode, data_path=data_path)
        output_dir_raw = prompt_text("输出目录 --output-dir，直接回车自动生成", None) if use_custom else None
        args = argparse.Namespace(
            tool_task="det",
            tool_action="infer",
            command="infer",
            experiment_dir=experiment_dir,
            checkpoint=None,
            image=None,
            image_dir=None,
            data=data_path,
            split=rt.INFER_DEFAULT_SPLIT,
            output_dir=Path(output_dir_raw).expanduser() if output_dir_raw else None,
            score_threshold=prompt_float("置信度阈值 --score-threshold", rt.INFER_DEFAULT_SCORE_THRESHOLD)
            if use_custom
            else rt.INFER_DEFAULT_SCORE_THRESHOLD,
            device=(prompt_text("推理设备 --device", rt.INFER_DEFAULT_DEVICE) or rt.INFER_DEFAULT_DEVICE)
            if use_custom
            else rt.INFER_DEFAULT_DEVICE,
            save_visualization=prompt_yes_no("是否保存可视化结果 --save-visualization", True)
            if use_custom
            else True,
            save_json=prompt_yes_no("是否保存 JSON 预测结果 --save-json", rt.INFER_DEFAULT_SAVE_JSON)
            if use_custom
            else rt.INFER_DEFAULT_SAVE_JSON,
            save_txt=prompt_yes_no("是否保存 TXT 预测结果 --save-txt", False)
            if use_custom
            else False,
            compute_metrics=False,
            metric_classwise=False,
            report_iou_threshold=rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD,
            bad_class_map50_threshold=rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD,
            save_test_report=False,
            report_path=None,
            overwrite=prompt_yes_no("输出目录非空时是否允许覆盖 --overwrite", False)
            if use_custom
            else False,
            infer_config_mode=config_mode,
            # SAHI 开关：由 launcher.py 的 det_sahi_enabled 决定，不在菜单里提问。
            sahi=rt.INFER_DEFAULT_SAHI,
            sahi_overlap=rt.INFER_DEFAULT_SAHI_OVERLAP,
            sahi_nms_iou=rt.INFER_DEFAULT_SAHI_NMS_IOU,
            sahi_global_local_iou=rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU,
            sahi_skip_small=rt.INFER_DEFAULT_SAHI_SKIP_SMALL,
        )
        if mode == "dataset":
            args.split = prompt_det_infer_split(
                experiment_dir=experiment_dir,
                data_path=args.data,
            )
        elif mode == "image":
            args.image = prompt_required_path("图片路径 --image")
        else:
            args.image_dir = prompt_required_path("图片目录 --image-dir", str(rt.INFER_DEFAULT_IMAGE_DIR))
        print(f"\n等价命令预览:\n  {build_det_cli_preview(args)}")
        return confirm_args("det/infer", args)

    if task == "det" and action == "optimize":
        print("\n训练后数据集优化：分析混淆矩阵和劣质类别，生成新数据集目录（原数据集不变）。")
        source_data_yaml = prompt_dataset_yaml("det", Path(rt.EXPORT_DEFAULT_SOURCE_DATA))
        optimize_candidates = _collect_optimize_analysis_candidates(source_data_yaml)
        selected_candidate = _prompt_optimize_report_json(source_data_yaml, optimize_candidates)
        report_json: Path | None = None
        report_jsons: list[Path] | None = None
        infer_output_dir: Path | None = None

        optimize_experiment_dir: Path | None = None
        auto_infer_temp_dir: Path | None = None

        if selected_candidate is None:
            report_json, infer_output_dir, optimize_experiment_dir, auto_infer_temp_dir = (
                _run_auto_infer_for_optimize(source_data_yaml)
            )
        else:
            selected_report_json = selected_candidate.get("report_json")
            if isinstance(selected_report_json, Path):
                report_json = selected_report_json
            report_jsons = selected_candidate.get("report_jsons")
            selected_infer_output_dir = selected_candidate.get("infer_output_dir")
            if isinstance(selected_infer_output_dir, Path):
                infer_output_dir = selected_infer_output_dir
            selected_experiment_dir = selected_candidate.get("experiment_dir")
            if isinstance(selected_experiment_dir, Path):
                optimize_experiment_dir = selected_experiment_dir
            if infer_output_dir is None and report_json is not None:
                infer_output_dir = _infer_output_dir_from_report(report_json)
            if infer_output_dir is not None:
                print("\n[det/optimize] 已从 test_report 对应的推理输出中找到预测 JSON。")
                print(f"  infer_output_dir : {compact_display_path(infer_output_dir)}")
            else:
                print("\n[det/optimize] 已选择 test_report；当前缺少可用于混淆矩阵的预测 JSON。")
                choice = prompt_choice(
                    "请选择处理方式",
                    [
                        ("auto_infer", "自动推理一次，生成混淆矩阵所需 JSON"),
                        ("cancel", "取消本次 optimize"),
                    ],
                )
                if choice == "cancel":
                    print("已取消本次执行。")
                    return None
                report_json, infer_output_dir, optimize_experiment_dir, auto_infer_temp_dir = (
                    _run_auto_infer_for_optimize(
                        source_data_yaml,
                        report_json_for_quality=report_json,
                        experiment_dir=optimize_experiment_dir or _experiment_dir_from_report(report_json),  # type: ignore[arg-type]
                    )
                )
                if infer_output_dir is None:
                    _cleanup_optimize_temp_dir(auto_infer_temp_dir)
                    print("自动推理完成后未找到预测 JSON，已取消本次执行。")
                    return None

        args = argparse.Namespace(
            tool_task="det",
            tool_action="optimize",
            source_data_yaml=source_data_yaml,
            report_json=report_json,
            report_jsons=report_jsons,
            infer_output_dir=infer_output_dir,
            confusion_threshold=0.15,
            optimize_experiment_dir=optimize_experiment_dir,
            auto_infer_temp_dir=auto_infer_temp_dir,
        )
        print("\ndet/optimize 配置确认")
        print(f"  source_data_yaml : {compact_display_path(source_data_yaml)}")
        if report_jsons:
            print(f"  report_json      : {len(report_jsons)} 份报告合并（{', '.join(r.stem for r in report_jsons)}）")
        else:
            print(f"  report_json      : {compact_display_path(report_json) if report_json else '(无)'}")
        print(f"  infer_output_dir : {compact_display_path(infer_output_dir) if infer_output_dir else '(无)'}")
        print(f"  confusion_threshold: 0.15 (强单向/双向混淆 >= 15% 建议合并)")
        if prompt_yes_no("确认执行以上配置吗", True):
            return args
        _cleanup_optimize_temp_dir(auto_infer_temp_dir)
        print("已取消本次执行。")
        return None

    if task == "det" and action == "report":
        args = argparse.Namespace(
            tool_task="det",
            tool_action="report",
            command="report",
            experiment_dir=prompt_experiment_dir("det", rt.INFER_DEFAULT_EXPERIMENT_DIR),
            search=None,
            output_dir=None,
            dry_run=False,
        )
        return confirm_args("det/report", args)

    return None


def _int_or_auto(value: str) -> int | str:
    if value.lower() == "auto":
        return "auto"
    return int(value)


def _devices_arg(value: str) -> int | str | list[int]:
    raw = value.strip().lower()
    if raw == "auto":
        return "auto"
    if "," in raw:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return int(raw)


def _validate_ratio_text(value: object, *, field: str, allow_empty: bool = False) -> None:
    text = str(value or "").strip()
    if not text and allow_empty:
        return
    parts = [part for part in re.split(r"[\s/:,]+", text) if part]
    if len(parts) != 3:
        raise ValueError(f"{field} 必须包含三个比例值，示例 8:1:1。")
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"{field} 必须包含三个数字。") from exc
    if any(not math.isfinite(item) or item < 0 for item in values) or sum(values) <= 0:
        raise ValueError(f"{field} 必须由非负有限数字组成，且总和大于 0。")


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate normalized CLI and interactive arguments with one shared contract."""
    for field in ("output_dir", "out_dir"):
        value = getattr(args, field, None)
        if value is not None and Path(value).expanduser().is_symlink():
            raise ValueError(f"输出路径不能是符号链接: {value}")
    unit_interval_fields = {
        "threshold",
        "score_threshold",
        "report_iou_threshold",
        "bad_class_map50_threshold",
        "sahi_nms_iou",
        "sahi_global_local_iou",
        "good_class_threshold",
        "threshold_percentile",
        "iou_dup",
        "iou_conflict",
        "match_iou_threshold",
        "low_conf_threshold",
        "outlier_class_ap",
        "confusion_threshold",
    }
    for field in unit_interval_fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"--{field.replace('_', '-')} 必须位于 [0, 1]。")

    overlap = getattr(args, "sahi_overlap", None)
    if overlap is not None:
        number = float(overlap)
        if not math.isfinite(number) or not 0.0 <= number < 1.0:
            raise ValueError("--sahi-overlap 必须位于 [0, 1)。")

    balance_ratio = getattr(args, "balance_ratio", None)
    if balance_ratio is not None:
        number = float(balance_ratio)
        if not math.isfinite(number) or (number != 0.0 and number < 1.0):
            raise ValueError("--balance-ratio 必须为 0（自动）或大于等于 1。")

    positive_fields = {"topk", "k", "cap", "num_shards"}
    nonnegative_fields = {
        "vis_max_images",
        "target_total_images",
        "min_class_images",
        "min_class_boxes",
        "min_class_instances",
        "target_images_per_class",
        "target_boxes_per_class",
        "target_instances_per_class",
        "max_boxes_per_image",
        "max_boxes_per_class_per_image",
        "max_instances_per_image",
        "max_instances_per_class_per_image",
        "max_images",
        "dense_top_n",
        "outlier_class_gt",
        "image_threshold",
    }
    for field in positive_fields | nonnegative_fields:
        value = getattr(args, field, None)
        if value is None:
            continue
        minimum = 1 if field in positive_fields else 0
        if int(value) < minimum:
            relation = "大于 0" if minimum else "大于或等于 0"
            raise ValueError(f"--{field.replace('_', '-')} 必须{relation}。")

    for field in (
        "box_density_penalty",
        "instance_density_penalty",
        "size_balance_weight",
        "min_box_px",
        "alpha",
    ):
        value = getattr(args, field, None)
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"--{field.replace('_', '-')} 必须是非负有限数字。")

    for field in ("steps", "batch_size"):
        value = getattr(args, field, None)
        if isinstance(value, int) and value <= 0:
            raise ValueError(f"--{field.replace('_', '-')} 必须大于 0 或使用 auto。")
    workers = getattr(args, "num_workers", None)
    if isinstance(workers, int) and workers < 0:
        raise ValueError("--num-workers 必须大于或等于 0，或使用 auto。")
    devices = getattr(args, "devices", None)
    device_ids = devices if isinstance(devices, list) else [devices]
    if any(isinstance(device, int) and device < 0 for device in device_ids):
        raise ValueError("--devices 中的 GPU 编号必须大于或等于 0。")

    shard_index = getattr(args, "shard_index", None)
    num_shards = int(getattr(args, "num_shards", 1) or 1)
    if shard_index is not None and not 0 <= int(shard_index) < num_shards:
        raise ValueError("--shard-index 必须位于 [0, num-shards)。")

    for field in ("split_ratio",):
        if hasattr(args, field):
            _validate_ratio_text(getattr(args, field), field=f"--{field.replace('_', '-')}")
    if hasattr(args, "size_ratio"):
        _validate_ratio_text(
            getattr(args, "size_ratio"), field="--size-ratio", allow_empty=True
        )

    for minimum_field, maximum_field in (
        ("avg_boxes_per_image_min", "avg_boxes_per_image_max"),
        ("avg_instances_per_image_min", "avg_instances_per_image_max"),
    ):
        minimum = getattr(args, minimum_field, None)
        maximum = getattr(args, maximum_field, None)
        if minimum is None or maximum is None:
            continue
        minimum_value = float(minimum)
        maximum_value = float(maximum)
        if any(not math.isfinite(item) or item < 0 for item in (minimum_value, maximum_value)):
            raise ValueError("平均每图数量范围必须由非负有限数字组成。")
        if maximum_value > 0 and minimum_value > maximum_value:
            raise ValueError("平均每图数量最小值不能大于最大值。")
    return args


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LightlyTrain 多任务训练、推理、评估与数据治理入口。"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="发生错误时显示完整 traceback；可放在命令前，launcher 也兼容放在末尾。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = _add_command_parser(subparsers, "train")
    train_parser.add_argument("--task", choices=("cls", "det", "seg"), default="det")
    train_parser.add_argument(
        "--seg-train-type",
        choices=("instance", "semantic"),
        default=rt.SEG_TRAIN_TYPE,
        help="Only used when --task seg.",
    )
    train_parser.add_argument("--data", "--data-yaml", dest="data_yaml", type=Path, default=None)
    train_parser.add_argument("--model", type=str, default=None)
    train_parser.add_argument("--backbone-weights", type=Path, default=None)
    train_parser.add_argument("--checkpoint", type=Path, default=None)
    train_parser.add_argument("--out-dir", type=Path, default=None)
    train_parser.add_argument("--steps", type=_int_or_auto, default="auto")
    train_parser.add_argument("--batch-size", type=_int_or_auto, default="auto")
    train_parser.add_argument("--num-workers", type=_int_or_auto, default="auto")
    train_parser.add_argument("--devices", type=_devices_arg, default="auto")
    train_parser.add_argument("--overwrite", action="store_true", default=False)
    train_parser.add_argument("--resume-interrupted", action="store_true", default=False)

    infer_parser = _add_command_parser(subparsers, "infer")
    infer_parser.add_argument("--task", choices=("det", "cls", "seg"), default="det")
    infer_parser.add_argument(
        "--seg-train-type",
        choices=("instance", "semantic"),
        default=None,
        help="Only used when --task seg; inferred from the experiment when omitted.",
    )
    infer_parser.add_argument("--experiment-dir", type=Path, default=None)
    infer_parser.add_argument("--checkpoint", type=Path, default=rt.INFER_DEFAULT_CHECKPOINT)
    infer_input = infer_parser.add_mutually_exclusive_group(required=False)
    infer_input.add_argument("--image", type=Path, default=rt.INFER_DEFAULT_IMAGE)
    infer_input.add_argument("--image-dir", type=Path, default=None)
    infer_input.add_argument("--data", type=Path, default=None)
    infer_parser.add_argument(
        "--split",
        choices=("train", "val", "test", "test+val", "all"),
        default=rt.INFER_DEFAULT_SPLIT,
    )
    infer_parser.add_argument("--output-dir", type=Path, default=None)
    infer_parser.add_argument("--score-threshold", type=float, default=rt.INFER_DEFAULT_SCORE_THRESHOLD)
    infer_parser.add_argument("--threshold", type=float, default=None)
    infer_parser.add_argument("--topk", type=int, default=1)
    infer_parser.add_argument("--device", type=str, default=rt.INFER_DEFAULT_DEVICE)
    infer_parser.add_argument("--save-visualization", dest="save_visualization", action="store_true")
    infer_parser.add_argument("--skip-visualization", dest="save_visualization", action="store_false")
    infer_parser.set_defaults(save_visualization=rt.INFER_DEFAULT_SAVE_VISUALIZATION)
    infer_parser.add_argument("--save-json", dest="save_json", action="store_true")
    infer_parser.add_argument("--skip-json", dest="save_json", action="store_false")
    infer_parser.set_defaults(save_json=rt.INFER_DEFAULT_SAVE_JSON)
    infer_parser.add_argument("--save-txt", action="store_true", default=rt.INFER_DEFAULT_SAVE_TXT)
    infer_parser.add_argument("--report-iou-threshold", type=float, default=rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD)
    infer_parser.add_argument(
        "--bad-class-map50-threshold",
        type=float,
        default=rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD,
    )
    infer_parser.add_argument("--compute-metrics", action="store_true", default=rt.INFER_DEFAULT_COMPUTE_METRICS)
    infer_parser.add_argument("--metric-classwise", action="store_true", default=rt.INFER_DEFAULT_METRIC_CLASSWISE)
    infer_parser.add_argument("--save-test-report", action="store_true", default=rt.INFER_DEFAULT_SAVE_TEST_REPORT)
    infer_parser.add_argument("--report-path", type=Path, default=None)
    # SAHI 切片推理：仅 --task det 时生效；不加 --sahi 时流程与旧版完全一致。
    infer_parser.add_argument("--sahi", action="store_true", default=rt.INFER_DEFAULT_SAHI)
    infer_parser.add_argument("--sahi-overlap", dest="sahi_overlap", type=float, default=rt.INFER_DEFAULT_SAHI_OVERLAP)
    infer_parser.add_argument("--sahi-nms-iou", dest="sahi_nms_iou", type=float, default=rt.INFER_DEFAULT_SAHI_NMS_IOU)
    infer_parser.add_argument(
        "--sahi-global-local-iou",
        dest="sahi_global_local_iou",
        type=float,
        default=rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU,
    )
    # 短边 < tile 的小图是否跳过 SAHI、回退普通 predict（小图上 SAHI 会更差）。
    infer_parser.add_argument("--sahi-skip-small", dest="sahi_skip_small", action="store_true")
    infer_parser.add_argument("--no-sahi-skip-small", dest="sahi_skip_small", action="store_false")
    infer_parser.set_defaults(sahi_skip_small=rt.INFER_DEFAULT_SAHI_SKIP_SMALL)
    infer_parser.add_argument("--overwrite", action="store_true", default=rt.INFER_DEFAULT_OVERWRITE)
    infer_parser.add_argument("--dry-run", action="store_true", default=False)
    infer_parser.add_argument("--skip-important-artifacts", action="store_true", default=False, help=argparse.SUPPRESS)
    infer_parser.add_argument("--selected-splits", type=str, default=None, help=argparse.SUPPRESS)
    infer_parser.add_argument("--multi-output-root", type=Path, default=None, help=argparse.SUPPRESS)
    infer_parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    infer_parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)

    eval_parser = _add_command_parser(subparsers, "eval")
    eval_parser.add_argument("--task", choices=("cls", "det", "seg"), default="seg")
    eval_parser.add_argument(
        "--seg-train-type",
        choices=("instance", "semantic"),
        default=None,
        help="Only used when --task seg; inferred from the experiment when omitted.",
    )
    eval_parser.add_argument("--experiment-dir", type=Path, default=None)
    eval_parser.add_argument("--checkpoint", type=Path, default=rt.INFER_DEFAULT_CHECKPOINT)
    eval_parser.add_argument("--data", type=Path, default=None)
    eval_parser.add_argument("--test-dir", type=Path, default=None)
    eval_parser.add_argument(
        "--split",
        nargs="+",
        choices=("train", "val", "test", "test+val", "all"),
        default=["test"],
        help="可指定多个 split，如 --split val test。",
    )
    eval_parser.add_argument("--output-dir", type=Path, default=None)
    eval_parser.add_argument("--threshold", type=float, default=None)
    eval_parser.add_argument("--score-threshold", type=float, default=rt.INFER_DEFAULT_SCORE_THRESHOLD)
    eval_parser.add_argument("--report-iou-threshold", type=float, default=rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD)
    eval_parser.add_argument(
        "--bad-class-map50-threshold",
        type=float,
        default=rt.INFER_DEFAULT_BAD_CLASS_MAP50_THRESHOLD,
    )
    eval_parser.add_argument("--report-path", type=Path, default=None)
    eval_parser.add_argument("--topk", type=int, default=1)
    eval_parser.add_argument("--classwise", action="store_true", default=False)
    eval_parser.add_argument("--device", type=str, default=rt.INFER_DEFAULT_DEVICE)
    eval_parser.add_argument("--overwrite", action="store_true", default=rt.INFER_DEFAULT_OVERWRITE)
    eval_parser.add_argument("--save-visualization", dest="save_visualization", action="store_true")
    eval_parser.add_argument(
        "--no-save-visualization", "--skip-visualization",
        dest="save_visualization", action="store_false",
    )
    eval_parser.set_defaults(save_visualization=True)
    eval_parser.add_argument(
        "--vis-max-images",
        dest="vis_max_images",
        type=int,
        default=None,
        help="评估对比图每个 split 最多出多少张；0=不限制。",
    )
    eval_parser.add_argument("--save-json", dest="save_json", action="store_true")
    eval_parser.add_argument("--skip-json", dest="save_json", action="store_false")
    eval_parser.set_defaults(save_json=rt.DET_EVAL_SAVE_JSON)
    eval_parser.add_argument("--sahi", action="store_true", default=rt.INFER_DEFAULT_SAHI)
    eval_parser.add_argument("--sahi-overlap", dest="sahi_overlap", type=float, default=rt.INFER_DEFAULT_SAHI_OVERLAP)
    eval_parser.add_argument("--sahi-nms-iou", dest="sahi_nms_iou", type=float, default=rt.INFER_DEFAULT_SAHI_NMS_IOU)
    eval_parser.add_argument(
        "--sahi-global-local-iou",
        dest="sahi_global_local_iou",
        type=float,
        default=rt.INFER_DEFAULT_SAHI_GLOBAL_LOCAL_IOU,
    )
    eval_parser.add_argument("--sahi-skip-small", dest="sahi_skip_small", action="store_true")
    eval_parser.add_argument("--no-sahi-skip-small", dest="sahi_skip_small", action="store_false")
    eval_parser.set_defaults(sahi_skip_small=rt.INFER_DEFAULT_SAHI_SKIP_SMALL)
    eval_parser.add_argument("--dry-run", action="store_true", default=False)
    eval_parser.add_argument("--skip-important-artifacts", action="store_true", default=False, help=argparse.SUPPRESS)
    eval_parser.add_argument("--selected-splits", type=str, default=None, help=argparse.SUPPRESS)
    eval_parser.add_argument("--multi-output-root", type=Path, default=None, help=argparse.SUPPRESS)
    eval_parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    eval_parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)

    export_parser = _add_command_parser(subparsers, "export")
    export_parser.add_argument("--report-json", type=Path, default=None)
    export_parser.add_argument("--export-source-data", type=Path, default=rt.EXPORT_DEFAULT_SOURCE_DATA)
    export_parser.add_argument("--good-class-threshold", type=float, default=rt.EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD)
    export_parser.add_argument("--auto-balance", dest="auto_balance", action="store_true")
    export_parser.add_argument("--no-auto-balance", dest="auto_balance", action="store_false")
    export_parser.set_defaults(auto_balance=rt.EXPORT_DEFAULT_AUTO_BALANCE)
    export_parser.add_argument(
        "--auto-relax-class-threshold",
        dest="auto_relax_class_threshold",
        action="store_true",
    )
    export_parser.add_argument(
        "--strict-class-threshold",
        dest="auto_relax_class_threshold",
        action="store_false",
    )
    export_parser.set_defaults(
        auto_relax_class_threshold=rt.EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD
    )
    export_parser.add_argument("--balance-ratio", type=float, default=rt.EXPORT_DEFAULT_BALANCE_RATIO)
    export_parser.add_argument("--min-class-images", type=int, default=rt.EXPORT_DEFAULT_MIN_CLASS_IMAGES)
    export_parser.add_argument("--min-class-boxes", type=int, default=rt.EXPORT_DEFAULT_MIN_CLASS_BOXES)
    export_parser.add_argument("--target-images-per-class", type=int, default=rt.EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS)
    export_parser.add_argument("--target-total-images", type=int, default=rt.EXPORT_DEFAULT_TARGET_TOTAL_IMAGES)
    export_parser.add_argument("--split-ratio", type=str, default=rt.EXPORT_DEFAULT_SPLIT_RATIO)
    export_parser.add_argument("--target-boxes-per-class", type=int, default=rt.EXPORT_DEFAULT_TARGET_BOXES_PER_CLASS)
    export_parser.add_argument("--max-boxes-per-image", type=int, default=rt.EXPORT_DEFAULT_MAX_BOXES_PER_IMAGE)
    export_parser.add_argument(
        "--max-boxes-per-class-per-image",
        type=int,
        default=rt.EXPORT_DEFAULT_MAX_BOXES_PER_CLASS_PER_IMAGE,
    )
    export_parser.add_argument("--box-density-penalty", type=float, default=rt.EXPORT_DEFAULT_BOX_DENSITY_PENALTY)
    export_parser.add_argument(
        "--trim-boxes",
        dest="trim_boxes",
        action="store_true",
        default=rt.EXPORT_DEFAULT_TRIM_BOXES,
        help="选图后把超出均衡窗口的类的多余框从标签里删掉(会产生漏标，默认关闭)。",
    )
    export_parser.add_argument("--no-trim-boxes", dest="trim_boxes", action="store_false")
    export_parser.add_argument("--size-ratio", type=str, default=rt.EXPORT_DEFAULT_SIZE_RATIO)
    export_parser.add_argument("--size-balance-weight", type=float, default=rt.EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT)
    export_parser.add_argument("--avg-boxes-per-image-min", type=float, default=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MIN)
    export_parser.add_argument("--avg-boxes-per-image-max", type=float, default=rt.EXPORT_DEFAULT_AVG_BOXES_PER_IMAGE_MAX)
    export_parser.add_argument("--export-suffix", type=str, default=rt.EXPORT_DEFAULT_EXPORT_SUFFIX)

    seg_export_parser = _add_command_parser(subparsers, "seg-export")
    seg_export_parser.add_argument(
        "--seg-train-type",
        choices=("instance", "semantic"),
        default="instance",
    )
    seg_export_parser.add_argument("--report-json", type=Path, default=None)
    seg_export_parser.add_argument("--export-source-data", type=Path, default=rt.SEG_EXPORT_DEFAULT_SOURCE_DATA)
    seg_export_parser.add_argument(
        "--good-class-threshold",
        type=float,
        default=rt.SEG_EXPORT_DEFAULT_GOOD_CLASS_THRESHOLD,
    )
    seg_export_parser.add_argument("--auto-balance", dest="auto_balance", action="store_true")
    seg_export_parser.add_argument("--no-auto-balance", dest="auto_balance", action="store_false")
    seg_export_parser.set_defaults(auto_balance=rt.SEG_EXPORT_DEFAULT_AUTO_BALANCE)
    seg_export_parser.add_argument(
        "--auto-relax-class-threshold",
        dest="auto_relax_class_threshold",
        action="store_true",
    )
    seg_export_parser.add_argument(
        "--strict-class-threshold",
        dest="auto_relax_class_threshold",
        action="store_false",
    )
    seg_export_parser.set_defaults(
        auto_relax_class_threshold=rt.SEG_EXPORT_DEFAULT_AUTO_RELAX_CLASS_THRESHOLD
    )
    seg_export_parser.add_argument(
        "--balance-ratio", type=float, default=rt.SEG_EXPORT_DEFAULT_BALANCE_RATIO
    )
    seg_export_parser.add_argument(
        "--min-class-images", type=int, default=rt.SEG_EXPORT_DEFAULT_MIN_CLASS_IMAGES
    )
    seg_export_parser.add_argument(
        "--min-class-instances", type=int, default=rt.SEG_EXPORT_DEFAULT_MIN_CLASS_INSTANCES
    )
    seg_export_parser.add_argument(
        "--target-images-per-class",
        type=int,
        default=rt.SEG_EXPORT_DEFAULT_TARGET_IMAGES_PER_CLASS,
    )
    seg_export_parser.add_argument(
        "--target-total-images",
        type=int,
        default=rt.SEG_EXPORT_DEFAULT_TARGET_TOTAL_IMAGES,
    )
    seg_export_parser.add_argument(
        "--split-ratio", type=str, default=rt.SEG_EXPORT_DEFAULT_SPLIT_RATIO
    )
    seg_export_parser.add_argument(
        "--target-instances-per-class",
        type=int,
        default=rt.SEG_EXPORT_DEFAULT_TARGET_INSTANCES_PER_CLASS,
    )
    seg_export_parser.add_argument(
        "--max-instances-per-image",
        type=int,
        default=rt.SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_IMAGE,
    )
    seg_export_parser.add_argument(
        "--max-instances-per-class-per-image",
        type=int,
        default=rt.SEG_EXPORT_DEFAULT_MAX_INSTANCES_PER_CLASS_PER_IMAGE,
    )
    seg_export_parser.add_argument(
        "--instance-density-penalty",
        type=float,
        default=rt.SEG_EXPORT_DEFAULT_INSTANCE_DENSITY_PENALTY,
    )
    seg_export_parser.add_argument("--size-ratio", type=str, default=rt.SEG_EXPORT_DEFAULT_SIZE_RATIO)
    seg_export_parser.add_argument("--size-balance-weight", type=float, default=rt.SEG_EXPORT_DEFAULT_SIZE_BALANCE_WEIGHT)
    seg_export_parser.add_argument("--avg-instances-per-image-min", type=float, default=rt.SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MIN)
    seg_export_parser.add_argument("--avg-instances-per-image-max", type=float, default=rt.SEG_EXPORT_DEFAULT_AVG_INSTANCES_PER_IMAGE_MAX)
    seg_export_parser.add_argument(
        "--export-suffix", type=str, default=rt.SEG_EXPORT_DEFAULT_EXPORT_SUFFIX
    )

    # seg-eda: 自动识别语义分割 / 实例分割 EDA
    seg_eda_parser = _add_command_parser(subparsers, "seg-eda")
    seg_eda_parser.add_argument("--data", type=Path, default=None)
    seg_eda_parser.add_argument(
        "--seg-type", choices=("auto", "instance", "semantic"), default="auto"
    )
    seg_eda_parser.add_argument("--output-dir", type=Path, default=None)
    seg_eda_parser.add_argument("--overwrite", action="store_true", default=False)
    seg_eda_parser.add_argument("--min-class-images", type=int, default=10)
    seg_eda_parser.add_argument("--threshold-percentile", type=float, default=0.9)

    # seg-curate: 语义分割交互式类别整理
    seg_curate_parser = _add_command_parser(subparsers, "seg-curate")
    seg_curate_parser.add_argument("--data", type=Path, default=rt.SEMANTIC_SEG_DEFAULT_DATA)
    seg_curate_parser.add_argument("--eda-dir", type=Path, default=None)
    seg_curate_parser.add_argument("--drop-classes", type=str, default=None, help="逗号分隔的类别 ID")
    seg_curate_parser.add_argument("--image-threshold", type=int, default=None, help="train 每类最多保留图片数，0=不压缩")
    seg_curate_parser.add_argument("--export-suffix", type=str, default="__curated")
    seg_curate_parser.add_argument("--contiguous-ids", action="store_true", default=False, help="将保留类 ID 重映射为 0..K-1 连续编号（改写 mask 像素值）")

    eda_parser = _add_command_parser(subparsers, "eda")
    eda_parser.add_argument("--data", type=Path, default=None)
    eda_parser.add_argument("--output-dir", type=Path, default=None)
    eda_parser.add_argument("--overwrite", action="store_true", default=False)

    # review-sample 子命令
    review_sample_parser = _add_command_parser(subparsers, "review-sample")
    review_sample_parser.add_argument("--data", type=Path, default=None)
    review_sample_parser.add_argument("--experiment-dir", type=Path, default=None)
    review_sample_parser.add_argument("--infer-output-dir", type=Path, default=None)
    review_sample_parser.add_argument("--report-path", type=Path, default=None)
    review_sample_parser.add_argument("--out-dir", type=Path, default=None)
    # 功能开关
    review_sample_parser.add_argument("--enable-geometry", dest="enable_geometry", action="store_true", default=True)
    review_sample_parser.add_argument("--disable-geometry", dest="enable_geometry", action="store_false")
    review_sample_parser.add_argument("--enable-model-analysis", dest="enable_model_analysis", action="store_true", default=True)
    review_sample_parser.add_argument("--disable-model-analysis", dest="enable_model_analysis", action="store_false")
    review_sample_parser.add_argument("--enable-outlier-class", dest="enable_outlier_class", action="store_true", default=True)
    review_sample_parser.add_argument("--disable-outlier-class", dest="enable_outlier_class", action="store_false")
    review_sample_parser.add_argument("--enable-visualization", dest="enable_visualization", action="store_true", default=False)
    review_sample_parser.add_argument("--disable-visualization", dest="enable_visualization", action="store_false")
    # 选图参数
    review_sample_parser.add_argument("--k", type=int, default=3)
    review_sample_parser.add_argument("--alpha", type=float, default=3.0)
    review_sample_parser.add_argument("--cap", type=int, default=10)
    review_sample_parser.add_argument("--max-images", type=int, default=500)
    review_sample_parser.add_argument("--min-only", action="store_true", default=False)
    review_sample_parser.add_argument("--problems-only", action="store_true", default=False)
    # 几何检测参数
    review_sample_parser.add_argument("--iou-dup", type=float, default=0.9)
    review_sample_parser.add_argument("--iou-conflict", type=float, default=0.5)
    review_sample_parser.add_argument("--min-box-px", type=float, default=4)
    review_sample_parser.add_argument("--dense-top-n", type=int, default=None)
    # 模型分析参数
    review_sample_parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    review_sample_parser.add_argument("--low-conf-threshold", type=float, default=0.3)
    review_sample_parser.add_argument("--outlier-class-ap", type=float, default=0.1)
    review_sample_parser.add_argument("--outlier-class-gt", type=int, default=5)
    # 其他
    review_sample_parser.add_argument("--groups-path", type=Path, default=None)
    review_sample_parser.add_argument("--groups-section", type=str, default="coco80")
    review_sample_parser.add_argument("--cache-path", type=Path, default=None)
    review_sample_parser.add_argument("--seed", type=int, default=42)
    review_sample_parser.add_argument("--embed-images", action="store_true", default=True)
    review_sample_parser.add_argument("--no-embed-images", dest="embed_images", action="store_false")
    review_sample_parser.add_argument("--interactive", action="store_true", default=False)

    report_parser = _add_command_parser(subparsers, "report")
    report_parser.add_argument("--experiment-dir", type=Path, default=None)
    report_parser.add_argument("--search", type=str, default=None)
    report_parser.add_argument("--output-dir", type=Path, default=None)
    report_parser.add_argument("--dry-run", action="store_true", default=False)

    clean_parser = _add_command_parser(subparsers, "clean")
    clean_mode = clean_parser.add_mutually_exclusive_group()
    clean_mode.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="扫描并预览全部可清理内容（默认）。",
    )
    clean_mode.add_argument(
        "--execute", dest="dry_run", action="store_false",
        help="执行清理；同时需要 --yes。",
    )
    clean_parser.set_defaults(dry_run=True)
    clean_parser.add_argument("--yes", action="store_true", help="确认执行全部计划项。")

    optimize_parser = _add_command_parser(subparsers, "optimize")
    optimize_parser.add_argument("--source-data", dest="source_data_yaml", type=Path, default=None)
    optimize_parser.add_argument("--report-json", type=Path, action="append", default=None)
    optimize_parser.add_argument("--infer-output-dir", type=Path, default=None)
    optimize_parser.add_argument("--experiment-dir", dest="optimize_experiment_dir", type=Path, default=None)
    optimize_parser.add_argument("--confusion-threshold", type=float, default=0.15)

    _complete_parser_help(subparsers)
    args = parser.parse_args(argv)
    if args.command == "train":
        from . import train_tools

        args.tool_task = args.task
        args.tool_action = "train"
        if args.data_yaml is None:
            args.data_yaml = _default_train_data_yaml(args.task, args.seg_train_type)
        if args.model is None:
            args.model = _default_train_model(args.task, args.seg_train_type)
        if args.resume_interrupted and args.out_dir is None:
            raise ValueError(
                "--resume-interrupted 需要 --out-dir 指向已有实验目录"
                "（续跑会读取该目录里的 checkpoint）；自动生成的新目录无法续跑。"
            )
        if args.out_dir is None:
            args.out_dir = train_tools.build_default_out_dir(args.data_yaml, args.model)
        return args
    if args.command == "clean":
        args.tool_task = "clean"
        args.tool_action = "clean"
        args.analyses = None
        if not args.dry_run and not args.yes:
            raise ValueError("clean --execute 需要同时提供 --yes。")
        return args
    if args.command == "optimize":
        args.tool_task = "det"
        args.tool_action = "optimize"
        if args.source_data_yaml is None:
            args.source_data_yaml = Path(rt.EXPORT_DEFAULT_SOURCE_DATA)
        reports = list(args.report_json or [])
        args.report_json = reports[0] if reports else None
        args.report_jsons = reports if len(reports) > 1 else None
        args.auto_infer_temp_dir = None
        return args
    if args.command == "seg-export":
        args.tool_task = "seg"
        args.tool_action = "export"
        return args
    if args.command == "seg-eda":
        args.tool_task = "seg"
        args.tool_action = "eda"
        if args.data is None:
            args.data = _default_train_data_yaml("seg", rt.SEG_TRAIN_TYPE)
        return args
    if args.command == "seg-curate":
        args.tool_task = "seg"
        args.tool_action = "curate"
        if args.data is None:
            args.data = rt.SEMANTIC_SEG_DEFAULT_DATA
        return args
    if args.command == "infer":
        args.tool_task = args.task
        args.tool_action = "infer"
        if args.task == "seg":
            if args.experiment_dir is None:
                args.experiment_dir = rt.auto_resolve_experiment_dir(
                    "seg", rt.SEG_DEFAULT_EXPERIMENT_DIR
                )
            if args.seg_train_type is None:
                seg_experiment = (
                    rt.experiment_dir_from_checkpoint_path(args.checkpoint)
                    if args.checkpoint is not None
                    else Path(args.experiment_dir)
                )
                args.seg_train_type = rt.experiment_seg_type(seg_experiment) or rt.SEG_TRAIN_TYPE
            if args.image is None and args.image_dir is None and args.data is None:
                args.data = _default_train_data_yaml(
                    "seg",
                    args.seg_train_type,
                    preferred_experiment_dir=Path(args.experiment_dir),
                )
            args.threshold = args.threshold if args.threshold is not None else rt.DEFAULT_SEG_THRESHOLD
            if args.output_dir is None:
                input_path = args.data or args.image_dir or args.image
                args.output_dir = rt.build_action_output_dir(
                    Path(args.experiment_dir),
                    "infer",
                    input_path=input_path,
                    split=args.split if args.data is not None else None,
                )
        elif args.task == "cls":
            if args.experiment_dir is None:
                args.experiment_dir = rt.auto_resolve_experiment_dir(
                    "cls", rt.EXPERIMENT_ROOT_DIR / "my_experiment_cls"
                )
            args.threshold = args.threshold if args.threshold is not None else rt.DEFAULT_CLS_THRESHOLD
            if args.output_dir is None:
                args.output_dir = Path(args.experiment_dir) / "infer"
            if args.image is None and args.image_dir is None:
                raise ValueError("cls infer requires --image or --image-dir.")
        else:
            if args.experiment_dir is None:
                args.experiment_dir = rt.auto_resolve_experiment_dir(
                    "det", rt.INFER_DEFAULT_EXPERIMENT_DIR
                )
            if args.image is None and args.image_dir is None and args.data is None:
                args.data = _default_train_data_yaml(
                    "det", preferred_experiment_dir=Path(args.experiment_dir)
                )
            if args.output_dir is None and rt.INFER_OUTPUT_DIR_CONFIGURED:
                args.output_dir = Path(rt.INFER_DEFAULT_OUTPUT_DIR)
        return args
    if args.command == "eval":
        args.tool_task = args.task
        args.tool_action = "eval"
        if args.task == "seg":
            if any(split in {"test+val", "all"} for split in args.split):
                raise ValueError("seg eval 的 split 使用 train、val、test，可一次传入多个。")
            if args.experiment_dir is None:
                args.experiment_dir = rt.auto_resolve_experiment_dir(
                    "seg", rt.SEG_DEFAULT_EXPERIMENT_DIR
                )
            if args.seg_train_type is None:
                seg_experiment = (
                    rt.experiment_dir_from_checkpoint_path(args.checkpoint)
                    if args.checkpoint is not None
                    else Path(args.experiment_dir)
                )
                args.seg_train_type = rt.experiment_seg_type(seg_experiment) or rt.SEG_TRAIN_TYPE
            if args.data is None:
                args.data = _default_train_data_yaml(
                    "seg",
                    args.seg_train_type,
                    preferred_experiment_dir=Path(args.experiment_dir),
                )
            args.threshold = args.threshold if args.threshold is not None else rt.DEFAULT_SEG_THRESHOLD
            args.vis_max_images = (
                rt.SEG_EVAL_VIS_MAX_IMAGES
                if args.vis_max_images is None
                else args.vis_max_images
            )
            if args.output_dir is None:
                args.output_dir = rt.build_action_output_dir(
                    Path(args.experiment_dir),
                    "eval",
                    input_path=args.data,
                    split=args.split,
                )
        elif args.task == "det":
            if args.experiment_dir is None:
                args.experiment_dir = rt.auto_resolve_experiment_dir(
                    "det", rt.INFER_DEFAULT_EXPERIMENT_DIR
                )
            if args.data is None:
                args.data = _default_train_data_yaml(
                    "det", preferred_experiment_dir=Path(args.experiment_dir)
                )
            requested = list(args.split)
            if len(requested) == 1:
                args.split = requested[0]
            elif set(requested) == {"test", "val"} and len(requested) == 2:
                args.split = "test+val"
            elif set(requested) == {"train", "test", "val"} and len(requested) == 3:
                args.split = "all"
            else:
                raise ValueError(
                    "det eval 的多 split 组合支持 val test 或 train val test。"
                )
            args.image = None
            args.image_dir = None
            args.metric_classwise = bool(args.classwise)
            args.compute_metrics = True
            args.save_test_report = True
            # eval 默认只出指标；需要预测明细做后处理时由 --save-json 打开。
            args.save_json = bool(getattr(args, "save_json", rt.DET_EVAL_SAVE_JSON))
            args.save_txt = False
            args.vis_max_images = (
                rt.DET_EVAL_VIS_MAX_IMAGES
                if args.vis_max_images is None
                else args.vis_max_images
            )
            if args.vis_max_images < 0:
                raise ValueError("det eval --vis-max-images 必须大于或等于 0。")
            if args.output_dir is None and rt.EVAL_OUTPUT_DIR_CONFIGURED:
                args.output_dir = Path(rt.EVAL_DEFAULT_OUTPUT_DIR)
            if args.report_path is None and rt.EVAL_REPORT_PATH_CONFIGURED:
                args.report_path = Path(rt.EVAL_DEFAULT_REPORT_PATH)
        else:
            if args.experiment_dir is None:
                args.experiment_dir = rt.auto_resolve_experiment_dir(
                    "cls", rt.EXPERIMENT_ROOT_DIR / "my_experiment_cls"
                )
            if args.test_dir is None:
                raise ValueError("cls eval requires --test-dir.")
            args.threshold = args.threshold if args.threshold is not None else rt.DEFAULT_CLS_THRESHOLD
            if args.output_dir is None:
                args.output_dir = Path(args.experiment_dir) / "eval"
        return args

    if args.command == "review-sample":
        args.tool_task = "det"
        args.tool_action = "review-sample"
        if args.data is None:
            args.data = Path(rt.EXPORT_DEFAULT_SOURCE_DATA)
        return args

    if args.command == "report" and args.experiment_dir is None:
        candidates = list_experiment_dirs(task="det")
        if args.search:
            candidates = filter_dirs_by_keyword(candidates, args.search)
        if not candidates:
            detail = f"（关键字: {args.search}）" if args.search else ""
            raise ValueError(f"未找到可生成报告的 det 实验目录{detail}。")
        args.experiment_dir = candidates[0]

    if args.command == "eda" and args.data is None:
        args.data = _default_train_data_yaml("det")

    args.tool_task = "det"
    args.tool_action = args.command
    return args


def build_review_sample_cli_preview(args: argparse.Namespace) -> str:
    """构建 review-sample 命令预览"""
    parts = ["python launcher.py review-sample"]
    if args.data:
        parts.append(f"--data {compact_display_path(args.data)}")
    if args.experiment_dir:
        parts.append(f"--experiment-dir {compact_display_path(args.experiment_dir)}")
    if args.k:
        parts.append(f"--k {args.k}")
    if args.max_images:
        parts.append(f"--max-images {args.max_images}")
    return " ".join(parts)


def _prompt_review_sample_experiment_dir(data_path: Path) -> Path | None:
    """为质检抽样选择实验目录"""
    from .det_problem_export import discover_infer_runs

    # 扫描所有实验目录
    all_dirs = list_experiment_dirs(task="det")
    if not all_dirs:
        all_dirs = list_experiment_dirs()

    # 过滤有推理结果的实验
    experiments_with_infer = []
    for exp_dir in all_dirs:
        runs = discover_infer_runs(exp_dir)
        if runs:
            # 获取最新推理结果的摘要
            latest_run = runs[0]
            report_path = latest_run.get("report_path")
            mAP = None
            if report_path and report_path.exists():
                try:
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
                    summary = report_payload.get("summary", {})
                    mAP = summary.get("map") or summary.get("map_50")
                except Exception:
                    pass

            experiments_with_infer.append({
                "dir": exp_dir,
                "runs": runs,
                "latest_split": latest_run.get("split", "unknown"),
                "mAP": mAP,
            })

    if not experiments_with_infer:
        print("\n  未发现有推理结果的实验目录。")
        print("  提示：请先运行 det infer 生成推理结果。")
        if prompt_yes_no("是否跳过模型分析，仅使用几何检测", True):
            return None
        return None

    print("\n  发现以下实验目录（含推理结果）：")
    for idx, exp in enumerate(experiments_with_infer, 1):
        dir_name = compact_display_path(exp["dir"])
        split = exp["latest_split"]
        mAP_str = f"mAP@0.5: {exp['mAP']:.3f}" if exp["mAP"] else "无报告"
        n_runs = len(exp["runs"])
        print(f"    [{idx}] {dir_name} ({split} split, {mAP_str}, {n_runs}个推理结果)")

    print(f"    [0] 跳过模型分析（仅几何检测）")

    while True:
        raw = read_input(f"\n  请选择实验目录 [0-{len(experiments_with_infer)}, 回车默认 1]: ").strip()
        if not raw:
            return experiments_with_infer[0]["dir"]
        if raw == "0":
            return None
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(experiments_with_infer):
                return experiments_with_infer[idx]["dir"]
        except ValueError:
            pass
        print("  无效选择，请重新输入。")


def _prompt_review_sample_infer_run(experiment_dir: Path) -> tuple[Path | None, Path | None]:
    """选择推理结果"""
    from .det_problem_export import discover_infer_runs

    runs = discover_infer_runs(experiment_dir)
    if not runs:
        print(f"\n  实验 {compact_display_path(experiment_dir)} 下未发现推理结果。")
        return None, None

    if len(runs) == 1:
        run = runs[0]
        print(f"\n  使用推理结果: {run['split']} split")
        return run.get("output_dir"), run.get("report_path")

    print(f"\n  实验目录下发现以下推理结果：")
    for idx, run in enumerate(runs, 1):
        split = run.get("split", "unknown")
        report_path = run.get("report_path")
        mAP_str = ""
        if report_path and report_path.exists():
            try:
                report_payload = json.loads(report_path.read_text(encoding="utf-8"))
                summary = report_payload.get("summary", {})
                mAP = summary.get("map") or summary.get("map_50")
                if mAP:
                    mAP_str = f", mAP@0.5: {mAP:.3f}"
            except Exception:
                pass
        print(f"    [{idx}] {split} split{mAP_str}")

    while True:
        raw = read_input(f"\n  请选择推理结果 [1-{len(runs)}, 回车默认 1]: ").strip()
        if not raw:
            run = runs[0]
            return run.get("output_dir"), run.get("report_path")
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(runs):
                run = runs[idx]
                return run.get("output_dir"), run.get("report_path")
        except ValueError:
            pass
        print("  无效选择，请重新输入。")


def _auto_detect_datasets() -> list[Path]:
    """使用共享检测器列出目标检测数据集。"""
    return list_dataset_yaml_candidates(task="det")


def _auto_detect_experiment_with_infer() -> list[dict[str, Any]]:
    """自动检测有推理结果的实验目录"""
    from .det_problem_export import discover_infer_runs

    all_dirs = list_experiment_dirs(task="det")
    if not all_dirs:
        all_dirs = list_experiment_dirs()

    experiments_with_infer = []
    for exp_dir in all_dirs:
        runs = discover_infer_runs(exp_dir)
        if runs:
            latest_run = runs[0]
            report_path = latest_run.get("report_path")
            mAP = None
            num_images = 0
            if report_path and report_path.exists():
                try:
                    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
                    summary = report_payload.get("summary", {})
                    mAP = summary.get("map") or summary.get("map_50")
                    num_images = summary.get("num_images", 0)
                except Exception:
                    pass

            experiments_with_infer.append({
                "dir": exp_dir,
                "runs": runs,
                "latest_split": latest_run.get("split", "unknown"),
                "mAP": mAP,
                "num_images": num_images,
            })

    return experiments_with_infer


def build_interactive_review_sample_args() -> argparse.Namespace | None:
    """交互式质检抽样流程 - 自动检测，只问关键开关"""
    import time

    print("\n" + "=" * 63)
    print("  数据集质检抽样 v2.0")
    print("=" * 63)

    # ── Step 0: 自动检测数据集 ──
    print("\n[Step 0] 自动检测数据集 ...")
    data_yamls = _auto_detect_datasets()

    if not data_yamls:
        print("  错误: datasets/ 目录下未找到 data.yaml 文件")
        return None

    # 选择数据集
    if len(data_yamls) == 1:
        data_path = data_yamls[0]
        print(f"  数据集: {compact_display_path(data_path)}")
    else:
        print(f"\n  发现 {len(data_yamls)} 个数据集：")
        for idx, p in enumerate(data_yamls[:10], 1):
            # 尝试读取类别数
            try:
                cfg = rt.load_data_config(p)
                names = rt.normalize_names(cfg.get("names"))
                nc = len(names)
                print(f"    [{idx}] {compact_display_path(p)} (nc={nc})")
            except Exception:
                print(f"    [{idx}] {compact_display_path(p)}")
        if len(data_yamls) > 10:
            print(f"    ... 还有 {len(data_yamls) - 10} 个")

        while True:
            raw = read_input(f"\n  请选择数据集 [1-{min(len(data_yamls), 10)}, 回车默认 1]: ").strip()
            if not raw:
                data_path = data_yamls[0]
                break
            try:
                idx = int(raw) - 1
                if 0 <= idx < min(len(data_yamls), 10):
                    data_path = data_yamls[idx]
                    break
            except ValueError:
                pass
            print("  无效选择，请重新输入。")

    # 读取数据集信息
    try:
        cfg = rt.load_data_config(data_path)
        class_names = rt.normalize_names(cfg.get("names"))
        nc = len(class_names)

        # 统计图片数
        total_images = 0
        for split in ("train", "val", "test"):
            split_dir = cfg.get(split)
            if split_dir:
                img_dir = Path(cfg["_root_dir"]) / split_dir
                if img_dir.exists():
                    total_images += len(list(img_dir.rglob("*.jpg"))) + len(list(img_dir.rglob("*.png")))

        print(f"  类别数: {nc}")
        print(f"  图片数: {total_images:,}")
    except Exception as e:
        print(f"  警告: 无法读取数据集详细信息: {e}")
        class_names = {}
        nc = 0
        total_images = 0

    # ── Step 1: 自动检测实验目录 ──
    print("\n" + "=" * 63)
    print("  模型分析配置")
    print("=" * 63)

    print("\n  自动检测实验目录 ...")
    experiments = _auto_detect_experiment_with_infer()

    experiment_dir = None
    infer_output_dir = None
    report_path = None
    mAP = None

    if not experiments:
        print("  未发现有推理结果的实验目录。")
        print("  将仅使用几何检测模式。")
    else:
        print(f"\n  发现 {len(experiments)} 个有推理结果的实验：")
        for idx, exp in enumerate(experiments[:5], 1):
            dir_name = compact_display_path(exp["dir"])
            split = exp["latest_split"]
            mAP_str = f"mAP={exp['mAP']:.3f}" if exp["mAP"] else "无报告"
            n_imgs = f"{exp['num_images']}张" if exp["num_images"] else ""
            print(f"    [{idx}] {dir_name} ({split}, {mAP_str}, {n_imgs})")
        print(f"    [0] 跳过模型分析（仅几何检测）")

        while True:
            raw = read_input(f"\n  请选择 [0-{min(len(experiments), 5)}, 回车默认 1]: ").strip()
            if not raw:
                exp = experiments[0]
                experiment_dir = exp["dir"]
                mAP = exp["mAP"]
                break
            if raw == "0":
                break
            try:
                idx = int(raw) - 1
                if 0 <= idx < min(len(experiments), 5):
                    exp = experiments[idx]
                    experiment_dir = exp["dir"]
                    mAP = exp["mAP"]
                    break
            except ValueError:
                pass
            print("  无效选择，请重新输入。")

        # 自动获取推理结果
        if experiment_dir:
            from .det_problem_export import discover_infer_runs
            runs = discover_infer_runs(experiment_dir)
            if runs:
                infer_output_dir = runs[0].get("output_dir")
                report_path = runs[0].get("report_path")
                split = runs[0].get("split", "unknown")
                print(f"\n  使用推理结果: {split} split")
                if mAP:
                    print(f"  mAP@0.5: {mAP:.3f}")

    # ── Step 2: 功能开关 ──
    print("\n  ── 功能开关（回车使用默认值）──")
    enable_geometry = prompt_yes_no("  [1] 几何检测（重复框/冲突标签/异常框）", True)
    enable_model = prompt_yes_no("  [2] 模型分析（漏检/错标/误检）", experiment_dir is not None)
    enable_outlier = prompt_yes_no("  [3] 劣质类别重点抽样", True)

    # ── Step 3: 抽样数量 ──
    print("\n  ── 抽样数量 ──")
    k = prompt_int("  每类软下限 k", 3)
    max_images = prompt_int("  总图数上限（0=不限）", 500)

    # ── 自动生成输出目录 ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_name = data_path.parent.parent.name + "_" + data_path.parent.name
    out_dir = rt.EXPERIMENT_ROOT_DIR / f"review_{dataset_name}_{timestamp}"

    # ── 确认执行 ──
    print("\n" + "=" * 63)
    print("  确认执行")
    print("=" * 63)

    print("\n  执行计划：")
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │ 1. 扫描数据集 + 一致性校验                              │")
    if enable_geometry:
        print("  │ 2. 几何检测：dup / conflict / bad_box / dense           │")
    if enable_model and infer_output_dir:
        print("  │ 3. 模型分析：missing / swapped / false_pos / loc        │")
    if enable_outlier:
        print("  │ 4. 劣质类别识别（AP < 0.1 或 GT < 5）                   │")
    print("  │ 5. 贪心选图（问题优先 + 覆盖均衡）                      │")
    print("  │ 6. 导出 review_subset/                                  │")
    print("  └─────────────────────────────────────────────────────────┘")

    print(f"\n  数据源：")
    print(f"  - 数据集: {compact_display_path(data_path)} (nc={nc}, {total_images}张)")
    if experiment_dir:
        print(f"  - 实验目录: {compact_display_path(experiment_dir)}")
    if infer_output_dir:
        print(f"  - 推理结果: {compact_display_path(infer_output_dir)}")
    print(f"  - test_report: {'✓ 已加载' if report_path else '✗ 无'}")
    print(f"\n  输出目录: {compact_display_path(out_dir)}")

    if not prompt_yes_no("\n  开始执行？", True):
        print("  已取消。")
        return None

    # 构建参数
    args = argparse.Namespace(
        tool_task="det",
        tool_action="review-sample",
        command="review-sample",
        data=data_path,
        experiment_dir=experiment_dir,
        infer_output_dir=infer_output_dir,
        report_path=report_path,
        # 功能开关
        enable_geometry=enable_geometry,
        enable_model_analysis=enable_model,
        enable_outlier_class=enable_outlier,
        enable_visualization=False,
        # 选图参数
        k=k,
        alpha=3.0,
        cap=10,
        max_images=max_images if max_images > 0 else None,
        min_only=False,
        problems_only=False,
        # 几何检测参数
        iou_dup=0.9,
        iou_conflict=0.5,
        min_box_px=4,
        dense_top_n=None,
        # 模型分析参数
        match_iou_threshold=0.5,
        low_conf_threshold=0.3,
        outlier_class_ap=0.1,
        outlier_class_gt=5,
        # 其他
        groups_path=None,
        groups_section="coco80",
        cache_path=None,
        seed=42,
        embed_images=True,
        out_dir=out_dir,
    )

    return args
