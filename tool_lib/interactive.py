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
import re
from pathlib import Path

from . import common as rt
from . import convert_tools


def read_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit("\n输入结束，已取消执行。")


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
    if not rt.EXPERIMENT_ROOT_DIR.exists():
        return []
    candidates = [path for path in rt.EXPERIMENT_ROOT_DIR.rglob("*") if rt.is_experiment_dir(path)]
    if task is None:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates
    candidates.sort(
        key=lambda p: (
            p.stat().st_mtime,
            1 if rt.is_task_experiment_dir(p, task) else 0,
        ),
        reverse=True,
    )
    return candidates


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
    raw = keyword.strip().lower()
    if not raw:
        return dirs
    normalized = re.sub(r"[^0-9a-z]+", " ", raw)
    terms = [term for term in normalized.split() if term]
    if not terms:
        return dirs
    return [
        path
        for path in dirs
        if all(term in re.sub(r"[^0-9a-z]+", " ", str(path).lower()) for term in terms)
    ]


def list_dataset_yaml_candidates(task: str | None = None) -> list[Path]:
    dataset_roots = list(rt.ROOT_DIR.glob("datasets/**/data.yaml"))
    candidates: list[Path] = []
    for path in dataset_roots:
        parent_name = path.parent.name.lower()
        path_text = str(path).lower()
        if "datasets/convert_datasets/" in path_text.replace("\\", "/"):
            continue
        if task == "det" and "dataset_det" not in parent_name and "detect" not in path_text:
            continue
        if task == "seg" and "dataset_seg" not in parent_name and "seg" not in path_text:
            continue
        candidates.append(path.resolve())
    candidates.sort(key=lambda p: (p.stat().st_mtime, str(p).lower()), reverse=True)
    return candidates


def prompt_convert_source_dir() -> Path:
    candidates = convert_tools.list_convert_source_dirs()
    if not candidates:
        return prompt_required_path("待转换源目录")

    while True:
        print("\n自动发现的待转换数据集:")
        for idx, path in enumerate(candidates, start=1):
            print(f"  {idx}. {compact_display_path(path)}")
        print("  输入编号直接选择")
        print("  输入名称直接选择，例如: TLPD")
        print("  输入 custom 手动输入路径")

        raw = read_input("请选择数据集: ").strip()
        lowered = raw.lower()
        if lowered == "custom":
            return prompt_required_path("待转换源目录")
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(candidates):
                return candidates[index]
        for path in candidates:
            if lowered == path.name.lower():
                return path
        print("无效选择，请重新输入。")


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


def prompt_dataset_yaml(task: str, default: Path) -> Path:
    candidates = list_dataset_yaml_candidates(task=task)
    if not candidates:
        return Path(prompt_text("数据配置 --data", str(default)) or str(default))

    visible_limit = 20
    visible_candidates = candidates[:visible_limit]

    while True:
        print(f"\n自动发现的 {task} 数据集:")
        for idx, path in enumerate(visible_candidates, start=1):
            print(f"  {idx}. {compact_display_path(path)}")
        if len(candidates) > visible_limit and len(visible_candidates) < len(candidates):
            print(f"  ... 当前仅显示前 {len(visible_candidates)} 个，共 {len(candidates)} 个")
        print("  输入编号直接选择")
        print("  输入 all 查看全部")
        print("  输入 custom 手动输入 data.yaml")

        raw = read_input("请选择数据集: ").strip()
        lowered = raw.lower()
        if not raw:
            return default
        if lowered == "custom":
            return Path(prompt_text("数据配置 --data", str(default)) or str(default))
        if lowered == "all":
            visible_candidates = candidates
            continue
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(visible_candidates):
                return visible_candidates[index]
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
            print(f"  {idx}. {compact_display_path(path)}")
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
    print(f"  export_source_data: {compact_display_value(args.export_source_data)}")
    print(f"  report_json: {report_value}")
    print(f"  target_total_images: {compact_display_value(args.target_total_images)}")
    print(f"  split_ratio: {compact_display_value(args.split_ratio)}")
    print(f"  export_suffix: {compact_display_value(args.export_suffix)}")

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


def confirm_args(title: str, args: argparse.Namespace) -> argparse.Namespace | None:
    if title == "det/export":
        print_det_export_preview(args)
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
    print("  save_json: False")
    print("  save_txt: False")
    split_text = "交互选择 test / val / all" if mode == "dataset" else "(not used)"
    print(f"  split: {split_text}")
    selected_data = data_path if data_path is not None else rt.INFER_DEFAULT_DATA
    default_data = compact_display_path(selected_data) if mode == "dataset" else "(not used)"
    print(f"  data: {default_data}")
    print(f"  output_dir: (auto)")
    print(f"  report_path: {'(auto)' if mode == 'dataset' else '(not used)'}")
    print(f"  compute_metrics: {'True' if mode == 'dataset' else 'False'}")
    print(f"  save_test_report: {'True' if mode == 'dataset' else 'False'}")
    print("  metric_classwise: False")
    print("  overwrite: False")


def _guess_det_dataset_dir_from_data_yaml(data_path: Path) -> Path:
    return data_path.expanduser().resolve().parent


def _collect_existing_det_infer_outputs(
    *,
    experiment_dir: Path,
    data_path: Path,
    splits: tuple[str, ...] = ("test", "val"),
) -> dict[str, Path]:
    try:
        checkpoint_path = rt.resolve_checkpoint_path(None, experiment_dir)
        dataset_dir = _guess_det_dataset_dir_from_data_yaml(data_path)
        existing: dict[str, Path] = {}
        for split in splits:
            output_dir = rt.derive_det_run_dir(
                checkpoint_path=checkpoint_path,
                dataset_dir=dataset_dir,
                split=split,
            )
            if output_dir.exists() and any(output_dir.iterdir()):
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
            ("test", "test"),
            ("val", "val"),
            ("all", "all (test + val)"),
        ],
    )


def build_det_cli_preview(args: argparse.Namespace) -> str:
    parts = ["python", "launcher.py", args.command]
    if args.command == "infer":
        parts.extend(["--experiment-dir", str(args.experiment_dir)])
        if args.checkpoint is not None:
            parts.extend(["--checkpoint", str(args.checkpoint)])
        if args.image is not None:
            parts.extend(["--image", str(args.image)])
        elif args.image_dir is not None:
            parts.extend(["--image-dir", str(args.image_dir)])
        elif args.data is not None:
            parts.extend(["--data", str(args.data), "--split", str(args.split)])
        if args.output_dir is not None:
            parts.extend(["--output-dir", str(args.output_dir)])
        parts.extend(["--score-threshold", str(args.score_threshold), "--device", str(args.device)])
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
        parts.extend(["--export-suffix", str(args.export_suffix)])
    elif args.command == "eda":
        parts.extend(["--data", str(args.data)])
        if args.output_dir is not None:
            parts.extend(["--output-dir", str(args.output_dir)])
        if getattr(args, "overwrite", False):
            parts.append("--overwrite")
    return " ".join(parts)


def build_convert_cli_preview(args: argparse.Namespace) -> str:
    source_dir = Path(args.source_dir)
    try:
        source_value = str(source_dir.relative_to(convert_tools.CONVERT_DATASETS_ROOT))
    except ValueError:
        source_value = str(source_dir)

    parts = ["python", "launcher.py", "convert", source_value]
    if getattr(args, "output_name", None):
        parts.extend(["--output-name", str(args.output_name)])
    if getattr(args, "task", "all") != "all":
        parts.extend(["--task", str(args.task)])
    if args.label_format != "auto":
        parts.extend(["--label-format", str(args.label_format)])
    if args.seed is not None:
        parts.extend(["--seed", str(args.seed)])
    if args.dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def build_interactive_args() -> argparse.Namespace | None:
    task = prompt_choice(
        "请选择任务类型",
        [("cls", "cls 分类"), ("det", "det 检测"), ("seg", "seg 分割"), ("data", "data 数据集转换")],
    )
    if task == "data":
        source_dir = prompt_convert_source_dir()
        output_name = prompt_directory_name("输出数据集目录名", source_dir.name)
        args = argparse.Namespace(
            tool_task="data",
            tool_action="convert",
            source_dir=source_dir,
            output_name=output_name,
            output_root=(rt.ROOT_DIR / "datasets" / output_name).resolve(),
            task=prompt_choice(
                "请选择输出任务",
                [
                    ("det", "det 检测"),
                    ("cls", "cls 分类"),
                    ("seg", "seg 分割"),
                    ("all", "all 全部"),
                ],
            ),
            label_format=prompt_choice(
                "请选择标注格式 --label-format",
                [("auto", "auto 自动判断"), ("labelme", "labelme JSON"), ("yolo", "yolo TXT")],
            ),
            seed=None,
            dry_run=False,
        )
        print(f"\n等价命令预览:\n  {build_convert_cli_preview(args)}")
        return confirm_args("data/convert", args)

    action_options = [("train", "train 训练"), ("infer", "infer 推理")]
    if task in {"cls", "seg"}:
        action_options.append(("eval", "eval 评估"))
    if task == "det":
        action_options.append(("eda", "EDA 数据集分析"))
        action_options.append(("export", "export 数据集筛选"))
        action_options.append(("report", "report 生成实验报告"))
    action = prompt_choice(f"请选择 {task} 功能", action_options)

    if action == "train":
        return confirm_args(
            f"{task}/train",
            argparse.Namespace(
                tool_task=task,
                tool_action=action,
                script_path={
                    "cls": rt.TRAIN_CLS_SCRIPT,
                    "det": rt.TRAIN_DET_SCRIPT,
                    "seg": rt.TRAIN_SEG_SCRIPT,
                }[task],
            ),
        )

    if task == "cls" and action == "eval":
        return confirm_args(
            "cls/eval",
            argparse.Namespace(tool_task="cls", tool_action="eval", script_path=rt.TEST_CLS_SCRIPT),
        )
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

    if task == "seg" and action in {"infer", "eval"}:
        experiment_dir = prompt_experiment_dir("seg", rt.EXPERIMENT_ROOT_DIR / "my_experiment_seg")
        if action == "infer":
            mode = prompt_choice("请选择 seg 推理输入方式", [("image", "image 单张图片"), ("image_dir", "image_dir 文件夹批量推理")])
            args = argparse.Namespace(
                tool_task="seg",
                tool_action="infer",
                experiment_dir=experiment_dir,
                checkpoint=None,
                image=None,
                image_dir=None,
                output_dir=Path(prompt_text("输出目录", str(experiment_dir / "infer")) or str(experiment_dir / "infer")),
                threshold=prompt_float("分割阈值 threshold", rt.DEFAULT_SEG_THRESHOLD),
                overwrite=prompt_yes_no("输出目录非空时是否允许覆盖", False),
                device=rt.DEFAULT_DEVICE,
            )
            if mode == "image":
                args.image = prompt_required_path("图片路径")
            else:
                args.image_dir = prompt_required_path("图片目录")
            return confirm_args("seg/infer", args)
        args = argparse.Namespace(
            tool_task="seg",
            tool_action="eval",
            experiment_dir=experiment_dir,
            checkpoint=None,
            data=prompt_dataset_yaml("seg", Path("datasets/dataset_seg/data.yaml")),
            split=prompt_choice("请选择数据集划分 --split", [("test", "test"), ("val", "val")]),
            output_dir=Path(prompt_text("输出目录", str(experiment_dir / "eval")) or str(experiment_dir / "eval")),
            threshold=prompt_float("分割阈值 threshold", rt.DEFAULT_SEG_THRESHOLD),
            overwrite=prompt_yes_no("输出目录非空时是否允许覆盖", False),
            classwise=prompt_yes_no("是否输出按类指标", False),
            device=rt.DEFAULT_DEVICE,
        )
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
            export_suffix=rt.EXPORT_DEFAULT_EXPORT_SUFFIX,
        )
        print("\n导出策略: 只询问总图数，其余阈值基于 EDA 和目标图数自动联合推导。")
        print(f"重划分比例固定为 train:val:test = {rt.EXPORT_DEFAULT_SPLIT_RATIO}")
        print("report_json 将自动优先匹配当前数据集最近的 test_report.json；找不到时按纯数据分布导出。")
        print(f"\n等价命令预览:\n  {build_det_cli_preview(args)}")
        return confirm_args("det/export", args)

    if task == "det" and action == "infer":
        experiment_dir = prompt_experiment_dir("det", rt.INFER_DEFAULT_EXPERIMENT_DIR)
        mode = prompt_choice("请选择输入方式", [("dataset", "dataset 数据集评测模式"), ("image", "image 单张图片"), ("image_dir", "image_dir 文件夹批量推理")])
        is_dataset_mode = mode == "dataset"
        data_path = prompt_dataset_yaml("det", Path(rt.INFER_DEFAULT_DATA)) if is_dataset_mode else None
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
            save_json=prompt_yes_no("是否保存 JSON 预测结果 --save-json", False)
            if use_custom
            else False,
            save_txt=prompt_yes_no("是否保存 TXT 预测结果 --save-txt", False)
            if use_custom
            else False,
            compute_metrics=is_dataset_mode,
            metric_classwise=is_dataset_mode and (
                prompt_yes_no("是否输出按类指标 --metric-classwise", False) if use_custom else False
            ),
            report_iou_threshold=rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD,
            save_test_report=is_dataset_mode,
            report_path=None,
            overwrite=prompt_yes_no("输出目录非空时是否允许覆盖 --overwrite", False)
            if use_custom
            else False,
            infer_config_mode=config_mode,
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


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Thin multi-task launcher. Detection CLI is still supported.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("source_dir", type=str, help="待转换数据集目录名或路径")
    convert_parser.add_argument("--output-name", type=str, default=None)
    convert_parser.add_argument("--output-root", type=Path, default=None)
    convert_parser.add_argument("--task", choices=("det", "cls", "seg", "all"), default="all")
    convert_parser.add_argument("--label-format", choices=("auto", "labelme", "yolo"), default="auto")
    convert_parser.add_argument("--seed", type=int, default=None)
    convert_parser.add_argument("--dry-run", action="store_true", default=False)

    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--experiment-dir", type=Path, default=rt.INFER_DEFAULT_EXPERIMENT_DIR)
    infer_parser.add_argument("--checkpoint", type=Path, default=rt.INFER_DEFAULT_CHECKPOINT)
    infer_input = infer_parser.add_mutually_exclusive_group(required=False)
    infer_input.add_argument("--image", type=Path, default=rt.INFER_DEFAULT_IMAGE)
    infer_input.add_argument("--image-dir", type=Path, default=None)
    infer_input.add_argument("--data", type=Path, default=None)
    infer_parser.add_argument("--split", choices=("val", "test", "all"), default=rt.INFER_DEFAULT_SPLIT)
    infer_parser.add_argument("--output-dir", type=Path, default=None)
    infer_parser.add_argument("--score-threshold", type=float, default=rt.INFER_DEFAULT_SCORE_THRESHOLD)
    infer_parser.add_argument("--device", type=str, default=rt.INFER_DEFAULT_DEVICE)
    infer_parser.add_argument("--save-visualization", dest="save_visualization", action="store_true")
    infer_parser.add_argument("--skip-visualization", dest="save_visualization", action="store_false")
    infer_parser.set_defaults(save_visualization=rt.INFER_DEFAULT_SAVE_VISUALIZATION)
    infer_parser.add_argument("--save-json", action="store_true", default=rt.INFER_DEFAULT_SAVE_JSON)
    infer_parser.add_argument("--save-txt", action="store_true", default=rt.INFER_DEFAULT_SAVE_TXT)
    infer_parser.add_argument("--report-iou-threshold", type=float, default=rt.INFER_DEFAULT_REPORT_IOU_THRESHOLD)
    infer_parser.add_argument("--compute-metrics", action="store_true", default=rt.INFER_DEFAULT_COMPUTE_METRICS)
    infer_parser.add_argument("--metric-classwise", action="store_true", default=rt.INFER_DEFAULT_METRIC_CLASSWISE)
    infer_parser.add_argument("--save-test-report", action="store_true", default=rt.INFER_DEFAULT_SAVE_TEST_REPORT)
    infer_parser.add_argument("--report-path", type=Path, default=None)
    infer_parser.add_argument("--overwrite", action="store_true", default=rt.INFER_DEFAULT_OVERWRITE)
    infer_parser.add_argument("--dry-run", action="store_true", default=False)
    infer_parser.add_argument("--shard-index", type=int, default=None, help=argparse.SUPPRESS)
    infer_parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)

    export_parser = subparsers.add_parser("export")
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
    export_parser.add_argument("--export-suffix", type=str, default=rt.EXPORT_DEFAULT_EXPORT_SUFFIX)

    eda_parser = subparsers.add_parser("eda")
    eda_parser.add_argument("--data", type=Path, default=rt.INFER_DEFAULT_DATA)
    eda_parser.add_argument("--output-dir", type=Path, default=None)
    eda_parser.add_argument("--overwrite", action="store_true", default=False)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--experiment-dir", type=Path, default=None)
    report_parser.add_argument("--search", type=str, default=None)
    report_parser.add_argument("--output-dir", type=Path, default=None)
    report_parser.add_argument("--dry-run", action="store_true", default=False)

    args = parser.parse_args(argv)
    if args.command == "convert":
        if args.output_root is None:
            args.output_name = args.output_name or Path(args.source_dir).name
            args.output_root = (rt.ROOT_DIR / "datasets" / args.output_name).resolve()
        else:
            args.output_root = args.output_root.expanduser().resolve()
            args.output_name = args.output_name or args.output_root.name
        args.tool_task = "data"
        args.tool_action = "convert"
        return args
    if args.command == "infer" and args.image is None and args.image_dir is None and args.data is None:
        args.data = rt.INFER_DEFAULT_DATA
    args.tool_task = "det"
    args.tool_action = args.command
    return args
