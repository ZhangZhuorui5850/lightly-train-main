#!/usr/bin/env python3
"""Generate an X-AnyLabeling mapping JSON for grayscale semantic masks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

try:
    from .dataset_discovery import format_modified_time
    from .dataset_detector import detect_datasets
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    from dataset_discovery import format_modified_time  # type: ignore[no-redef]
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}
SPLITS = ("train", "val", "test")
SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


@dataclass(frozen=True)
class ValidationStats:
    images: int
    masks: int
    checked_masks: int
    observed_values: tuple[int, ...]


@dataclass(frozen=True)
class SemanticDataset:
    root: Path
    config_path: Path
    splits: tuple[str, ...]
    class_count: int
    modified_time: float = 0.0


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(read_text_auto(path)) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层需要为字典: {path}")
    return value


def dataset_root_from_config(config_path: Path, config: dict[str, Any]) -> Path:
    raw_root = config.get("path")
    if raw_root is None:
        return config_path.parent.resolve()
    root = Path(str(raw_root)).expanduser()
    return root.resolve() if root.is_absolute() else (config_path.parent / root).resolve()


def class_names(config: dict[str, Any], root: Path) -> dict[int, str]:
    raw_names = config.get("names", config.get("classes"))
    if isinstance(raw_names, list):
        return {index: str(name) for index, name in enumerate(raw_names)}
    if isinstance(raw_names, dict):
        names: dict[int, str] = {}
        for raw_id, raw_name in raw_names.items():
            try:
                class_id = int(raw_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"类别 ID 需要为整数: {raw_id!r}") from exc
            if isinstance(raw_name, dict):
                raw_name = raw_name.get("name", f"class_{class_id}")
            names[class_id] = str(raw_name)
        return names

    classes_path = root / "classes.txt"
    if classes_path.is_file():
        lines = classes_path.read_text(encoding="utf-8-sig").splitlines()
        return {
            index: name.strip()
            for index, name in enumerate(lines)
            if name.strip()
        }
    return {}


def _resolve_path(root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _replace_images_component(image_dir: Path) -> Path | None:
    parts = list(image_dir.parts)
    positions = [index for index, part in enumerate(parts) if part.casefold() == "images"]
    if not positions:
        return None
    parts[positions[-1]] = "masks"
    return Path(*parts)


def split_directories(
    root: Path, config: dict[str, Any]
) -> dict[str, tuple[Path, Path]]:
    """Resolve images/masks directories for common semantic dataset layouts."""
    result: dict[str, tuple[Path, Path]] = {}
    for split in SPLITS:
        split_config = next(
            (
                config[key]
                for key in SPLIT_ALIASES[split]
                if config.get(key) is not None
            ),
            None,
        )
        raw_image: Any = None
        raw_mask: Any = None
        if isinstance(split_config, dict):
            raw_image = split_config.get("images", split_config.get("image"))
            raw_mask = split_config.get("masks", split_config.get("mask"))
        elif split_config is not None:
            raw_image = split_config

        image_candidates: list[Path] = []
        if raw_image is not None:
            configured = _resolve_path(root, raw_image)
            image_candidates.extend((configured / "images", configured))
        image_candidates.extend(
            candidate
            for alias in SPLIT_ALIASES[split]
            for candidate in (root / "images" / alias, root / alias / "images")
        )
        image_dir = next((path for path in image_candidates if path.is_dir()), None)

        mask_candidates: list[Path] = []
        if raw_mask is not None:
            mask_candidates.append(_resolve_path(root, raw_mask))
        if image_dir is not None:
            inferred = _replace_images_component(image_dir)
            if inferred is not None:
                mask_candidates.append(inferred)
        mask_candidates.extend(
            candidate
            for alias in SPLIT_ALIASES[split]
            for candidate in (root / "masks" / alias, root / alias / "masks")
        )
        mask_dir = next((path for path in mask_candidates if path.is_dir()), None)

        if image_dir is not None or mask_dir is not None:
            if image_dir is None or mask_dir is None:
                raise ValueError(f"{split} split 的 images/masks 目录不完整")
            result[split] = (image_dir.resolve(), mask_dir.resolve())
    return result


def default_search_root() -> Path:
    """Find the nearest datasets directory for interactive discovery."""
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if parent.name == "datasets" and parent.is_dir():
            return parent
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        candidate = parent / "datasets"
        if candidate.is_dir():
            return candidate
    return cwd


def discover_semantic_datasets(search_root: Path) -> list[SemanticDataset]:
    """Discover YAML-configured datasets containing image/mask split pairs."""
    search_root = search_root.expanduser().resolve()
    return [
        SemanticDataset(
            root=candidate.path,
            config_path=candidate.config_path,
            splits=candidate.splits,
            class_count=candidate.class_count,
            modified_time=candidate.modified_time,
        )
        for candidate in detect_datasets(
            search_root, kinds={"semantic_mask"}
        )
        if candidate.kind == "semantic_mask" and candidate.config_path is not None
    ]


def interactive_source(search_root: Path | None = None) -> Path:
    """Scan semantic datasets and let the user choose one."""
    root = (search_root or default_search_root()).expanduser().resolve()
    print(f"扫描语义分割数据集: {root}")
    datasets = discover_semantic_datasets(root)
    if datasets:
        print("\n检测到以下 PNG 语义分割数据集:")
        for index, dataset in enumerate(datasets, start=1):
            split_text = ",".join(dataset.splits)
            print(
                f"  {index:>2}. {dataset.root} "
                f"[split={split_text}; classes={dataset.class_count}; "
                f"modified={format_modified_time(dataset.modified_time)}]"
            )
        while True:
            raw = input("\n选择数据集 [序号，m 手动输入，q 退出]: ").strip()
            if raw.casefold() in {"q", "quit", "exit"}:
                raise KeyboardInterrupt
            if raw.casefold() in {"m", "manual", "手动"}:
                break
            if raw.isdigit() and 1 <= int(raw) <= len(datasets):
                return datasets[int(raw) - 1].config_path
            print(f"请输入 1..{len(datasets)}、m 或 q")
    else:
        print("扫描范围内未发现语义分割数据集，进入路径输入。")

    raw_path = input("输入 dataset_semantic 目录或 data.yaml 路径: ").strip()
    if not raw_path:
        raise ValueError("数据集路径为空")
    return Path(raw_path).expanduser()


def resolve_dataset(source: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Resolve a semantic dataset directory or its YAML configuration."""
    source = source.expanduser().resolve()
    if source.is_file():
        config_path = source
    else:
        discovered = [
            candidate
            for candidate in detect_datasets(
                source,
                kinds={"semantic_mask"},
                show_progress=False,
            )
            if candidate.kind == "semantic_mask"
            and candidate.config_path is not None
            and candidate.path == source
        ]
        config_path = (
            discovered[0].config_path
            if discovered
            else source / "data.yaml"
        )
    if not config_path.is_file():
        raise FileNotFoundError(f"数据集配置文件不存在: {config_path}")

    config = load_yaml(config_path)
    if not config:
        raise ValueError(f"数据集配置为空或格式无效: {config_path}")
    root = dataset_root_from_config(config_path, config)
    return root, config_path, config


def build_grayscale_mapping(
    names: dict[int, str],
    *,
    include_background: bool = False,
    strict_classes: bool = False,
    messages: list[str] | None = None,
) -> dict[str, Any]:
    """Build the mapping schema consumed by X-AnyLabeling MASK import."""
    if not names:
        raise ValueError("mask 中缺少可生成映射的前景像素值")

    colors: dict[str, int] = {}
    for class_id, raw_name in sorted(names.items()):
        name = raw_name.strip()
        if not name:
            if strict_classes:
                raise ValueError(f"类别 {class_id} 的名称为空")
            name = f"class_{class_id}"
            if messages is not None:
                messages.append(f"类别 {class_id} 名称为空，已命名为 {name!r}")
        if class_id < 0 or class_id > 255:
            raise ValueError(
                f"类别 {name!r} 的灰度值 {class_id} 超出 X-AnyLabeling 的 0..255 范围"
            )
        if not include_background and class_id == 0:
            continue
        if name in colors:
            if strict_classes:
                raise ValueError(f"类别名称重复，JSON 无法保留两个同名键: {name!r}")
            original_name = name
            name = f"{original_name}__id_{class_id}"
            suffix = 2
            while name in colors:
                name = f"{original_name}__id_{class_id}_{suffix}"
                suffix += 1
            if messages is not None:
                messages.append(
                    f"类别名 {original_name!r} 重复，像素 ID {class_id} 已命名为 {name!r}"
                )
        colors[name] = class_id

    if not colors:
        raise ValueError("映射中缺少可导入的前景类别")
    return {"type": "grayscale", "colors": colors}


def _files_by_stem(directory: Path, extensions: set[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in extensions:
            continue
        relative_stem = str(path.relative_to(directory).with_suffix(""))
        if relative_stem in files:
            raise ValueError(
                f"目录中存在相同相对 stem 的多个文件: {files[relative_stem]} / {path}"
            )
        files[relative_stem] = path
    return files


def _mask_values(mask_path: Path) -> tuple[set[int], tuple[int, int]]:
    with Image.open(mask_path) as mask_image:
        if len(mask_image.getbands()) != 1:
            raise ValueError(
                f"MASK 导入需要单通道灰度图，当前 mask 模式为 {mask_image.mode}: {mask_path}"
            )
        array = np.asarray(mask_image)
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"mask 像素类型需要为整数，当前为 {array.dtype}: {mask_path}")
        values = {int(value) for value in np.unique(array)}
        return values, mask_image.size


def validate_dataset(
    root: Path,
    config: dict[str, Any],
) -> ValidationStats:
    """Validate image/mask pairing and collect actual mask pixel values."""
    split_dirs = split_directories(root, config)
    if not split_dirs:
        raise ValueError("数据集中缺少可识别的 images/masks split")

    total_images = 0
    total_masks = 0
    checked_masks = 0
    observed_values: set[int] = set()

    for split in SPLITS:
        directories = split_dirs.get(split)
        if directories is None:
            continue
        image_dir, mask_dir = directories

        images = _files_by_stem(image_dir, IMAGE_EXTENSIONS)
        masks = _files_by_stem(mask_dir, MASK_EXTENSIONS)
        total_images += len(images)
        total_masks += len(masks)

        missing_masks = sorted(images.keys() - masks.keys())
        missing_images = sorted(masks.keys() - images.keys())
        if missing_masks:
            raise ValueError(
                f"{split} split 有 {len(missing_masks)} 张图片缺少同名 mask，"
                f"示例: {missing_masks[0]}"
            )
        if missing_images:
            raise ValueError(
                f"{split} split 有 {len(missing_images)} 个 mask 缺少同名图片，"
                f"示例: {missing_images[0]}"
            )

        for stem, mask_path in tqdm(
            masks.items(),
            total=len(masks),
            desc=f"校验 {split} masks",
            unit="mask",
            leave=False,
        ):
            values, mask_size = _mask_values(mask_path)
            observed_values.update(values)
            checked_masks += 1
            with Image.open(images[stem]) as image:
                if image.size != mask_size:
                    raise ValueError(
                        f"图片与 mask 尺寸不同: {images[stem]}={image.size}, "
                        f"{mask_path}={mask_size}"
                    )

    return ValidationStats(
        images=total_images,
        masks=total_masks,
        checked_masks=checked_masks,
        observed_values=tuple(sorted(observed_values)),
    )


def write_json_atomic(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def create_mapping(
    source: Path,
    output: Path | None = None,
    *,
    include_background: bool = False,
    validate: bool = True,
    ignore_values: set[int] | None = None,
    strict_classes: bool = False,
    messages: list[str] | None = None,
    dry_run: bool = False,
) -> tuple[Path, dict[str, Any], ValidationStats | None]:
    root, _config_path, config = resolve_dataset(source)
    configured_names = class_names(config, root)
    ignored = ignore_values or {255}
    stats = validate_dataset(root, config) if validate else None

    if stats is not None:
        observed = set(stats.observed_values) - ignored
        missing_ids = sorted(observed - set(configured_names))
        if missing_ids and strict_classes:
            raise ValueError(f"mask 含有 data.yaml 未定义的像素值 {missing_ids}")
        if missing_ids and messages is not None:
            inferred = ", ".join(f"{class_id}->class_{class_id}" for class_id in missing_ids)
            messages.append(f"mask 像素 ID 在 data.yaml 中未定义，已自动命名: {inferred}")
        names = {
            class_id: configured_names.get(class_id, f"class_{class_id}")
            for class_id in observed
        }
    else:
        names = {
            class_id: name
            for class_id, name in configured_names.items()
            if class_id not in ignored
        }

    payload = build_grayscale_mapping(
        names,
        include_background=include_background,
        strict_classes=strict_classes,
        messages=messages,
    )
    output_path = (
        output.expanduser().resolve()
        if output is not None
        else root / "mask_grayscale_map.json"
    )
    if not dry_run:
        write_json_atomic(payload, output_path)
    return output_path, payload, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从 PNG 语义分割 data.yaml 生成 X-AnyLabeling MASK 导入映射 JSON"
    )
    parser.add_argument(
        "--src",
        type=Path,
        help="语义分割数据集目录或 data.yaml 路径；省略后进入交互扫描",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        help="交互扫描范围；默认自动定位项目 datasets 目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="输出 JSON；默认写到数据集根目录 mask_grayscale_map.json",
    )
    parser.add_argument(
        "--include-background",
        action="store_true",
        help="把像素 ID=0 加入映射；默认将其作为画布背景",
    )
    parser.add_argument(
        "--validation",
        choices=("all", "none"),
        default="all",
        help="all 检查全部图片/mask；none 仅生成映射（默认: all）",
    )
    parser.add_argument(
        "--ignore-value",
        type=int,
        action="append",
        default=[255],
        help="校验时允许的忽略像素值，可重复传入（默认: 255）",
    )
    parser.add_argument(
        "--strict-classes",
        action="store_true",
        help="要求 mask 像素 ID 全部在 YAML 定义且类别名称唯一",
    )
    parser.add_argument("--dry-run", action="store_true", help="执行完整校验并显示映射计划，保持零写入")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        source = args.src or interactive_source(args.datasets)
        messages: list[str] = []
        output, payload, stats = create_mapping(
            source,
            args.out,
            include_background=args.include_background,
            validate=args.validation == "all",
            ignore_values=set(args.ignore_value),
            strict_classes=args.strict_classes,
            messages=messages,
            dry_run=args.dry_run,
        )
    except KeyboardInterrupt:
        print("\n已退出")
        return 130
    except (EOFError, OSError, ValueError) as exc:
        print(f"生成失败: {exc}")
        return 2

    print(f"{'[dry-run] 计划生成' if args.dry_run else '已生成'}: {output}")
    print(f"映射类别: {len(payload['colors'])}")
    for message in messages:
        print(f"提示: {message}")
    if stats is not None:
        print(
            f"校验完成: images={stats.images}, masks={stats.masks}, "
            f"像素值={list(stats.observed_values)}"
        )
    print("X-AnyLabeling: 打开 images/<split>，选择导入 MASK 标注，依次选择此 JSON 和 masks/<split>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
