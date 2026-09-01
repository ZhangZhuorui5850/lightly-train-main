#!/usr/bin/env python3
"""Convert YOLO polygon labels into semantic class-ID PNG masks."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw

try:
    from .dataset_discovery import (
        IMAGE_EXTENSIONS,
        SPLITS,
        auto_datasets_root,
        class_names,
        dataset_root_from_config,
        find_dataset_config,
        load_yaml,
        split_sample_files,
    )
    from .dataset_detector import detect_datasets, inspect_dataset
    from .output_naming import default_output_dir
    from .dataset_transaction import staged_output, validate_output_location
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    from dataset_discovery import (  # type: ignore[no-redef]
        IMAGE_EXTENSIONS,
        SPLITS,
        auto_datasets_root,
        class_names,
        dataset_root_from_config,
        find_dataset_config,
        load_yaml,
        split_sample_files,
    )
    from dataset_detector import (  # type: ignore[no-redef]
        detect_datasets,
        inspect_dataset,
    )
    from output_naming import default_output_dir  # type: ignore[no-redef]
    from dataset_transaction import (  # type: ignore[no-redef]
        staged_output,
        validate_output_location,
    )
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


@dataclass
class ConversionIssue:
    split: str
    image: str
    label: str
    line: int | None
    message: str


@dataclass
class ConversionStats:
    images: int = 0
    polygons: int = 0
    background_images: int = 0
    invalid_rows: int = 0


def resolve_source(source: Path) -> tuple[Path, Path, dict]:
    source = source.expanduser().resolve()
    config_path = find_dataset_config(source)
    config = load_yaml(config_path)
    root = dataset_root_from_config(config_path, config)
    return root, config_path, config


def parse_polygon_row(row: str) -> tuple[int, list[float]]:
    tokens = row.split()
    if len(tokens) < 7 or len(tokens) % 2 == 0:
        raise ValueError("YOLO polygon 行需要 class_id 和至少 3 个坐标点")
    try:
        class_value = float(tokens[0])
        coords = [float(value) for value in tokens[1:]]
    except ValueError as exc:
        raise ValueError("标签行包含非数字内容") from exc
    class_id = int(class_value)
    if class_value != class_id or class_id < 0:
        raise ValueError("class_id 需要使用非负整数")
    if any(not math.isfinite(value) for value in coords):
        raise ValueError("坐标需要使用有限数值")
    return class_id, coords


def polygon_points(coords: list[float], width: int, height: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for x_norm, y_norm in zip(coords[0::2], coords[1::2]):
        x = round(min(max(x_norm, 0.0), 1.0) * max(width - 1, 0))
        y = round(min(max(y_norm, 0.0), 1.0) * max(height - 1, 0))
        points.append((x, y))
    return points


def _polygon_area(points: list[tuple[int, int]]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def rasterize_polygons(
    *,
    width: int,
    height: int,
    polygons: list[tuple[int, list[tuple[int, int]]]],
    class_mapping: dict[int, int],
    background_id: int,
    overlap: str,
) -> np.ndarray:
    max_id = max([background_id, *class_mapping.values()])
    dtype = np.uint8 if max_id <= 255 else np.uint16
    mask = np.full((height, width), background_id, dtype=dtype)
    ordered = polygons
    policy = overlap
    if overlap == "larger":
        ordered = sorted(polygons, key=lambda item: _polygon_area(item[1]), reverse=True)
        policy = "first"

    for source_class, points in ordered:
        target_class = class_mapping[source_class]
        if policy == "last":
            mask_image = Image.fromarray(mask)
            ImageDraw.Draw(mask_image).polygon(points, fill=int(target_class))
            mask = np.asarray(mask_image, dtype=dtype).copy()
            continue
        region_image = Image.new("1", (width, height), 0)
        ImageDraw.Draw(region_image).polygon(points, fill=1)
        region = np.asarray(region_image, dtype=bool)
        mask[region & (mask == background_id)] = target_class
    return mask


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype == np.uint16:
        Image.fromarray(mask, mode="I;16").save(path)
    else:
        Image.fromarray(mask).save(path)


def _transfer_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"未知文件传输模式: {mode}")


def _image_files(path: Path) -> Iterable[Path]:
    return sorted(
        file
        for file in path.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_polygons(
    label_path: Path,
    *,
    split: str,
    image_path: Path,
    width: int,
    height: int,
    class_mapping: dict[int, int],
    issues: list[ConversionIssue],
) -> list[tuple[int, list[tuple[int, int]]]]:
    if not label_path.is_file():
        return []
    polygons: list[tuple[int, list[tuple[int, int]]]] = []
    for line_number, raw in enumerate(
        read_text_auto(label_path).splitlines(), start=1
    ):
        row = raw.strip()
        if not row:
            continue
        try:
            class_id, coords = parse_polygon_row(row)
            if class_id not in class_mapping:
                raise ValueError(f"class_id {class_id} 未出现在 data.yaml names/classes 中")
            points = polygon_points(coords, width, height)
            if len(set(points)) < 3 or _polygon_area(points) == 0:
                raise ValueError("多边形栅格化后面积为 0")
            polygons.append((class_id, points))
        except ValueError as exc:
            issues.append(
                ConversionIssue(
                    split=split,
                    image=str(image_path),
                    label=str(label_path),
                    line=line_number,
                    message=str(exc),
                )
            )
    return polygons


def _convert_dataset_unpublished(
    source: Path,
    output: Path,
    *,
    background_id: int = 0,
    class_offset: int = 1,
    background_name: str = "background",
    overlap: str = "last",
    image_mode: str = "copy",
    report_output: Path | None = None,
) -> dict:
    root, config_path, config = resolve_source(source)
    inspected = inspect_dataset(config_path)
    if inspected.kind != "yolo_instance":
        raise ValueError(f"源数据集类型为 {inspected.kind}，需要 YOLO polygon 实例分割数据集")
    names = class_names(config, root)
    if not names:
        raise ValueError("data.yaml/classes.txt 中缺少类别定义")
    if background_id < 0 or class_offset < 0:
        raise ValueError("background_id 和 class_offset 需要使用非负整数")
    class_mapping = {class_id: class_id + class_offset for class_id in names}
    if background_id in class_mapping.values():
        raise ValueError("background_id 与转换后的类别 ID 冲突，请调整 --class-offset")
    if overlap not in {"last", "first", "larger"}:
        raise ValueError(f"未知重叠策略: {overlap}")

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    samples_by_split = {
        split: split_sample_files(root, config, split, annotation="labels")
        for split in SPLITS
    }
    if not samples_by_split["train"] or not samples_by_split["val"]:
        raise ValueError("源数据集需要提供 train 和 val images split")

    issues: list[ConversionIssue] = []
    stats: dict[str, ConversionStats] = {}
    for split in SPLITS:
        split_samples = samples_by_split[split]
        if not split_samples:
            continue
        split_stats = ConversionStats()
        stats[split] = split_stats
        for image_path, label_path, relative_path in tqdm(
            split_samples,
            desc=f"转换 {root.name}/{split}",
            unit="图片",
        ):
            with Image.open(image_path) as image:
                width, height = image.size
            polygons = _read_polygons(
                label_path,
                split=split,
                image_path=image_path,
                width=width,
                height=height,
                class_mapping=class_mapping,
                issues=issues,
            )
            mask = rasterize_polygons(
                width=width,
                height=height,
                polygons=polygons,
                class_mapping=class_mapping,
                background_id=background_id,
                overlap=overlap,
            )
            _transfer_file(image_path, output / "images" / split / relative_path, image_mode)
            _save_mask(
                mask,
                (output / "masks" / split / relative_path).with_suffix(".png"),
            )
            split_stats.images += 1
            split_stats.polygons += len(polygons)
            if not polygons:
                split_stats.background_images += 1
        split_stats.invalid_rows = sum(1 for issue in issues if issue.split == split)

    semantic_classes = {background_id: background_name}
    semantic_classes.update({class_mapping[class_id]: name for class_id, name in names.items()})
    data: dict = {"task": "semantic_segmentation"}
    for split in SPLITS:
        if split in stats:
            data[split] = {"images": f"images/{split}", "masks": f"masks/{split}"}
    data["classes"] = dict(sorted(semantic_classes.items()))
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    max_class_id = max(semantic_classes)
    (output / "classes.txt").write_text(
        "\n".join(
            semantic_classes.get(class_id, f"class_{class_id}")
            for class_id in range(max_class_id + 1)
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "source": str(root),
        "source_config": str(config_path),
        "output": str(report_output or output),
        "class_mapping": {str(key): value for key, value in class_mapping.items()},
        "background_id": background_id,
        "overlap": overlap,
        "image_mode": image_mode,
        "splits": {split: asdict(value) for split, value in stats.items()},
        "issues": [asdict(issue) for issue in issues],
    }
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def convert_dataset(
    source: Path,
    output: Path,
    *,
    background_id: int = 0,
    class_offset: int = 1,
    background_name: str = "background",
    overlap: str = "last",
    image_mode: str = "copy",
    clean: bool = False,
) -> dict:
    """Convert into a validated staging tree and publish it atomically."""
    source_root, _, _ = resolve_source(source)
    published_output = validate_output_location(output, [source_root])
    with staged_output(published_output, clean=clean) as stage:
        report = _convert_dataset_unpublished(
            source,
            stage,
            background_id=background_id,
            class_offset=class_offset,
            background_name=background_name,
            overlap=overlap,
            image_mode=image_mode,
            report_output=published_output,
        )
    return report


def _choose_source(search_root: Path) -> Path | None:
    candidates = detect_datasets(search_root, kinds={"yolo_instance"})
    if not candidates:
        print(f"在 {search_root} 下没有检索到 YOLO 实例分割数据集。")
        return None
    print(f"\n检索到 {len(candidates)} 个可转换的 YOLO 实例分割数据集:\n")
    for index, item in enumerate(candidates, start=1):
        try:
            display = item.path.relative_to(search_root)
        except ValueError:
            display = item.path
        print(
            f"  {index:>2}. {display}  "
            f"图像={item.image_count} 标签={item.annotation_count} 类别={item.class_count}"
        )
    while True:
        try:
            raw = input(f"\n选择数据集 [1-{len(candidates)}，q 退出]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            candidate = candidates[int(raw) - 1]
            return candidate.config_path or candidate.path
        print("请输入列表中的序号。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO polygon txt → train_seg.py 可用的 PNG 语义掩码数据集"
    )
    parser.add_argument("--src", type=Path, help="YOLO-seg 数据集目录或 data.yaml；省略时自动扫描")
    parser.add_argument("--out", type=Path, help="输出 dataset_semantic 目录")
    parser.add_argument("--datasets", type=Path, help="自动扫描根目录，默认仓库 datasets/")
    parser.add_argument("--background-id", type=int, default=0)
    parser.add_argument("--background-name", default="background")
    parser.add_argument("--class-offset", type=int, default=1, help="YOLO class ID 的偏移量")
    parser.add_argument(
        "--overlap",
        choices=("last", "first", "larger"),
        default="last",
        help="实例重叠区域归属：后标注、先标注、较大实例",
    )
    parser.add_argument(
        "--image-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    parser.add_argument("--clean", action="store_true", help="清理已有输出后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只验证输入并显示输出计划，保持零写入")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.src
    if source is None:
        source = _choose_source((args.datasets or auto_datasets_root()).expanduser().resolve())
        if source is None:
            return 1
    source_root, _, _ = resolve_source(source)
    output = args.out or default_output_dir(source_root, "to-semantic")
    if args.out is None:
        try:
            raw = input(f"输出目录 [{output}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 1
        if raw:
            output = Path(raw)
    output = validate_output_location(output, [source_root])
    if args.dry_run:
        print("[dry-run] YOLO Seg → PNG Semantic")
        print(f"  输入: {source_root}")
        print(f"  输出: {output}")
        print(f"  overlap={args.overlap}, image_mode={args.image_mode}, class_offset={args.class_offset}")
        return 0
    report = convert_dataset(
        source,
        output,
        background_id=args.background_id,
        class_offset=args.class_offset,
        background_name=args.background_name,
        overlap=args.overlap,
        image_mode=args.image_mode,
        clean=args.clean,
    )
    totals = report["splits"]
    image_count = sum(item["images"] for item in totals.values())
    polygon_count = sum(item["polygons"] for item in totals.values())
    print(f"\n转换完成: 图像 {image_count}，多边形 {polygon_count}")
    print(f"输出目录: {report['output']}")
    print(f"审计报告: {Path(report['output']) / 'conversion_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
