#!/usr/bin/env python3
"""Convert standard MVTec AD directories to YOLO segmentation and detection.

Supported input:

    <root>/<category>/
        train/good/<image>
        test/good/<image>
        test/<defect>/<image>
        ground_truth/<defect>/<stem>_mask.png

``src`` may point to ``<root>`` or directly to one ``<category>`` directory.
By default every image is written to YOLO ``train`` so generated MVTec anomaly
images can be used for supervised training. ``--split-mode preserve`` keeps
MVTec train/test membership.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import cv2
    import numpy as np
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
    if not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        sys.exit(f"缺少依赖 {exc.name}，请在项目训练环境中运行。当前 Python: {sys.executable}")

try:
    from .generated_mask_to_yoloseg import mask_to_polygons
    from .dataset_transaction import staged_output, validate_output_location
    from .dataset_discovery import auto_datasets_root
    from .dataset_detector import detect_datasets
    from .interactive_helpers import prompt_choice, prompt_path
    from .output_naming import default_output_dir
    from .progress import tqdm
except ImportError:
    from generated_mask_to_yoloseg import mask_to_polygons  # type: ignore[no-redef]
    from dataset_transaction import (  # type: ignore[no-redef]
        staged_output,
        validate_output_location,
    )
    from dataset_discovery import auto_datasets_root  # type: ignore[no-redef]
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from interactive_helpers import prompt_choice, prompt_path  # type: ignore[no-redef]
    from output_naming import default_output_dir  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
SPLITS = ("train", "val", "test")
TASKS = ("segment", "detect")


@dataclass(frozen=True)
class Sample:
    category: str
    defect: str
    source_split: str
    image: Path
    mask: Path | None


@dataclass(frozen=True)
class ScanIssue:
    status: str
    category: str
    defect: str
    stem: str
    image: str = ""
    mask: str = ""
    detail: str = ""


def natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", value)]


def _image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda path: natural_key(path.name),
    )


def discover_category_roots(src: Path) -> list[Path]:
    """Return MVTec category directories below ``src``."""
    src = src.expanduser().resolve()
    if not src.is_dir():
        raise ValueError(f"输入目录不存在: {src}")
    if (src / "test").is_dir() and (src / "ground_truth").is_dir():
        return [src]
    categories = [
        path for path in src.iterdir()
        if path.is_dir()
        and (path / "test").is_dir()
        and (path / "ground_truth").is_dir()
    ]
    return sorted(categories, key=lambda path: natural_key(path.name))


def _unique_files_by_stem(directory: Path) -> tuple[dict[str, Path], set[str]]:
    grouped: dict[str, list[Path]] = {}
    for path in _image_files(directory):
        grouped.setdefault(path.stem, []).append(path)
    unique = {stem: paths[0] for stem, paths in grouped.items() if len(paths) == 1}
    duplicate = {stem for stem, paths in grouped.items() if len(paths) > 1}
    return unique, duplicate


def _mask_for_image(mask_dir: Path, image_stem: str) -> tuple[Path | None, bool]:
    """Resolve ``stem_mask`` first, then the less common same-stem form."""
    masks, duplicates = _unique_files_by_stem(mask_dir)
    for stem in (f"{image_stem}_mask", image_stem):
        if stem in duplicates:
            return None, True
        if stem in masks:
            return masks[stem], False
    return None, False


def _mask_from_index(
    masks: dict[str, Path], duplicates: set[str], image_stem: str
) -> tuple[Path | None, bool]:
    """Resolve one image against a mask index built once for its defect."""
    for stem in (f"{image_stem}_mask", image_stem):
        if stem in duplicates:
            return None, True
        if stem in masks:
            return masks[stem], False
    return None, False


def discover_samples(src: Path) -> tuple[list[Sample], list[ScanIssue]]:
    categories = discover_category_roots(src)
    if not categories:
        raise ValueError(f"未发现标准 MVTec category（需要 test/ 和 ground_truth/）: {src}")

    samples: list[Sample] = []
    issues: list[ScanIssue] = []
    for category_dir in categories:
        category = category_dir.name
        for image in _image_files(category_dir / "train" / "good"):
            samples.append(Sample(category, "good", "train", image, None))
        for image in _image_files(category_dir / "test" / "good"):
            samples.append(Sample(category, "good", "test", image, None))

        test_dir = category_dir / "test"
        for defect_dir in sorted(
            (
                path for path in test_dir.iterdir()
                if path.is_dir() and path.name != "good"
            ),
            key=lambda path: natural_key(path.name),
        ):
            defect = defect_dir.name
            mask_dir = category_dir / "ground_truth" / defect
            masks, duplicates = _unique_files_by_stem(mask_dir)
            for image in _image_files(defect_dir):
                mask, duplicate = _mask_from_index(masks, duplicates, image.stem)
                if duplicate:
                    issues.append(ScanIssue(
                        "duplicate_mask", category, defect, image.stem,
                        image=str(image), detail="同一 mask stem 存在多个图片扩展名",
                    ))
                    continue
                if mask is None:
                    issues.append(ScanIssue(
                        "missing_mask", category, defect, image.stem,
                        image=str(image), detail="缺少 <stem>_mask 或同 stem 掩码",
                    ))
                    continue
                samples.append(Sample(category, defect, "test", image, mask))
    return samples, issues


def class_name(sample: Sample, class_mode: str) -> str:
    if class_mode == "object":
        return sample.category
    if class_mode == "object-defect":
        return f"{sample.category}__{sample.defect}"
    return sample.defect


def resolve_class_names(samples: list[Sample], class_mode: str) -> list[str]:
    names = {
        class_name(sample, class_mode)
        for sample in samples
        if sample.mask is not None
    }
    return sorted(names, key=natural_key)


def _safe_component(value: str) -> str:
    value = re.sub(r"[\\/\s]+", "_", value.strip())
    value = re.sub(r"[^\w.()\-\u4e00-\u9fff]+", "_", value, flags=re.UNICODE)
    return value.strip("._") or "unknown"


def _output_stem(sample: Sample, src: Path, used: set[str]) -> str:
    base = "__".join(map(
        _safe_component,
        (sample.category, sample.defect, sample.image.stem),
    ))
    if base not in used:
        used.add(base)
        return base
    relative = (
        str(sample.image.relative_to(src))
        if sample.image.is_relative_to(src)
        else str(sample.image)
    )
    suffix = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
    candidate = f"{base}__{suffix}"
    counter = 2
    while candidate in used:
        candidate = f"{base}__{suffix}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


def _task_outputs(out: Path, task: str) -> dict[str, Path]:
    if task == "both":
        return {
            "segment": out / "dataset_seg",
            "detect": out / "dataset_det",
        }
    return {task: out}


def _write_metadata(
    root: Path,
    task: str,
    names: list[str],
    sources: dict[str, set[str]],
) -> None:
    data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": task,
        "nc": len(names),
        "names": {index: name for index, name in enumerate(names)},
    }
    (root / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    mapping = {
        str(index): {
            "name": name,
            "sources": sorted(sources.get(name, set())),
        }
        for index, name in enumerate(names)
    }
    (root / "class_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _prepare_outputs(outputs: dict[str, Path], clean: bool) -> None:
    for root in outputs.values():
        if clean and root.exists():
            shutil.rmtree(root)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"输出目录已存在且含有文件: {root}")
    for root in outputs.values():
        for split in SPLITS:
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _copy_image(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def _segment_lines(polygons: list[np.ndarray], class_id: int, width: int, height: int) -> list[str]:
    lines: list[str] = []
    for polygon in polygons:
        coordinates: list[str] = []
        for x, y in polygon:
            nx = min(max(float(x) / float(width), 0.0), 1.0)
            ny = min(max(float(y) / float(height), 0.0), 1.0)
            coordinates.extend((f"{nx:.6f}", f"{ny:.6f}"))
        lines.append(f"{class_id} " + " ".join(coordinates))
    return lines


def _detect_lines(polygons: list[np.ndarray], class_id: int, width: int, height: int) -> list[str]:
    lines: list[str] = []
    for polygon in polygons:
        x, y, box_width, box_height = cv2.boundingRect(polygon.reshape(-1, 1, 2))
        center_x = (float(x) + float(box_width) / 2.0) / float(width)
        center_y = (float(y) + float(box_height) / 2.0) / float(height)
        normalized_width = float(box_width) / float(width)
        normalized_height = float(box_height) / float(height)
        lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{normalized_width:.6f} {normalized_height:.6f}"
        )
    return lines


REPORT_FIELDS = (
    "status", "category", "defect", "class_name", "class_id", "source_split",
    "output_split", "stem", "source_image", "source_mask", "output_stem",
    "instances", "foreground_ratio", "detail",
)


def _convert_unpublished(
    src: Path,
    out: Path,
    *,
    task: str = "both",
    class_mode: str = "defect",
    split_mode: str = "all-train",
    threshold: int = 127,
    min_area: float = 1.0,
    epsilon: float = 0.001,
    verbose: bool = True,
    report_output: Path | None = None,
) -> dict[str, object]:
    """Convert a MVTec dataset into YOLO segment, detect, or both."""
    if task not in (*TASKS, "both"):
        raise ValueError(f"未知任务: {task}")
    if class_mode not in ("defect", "object", "object-defect"):
        raise ValueError(f"未知类别模式: {class_mode}")
    if split_mode not in ("all-train", "preserve"):
        raise ValueError(f"未知划分模式: {split_mode}")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold 必须位于 0..255")
    if min_area < 0 or epsilon < 0:
        raise ValueError("min-area 和 epsilon 必须大于等于 0")

    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    published_root = report_output or out
    samples, scan_issues = discover_samples(src)
    if not samples:
        raise ValueError(f"MVTec 数据集中未发现图片: {src}")
    names = resolve_class_names(samples, class_mode)
    if not names:
        raise ValueError(f"MVTec 数据集中未发现带掩码的缺陷样本: {src}")
    class_ids = {name: index for index, name in enumerate(names)}
    outputs = _task_outputs(out, task)
    _prepare_outputs(outputs, clean=False)

    sources: dict[str, set[str]] = {name: set() for name in names}
    rows: list[dict[str, object]] = [
        {
            "status": issue.status,
            "category": issue.category,
            "defect": issue.defect,
            "class_name": "",
            "class_id": "",
            "source_split": "test",
            "output_split": "",
            "stem": issue.stem,
            "source_image": issue.image,
            "source_mask": issue.mask,
            "output_stem": "",
            "instances": 0,
            "foreground_ratio": "",
            "detail": issue.detail,
        }
        for issue in scan_issues
    ]

    used_stems: set[str] = set()
    converted = empty_masks = skipped = 0
    for sample in tqdm(
        samples,
        desc=f"转换 {src.name}",
        unit="图片",
        disable=not verbose,
    ):
        image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR)
        mask = (
            cv2.imread(str(sample.mask), cv2.IMREAD_UNCHANGED)
            if sample.mask is not None
            else None
        )
        status = "converted"
        detail = ""
        polygons: list[np.ndarray] = []
        foreground_ratio = 0.0
        if image is None:
            status, detail = "unreadable", "图片读取失败"
        elif sample.mask is not None and mask is None:
            status, detail = "unreadable", "掩码读取失败"
        elif mask is not None and image.shape[:2] != mask.shape[:2]:
            status = "size_mismatch"
            detail = (
                f"image={image.shape[1]}x{image.shape[0]}, "
                f"mask={mask.shape[1]}x{mask.shape[0]}"
            )
        elif mask is not None:
            polygons, _, foreground_ratio = mask_to_polygons(
                mask,
                threshold=threshold,
                min_area=min_area,
                epsilon=epsilon,
                auto_invert=False,
            )

        output_split = (
            sample.source_split if split_mode == "preserve" else "train"
        )
        output_stem = ""
        resolved_class_name = ""
        resolved_class_id: int | str = ""
        if status == "converted":
            output_stem = _output_stem(sample, src, used_stems)
            height, width = image.shape[:2]
            if sample.mask is not None:
                resolved_class_name = class_name(sample, class_mode)
                resolved_class_id = class_ids[resolved_class_name]
                sources[resolved_class_name].add(
                    f"{sample.category}/{sample.defect}"
                )
            for output_task, root in outputs.items():
                image_destination = (
                    root / "images" / output_split
                    / f"{output_stem}{sample.image.suffix.lower()}"
                )
                label_destination = (
                    root / "labels" / output_split / f"{output_stem}.txt"
                )
                _copy_image(sample.image, image_destination)
                if sample.mask is None:
                    lines: list[str] = []
                elif output_task == "segment":
                    lines = _segment_lines(
                        polygons, int(resolved_class_id), width, height
                    )
                else:
                    lines = _detect_lines(
                        polygons, int(resolved_class_id), width, height
                    )
                label_destination.write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )
            converted += 1
            if sample.mask is not None and not polygons:
                status = "empty_mask"
                detail = "阈值处理后未提取到有效轮廓"
                empty_masks += 1
        else:
            skipped += 1

        rows.append({
            "status": status,
            "category": sample.category,
            "defect": sample.defect,
            "class_name": resolved_class_name,
            "class_id": resolved_class_id,
            "source_split": sample.source_split,
            "output_split": output_split if status != "unreadable" else "",
            "stem": sample.image.stem,
            "source_image": str(sample.image),
            "source_mask": str(sample.mask or ""),
            "output_stem": output_stem,
            "instances": len(polygons),
            "foreground_ratio": f"{foreground_ratio:.6f}",
            "detail": detail,
        })

    for output_task, root in outputs.items():
        _write_metadata(root, output_task, names, sources)
    report_path = out / "conversion_report.csv" if task == "both" else next(
        iter(outputs.values())
    ) / "conversion_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    published_outputs = {
        output_task: published_root / root.relative_to(out)
        for output_task, root in outputs.items()
    }
    published_report = published_root / report_path.relative_to(out)
    result: dict[str, object] = {
        "src": src,
        "out": published_root,
        "outputs": published_outputs,
        "samples": len(samples),
        "converted": converted,
        "empty_masks": empty_masks,
        "skipped": skipped,
        "scan_issues": len(scan_issues),
        "names": names,
        "report": published_report,
    }
    if verbose:
        print(f"转换完成: {src} -> {published_root}")
        print(
            f"样本 {len(samples)}，写入 {converted}，空掩码 {empty_masks}，"
            f"跳过 {skipped}，配对问题 {len(scan_issues)}"
        )
        print("类别: " + ", ".join(f"{index}={name}" for index, name in enumerate(names)))
        for output_task, root in published_outputs.items():
            print(f"{output_task}: {root / 'data.yaml'}")
        print(f"报告: {published_report}")
    return result


def convert(
    src: Path,
    out: Path,
    *,
    task: str = "both",
    class_mode: str = "defect",
    split_mode: str = "all-train",
    threshold: int = 127,
    min_area: float = 1.0,
    epsilon: float = 0.001,
    clean: bool = False,
    verbose: bool = True,
) -> dict[str, object]:
    """Convert into a validated staging tree and publish it atomically."""
    source = src.expanduser().resolve()
    published_output = validate_output_location(out, [source])
    with staged_output(published_output, clean=clean) as stage:
        result = _convert_unpublished(
            source,
            stage,
            task=task,
            class_mode=class_mode,
            split_mode=split_mode,
            threshold=threshold,
            min_area=min_area,
            epsilon=epsilon,
            verbose=verbose,
            report_output=published_output,
        )
    return result


def choose_source(search_root: Path) -> Path | None:
    """Scan and interactively select one MVTec dataset root."""
    search_root = search_root.expanduser().resolve()
    print(f"\n扫描 MVTec AD 数据: {search_root}")
    candidates = detect_datasets(search_root, kinds={"mvtec"})
    if not candidates:
        print("扫描范围内暂未发现包含 test/ 和 ground_truth/ 的 MVTec 数据。")
        return None
    print(f"\n{'#':>3} {'图像':>8} {'掩码':>8} {'类别':>8}  路径")
    print("-" * 78)
    for index, candidate in enumerate(candidates, start=1):
        try:
            display = candidate.path.relative_to(search_root)
        except ValueError:
            display = candidate.path
        print(
            f"{index:>3} {candidate.image_count:>8} "
            f"{candidate.annotation_count:>8} {candidate.class_count:>8}  {display}"
        )
    default = 0 if len(candidates) == 1 else None
    default_hint = "，回车选 1" if default is not None else ""
    selected = prompt_choice(
        f"\n选择数据集 [1-{len(candidates)}{default_hint}，q 退出]: ",
        len(candidates),
        default=default,
    )
    return candidates[selected].path if selected is not None else None


def print_analysis(src: Path, class_mode: str) -> None:
    samples, issues = discover_samples(src)
    names = resolve_class_names(samples, class_mode)
    categories = sorted({sample.category for sample in samples}, key=natural_key)
    anomaly_count = sum(sample.mask is not None for sample in samples)
    print("\n数据分析")
    print(f"  输入       : {src}")
    print(f"  category   : {len(categories)}")
    print(f"  图片       : {len(samples)}")
    print(f"  异常掩码   : {anomaly_count}")
    print(f"  配对问题   : {len(issues)}")
    print(f"  类别模式   : {class_mode}")
    print("  YOLO 类别  : " + ", ".join(
        f"{index}={name}" for index, name in enumerate(names)
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="标准 MVTec AD → YOLO Seg/Det")
    parser.add_argument("--src", type=Path, help="MVTec 数据集根或单个 category；省略时自动扫描")
    parser.add_argument("--out", type=Path, help="输出目录；省略时交互确认默认路径")
    parser.add_argument("--datasets", type=Path, help="自动扫描根目录，默认仓库 datasets/")
    parser.add_argument(
        "--task",
        choices=("segment", "detect", "both"),
        default="both",
        help="输出任务，默认同时生成 Seg 和 Det",
    )
    parser.add_argument(
        "--class-mode",
        choices=("defect", "object", "object-defect"),
        default="defect",
        help="类别来自缺陷名、物体名或物体+缺陷，默认 defect",
    )
    parser.add_argument(
        "--split-mode",
        choices=("all-train", "preserve"),
        default="all-train",
        help="全部写入 train，或保留 MVTec train/test，默认 all-train",
    )
    parser.add_argument("--threshold", type=int, default=127, help="掩码二值阈值")
    parser.add_argument("--min-area", type=float, default=1.0, help="最小实例面积")
    parser.add_argument("--epsilon", type=float, default=0.001, help="轮廓简化比例")
    parser.add_argument("--clean", action="store_true", help="清空已有输出后重建")
    parser.add_argument("--yes", action="store_true", help="使用默认输出并直接执行")
    parser.add_argument("--dry-run", action="store_true", help="分析输入并显示输出计划，保持零写入")
    args = parser.parse_args(argv)
    interactive_mode = args.src is None or args.out is None
    try:
        if args.src:
            src = args.src.expanduser().resolve()
        else:
            src = choose_source(args.datasets or auto_datasets_root())
            if src is None:
                print("已退出。")
                return 0

        print_analysis(src, args.class_mode)
        default_out = default_output_dir(src, "mvtec-to-yolo")
        if args.out:
            out = args.out.expanduser().resolve()
        elif args.yes:
            out = default_out
        else:
            out = prompt_path("\n输出目录", default_out)
            if out is None:
                print("已退出。")
                return 0

        if args.dry_run:
            validate_output_location(out, [src])
            print("\n[dry-run] MVTec AD → YOLO")
            print(f"  输入: {src}")
            print(f"  输出: {out}")
            print(f"  task={args.task}, class_mode={args.class_mode}, split_mode={args.split_mode}")
            return 0

        clean = args.clean
        if out.exists() and any(out.iterdir()) and not clean:
            if args.yes:
                raise FileExistsError(f"输出目录已存在且含有文件: {out}；使用 --clean 可重建")
            answer = input(f"输出目录已有内容，清空并重建 {out}？[y/N]: ").strip().lower()
            clean = answer in ("y", "yes", "是")
            if not clean:
                print("已取消。")
                return 0
        if interactive_mode and not args.yes:
            answer = input(
                f"开始生成 {args.task} 数据到 {out}？[Y/n]: "
            ).strip().lower()
            if answer in ("n", "no", "否"):
                print("已取消。")
                return 0

        convert(
            src,
            out,
            task=args.task,
            class_mode=args.class_mode,
            split_mode=args.split_mode,
            threshold=args.threshold,
            min_area=args.min_area,
            epsilon=args.epsilon,
            clean=clean,
        )
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    except EOFError:
        print("\n已退出。")
        return 0
    except KeyboardInterrupt:
        print("\n已退出。")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
