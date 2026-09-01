#!/usr/bin/env python3
"""交互式编辑 YOLO 检测或实例分割数据集的类别。

删除类别会删除对应标注行；合并、重命名后会把全部类别连续重排为
0..N-1，并同步生成 labels、data.yaml、classes.txt 和审计报告。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

try:
    from . import semantic_class_editor as common
    from .dataset_discovery import (
        IMAGE_EXTENSIONS,
        SPLITS,
        auto_datasets_root,
        dataset_path_diagnostics,
        dataset_root_from_config,
        label_dir_from_image_dir,
        split_annotation_dirs,
        split_image_files,
        split_image_dirs,
    )
    from .dataset_detector import detect_datasets, inspect_dataset
    from .dataset_transaction import staged_output, validate_output_location
    from .output_naming import default_output_dir
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    import semantic_class_editor as common  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        IMAGE_EXTENSIONS,
        SPLITS,
        auto_datasets_root,
        dataset_path_diagnostics,
        dataset_root_from_config,
        label_dir_from_image_dir,
        split_annotation_dirs,
        split_image_files,
        split_image_dirs,
    )
    from dataset_detector import (  # type: ignore[no-redef]
        detect_datasets,
        inspect_dataset,
    )
    from dataset_transaction import staged_output, validate_output_location  # type: ignore[no-redef]
    from output_naming import default_output_dir  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


YOLO_KINDS = {"yolo_detection", "yolo_instance"}


@dataclass(frozen=True)
class YoloSplitPaths:
    images: Path
    labels: Path
    image_files: tuple[Path, ...] = ()
    label_files: tuple[Path, ...] = ()


@dataclass
class YoloDataset:
    config_path: Path
    root: Path
    config: dict[str, Any]
    class_names: dict[int, str]
    raw_class_names: dict[int, Any]
    splits: dict[str, YoloSplitPaths]
    kind: str


@dataclass
class YoloAnalysis:
    shared: common.DatasetAnalysis
    annotation_count: int
    malformed_count: int


def resolve_dataset(
    source: Path,
    *,
    forced_kind: str | None = None,
    selected_splits: tuple[str, ...] | None = None,
) -> YoloDataset:
    config_path = common._find_config(source)
    config = common.read_yaml(config_path)
    root = dataset_root_from_config(config_path, config)
    names, raw_names = common.parse_class_names(config)
    requested_splits = selected_splits or SPLITS
    image_dirs = split_image_dirs(
        root, config, selected_splits=requested_splits
    )
    configured_images = split_image_files(
        root, config, selected_splits=requested_splits
    )
    label_dirs = split_annotation_dirs(
        root,
        config,
        image_dirs,
        annotation="labels",
        include_missing=True,
        selected_splits=requested_splits,
    )
    splits: dict[str, YoloSplitPaths] = {}
    for split in requested_splits:
        files = tuple(configured_images.get(split, ()))
        image_dir = image_dirs.get(split)
        if image_dir is None and files:
            image_dir = _common_parent(files)
        if image_dir is None:
            continue
        label_dir = label_dirs.get(
            split, label_dir_from_image_dir(image_dir, root, split)
        ).resolve()
        inferred_labels = {
            _label_path_for_image(path, root, split, image_dir, label_dir)
            for path in files
        }
        discovered_labels = set(common._iter_files(label_dir, {".txt"}))
        splits[split] = YoloSplitPaths(
            images=image_dir,
            labels=label_dir,
            image_files=files,
            label_files=tuple(sorted(inferred_labels | discovered_labels)),
        )
    if not splits:
        diagnostics = "\n  ".join(dataset_path_diagnostics(config_path, config))
        raise ValueError(
            "data.yaml 中没有可用的 train/val/test 图片目录: "
            f"{config_path}\n  {diagnostics}"
        )

    kind = forced_kind
    if kind is None:
        task = str(config.get("task", "")).casefold()
        if task in {"segment", "seg", "instance_segmentation"}:
            kind = "yolo_instance"
        elif task in {"detect", "detection", "object_detection"}:
            kind = "yolo_detection"
        else:
            candidate = inspect_dataset(config_path, count_limit=1)
            if candidate.kind in YOLO_KINDS:
                kind = candidate.kind
    if kind not in YOLO_KINDS:
        raise ValueError(
            "无法确认 YOLO 标注类型；请使用 --format yolo-detect 或 --format yolo-seg"
        )
    return YoloDataset(
        config_path=config_path,
        root=root,
        config=config,
        class_names=names,
        raw_class_names=raw_names,
        splits=splits,
        kind=kind,
    )


def _common_parent(paths: tuple[Path, ...]) -> Path:
    if len(paths) == 1:
        return paths[0].parent.resolve()
    try:
        return Path(__import__("os").path.commonpath([str(path) for path in paths])).resolve()
    except ValueError:
        return paths[0].parent.resolve()


def _label_path_for_image(
    image: Path,
    root: Path,
    split: str,
    image_dir: Path,
    label_dir: Path,
) -> Path:
    parts = list(image.parts)
    positions = [index for index, part in enumerate(parts) if part.casefold() == "images"]
    if positions:
        parts[positions[-1]] = "labels"
        return Path(*parts).with_suffix(".txt").resolve()
    try:
        relative = image.relative_to(image_dir)
    except ValueError:
        relative = Path(image.name)
    return (label_dir / relative).with_suffix(".txt").resolve()


def split_image_paths(paths: YoloSplitPaths) -> list[Path]:
    return list(paths.image_files) if paths.image_files else common._iter_files(
        paths.images, IMAGE_EXTENSIONS
    )


def split_label_paths(paths: YoloSplitPaths) -> list[Path]:
    if paths.label_files:
        return [path for path in paths.label_files if path.is_file()]
    return common._iter_files(paths.labels, {".txt"})


def relative_sample_path(path: Path, primary_root: Path) -> Path:
    try:
        return path.relative_to(primary_root)
    except ValueError:
        # Preserve a small amount of parent context for manifest entries while
        # keeping the generated path safely relative.
        return Path(path.parent.name) / path.name


def logical_sample_stem(path: Path, primary_root: Path) -> str:
    """Return the output pairing key for directory and manifest layouts."""
    parts = list(path.parts)
    positions = [
        index
        for index, part in enumerate(parts)
        if part.casefold() in {"images", "labels"}
    ]
    if positions:
        tail = parts[positions[-1] + 1 :]
        if tail and tail[0].casefold() in SPLITS:
            tail = tail[1:]
        if tail:
            return str(Path(*tail).with_suffix("")).replace("\\", "/")
    return str(relative_sample_path(path, primary_root).with_suffix("")).replace(
        "\\", "/"
    )


def parse_yolo_row(line: str, kind: str) -> tuple[int, list[str]]:
    tokens = line.split()
    expected = "class x y w h" if kind == "yolo_detection" else "class x1 y1 x2 y2 x3 y3 ..."
    if kind == "yolo_detection":
        valid_length = len(tokens) == 5
    else:
        valid_length = len(tokens) >= 7 and len(tokens) % 2 == 1
    if not valid_length:
        raise ValueError(f"YOLO 行格式应为 {expected}，当前字段数={len(tokens)}")
    try:
        class_id = int(tokens[0])
    except ValueError as exc:
        raise ValueError(f"类别 ID 需要是整数，当前为 {tokens[0]!r}") from exc
    if class_id < 0:
        raise ValueError(f"类别 ID 需要为非负整数，当前为 {class_id}")
    try:
        coordinates = [float(value) for value in tokens[1:]]
    except ValueError as exc:
        raise ValueError("坐标字段需要是数字") from exc
    if any(not math.isfinite(value) for value in coordinates):
        raise ValueError("坐标字段需要是有限数字")
    outside = [value for value in coordinates if value < 0.0 or value > 1.0]
    if outside:
        raise ValueError(f"归一化坐标需要位于 0..1，检测到 {outside[0]}")
    if kind == "yolo_detection":
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError("检测框宽高需要大于 0")
    else:
        points = list(zip(coordinates[0::2], coordinates[1::2]))
        if len(set(points)) < 3:
            raise ValueError("polygon 需要至少三个不同顶点")
        area2 = abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
            )
        )
        if area2 <= 1e-12:
            raise ValueError("polygon 面积需要大于 0")
    return class_id, tokens


def analyze_dataset(dataset: YoloDataset) -> YoloAnalysis:
    stats = {class_id: common.ClassStats() for class_id in dataset.class_names}
    unknown_stats: dict[int, common.ClassStats] = defaultdict(common.ClassStats)
    observed_ids: set[int] = set()
    issues: list[common.AnalysisIssue] = []
    label_file_count = 0
    image_count = 0
    annotation_count = 0
    malformed_count = 0

    for split, paths in dataset.splits.items():
        images = split_image_paths(paths)
        labels = split_label_paths(paths)
        image_count += len(images)
        label_file_count += len(labels)
        if not paths.labels.is_dir():
            issues.append(
                common.AnalysisIssue(
                    "missing_label_dir", "labels 目录不存在", split, str(paths.labels)
                )
            )

        image_index: dict[str, list[Path]] = defaultdict(list)
        label_index: dict[str, list[Path]] = defaultdict(list)
        for image in images:
            image_index[logical_sample_stem(image, paths.images)].append(image)
            try:
                with Image.open(image) as opened:
                    opened.verify()
                    if opened.width <= 0 or opened.height <= 0:
                        raise ValueError("图片宽高需要大于 0")
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                issues.append(
                    common.AnalysisIssue(
                        "invalid_image", f"图片无法读取: {exc}", split, str(image)
                    )
                )
        for label in labels:
            label_index[logical_sample_stem(label, paths.labels)].append(label)
        for stem, values in sorted(image_index.items()):
            if len(values) > 1:
                issues.append(
                    common.AnalysisIssue(
                        "duplicate_image_stem",
                        "多个图片映射到相同相对 stem",
                        split,
                        stem,
                        {"paths": [str(path) for path in values]},
                    )
                )
        for stem, values in sorted(label_index.items()):
            if len(values) > 1:
                issues.append(
                    common.AnalysisIssue(
                        "duplicate_label_stem",
                        "多个标注映射到相同相对 stem",
                        split,
                        stem,
                        {"paths": [str(path) for path in values]},
                    )
                )
        image_stems = set(image_index)
        label_stems = set(label_index)
        for stem in sorted(label_stems - image_stems):
            issues.append(
                common.AnalysisIssue(
                    "missing_image", "label 缺少同路径 stem 的图片", split, stem
                )
            )

        for label_path in tqdm(
            labels,
            desc=f"分析 {dataset.root.name}/{split}",
            unit="label",
            leave=False,
        ):
            ids_in_file: set[int] = set()
            try:
                lines = read_text_auto(label_path).splitlines()
            except (OSError, ValueError) as exc:
                malformed_count += 1
                issues.append(
                    common.AnalysisIssue(
                        "invalid_label",
                        f"标注文件编码无法读取: {exc}",
                        split,
                        str(label_path),
                    )
                )
                continue
            for line_number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    class_id, _ = parse_yolo_row(line, dataset.kind)
                except ValueError as exc:
                    malformed_count += 1
                    issues.append(
                        common.AnalysisIssue(
                            "invalid_label",
                            str(exc),
                            split,
                            str(label_path),
                            {"line": line_number, "content": line},
                        )
                    )
                    continue
                annotation_count += 1
                observed_ids.add(class_id)
                ids_in_file.add(class_id)
                target_stats = stats.get(class_id)
                if target_stats is None:
                    target_stats = unknown_stats[class_id]
                target_stats.pixel_count += 1
            for class_id in ids_in_file:
                target_stats = stats.get(class_id)
                if target_stats is None:
                    target_stats = unknown_stats[class_id]
                target_stats.image_count += 1

    class_ids = set(dataset.class_names)
    unknown_ids = observed_ids - class_ids
    unused_ids = class_ids - observed_ids
    null_like_ids = {
        class_id
        for class_id, name in dataset.class_names.items()
        if common.is_null_like_name(name, dataset.raw_class_names.get(class_id))
    }
    if unknown_ids:
        issues.append(
            common.AnalysisIssue(
                "unknown_label_ids",
                f"标注中存在 data.yaml 未定义的 ID: {sorted(unknown_ids)}",
                details={"ids": sorted(unknown_ids)},
            )
        )
    if unused_ids:
        issues.append(
            common.AnalysisIssue(
                "unused_yaml_ids",
                f"data.yaml 中存在标注未使用的 ID: {sorted(unused_ids)}",
                details={"ids": sorted(unused_ids)},
            )
        )
    if null_like_ids:
        issues.append(
            common.AnalysisIssue(
                "null_like_names",
                f"检测到空/占位类别名称: {sorted(null_like_ids)}",
                details={"ids": sorted(null_like_ids)},
            )
        )
    sorted_ids = sorted(class_ids)
    if sorted_ids != list(range(len(sorted_ids))):
        issues.append(
            common.AnalysisIssue(
                "non_contiguous_ids",
                f"类别 ID 不连续: {sorted_ids}",
                details={"ids": sorted_ids},
            )
        )
    exact_groups, similar_groups = common._name_candidate_groups(dataset.class_names)
    if exact_groups:
        issues.append(
            common.AnalysisIssue(
                "duplicate_names",
                f"检测到重复类别名候选: {exact_groups}",
                details={"groups": exact_groups},
            )
        )
    if similar_groups:
        issues.append(
            common.AnalysisIssue(
                "similar_names",
                f"检测到近似/带来源前缀的类别名候选: {similar_groups}",
                details={"groups": similar_groups},
            )
        )

    shared = common.DatasetAnalysis(
        class_stats=stats,
        unknown_stats=dict(sorted(unknown_stats.items())),
        observed_ids=observed_ids,
        unknown_ids=unknown_ids,
        unused_ids=unused_ids,
        null_like_ids=null_like_ids,
        exact_name_groups=exact_groups,
        similar_name_groups=similar_groups,
        issues=issues,
        mask_count=label_file_count,
        image_count=image_count,
    )
    return YoloAnalysis(
        shared=shared,
        annotation_count=annotation_count,
        malformed_count=malformed_count,
    )


def print_analysis(dataset: YoloDataset, analysis: YoloAnalysis) -> None:
    shared = analysis.shared
    kind_name = "YOLO 实例分割" if dataset.kind == "yolo_instance" else "YOLO 目标检测"
    print(f"\n数据集: {dataset.root}")
    print(
        f"格式 {kind_name}，图片 {shared.image_count}，label {shared.mask_count}，"
        f"标注对象 {analysis.annotation_count}，YAML 类别 {len(dataset.class_names)}"
    )
    print(f"\n  {'ID':>5}  {'类别名':<32}{'文件数':>10}{'对象数':>14}  状态")
    print("  " + "-" * 78)
    for class_id, name in dataset.class_names.items():
        stats = shared.class_stats[class_id]
        flags: list[str] = []
        if class_id in shared.null_like_ids:
            flags.append("空/占位名")
        if class_id in shared.unused_ids:
            flags.append("标注未使用")
        print(
            f"  {class_id:>5}  {(name or '<empty>'):<32}"
            f"{stats.image_count:>10}{stats.pixel_count:>14}  {','.join(flags)}"
        )
    if shared.unknown_ids:
        print("\n  标注中 YAML 未定义的 ID:")
        for class_id in sorted(shared.unknown_ids):
            stats = shared.unknown_stats[class_id]
            print(
                f"    ID {class_id}: 文件 {stats.image_count}，对象 {stats.pixel_count}"
            )
    if shared.exact_name_groups:
        print(f"  重复名称候选: {shared.exact_name_groups}")
    if shared.similar_name_groups:
        print(f"  近似/来源前缀名称候选: {shared.similar_name_groups}")
    issue_counts = Counter(issue.issue_type for issue in shared.issues)
    if issue_counts:
        print(f"  问题统计: {dict(sorted(issue_counts.items()))}")


def remap_label_text(
    text: str,
    kind: str,
    resolved: common.ResolvedPlan,
    *,
    unknown_policy: str,
) -> tuple[str, int, int]:
    output_lines: list[str] = []
    kept = 0
    removed = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            class_id, tokens = parse_yolo_row(line, kind)
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行: {exc}") from exc
        target_id = resolved.source_to_target.get(class_id)
        if target_id is None:
            if class_id in resolved.dropped_ids or unknown_policy == "ignore":
                removed += 1
                continue
            raise ValueError(f"第 {line_number} 行包含映射计划未定义的 ID: {class_id}")
        tokens[0] = str(target_id)
        output_lines.append(" ".join(tokens))
        kept += 1
    output = "\n".join(output_lines)
    return (output + "\n" if output_lines else ""), kept, removed


def _analysis_json(analysis: YoloAnalysis) -> dict[str, Any]:
    shared = analysis.shared
    return {
        "class_stats": {
            str(class_id): {
                "file_count": stats.image_count,
                "annotation_count": stats.pixel_count,
            }
            for class_id, stats in sorted(shared.class_stats.items())
        },
        "unknown_stats": {
            str(class_id): {
                "file_count": stats.image_count,
                "annotation_count": stats.pixel_count,
            }
            for class_id, stats in sorted(shared.unknown_stats.items())
        },
        "observed_ids": sorted(shared.observed_ids),
        "unknown_ids": sorted(shared.unknown_ids),
        "unused_ids": sorted(shared.unused_ids),
        "null_like_ids": sorted(shared.null_like_ids),
        "exact_name_groups": shared.exact_name_groups,
        "similar_name_groups": shared.similar_name_groups,
        "issues": [asdict(issue) for issue in shared.issues],
        "label_file_count": shared.mask_count,
        "image_count": shared.image_count,
        "annotation_count": analysis.annotation_count,
        "malformed_count": analysis.malformed_count,
    }


def edit_dataset(
    dataset: YoloDataset,
    output: Path,
    plan: common.EditPlan,
    *,
    analysis: YoloAnalysis | None = None,
    image_mode: str = "copy",
    clean: bool = False,
    dry_run: bool = False,
    require_train_val: bool = False,
) -> dict[str, Any]:
    analysis = analysis or analyze_dataset(dataset)
    if analysis.malformed_count:
        first = next(
            issue for issue in analysis.shared.issues if issue.issue_type == "invalid_label"
        )
        raise ValueError(
            f"检测到 {analysis.malformed_count} 行无效 YOLO 标注，首个问题: "
            f"{first.path}: {first.message}"
        )
    if analysis.shared.unknown_ids and plan.unknown_policy == "error":
        raise ValueError(
            f"标注存在 data.yaml 未定义的 ID: {sorted(analysis.shared.unknown_ids)}；"
            "请选择 unknown_policy=ignore 删除对应标注行，或补充类别定义"
        )
    resolved = common.resolve_plan(plan, dataset.class_names)
    class_ids = sorted(resolved.output_names)
    if not class_ids:
        raise ValueError("最终类别为空，YOLO 数据集无法用于训练")
    if class_ids != list(range(len(class_ids))):
        raise ValueError(
            "YOLO 输出类别 ID 需要连续为 0..N-1；"
            f"当前为 {class_ids}"
        )
    fatal_types = {
        "invalid_image",
        "duplicate_image_stem",
        "duplicate_label_stem",
        "missing_image",
    }
    fatal_issues = [
        issue for issue in analysis.shared.issues if issue.issue_type in fatal_types
    ]
    if fatal_issues:
        first = fatal_issues[0]
        raise ValueError(
            f"检测到 {len(fatal_issues)} 个输入完整性问题，首个问题: "
            f"{first.path}: {first.message}"
        )
    output = validate_output_location(output, [dataset.root])

    report: dict[str, Any] = {
        "source": str(dataset.root),
        "source_config": str(dataset.config_path),
        "output": str(output),
        "format": dataset.kind,
        "image_mode": image_mode,
        "dry_run": dry_run,
        "source_names": dataset.class_names,
        "output_names": resolved.output_names,
        "source_to_target": {
            class_id: resolved.source_to_target.get(class_id)
            for class_id in dataset.class_names
        },
        "drop_ids": sorted(plan.drop_ids),
        "merges": [asdict(merge) for merge in plan.merges],
        "renames": dict(sorted(plan.renames.items())),
        "unknown_policy": plan.unknown_policy,
        "unknown_policy_effect": "删除对应标注行",
        "require_train_val": require_train_val,
        "warnings": [],
        "analysis": _analysis_json(analysis),
        "splits": {},
    }
    if "train" not in dataset.splits:
        raise ValueError("YOLO 输出缺少 train split，无法用于训练")
    if "val" not in dataset.splits:
        message = "YOLO 输出缺少 val split；直接训练前需要提供验证集"
        if require_train_val:
            raise ValueError(message)
        report["warnings"].append(message)
    if dry_run:
        return report

    total_kept = 0
    total_removed = 0
    with staged_output(output, clean=clean) as stage:
        for split, paths in dataset.splits.items():
            split_report = {
                "images": 0,
                "labels": 0,
                "kept_annotations": 0,
                "removed_annotations": 0,
            }
            image_index = {
                logical_sample_stem(path, paths.images): path
                for path in split_image_paths(paths)
            }
            label_index = {
                logical_sample_stem(path, paths.labels): path
                for path in split_label_paths(paths)
            }
            with tqdm(
                total=len(image_index),
                desc=f"处理 {dataset.root.name}/{split}",
                unit="样本",
            ) as progress:
                for stem, image_path in sorted(image_index.items()):
                    image_target = (
                        stage
                        / "images"
                        / split
                        / Path(stem + image_path.suffix.lower())
                    )
                    common._transfer_file(image_path, image_target, image_mode)
                    split_report["images"] += 1
                    label_path = label_index.get(stem)
                    if label_path is not None:
                        rewritten, kept, removed = remap_label_text(
                            read_text_auto(label_path),
                            dataset.kind,
                            resolved,
                            unknown_policy=plan.unknown_policy,
                        )
                        target = stage / "labels" / split / Path(stem + ".txt")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(rewritten, encoding="utf-8")
                        split_report["labels"] += 1
                        split_report["kept_annotations"] += kept
                        split_report["removed_annotations"] += removed
                    progress.update()
            total_kept += split_report["kept_annotations"]
            total_removed += split_report["removed_annotations"]
            report["splits"][split] = split_report
        report["kept_annotations"] = total_kept
        report["removed_annotations"] = total_removed

        if report["splits"]["train"]["images"] == 0:
            raise ValueError("train split 中没有可用图片")
        if require_train_val and report["splits"].get("val", {}).get("images", 0) == 0:
            raise ValueError("val split 中没有可用图片")

        data: dict[str, Any] = {
            "path": str(output),
            "task": "segment" if dataset.kind == "yolo_instance" else "detect",
            "nc": len(resolved.output_names),
            "names": resolved.output_names,
        }
        for split in dataset.splits:
            data[split] = f"images/{split}"
        (stage / "data.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (stage / "classes.txt").write_text(
            "\n".join(
                resolved.output_names[class_id]
                for class_id in sorted(resolved.output_names)
            )
            + "\n",
            encoding="utf-8",
        )
        mapping = {key: value for key, value in report.items() if key != "analysis"}
        (stage / "class_edit_mapping.yaml").write_text(
            yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (stage / "class_analysis.json").write_text(
            json.dumps(report["analysis"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stage / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return report


def _choose_source(search_root: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in detect_datasets(search_root, kinds=YOLO_KINDS)
    ]
    print(f"\n在 {search_root} 中检测到 {len(candidates)} 个 YOLO Det/Seg 数据集:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. {candidate.kind:<18} 类别 {candidate.class_count:>4}，"
            f"label {candidate.annotation_count:>7}  {candidate.path}"
        )
    print("   p. 输入任意 data.yaml 或数据集路径")
    while True:
        try:
            raw = input("选择数据集 [序号/p/q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        if raw.casefold() == "p":
            value = input("data.yaml 或数据集目录: ").strip()
            return Path(value).expanduser().resolve() if value else None
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            candidate = candidates[int(raw) - 1]
            return candidate.config_path or candidate.path
        print("请输入有效序号、p 或 q。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="源 data.yaml 或 YOLO 数据集目录")
    parser.add_argument("--out", type=Path, help="输出目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描目录，默认仓库 datasets/")
    parser.add_argument("--plan", type=Path, help="读取已有 class_edit_mapping/计划 YAML")
    parser.add_argument(
        "--id-policy",
        choices=("compact", "preserve", "explicit"),
        help="覆盖计划中的类别 ID 策略；explicit 的映射取自计划 explicit_ids",
    )
    parser.add_argument(
        "--format",
        choices=("yolo-detect", "yolo-seg"),
        help="标签为空时可显式指定 YOLO 类型",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink", "reflink"),
        default="copy",
    )
    parser.add_argument("--clean", action="store_true", help="清理已有输出后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只分析和生成操作计划")
    parser.add_argument("--yes", action="store_true", help="跳过最终交互确认")
    parser.add_argument("--require-train-val", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.src
    if source is None:
        source = _choose_source((args.datasets or auto_datasets_root()).resolve())
        if source is None:
            return 0
    forced_kind = {
        "yolo-detect": "yolo_detection",
        "yolo-seg": "yolo_instance",
        None: None,
    }[args.format]
    dataset = resolve_dataset(source, forced_kind=forced_kind)
    print("\n正在扫描全部 YOLO 标注...")
    analysis = analyze_dataset(dataset)
    print_analysis(dataset, analysis)
    plan = (
        common.plan_from_yaml(args.plan)
        if args.plan
        else common.interactive_plan(
            dataset,  # type: ignore[arg-type]
            analysis.shared,
            annotation_label="标注",
            drop_effect="删除对应标注行",
            unknown_effect="删除对应标注行",
        )
    )
    if plan is None:
        return 0
    if args.id_policy is not None:
        plan.id_policy = args.id_policy
    resolved = common.resolve_plan(plan, dataset.class_names)
    print("\n最终类别:")
    for class_id, name in resolved.output_names.items():
        print(f"  {class_id}: {name}")
    print("删除类别及未知 ID 对应的标注行会从输出中移除。")

    output = args.out or default_output_dir(dataset.root, "edit-classes")
    if args.out is None:
        raw = input(f"输出目录 [{output}]: ").strip()
        if raw:
            output = Path(raw)
    if not args.dry_run and not args.yes:
        answer = input("确认重写 labels 并生成新数据集？[y/N]: ").strip().casefold()
        if answer not in {"y", "yes", "是"}:
            print("已取消。")
            return 0
    report = edit_dataset(
        dataset,
        output,
        plan,
        analysis=analysis,
        image_mode=args.image_mode,
        clean=args.clean,
        dry_run=args.dry_run,
        require_train_val=args.require_train_val,
    )
    if args.dry_run:
        print("\ndry-run 完成，源数据保持原样。")
        print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    else:
        print(f"\n处理完成: {report['output']}")
        print(f"保留标注: {report['kept_annotations']}")
        print(f"删除标注: {report['removed_annotations']}")
        print(f"类别映射: {Path(report['output']) / 'class_edit_mapping.yaml'}")
        print(f"问题报告: {Path(report['output']) / 'class_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
