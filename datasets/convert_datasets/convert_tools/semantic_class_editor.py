#!/usr/bin/env python3
"""交互式检查并编辑现有 PNG 语义分割数据集的类别。

支持：
  - 识别空名称、None/null、重复/近似名称、稀疏 ID、未使用类别和未知 mask ID；
  - 删除类别并将其像素写为 ignore label；
  - 合并任意多个类别，可同时指定新的类别名；
  - 重命名类别；
  - 将保留类别重新编号为连续的 0..N-1；
  - 同步重写 mask、data.yaml、classes.txt 和审计报告。

默认写入新的输出目录，源数据保持原样。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

try:
    from .dataset_discovery import (
        IMAGE_EXTENSIONS,
        SPLITS,
        auto_datasets_root,
        find_dataset_config,
        split_annotation_dirs,
        split_image_dirs,
    )
    from .dataset_detector import detect_datasets
    from .progress import tqdm
    from .text_encoding import read_text_auto
    from .output_naming import default_output_dir
    from .dataset_transaction import staged_output, validate_output_location
except ImportError:
    from dataset_discovery import (  # type: ignore[no-redef]
        IMAGE_EXTENSIONS,
        SPLITS,
        auto_datasets_root,
        find_dataset_config,
        split_annotation_dirs,
        split_image_dirs,
    )
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]
    from output_naming import default_output_dir  # type: ignore[no-redef]
    from dataset_transaction import staged_output, validate_output_location  # type: ignore[no-redef]


MASK_EXTENSIONS = {".png", ".tif", ".tiff", ".bmp"}
NULL_LIKE_NAMES = {
    "",
    "none",
    "null",
    "nil",
    "nan",
    "void",
    "ignore",
    "ignored",
    "unlabeled",
    "unlabelled",
    "未标注",
    "无标签",
}

MaskLabel = int | tuple[int, ...]


@dataclass(frozen=True)
class SplitPaths:
    images: Path
    masks: Path


@dataclass
class SemanticDataset:
    config_path: Path
    root: Path
    config: dict[str, Any]
    class_names: dict[int, str]
    raw_class_names: dict[int, Any]
    splits: dict[str, SplitPaths]
    ignore_label: int | None = None
    class_labels: dict[int, tuple[MaskLabel, ...]] = field(default_factory=dict)
    label_to_class: dict[MaskLabel, int] = field(default_factory=dict)
    label_kind: str = "integer"
    label_channels: int = 1


@dataclass
class ClassStats:
    image_count: int = 0
    pixel_count: int = 0


@dataclass
class AnalysisIssue:
    issue_type: str
    message: str
    split: str = ""
    path: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetAnalysis:
    class_stats: dict[int, ClassStats]
    unknown_stats: dict[MaskLabel, ClassStats]
    observed_ids: set[int]
    unknown_ids: set[MaskLabel]
    unused_ids: set[int]
    null_like_ids: set[int]
    exact_name_groups: list[list[int]]
    similar_name_groups: list[list[int]]
    issues: list[AnalysisIssue]
    mask_count: int
    image_count: int


@dataclass
class MergeSpec:
    ids: list[int]
    name: str


@dataclass
class EditPlan:
    drop_ids: set[int] = field(default_factory=set)
    merges: list[MergeSpec] = field(default_factory=list)
    renames: dict[int, str] = field(default_factory=dict)
    unknown_policy: str = "error"
    id_policy: str = "compact"
    explicit_ids: dict[int, int] = field(default_factory=dict)


@dataclass
class ResolvedPlan:
    source_to_target: dict[int, int]
    output_names: dict[int, str]
    dropped_ids: set[int]


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(read_text_auto(path)) or {}
    except (OSError, ValueError) as exc:
        raise ValueError(f"YAML 编码无法识别: {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 解析失败: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"YAML 顶层需要是 object/dict: {path}")
    return value


def normalize_class_value(value: Any) -> tuple[str, Any]:
    raw = value
    if isinstance(value, dict):
        value = value.get("name")
    if value is None:
        return "", raw
    return str(value).strip(), raw


def parse_class_names(config: dict[str, Any]) -> tuple[dict[int, str], dict[int, Any]]:
    raw_names = config.get("names", config.get("classes"))
    if isinstance(raw_names, list):
        items = enumerate(raw_names)
    elif isinstance(raw_names, dict):
        items = raw_names.items()
    else:
        raise ValueError("data.yaml 中缺少 names/classes")

    names: dict[int, str] = {}
    raw_values: dict[int, Any] = {}
    for raw_id, value in items:
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"类别 ID 无法转换为整数: {raw_id!r}") from exc
        if class_id < 0:
            raise ValueError(f"类别 ID 需要为非负整数: {class_id}")
        name, raw_value = normalize_class_value(value)
        names[class_id] = name
        raw_values[class_id] = raw_value
    if not names:
        raise ValueError("data.yaml 中的 names/classes 为空")
    return dict(sorted(names.items())), dict(sorted(raw_values.items()))


def _label_sort_key(label: MaskLabel) -> tuple[int, tuple[int, ...]]:
    if isinstance(label, tuple):
        return 1, label
    return 0, (label,)


def _sorted_labels(labels: Iterable[MaskLabel]) -> list[MaskLabel]:
    return sorted(labels, key=_label_sort_key)


def _normalize_mask_label(raw_label: Any, *, class_id: int) -> MaskLabel:
    if isinstance(raw_label, bool):
        raise ValueError(f"类别 {class_id} 的 label 需要是整数或 RGB/RGBA 元组")
    if isinstance(raw_label, Integral):
        label = int(raw_label)
        if not 0 <= label <= 65535:
            raise ValueError(
                f"类别 {class_id} 的整数 label 需要在 0..65535 范围内: {label}"
            )
        return label
    if isinstance(raw_label, (list, tuple)):
        if len(raw_label) not in {3, 4}:
            raise ValueError(
                f"类别 {class_id} 的颜色 label 需要是 RGB/RGBA，"
                f"当前通道数={len(raw_label)}"
            )
        components: list[int] = []
        for component in raw_label:
            if isinstance(component, bool) or not isinstance(component, Integral):
                raise ValueError(
                    f"类别 {class_id} 的颜色 label 分量需要是整数: {raw_label!r}"
                )
            value = int(component)
            if not 0 <= value <= 255:
                raise ValueError(
                    f"类别 {class_id} 的颜色 label 分量需要在 0..255 范围内: "
                    f"{raw_label!r}"
                )
            components.append(value)
        return tuple(components)
    raise ValueError(
        f"类别 {class_id} 的 label 需要是整数或 RGB/RGBA 元组: {raw_label!r}"
    )


def parse_semantic_classes(
    config: dict[str, Any],
) -> tuple[
    dict[int, str],
    dict[int, Any],
    dict[int, tuple[MaskLabel, ...]],
    dict[MaskLabel, int],
    str,
    int,
]:
    """Parse Lightly semantic class names and their raw mask labels.

    String/null entries retain the historical ``class id == pixel value``
    convention. Object entries accept ``labels`` and its legacy alias
    ``values``. A dataset has one raw-label representation: integers, RGB, or
    RGBA.
    """
    names, raw_values = parse_class_names(config)
    class_labels: dict[int, tuple[MaskLabel, ...]] = {}
    label_to_class: dict[MaskLabel, int] = {}
    label_kinds: set[str] = set()
    color_channels: set[int] = set()

    for class_id, raw_value in raw_values.items():
        if isinstance(raw_value, dict):
            has_labels = "labels" in raw_value
            has_values = "values" in raw_value
            if has_labels and has_values:
                raise ValueError(
                    f"类别 {class_id} 同时配置 labels 和 values，"
                    "请保留一个字段以消除歧义"
                )
            if not has_labels and not has_values:
                raise ValueError(
                    f"类别 {class_id} 使用 object 配置时需要 labels 或 values"
                )
            raw_labels = raw_value["labels" if has_labels else "values"]
            if not isinstance(raw_labels, (list, tuple, set, frozenset)):
                raise ValueError(f"类别 {class_id} 的 labels/values 需要是列表或集合")
            if not raw_labels:
                raise ValueError(f"类别 {class_id} 的 labels/values 不能为空")
            labels = {
                _normalize_mask_label(raw_label, class_id=class_id)
                for raw_label in raw_labels
            }
        else:
            labels = {class_id}

        kinds = {"color" if isinstance(label, tuple) else "integer" for label in labels}
        if len(kinds) != 1:
            raise ValueError(
                f"类别 {class_id} 混合了整数与颜色 label；"
                "整个数据集需要使用一致的 label 类型"
            )
        kind = kinds.pop()
        label_kinds.add(kind)
        if kind == "color":
            color_channels.update(len(label) for label in labels if isinstance(label, tuple))
        ordered = tuple(_sorted_labels(labels))
        class_labels[class_id] = ordered
        for label in ordered:
            previous = label_to_class.get(label)
            if previous is not None and previous != class_id:
                raise ValueError(
                    f"原始 mask label {label!r} 同时映射到类别 "
                    f"{previous} 和 {class_id}"
                )
            label_to_class[label] = class_id

    if len(label_kinds) != 1:
        raise ValueError(
            "classes 混合了整数与颜色 label；整个数据集需要使用一致的 label 类型"
        )
    label_kind = label_kinds.pop()
    if label_kind == "color":
        if len(color_channels) != 1:
            raise ValueError(
                "classes 混合了 RGB 与 RGBA label；所有颜色 label 通道数需要一致"
            )
        label_channels = color_channels.pop()
        label_kind = "rgb" if label_channels == 3 else "rgba"
    else:
        label_channels = 1
    return (
        names,
        raw_values,
        class_labels,
        label_to_class,
        label_kind,
        label_channels,
    )


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _find_config(source: Path) -> Path:
    return find_dataset_config(source)


def resolve_dataset(source: Path) -> SemanticDataset:
    config_path = _find_config(source)
    config = read_yaml(config_path)
    configured_root = config.get("path")
    root = (
        _resolve_path(config_path.parent, configured_root)
        if configured_root is not None
        else config_path.parent.resolve()
    )
    (
        names,
        raw_names,
        class_labels,
        label_to_class,
        label_kind,
        label_channels,
    ) = parse_semantic_classes(config)
    image_dirs = split_image_dirs(root, config)
    mask_dirs = split_annotation_dirs(
        root,
        config,
        image_dirs,
        annotation="masks",
        include_missing=True,
    )
    splits: dict[str, SplitPaths] = {}
    for split in SPLITS:
        image_dir = image_dirs.get(split)
        mask_dir = mask_dirs.get(split)
        if image_dir is None:
            image_dir = (root / "images" / split).resolve()
        if mask_dir is None:
            mask_dir = (root / "masks" / split).resolve()
        if image_dir.is_dir() or mask_dir.is_dir():
            splits[split] = SplitPaths(images=image_dir, masks=mask_dir)
    if not splits:
        raise ValueError(f"没有找到 images/masks split: {root}")
    raw_ignore = config.get("ignore_label", config.get("ignore_index"))
    ignore_label = int(raw_ignore) if raw_ignore is not None else None
    if ignore_label is not None and not 0 <= ignore_label <= 65535:
        raise ValueError(f"ignore_label 需要在 0..65535 范围内: {ignore_label}")
    return SemanticDataset(
        config_path=config_path,
        root=root,
        config=config,
        class_names=names,
        raw_class_names=raw_names,
        splits=splits,
        ignore_label=ignore_label,
        class_labels=class_labels,
        label_to_class=label_to_class,
        label_kind=label_kind,
        label_channels=label_channels,
    )


def source_ignore_label_for_edit(
    dataset: SemanticDataset, output_ignore_label: int
) -> int | None:
    """Resolve input ignore semantics while preserving a declared real class."""
    if dataset.ignore_label is not None:
        if dataset.ignore_label in dataset.label_to_class:
            raise ValueError(
                f"{dataset.root} 同时把 {dataset.ignore_label} 声明为 ignore_label "
                "和原始 mask label"
            )
        return dataset.ignore_label
    if output_ignore_label in dataset.label_to_class:
        return None
    return output_ignore_label


def _iter_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def read_mask(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            mode = image.mode
            array = np.asarray(image)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"mask 无法读取: {exc}") from exc
    if array.ndim not in {2, 3}:
        raise ValueError(f"mask 需要是单通道、RGB 或 RGBA 类别图，当前 shape={array.shape}")
    if array.ndim == 3 and array.shape[2] not in {3, 4}:
        raise ValueError(f"彩色 mask 需要是 RGB 或 RGBA，当前 shape={array.shape}")
    if array.ndim == 3 and mode not in {"RGB", "RGBA"}:
        raise ValueError(f"彩色 mask 需要使用 RGB/RGBA mode，当前 mode={mode}")
    if not np.issubdtype(array.dtype, np.integer):
        if np.issubdtype(array.dtype, np.bool_):
            return array.astype(np.uint8)
        raise ValueError(f"mask dtype 需要是整数，当前 dtype={array.dtype}")
    return array


def _validate_mask_representation(mask: np.ndarray, dataset: SemanticDataset) -> None:
    if dataset.label_kind == "integer":
        if mask.ndim != 2:
            raise ValueError(
                "classes 使用整数 label，mask 需要是单通道类别图，"
                f"当前 shape={mask.shape}"
            )
        return
    if mask.ndim != 3 or mask.shape[2] != dataset.label_channels:
        expected = dataset.label_kind.upper()
        raise ValueError(
            f"classes 使用 {expected} label，mask 需要有 {dataset.label_channels} 个通道，"
            f"当前 shape={mask.shape}"
        )
    if mask.size and (int(mask.min()) < 0 or int(mask.max()) > 255):
        raise ValueError("RGB/RGBA mask 的通道值需要在 0..255 范围内")


def _mask_label_counts(
    mask: np.ndarray, dataset: SemanticDataset
) -> list[tuple[MaskLabel, int]]:
    _validate_mask_representation(mask, dataset)
    if dataset.label_kind == "integer":
        values, counts = np.unique(mask, return_counts=True)
        return [
            (int(raw_value), int(raw_count))
            for raw_value, raw_count in zip(values, counts)
        ]
    pixels = mask.reshape(-1, dataset.label_channels)
    values, counts = np.unique(pixels, axis=0, return_counts=True)
    return [
        (tuple(int(component) for component in raw_value), int(raw_count))
        for raw_value, raw_count in zip(values, counts)
    ]


def _pack_color_mask(mask: np.ndarray) -> np.ndarray:
    packed = np.zeros(mask.shape[:2], dtype=np.uint32)
    for channel in range(mask.shape[2]):
        packed = (packed << np.uint32(8)) | mask[..., channel].astype(
            np.uint32, copy=False
        )
    return packed


def _pack_color_label(label: tuple[int, ...]) -> int:
    packed = 0
    for component in label:
        packed = (packed << 8) | component
    return packed


def _unpack_color_label(value: int, channels: int) -> tuple[int, ...]:
    components = [0] * channels
    for index in range(channels - 1, -1, -1):
        components[index] = value & 0xFF
        value >>= 8
    return tuple(components)


def normalize_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold().strip()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def semantic_tail(name: str) -> str:
    normalized = unicodedata.normalize("NFKC", name).casefold().strip()
    normalized = re.sub(
        r"^(?:dataset|source|data|ds)\d+[_\-\s]+",
        "",
        normalized,
    )
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def is_numbered_placeholder_name(name: str) -> bool:
    normalized = semantic_tail(name)
    return bool(
        re.fullmatch(
            r"(?:class|category|类别|类别id)\d+",
            normalized,
        )
    )


def is_null_like_name(name: str, raw_value: Any) -> bool:
    if raw_value is None:
        return True
    return unicodedata.normalize("NFKC", name).casefold().strip() in NULL_LIKE_NAMES


def _name_candidate_groups(names: dict[int, str]) -> tuple[list[list[int]], list[list[int]]]:
    exact_by_key: dict[str, list[int]] = defaultdict(list)
    tail_by_key: dict[str, list[int]] = defaultdict(list)
    for class_id, name in names.items():
        if not name:
            continue
        exact_by_key[normalize_name(name)].append(class_id)
        if not is_numbered_placeholder_name(name):
            tail_by_key[semantic_tail(name)].append(class_id)
    exact = sorted(
        (sorted(ids) for key, ids in exact_by_key.items() if key and len(ids) > 1),
        key=lambda ids: ids[0],
    )
    exact_sets = {tuple(ids) for ids in exact}
    similar: set[tuple[int, ...]] = {
        tuple(sorted(ids))
        for key, ids in tail_by_key.items()
        if key and len(ids) > 1 and tuple(sorted(ids)) not in exact_sets
    }

    valid = [(class_id, semantic_tail(name)) for class_id, name in names.items() if name]
    for index, (left_id, left_name) in enumerate(valid):
        if is_numbered_placeholder_name(names[left_id]):
            continue
        if len(left_name) < 4:
            continue
        for right_id, right_name in valid[index + 1 :]:
            if is_numbered_placeholder_name(names[right_id]):
                continue
            if len(right_name) < 4:
                continue
            if SequenceMatcher(None, left_name, right_name).ratio() >= 0.9:
                pair = tuple(sorted((left_id, right_id)))
                if pair not in exact_sets:
                    similar.add(pair)
    return exact, [list(ids) for ids in sorted(similar)]


def _relative_stems(files: Iterable[Path], root: Path) -> set[str]:
    return {
        str(path.relative_to(root).with_suffix("")).replace("\\", "/")
        for path in files
    }


def analyze_dataset(
    dataset: SemanticDataset,
    *,
    ignore_label: int | None = 255,
) -> DatasetAnalysis:
    # A configured class has explicit semantics and takes priority over the
    # legacy convention that an undeclared output ignore value means ignore.
    if ignore_label in dataset.label_to_class:
        ignore_label = None
    stats = {class_id: ClassStats() for class_id in dataset.class_names}
    unknown_stats: dict[MaskLabel, ClassStats] = defaultdict(ClassStats)
    observed_ids: set[int] = set()
    issues: list[AnalysisIssue] = []
    mask_count = 0
    image_count = 0

    for split, paths in dataset.splits.items():
        images = _iter_files(paths.images, IMAGE_EXTENSIONS)
        masks = _iter_files(paths.masks, MASK_EXTENSIONS)
        image_count += len(images)
        mask_count += len(masks)
        if not paths.images.is_dir():
            issues.append(
                AnalysisIssue("missing_image_dir", "images 目录不存在", split, str(paths.images))
            )
        if not paths.masks.is_dir():
            issues.append(
                AnalysisIssue("missing_mask_dir", "masks 目录不存在", split, str(paths.masks))
            )
        image_index: dict[str, list[Path]] = defaultdict(list)
        mask_index: dict[str, list[Path]] = defaultdict(list)
        image_sizes: dict[str, tuple[int, int]] = {}
        for image_path in images:
            stem = str(image_path.relative_to(paths.images).with_suffix("")).replace("\\", "/")
            image_index[stem].append(image_path)
            try:
                with Image.open(image_path) as image:
                    image.load()
                    image_sizes[stem] = image.size
            except (OSError, UnidentifiedImageError) as exc:
                issues.append(
                    AnalysisIssue(
                        "invalid_image", f"图片无法读取: {exc}", split, str(image_path)
                    )
                )
        for mask_path in masks:
            stem = str(mask_path.relative_to(paths.masks).with_suffix("")).replace("\\", "/")
            mask_index[stem].append(mask_path)
        for stem, values in sorted(image_index.items()):
            if len(values) > 1:
                issues.append(
                    AnalysisIssue(
                        "duplicate_image_stem",
                        "多个图片映射到相同相对 stem",
                        split,
                        stem,
                        {"paths": [str(path) for path in values]},
                    )
                )
        for stem, values in sorted(mask_index.items()):
            if len(values) > 1:
                issues.append(
                    AnalysisIssue(
                        "duplicate_mask_stem",
                        "多个 mask 映射到相同 PNG 输出路径",
                        split,
                        stem,
                        {"paths": [str(path) for path in values]},
                    )
                )
        image_stems = set(image_index)
        mask_stems = set(mask_index)
        for stem in sorted(image_stems - mask_stems):
            issues.append(
                AnalysisIssue("missing_mask", "图片缺少同路径 stem 的 mask", split, stem)
            )
        for stem in sorted(mask_stems - image_stems):
            issues.append(
                AnalysisIssue("missing_image", "mask 缺少同路径 stem 的图片", split, stem)
            )

        for mask_path in tqdm(
            masks,
            desc=f"分析 {dataset.root.name}/{split}",
            unit="mask",
            leave=False,
        ):
            try:
                mask = read_mask(mask_path)
            except ValueError as exc:
                issues.append(
                    AnalysisIssue("invalid_mask", str(exc), split, str(mask_path))
                )
                continue
            stem = str(mask_path.relative_to(paths.masks).with_suffix("")).replace("\\", "/")
            expected_size = image_sizes.get(stem)
            if expected_size is not None and (mask.shape[1], mask.shape[0]) != expected_size:
                issues.append(
                    AnalysisIssue(
                        "size_mismatch",
                        f"图片尺寸 {expected_size} 与 mask 尺寸 "
                        f"{(mask.shape[1], mask.shape[0])} 不一致",
                        split,
                        str(mask_path),
                        {"image": str(image_index[stem][0])},
                    )
                )
            try:
                label_counts = _mask_label_counts(mask, dataset)
            except ValueError as exc:
                issues.append(
                    AnalysisIssue("invalid_mask", str(exc), split, str(mask_path))
                )
                continue
            image_class_pixels: dict[int, int] = defaultdict(int)
            for raw_label, count in label_counts:
                if ignore_label is not None and raw_label == ignore_label:
                    continue
                class_id = dataset.label_to_class.get(raw_label)
                if class_id is None:
                    unknown_stats[raw_label].image_count += 1
                    unknown_stats[raw_label].pixel_count += count
                    continue
                observed_ids.add(class_id)
                image_class_pixels[class_id] += count
            for class_id, count in image_class_pixels.items():
                stats[class_id].image_count += 1
                stats[class_id].pixel_count += count

    class_ids = set(dataset.class_names)
    unknown_ids = set(unknown_stats)
    unused_ids = class_ids - observed_ids
    null_like_ids = {
        class_id
        for class_id, name in dataset.class_names.items()
        if is_null_like_name(name, dataset.raw_class_names.get(class_id))
    }
    if unknown_ids:
        issues.append(
            AnalysisIssue(
                "unknown_mask_ids",
                f"mask 中存在 classes 未映射的 label: {_sorted_labels(unknown_ids)}",
                details={"labels": _sorted_labels(unknown_ids)},
            )
        )
    if unused_ids:
        issues.append(
            AnalysisIssue(
                "unused_yaml_ids",
                f"data.yaml 中存在 mask 未使用的 ID: {sorted(unused_ids)}",
                details={"ids": sorted(unused_ids)},
            )
        )
    if null_like_ids:
        issues.append(
            AnalysisIssue(
                "null_like_names",
                f"检测到空/占位类别名称: {sorted(null_like_ids)}",
                details={"ids": sorted(null_like_ids)},
            )
        )
    if ignore_label is not None and ignore_label in dataset.label_to_class:
        issues.append(
            AnalysisIssue(
                "ignore_label_defined_as_class",
                f"ignore_label={ignore_label} 同时出现在类别定义中",
                details={"id": ignore_label},
            )
        )
    sorted_ids = sorted(class_ids)
    if sorted_ids != list(range(len(sorted_ids))):
        issues.append(
            AnalysisIssue(
                "non_contiguous_ids",
                f"类别 ID 不连续: {sorted_ids}",
                details={"ids": sorted_ids},
            )
        )
    exact_groups, similar_groups = _name_candidate_groups(dataset.class_names)
    if exact_groups:
        issues.append(
            AnalysisIssue(
                "duplicate_names",
                f"检测到重复类别名候选: {exact_groups}",
                details={"groups": exact_groups},
            )
        )
    if similar_groups:
        issues.append(
            AnalysisIssue(
                "similar_names",
                f"检测到近似/带来源前缀的类别名候选: {similar_groups}",
                details={"groups": similar_groups},
            )
        )
    return DatasetAnalysis(
        class_stats=stats,
        unknown_stats={
            label: unknown_stats[label] for label in _sorted_labels(unknown_stats)
        },
        observed_ids=observed_ids,
        unknown_ids=unknown_ids,
        unused_ids=unused_ids,
        null_like_ids=null_like_ids,
        exact_name_groups=exact_groups,
        similar_name_groups=similar_groups,
        issues=issues,
        mask_count=mask_count,
        image_count=image_count,
    )


def _merge_members(plan: EditPlan) -> set[int]:
    return {class_id for merge in plan.merges for class_id in merge.ids}


def validate_plan(plan: EditPlan, class_names: dict[int, str]) -> None:
    valid_ids = set(class_names)
    unknown_drop = plan.drop_ids - valid_ids
    if unknown_drop:
        raise ValueError(f"删除计划包含未知类别 ID: {sorted(unknown_drop)}")
    seen: set[int] = set()
    for merge in plan.merges:
        ids = set(merge.ids)
        if len(ids) < 2:
            raise ValueError("每组合并至少需要两个类别 ID")
        if ids - valid_ids:
            raise ValueError(f"合并计划包含未知类别 ID: {sorted(ids - valid_ids)}")
        if ids & plan.drop_ids:
            raise ValueError(f"同一类别同时出现在删除和合并计划: {sorted(ids & plan.drop_ids)}")
        if ids & seen:
            raise ValueError(f"类别出现在多个合并组: {sorted(ids & seen)}")
        if not merge.name.strip():
            raise ValueError(f"合并类别 {sorted(ids)} 缺少输出名称")
        seen.update(ids)
    if set(plan.renames) - valid_ids:
        raise ValueError(
            f"重命名计划包含未知类别 ID: {sorted(set(plan.renames) - valid_ids)}"
        )
    if set(plan.renames) & (plan.drop_ids | seen):
        raise ValueError("删除/合并组中的类别请直接在对应操作中指定输出名称")
    if plan.unknown_policy not in {"error", "ignore"}:
        raise ValueError(f"未知 unknown_policy: {plan.unknown_policy}")
    if plan.id_policy not in {"compact", "preserve", "explicit"}:
        raise ValueError(f"未知 id_policy: {plan.id_policy}")
    invalid_explicit_sources = set(plan.explicit_ids) - valid_ids
    if invalid_explicit_sources:
        raise ValueError(
            f"explicit_ids 包含未知类别 ID: {sorted(invalid_explicit_sources)}"
        )
    if any(target < 0 for target in plan.explicit_ids.values()):
        raise ValueError("explicit_ids 的目标 ID 需要为非负整数")


def resolve_plan(plan: EditPlan, class_names: dict[int, str]) -> ResolvedPlan:
    validate_plan(plan, class_names)
    merge_by_id: dict[int, MergeSpec] = {}
    for merge in plan.merges:
        for class_id in merge.ids:
            merge_by_id[class_id] = merge

    groups: list[tuple[list[int], str]] = []
    handled: set[int] = set(plan.drop_ids)
    for class_id in sorted(class_names):
        if class_id in handled:
            continue
        merge = merge_by_id.get(class_id)
        if merge is not None:
            ids = sorted(merge.ids)
            groups.append((ids, merge.name.strip()))
            handled.update(ids)
            continue
        name = plan.renames.get(class_id, class_names[class_id]).strip()
        if not name:
            raise ValueError(f"保留类别 {class_id} 的名称为空，请删除或重命名")
        groups.append(([class_id], name))
        handled.add(class_id)

    source_to_target: dict[int, int] = {}
    output_names: dict[int, str] = {}
    used_targets: set[int] = set()
    for compact_id, (source_ids, name) in enumerate(groups):
        if plan.id_policy == "compact":
            target_id = compact_id
        elif plan.id_policy == "preserve":
            target_id = min(source_ids)
        else:
            explicit = {plan.explicit_ids[value] for value in source_ids if value in plan.explicit_ids}
            if not explicit:
                raise ValueError(f"explicit id_policy 缺少类别 {source_ids} 的目标 ID")
            if len(explicit) != 1:
                raise ValueError(f"合并组 {source_ids} 配置了多个目标 ID: {sorted(explicit)}")
            target_id = explicit.pop()
        if target_id in used_targets:
            raise ValueError(f"多个输出类别占用目标 ID {target_id}")
        used_targets.add(target_id)
        output_names[target_id] = name
        for source_id in source_ids:
            source_to_target[source_id] = target_id
    duplicate_output_names: dict[str, list[int]] = defaultdict(list)
    for class_id, name in output_names.items():
        duplicate_output_names[normalize_name(name)].append(class_id)
    duplicates = {
        name: ids
        for name, ids in duplicate_output_names.items()
        if name and len(ids) > 1
    }
    if duplicates:
        raise ValueError(f"输出仍有重复类别名，请继续合并或重命名: {duplicates}")
    return ResolvedPlan(
        source_to_target=source_to_target,
        output_names=output_names,
        dropped_ids=set(plan.drop_ids),
    )


def remap_mask(
    mask: np.ndarray,
    resolved: ResolvedPlan,
    *,
    ignore_label: int,
    unknown_policy: str,
    source_ignore_label: int | None = None,
    dataset: SemanticDataset | None = None,
) -> np.ndarray:
    if not 0 <= ignore_label <= 65535:
        raise ValueError("ignore_label 需要在 0..65535 范围内")
    if unknown_policy not in {"error", "ignore"}:
        raise ValueError(f"未知 unknown_policy: {unknown_policy}")
    if not (
        np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.bool_)
    ):
        raise ValueError(f"mask dtype 需要是整数，当前 dtype={mask.dtype}")
    max_output_id = max(resolved.output_names, default=0)
    if ignore_label in resolved.output_names:
        raise ValueError(f"ignore_label={ignore_label} 与输出类别 ID 冲突")
    if max_output_id > 65535:
        raise ValueError("输出类别 ID 超过 PNG 单通道整数范围 0..65535")
    dtype = np.uint8 if max(max_output_id, ignore_label) <= 255 else np.uint16

    if dataset is None:
        if mask.ndim != 2:
            raise ValueError(
                f"缺少 dataset label schema 时，mask 需要是二维类别图: {mask.shape}"
            )
        label_kind = "integer"
        label_channels = 1
        label_to_class: dict[MaskLabel, int] = {
            class_id: class_id
            for class_id in set(resolved.source_to_target) | resolved.dropped_ids
        }
    else:
        _validate_mask_representation(mask, dataset)
        label_kind = dataset.label_kind
        label_channels = dataset.label_channels
        label_to_class = dataset.label_to_class

    raw_to_target: dict[MaskLabel, int] = {}
    for raw_label, class_id in label_to_class.items():
        if class_id in resolved.source_to_target:
            raw_to_target[raw_label] = resolved.source_to_target[class_id]
        elif class_id in resolved.dropped_ids:
            raw_to_target[raw_label] = ignore_label
        else:
            raise ValueError(
                f"类别 {class_id} 缺少最终映射，原始 label={raw_label!r}"
            )

    if label_kind == "integer":
        if source_ignore_label is not None:
            previous = raw_to_target.get(source_ignore_label)
            if previous is not None:
                raise ValueError(
                    f"source_ignore_label={source_ignore_label} 同时是已声明的原始 label"
                )
            raw_to_target[source_ignore_label] = ignore_label
        elif dataset is None and ignore_label not in raw_to_target:
            # Preserve the historical standalone remap behavior.
            raw_to_target[ignore_label] = ignore_label
        keys = mask.astype(np.int64, copy=False)
        output_shape = mask.shape
        encoded_mapping = {
            int(raw_label): target
            for raw_label, target in raw_to_target.items()
            if isinstance(raw_label, int)
        }
    else:
        keys = _pack_color_mask(mask)
        output_shape = mask.shape[:2]
        encoded_mapping = {
            _pack_color_label(raw_label): target
            for raw_label, target in raw_to_target.items()
            if isinstance(raw_label, tuple)
        }

    if mask.size == 0:
        return np.empty(output_shape, dtype=dtype)

    # Integer PNG masks receive an O(pixels) LUT path. Color masks and unusual
    # integer dtypes use a compact sorted lookup without allocating a 2^24 LUT.
    min_value = int(keys.min())
    max_value = int(keys.max())
    if label_kind == "integer" and min_value >= 0 and max_value <= 65535:
        valid_lut = np.zeros(max_value + 1, dtype=np.bool_)
        lut = np.full(max_value + 1, ignore_label, dtype=dtype)
        for raw_value, target_id in encoded_mapping.items():
            if 0 <= raw_value <= max_value:
                valid_lut[raw_value] = True
                lut[raw_value] = target_id
        indices = keys.astype(np.int64, copy=False)
        valid_pixels = valid_lut[indices]
        if unknown_policy == "error" and not bool(valid_pixels.all()):
            unknown = [int(value) for value in np.unique(keys[~valid_pixels])]
            raise ValueError(f"mask 包含 classes 未映射的 label: {unknown}")
        return lut[indices]

    mapping_keys = np.asarray(sorted(encoded_mapping), dtype=keys.dtype)
    output = np.full(output_shape, ignore_label, dtype=dtype)
    if mapping_keys.size:
        flat_keys = keys.reshape(-1)
        positions = np.searchsorted(mapping_keys, flat_keys)
        bounded = positions < mapping_keys.size
        safe_positions = np.minimum(positions, mapping_keys.size - 1)
        valid = bounded & (mapping_keys[safe_positions] == flat_keys)
        mapping_values = np.asarray(
            [encoded_mapping[int(key)] for key in mapping_keys], dtype=dtype
        )
        output.reshape(-1)[valid] = mapping_values[safe_positions[valid]]
    else:
        valid = np.zeros(keys.size, dtype=np.bool_)
        flat_keys = keys.reshape(-1)
    if unknown_policy == "error" and not bool(valid.all()):
        unknown_values = [int(value) for value in np.unique(flat_keys[~valid])]
        unknown: list[MaskLabel]
        if label_kind == "integer":
            unknown = unknown_values
        else:
            unknown = [
                _unpack_color_label(value, label_channels)
                for value in unknown_values
            ]
        raise ValueError(f"mask 包含 classes 未映射的 label: {unknown}")
    return output


def _transfer_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "reflink":
        try:
            import errno
            import fcntl
        except ImportError:
            shutil.copy2(source, target)
            return
        target_created = False
        try:
            with source.open("rb") as source_stream, target.open("xb") as target_stream:
                target_created = True
                # Linux FICLONE creates a copy-on-write clone when supported by
                # the source and target filesystems.
                fcntl.ioctl(target_stream.fileno(), 0x40049409, source_stream.fileno())
        except OSError as exc:
            if target_created:
                target.unlink(missing_ok=True)
            unsupported = {
                errno.EINVAL,
                errno.ENOSYS,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EPERM,
                errno.EXDEV,
            }
            if exc.errno not in unsupported:
                raise
            shutil.copy2(source, target)
        else:
            shutil.copystat(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        raise ValueError(f"未知 image_mode: {mode}")


def _json_analysis(analysis: DatasetAnalysis) -> dict[str, Any]:
    return {
        "class_stats": {
            str(class_id): asdict(stats)
            for class_id, stats in sorted(analysis.class_stats.items())
        },
        "unknown_stats": {
            str(label): asdict(analysis.unknown_stats[label])
            for label in _sorted_labels(analysis.unknown_stats)
        },
        "observed_ids": sorted(analysis.observed_ids),
        "unknown_ids": _sorted_labels(analysis.unknown_ids),
        "unused_ids": sorted(analysis.unused_ids),
        "null_like_ids": sorted(analysis.null_like_ids),
        "exact_name_groups": analysis.exact_name_groups,
        "similar_name_groups": analysis.similar_name_groups,
        "issues": [asdict(issue) for issue in analysis.issues],
        "mask_count": analysis.mask_count,
        "image_count": analysis.image_count,
    }


def edit_dataset(
    dataset: SemanticDataset,
    output: Path,
    plan: EditPlan,
    *,
    analysis: DatasetAnalysis | None = None,
    ignore_label: int = 255,
    image_mode: str = "copy",
    clean: bool = False,
    dry_run: bool = False,
    require_train_val: bool = False,
) -> dict[str, Any]:
    if not 0 <= ignore_label <= 65535:
        raise ValueError("ignore_label 需要在 0..65535 范围内")
    source_ignore_label = source_ignore_label_for_edit(dataset, ignore_label)
    analysis = analysis or analyze_dataset(
        dataset, ignore_label=source_ignore_label
    )
    if analysis.unknown_ids and plan.unknown_policy == "error":
        raise ValueError(
            f"mask 存在 classes 未定义/未映射的 label: "
            f"{_sorted_labels(analysis.unknown_ids)}；"
            "请选择 unknown_policy=ignore 或补充类别定义"
        )
    resolved = resolve_plan(plan, dataset.class_names)
    if not resolved.output_names:
        raise ValueError("最终类别为空，语义分割数据集无法用于训练")
    if ignore_label in resolved.output_names:
        raise ValueError(
            f"输出类别数量占用了 ignore_label={ignore_label}；"
            "请更换 --ignore-label"
        )
    output = validate_output_location(output, [dataset.root])
    fatal_types = {
        "invalid_mask",
        "invalid_image",
        "size_mismatch",
        "duplicate_image_stem",
        "duplicate_mask_stem",
        "missing_image_dir",
        "missing_mask_dir",
        "missing_image",
        "missing_mask",
    }
    fatal_issues = [
        issue for issue in analysis.issues if issue.issue_type in fatal_types
    ]
    if fatal_issues:
        category = "无效 mask" if fatal_issues[0].issue_type == "invalid_mask" else "输入完整性问题"
        raise ValueError(
            f"检测到 {len(fatal_issues)} 个{category}，首个问题: "
            f"{fatal_issues[0].path}: {fatal_issues[0].message}"
        )

    report: dict[str, Any] = {
        "source": str(dataset.root),
        "source_config": str(dataset.config_path),
        "output": str(output),
        "ignore_label": ignore_label,
        "source_ignore_label": source_ignore_label,
        "image_mode": image_mode,
        "dry_run": dry_run,
        "source_names": dataset.class_names,
        "source_class_labels": {
            class_id: [list(label) if isinstance(label, tuple) else label for label in labels]
            for class_id, labels in dataset.class_labels.items()
        },
        "source_label_kind": dataset.label_kind,
        "output_names": resolved.output_names,
        "source_to_target": {
            class_id: resolved.source_to_target.get(class_id, ignore_label)
            for class_id in dataset.class_names
        },
        "drop_ids": sorted(plan.drop_ids),
        "merges": [asdict(merge) for merge in plan.merges],
        "renames": dict(sorted(plan.renames.items())),
        "unknown_policy": plan.unknown_policy,
        "require_train_val": require_train_val,
        "warnings": [],
        "analysis": _json_analysis(analysis),
        "splits": {},
    }
    if "train" not in dataset.splits:
        raise ValueError("语义输出缺少 train split，无法用于训练")
    if "val" not in dataset.splits:
        message = "语义输出缺少 val split；直接训练前需要提供验证集"
        if require_train_val:
            raise ValueError(message)
        report["warnings"].append(message)
    if dry_run:
        return report

    with staged_output(output, clean=clean) as stage:
        for split, paths in dataset.splits.items():
            split_report = {"images": 0, "masks": 0}
            image_files = _iter_files(paths.images, IMAGE_EXTENSIONS)
            mask_files = _iter_files(paths.masks, MASK_EXTENSIONS)
            with tqdm(
                total=len(image_files) + len(mask_files),
                desc=f"处理 {dataset.root.name}/{split}",
                unit="文件",
            ) as progress:
                for image_path in image_files:
                    relative = image_path.relative_to(paths.images)
                    _transfer_file(
                        image_path, stage / "images" / split / relative, image_mode
                    )
                    split_report["images"] += 1
                    progress.update()
                for mask_path in mask_files:
                    relative = mask_path.relative_to(paths.masks).with_suffix(".png")
                    mask = read_mask(mask_path)
                    remapped = remap_mask(
                        mask,
                        resolved,
                        ignore_label=ignore_label,
                        unknown_policy=plan.unknown_policy,
                        source_ignore_label=source_ignore_label,
                        dataset=dataset,
                    )
                    target = stage / "masks" / split / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(remapped).save(target)
                    split_report["masks"] += 1
                    progress.update()
            report["splits"][split] = split_report

        if report["splits"]["train"]["images"] == 0:
            raise ValueError("train split 中没有可用图片")
        if require_train_val and report["splits"].get("val", {}).get("images", 0) == 0:
            raise ValueError("val split 中没有可用图片")

        # Absolute paths make the config independent of the caller's cwd.
        data: dict[str, Any] = {
            "classes": resolved.output_names,
            "task": "semantic_segmentation",
        }
        for split in dataset.splits:
            data[split] = {
                "images": str(output / "images" / split),
                "masks": str(output / "masks" / split),
            }
        (stage / "data.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (stage / "classes.txt").write_text(
            "\n".join(
                resolved.output_names[class_id]
                for class_id in sorted(resolved.output_names)
            )
            + "\n",
            encoding="utf-8",
        )
        mapping = {
            key: value for key, value in report.items() if key not in {"analysis"}
        }
        (stage / "class_edit_mapping.yaml").write_text(
            yaml.safe_dump(mapping, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (stage / "class_analysis.json").write_text(
            json.dumps(report["analysis"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (stage / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return report


def print_analysis(dataset: SemanticDataset, analysis: DatasetAnalysis) -> None:
    print(f"\n数据集: {dataset.root}")
    print(
        f"图片 {analysis.image_count}，mask {analysis.mask_count}，"
        f"YAML 类别 {len(dataset.class_names)}，实际 ID {len(analysis.observed_ids)}"
    )
    print(
        f"\n  {'ID':>5}  {'类别名':<32}{'图片数':>10}{'像素数':>16}  状态"
    )
    print("  " + "-" * 82)
    for class_id, name in dataset.class_names.items():
        stats = analysis.class_stats[class_id]
        flags: list[str] = []
        if class_id in analysis.null_like_ids:
            flags.append("空/占位名")
        if class_id in analysis.unused_ids:
            flags.append("mask未使用")
        display_name = name or "<empty>"
        print(
            f"  {class_id:>5}  {display_name:<32}"
            f"{stats.image_count:>10}{stats.pixel_count:>16}  {','.join(flags)}"
        )
    if analysis.unknown_ids:
        print("\n  mask 中 classes 未映射的 label:")
        for label in _sorted_labels(analysis.unknown_ids):
            stats = analysis.unknown_stats[label]
            print(
                f"    label {label}: 图片 {stats.image_count}，"
                f"像素 {stats.pixel_count}"
            )
    if analysis.exact_name_groups:
        print(f"  重复名称候选: {analysis.exact_name_groups}")
    if analysis.similar_name_groups:
        print(f"  近似/来源前缀名称候选: {analysis.similar_name_groups}")
    issue_counts = Counter(issue.issue_type for issue in analysis.issues)
    if issue_counts:
        print(f"  问题统计: {dict(sorted(issue_counts.items()))}")


def _parse_ids(value: str) -> list[int]:
    tokens = [token for token in re.split(r"[\s,，]+", value.strip()) if token]
    if not tokens:
        raise ValueError("请输入至少一个类别 ID")
    try:
        return sorted({int(token) for token in tokens})
    except ValueError as exc:
        raise ValueError("类别 ID 需要使用整数，并以空格或逗号分隔") from exc


def _print_plan(
    plan: EditPlan,
    *,
    unknown_label: str = "YAML 未定义标注 ID",
) -> None:
    print("\n当前操作计划:")
    print(f"  删除/忽略: {sorted(plan.drop_ids)}")
    print(f"  合并: {[asdict(merge) for merge in plan.merges]}")
    print(f"  重命名: {dict(sorted(plan.renames.items()))}")
    print(f"  {unknown_label}: {plan.unknown_policy}")
    print(f"  ID 策略: {plan.id_policy}")


def _copy_plan(plan: EditPlan) -> EditPlan:
    return EditPlan(
        drop_ids=set(plan.drop_ids),
        merges=[
            MergeSpec(ids=list(merge.ids), name=merge.name)
            for merge in plan.merges
        ],
        renames=dict(plan.renames),
        unknown_policy=plan.unknown_policy,
        id_policy=plan.id_policy,
        explicit_ids=dict(plan.explicit_ids),
    )


def choose_merge_reference(
    source_paths: list[Path],
    *,
    selected: int | None = None,
    assume_yes: bool = False,
) -> list[Path] | None:
    """Choose the source whose class order and names define the merged dataset.

    The selected source is moved to the front.  Merge inventories allocate
    provisional IDs in source order, so a compact final plan then retains the
    reference YAML's class order and appends classes found only in other sources.
    """
    if len(source_paths) < 2:
        raise ValueError("合并至少需要两个数据集")
    print("\n第 2 步 · 选择合并后的类别 ID 与名称基准 data.yaml:\n")
    for index, path in enumerate(source_paths, start=1):
        print(f"  {index}. {path.expanduser().resolve()}")
    if selected is None and assume_yes:
        selected = 1
    if selected is not None and not 1 <= selected <= len(source_paths):
        raise ValueError(
            f"--reference-source 需要在 1..{len(source_paths)} 范围内: {selected}"
        )
    while selected is None:
        try:
            raw = input("选择基准 [序号，默认 1，q 退出]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        if not raw:
            selected = 1
        elif raw.isdigit():
            selected = int(raw)
        else:
            print("请输入数据集序号或 q。")
            continue
        if not 1 <= selected <= len(source_paths):
            print(f"请输入 1..{len(source_paths)}。")
            selected = None
    reference = source_paths[selected - 1]
    ordered = [reference]
    ordered.extend(
        path for index, path in enumerate(source_paths, start=1) if index != selected
    )
    print(f"基准 data.yaml: {reference.expanduser().resolve()}")
    return ordered


def recommend_reference_merge_plan(
    class_names: dict[int, str],
    analysis: DatasetAnalysis,
    *,
    reference_ids: set[int],
    base_names: dict[int, str],
) -> EditPlan:
    """Build one complete, reviewable class plan around a reference taxonomy."""
    valid_ids = set(class_names)
    reference_ids = reference_ids & valid_ids
    drop_ids = set(analysis.null_like_ids)
    drop_ids.update(analysis.unused_ids - reference_ids)
    available_ids = valid_ids - drop_ids

    parent = {class_id: class_id for class_id in available_ids}
    preferred = {
        class_id: ({class_id} if class_id in reference_ids else set())
        for class_id in available_ids
    }

    def find(class_id: int) -> int:
        while parent[class_id] != class_id:
            parent[class_id] = parent[parent[class_id]]
            class_id = parent[class_id]
        return class_id

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        combined_reference_ids = preferred[left_root] | preferred[right_root]
        # One output class can inherit at most one reference YAML class.
        if len(combined_reference_ids) > 1:
            return
        if right_root < left_root:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        preferred[left_root] = combined_reference_ids

    candidate_groups = analysis.exact_name_groups + analysis.similar_name_groups
    for candidate_ids in candidate_groups:
        ids = [class_id for class_id in candidate_ids if class_id in available_ids]
        for index, left in enumerate(ids):
            for right in ids[index + 1 :]:
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for class_id in sorted(available_ids):
        groups[find(class_id)].append(class_id)

    merges: list[MergeSpec] = []
    merged_ids: set[int] = set()
    for ids in sorted(groups.values(), key=lambda values: values[0]):
        if len(ids) < 2:
            continue
        reference_member = next(
            (class_id for class_id in ids if class_id in reference_ids),
            None,
        )
        name_source = reference_member if reference_member is not None else ids[0]
        merge_name = base_names.get(name_source, "").strip()
        if not merge_name:
            merge_name = class_names[name_source].strip() or f"class_{name_source}"
        merges.append(MergeSpec(ids=ids, name=merge_name))
        merged_ids.update(ids)

    renames = {
        class_id: base_names[class_id].strip()
        for class_id in sorted(available_ids - merged_ids)
        if base_names.get(class_id, "").strip()
        and base_names[class_id].strip() != class_names[class_id].strip()
    }
    plan = EditPlan(
        drop_ids=drop_ids,
        merges=merges,
        renames=renames,
        unknown_policy="ignore" if analysis.unknown_ids else "error",
        id_policy="compact",
    )
    resolve_plan(plan, class_names)
    return plan


def print_reference_merge_plan(
    inventory: Any,
    plan: EditPlan,
    *,
    annotation_label: str,
) -> None:
    """Print the proposed source-ID to final-ID mapping before one confirmation."""
    resolved = resolve_plan(plan, inventory.class_names)
    print("\n推荐的完整类别映射方案:")
    print(f"  基准: {inventory.sources[0].dataset.config_path}")
    print(f"  标注处理: {annotation_label}")
    for source in inventory.sources:
        print(f"\n  {source.key}: {source.dataset.config_path}")
        for old_id, provisional_id in sorted(source.old_to_provisional.items()):
            old_name = source.dataset.class_names.get(old_id, f"unknown_{old_id}")
            target_id = resolved.source_to_target.get(provisional_id)
            target = "删除/忽略" if target_id is None else (
                f"{target_id} {resolved.output_names[target_id]}"
            )
            print(f"    {old_id:>4} {old_name:<28} -> {target}")
    print("\n  最终 data.yaml 类别:")
    for class_id, name in sorted(resolved.output_names.items()):
        print(f"    {class_id}: {name}")


def confirm_reference_merge_plan() -> bool | None:
    """Accept the complete plan with y, or enter item review with n."""
    while True:
        try:
            answer = input("一次性采用以上全部类别映射？[Y/n]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if answer in {"", "y", "yes", "是"}:
            return True
        if answer in {"n", "no", "否"}:
            return False
        print("请输入 y 一次性采用，或输入 n 逐项确认。")


def interactive_plan(
    dataset: SemanticDataset,
    analysis: DatasetAnalysis,
    *,
    annotation_label: str = "mask",
    drop_effect: str = "像素映射 ignore",
    unknown_effect: str = "映射 ignore",
    merge_default_names: dict[int, str] | None = None,
) -> EditPlan | None:
    plan = EditPlan()
    if analysis.null_like_ids:
        ids = sorted(analysis.null_like_ids)
        answer = input(
            f"\n检测到空/None/null 类别 {ids}，加入删除并映射 ignore？[Y/n]: "
        ).strip().casefold()
        if answer in {"", "y", "yes", "是"}:
            plan.drop_ids.update(ids)
    unused_ids = sorted(analysis.unused_ids - plan.drop_ids)
    if unused_ids:
        answer = input(
            f"YAML 中这些类别未出现在任何{annotation_label}中 {unused_ids}，"
            "加入删除并重排 ID？[Y/n]: "
        ).strip().casefold()
        if answer in {"", "y", "yes", "是"}:
            plan.drop_ids.update(unused_ids)
    candidates = analysis.exact_name_groups + analysis.similar_name_groups
    for candidate_ids in candidates:
        occupied_ids = plan.drop_ids | _merge_members(plan)
        ids = [class_id for class_id in candidate_ids if class_id not in occupied_ids]
        if len(ids) < 2:
            continue
        names = [dataset.class_names[class_id] for class_id in ids]
        answer = input(
            f"候选同义类别 {list(zip(ids, names))}，合并？[y/N]: "
        ).strip().casefold()
        if answer not in {"y", "yes", "是"}:
            continue
        default_name = next(
            (
                merge_default_names.get(class_id, "")
                for class_id in ids
                if merge_default_names and merge_default_names.get(class_id, "")
            ),
            next((name for name in names if name), f"class_{ids[0]}"),
        )
        name = input(f"合并后的类别名 [{default_name}]: ").strip() or default_name
        plan.merges.append(MergeSpec(ids=ids, name=name))
    if analysis.unknown_ids:
        answer = input(
            f"{annotation_label}中存在 classes 未映射 label "
            f"{_sorted_labels(analysis.unknown_ids)}，"
            f"全部{unknown_effect}？[y/N]: "
        ).strip().casefold()
        if answer in {"y", "yes", "是"}:
            plan.unknown_policy = "ignore"

    print(
        "\n继续编辑类别，命令格式:\n"
        f"  d 7,8,12                 删除类别，{drop_effect}\n"
        "  m 1,5,9 = road           合并类别并命名\n"
        "  r 3 = paved_road         重命名类别\n"
        f"  u ignore                  YAML 未定义的{annotation_label} ID {unknown_effect}\n"
        f"  u error                   YAML 未定义的{annotation_label} ID 作为错误\n"
        "  show                      查看当前计划\n"
        "  reset                     清空计划\n"
        "  done                      完成编辑\n"
        "  q                         退出\n"
    )
    while True:
        try:
            raw = input("类别操作> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            continue
        command = raw.casefold()
        if command in {"q", "quit", "exit"}:
            return None
        if command in {"show", "s", "查看"}:
            _print_plan(plan, unknown_label=f"YAML 未定义{annotation_label} ID")
            continue
        if command in {"reset", "清空"}:
            plan = EditPlan()
            print("计划已清空。")
            continue
        if command in {"done", "完成"}:
            try:
                resolve_plan(plan, dataset.class_names)
            except ValueError as exc:
                print(f"计划需要调整: {exc}")
                continue
            return plan
        try:
            operation, _, payload = raw.partition(" ")
            operation = operation.casefold()
            proposed = _copy_plan(plan)
            if operation == "d":
                ids = _parse_ids(payload)
                proposed.drop_ids.update(ids)
                proposed.merges = [
                    merge
                    for merge in proposed.merges
                    if not set(merge.ids) & set(ids)
                ]
                for class_id in ids:
                    proposed.renames.pop(class_id, None)
            elif operation == "m":
                id_text, separator, name = payload.partition("=")
                if not separator:
                    raise ValueError("合并命令格式: m 1,5,9 = road")
                ids = _parse_ids(id_text)
                merge_name = name.strip()
                if not merge_name:
                    raise ValueError("请输入合并后的类别名")
                proposed.merges.append(MergeSpec(ids=ids, name=merge_name))
            elif operation == "r":
                id_text, separator, name = payload.partition("=")
                if not separator:
                    raise ValueError("重命名命令格式: r 3 = paved_road")
                ids = _parse_ids(id_text)
                if len(ids) != 1 or not name.strip():
                    raise ValueError("重命名需要一个类别 ID 和新名称")
                proposed.renames[ids[0]] = name.strip()
            elif operation == "u":
                policy = payload.strip().casefold()
                if policy not in {"error", "ignore"}:
                    raise ValueError("未知 ID 命令格式: u error 或 u ignore")
                proposed.unknown_policy = policy
            else:
                raise ValueError(f"未知命令: {operation}")
            validate_plan(proposed, dataset.class_names)
            plan = proposed
            _print_plan(plan, unknown_label=f"YAML 未定义{annotation_label} ID")
        except ValueError as exc:
            print(f"操作无效: {exc}")


def _choose_source(search_root: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in detect_datasets(search_root, kinds={"semantic_mask"})
        if candidate.kind == "semantic_mask"
    ]
    print(f"\n在 {search_root} 中检测到 {len(candidates)} 个 PNG 语义分割数据集:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. 类别 {candidate.class_count:>4}，"
            f"mask {candidate.annotation_count:>7}  {candidate.path}"
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
            return candidates[int(raw) - 1].config_path or candidates[int(raw) - 1].path
        print("请输入有效序号、p 或 q。")


def plan_from_yaml(path: Path) -> EditPlan:
    data = read_yaml(path)
    merges = [
        MergeSpec(ids=[int(value) for value in item["ids"]], name=str(item["name"]))
        for item in data.get("merges", [])
    ]
    return EditPlan(
        drop_ids={int(value) for value in data.get("drop_ids", [])},
        merges=merges,
        renames={int(key): str(value) for key, value in data.get("renames", {}).items()},
        unknown_policy=str(data.get("unknown_policy", "error")),
        id_policy=str(data.get("id_policy", "compact")),
        explicit_ids={
            int(key): int(value) for key, value in data.get("explicit_ids", {}).items()
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="源 data.yaml 或语义数据集目录")
    parser.add_argument("--out", type=Path, help="输出目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描目录，默认仓库 datasets/")
    parser.add_argument("--plan", type=Path, help="读取已有 class_edit_mapping/计划 YAML")
    parser.add_argument(
        "--id-policy",
        choices=("compact", "preserve", "explicit"),
        help="覆盖计划中的类别 ID 策略；explicit 的映射取自计划 explicit_ids",
    )
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument(
        "--image-mode",
        choices=("copy", "reflink", "hardlink", "symlink"),
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
    dataset = resolve_dataset(source)
    print("\n正在扫描全部 mask 像素值...")
    analysis = analyze_dataset(
        dataset,
        ignore_label=source_ignore_label_for_edit(dataset, args.ignore_label),
    )
    print_analysis(dataset, analysis)
    plan = plan_from_yaml(args.plan) if args.plan else interactive_plan(dataset, analysis)
    if plan is None:
        return 0
    if args.id_policy is not None:
        plan.id_policy = args.id_policy
    resolved = resolve_plan(plan, dataset.class_names)
    print("\n最终类别:")
    for class_id, name in resolved.output_names.items():
        print(f"  {class_id}: {name}")
    print(f"忽略值: {args.ignore_label}")

    output = args.out or default_output_dir(dataset.root, "edit-classes")
    if args.out is None:
        raw = input(f"输出目录 [{output}]: ").strip()
        if raw:
            output = Path(raw)
    if not args.dry_run and not args.yes:
        answer = input("确认重写 mask 并生成新数据集？[y/N]: ").strip().casefold()
        if answer not in {"y", "yes", "是"}:
            print("已取消。")
            return 0
    report = edit_dataset(
        dataset,
        output,
        plan,
        analysis=analysis,
        ignore_label=args.ignore_label,
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
        print(f"类别映射: {Path(report['output']) / 'class_edit_mapping.yaml'}")
        print(f"问题报告: {Path(report['output']) / 'class_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
