#!/usr/bin/env python3
"""按参考 data.yaml 对齐类别 ID，再挑选固定数量的代表性测试图片。

支持 YOLO Detection 与 YOLO Instance Segmentation。工具会先报告长尾类别占比，
再按类别图片数门槛决定参与抽样的类别。候选范围可以限制为 val/test，也可以包含
train。``--previous`` 是可选的图片排除集合，``--count`` 是最终测试集图片总数。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

try:
    from . import semantic_class_editor as common
    from . import yolo_class_editor as yolo
    from .dataset_discovery import (
        IMAGE_EXTENSIONS,
        KIND_LABELS,
        auto_datasets_root,
        dataset_root_from_config,
        format_candidate_count,
        format_modified_time,
        split_image_dirs,
    )
    from .dataset_transaction import (
        file_digest,
        staged_output,
        validate_output_location,
    )
    from .dataset_detector import detect_datasets
    from .interactive_helpers import prompt_path
    from .output_naming import allocate_flat_sample_stem, default_output_dir
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    import semantic_class_editor as common  # type: ignore[no-redef]
    import yolo_class_editor as yolo  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        IMAGE_EXTENSIONS,
        KIND_LABELS,
        auto_datasets_root,
        dataset_root_from_config,
        format_candidate_count,
        format_modified_time,
        split_image_dirs,
    )
    from dataset_transaction import (  # type: ignore[no-redef]
        file_digest,
        staged_output,
        validate_output_location,
    )
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from interactive_helpers import prompt_path  # type: ignore[no-redef]
    from output_naming import (  # type: ignore[no-redef]
        allocate_flat_sample_stem,
        default_output_dir,
    )
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


FATAL_ISSUE_TYPES = {
    "invalid_image",
    "invalid_label",
    "missing_image",
    "duplicate_image_stem",
    "duplicate_label_stem",
    "duplicate_names",
    "unknown_label_ids",
}

SUPPORTED_SPLITS = ("train", "val", "test")
DEFAULT_LONG_TAIL_THRESHOLDS = tuple(range(10, 101, 10))
SELECTION_STRATEGIES = ("representative", "balanced", "coverage")
RARE_CLASS_POLICIES = ("exclude-images", "keep-incidental")


@dataclass(frozen=True)
class Sample:
    split: str
    logical_stem: str
    image_path: Path
    label_path: Path | None
    label_lines: tuple[str, ...]
    class_box_counts: dict[int, int]

    @property
    def key(self) -> str:
        return f"{self.split}/{self.logical_stem}"

    @property
    def class_ids(self) -> frozenset[int]:
        return frozenset(self.class_box_counts)


@dataclass
class SelectionPlan:
    source: yolo.YoloDataset
    source_analysis: yolo.YoloAnalysis
    reference_config_path: Path
    final_class_names: dict[int, str]
    exclusion_path: Path | None
    exclusion_root: Path | None
    exclusion_images: list[Path]
    exclusion_hashes: dict[Path, str]
    source_hashes: dict[Path, str]
    available_candidates: list[Sample]
    selected: list[Sample]
    remaining: list[Sample]
    excluded_source_samples: list[Sample]
    exclusion_matches: list[dict[str, Any]]
    unmatched_exclusion_images: list[Path]
    ambiguous_exclusion_matches: list[dict[str, Any]]
    exclusion_duplicate_count: int
    source_duplicate_count: int
    source_annotation_events: list[dict[str, Any]]
    reference_filtered_image_count: int
    reference_filtered_annotation_count: int
    active_class_ids_before_exclusion: set[int]
    active_class_ids_available: set[int]
    unavailable_after_exclusion: list[int]
    missing_selected_classes: list[int]
    selected_image_counts: dict[int, int]
    selected_box_counts: dict[int, int]
    source_splits: tuple[str, ...]
    total_source_image_count: int
    pool_image_count: int
    class_image_counts_before_exclusion: dict[int, int]
    class_image_counts_available: dict[int, int]
    eligible_class_ids: set[int]
    rare_class_ids: set[int]
    min_class_images: int
    rare_class_policy: str
    rare_filtered_image_count: int
    rare_filtered_samples: list[Sample]
    selection_strategy: str
    long_tail_summary: list[dict[str, Any]]
    final_id_by_source_id: dict[int, int]
    requested_count: int
    seed: int
    duplicate_annotation_policy: str
    require_all_previous_matched: bool
    deep_validate: bool
    deduplicate_source: bool


def normalize_source_splits(
    dataset: yolo.YoloDataset,
    source_splits: Sequence[str] | None,
) -> tuple[str, ...]:
    if source_splits is None:
        return tuple(split for split in SUPPORTED_SPLITS if split in dataset.splits)
    requested = parse_source_splits(source_splits)
    missing = [split for split in requested if split not in dataset.splits]
    if missing:
        raise ValueError(
            f"数据集缺少请求的 split: {missing}；现有 split: {list(dataset.splits)}"
        )
    return requested


def parse_source_splits(source_splits: Sequence[str]) -> tuple[str, ...]:
    requested: list[str] = []
    for raw_split in source_splits:
        for split in str(raw_split).replace(",", " ").split():
            normalized = split.casefold()
            if normalized not in SUPPORTED_SPLITS:
                raise ValueError(
                    f"未知 split={split!r}；可选值为 train、val、test"
                )
            if normalized not in requested:
                requested.append(normalized)
    if not requested:
        raise ValueError("候选 split 至少需要一个")
    return tuple(requested)


def collect_samples(
    dataset: yolo.YoloDataset,
    source_splits: Sequence[str] | None = None,
) -> list[Sample]:
    selected_splits = normalize_source_splits(dataset, source_splits)
    result: list[Sample] = []
    for split in selected_splits:
        paths = dataset.splits[split]
        labels_by_stem = {
            yolo.logical_sample_stem(path, paths.labels): path
            for path in yolo.split_label_paths(paths)
        }
        for image_path in sorted(yolo.split_image_paths(paths)):
            logical_stem = yolo.logical_sample_stem(image_path, paths.images)
            label_path = labels_by_stem.get(logical_stem)
            lines: list[str] = []
            counts: Counter[int] = Counter()
            if label_path is not None and label_path.is_file():
                for raw_line in read_text_auto(label_path).splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        class_id, tokens = yolo.parse_yolo_row(line, dataset.kind)
                    except ValueError as exc:
                        raise ValueError(
                            f"标注格式错误: {label_path}: {exc}"
                        ) from exc
                    lines.append(" ".join(tokens))
                    counts[class_id] += 1
            result.append(
                Sample(
                    split=split,
                    logical_stem=logical_stem,
                    image_path=image_path.resolve(),
                    label_path=label_path.resolve() if label_path is not None else None,
                    label_lines=tuple(lines),
                    class_box_counts=dict(counts),
                )
            )
    return result


def _analyze_or_raise(dataset: yolo.YoloDataset) -> yolo.YoloAnalysis:
    analysis = yolo.analyze_dataset(dataset)
    fatal = [
        issue
        for issue in analysis.shared.issues
        if issue.issue_type in FATAL_ISSUE_TYPES
    ]
    if fatal:
        preview = "; ".join(issue.message for issue in fatal[:5])
        raise ValueError(f"候选池 {dataset.root} 校验失败: {preview}")
    observed_null = analysis.shared.observed_ids & analysis.shared.null_like_ids
    if observed_null:
        raise ValueError(
            f"候选池有效标注使用了空/占位类别: {sorted(observed_null)}"
        )
    return analysis


def _fast_analysis(
    dataset: yolo.YoloDataset,
    samples: Sequence[Sample],
) -> yolo.YoloAnalysis:
    """Build selection statistics from already parsed labels without decoding images."""
    stats = {
        class_id: common.ClassStats() for class_id in dataset.class_names
    }
    unknown_stats: dict[int, common.ClassStats] = defaultdict(common.ClassStats)
    observed_ids: set[int] = set()
    annotation_count = 0
    for sample in samples:
        annotation_count += sum(sample.class_box_counts.values())
        for class_id, count in sample.class_box_counts.items():
            observed_ids.add(class_id)
            target = stats.get(class_id)
            if target is None:
                target = unknown_stats[class_id]
            target.image_count += 1
            target.pixel_count += count
    class_ids = set(dataset.class_names)
    unknown_ids = observed_ids - class_ids
    null_like_ids = {
        class_id
        for class_id, name in dataset.class_names.items()
        if common.is_null_like_name(name, dataset.raw_class_names.get(class_id))
    }
    issues: list[common.AnalysisIssue] = []
    if unknown_ids:
        issues.append(
            common.AnalysisIssue(
                "unknown_label_ids",
                f"标注中存在 data.yaml 未定义的 ID: {sorted(unknown_ids)}",
                details={"ids": sorted(unknown_ids)},
            )
        )
    exact_groups, similar_groups = common._name_candidate_groups(
        dataset.class_names
    )
    if exact_groups:
        issues.append(
            common.AnalysisIssue(
                "duplicate_names",
                f"检测到重复类别名候选: {exact_groups}",
                details={"groups": exact_groups},
            )
        )
    shared = common.DatasetAnalysis(
        class_stats=stats,
        unknown_stats=dict(unknown_stats),
        observed_ids=observed_ids,
        unknown_ids=unknown_ids,
        unused_ids=class_ids - observed_ids,
        null_like_ids=null_like_ids,
        exact_name_groups=exact_groups,
        similar_name_groups=similar_groups,
        issues=issues,
        mask_count=sum(sample.label_path is not None for sample in samples),
        image_count=len(samples),
    )
    analysis = yolo.YoloAnalysis(
        shared=shared,
        annotation_count=annotation_count,
        malformed_count=0,
    )
    fatal = [issue for issue in issues if issue.issue_type in FATAL_ISSUE_TYPES]
    if fatal:
        raise ValueError(fatal[0].message)
    observed_null = observed_ids & null_like_ids
    if observed_null:
        raise ValueError(
            f"候选池有效标注使用了空/占位类别: {sorted(observed_null)}"
        )
    return analysis


def _resolve_exclusion_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path.parent
    return path


def collect_exclusion_images(path: Path) -> tuple[Path, list[Path]]:
    """读取排除路径中的图片；有效 YOLO 配置和普通图片目录均可使用。"""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"已有图片路径不存在: {path}")

    config_candidate: Path | None = None
    if path.is_file() and path.suffix.casefold() in {".yaml", ".yml"}:
        config_candidate = path
    elif path.is_dir():
        for name in ("data.yaml", "dataset.yaml"):
            candidate = path / name
            if candidate.is_file():
                config_candidate = candidate
                break

    if config_candidate is not None:
        try:
            dataset = yolo.resolve_dataset(config_candidate)
        except (FileNotFoundError, ValueError):
            dataset = None
        if dataset is not None:
            images = sorted(
                {
                    image.resolve()
                    for paths in dataset.splits.values()
                    for image in yolo.split_image_paths(paths)
                },
                key=str,
            )
            if images:
                return dataset.root, images

    root = _resolve_exclusion_root(path)
    if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
        return root, [path]
    images = sorted(
        (
            item.resolve()
            for item in root.rglob("*")
            if item.is_file() and item.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=str,
    )
    if not images:
        raise ValueError(f"已有图片路径中没有检测到图片: {path}")
    return root, images


def _hash_paths(paths: Iterable[Path], workers: int) -> dict[Path, str]:
    unique = sorted({path.resolve() for path in paths}, key=str)
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        digests = tqdm(
            executor.map(file_digest, unique),
            total=len(unique),
            desc="计算图片摘要",
            unit="img",
        )
        return dict(zip(unique, digests))


_ROBOFLOW_RE = re.compile(
    r"(?:^|[._-])rf[._-]([0-9a-f]{8,64})(?=$|[._-])",
    re.IGNORECASE,
)


def _normalized_stem(path: Path) -> str:
    return unicodedata.normalize("NFKC", path.stem).casefold().strip()


def _name_variants(path: Path) -> set[str]:
    stem = _normalized_stem(path)
    variants = {stem}
    match = _ROBOFLOW_RE.search(stem)
    if match is not None:
        base = stem[: match.start()].rstrip("._-")
        if base:
            variants.add(base)
    return variants


def _roboflow_id(path: Path) -> str | None:
    match = _ROBOFLOW_RE.search(_normalized_stem(path))
    return match.group(1).casefold() if match is not None else None


def _contains_on_field_boundary(longer: str, shorter: str) -> bool:
    start = 0
    while True:
        index = longer.find(shorter, start)
        if index < 0:
            return False
        end = index + len(shorter)
        left_ok = index == 0 or not longer[index - 1].isalnum()
        right_ok = end == len(longer) or not longer[end].isalnum()
        if left_ok and right_ok:
            return True
        start = index + 1


def _filename_match_score(left: Path, right: Path) -> int:
    left_rf = _roboflow_id(left)
    right_rf = _roboflow_id(right)
    if left_rf is not None and left_rf == right_rf:
        return 100_000 + len(left_rf)

    best = 0
    for left_name in _name_variants(left):
        for right_name in _name_variants(right):
            if left_name == right_name:
                best = max(best, 10_000 + len(left_name))
                continue
            shorter, longer = sorted((left_name, right_name), key=len)
            if shorter and _contains_on_field_boundary(longer, shorter):
                best = max(best, 1_000 + len(shorter))
    return best


def match_exclusion_images(
    *,
    source_samples: Sequence[Sample],
    source_hashes: dict[Path, str],
    exclusion_images: Sequence[Path],
    exclusion_hashes: dict[Path, str],
) -> tuple[set[Path], list[dict[str, Any]], list[Path], list[dict[str, Any]]]:
    """按内容、Roboflow ID、带边界文件名片段匹配排除图片。"""
    source_by_digest: dict[str, list[Path]] = defaultdict(list)
    for sample in source_samples:
        digest = source_hashes.get(sample.image_path)
        if digest is not None:
            source_by_digest[digest].append(sample.image_path)

    matched_source_paths: set[Path] = set()
    matches: list[dict[str, Any]] = []
    unmatched: list[Path] = []
    ambiguous: list[dict[str, Any]] = []
    source_paths = sorted(
        {sample.image_path for sample in source_samples}, key=str
    )

    for exclusion_image in sorted(exclusion_images, key=str):
        digest_matches = source_by_digest.get(exclusion_hashes[exclusion_image], [])
        if digest_matches:
            matched_source_paths.update(digest_matches)
            matches.append(
                {
                    "exclusion_image": str(exclusion_image),
                    "method": "sha256",
                    "source_images": [str(path) for path in digest_matches],
                }
            )
            continue

        scored = [
            (_filename_match_score(exclusion_image, source_path), source_path)
            for source_path in source_paths
        ]
        best_score = max((score for score, _ in scored), default=0)
        best_paths = [
            path for score, path in scored if score == best_score and score > 0
        ]
        if not best_paths:
            unmatched.append(exclusion_image)
            continue

        for path in best_paths:
            if path not in source_hashes:
                source_hashes[path] = file_digest(path)
        best_digests = {source_hashes[path] for path in best_paths}
        if len(best_digests) > 1:
            ambiguous.append(
                {
                    "exclusion_image": str(exclusion_image),
                    "score": best_score,
                    "source_images": [str(path) for path in best_paths],
                }
            )
            continue

        matched_source_paths.update(best_paths)
        method = "roboflow_id" if best_score >= 100_000 else "filename_field"
        matches.append(
            {
                "exclusion_image": str(exclusion_image),
                "method": method,
                "source_images": [str(path) for path in best_paths],
            }
        )

    # 候选池自身可能保存了同一内容的多个副本；命中任一副本后同步排除全部副本。
    matched_digests = {source_hashes[path] for path in matched_source_paths}
    for source_path, digest in source_hashes.items():
        if digest in matched_digests:
            matched_source_paths.add(source_path)
    return matched_source_paths, matches, unmatched, ambiguous


def remap_label_lines(lines: Sequence[str], mapping: dict[int, int]) -> tuple[str, ...]:
    result: list[str] = []
    for line in lines:
        parts = line.split()
        old_id = int(parts[0])
        if old_id not in mapping:
            raise ValueError(f"标签类别 ID={old_id} 缺少输出映射")
        result.append(" ".join([str(mapping[old_id]), *parts[1:]]))
    return tuple(result)


def resolve_reference_mapping(
    *,
    source: yolo.YoloDataset,
    reference_path: Path | None,
) -> tuple[Path, dict[int, str], dict[int, int]]:
    config_path = (
        common._find_config(reference_path)
        if reference_path is not None
        else source.config_path
    )
    config = common.read_yaml(config_path)
    configured_names, _ = common.parse_class_names(config)
    reference_names = {
        final_id: name
        for final_id, (_, name) in enumerate(sorted(configured_names.items()))
    }

    reference_id_by_name: dict[str, int] = {}
    for class_id, name in reference_names.items():
        normalized = common.normalize_name(name)
        if not normalized:
            raise ValueError(f"参考 data.yaml 的类别 ID={class_id} 名称为空")
        if normalized in reference_id_by_name:
            previous_id = reference_id_by_name[normalized]
            raise ValueError(
                "参考 data.yaml 存在重复类别名: "
                f"ID={previous_id} 和 ID={class_id} ({name})"
            )
        reference_id_by_name[normalized] = class_id

    source_to_final: dict[int, int] = {}
    for source_id, source_name in source.class_names.items():
        final_id = reference_id_by_name.get(common.normalize_name(source_name))
        if final_id is not None:
            source_to_final[source_id] = final_id
    if not source_to_final:
        raise ValueError("候选池与参考 data.yaml 之间没有同名类别")
    return config_path.resolve(), reference_names, source_to_final


def filter_samples_by_class_mapping(
    samples: Sequence[Sample],
    source_to_final: dict[int, int],
) -> tuple[list[Sample], int, int]:
    kept_samples: list[Sample] = []
    dropped_images = 0
    dropped_annotations = 0
    for sample in samples:
        kept_lines = tuple(
            line
            for line in sample.label_lines
            if int(line.split()[0]) in source_to_final
        )
        dropped_annotations += len(sample.label_lines) - len(kept_lines)
        if not kept_lines:
            dropped_images += 1
            continue
        counts: Counter[int] = Counter(
            int(line.split()[0]) for line in kept_lines
        )
        kept_samples.append(
            Sample(
                split=sample.split,
                logical_stem=sample.logical_stem,
                image_path=sample.image_path,
                label_path=sample.label_path,
                label_lines=kept_lines,
                class_box_counts=dict(counts),
            )
        )
    return kept_samples, dropped_images, dropped_annotations


def _annotation_key(line: str) -> tuple[int, tuple[float, ...]]:
    parts = line.split()
    return int(parts[0]), tuple(float(value) for value in parts[1:])


def _merge_duplicate_source_samples(
    primary: Sample,
    duplicate: Sample,
    *,
    policy: str,
) -> tuple[Sample, dict[str, Any] | None]:
    primary_rows = {_annotation_key(line): line for line in primary.label_lines}
    duplicate_rows = {_annotation_key(line): line for line in duplicate.label_lines}
    added_keys = [key for key in duplicate_rows if key not in primary_rows]
    if not added_keys:
        return primary, None
    if policy == "error":
        raise ValueError(
            "候选池包含内容相同、标注不同的图片: "
            f"{primary.image_path} <-> {duplicate.image_path}"
        )

    geometry_classes: dict[tuple[float, ...], set[int]] = defaultdict(set)
    for class_id, geometry in [*primary_rows, *duplicate_rows]:
        geometry_classes[geometry].add(class_id)
    event: dict[str, Any] = {
        "kept_image": str(primary.image_path),
        "duplicate_image": str(duplicate.image_path),
        "policy": policy,
        "added_annotations": len(added_keys) if policy == "merge" else 0,
        "discarded_annotations": len(added_keys) if policy == "keep-first" else 0,
        "class_conflicts": [
            {"class_ids": sorted(class_ids), "geometry": list(geometry)}
            for geometry, class_ids in geometry_classes.items()
            if len(class_ids) > 1
        ],
    }
    if policy == "keep-first":
        return primary, event

    merged_lines = [*primary.label_lines]
    merged_lines.extend(duplicate_rows[key] for key in added_keys)
    counts: Counter[int] = Counter(int(line.split()[0]) for line in merged_lines)
    return (
        Sample(
            split=primary.split,
            logical_stem=primary.logical_stem,
            image_path=primary.image_path,
            label_path=primary.label_path,
            label_lines=tuple(merged_lines),
            class_box_counts=dict(counts),
        ),
        event,
    )


def filter_and_dedupe_source(
    samples: Sequence[Sample],
    source_hashes: dict[Path, str],
    excluded_source_paths: set[Path],
    *,
    annotation_policy: str,
) -> tuple[list[Sample], list[Sample], int, list[dict[str, Any]]]:
    seen: dict[str, int] = {}
    available: list[Sample] = []
    excluded: list[Sample] = []
    duplicate_count = 0
    events: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: item.key):
        digest = source_hashes[sample.image_path]
        if sample.image_path in excluded_source_paths:
            excluded.append(sample)
            continue
        prior_index = seen.get(digest)
        if prior_index is not None:
            merged, event = _merge_duplicate_source_samples(
                available[prior_index], sample, policy=annotation_policy
            )
            available[prior_index] = merged
            duplicate_count += 1
            if event is not None:
                events.append(event)
            continue
        seen[digest] = len(available)
        available.append(sample)
    return available, excluded, duplicate_count, events


def _count_images(samples: Sequence[Sample]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(sample.class_ids)
    return dict(counts)


def _count_boxes(samples: Sequence[Sample]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for sample in samples:
        counts.update(sample.class_box_counts)
    return dict(counts)


def build_long_tail_summary(
    *,
    class_image_counts: dict[int, int],
    class_ids: Iterable[int],
    thresholds: Sequence[int] = DEFAULT_LONG_TAIL_THRESHOLDS,
) -> list[dict[str, Any]]:
    """统计各图片数门槛下的长尾类别数量与占比。

    分母使用候选池与参考 YAML 能够映射的全部类别，因此当前 split 中零样本的
    配置类别也会进入统计，避免把缺失类别藏起来。
    """
    ids = sorted(set(class_ids))
    total = len(ids)
    result: list[dict[str, Any]] = []
    normalized_thresholds = {int(value) for value in thresholds if value > 0}
    for threshold in sorted(normalized_thresholds):
        below = [
            class_id
            for class_id in ids
            if class_image_counts.get(class_id, 0) < threshold
        ]
        result.append(
            {
                "threshold": threshold,
                "below_class_count": len(below),
                "total_class_count": total,
                "below_class_ratio": (len(below) / total) if total else 0.0,
                "below_class_ids": below,
            }
        )
    return result


def filter_candidates_by_class_threshold(
    samples: Sequence[Sample],
    *,
    eligible_class_ids: set[int],
    rare_class_ids: set[int],
    policy: str,
) -> tuple[list[Sample], list[Sample]]:
    """按长尾策略建立候选池，始终保留入选图片的完整标注。"""
    if policy not in RARE_CLASS_POLICIES:
        raise ValueError(f"未知 rare_class_policy: {policy}")
    kept: list[Sample] = []
    dropped: list[Sample] = []
    for sample in samples:
        if not (sample.class_ids & eligible_class_ids):
            dropped.append(sample)
            continue
        if policy == "exclude-images" and sample.class_ids & rare_class_ids:
            dropped.append(sample)
            continue
        kept.append(sample)
    return kept, dropped


def _selection_targets(
    *,
    candidates: Sequence[Sample],
    count: int,
    active_class_ids: set[int],
    strategy: str,
) -> dict[int, float]:
    available = Counter()
    for sample in candidates:
        available.update(sample.class_ids & active_class_ids)
    if strategy == "representative":
        return {
            class_id: count * available[class_id] / len(candidates)
            for class_id in active_class_ids
        }

    mean_cardinality = sum(
        len(sample.class_ids & active_class_ids) for sample in candidates
    ) / len(candidates)
    equal_target = count * mean_cardinality / max(len(active_class_ids), 1)
    return {
        class_id: min(float(available[class_id]), equal_target)
        for class_id in active_class_ids
    }


def _distribution_objective(
    counts: Counter[int],
    targets: dict[int, float],
) -> float:
    return sum(
        ((counts[class_id] - target) / max(target, 1.0)) ** 2
        for class_id, target in targets.items()
    )


def select_samples(
    *,
    candidates: Sequence[Sample],
    count: int,
    active_class_ids: set[int],
    seed: int,
    strategy: str = "representative",
) -> tuple[list[Sample], list[int]]:
    if count <= 0:
        raise ValueError("--count 需要大于 0")
    if count > len(candidates):
        raise ValueError(
            f"最终测试集需要 {count} 张，完成 split、排重和类别门槛筛选后仅有 "
            f"{len(candidates)} 张候选"
        )
    if strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"未知 selection_strategy: {strategy}")
    if not active_class_ids:
        raise ValueError("类别图片数门槛过滤后没有可参与抽样的类别")

    available_per_class: Counter[int] = Counter()
    for sample in candidates:
        available_per_class.update(sample.class_ids & active_class_ids)
    rng = random.Random(seed)
    jitter = {
        sample.key: rng.random()
        for sample in sorted(candidates, key=lambda item: item.key)
    }
    relevant_class_ids = {
        sample.key: tuple(sorted(sample.class_ids & active_class_ids))
        for sample in candidates
    }
    remaining = list(sorted(candidates, key=lambda item: item.key))
    selected: list[Sample] = []
    selected_counts: Counter[int] = Counter()
    missing = set(active_class_ids)

    # coverage 是显式的类别覆盖模式。代表性/均衡模式直接拟合目标分布，避免小批量
    # 导出时为了覆盖门槛边缘类别而牺牲主体类别分布。
    while strategy == "coverage" and missing and remaining and len(selected) < count:
        def coverage_key(sample: Sample) -> tuple[float, float, float, str]:
            covered = sample.class_ids & missing
            scarcity = sum(
                1.0 / max(available_per_class[class_id], 1)
                for class_id in covered
            )
            return len(covered), scarcity, jitter[sample.key], sample.key

        best = max(remaining, key=coverage_key)
        if not (best.class_ids & missing):
            break
        remaining.remove(best)
        selected.append(best)
        selected_counts.update(best.class_ids & active_class_ids)
        missing.difference_update(best.class_ids)

    target_strategy = "balanced" if strategy == "coverage" else strategy
    targets = _selection_targets(
        candidates=candidates,
        count=count,
        active_class_ids=active_class_ids,
        strategy=target_strategy,
    )
    while remaining and len(selected) < count:
        def distribution_key(sample: Sample) -> tuple[float, float, str]:
            # Unchanged classes cancel out. Scoring only the classes carried by
            # this image keeps each greedy pass proportional to image cardinality.
            improvement = sum(
                (
                    (selected_counts[class_id] - targets[class_id]) ** 2
                    - (selected_counts[class_id] + 1 - targets[class_id]) ** 2
                )
                / max(targets[class_id], 1.0) ** 2
                for class_id in relevant_class_ids[sample.key]
            )
            return improvement, jitter[sample.key], sample.key

        best = max(remaining, key=distribution_key)
        remaining.remove(best)
        selected.append(best)
        selected_counts.update(best.class_ids & active_class_ids)

    missing_final = sorted(
        class_id for class_id in active_class_ids if selected_counts[class_id] <= 0
    )
    return selected, missing_final


def select_balanced_samples(
    *,
    candidates: Sequence[Sample],
    count: int,
    active_class_ids: set[int],
    seed: int,
) -> tuple[list[Sample], list[int]]:
    """兼容旧调用，保留原先先覆盖类别、再均衡补齐的语义。"""
    return select_samples(
        candidates=candidates,
        count=count,
        active_class_ids=active_class_ids,
        seed=seed,
        strategy="coverage",
    )


def build_selection_plan(
    *,
    source_path: Path,
    previous_path: Path | None = None,
    count: int,
    seed: int = 42,
    workers: int = 4,
    duplicate_annotation_policy: str = "merge",
    require_all_previous_matched: bool = True,
    reference_path: Path | None = None,
    source_splits: Sequence[str] | None = None,
    min_class_images: int = 0,
    rare_class_policy: str = "exclude-images",
    selection_strategy: str = "representative",
    long_tail_thresholds: Sequence[int] = DEFAULT_LONG_TAIL_THRESHOLDS,
    forced_kind: str | None = None,
    deep_validate: bool = False,
    deduplicate_source: bool = True,
) -> SelectionPlan:
    if duplicate_annotation_policy not in {"merge", "error", "keep-first"}:
        raise ValueError(
            f"未知 duplicate_annotation_policy: {duplicate_annotation_policy}"
        )
    if min_class_images < 0:
        raise ValueError("min_class_images 需要大于或等于 0")
    if rare_class_policy not in RARE_CLASS_POLICIES:
        raise ValueError(f"未知 rare_class_policy: {rare_class_policy}")
    if selection_strategy not in SELECTION_STRATEGIES:
        raise ValueError(f"未知 selection_strategy: {selection_strategy}")
    requested_splits = (
        parse_source_splits(source_splits)
        if source_splits is not None
        else None
    )
    source = yolo.resolve_dataset(
        source_path,
        forced_kind=forced_kind,
        selected_splits=requested_splits,
    )
    normalized_splits = normalize_source_splits(source, source_splits)
    pool_source = replace(
        source,
        splits={split: source.splits[split] for split in normalized_splits},
    )
    total_source_image_count = sum(
        len(yolo.split_image_paths(paths)) for paths in source.splits.values()
    )
    reference_config_path, final_class_names, source_to_final = (
        resolve_reference_mapping(
            source=source,
            reference_path=reference_path,
        )
    )
    all_source_samples = collect_samples(source, normalized_splits)
    source_analysis = (
        _analyze_or_raise(pool_source)
        if deep_validate
        else _fast_analysis(pool_source, all_source_samples)
    )
    source_samples, filtered_image_count, filtered_annotation_count = (
        filter_samples_by_class_mapping(all_source_samples, source_to_final)
    )
    if not source_samples:
        raise ValueError("所选 split 中没有包含参考类别的图片")
    source_hashes: dict[Path, str] = {}
    if previous_path is None:
        exclusion_root = None
        exclusion_images: list[Path] = []
        exclusion_hashes: dict[Path, str] = {}
        exclusion_duplicate_count = 0
        excluded_source_paths: set[Path] = set()
        exclusion_matches: list[dict[str, Any]] = []
        unmatched_exclusion_images: list[Path] = []
        ambiguous_exclusion_matches: list[dict[str, Any]] = []
    else:
        exclusion_root, exclusion_images = collect_exclusion_images(previous_path)
        exclusion_hashes = _hash_paths(exclusion_images, workers)
        exclusion_sizes = {
            path.stat().st_size for path in exclusion_images if path.is_file()
        }
        source_hashes = _hash_paths(
            (
                sample.image_path
                for sample in all_source_samples
                if sample.image_path.stat().st_size in exclusion_sizes
            ),
            workers,
        )
        exclusion_duplicate_count = len(exclusion_images) - len(
            set(exclusion_hashes.values())
        )
        (
            excluded_source_paths,
            exclusion_matches,
            unmatched_exclusion_images,
            ambiguous_exclusion_matches,
        ) = match_exclusion_images(
            source_samples=all_source_samples,
            source_hashes=source_hashes,
            exclusion_images=exclusion_images,
            exclusion_hashes=exclusion_hashes,
        )
    if deduplicate_source:
        missing_hashes = [
            sample.image_path
            for sample in all_source_samples
            if sample.image_path not in source_hashes
        ]
        source_hashes.update(_hash_paths(missing_hashes, workers))
        (
            unfiltered_candidates,
            excluded_source_samples,
            source_duplicate_count,
            source_annotation_events,
        ) = filter_and_dedupe_source(
            all_source_samples,
            source_hashes,
            excluded_source_paths,
            annotation_policy=duplicate_annotation_policy,
        )
    else:
        excluded_source_samples = [
            sample
            for sample in all_source_samples
            if sample.image_path in excluded_source_paths
        ]
        unfiltered_candidates = [
            sample
            for sample in all_source_samples
            if sample.image_path not in excluded_source_paths
        ]
        source_duplicate_count = 0
        source_annotation_events = []
    candidates, _, _ = filter_samples_by_class_mapping(
        unfiltered_candidates, source_to_final
    )

    active_before = {
        class_id for sample in source_samples for class_id in sample.class_ids
    }
    counts_before = _count_images(source_samples)
    counts_available = _count_images(candidates)
    mapped_class_ids = set(source_to_final)
    active_available = {
        class_id for class_id, value in counts_available.items() if value
    }
    unavailable_after_exclusion = sorted(active_before - active_available)
    eligible_class_ids = {
        class_id
        for class_id in active_available
        if counts_available.get(class_id, 0) >= min_class_images
    }
    rare_class_ids = mapped_class_ids - eligible_class_ids
    threshold_candidates, rare_filtered_samples = (
        filter_candidates_by_class_threshold(
            candidates,
            eligible_class_ids=eligible_class_ids,
            rare_class_ids=rare_class_ids,
            policy=rare_class_policy,
        )
    )
    thresholds = [*long_tail_thresholds, min_class_images]
    long_tail_summary = build_long_tail_summary(
        class_image_counts=counts_available,
        class_ids=mapped_class_ids,
        thresholds=thresholds,
    )
    selected, missing = select_samples(
        candidates=threshold_candidates,
        count=count,
        active_class_ids=eligible_class_ids,
        seed=seed,
        strategy=selection_strategy,
    )
    selected_keys = {sample.key for sample in selected}
    remaining = [
        sample for sample in threshold_candidates if sample.key not in selected_keys
    ]
    return SelectionPlan(
        source=source,
        source_analysis=source_analysis,
        reference_config_path=reference_config_path,
        final_class_names=final_class_names,
        exclusion_path=(
            previous_path.expanduser().resolve() if previous_path is not None else None
        ),
        exclusion_root=exclusion_root,
        exclusion_images=exclusion_images,
        exclusion_hashes=exclusion_hashes,
        source_hashes=source_hashes,
        available_candidates=threshold_candidates,
        selected=selected,
        remaining=remaining,
        excluded_source_samples=excluded_source_samples,
        exclusion_matches=exclusion_matches,
        unmatched_exclusion_images=unmatched_exclusion_images,
        ambiguous_exclusion_matches=ambiguous_exclusion_matches,
        exclusion_duplicate_count=exclusion_duplicate_count,
        source_duplicate_count=source_duplicate_count,
        source_annotation_events=source_annotation_events,
        reference_filtered_image_count=filtered_image_count,
        reference_filtered_annotation_count=filtered_annotation_count,
        active_class_ids_before_exclusion=active_before,
        active_class_ids_available=active_available,
        unavailable_after_exclusion=unavailable_after_exclusion,
        missing_selected_classes=missing,
        selected_image_counts=_count_images(selected),
        selected_box_counts=_count_boxes(selected),
        source_splits=normalized_splits,
        total_source_image_count=total_source_image_count,
        pool_image_count=len(all_source_samples),
        class_image_counts_before_exclusion=counts_before,
        class_image_counts_available=counts_available,
        eligible_class_ids=eligible_class_ids,
        rare_class_ids=rare_class_ids,
        min_class_images=min_class_images,
        rare_class_policy=rare_class_policy,
        rare_filtered_image_count=len(rare_filtered_samples),
        rare_filtered_samples=rare_filtered_samples,
        selection_strategy=selection_strategy,
        long_tail_summary=long_tail_summary,
        final_id_by_source_id=source_to_final,
        requested_count=count,
        seed=seed,
        duplicate_annotation_policy=duplicate_annotation_policy,
        require_all_previous_matched=require_all_previous_matched,
        deep_validate=deep_validate,
        deduplicate_source=deduplicate_source,
    )


def _report_dict(plan: SelectionPlan, output: Path) -> dict[str, Any]:
    match_method_counts = Counter(
        row["method"] for row in plan.exclusion_matches
    )
    class_rows = [
        {
            "source_id": source_id,
            "final_id": plan.final_id_by_source_id[source_id],
            "name": name,
            "active_before_exclusion": source_id
            in plan.active_class_ids_before_exclusion,
            "available_after_exclusion": source_id
            in plan.active_class_ids_available,
            "images_before_exclusion": plan.class_image_counts_before_exclusion.get(
                source_id, 0
            ),
            "images_available": plan.class_image_counts_available.get(source_id, 0),
            "eligible_for_selection": source_id in plan.eligible_class_ids,
            "selected_images": plan.selected_image_counts.get(source_id, 0),
            "selected_instances": plan.selected_box_counts.get(source_id, 0),
        }
        for source_id, name in sorted(plan.source.class_names.items())
        if source_id in plan.final_id_by_source_id
    ]
    return {
        "version": 4,
        "source": str(plan.source.root),
        "source_splits": list(plan.source_splits),
        "reference_data_yaml": str(plan.reference_config_path),
        "exclusion_path": str(plan.exclusion_path) if plan.exclusion_path else None,
        "output": str(output),
        "task": "segment" if plan.source.kind == "yolo_instance" else "detect",
        "seed": plan.seed,
        "requested_test_images": plan.requested_count,
        "selected_test_images": len(plan.selected),
        "final_test_images": len(plan.selected),
        "source_input_images": plan.total_source_image_count,
        "selected_split_input_images": plan.pool_image_count,
        "exclusion_input_images": len(plan.exclusion_images),
        "exclusion_unique_images": len(set(plan.exclusion_hashes.values())),
        "exclusion_internal_duplicates": plan.exclusion_duplicate_count,
        "matched_exclusion_images": len(plan.exclusion_matches),
        "unmatched_exclusion_images": [
            str(path) for path in plan.unmatched_exclusion_images
        ],
        "ambiguous_exclusion_matches": plan.ambiguous_exclusion_matches,
        "exclusion_match_methods": dict(match_method_counts),
        "exclusion_matches": plan.exclusion_matches,
        "source_images_excluded": len(plan.excluded_source_samples),
        "source_internal_duplicates": plan.source_duplicate_count,
        "available_candidates": len(plan.available_candidates),
        "min_class_images": plan.min_class_images,
        "rare_class_policy": plan.rare_class_policy,
        "rare_class_ids": sorted(plan.rare_class_ids),
        "eligible_class_ids": sorted(plan.eligible_class_ids),
        "rare_filtered_images": plan.rare_filtered_image_count,
        "selection_strategy": plan.selection_strategy,
        "deep_validate": plan.deep_validate,
        "deduplicate_source": plan.deduplicate_source,
        "train_split_included": "train" in plan.source_splits,
        "long_tail_summary": plan.long_tail_summary,
        "duplicate_annotation_policy": plan.duplicate_annotation_policy,
        "source_duplicate_annotation_events": plan.source_annotation_events,
        "reference_filtered_images": plan.reference_filtered_image_count,
        "reference_filtered_annotations": plan.reference_filtered_annotation_count,
        "unavailable_class_ids_after_exclusion": plan.unavailable_after_exclusion,
        "missing_selected_class_ids": plan.missing_selected_classes,
        "classes": class_rows,
        "selected": [
            {
                "key": sample.key,
                "image": str(sample.image_path),
                "classes": sorted(sample.class_ids),
            }
            for sample in plan.selected
        ],
    }


def _render_markdown(report: dict[str, Any]) -> str:
    split_text = ", ".join(report["source_splits"])
    exclusion_text = report["exclusion_path"] or "未设置"
    lines = [
        "# 代表性测试集选图报告",
        "",
        f"- 候选池：`{report['source']}`",
        f"- 候选 split：`{split_text}`",
        f"- 参考类别 YAML：`{report['reference_data_yaml']}`",
        f"- 排除图片路径：`{exclusion_text}`",
        f"- 要求测试集图片：{report['requested_test_images']}",
        f"- 实际测试集图片：{report['selected_test_images']}",
        f"- 类别图片数门槛：{report['min_class_images']}",
        f"- 长尾处理：{report['rare_class_policy']}",
        f"- 抽样策略：{report['selection_strategy']}",
        f"- 门槛过滤图片：{report['rare_filtered_images']}",
        f"- 已有图片输入：{report['exclusion_input_images']}",
        f"- 已有图片内部重复：{report['exclusion_internal_duplicates']}",
        f"- 已有图片成功匹配：{report['matched_exclusion_images']}",
        f"- 已有图片匹配歧义：{len(report['ambiguous_exclusion_matches'])}",
        f"- 已有图片未匹配：{len(report['unmatched_exclusion_images'])}",
        f"- 匹配方式：{report['exclusion_match_methods']}",
        f"- 候选池命中排除集合：{report['source_images_excluded']}",
        f"- 排重后可用候选：{report['available_candidates']}",
        f"- 无参考类别而排除的图片：{report['reference_filtered_images']}",
        f"- 从混合标签中删除的额外实例：{report['reference_filtered_annotations']}",
        f"- 随机种子：{report['seed']}",
        "",
        "## 长尾诊断",
        "",
        "| 图片数门槛 | 低于门槛类别 | 类别总数 | 占比 |",
        "|---:|---:|---:|---:|",
    ]
    if report["train_split_included"]:
        lines[lines.index("## 长尾诊断"):lines.index("## 长尾诊断")] = [
            "## 评估提醒",
            "",
            "候选池包含 train。评估已在该训练集上训练的模型会产生数据泄漏。",
            "",
        ]
    for row in report["long_tail_summary"]:
        lines.append(
            f"| < {row['threshold']} | {row['below_class_count']} | "
            f"{row['total_class_count']} | {row['below_class_ratio']:.1%} |"
        )
    lines.extend([
        "",
        "## 测试集类别覆盖",
        "",
        "| final_id | source_id | 类别 | 可用图片 | 选中图片 | 选中实例 | 状态 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ])
    unavailable = set(report["unavailable_class_ids_after_exclusion"])
    missing = set(report["missing_selected_class_ids"])
    for row in report["classes"]:
        status = "所选 split 未出现"
        if row["source_id"] in unavailable:
            status = "排除后无候选"
        elif not row["eligible_for_selection"]:
            status = "低于图片数门槛"
        elif row["source_id"] in missing:
            status = "测试集缺失"
        elif row["selected_images"] > 0:
            status = "已覆盖"
        lines.append(
            f"| {row['final_id']} | {row['source_id']} | {row['name']} | "
            f"{row['images_available']} | {row['selected_images']} | "
            f"{row['selected_instances']} | {status} |"
        )
    lines.append("")
    return "\n".join(lines)


def print_plan_summary(plan: SelectionPlan) -> None:
    print("\n" + "=" * 72)
    print(f"已加载候选图片   : {plan.total_source_image_count}")
    print(f"候选 split       : {', '.join(plan.source_splits)}")
    print(f"所选 split 图片  : {plan.pool_image_count}")
    print(f"参考类别 YAML    : {plan.reference_config_path}")
    print(f"无目标类别排除   : {plan.reference_filtered_image_count}")
    print(f"多余标注删除     : {plan.reference_filtered_annotation_count}")
    print(f"已有图片输入     : {len(plan.exclusion_images)}")
    print(f"已有图片内部重复 : {plan.exclusion_duplicate_count}")
    print(f"已有图片成功匹配 : {len(plan.exclusion_matches)}")
    print(f"已有图片匹配歧义 : {len(plan.ambiguous_exclusion_matches)}")
    print(f"已有图片未匹配   : {len(plan.unmatched_exclusion_images)}")
    method_counts = Counter(row["method"] for row in plan.exclusion_matches)
    print(
        "匹配方式           : "
        f"内容 {method_counts['sha256']} / "
        f"RF字段 {method_counts['roboflow_id']} / "
        f"文件名字段 {method_counts['filename_field']}"
    )
    print(f"候选池命中排除   : {len(plan.excluded_source_samples)}")
    print(f"候选池内部重复   : {plan.source_duplicate_count}")
    print(f"排重后可用候选   : {len(plan.available_candidates)}")
    print(f"类别图片数门槛   : {plan.min_class_images}")
    print(f"长尾图片处理     : {plan.rare_class_policy}")
    print(f"门槛过滤图片     : {plan.rare_filtered_image_count}")
    print(f"抽样策略         : {plan.selection_strategy}")
    print(f"输入校验         : {'完整' if plan.deep_validate else '快速'}")
    print(f"源池内容排重     : {'开启' if plan.deduplicate_source else '按路径'}")
    print(f"最终测试集       : {len(plan.selected)} / {plan.requested_count}")
    if "train" in plan.source_splits:
        print("评估提醒         : train 已进入候选；已训练模型可能产生数据泄漏")
    print("-" * 72)
    print("长尾类别诊断（分母包含参考 YAML 可映射、当前 split 为 0 张的类别）")
    for row in plan.long_tail_summary:
        print(
            f"  少于 {row['threshold']:>4} 张: "
            f"{row['below_class_count']:>4}/{row['total_class_count']:<4} 类 "
            f"({row['below_class_ratio']:.1%})"
        )
    print("-" * 72)
    print(
        f"{'ID':>5}  {'类别':<24}{'可用图片':>10}"
        f"{'选中图片':>10}{'选中实例':>10}"
    )
    for source_id, name in sorted(plan.source.class_names.items()):
        if source_id not in plan.final_id_by_source_id:
            continue
        final_id = plan.final_id_by_source_id[source_id]
        marker = ""
        if source_id in plan.unavailable_after_exclusion:
            marker = "  排除后无候选"
        elif source_id not in plan.eligible_class_ids:
            marker = "  低于门槛"
        elif source_id in plan.missing_selected_classes:
            marker = "  测试集缺失"
        print(
            f"{source_id:>3}->{final_id:<3}  {name:<22}"
            f"{plan.class_image_counts_available.get(source_id, 0):>10}"
            f"{plan.selected_image_counts.get(source_id, 0):>10}"
            f"{plan.selected_box_counts.get(source_id, 0):>10}{marker}"
        )
    print("=" * 72)


def validate_exclusion_coverage(plan: SelectionPlan) -> None:
    if not plan.require_all_previous_matched:
        return
    unresolved_count = len(plan.unmatched_exclusion_images) + len(
        plan.ambiguous_exclusion_matches
    )
    if unresolved_count <= 0:
        return
    preview = [str(path) for path in plan.unmatched_exclusion_images[:3]]
    preview.extend(
        row["exclusion_image"] for row in plan.ambiguous_exclusion_matches[:3]
    )
    raise ValueError(
        f"已有图片中有 {unresolved_count} 张未能唯一匹配候选池；"
        f"示例: {preview}。先检查命名字段，或使用 --allow-unmatched-previous 继续。"
    )


def write_selection(
    plan: SelectionPlan,
    *,
    output: Path,
    image_mode: str,
    clean: bool,
) -> Path:
    output = validate_output_location(
        output,
        [
            path
            for path in (plan.source.root, plan.exclusion_root)
            if path is not None
        ],
    )
    validate_exclusion_coverage(plan)
    if len(plan.selected) != plan.requested_count:
        raise ValueError(
            f"测试集要求 {plan.requested_count} 张，当前选中 {len(plan.selected)} 张"
        )
    source_to_final = dict(plan.final_id_by_source_id)
    final_names = dict(plan.final_class_names)
    report = _report_dict(plan, output)
    with staged_output(output, clean=clean) as stage:
        image_dir = stage / "images" / "test"
        label_dir = stage / "labels" / "test"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        used_stems: set[str] = set()
        output_rows: list[dict[str, str]] = []
        for sample in tqdm(plan.selected, desc="生成测试集", unit="img"):
            allocated_stem = allocate_flat_sample_stem(
                sample.image_path.stem, used_stems
            )
            image_target = (
                image_dir / f"{allocated_stem}{sample.image_path.suffix.lower()}"
            )
            label_target = label_dir / f"{allocated_stem}.txt"
            common._transfer_file(sample.image_path, image_target, image_mode)
            rewritten = remap_label_lines(sample.label_lines, source_to_final)
            label_target.write_text(
                "\n".join(rewritten) + ("\n" if rewritten else ""),
                encoding="utf-8",
            )
            output_rows.append(
                {
                    "source_split": sample.split,
                    "source_image": str(sample.image_path),
                    "output_image": str(Path("images/test") / image_target.name),
                    "output_label": str(Path("labels/test") / label_target.name),
                }
            )

        data = {
            "path": str(output),
            "test": "images/test",
            "task": "segment" if plan.source.kind == "yolo_instance" else "detect",
            "nc": len(final_names),
            "names": final_names,
        }
        (stage / "data.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        (stage / "classes.txt").write_text(
            "\n".join(final_names[class_id] for class_id in sorted(final_names))
            + "\n",
            encoding="utf-8",
        )
        (stage / "class_mapping.yaml").write_text(
            yaml.safe_dump(
                {
                    "reference_data_yaml": str(plan.reference_config_path),
                    "final_names": final_names,
                    "source_to_final": [
                        {
                            "source_id": source_id,
                            "source_name": plan.source.class_names[source_id],
                            "final_id": final_id,
                        }
                        for source_id, final_id in sorted(source_to_final.items())
                    ]
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (stage / "output_mapping.json").write_text(
            json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "selected_test.txt").write_text(
            "\n".join(str(sample.image_path) for sample in plan.selected) + "\n",
            encoding="utf-8",
        )
        (stage / "excluded_previous_matches.txt").write_text(
            "\n".join(
                str(sample.image_path) for sample in plan.excluded_source_samples
            )
            + "\n",
            encoding="utf-8",
        )
        (stage / "exclusion_matches.json").write_text(
            json.dumps(plan.exclusion_matches, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "unmatched_previous.txt").write_text(
            "\n".join(str(path) for path in plan.unmatched_exclusion_images)
            + "\n",
            encoding="utf-8",
        )
        (stage / "ambiguous_previous_matches.json").write_text(
            json.dumps(
                plan.ambiguous_exclusion_matches, ensure_ascii=False, indent=2
            )
            + "\n",
            encoding="utf-8",
        )
        (stage / "remaining_source.txt").write_text(
            "\n".join(str(sample.image_path) for sample in plan.remaining) + "\n",
            encoding="utf-8",
        )
        (stage / "filtered_by_class_threshold.txt").write_text(
            "\n".join(str(sample.image_path) for sample in plan.rare_filtered_samples)
            + "\n",
            encoding="utf-8",
        )
        (stage / "selection_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "selection_report.md").write_text(
            _render_markdown(report), encoding="utf-8"
        )
        (stage / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return output


def _choose_task_kind() -> str | None:
    print("\n第 1 步 · 选择数据集任务:")
    print("  1. Det（YOLO Detection）")
    print("  2. Seg（YOLO Instance Segmentation polygon）")
    while True:
        try:
            raw = input("选择 [1/2，q 退出]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        if raw == "1":
            return "yolo_detection"
        if raw == "2":
            return "yolo_instance"
        print("请输入 1 或 2。")


def _choose_source(search_root: Path, required_kind: str) -> Path | None:
    candidates = [
        candidate
        for candidate in detect_datasets(
            search_root,
            kinds={required_kind},
            include_unknown=True,
        )
        if candidate.image_count > 0
        and (
            candidate.kind == required_kind
            or (candidate.kind == "unknown" and candidate.config_path is not None)
        )
    ]
    task_name = "Det" if required_kind == "yolo_detection" else "Seg"
    print(f"\n第 2 步 · 选择 YOLO {task_name} 候选池（最近修改优先）:\n")
    print(f"  扫描根目录: {search_root.resolve()}")
    print(
        f"  {'#':>3} {'类型':<18}{'图片(快扫)':>12}{'标注文件':>10}{'类别':>6}  "
        f"{'split':<16}{'最后修改':<16} 配置/路径"
    )
    for index, candidate in enumerate(candidates, start=1):
        kind_label = KIND_LABELS[candidate.kind]
        if candidate.kind == "unknown":
            kind_label = f"待按 {task_name} 读取"
        image_count = format_candidate_count(
            candidate.image_count, candidate.image_count_is_exact
        )
        annotation_count = format_candidate_count(
            candidate.annotation_count, candidate.annotation_count_is_exact
        )
        split_text = ",".join(candidate.splits) or "-"
        print(
            f"  {index:>3} {kind_label:<18}{image_count:>12}{annotation_count:>10}"
            f"{candidate.class_count:>6}  "
            f"{split_text:<16}"
            f"{format_modified_time(candidate.modified_time):<16} "
            f"{candidate.config_path or candidate.path}"
        )
    if not candidates:
        print("  自动候选为空；可输入 p 直接指定 data.yaml，并查看路径诊断。")
    print("   p. 输入任意 data.yaml 或数据集目录")
    while True:
        try:
            raw = input(f"选择 [1-{len(candidates)}/p/q]: ").strip()
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


def _choose_exclusion_path(search_root: Path, source_root: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in detect_datasets(search_root)
        if candidate.image_count > 0
        and candidate.path.resolve() != source_root.resolve()
    ]
    print("\n选择之前挑过的图片路径（仅用于排除）:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. 图片 {candidate.image_count:>6}  "
            f"{KIND_LABELS.get(candidate.kind, candidate.kind):<18} {candidate.path}"
        )
    print("   p. 输入任意图片目录、data.yaml 或单张图片")
    while True:
        try:
            raw = input(f"选择 [1-{len(candidates)}/p/q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        if raw.casefold() == "p":
            value = input("之前挑过的图片路径: ").strip()
            return Path(value).expanduser().resolve() if value else None
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            candidate = candidates[int(raw) - 1]
            return candidate.config_path or candidate.path
        print("请输入有效序号、p 或 q。")


def _prompt_reference_yaml(default: Path) -> Path | None:
    print("\n第 2 步 · 输入类别顺序对照 data.yaml:\n")
    while True:
        try:
            raw = input(f"参考 data.yaml [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        chosen = Path(raw).expanduser().resolve() if raw else default.resolve()
        try:
            common._find_config(chosen)
        except FileNotFoundError as exc:
            print(exc)
            continue
        return chosen


def _prompt_count(default: int = 200) -> int | None:
    print("\n设置最终测试集图片数量:\n")
    while True:
        try:
            raw = input(f"最终测试集需要多少张 [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return default
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("请输入大于 0 的整数。")


def _prompt_source_splits(
    available_splits: Sequence[str],
) -> tuple[str, ...] | None:
    available = tuple(
        split for split in SUPPORTED_SPLITS if split in available_splits
    )
    eval_splits = tuple(split for split in ("val", "test") if split in available)
    print("\n第 3 步 · 选择候选图片来源:")
    if eval_splits:
        print(f"  1. 评估集优先：{', '.join(eval_splits)}")
    print(f"  2. 全部 split：{', '.join(available)}")
    print("  3. 自定义 split")
    default = "1" if eval_splits else "2"
    while True:
        try:
            raw = input(f"选择 [默认 {default}，q 退出]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        raw = raw or default
        if raw in {"q", "quit", "exit"}:
            return None
        if raw == "1" and eval_splits:
            return eval_splits
        if raw == "2":
            return available
        if raw == "3":
            custom = input("输入 split，以空格分隔（train val test）: ").strip()
            try:
                requested = parse_source_splits(custom.split())
            except ValueError as exc:
                print(exc)
                continue
            missing = [split for split in requested if split not in available]
            if missing:
                print(
                    f"数据集缺少请求的 split: {missing}；"
                    f"现有 split: {list(available)}"
                )
                continue
            return requested
        print("请输入有效序号。")


def preview_long_tail(
    *,
    source_path: Path,
    reference_path: Path | None,
    source_splits: Sequence[str] | None,
    thresholds: Sequence[int] = DEFAULT_LONG_TAIL_THRESHOLDS,
    forced_kind: str | None = None,
) -> tuple[
    dict[int, int],
    dict[int, str],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """快速预览所选 split 的类别图片分布，供交互式门槛决策。"""
    requested_splits = (
        parse_source_splits(source_splits)
        if source_splits is not None
        else None
    )
    source = yolo.resolve_dataset(
        source_path,
        forced_kind=forced_kind,
        selected_splits=requested_splits,
    )
    _, _, source_to_final = resolve_reference_mapping(
        source=source, reference_path=reference_path
    )
    samples, _, _ = filter_samples_by_class_mapping(
        collect_samples(source, source_splits), source_to_final
    )
    counts = _count_images(samples)
    summary = build_long_tail_summary(
        class_image_counts=counts,
        class_ids=source_to_final,
        thresholds=thresholds,
    )
    names = {
        class_id: source.class_names[class_id]
        for class_id in source_to_final
    }
    pool_stats = {
        "root": source.root,
        "config_path": source.config_path,
        "kind": source.kind,
        "splits": normalize_source_splits(source, source_splits),
        "image_count": len(samples),
        "annotation_file_count": sum(
            sample.label_path is not None and sample.label_path.is_file()
            for sample in samples
        ),
        "instance_count": sum(len(sample.label_lines) for sample in samples),
        "class_count": len(names),
        "active_class_count": sum(counts.get(class_id, 0) > 0 for class_id in names),
    }
    return counts, names, summary, pool_stats


def print_long_tail_preview(
    counts: dict[int, int],
    names: dict[int, str],
    summary: Sequence[dict[str, Any]],
    pool_stats: dict[str, Any],
) -> None:
    task_label = (
        "YOLO Detection"
        if pool_stats["kind"] == "yolo_detection"
        else "YOLO Instance Segmentation"
    )
    print("\n第 4 步 · 数据集基础信息与长尾诊断:")
    print(f"  配置文件: {pool_stats['config_path']}")
    print(f"  数据集根: {pool_stats['root']}")
    print(f"  任务类型: {task_label}")
    print(f"  统计 split: {', '.join(pool_stats['splits'])}")
    print(f"  图片总数: {pool_stats['image_count']}")
    print(f"  标注文件: {pool_stats['annotation_file_count']}")
    print(f"  框/实例数: {pool_stats['instance_count']}")
    print(
        f"  类别数量: {pool_stats['class_count']}，"
        f"当前 split 有图类别: {pool_stats['active_class_count']}"
    )
    print("  长尾分布:")
    for row in summary:
        print(
            f"  少于 {row['threshold']:>3} 张："
            f"{row['below_class_count']}/{row['total_class_count']} 类，"
            f"占 {row['below_class_ratio']:.1%}"
        )
    tail_rows = sorted(names, key=lambda class_id: (counts.get(class_id, 0), class_id))
    if tail_rows:
        print("  图片数最少的类别：")
        for class_id in tail_rows[:20]:
            print(
                f"    ID {class_id:>4}  {counts.get(class_id, 0):>6} 张  "
                f"{names[class_id]}"
            )
        if len(tail_rows) > 20:
            print(f"    其余 {len(tail_rows) - 20} 类写入最终报告")


def _prompt_min_class_images(default: int = 80) -> int | None:
    while True:
        try:
            raw = input(f"类别至少多少张图片才参与抽样 [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not raw:
            return default
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        if raw.isdigit():
            return int(raw)
        print("请输入大于或等于 0 的整数。")


def _prompt_selection_strategy() -> str | None:
    print("\n第 5 步 · 选择抽样目标:")
    print("  1. representative：保持候选池类别分布，适合稳定评估")
    print("  2. balanced：拉平合格类别，适合类别专项对比")
    print("  3. coverage：先覆盖合格类别，适合覆盖检查")
    while True:
        try:
            raw = input("选择 [默认 1，q 退出]: ").strip().casefold() or "1"
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        mapping = {"1": "representative", "2": "balanced", "3": "coverage"}
        if raw in mapping:
            return mapping[raw]
        print("请输入 1、2 或 3。")


def _prompt_rare_class_policy() -> str | None:
    print("\n第 6 步 · 选择含长尾类别图片的处理方式:")
    print("  1. exclude-images：整张移出候选池，评估标签最干净")
    print("  2. keep-incidental：合格类别驱动选图，同时保留图片内全部标注")
    while True:
        try:
            raw = input("选择 [默认 1，q 退出]: ").strip().casefold() or "1"
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw in {"q", "quit", "exit"}:
            return None
        if raw == "1":
            return "exclude-images"
        if raw == "2":
            return "keep-incidental"
        print("请输入 1 或 2。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="YOLO Det/Seg 候选池")
    parser.add_argument(
        "--task",
        choices=("auto", "detect", "segment"),
        default="auto",
        help="候选任务；交互模式先选择 Det 或 Seg",
    )
    parser.add_argument(
        "--reference-yaml",
        type=Path,
        help="类别名称和 ID 顺序的参考 data.yaml；输出完整继承该类别表",
    )
    parser.add_argument("--previous", type=Path, help="可选的已有图片路径，仅用于排除")
    parser.add_argument("--count", type=int, help="最终测试集图片总数")
    parser.add_argument(
        "--splits",
        nargs="+",
        metavar="SPLIT",
        help="候选来源，例如 --splits val test 或 --splits train val test",
    )
    parser.add_argument(
        "--min-class-images",
        type=int,
        default=0,
        help="类别在候选池中至少出现于多少张图片；0 表示保留全部活跃类别",
    )
    parser.add_argument(
        "--strategy",
        choices=SELECTION_STRATEGIES,
        default="representative",
        help="representative=保持分布，balanced=拉平类别，coverage=覆盖优先",
    )
    parser.add_argument(
        "--rare-class-policy",
        choices=RARE_CLASS_POLICIES,
        default="exclude-images",
        help="exclude-images=整图过滤，keep-incidental=保留混合图完整标注",
    )
    parser.add_argument(
        "--tail-thresholds",
        type=int,
        nargs="+",
        default=list(DEFAULT_LONG_TAIL_THRESHOLDS),
        help="报告的长尾诊断门槛，默认从 10 到 100、每 10 张一档",
    )
    parser.add_argument("--out", type=Path, help="测试集输出目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描根目录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--deep-validate",
        action="store_true",
        help="逐张解码图片并执行完整性校验；大型数据集耗时较长",
    )
    parser.add_argument(
        "--deduplicate-source",
        action="store_true",
        help="计算候选池全部图片摘要并合并重复内容；默认按路径去重",
    )
    parser.add_argument(
        "--duplicate-annotation-policy",
        choices=("merge", "error", "keep-first"),
        default="merge",
        help="候选池内部同图异标时：合并、报错或保留首份",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "reflink", "hardlink", "symlink"),
        default="reflink",
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--allow-unmatched-previous",
        action="store_true",
        help="允许已有图片存在未匹配项或一对多歧义",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = args.source is None or args.count is None
    full_wizard = args.source is None
    if interactive:
        print(
            "\n流程：选择候选 split，查看长尾占比，设置类别门槛，再挑选固定数量。\n"
            "输出兼容 YOLO Detection 与 YOLO Instance Segmentation。\n"
        )
        search_root = (args.datasets or auto_datasets_root()).expanduser().resolve()
        if args.task == "detect":
            forced_kind = "yolo_detection"
        elif args.task == "segment":
            forced_kind = "yolo_instance"
        else:
            forced_kind = None
        if full_wizard:
            forced_kind = _choose_task_kind()
            if forced_kind is None:
                return 0
        source_path = args.source or _choose_source(search_root, forced_kind)
        if source_path is None:
            return 0
        source_config_path = common._find_config(source_path)
        source_config = common.read_yaml(source_config_path)
        source_root = dataset_root_from_config(source_config_path, source_config)
        reference_path = args.reference_yaml or _prompt_reference_yaml(
            source_config_path
        )
        if reference_path is None:
            return 0
        available_splits = tuple(
            split
            for split in SUPPORTED_SPLITS
            if split in split_image_dirs(source_root, source_config)
        )
        if full_wizard and args.splits is None:
            source_splits = _prompt_source_splits(available_splits)
            if source_splits is None:
                return 0
        else:
            source_splits = (
                parse_source_splits(args.splits)
                if args.splits is not None
                else available_splits
            )
            missing_splits = [
                split for split in source_splits if split not in available_splits
            ]
            if missing_splits:
                raise ValueError(
                    f"数据集缺少请求的 split: {missing_splits}；"
                    f"现有 split: {list(available_splits)}"
                )

        min_class_images = args.min_class_images
        selection_strategy = args.strategy
        rare_class_policy = args.rare_class_policy
        if full_wizard:
            counts, names, summary, pool_stats = preview_long_tail(
                source_path=source_path,
                reference_path=reference_path,
                source_splits=source_splits,
                thresholds=args.tail_thresholds,
                forced_kind=forced_kind,
            )
            print_long_tail_preview(counts, names, summary, pool_stats)
            chosen_minimum = _prompt_min_class_images()
            if chosen_minimum is None:
                return 0
            min_class_images = chosen_minimum
            chosen_strategy = _prompt_selection_strategy()
            if chosen_strategy is None:
                return 0
            selection_strategy = chosen_strategy
            chosen_policy = _prompt_rare_class_policy()
            if chosen_policy is None:
                return 0
            rare_class_policy = chosen_policy

        previous_path = args.previous
        if full_wizard and previous_path is None:
            try:
                use_previous = input(
                    "\n第 7 步 · 添加已有图片排除集合？[y/N]: "
                ).strip().casefold()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if use_previous in {"y", "yes", "是"}:
                previous_path = _choose_exclusion_path(search_root, source_root)
                if previous_path is None:
                    return 0
        count = args.count if args.count is not None else _prompt_count()
        if count is None:
            return 0
    else:
        source_path = args.source
        reference_path = args.reference_yaml
        previous_path = args.previous
        count = args.count
        if args.task == "detect":
            forced_kind = "yolo_detection"
        elif args.task == "segment":
            forced_kind = "yolo_instance"
        else:
            forced_kind = None
        source_splits = (
            parse_source_splits(args.splits)
            if args.splits is not None
            else None
        )
        min_class_images = args.min_class_images
        selection_strategy = args.strategy
        rare_class_policy = args.rare_class_policy

    source_path = source_path.expanduser().resolve()
    if previous_path is not None:
        previous_path = previous_path.expanduser().resolve()
    source_config_path = common._find_config(source_path)
    source_config = common.read_yaml(source_config_path)
    source_root = dataset_root_from_config(source_config_path, source_config)
    output = (
        args.out.expanduser()
        if args.out is not None
        else default_output_dir(source_root, "balanced-test")
    )
    if interactive and args.out is None:
        chosen = prompt_path("第 5 步 · 测试集输出目录", output)
        if chosen is None:
            return 0
        output = chosen

    plan = build_selection_plan(
        source_path=source_path,
        previous_path=previous_path,
        count=count,
        seed=args.seed,
        workers=args.workers,
        duplicate_annotation_policy=args.duplicate_annotation_policy,
        require_all_previous_matched=not args.allow_unmatched_previous,
        reference_path=reference_path,
        source_splits=source_splits,
        min_class_images=min_class_images,
        rare_class_policy=rare_class_policy,
        selection_strategy=selection_strategy,
        long_tail_thresholds=args.tail_thresholds,
        forced_kind=forced_kind,
        deep_validate=args.deep_validate,
        deduplicate_source=args.deduplicate_source,
    )
    print_plan_summary(plan)
    if args.dry_run:
        print("DRY-RUN 完成。")
        return 0
    validate_exclusion_coverage(plan)
    if not args.yes:
        try:
            raw = input(f"生成 {count} 张测试图片到 {output}？[y/N]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if raw not in {"y", "yes", "是"}:
            print("已取消。")
            return 0
    result = write_selection(
        plan,
        output=output,
        image_mode=args.image_mode,
        clean=args.clean,
    )
    print(f"完成: {result}")
    print(f"测试集图片: {len(plan.selected)}")
    print(f"报告: {result / 'selection_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
