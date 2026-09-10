#!/usr/bin/env python3
"""汇总数据集 train/val/test，并按类别重新进行多标签分层划分。

支持 YOLO Detection、YOLO Instance Segmentation 和 PNG Semantic Segmentation。
脚本保持图片及对应 TXT/mask 为一个整体，优先让具备足够样本的类别覆盖
train/val/test，再尽量贴近指定比例。结果通过暂存目录原子发布到新的输出目录。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import yaml

try:
    from . import semantic_class_editor as common
    from . import yolo_class_editor as yolo
    from .dataset_discovery import (
        IMAGE_EXTENSIONS,
        KIND_LABELS,
        auto_datasets_root,
    )
    from .dataset_detector import detect_datasets, inspect_dataset
    from .dataset_transaction import file_digest, staged_output, validate_output_location
    from .interactive_helpers import prompt_path
    from .output_naming import allocate_flat_sample_stem
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    import semantic_class_editor as common  # type: ignore[no-redef]
    import yolo_class_editor as yolo  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        IMAGE_EXTENSIONS,
        KIND_LABELS,
        auto_datasets_root,
    )
    from dataset_detector import (  # type: ignore[no-redef]
        detect_datasets,
        inspect_dataset,
    )
    from dataset_transaction import (  # type: ignore[no-redef]
        file_digest,
        staged_output,
        validate_output_location,
    )
    from interactive_helpers import prompt_path  # type: ignore[no-redef]
    from output_naming import allocate_flat_sample_stem  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


SPLITS = ("train", "val", "test")
FATAL_ISSUES = {
    "invalid_image",
    "invalid_label",
    "missing_image",
    "duplicate_image_stem",
    "duplicate_label_stem",
    "duplicate_names",
    "unknown_label_ids",
}
SEMANTIC_FATAL_ISSUES = {
    "invalid_mask",
    "invalid_image",
    "size_mismatch",
    "duplicate_image_stem",
    "duplicate_mask_stem",
    "missing_image_dir",
    "missing_mask_dir",
    "missing_image",
    "missing_mask",
    "unknown_mask_ids",
    "duplicate_names",
}


@dataclass(frozen=True)
class Sample:
    source_split: str
    logical_stem: str
    image_path: Path
    annotation_path: Path | None
    label_lines: tuple[str, ...]
    amount_counts: dict[int, int]

    @property
    def key(self) -> str:
        return f"{self.source_split}/{self.logical_stem}"

    @property
    def class_ids(self) -> frozenset[int]:
        return frozenset(self.amount_counts)


@dataclass
class SplitPlan:
    dataset: Any
    format_kind: str
    amount_unit: str
    samples: list[Sample]
    assignments: dict[str, str]
    ratios: dict[str, float]
    target_sizes: dict[str, int]
    duplicate_rows: list[dict[str, str]]
    class_image_counts: dict[int, dict[str, int]]
    class_amount_counts: dict[int, dict[str, int]]
    missing_coverage: list[dict[str, Any]]
    seed: int


def _validate_ratios(values: Sequence[float]) -> dict[str, float]:
    if len(values) != len(SPLITS):
        raise ValueError("--ratios 需要依次提供 train、val、test 三个数")
    if any((not math.isfinite(value)) or value < 0 for value in values):
        raise ValueError("--ratios 需要使用有限的非负数")
    total = sum(values)
    if total <= 0:
        raise ValueError("--ratios 的总和需要大于 0")
    return {split: value / total for split, value in zip(SPLITS, values)}


def calculate_target_sizes(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Use largest remainders while keeping active splits non-empty when possible."""
    raw = {split: total * ratios[split] for split in SPLITS}
    result = {split: int(math.floor(raw[split])) for split in SPLITS}
    remainder = total - sum(result.values())
    order = sorted(SPLITS, key=lambda split: (-(raw[split] - result[split]), SPLITS.index(split)))
    for split in order[:remainder]:
        result[split] += 1

    active = [split for split in SPLITS if ratios[split] > 0]
    if total >= len(active):
        for split in active:
            if result[split] > 0:
                continue
            donors = sorted(
                active,
                key=lambda item: (result[item] - 1, ratios[item], -SPLITS.index(item)),
                reverse=True,
            )
            donor = next(item for item in donors if result[item] > 1)
            result[donor] -= 1
            result[split] += 1
    return result


def collect_samples(dataset: yolo.YoloDataset) -> list[Sample]:
    result: list[Sample] = []
    split_rank = {split: index for index, split in enumerate(SPLITS)}
    for split, paths in dataset.splits.items():
        images_by_stem: dict[str, Path] = {}
        for path in yolo.split_image_paths(paths):
            stem = yolo.logical_sample_stem(path, paths.images)
            if stem in images_by_stem:
                raise ValueError(f"{split} 中多个图片具有相同配对 stem: {stem}")
            images_by_stem[stem] = path.resolve()
        labels_by_stem: dict[str, Path] = {}
        for path in yolo.split_label_paths(paths):
            stem = yolo.logical_sample_stem(path, paths.labels)
            if stem in labels_by_stem:
                raise ValueError(f"{split} 中多个标签具有相同配对 stem: {stem}")
            labels_by_stem[stem] = path.resolve()

        for stem, image_path in sorted(images_by_stem.items()):
            label_path = labels_by_stem.get(stem)
            lines: list[str] = []
            counts: Counter[int] = Counter()
            if label_path is not None:
                for raw_line in read_text_auto(label_path).splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    class_id, tokens = yolo.parse_yolo_row(line, dataset.kind)
                    lines.append(" ".join(tokens))
                    counts[class_id] += 1
            result.append(
                Sample(
                    source_split=split,
                    logical_stem=stem,
                    image_path=image_path,
                    annotation_path=label_path,
                    label_lines=tuple(lines),
                    amount_counts=dict(counts),
                )
            )
    return sorted(result, key=lambda sample: (split_rank.get(sample.source_split, 99), sample.key))


def collect_semantic_samples(dataset: common.SemanticDataset) -> list[Sample]:
    result: list[Sample] = []
    split_rank = {split: index for index, split in enumerate(SPLITS)}
    ignore_label = common.source_ignore_label_for_edit(dataset, 255)
    for split, paths in dataset.splits.items():
        images_by_stem = {
            str(path.relative_to(paths.images).with_suffix("")).replace("\\", "/"): path.resolve()
            for path in common._iter_files(paths.images, IMAGE_EXTENSIONS)
        }
        masks_by_stem = {
            str(path.relative_to(paths.masks).with_suffix("")).replace("\\", "/"): path.resolve()
            for path in common._iter_files(paths.masks, common.MASK_EXTENSIONS)
        }
        for stem, image_path in sorted(images_by_stem.items()):
            mask_path = masks_by_stem.get(stem)
            if mask_path is None:
                raise ValueError(f"{split} 中图片缺少对应 mask: {stem}")
            mask = common.read_mask(mask_path)
            pixel_counts: Counter[int] = Counter()
            for raw_label, count in common._mask_label_counts(mask, dataset):
                if ignore_label is not None and raw_label == ignore_label:
                    continue
                class_id = dataset.label_to_class.get(raw_label)
                if class_id is None:
                    raise ValueError(
                        f"mask {mask_path} 包含 classes 未映射的 label: {raw_label!r}"
                    )
                pixel_counts[class_id] += count
            result.append(
                Sample(
                    source_split=split,
                    logical_stem=stem,
                    image_path=image_path,
                    annotation_path=mask_path,
                    label_lines=(),
                    amount_counts=dict(pixel_counts),
                )
            )
    return sorted(
        result,
        key=lambda sample: (split_rank.get(sample.source_split, 99), sample.key),
    )


def deduplicate_samples(
    samples: Sequence[Sample],
    *,
    merge_yolo_annotations: bool = False,
) -> tuple[list[Sample], list[dict[str, str]]]:
    """Collapse byte-identical images and optionally merge their YOLO rows."""
    seen_index: dict[str, int] = {}
    kept: list[Sample] = []
    duplicates: list[dict[str, str]] = []
    for sample in tqdm(samples, desc="检查跨 split 重复图片", unit="img"):
        digest = file_digest(sample.image_path)
        previous_index = seen_index.get(digest)
        if previous_index is None:
            seen_index[digest] = len(kept)
            kept.append(sample)
            continue
        previous = kept[previous_index]
        if previous.label_lines or sample.label_lines:
            left: Any = tuple(sorted(previous.label_lines))
            right: Any = tuple(sorted(sample.label_lines))
        else:
            left = (
                file_digest(previous.annotation_path)
                if previous.annotation_path is not None
                else None
            )
            right = (
                file_digest(sample.annotation_path)
                if sample.annotation_path is not None
                else None
            )
        if left != right:
            if not merge_yolo_annotations:
                raise ValueError(
                    "检测到内容相同且标注不同的图片: "
                    f"{previous.image_path} <-> {sample.image_path}"
                )
            merged_lines = tuple(
                dict.fromkeys((*previous.label_lines, *sample.label_lines))
            )
            merged_counts: Counter[int] = Counter(
                int(line.split()[0]) for line in merged_lines
            )
            kept[previous_index] = replace(
                previous,
                label_lines=merged_lines,
                amount_counts=dict(merged_counts),
            )
        duplicates.append(
            {
                "kept": str(previous.image_path),
                "removed": str(sample.image_path),
                "source_split": sample.source_split,
                "status": "merged-yolo-annotations" if left != right else "duplicate-removed",
            }
        )
    return kept, duplicates


def _desired_class_counts(
    class_frequency: Counter[int], ratios: dict[str, float]
) -> dict[int, dict[str, float]]:
    return {
        class_id: {
            split: frequency * ratios[split]
            for split in SPLITS
        }
        for class_id, frequency in class_frequency.items()
    }


def stratified_assign(
    samples: Sequence[Sample],
    target_sizes: dict[str, int],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Greedy rarest-first multilabel stratification with exact split sizes."""
    rng = random.Random(seed)
    by_key = {sample.key: sample for sample in samples}
    unassigned = set(by_key)
    remaining_by_class: dict[int, set[str]] = defaultdict(set)
    class_frequency: Counter[int] = Counter()
    for sample in samples:
        class_frequency.update(sample.class_ids)
        for class_id in sample.class_ids:
            remaining_by_class[class_id].add(sample.key)

    desired = _desired_class_counts(class_frequency, ratios)
    actual: dict[int, Counter[str]] = defaultdict(Counter)
    remaining_capacity = dict(target_sizes)
    assignments: dict[str, str] = {}
    active_splits = [split for split in SPLITS if target_sizes[split] > 0]
    coverage_classes = {
        class_id
        for class_id, frequency in class_frequency.items()
        if frequency >= len(active_splits)
    }
    jitter = {
        (sample.key, split): rng.random()
        for sample in samples
        for split in SPLITS
    }
    sample_jitter = {sample.key: rng.random() for sample in samples}

    def choose_sample() -> Sample:
        present_classes = [
            class_id for class_id, keys in remaining_by_class.items() if keys
        ]
        if not present_classes:
            return by_key[min(unassigned)]
        rarest = min(
            present_classes,
            key=lambda class_id: (
                len(remaining_by_class[class_id]),
                class_frequency[class_id],
                class_id,
            ),
        )
        candidates = [by_key[key] for key in sorted(remaining_by_class[rarest])]

        def priority(sample: Sample) -> tuple[float, int, float, str]:
            scarcity = sum(1.0 / class_frequency[class_id] for class_id in sample.class_ids)
            uncovered = sum(
                1
                for class_id in sample.class_ids & coverage_classes
                if any(actual[class_id][split] == 0 for split in active_splits)
            )
            return uncovered, scarcity, sample_jitter[sample.key], sample.key

        return max(candidates, key=priority)

    while unassigned:
        sample = choose_sample()
        candidate_splits = [split for split in active_splits if remaining_capacity[split] > 0]
        if not candidate_splits:
            raise RuntimeError("内部划分错误: split 容量提前耗尽")

        def split_score(split: str) -> tuple[float, float, float, float]:
            coverage_gain = sum(
                (1.0 + 1.0 / class_frequency[class_id])
                for class_id in sample.class_ids & coverage_classes
                if actual[class_id][split] == 0
            )
            deficit_gain = sum(
                max(desired[class_id][split] - actual[class_id][split], 0.0)
                / math.sqrt(class_frequency[class_id])
                for class_id in sample.class_ids
            )
            capacity_ratio = remaining_capacity[split] / max(target_sizes[split], 1)
            return coverage_gain, deficit_gain, capacity_ratio, jitter[(sample.key, split)]

        chosen_split = max(candidate_splits, key=split_score)
        assignments[sample.key] = chosen_split
        remaining_capacity[chosen_split] -= 1
        unassigned.remove(sample.key)
        for class_id in sample.class_ids:
            actual[class_id][chosen_split] += 1
            remaining_by_class[class_id].discard(sample.key)
    return assignments


def _class_split_image_counts(
    samples: Sequence[Sample], assignments: dict[str, str]
) -> dict[int, dict[str, int]]:
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        split = assignments[sample.key]
        for class_id in sample.class_ids:
            counts[class_id][split] += 1
    return {
        class_id: {split: split_counts[split] for split in SPLITS}
        for class_id, split_counts in counts.items()
    }


def _distribution_cost(
    class_id: int,
    counts: dict[int, dict[str, int]],
    desired: dict[int, dict[str, float]],
    frequency: Counter[int],
) -> float:
    return sum(
        (counts[class_id][split] - desired[class_id][split]) ** 2
        / max(frequency[class_id], 1)
        for split in SPLITS
    )


def repair_coverage(
    samples: Sequence[Sample],
    assignments: dict[str, str],
    ratios: dict[str, float],
) -> None:
    """Swap equal-size samples to fill feasible missing class/split cells."""
    by_split: dict[str, list[Sample]] = {split: [] for split in SPLITS}
    class_frequency: Counter[int] = Counter()
    for sample in samples:
        by_split[assignments[sample.key]].append(sample)
        class_frequency.update(sample.class_ids)
    active = [split for split in SPLITS if by_split[split]]
    eligible = {
        class_id for class_id, frequency in class_frequency.items() if frequency >= len(active)
    }
    desired = _desired_class_counts(class_frequency, ratios)

    for _ in range(max(1, len(eligible) * len(active))):
        counts = _class_split_image_counts(samples, assignments)
        missing = [
            (class_id, split)
            for class_id in sorted(eligible, key=lambda item: (class_frequency[item], item))
            for split in active
            if counts[class_id][split] == 0
        ]
        if not missing:
            return
        changed = False
        for class_id, destination in missing:
            best: tuple[float, Sample, Sample] | None = None
            donors = [
                sample
                for split in active
                if split != destination
                for sample in by_split[split]
                if class_id in sample.class_ids
            ]
            for donor in donors:
                source = assignments[donor.key]
                for receiver in by_split[destination]:
                    if class_id in receiver.class_ids:
                        continue
                    source_safe = all(
                        item in receiver.class_ids or counts[item][source] > 1
                        for item in donor.class_ids & eligible
                    )
                    destination_safe = all(
                        item in donor.class_ids or counts[item][destination] > 1
                        for item in receiver.class_ids & eligible
                    )
                    if not source_safe or not destination_safe:
                        continue
                    affected = donor.class_ids | receiver.class_ids
                    before = sum(
                        _distribution_cost(item, counts, desired, class_frequency)
                        for item in affected
                    )
                    for item in donor.class_ids - receiver.class_ids:
                        counts[item][source] -= 1
                        counts[item][destination] += 1
                    for item in receiver.class_ids - donor.class_ids:
                        counts[item][destination] -= 1
                        counts[item][source] += 1
                    after = sum(
                        _distribution_cost(item, counts, desired, class_frequency)
                        for item in affected
                    )
                    for item in donor.class_ids - receiver.class_ids:
                        counts[item][source] += 1
                        counts[item][destination] -= 1
                    for item in receiver.class_ids - donor.class_ids:
                        counts[item][destination] += 1
                        counts[item][source] -= 1
                    candidate = (after - before, donor, receiver)
                    if best is None or (candidate[0], donor.key, receiver.key) < (
                        best[0], best[1].key, best[2].key
                    ):
                        best = candidate
            if best is None:
                continue
            _, donor, receiver = best
            source = assignments[donor.key]
            assignments[donor.key] = destination
            assignments[receiver.key] = source
            by_split[source].remove(donor)
            by_split[source].append(receiver)
            by_split[destination].remove(receiver)
            by_split[destination].append(donor)
            changed = True
            break
        if not changed:
            return


def _amount_counts(
    samples: Sequence[Sample], assignments: dict[str, str]
) -> dict[int, dict[str, int]]:
    result: dict[int, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        split = assignments[sample.key]
        for class_id, count in sample.amount_counts.items():
            result[class_id][split] += count
    return {
        class_id: {split: counts[split] for split in SPLITS}
        for class_id, counts in result.items()
    }


def _finalize_plan(
    *,
    dataset: Any,
    samples: list[Sample],
    normalized_ratios: dict[str, float],
    seed: int,
    deduplicate: bool,
    format_kind: str,
    amount_unit: str,
) -> SplitPlan:
    if not samples:
        raise ValueError("数据集中没有可划分的图片")
    duplicate_rows: list[dict[str, str]] = []
    if deduplicate:
        samples, duplicate_rows = deduplicate_samples(
            samples,
            merge_yolo_annotations=format_kind in yolo.YOLO_KINDS,
        )

    target_sizes = calculate_target_sizes(len(samples), normalized_ratios)
    assignments = stratified_assign(samples, target_sizes, normalized_ratios, seed)
    repair_coverage(samples, assignments, normalized_ratios)
    image_counts = _class_split_image_counts(samples, assignments)
    amount_counts = _amount_counts(samples, assignments)
    active = [split for split in SPLITS if target_sizes[split] > 0]
    total_by_class = Counter(
        class_id for sample in samples for class_id in sample.class_ids
    )
    missing = [
        {
            "class_id": class_id,
            "class_name": dataset.class_names.get(class_id, f"class_{class_id}"),
            "split": split,
            "total_images": total_by_class[class_id],
            "coverage_possible_by_count": total_by_class[class_id] >= len(active),
        }
        for class_id in sorted(total_by_class)
        for split in active
        if image_counts[class_id][split] == 0
    ]
    return SplitPlan(
        dataset=dataset,
        format_kind=format_kind,
        amount_unit=amount_unit,
        samples=samples,
        assignments=assignments,
        ratios=normalized_ratios,
        target_sizes=target_sizes,
        duplicate_rows=duplicate_rows,
        class_image_counts=image_counts,
        class_amount_counts=amount_counts,
        missing_coverage=missing,
        seed=seed,
    )


def build_plan(
    source: Path,
    *,
    ratios: Sequence[float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    deduplicate: bool = False,
    forced_kind: str | None = None,
) -> SplitPlan:
    normalized_ratios = _validate_ratios(ratios)
    config_path = common._find_config(source)
    discovered_kind = inspect_dataset(config_path).kind
    format_kind = forced_kind or discovered_kind
    if format_kind == "semantic_mask":
        dataset = common.resolve_dataset(source)
        source_ignore = common.source_ignore_label_for_edit(dataset, 255)
        analysis = common.analyze_dataset(dataset, ignore_label=source_ignore)
        fatal = [
            issue
            for issue in analysis.issues
            if issue.issue_type in SEMANTIC_FATAL_ISSUES
        ]
        if fatal:
            preview = "; ".join(issue.message for issue in fatal[:8])
            raise ValueError(f"语义分割数据集校验失败: {preview}")
        return _finalize_plan(
            dataset=dataset,
            samples=collect_semantic_samples(dataset),
            normalized_ratios=normalized_ratios,
            seed=seed,
            deduplicate=deduplicate,
            format_kind="semantic_mask",
            amount_unit="pixels",
        )

    if format_kind not in yolo.YOLO_KINDS:
        raise ValueError(
            "支持的数据格式为 YOLO Detection、YOLO Instance Segmentation "
            "和 PNG Semantic Segmentation"
        )
    dataset = yolo.resolve_dataset(source, forced_kind=format_kind)
    analysis = yolo.analyze_dataset(dataset)
    fatal = [
        issue for issue in analysis.shared.issues if issue.issue_type in FATAL_ISSUES
    ]
    if fatal:
        preview = "; ".join(issue.message for issue in fatal[:8])
        raise ValueError(f"YOLO 数据集校验失败: {preview}")
    return _finalize_plan(
        dataset=dataset,
        samples=collect_samples(dataset),
        normalized_ratios=normalized_ratios,
        seed=seed,
        deduplicate=deduplicate,
        format_kind=format_kind,
        amount_unit="objects",
    )


def _report_dict(plan: SplitPlan, output: Path | None = None) -> dict[str, Any]:
    classes: list[dict[str, Any]] = []
    for class_id in sorted(plan.dataset.class_names):
        images = plan.class_image_counts.get(class_id, {split: 0 for split in SPLITS})
        amounts = plan.class_amount_counts.get(
            class_id, {split: 0 for split in SPLITS}
        )
        classes.append(
            {
                "id": class_id,
                "name": plan.dataset.class_names[class_id],
                "images": {split: images.get(split, 0) for split in SPLITS},
                "amounts": {split: amounts.get(split, 0) for split in SPLITS},
                "total_images": sum(images.values()),
                "total_amount": sum(amounts.values()),
            }
        )
    tasks = {
        "yolo_detection": "detect",
        "yolo_instance": "segment",
        "semantic_mask": "semantic_segmentation",
    }
    return {
        "source": str(plan.dataset.root),
        "output": str(output) if output is not None else None,
        "task": tasks[plan.format_kind],
        "format": plan.format_kind,
        "amount_unit": plan.amount_unit,
        "seed": plan.seed,
        "ratios": plan.ratios,
        "images": plan.target_sizes,
        "total_images": len(plan.samples),
        "removed_duplicate_images": len(plan.duplicate_rows),
        "merged_yolo_annotation_sources": sum(
            row.get("status") == "merged-yolo-annotations"
            for row in plan.duplicate_rows
        ),
        "classes": classes,
        "missing_coverage": plan.missing_coverage,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    amount_label = "像素" if report["amount_unit"] == "pixels" else "对象"
    lines = [
        "# 数据集 split 重划分报告",
        "",
        f"- 图片总数: {report['total_images']}",
        f"- 重复图片移除数: {report['removed_duplicate_images']}",
        f"- YOLO 标注合并来源数: {report['merged_yolo_annotation_sources']}",
        "- train/val/test: "
        f"{report['images']['train']} / {report['images']['val']} / "
        f"{report['images']['test']}",
        f"- 缺失的 类别×split: {len(report['missing_coverage'])}",
        "",
        f"| ID | 类别 | train 图片/{amount_label} | val 图片/{amount_label} | "
        f"test 图片/{amount_label} |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in report["classes"]:
        cells = [
            f"{row['images'][split]}/{row['amounts'][split]}" for split in SPLITS
        ]
        lines.append(
            f"| {row['id']} | {row['name']} | {cells[0]} | {cells[1]} | {cells[2]} |"
        )
    if report["missing_coverage"]:
        lines.extend(["", "## 仍缺失的类别覆盖", ""])
        for item in report["missing_coverage"]:
            reason = (
                "样本数量允许，组合或 split 容量限制了当前划分"
                if item["coverage_possible_by_count"]
                else "该类别的图片总数少于有效 split 数"
            )
            lines.append(
                f"- `{item['split']}`: ID {item['class_id']} {item['class_name']}，"
                f"共 {item['total_images']} 张；{reason}。"
            )
    return "\n".join(lines) + "\n"


def print_summary(plan: SplitPlan) -> None:
    merged_sources = sum(
        row.get("status") == "merged-yolo-annotations" for row in plan.duplicate_rows
    )
    print(f"\n数据集: {plan.dataset.root}")
    print(f"图片: {len(plan.samples)}，跨 split 重复移除: {len(plan.duplicate_rows)}")
    if merged_sources:
        print(f"同图不同 YOLO 标注已自动合并: {merged_sources} 个来源")
    print(
        "划分: "
        + "，".join(f"{split}={plan.target_sizes[split]}" for split in SPLITS)
    )
    print(f"类别×split 缺失: {len(plan.missing_coverage)}")
    if plan.missing_coverage:
        for item in plan.missing_coverage[:12]:
            print(
                f"  {item['split']}: ID {item['class_id']} {item['class_name']} "
                f"(共 {item['total_images']} 张)"
            )


def write_plan(
    plan: SplitPlan,
    output: Path,
    *,
    image_mode: str = "reflink",
    clean: bool = False,
) -> Path:
    output = validate_output_location(output, [plan.dataset.root])
    report = _report_dict(plan, output)
    used_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    mapping_rows: list[dict[str, str]] = []
    annotation_dir_name = "masks" if plan.format_kind == "semantic_mask" else "labels"
    with staged_output(output, clean=clean) as stage:
        for split in SPLITS:
            (stage / "images" / split).mkdir(parents=True, exist_ok=True)
            (stage / annotation_dir_name / split).mkdir(parents=True, exist_ok=True)
        for sample in tqdm(plan.samples, desc="生成重新划分的数据集", unit="img"):
            split = plan.assignments[sample.key]
            output_stem = allocate_flat_sample_stem(
                sample.logical_stem,
                used_by_split[split],
            )
            image_target = (
                stage
                / "images"
                / split
                / f"{output_stem}{sample.image_path.suffix.lower()}"
            )
            common._transfer_file(sample.image_path, image_target, image_mode)
            mapping_row = {
                "source_split": sample.source_split,
                "source_image": str(sample.image_path),
                "output_split": split,
                "output_image": str(Path("images") / split / image_target.name),
            }
            if plan.format_kind == "semantic_mask":
                if sample.annotation_path is None:
                    raise RuntimeError(f"语义样本缺少 mask: {sample.image_path}")
                mask_target = (
                    stage
                    / "masks"
                    / split
                    / f"{output_stem}{sample.annotation_path.suffix.lower()}"
                )
                common._transfer_file(sample.annotation_path, mask_target, image_mode)
                mapping_row["source_mask"] = str(sample.annotation_path)
                mapping_row["output_mask"] = str(
                    Path("masks") / split / mask_target.name
                )
            else:
                label_target = stage / "labels" / split / f"{output_stem}.txt"
                label_target.write_text(
                    "\n".join(sample.label_lines)
                    + ("\n" if sample.label_lines else ""),
                    encoding="utf-8",
                )
                mapping_row["source_label"] = (
                    str(sample.annotation_path)
                    if sample.annotation_path is not None
                    else ""
                )
                mapping_row["output_label"] = str(
                    Path("labels") / split / label_target.name
                )
            mapping_rows.append(mapping_row)

        if plan.format_kind == "semantic_mask":
            raw_classes = plan.dataset.config.get(
                "classes", plan.dataset.config.get("names")
            )
            data: dict[str, Any] = {
                "path": str(output),
                "task": "semantic_segmentation",
                "classes": raw_classes,
            }
            for split in SPLITS:
                data[split] = {
                    "images": f"images/{split}",
                    "masks": f"masks/{split}",
                }
            if "ignore_label" in plan.dataset.config:
                data["ignore_label"] = plan.dataset.config["ignore_label"]
            elif "ignore_index" in plan.dataset.config:
                data["ignore_index"] = plan.dataset.config["ignore_index"]
        else:
            data = {
                "path": str(output),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "task": (
                    "segment" if plan.format_kind == "yolo_instance" else "detect"
                ),
                "nc": len(plan.dataset.class_names),
                "names": plan.dataset.class_names,
            }
        (stage / "data.yaml").write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        max_class_id = max(plan.dataset.class_names, default=-1)
        class_lines = [
            plan.dataset.class_names.get(class_id, f"class_{class_id}")
            for class_id in range(max_class_id + 1)
        ]
        (stage / "classes.txt").write_text("\n".join(class_lines) + "\n", encoding="utf-8")
        (stage / "split_mapping.json").write_text(
            json.dumps(mapping_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "duplicate_images.json").write_text(
            json.dumps(plan.duplicate_rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "split_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (stage / "split_report.md").write_text(_render_markdown(report), encoding="utf-8")
        (stage / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return output


def _choose_source(search_root: Path) -> Path | None:
    candidates = [
        candidate
        for candidate in detect_datasets(
            search_root,
            kinds=yolo.YOLO_KINDS | {"semantic_mask"},
        )
        if candidate.kind in yolo.YOLO_KINDS | {"semantic_mask"}
    ]
    print(f"\n在 {search_root} 中找到 {len(candidates)} 个可重划分数据集:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. {KIND_LABELS[candidate.kind]:<18} "
            f"图片 {candidate.image_count:>6}  类别 {candidate.class_count:>4}  {candidate.path}"
        )
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
            selected = candidates[int(raw) - 1]
            return selected.config_path or selected.path
        print("请输入有效序号、p 或 q。")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="源 data.yaml 或数据集目录")
    parser.add_argument("--out", type=Path, help="输出数据集目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描根目录")
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test 比例，默认 0.8 0.1 0.1",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--format",
        choices=("detect", "segment", "semantic"),
        help="data.yaml 格式识别有歧义时显式指定格式",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "reflink", "hardlink", "symlink"),
        default="reflink",
        help="输出图片方式，reflink 会在支持时节省空间",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="按图片内容摘要排重；同图 YOLO 标注会自动合并",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅分析和预览划分")
    parser.add_argument("--clean", action="store_true", help="安全替换已有输出目录")
    parser.add_argument("--yes", action="store_true", help="直接执行写入")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    interactive = args.src is None
    source = args.src
    if source is None:
        search_root = (args.datasets or auto_datasets_root()).expanduser().resolve()
        source = _choose_source(search_root)
        if source is None:
            return 0
    source = source.expanduser().resolve()
    forced_kind = {
        "detect": "yolo_detection",
        "segment": "yolo_instance",
        "semantic": "semantic_mask",
        None: None,
    }[args.format]
    plan = build_plan(
        source,
        ratios=args.ratios,
        seed=args.seed,
        deduplicate=args.deduplicate,
        forced_kind=forced_kind,
    )
    print_summary(plan)
    if args.out is not None:
        validate_output_location(args.out, [plan.dataset.root])
    if args.dry_run:
        print("DRY-RUN 完成。")
        return 0

    default_output = plan.dataset.root.parent / f"{plan.dataset.root.name}__balanced_splits"
    output = args.out.expanduser() if args.out is not None else default_output
    if interactive and args.out is None:
        chosen = prompt_path("输出目录", default_output)
        if chosen is None:
            return 0
        output = chosen
    if not args.yes:
        try:
            answer = input(f"生成重新划分的数据集到 {output}？[y/N]: ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if answer not in {"y", "yes", "是"}:
            print("已取消。")
            return 0
    result = write_plan(plan, output, image_mode=args.image_mode, clean=args.clean)
    print(f"完成: {result}")
    print(f"报告: {result / 'split_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
