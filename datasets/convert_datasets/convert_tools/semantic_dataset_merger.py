#!/usr/bin/env python3
"""交互式合并多个 PNG 语义分割数据集并统一类别 ID。

输入采用 data.yaml + images/{split} + masks/{split}，或
{split}/images + {split}/masks。程序先建立跨来源临时类别 ID，循环收集删除、
合并、重命名操作；用户输入 done 并确认后，才会一次性重写全部 mask 像素和最终
data.yaml。
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Iterator

import yaml
from PIL import Image

try:
    from . import semantic_class_editor as common
    from .dataset_discovery import (
        IMAGE_EXTENSIONS,
        KIND_LABELS,
        auto_datasets_root,
    )
    from .dataset_detector import detect_datasets
    from .output_naming import allocate_flat_sample_stem, default_output_dir
    from .progress import tqdm
    from .dataset_transaction import (
        file_digest,
        metadata_fingerprint,
        staged_output,
        validate_distinct_sources,
        validate_output_location,
    )
except ImportError:
    import semantic_class_editor as common  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        IMAGE_EXTENSIONS,
        KIND_LABELS,
        auto_datasets_root,
    )
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from output_naming import (  # type: ignore[no-redef]
        allocate_flat_sample_stem,
        default_output_dir,
    )
    from progress import tqdm  # type: ignore[no-redef]
    from dataset_transaction import (  # type: ignore[no-redef]
        file_digest,
        metadata_fingerprint,
        staged_output,
        validate_distinct_sources,
        validate_output_location,
    )


@dataclass
class MergeSource:
    key: str
    source_id: str
    fingerprint: str
    ignore_label: int | None
    dataset: common.SemanticDataset
    analysis: common.DatasetAnalysis
    old_to_provisional: dict[int, int]


@dataclass
class MergeInventory:
    sources: list[MergeSource]
    class_names: dict[int, str]
    base_names: dict[int, str]
    analysis: common.DatasetAnalysis
    image_count: int
    mask_count: int
    ignore_label: int
    taxonomy_mode: str = "namespace"


@dataclass(frozen=True)
class _MaskTask:
    source: MergeSource
    split: str
    stem: str
    image_path: Path
    mask_path: Path
    resolved: common.ResolvedPlan
    can_fast_copy: bool
    output_ignore_label: int
    unknown_policy: str
    inspect_all_ignore: bool
    build_signature: bool


@dataclass(frozen=True)
class _PreparedMask:
    task: _MaskTask
    remapped: Any | None
    all_ignore: bool
    signature: str


def _has_canonical_integer_encoding(dataset: common.SemanticDataset) -> bool:
    return dataset.label_kind == "integer" and all(
        dataset.class_labels.get(class_id) == (class_id,)
        for class_id in dataset.class_names
    )


def _prepare_mask_task(task: _MaskTask) -> _PreparedMask:
    needs_decode = (
        not task.can_fast_copy
        or task.inspect_all_ignore
        or task.build_signature
    )
    remapped = None
    if needs_decode:
        mask = common.read_mask(task.mask_path)
        remapped = common.remap_mask(
            mask,
            task.resolved,
            ignore_label=task.output_ignore_label,
            unknown_policy=task.unknown_policy,
            source_ignore_label=task.source.ignore_label,
            dataset=task.source.dataset,
        )
    all_ignore = bool(
        task.inspect_all_ignore
        and remapped is not None
        and remapped.size
        and (remapped == task.output_ignore_label).all()
    )
    signature = (
        _mask_signature(remapped)
        if task.build_signature and remapped is not None
        else ""
    )
    return _PreparedMask(
        task=task,
        remapped=remapped,
        all_ignore=all_ignore,
        signature=signature,
    )


def _prepare_masks_ordered(
    executor: ThreadPoolExecutor,
    tasks: Iterable[_MaskTask],
    *,
    max_pending: int,
) -> Iterator[_PreparedMask]:
    """Yield parallel conversion results in input order with bounded memory."""
    task_iterator = iter(tasks)
    pending = deque()
    for _ in range(max_pending):
        try:
            task = next(task_iterator)
        except StopIteration:
            break
        pending.append(executor.submit(_prepare_mask_task, task))
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            task = next(task_iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(_prepare_mask_task, task))


def _source_metadata(
    dataset: common.SemanticDataset, source_ignore_label: int | None
) -> tuple[str, str]:
    fingerprint = metadata_fingerprint(
        {
            "format": "semantic_mask",
            "classes": dataset.class_names,
            "class_labels": {
                class_id: [list(label) if isinstance(label, tuple) else label for label in labels]
                for class_id, labels in dataset.class_labels.items()
            },
            "label_kind": dataset.label_kind,
            "source_ignore_label": source_ignore_label,
            "config": dataset.config_path.name,
        }
    )
    source_id = "src_" + metadata_fingerprint(
        {"root": str(dataset.root), "config": str(dataset.config_path)}
    )[:16]
    return source_id, fingerprint


def _load_taxonomy_mapping(path: Path | None) -> dict[tuple[str, int], str]:
    if path is None:
        raise ValueError("taxonomy=mapping-file 需要 --mapping-file")
    data = common.read_yaml(path)
    rows = data.get("mappings", data)
    result: dict[tuple[str, int], str] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("mapping-file 的 mappings 项需要是 object")
            source = str(row.get("source", row.get("source_id", ""))).strip()
            old_id = int(row.get("old_id", row.get("id")))
            target = str(row.get("target", row.get("name", ""))).strip()
            if not source or not target:
                raise ValueError(f"mapping-file 项缺少 source/target: {row}")
            result[(source, old_id)] = target
    elif isinstance(rows, dict):
        for source, values in rows.items():
            if source in {"version", "taxonomy"}:
                continue
            if not isinstance(values, dict):
                raise ValueError(f"mapping-file 来源 {source} 的值需要是 ID 映射")
            for old_id, target in values.items():
                result[(str(source), int(old_id))] = str(target).strip()
    else:
        raise ValueError("mapping-file 需要 mappings 列表或 source -> ID 映射")
    return result


def _source_ignore_for(
    dataset: common.SemanticDataset,
    output_ignore_label: int,
    overrides: dict[str, int | None],
    key: str,
) -> int | None:
    selectors = (key, str(dataset.root), dataset.root.name)
    for selector in selectors:
        if selector in overrides:
            return overrides[selector]
    if dataset.ignore_label is not None:
        if dataset.ignore_label in dataset.label_to_class:
            raise ValueError(
                f"{dataset.root} 同时把 {dataset.ignore_label} 声明为 ignore_label "
                "和原始 mask label；"
                "请修正配置或使用来源 ignore override"
            )
        return dataset.ignore_label
    # Legacy datasets commonly use 255 without declaring it.  A declared class
    # with that ID has priority and remains a real class.
    return None if output_ignore_label in dataset.label_to_class else output_ignore_label


def build_inventory(
    source_paths: list[Path],
    *,
    ignore_label: int = 255,
    source_ignore_labels: dict[str, int | None] | None = None,
    taxonomy_mode: str = "namespace",
    mapping_file: Path | None = None,
    use_first_source_names: bool = False,
) -> MergeInventory:
    if len(source_paths) < 2:
        raise ValueError("合并至少需要两个 PNG 语义分割数据集")
    if not 0 <= ignore_label <= 65535:
        raise ValueError("ignore_label 需要在 0..65535 范围内")
    if taxonomy_mode not in {"namespace", "strict", "union-by-name", "mapping-file"}:
        raise ValueError(f"未知 taxonomy_mode: {taxonomy_mode}")
    overrides = source_ignore_labels or {}
    datasets = [common.resolve_dataset(path.expanduser().resolve()) for path in source_paths]
    validate_distinct_sources(
        [(dataset.root, dataset.config_path) for dataset in datasets]
    )
    if taxonomy_mode == "strict":
        expected = datasets[0].class_names
        mismatches = [
            str(dataset.root)
            for dataset in datasets[1:]
            if dataset.class_names != expected
        ]
        if mismatches:
            raise ValueError(
                "strict taxonomy 要求所有来源使用完全一致的 ID/类别名；"
                f"不一致来源: {mismatches}"
            )
    explicit_mapping = _load_taxonomy_mapping(mapping_file) if taxonomy_mode == "mapping-file" else {}

    sources: list[MergeSource] = []
    class_names: dict[int, str] = {}
    base_names: dict[int, str] = {}
    raw_names: dict[int, Any] = {}
    class_stats: dict[int, common.ClassStats] = {}
    observed_ids: set[int] = set()
    unknown_stats: dict[common.MaskLabel, common.ClassStats] = {}
    null_like_ids: set[int] = set()
    image_count = 0
    mask_count = 0
    next_id = 0
    taxonomy_ids: dict[tuple[str, Any], int] = {}

    for index, dataset in enumerate(
        tqdm(datasets, desc="分析合并来源", unit="数据集")
    ):
        key = f"dataset{index + 1}"
        source_ignore_label = _source_ignore_for(
            dataset, ignore_label, overrides, key
        )
        analysis = common.analyze_dataset(
            dataset, ignore_label=source_ignore_label
        )
        invalid = [
            issue
            for issue in analysis.issues
            if issue.issue_type
            in {
                "invalid_mask",
                "missing_image_dir",
                "missing_mask_dir",
                "missing_mask",
                "missing_image",
                "invalid_image",
                "size_mismatch",
                "duplicate_image_stem",
                "duplicate_mask_stem",
            }
        ]
        if invalid:
            first = invalid[0]
            raise ValueError(
                f"{dataset.root} 存在 {len(invalid)} 个图片/mask 问题；"
                f"首个问题: {first.path}: {first.message}"
            )

        source_id, fingerprint = _source_metadata(dataset, source_ignore_label)
        old_to_provisional: dict[int, int] = {}
        all_old_ids = sorted(dataset.class_names)
        for old_id in all_old_ids:
            base_name = dataset.class_names[old_id]
            raw_name = dataset.raw_class_names.get(old_id)
            source_stats = analysis.class_stats[old_id]
            if taxonomy_mode == "namespace":
                taxonomy_key: tuple[str, Any] = (source_id, old_id)
                output_name = (
                    base_name
                    if use_first_source_names and index == 0
                    else (f"{key}_{base_name}" if base_name else "")
                )
            elif taxonomy_mode == "strict":
                if old_id not in dataset.class_names:
                    raise ValueError(
                        f"strict taxonomy 不接受 YAML 未定义类别: {dataset.root} ID={old_id}"
                    )
                taxonomy_key = ("strict", old_id)
                output_name = base_name
            elif taxonomy_mode == "union-by-name":
                normalized = common.normalize_name(base_name)
                taxonomy_key = (
                    "name", normalized if normalized else f"{source_id}:{old_id}"
                )
                output_name = base_name
            else:
                selectors = (source_id, key, dataset.root.name, str(dataset.root))
                target = next(
                    (
                        explicit_mapping[(selector, old_id)]
                        for selector in selectors
                        if (selector, old_id) in explicit_mapping
                    ),
                    "",
                )
                if not target:
                    raise ValueError(
                        f"mapping-file 缺少来源 {key}/{source_id} 的类别 ID {old_id}"
                    )
                taxonomy_key = ("mapped", common.normalize_name(target))
                output_name = target
            provisional_id = taxonomy_ids.get(taxonomy_key, -1)
            if provisional_id < 0:
                provisional_id = next_id
                next_id += 1
                taxonomy_ids[taxonomy_key] = provisional_id
                class_names[provisional_id] = output_name
                base_names[provisional_id] = output_name if taxonomy_mode == "mapping-file" else base_name
                raw_names[provisional_id] = raw_name
                class_stats[provisional_id] = common.ClassStats()
            old_to_provisional[old_id] = provisional_id
            class_stats[provisional_id].image_count += source_stats.image_count
            class_stats[provisional_id].pixel_count += source_stats.pixel_count
            if source_stats.pixel_count:
                observed_ids.add(provisional_id)
            if common.is_null_like_name(base_name, raw_name):
                null_like_ids.add(provisional_id)

        for label, stats in analysis.unknown_stats.items():
            shared = unknown_stats.setdefault(label, common.ClassStats())
            shared.image_count += stats.image_count
            shared.pixel_count += stats.pixel_count

        sources.append(
            MergeSource(
                key=key,
                source_id=source_id,
                fingerprint=fingerprint,
                ignore_label=source_ignore_label,
                dataset=dataset,
                analysis=analysis,
                old_to_provisional=old_to_provisional,
            )
        )
        image_count += analysis.image_count
        mask_count += analysis.mask_count

    unused_ids = set(class_names) - observed_ids
    exact_groups, similar_groups = common._name_candidate_groups(class_names)
    issues: list[common.AnalysisIssue] = []
    if unused_ids:
        issues.append(
            common.AnalysisIssue(
                "unused_yaml_ids",
                f"类别表中存在 mask 未使用的临时 ID: {sorted(unused_ids)}",
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
    if unknown_stats:
        issues.append(
            common.AnalysisIssue(
                "unknown_mask_ids",
                "mask 中存在 classes 未映射的 label",
                details={"labels": common._sorted_labels(unknown_stats)},
            )
        )
    if exact_groups or similar_groups:
        issues.append(
            common.AnalysisIssue(
                "merge_candidates",
                "检测到跨来源同名/近似类别候选",
                details={"groups": exact_groups + similar_groups},
            )
        )
    shared_analysis = common.DatasetAnalysis(
        class_stats=class_stats,
        unknown_stats={
            label: unknown_stats[label]
            for label in common._sorted_labels(unknown_stats)
        },
        observed_ids=observed_ids,
        unknown_ids=set(unknown_stats),
        unused_ids=unused_ids,
        null_like_ids=null_like_ids,
        exact_name_groups=exact_groups,
        similar_name_groups=similar_groups,
        issues=issues,
        mask_count=mask_count,
        image_count=image_count,
    )
    return MergeInventory(
        sources=sources,
        class_names=class_names,
        base_names=base_names,
        analysis=shared_analysis,
        image_count=image_count,
        mask_count=mask_count,
        ignore_label=ignore_label,
        taxonomy_mode=taxonomy_mode,
    )


def print_inventory(inventory: MergeInventory) -> None:
    print(
        f"\n格式 PNG 语义分割，来源 {len(inventory.sources)}，"
        f"图片 {inventory.image_count}，mask {inventory.mask_count}"
    )
    for source in inventory.sources:
        print(f"  {source.key}: {source.dataset.root}")
    print(
        f"\n  {'临时ID':>6}  {'来源':<10}{'原ID':>8}  "
        f"{'原类别名':<28}{'图片数':>10}{'像素数':>16}"
    )
    print("  " + "-" * 88)
    for source in inventory.sources:
        reverse = {
            provisional_id: old_id
            for old_id, provisional_id in source.old_to_provisional.items()
        }
        for provisional_id in sorted(reverse):
            old_id = reverse[provisional_id]
            stats = inventory.analysis.class_stats[provisional_id]
            print(
                f"  {provisional_id:>6}  {source.key:<10}{old_id:>8}  "
                f"{(inventory.base_names[provisional_id] or '<empty>'):<28}"
                f"{stats.image_count:>10}{stats.pixel_count:>16}"
            )
    candidates = (
        inventory.analysis.exact_name_groups + inventory.analysis.similar_name_groups
    )
    if candidates:
        print("\n自动检测到的合并候选:")
        for ids in candidates:
            print(
                f"  {[(class_id, inventory.class_names[class_id]) for class_id in ids]}"
            )
    print(
        f"\nmask ignore 值为 {inventory.ignore_label}。临时 ID 只用于本次交互；"
        "done 后统一生成连续最终 ID。"
    )


def build_final_source_maps(
    inventory: MergeInventory,
    resolved: common.ResolvedPlan,
) -> dict[str, dict[int, int | None]]:
    return {
        source.key: {
            old_id: resolved.source_to_target.get(provisional_id)
            for old_id, provisional_id in source.old_to_provisional.items()
        }
        for source in inventory.sources
    }


def _provisional_refs(
    inventory: MergeInventory,
) -> dict[int, list[dict[str, Any]]]:
    refs: dict[int, list[dict[str, Any]]] = {}
    for source in inventory.sources:
        for old_id, provisional_id in source.old_to_provisional.items():
            refs.setdefault(provisional_id, []).append(
                {"source_id": source.source_id, "old_id": old_id}
            )
    return refs


def _taxonomy_fingerprint(inventory: MergeInventory) -> str:
    groups: list[dict[str, Any]] = []
    for provisional_id, members in _provisional_refs(inventory).items():
        stable_members = sorted(
            (str(member["source_id"]), int(member["old_id"]))
            for member in members
        )
        if inventory.taxonomy_mode == "mapping-file":
            taxonomy_name = inventory.class_names[provisional_id]
        elif inventory.taxonomy_mode == "union-by-name":
            taxonomy_name = common.normalize_name(
                inventory.class_names[provisional_id]
            )
        else:
            taxonomy_name = inventory.base_names[provisional_id]
        groups.append({"members": stable_members, "name": taxonomy_name})
    return metadata_fingerprint(
        {
            "taxonomy_mode": inventory.taxonomy_mode,
            "groups": sorted(groups, key=lambda item: item["members"]),
        }
    )


def serialize_merge_plan(
    inventory: MergeInventory, plan: common.EditPlan
) -> dict[str, Any]:
    common.validate_plan(plan, inventory.class_names)
    refs = _provisional_refs(inventory)
    resolved = common.resolve_plan(plan, inventory.class_names)
    return {
        "version": 2,
        "taxonomy_mode": inventory.taxonomy_mode,
        "taxonomy_fingerprint": _taxonomy_fingerprint(inventory),
        "sources": [
            {
                "source_id": source.source_id,
                "fingerprint": source.fingerprint,
                "root": str(source.dataset.root),
                "config": str(source.dataset.config_path),
                "ignore_label": source.ignore_label,
            }
            for source in inventory.sources
        ],
        "operations": {
            "drop": [ref for value in sorted(plan.drop_ids) for ref in refs[value]],
            "merges": [
                {
                    "members": [ref for value in merge.ids for ref in refs[value]],
                    "name": merge.name,
                }
                for merge in plan.merges
            ],
            "renames": [
                {"member": refs[value][0], "name": name}
                for value, name in sorted(plan.renames.items())
            ],
            "unknown_policy": plan.unknown_policy,
            "id_policy": plan.id_policy,
            "explicit_ids": [
                {"member": refs[value][0], "target_id": target}
                for value, target in sorted(plan.explicit_ids.items())
            ],
        },
        "resolved_classes": [
            {
                "target_id": target_id,
                "name": resolved.output_names[target_id],
                "members": [
                    ref
                    for provisional_id, final_id in sorted(
                        resolved.source_to_target.items()
                    )
                    if final_id == target_id
                    for ref in refs[provisional_id]
                ],
            }
            for target_id in sorted(resolved.output_names)
        ],
        "drop_ids": sorted(plan.drop_ids),
        "merges": [asdict(merge) for merge in plan.merges],
        "renames": dict(sorted(plan.renames.items())),
        "unknown_policy": plan.unknown_policy,
        "id_policy": plan.id_policy,
        "explicit_ids": dict(sorted(plan.explicit_ids.items())),
    }


def load_merge_plan(path: Path, inventory: MergeInventory) -> common.EditPlan:
    data = common.read_yaml(path)
    if int(data.get("version", 1)) < 2 or "operations" not in data:
        return common.plan_from_yaml(path)
    plan_taxonomy = str(data.get("taxonomy_mode", "namespace"))
    if plan_taxonomy != inventory.taxonomy_mode:
        raise ValueError(
            f"plan taxonomy={plan_taxonomy} 与当前 taxonomy="
            f"{inventory.taxonomy_mode} 不一致"
        )
    stored_taxonomy_fingerprint = str(data.get("taxonomy_fingerprint", ""))
    if (
        stored_taxonomy_fingerprint
        and stored_taxonomy_fingerprint != _taxonomy_fingerprint(inventory)
    ):
        raise ValueError("plan taxonomy 分组指纹已变化")
    expected = {source.source_id: source for source in inventory.sources}
    listed = data.get("sources", [])
    if not isinstance(listed, list):
        raise ValueError("v2 plan 的 sources 需要是列表")
    for item in listed:
        source_id = str(item.get("source_id", ""))
        source = expected.get(source_id)
        if source is None:
            raise ValueError(f"plan 来源与当前输入不匹配: {source_id}")
        if str(item.get("fingerprint", "")) != source.fingerprint:
            raise ValueError(f"plan 来源类别指纹已变化: {source.dataset.root}")
    if {str(item.get("source_id", "")) for item in listed} != set(expected):
        raise ValueError("plan 来源集合与当前输入不一致")
    ref_to_provisional = {
        (source.source_id, old_id): provisional_id
        for source in inventory.sources
        for old_id, provisional_id in source.old_to_provisional.items()
    }

    def resolve_ref(item: dict[str, Any]) -> int:
        key = (str(item.get("source_id", "")), int(item.get("old_id")))
        if key not in ref_to_provisional:
            raise ValueError(f"plan 类别引用与当前输入不匹配: {item}")
        return ref_to_provisional[key]

    operations = data["operations"]
    resolved_classes = data.get("resolved_classes")
    if resolved_classes is not None:
        if not isinstance(resolved_classes, list):
            raise ValueError("v2 plan 的 resolved_classes 需要是列表")
        drops = {resolve_ref(item) for item in operations.get("drop", [])}
        merges: list[common.MergeSpec] = []
        renames: dict[int, str] = {}
        explicit_ids: dict[int, int] = {}
        assigned: set[int] = set()
        target_ids: set[int] = set()
        for row in resolved_classes:
            if not isinstance(row, dict):
                raise ValueError("resolved_classes 项需要是 object")
            target_id = int(row.get("target_id"))
            if target_id < 0 or target_id in target_ids:
                raise ValueError(f"resolved_classes 目标 ID 无效或重复: {target_id}")
            target_ids.add(target_id)
            name = str(row.get("name", "")).strip()
            if not name:
                raise ValueError(f"resolved_classes 目标 {target_id} 缺少名称")
            members = {
                resolve_ref(member) for member in row.get("members", [])
            }
            if not members:
                raise ValueError(f"resolved_classes 目标 {target_id} 缺少成员")
            overlap = assigned & members
            if overlap:
                raise ValueError(f"resolved_classes 成员重复: {sorted(overlap)}")
            assigned.update(members)
            explicit_ids.update({member: target_id for member in members})
            if len(members) >= 2:
                merges.append(common.MergeSpec(ids=sorted(members), name=name))
            else:
                member = next(iter(members))
                if inventory.class_names[member] != name:
                    renames[member] = name
        expected_ids = set(inventory.class_names)
        if assigned & drops:
            raise ValueError("resolved_classes 与删除成员冲突")
        missing = expected_ids - assigned - drops
        extra = (assigned | drops) - expected_ids
        if missing or extra:
            raise ValueError(
                f"resolved_classes 与当前类别集不一致: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        plan = common.EditPlan(
            drop_ids=drops,
            merges=merges,
            renames=renames,
            unknown_policy=str(operations.get("unknown_policy", "error")),
            id_policy="explicit",
            explicit_ids=explicit_ids,
        )
        common.validate_plan(plan, inventory.class_names)
        return plan

    plan = common.EditPlan(
        drop_ids={resolve_ref(item) for item in operations.get("drop", [])},
        merges=[
            common.MergeSpec(
                ids=sorted(
                    {resolve_ref(member) for member in item.get("members", [])}
                ),
                name=str(item["name"]),
            )
            for item in operations.get("merges", [])
            if len(
                {resolve_ref(member) for member in item.get("members", [])}
            )
            >= 2
        ],
        renames={
            resolve_ref(item["member"]): str(item["name"])
            for item in operations.get("renames", [])
        },
        unknown_policy=str(operations.get("unknown_policy", "error")),
        id_policy=str(operations.get("id_policy", "compact")),
        explicit_ids={
            resolve_ref(item["member"]): int(item["target_id"])
            for item in operations.get("explicit_ids", [])
        },
    )
    common.validate_plan(plan, inventory.class_names)
    return plan


def _validate_output(output: Path, inventory: MergeInventory) -> Path:
    return validate_output_location(
        output, [source.dataset.root for source in inventory.sources]
    )


def _mask_signature(mask: Any) -> str:
    digest = hashlib.sha256()
    digest.update(str(mask.shape).encode("ascii"))
    digest.update(str(mask.dtype).encode("ascii"))
    digest.update(mask.tobytes(order="C"))
    return digest.hexdigest()


def merge_datasets(
    inventory: MergeInventory,
    output: Path,
    plan: common.EditPlan,
    *,
    image_mode: str = "copy",
    clean: bool = False,
    dry_run: bool = False,
    all_ignore_policy: str = "keep",
    duplicate_policy: str = "keep",
    split_leakage_policy: str = "warn",
    workers: int = 1,
    require_train_val: bool = False,
) -> dict[str, Any]:
    if all_ignore_policy not in {"keep", "drop", "quarantine"}:
        raise ValueError(f"未知 all_ignore_policy: {all_ignore_policy}")
    if duplicate_policy not in {"keep", "error", "hash-dedupe", "drop"}:
        raise ValueError(f"未知 duplicate_policy: {duplicate_policy}")
    if split_leakage_policy not in {"warn", "error", "drop"}:
        raise ValueError(f"未知 split_leakage_policy: {split_leakage_policy}")
    if workers < 1:
        raise ValueError("workers 需要大于等于 1")
    resolved = common.resolve_plan(plan, inventory.class_names)
    if not resolved.output_names:
        raise ValueError("最终类别为空，语义分割数据集无法用于训练")
    if inventory.ignore_label in resolved.output_names:
        raise ValueError(
            f"最终类别 ID 占用了 ignore_label={inventory.ignore_label}，"
            "请减少类别或更换 ignore label"
        )
    output = _validate_output(output, inventory)

    final_maps = build_final_source_maps(inventory, resolved)
    report: dict[str, Any] = {
        "output": str(output),
        "format": "semantic_mask",
        "source_count": len(inventory.sources),
        "ignore_label": inventory.ignore_label,
        "image_mode": image_mode,
        "dry_run": dry_run,
        "output_names": resolved.output_names,
        "drop_ids": sorted(plan.drop_ids),
        "merges": [asdict(merge) for merge in plan.merges],
        "renames": dict(sorted(plan.renames.items())),
        "id_policy": plan.id_policy,
        "taxonomy_mode": inventory.taxonomy_mode,
        "all_ignore_policy": all_ignore_policy,
        "duplicate_policy": duplicate_policy,
        "split_leakage_policy": split_leakage_policy,
        "workers": workers,
        "require_train_val": require_train_val,
        "warnings": [],
        "sources": {},
        "splits": {},
    }
    for source in inventory.sources:
        report["sources"][source.key] = {
            "root": str(source.dataset.root),
            "source_id": source.source_id,
            "fingerprint": source.fingerprint,
            "source_ignore_label": source.ignore_label,
            "old_names": source.dataset.class_names,
            "old_class_labels": {
                class_id: [
                    list(label) if isinstance(label, tuple) else label
                    for label in labels
                ]
                for class_id, labels in source.dataset.class_labels.items()
            },
            "source_label_kind": source.dataset.label_kind,
            "old_to_provisional": source.old_to_provisional,
            "old_to_final": final_maps[source.key],
        }
    split_names = [
        split
        for split in ("train", "val", "test")
        if any(split in source.dataset.splits for source in inventory.sources)
    ]
    if "train" not in split_names:
        raise ValueError("语义输出缺少 train split，无法用于训练")
    if "val" not in split_names:
        message = "语义输出缺少 val split；直接训练前需要提供验证集"
        if require_train_val:
            raise ValueError(message)
        report["warnings"].append(message)
    if dry_run:
        return report
    for split in split_names:
        report["splits"][split] = {
            "images": 0,
            "masks": 0,
            "dropped_samples": 0,
            "quarantined_samples": 0,
            "quarantined_masks": 0,
            "duplicate_samples": 0,
            "identity_fast_path_masks": 0,
            "renamed_samples": 0,
        }

    tasks: list[_MaskTask] = []
    for source in inventory.sources:
        source_map = final_maps[source.key]
        source_resolved = common.ResolvedPlan(
            source_to_target={
                old_id: final_id
                for old_id, final_id in source_map.items()
                if final_id is not None
            },
            output_names=resolved.output_names,
            dropped_ids={
                old_id for old_id, final_id in source_map.items() if final_id is None
            },
        )
        identity_mapping = (
            source.ignore_label == inventory.ignore_label
            and not source.analysis.unknown_ids
            and _has_canonical_integer_encoding(source.dataset)
            and all(
                final_id == old_id
                for old_id, final_id in source_map.items()
            )
        )
        for split, paths in source.dataset.splits.items():
            image_files = common._iter_files(paths.images, IMAGE_EXTENSIONS)
            mask_files = common._iter_files(paths.masks, common.MASK_EXTENSIONS)
            image_index = {
                str(path.relative_to(paths.images).with_suffix("")).replace("\\", "/"): path
                for path in image_files
            }
            mask_index = {
                str(path.relative_to(paths.masks).with_suffix("")).replace("\\", "/"): path
                for path in mask_files
            }
            for stem, image_path in sorted(image_index.items()):
                mask_path = mask_index[stem]
                tasks.append(
                    _MaskTask(
                        source=source,
                        split=split,
                        stem=stem,
                        image_path=image_path,
                        mask_path=mask_path,
                        resolved=source_resolved,
                        can_fast_copy=(
                            identity_mapping
                            and mask_path.suffix.casefold() == ".png"
                        ),
                        output_ignore_label=inventory.ignore_label,
                        unknown_policy=plan.unknown_policy,
                        inspect_all_ignore=all_ignore_policy != "keep",
                        build_signature=duplicate_policy == "hash-dedupe",
                    )
                )

    all_images = [task.image_path for task in tasks]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        digests = dict(zip(
            all_images,
            tqdm(
                executor.map(file_digest, all_images),
                total=len(all_images),
                desc="计算图片指纹",
                unit="图片",
            ),
        ))

    seen_hashes: dict[str, dict[str, tuple[Path, str]]] = {}
    used_output_stems: dict[str, set[str]] = {}
    used_quarantine_stems: dict[str, set[str]] = {}
    with staged_output(output, clean=clean) as stage:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared_masks = _prepare_masks_ordered(
                executor,
                tasks,
                max_pending=max(1, workers * 2),
            )
            with tqdm(total=len(tasks), desc="合并语义 mask", unit="样本") as progress:
                for prepared in prepared_masks:
                    task = prepared.task
                    source = task.source
                    split = task.split
                    stem = task.stem
                    image_path = task.image_path
                    mask_path = task.mask_path
                    split_report = report["splits"][split]
                    if prepared.all_ignore and all_ignore_policy == "drop":
                        split_report["dropped_samples"] += 1
                        progress.update()
                        continue
                    quarantined = (
                        prepared.all_ignore and all_ignore_policy == "quarantine"
                    )
                    if not quarantined:
                        digest = digests[image_path]
                        by_split = seen_hashes.setdefault(digest, {})
                        other = next(
                            (
                                (previous_split, previous[0])
                                for previous_split, previous in sorted(by_split.items())
                                if previous_split != split
                            ),
                            None,
                        )
                        if other is not None:
                            previous_split, previous_path = other
                            message = (
                                f"跨 split 重复图片: {previous_path} "
                                f"({previous_split}) <-> {image_path} ({split})"
                            )
                            if split_leakage_policy == "error":
                                raise ValueError(message)
                            report["warnings"].append(message)
                            if split_leakage_policy == "drop":
                                split_report["dropped_samples"] += 1
                                progress.update()
                                continue
                        previous = by_split.get(split)
                        if previous is not None:
                            previous_path, previous_signature = previous
                            split_report["duplicate_samples"] += 1
                            if duplicate_policy == "error":
                                raise ValueError(
                                    f"重复图片内容: {previous_path} <-> {image_path}"
                                )
                            if duplicate_policy == "hash-dedupe":
                                if previous_signature != prepared.signature:
                                    raise ValueError(
                                        "重复图片的重映射 mask 不一致: "
                                        f"{previous_path} <-> {image_path}"
                                    )
                                split_report["dropped_samples"] += 1
                                progress.update()
                                continue
                            if duplicate_policy == "drop":
                                split_report["dropped_samples"] += 1
                                progress.update()
                                continue
                        else:
                            by_split[split] = (image_path, prepared.signature)
                    if quarantined:
                        split_report["quarantined_samples"] += 1
                        split_report["quarantined_masks"] += 1
                        stem_pool = used_quarantine_stems.setdefault(split, set())
                        base = stage / "quarantine" / split
                    else:
                        stem_pool = used_output_stems.setdefault(split, set())
                        base = stage
                    output_stem = allocate_flat_sample_stem(stem, stem_pool)
                    flattened_stem = stem.replace("\\", "/").replace("/", "__")
                    if output_stem != flattened_stem:
                        split_report["renamed_samples"] += 1
                    if quarantined:
                        image_target = (
                            base
                            / "images"
                            / f"{output_stem}{image_path.suffix.lower()}"
                        )
                        mask_target = base / "masks" / f"{output_stem}.png"
                    else:
                        image_target = (
                            stage
                            / "images"
                            / split
                            / f"{output_stem}{image_path.suffix.lower()}"
                        )
                        mask_target = stage / "masks" / split / f"{output_stem}.png"
                    common._transfer_file(image_path, image_target, image_mode)
                    if task.can_fast_copy:
                        common._transfer_file(mask_path, mask_target, image_mode)
                        if not quarantined:
                            split_report["identity_fast_path_masks"] += 1
                    else:
                        if prepared.remapped is None:
                            raise RuntimeError(f"mask 转换结果缺失: {mask_path}")
                        mask_target.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(prepared.remapped).save(mask_target)
                    if not quarantined:
                        split_report["images"] += 1
                        split_report["masks"] += 1
                    progress.update()

        if report["splits"]["train"]["images"] == 0:
            raise ValueError("所有 train 样本均被删除或隔离，输出无法用于训练")
        if "val" in report["splits"] and report["splits"]["val"]["images"] == 0:
            message = "所有 val 样本均被删除或隔离"
            if require_train_val:
                raise ValueError(message)
            report["warnings"].append(message)

        # Absolute paths keep the generated config consumable when the caller
        # loads data.yaml from a working directory outside the output tree.
        data: dict[str, Any] = {
            "classes": resolved.output_names,
            "task": "semantic_segmentation",
        }
        for split in split_names:
            data[split] = {
                "images": str(output / "images" / split),
                "masks": str(output / "masks" / split),
            }
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
        (stage / "dataset_merge_report.yaml").write_text(
            yaml.safe_dump(report, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        (stage / "class_plan.yaml").write_text(
            yaml.safe_dump(
                serialize_merge_plan(inventory, plan),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (stage / "dataset_manifest.yaml").write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "format": "semantic_mask",
                    "taxonomy_mode": inventory.taxonomy_mode,
                    "ignore_label": inventory.ignore_label,
                    "sources": report["sources"],
                    "splits": report["splits"],
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (stage / "_SUCCESS").write_text("ok\n", encoding="utf-8")
    return report


def _parse_indices(value: str, count: int) -> list[int]:
    tokens = [token for token in re.split(r"[\s,，]+", value.strip()) if token]
    try:
        indices = sorted({int(token) for token in tokens})
    except ValueError as exc:
        raise ValueError("数据集序号需要使用整数，并以空格或逗号分隔") from exc
    if len(indices) < 2:
        raise ValueError("请选择至少两个数据集")
    invalid = [index for index in indices if index < 1 or index > count]
    if invalid:
        raise ValueError(f"数据集序号超出范围: {invalid}")
    return indices


def choose_sources(search_root: Path) -> list[Path] | None:
    candidates = [
        candidate
        for candidate in detect_datasets(
            search_root, kinds={"semantic_mask"}
        )
        if candidate.kind == "semantic_mask"
    ]
    print(f"\n在 {search_root} 中检测到 {len(candidates)} 个 PNG 语义分割数据集:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. {KIND_LABELS[candidate.kind]:<18}"
            f"类别 {candidate.class_count:>4}，mask {candidate.annotation_count:>7}  "
            f"{candidate.path}"
        )
    print("   p. 手动输入多个 data.yaml 或数据集路径")
    while True:
        try:
            raw = input("选择至少两个数据集 [如 1,3 / p / q]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if raw.casefold() in {"q", "quit", "exit"}:
            return None
        if raw.casefold() == "p":
            paths: list[Path] = []
            print("逐行输入路径，空行结束。")
            while True:
                value = input(f"来源 {len(paths) + 1}: ").strip()
                if not value:
                    break
                paths.append(Path(value).expanduser().resolve())
            if len(paths) >= 2:
                return paths
            print("请输入至少两个路径。")
            continue
        try:
            indices = _parse_indices(raw, len(candidates))
        except ValueError as exc:
            print(exc)
            continue
        return [
            candidates[index - 1].config_path or candidates[index - 1].path
            for index in indices
        ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        action="append",
        help="输入 data.yaml 或语义数据集目录；至少重复两次",
    )
    parser.add_argument("--out", type=Path, help="最终合并数据集目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描目录，默认仓库 datasets/")
    parser.add_argument("--plan", type=Path, help="读取已有完整类别操作计划 YAML")
    parser.add_argument(
        "--id-policy",
        choices=("compact", "preserve", "explicit"),
        help="覆盖计划中的类别 ID 策略；explicit 的映射取自计划 explicit_ids",
    )
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument(
        "--source-ignore-label",
        action="append",
        default=[],
        metavar="SOURCE=ID|none",
        help="覆盖单个来源的输入 ignore 值，可重复使用",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "reflink", "hardlink", "symlink"),
        default="copy",
    )
    parser.add_argument(
        "--taxonomy",
        choices=("namespace", "strict", "union-by-name", "mapping-file"),
        default="namespace",
    )
    parser.add_argument("--mapping-file", type=Path)
    parser.add_argument(
        "--reference-source",
        type=int,
        metavar="N",
        help="以第 N 个 --src 的 data.yaml 作为类别 ID 与名称基准",
    )
    parser.add_argument(
        "--all-ignore-policy",
        choices=("keep", "drop", "quarantine"),
        default="keep",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("keep", "error", "hash-dedupe", "drop"),
        default="keep",
    )
    parser.add_argument(
        "--split-leakage-policy", choices=("warn", "error", "drop"), default="warn"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require-train-val", action="store_true")
    parser.add_argument("--clean", action="store_true", help="清理已有输出后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只分析并输出合并计划")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="采用第一个来源及推荐类别方案，并跳过最终确认",
    )
    return parser.parse_args(argv)


def _parse_source_ignore_overrides(values: list[str]) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for value in values:
        source, separator, raw_id = value.rpartition("=")
        if not separator or not source.strip():
            raise ValueError(
                f"--source-ignore-label 格式需要 SOURCE=ID|none: {value!r}"
            )
        raw_id = raw_id.strip().casefold()
        parsed = None if raw_id in {"none", "null", "class"} else int(raw_id)
        if parsed is not None and not 0 <= parsed <= 65535:
            raise ValueError(f"来源 ignore label 超出 0..65535: {parsed}")
        result[source.strip()] = parsed
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_paths = args.src
    if not source_paths:
        source_paths = choose_sources((args.datasets or auto_datasets_root()).resolve())
        if source_paths is None:
            return 0
    if args.plan is None:
        source_paths = common.choose_merge_reference(
            source_paths,
            selected=args.reference_source,
            assume_yes=args.yes,
        )
        if source_paths is None:
            return 0
    print("\n正在扫描全部来源 mask 和类别...")
    inventory = build_inventory(
        source_paths,
        ignore_label=args.ignore_label,
        source_ignore_labels=_parse_source_ignore_overrides(args.source_ignore_label),
        taxonomy_mode=args.taxonomy,
        mapping_file=args.mapping_file,
        use_first_source_names=args.plan is None,
    )
    print_inventory(inventory)
    planning_dataset = SimpleNamespace(class_names=inventory.class_names)
    if args.plan:
        plan = load_merge_plan(args.plan, inventory)
    else:
        reference_ids = set(inventory.sources[0].old_to_provisional.values())
        recommended_plan = common.recommend_reference_merge_plan(
            inventory.class_names,
            inventory.analysis,
            reference_ids=reference_ids,
            base_names=inventory.base_names,
        )
        common.print_reference_merge_plan(
            inventory,
            recommended_plan,
            annotation_label=f"重写对应 mask 像素值，忽略值 {args.ignore_label}",
        )
        if args.yes:
            plan = recommended_plan
        else:
            accept_all = common.confirm_reference_merge_plan()
            if accept_all is None:
                return 0
            if accept_all:
                plan = recommended_plan
            else:
                print("\n进入逐项确认模式。")
                plan = common.interactive_plan(
                    planning_dataset,  # type: ignore[arg-type]
                    inventory.analysis,
                    annotation_label="mask",
                    drop_effect=f"像素映射为 {args.ignore_label}",
                    unknown_effect=f"映射为 {args.ignore_label}",
                    merge_default_names=inventory.base_names,
                )
    if plan is None:
        return 0
    if args.id_policy is not None:
        plan.id_policy = args.id_policy
    resolved = common.resolve_plan(plan, inventory.class_names)
    print("\n最终类别:")
    for class_id, name in resolved.output_names.items():
        print(f"  {class_id}: {name}")
    print(f"ignore 值: {args.ignore_label}")

    output = args.out or default_output_dir(
        inventory.sources[0].dataset.root, "merge-semantic"
    )
    if args.out is None:
        raw = input(f"输出目录 [{output}]: ").strip()
        if raw:
            output = Path(raw)
    if not args.dry_run and not args.yes:
        answer = input(
            "确认一次性重写所有 mask、复制图片并生成最终数据集？[y/N]: "
        ).strip().casefold()
        if answer not in {"y", "yes", "是"}:
            print("已取消，尚未生成输出数据集。")
            return 0
    report = merge_datasets(
        inventory,
        output,
        plan,
        image_mode=args.image_mode,
        clean=args.clean,
        dry_run=args.dry_run,
        all_ignore_policy=args.all_ignore_policy,
        duplicate_policy=args.duplicate_policy,
        split_leakage_policy=args.split_leakage_policy,
        workers=args.workers,
        require_train_val=args.require_train_val,
    )
    if args.dry_run:
        print("\ndry-run 完成，尚未生成输出数据集。")
        print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    else:
        print(f"\n合并完成: {report['output']}")
        print(f"最终类别数: {len(report['output_names'])}")
        print(f"合并报告: {Path(report['output']) / 'dataset_merge_report.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
