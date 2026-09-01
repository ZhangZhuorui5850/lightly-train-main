#!/usr/bin/env python3
"""交互式合并多个 YOLO Det/Seg 数据集并统一类别 ID。

程序先为每个来源建立独立的临时类别 ID，展示同名/近似类别候选，然后循环接收
删除、合并、重命名操作。只有用户输入 done 并确认后，程序才会一次性重写全部
YOLO TXT、复制图片并生成一个 data.yaml。
"""

from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

try:
    from . import semantic_class_editor as common
    from . import yolo_class_editor as yolo
    from .dataset_discovery import (
        KIND_LABELS,
        auto_datasets_root,
    )
    from .dataset_detector import detect_datasets
    from .output_naming import allocate_flat_sample_stem, default_output_dir
    from .progress import tqdm
    from .text_encoding import read_text_auto
    from .dataset_transaction import (
        file_digest,
        metadata_fingerprint,
        staged_output,
        validate_distinct_sources,
        validate_output_location,
    )
except ImportError:
    import semantic_class_editor as common  # type: ignore[no-redef]
    import yolo_class_editor as yolo  # type: ignore[no-redef]
    from dataset_discovery import (  # type: ignore[no-redef]
        KIND_LABELS,
        auto_datasets_root,
    )
    from dataset_detector import detect_datasets  # type: ignore[no-redef]
    from output_naming import (  # type: ignore[no-redef]
        allocate_flat_sample_stem,
        default_output_dir,
    )
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]
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
    dataset: yolo.YoloDataset
    analysis: yolo.YoloAnalysis
    old_to_provisional: dict[int, int]


@dataclass
class MergeInventory:
    sources: list[MergeSource]
    kind: str
    class_names: dict[int, str]
    base_names: dict[int, str]
    analysis: common.DatasetAnalysis
    image_count: int
    label_count: int
    annotation_count: int
    taxonomy_mode: str = "namespace"


@dataclass(frozen=True)
class _CanonicalAnnotation:
    """A remapped annotation retained with enough context for conflict errors."""

    class_id: int
    geometry: tuple[float, ...]
    line: str
    source_path: Path
    line_number: int


@dataclass
class _SeenSample:
    """The deterministic first output selected for one image-content hash/split."""

    image_path: Path
    label_target: Path
    annotations: list[_CanonicalAnnotation]
    label_written: bool


@dataclass
class _AnnotationMergeResult:
    annotations: list[_CanonicalAnnotation]
    accepted: int = 0
    deduplicated: int = 0
    conflicted: int = 0
    conflict_pairs: int = 0
    replaced: int = 0


def _source_key(index: int) -> str:
    return f"dataset{index + 1}"


def _safe_component(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", value).strip("._-")
    return value or "dataset"


def _source_metadata(dataset: yolo.YoloDataset) -> tuple[str, str]:
    fingerprint = metadata_fingerprint(
        {
            "format": dataset.kind,
            "classes": dataset.class_names,
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


def build_inventory(
    source_paths: list[Path],
    *,
    forced_kind: str | None = None,
    taxonomy_mode: str = "namespace",
    mapping_file: Path | None = None,
    orphan_policy: str = "error",
    use_first_source_names: bool = False,
) -> MergeInventory:
    if len(source_paths) < 2:
        raise ValueError("合并至少需要两个 YOLO 数据集")
    if taxonomy_mode not in {"namespace", "strict", "union-by-name", "mapping-file"}:
        raise ValueError(f"未知 taxonomy_mode: {taxonomy_mode}")
    if orphan_policy not in {"error", "drop"}:
        raise ValueError(f"未知 orphan_policy: {orphan_policy}")
    resolved_datasets = [
        yolo.resolve_dataset(path.expanduser().resolve(), forced_kind=forced_kind)
        for path in source_paths
    ]
    validate_distinct_sources(
        [(dataset.root, dataset.config_path) for dataset in resolved_datasets]
    )
    if taxonomy_mode == "strict":
        expected = resolved_datasets[0].class_names
        mismatches = [
            str(dataset.root)
            for dataset in resolved_datasets[1:]
            if dataset.class_names != expected
        ]
        if mismatches:
            raise ValueError(
                "strict taxonomy 要求所有来源使用完全一致的 ID/类别名；"
                f"不一致来源: {mismatches}"
            )
    explicit_mapping = (
        _load_taxonomy_mapping(mapping_file)
        if taxonomy_mode == "mapping-file"
        else {}
    )

    sources: list[MergeSource] = []
    class_names: dict[int, str] = {}
    base_names: dict[int, str] = {}
    raw_names: dict[int, Any] = {}
    class_stats: dict[int, common.ClassStats] = {}
    observed_ids: set[int] = set()
    null_like_ids: set[int] = set()
    image_count = 0
    label_count = 0
    annotation_count = 0
    kind: str | None = None
    next_id = 0
    taxonomy_ids: dict[tuple[str, Any], int] = {}

    for index, dataset in enumerate(
        tqdm(resolved_datasets, desc="分析合并来源", unit="数据集")
    ):
        if kind is None:
            kind = dataset.kind
        elif dataset.kind != kind:
            raise ValueError(
                "所有输入需要使用同一种 YOLO 标注格式，"
                f"当前同时出现 {kind} 和 {dataset.kind}"
            )
        analysis = yolo.analyze_dataset(dataset)
        fatal_types = {
            "invalid_label",
            "invalid_image",
            "duplicate_image_stem",
            "duplicate_label_stem",
        }
        if orphan_policy == "error":
            fatal_types.add("missing_image")
        fatal = [issue for issue in analysis.shared.issues if issue.issue_type in fatal_types]
        if fatal:
            first = fatal[0]
            raise ValueError(
                f"{dataset.root} 存在 {len(fatal)} 个 preflight 问题；"
                f"首个问题: {first.path}: {first.message}"
            )

        key = _source_key(index)
        source_id, fingerprint = _source_metadata(dataset)
        old_to_provisional: dict[int, int] = {}
        all_old_ids = sorted(set(dataset.class_names) | analysis.shared.unknown_ids)
        for old_id in all_old_ids:
            if old_id in dataset.class_names:
                base_name = dataset.class_names[old_id]
                raw_name = dataset.raw_class_names.get(old_id)
                source_stats = analysis.shared.class_stats[old_id]
            else:
                base_name = f"unknown_{old_id}"
                raw_name = base_name
                source_stats = analysis.shared.unknown_stats[old_id]
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
                    "name",
                    normalized if normalized else f"{source_id}:{old_id}",
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
                base_names[provisional_id] = (
                    output_name if taxonomy_mode == "mapping-file" else base_name
                )
                raw_names[provisional_id] = raw_name
                class_stats[provisional_id] = common.ClassStats()
            old_to_provisional[old_id] = provisional_id
            class_stats[provisional_id].image_count += source_stats.image_count
            class_stats[provisional_id].pixel_count += source_stats.pixel_count
            if source_stats.pixel_count:
                observed_ids.add(provisional_id)
            if common.is_null_like_name(base_name, raw_name):
                null_like_ids.add(provisional_id)

        sources.append(
            MergeSource(
                key=key,
                source_id=source_id,
                fingerprint=fingerprint,
                dataset=dataset,
                analysis=analysis,
                old_to_provisional=old_to_provisional,
            )
        )
        image_count += analysis.shared.image_count
        label_count += analysis.shared.mask_count
        annotation_count += analysis.annotation_count

    unused_ids = set(class_names) - observed_ids
    exact_groups, similar_groups = common._name_candidate_groups(class_names)
    issues: list[common.AnalysisIssue] = []
    if unused_ids:
        issues.append(
            common.AnalysisIssue(
                "unused_yaml_ids",
                f"类别表中存在标注未使用的临时 ID: {sorted(unused_ids)}",
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
        unknown_stats={},
        observed_ids=observed_ids,
        unknown_ids=set(),
        unused_ids=unused_ids,
        null_like_ids=null_like_ids,
        exact_name_groups=exact_groups,
        similar_name_groups=similar_groups,
        issues=issues,
        mask_count=label_count,
        image_count=image_count,
    )
    return MergeInventory(
        sources=sources,
        kind=kind or "yolo_detection",
        class_names=class_names,
        base_names=base_names,
        analysis=shared_analysis,
        image_count=image_count,
        label_count=label_count,
        annotation_count=annotation_count,
        taxonomy_mode=taxonomy_mode,
    )


def print_inventory(inventory: MergeInventory) -> None:
    kind_name = (
        "YOLO 实例分割" if inventory.kind == "yolo_instance" else "YOLO 目标检测"
    )
    print(
        f"\n格式 {kind_name}，来源 {len(inventory.sources)}，图片 {inventory.image_count}，"
        f"label {inventory.label_count}，对象 {inventory.annotation_count}"
    )
    for source in inventory.sources:
        print(f"  {source.key}: {source.dataset.root}")
    print(
        f"\n  {'临时ID':>6}  {'来源':<10}{'原ID':>8}  "
        f"{'原类别名':<28}{'文件数':>10}{'对象数':>12}"
    )
    print("  " + "-" * 82)
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
                f"{stats.image_count:>10}{stats.pixel_count:>12}"
            )
    candidates = (
        inventory.analysis.exact_name_groups + inventory.analysis.similar_name_groups
    )
    if candidates:
        print("\n自动检测到的合并候选:")
        for ids in candidates:
            values = [
                (class_id, inventory.class_names[class_id]) for class_id in ids
            ]
            print(f"  {values}")
    print(
        "\n临时 ID 只用于本次交互。输入多条 m/d/r 命令并执行 done 后，"
        "程序统一生成连续最终 ID。"
    )


def build_final_source_maps(
    inventory: MergeInventory,
    resolved: common.ResolvedPlan,
) -> dict[str, dict[int, int | None]]:
    result: dict[str, dict[int, int | None]] = {}
    for source in inventory.sources:
        result[source.key] = {
            old_id: resolved.source_to_target.get(provisional_id)
            for old_id, provisional_id in source.old_to_provisional.items()
        }
    return result


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
    """Fingerprint source-to-taxonomy grouping independently of source order."""
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
    """Serialize a source-bound v2 plan while retaining legacy fields."""
    common.validate_plan(plan, inventory.class_names)
    refs = _provisional_refs(inventory)
    resolved = common.resolve_plan(plan, inventory.class_names)
    data: dict[str, Any] = {
        "version": 2,
        "taxonomy_mode": inventory.taxonomy_mode,
        "taxonomy_fingerprint": _taxonomy_fingerprint(inventory),
        "sources": [
            {
                "source_id": source.source_id,
                "fingerprint": source.fingerprint,
                "root": str(source.dataset.root),
                "config": str(source.dataset.config_path),
            }
            for source in inventory.sources
        ],
        "operations": {
            "drop": [ref for value in sorted(plan.drop_ids) for ref in refs[value]],
            "merges": [
                {
                    "members": [
                        ref for value in merge.ids for ref in refs[value]
                    ],
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
        # Lock every final class to stable source references.  Operations alone
        # cannot preserve untouched compact IDs when --src ordering changes.
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
        # A legacy reader can still consume plans produced by this version when
        # source ordering remains unchanged.
        "drop_ids": sorted(plan.drop_ids),
        "merges": [asdict(merge) for merge in plan.merges],
        "renames": dict(sorted(plan.renames.items())),
        "unknown_policy": plan.unknown_policy,
        "id_policy": plan.id_policy,
        "explicit_ids": dict(sorted(plan.explicit_ids.items())),
    }
    return data


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
            raise ValueError(
                f"plan 来源类别指纹已变化: {source.dataset.root}"
            )
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

    drops = {resolve_ref(item) for item in operations.get("drop", [])}
    merges: list[common.MergeSpec] = []
    for item in operations.get("merges", []):
        ids = sorted({resolve_ref(member) for member in item.get("members", [])})
        if len(ids) >= 2:
            merges.append(common.MergeSpec(ids=ids, name=str(item["name"])))
    renames = {
        resolve_ref(item["member"]): str(item["name"])
        for item in operations.get("renames", [])
    }
    explicit_ids = {
        resolve_ref(item["member"]): int(item["target_id"])
        for item in operations.get("explicit_ids", [])
    }
    plan = common.EditPlan(
        drop_ids=drops,
        merges=merges,
        renames=renames,
        unknown_policy=str(operations.get("unknown_policy", "error")),
        id_policy=str(operations.get("id_policy", "compact")),
        explicit_ids=explicit_ids,
    )
    common.validate_plan(plan, inventory.class_names)
    return plan


def _validate_output(output: Path, inventory: MergeInventory) -> Path:
    return validate_output_location(
        output, [source.dataset.root for source in inventory.sources]
    )


def _validate_yolo_output_classes(resolved: common.ResolvedPlan) -> None:
    class_ids = sorted(resolved.output_names)
    if not class_ids:
        raise ValueError("最终类别为空，YOLO 数据集无法用于训练")
    if class_ids != list(range(len(class_ids))):
        raise ValueError(
            "YOLO 输出类别 ID 需要连续为 0..N-1；"
            f"当前为 {class_ids}"
        )


def _detection_box(geometry: tuple[float, ...]) -> tuple[float, float, float, float]:
    center_x, center_y, width, height = geometry
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _box_intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = _box_intersection_area(first, second)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _polygon_points(geometry: tuple[float, ...]) -> list[tuple[float, float]]:
    return list(zip(geometry[0::2], geometry[1::2]))


def _signed_polygon_area(points: list[tuple[float, float]]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(_signed_polygon_area(points))


def _polygon_box(
    points: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def _orientation(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _point_on_segment(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    epsilon: float = 1e-12,
) -> bool:
    return (
        abs(_orientation(first, second, point)) <= epsilon
        and min(first[0], second[0]) - epsilon
        <= point[0]
        <= max(first[0], second[0]) + epsilon
        and min(first[1], second[1]) - epsilon
        <= point[1]
        <= max(first[1], second[1]) + epsilon
    )


def _segments_intersect(
    first_start: tuple[float, float],
    first_end: tuple[float, float],
    second_start: tuple[float, float],
    second_end: tuple[float, float],
    *,
    epsilon: float = 1e-12,
) -> bool:
    orientations = (
        _orientation(first_start, first_end, second_start),
        _orientation(first_start, first_end, second_end),
        _orientation(second_start, second_end, first_start),
        _orientation(second_start, second_end, first_end),
    )
    if (
        orientations[0] * orientations[1] < -epsilon
        and orientations[2] * orientations[3] < -epsilon
    ):
        return True
    return (
        (
            abs(orientations[0]) <= epsilon
            and _point_on_segment(second_start, first_start, first_end)
        )
        or (
            abs(orientations[1]) <= epsilon
            and _point_on_segment(second_end, first_start, first_end)
        )
        or (
            abs(orientations[2]) <= epsilon
            and _point_on_segment(first_start, second_start, second_end)
        )
        or (
            abs(orientations[3]) <= epsilon
            and _point_on_segment(first_end, second_start, second_end)
        )
    )


def _is_simple_polygon(points: list[tuple[float, float]]) -> bool:
    count = len(points)
    for first_index in range(count):
        first_end = (first_index + 1) % count
        for second_index in range(first_index + 1, count):
            second_end = (second_index + 1) % count
            if (
                first_index == second_index
                or first_end == second_index
                or second_end == first_index
            ):
                continue
            if _segments_intersect(
                points[first_index],
                points[first_end],
                points[second_index],
                points[second_end],
            ):
                return False
    return True


def _is_convex_polygon(points: list[tuple[float, float]]) -> bool:
    direction = 0
    for index in range(len(points)):
        cross = _orientation(
            points[index - 1], points[index], points[(index + 1) % len(points)]
        )
        if abs(cross) <= 1e-12:
            continue
        current = 1 if cross > 0.0 else -1
        if direction and current != direction:
            return False
        direction = current
    return direction != 0 and _is_simple_polygon(points)


def _line_intersection(
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
    clip_start: tuple[float, float],
    clip_end: tuple[float, float],
) -> tuple[float, float]:
    segment_dx = segment_end[0] - segment_start[0]
    segment_dy = segment_end[1] - segment_start[1]
    clip_dx = clip_end[0] - clip_start[0]
    clip_dy = clip_end[1] - clip_start[1]
    denominator = segment_dx * clip_dy - segment_dy * clip_dx
    if abs(denominator) <= 1e-15:
        return segment_end
    offset_x = clip_start[0] - segment_start[0]
    offset_y = clip_start[1] - segment_start[1]
    fraction = (offset_x * clip_dy - offset_y * clip_dx) / denominator
    return (
        segment_start[0] + fraction * segment_dx,
        segment_start[1] + fraction * segment_dy,
    )


def _convex_polygon_intersection(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    output = list(subject)
    clip_direction = 1.0 if _signed_polygon_area(clip) > 0.0 else -1.0
    for clip_start, clip_end in zip(clip, clip[1:] + clip[:1]):
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = (
            clip_direction * _orientation(clip_start, clip_end, previous)
            >= -1e-12
        )
        for current in input_points:
            current_inside = (
                clip_direction * _orientation(clip_start, clip_end, current)
                >= -1e-12
            )
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, clip_start, clip_end)
                )
            previous = current
            previous_inside = current_inside
    return output


def _polygon_iou_or_upper_bound(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> tuple[float | None, float]:
    """Return exact IoU for convex simple polygons, otherwise a safe upper bound."""

    first_area = _polygon_area(first)
    second_area = _polygon_area(second)
    first_box = _polygon_box(first)
    second_box = _polygon_box(second)
    box_intersection = _box_intersection_area(first_box, second_box)
    max_intersection = min(box_intersection, first_area, second_area)
    min_union = first_area + second_area - max_intersection
    upper_bound = max_intersection / min_union if min_union > 0.0 else 0.0
    if not (_is_convex_polygon(first) and _is_convex_polygon(second)):
        return None, upper_bound
    intersection = _polygon_area(_convex_polygon_intersection(first, second))
    union = first_area + second_area - intersection
    iou = intersection / union if union > 0.0 else 0.0
    return iou, iou


def _canonical_polygon(
    geometry: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    points = tuple(_polygon_points(geometry))
    variants: list[tuple[tuple[float, float], ...]] = []
    for sequence in (points, tuple(reversed(points))):
        variants.extend(
            sequence[index:] + sequence[:index] for index in range(len(sequence))
        )
    return min(variants)


def _geometry_relation(
    first: _CanonicalAnnotation,
    second: _CanonicalAnnotation,
    *,
    kind: str,
    conflict_iou: float,
) -> tuple[str, float | None]:
    """Return same, conflict, ambiguous, or distinct for two geometries."""

    if kind == "yolo_detection":
        if first.geometry == second.geometry:
            return "same", 1.0
        iou = _box_iou(_detection_box(first.geometry), _detection_box(second.geometry))
        return ("conflict", iou) if iou >= conflict_iou else ("distinct", iou)

    if _canonical_polygon(first.geometry) == _canonical_polygon(second.geometry):
        return "same", 1.0
    iou, upper_bound = _polygon_iou_or_upper_bound(
        _polygon_points(first.geometry), _polygon_points(second.geometry)
    )
    if iou is not None:
        if math.isclose(iou, 1.0, rel_tol=0.0, abs_tol=1e-12):
            return "same", iou
        return ("conflict", iou) if iou >= conflict_iou else ("distinct", iou)
    if upper_bound >= conflict_iou:
        return "ambiguous", upper_bound
    return "distinct", upper_bound


def _annotation_description(annotation: _CanonicalAnnotation) -> str:
    return (
        f"{annotation.source_path}:{annotation.line_number} "
        f"class={annotation.class_id} geometry={annotation.geometry}"
    )


def _merge_annotations(
    existing: list[_CanonicalAnnotation],
    incoming: list[_CanonicalAnnotation],
    *,
    kind: str,
    conflict_policy: str,
    conflict_iou: float,
) -> _AnnotationMergeResult:
    result = _AnnotationMergeResult(annotations=list(existing))
    for candidate in incoming:
        duplicate = False
        conflicts: list[tuple[int, str, float | None]] = []
        for index, current in enumerate(result.annotations):
            # Rows originating in one label file are already a coherent source
            # annotation set; conflict checks apply across duplicate samples.
            if current.source_path == candidate.source_path:
                continue
            relation, overlap = _geometry_relation(
                current,
                candidate,
                kind=kind,
                conflict_iou=conflict_iou,
            )
            if relation == "same" and current.class_id == candidate.class_id:
                duplicate = True
                break
            if relation in {"same", "conflict", "ambiguous"}:
                conflicts.append((index, relation, overlap))
        if duplicate:
            result.deduplicated += 1
            continue
        if not conflicts:
            result.annotations.append(candidate)
            result.accepted += 1
            continue

        result.conflicted += 1
        result.conflict_pairs += len(conflicts)
        first_index, relation, overlap = conflicts[0]
        current = result.annotations[first_index]
        if conflict_policy == "error":
            overlap_text = "unknown" if overlap is None else f"{overlap:.6f}"
            raise ValueError(
                "重复图片标注冲突: "
                f"type={relation}, iou_or_upper_bound={overlap_text}, "
                f"threshold={conflict_iou:.6f}; "
                f"existing=({_annotation_description(current)}); "
                f"incoming=({_annotation_description(candidate)})"
            )
        if conflict_policy == "keep-first":
            continue
        if conflict_policy == "keep-both":
            result.annotations.append(candidate)
            result.accepted += 1
            continue

        # keep-last replaces every prior annotation that conflicts with this
        # incoming annotation and preserves the earliest stable list position.
        conflict_indices = {index for index, _, _ in conflicts}
        insert_at = min(conflict_indices)
        result.annotations = [
            annotation
            for index, annotation in enumerate(result.annotations)
            if index not in conflict_indices
        ]
        result.annotations.insert(insert_at, candidate)
        result.accepted += 1
        result.replaced += len(conflict_indices)
    return result


def _annotation_text(annotations: list[_CanonicalAnnotation]) -> str:
    return "".join(f"{annotation.line}\n" for annotation in annotations)


def merge_datasets(
    inventory: MergeInventory,
    output: Path,
    plan: common.EditPlan,
    *,
    image_mode: str = "copy",
    clean: bool = False,
    dry_run: bool = False,
    empty_image_policy: str = "keep",
    duplicate_policy: str = "keep",
    duplicate_annotation_conflict_policy: str = "error",
    duplicate_annotation_conflict_iou: float = 0.95,
    split_leakage_policy: str = "warn",
    orphan_policy: str = "error",
    workers: int = 1,
    require_train_val: bool = False,
) -> dict[str, Any]:
    if empty_image_policy not in {"keep", "drop", "quarantine"}:
        raise ValueError(f"未知 empty_image_policy: {empty_image_policy}")
    if duplicate_policy not in {
        "keep",
        "error",
        "hash-dedupe",
        "drop",
        "merge-annotations",
    }:
        raise ValueError(f"未知 duplicate_policy: {duplicate_policy}")
    if duplicate_annotation_conflict_policy not in {
        "error",
        "keep-first",
        "keep-last",
        "keep-both",
    }:
        raise ValueError(
            "未知 duplicate_annotation_conflict_policy: "
            f"{duplicate_annotation_conflict_policy}"
        )
    if not 0.0 < duplicate_annotation_conflict_iou <= 1.0:
        raise ValueError("duplicate_annotation_conflict_iou 需要位于 (0, 1]")
    if split_leakage_policy not in {"warn", "error", "drop"}:
        raise ValueError(f"未知 split_leakage_policy: {split_leakage_policy}")
    if orphan_policy not in {"error", "drop"}:
        raise ValueError(f"未知 orphan_policy: {orphan_policy}")
    if workers < 1:
        raise ValueError("workers 需要大于等于 1")
    resolved = common.resolve_plan(plan, inventory.class_names)
    _validate_yolo_output_classes(resolved)
    output = _validate_output(output, inventory)

    final_maps = build_final_source_maps(inventory, resolved)
    report: dict[str, Any] = {
        "output": str(output),
        "format": inventory.kind,
        "source_count": len(inventory.sources),
        "image_mode": image_mode,
        "dry_run": dry_run,
        "output_names": resolved.output_names,
        "drop_ids": sorted(plan.drop_ids),
        "merges": [asdict(merge) for merge in plan.merges],
        "renames": dict(sorted(plan.renames.items())),
        "id_policy": plan.id_policy,
        "taxonomy_mode": inventory.taxonomy_mode,
        "empty_image_policy": empty_image_policy,
        "duplicate_policy": duplicate_policy,
        "duplicate_annotation_conflict_policy": (
            duplicate_annotation_conflict_policy
        ),
        "duplicate_annotation_conflict_iou": duplicate_annotation_conflict_iou,
        "split_leakage_policy": split_leakage_policy,
        "orphan_policy": orphan_policy,
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
            "old_names": source.dataset.class_names,
            "old_to_provisional": source.old_to_provisional,
            "old_to_final": final_maps[source.key],
        }
    split_names = [
        split
        for split in ("train", "val", "test")
        if any(split in source.dataset.splits for source in inventory.sources)
    ]
    if "train" not in split_names:
        raise ValueError("YOLO 输出缺少 train split，无法用于训练")
    if "val" not in split_names:
        message = "YOLO 输出缺少 val split；直接训练前需要提供验证集"
        if require_train_val:
            raise ValueError(message)
        report["warnings"].append(message)
    if dry_run:
        return report
    for split in split_names:
        report["splits"][split] = {
            "images": 0,
            "labels": 0,
            "kept_annotations": 0,
            "removed_annotations": 0,
            "dropped_samples": 0,
            "quarantined_samples": 0,
            "quarantined_labels": 0,
            "quarantined_annotations": 0,
            "quarantined_removed_annotations": 0,
            "dropped_annotations": 0,
            "duplicate_samples": 0,
            "merged_samples": 0,
            "merged_annotations": 0,
            "deduplicated_annotations": 0,
            "conflicted_annotations": 0,
            "annotation_conflict_pairs": 0,
            "replaced_annotations": 0,
            "orphan_labels": 0,
            "renamed_samples": 0,
        }

    all_images = [
        image
        for source in inventory.sources
        for paths in source.dataset.splits.values()
        for image in yolo.split_image_paths(paths)
    ]
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

    seen_hashes: dict[str, dict[str, _SeenSample]] = {}
    used_output_stems: dict[str, set[str]] = {}
    used_quarantine_stems: dict[str, set[str]] = {}
    with staged_output(output, clean=clean) as stage:
        for source in inventory.sources:
            source_component = _safe_component(source.key)
            source_map = final_maps[source.key]
            for split, paths in source.dataset.splits.items():
                split_report = report["splits"][split]
                image_files = yolo.split_image_paths(paths)
                label_files = yolo.split_label_paths(paths)
                image_index = {
                    yolo.logical_sample_stem(path, paths.images): path
                    for path in image_files
                }
                label_index = {
                    yolo.logical_sample_stem(path, paths.labels): path
                    for path in label_files
                }
                orphan_stems = sorted(set(label_index) - set(image_index))
                if orphan_stems and orphan_policy == "error":
                    raise ValueError(
                        f"{source.dataset.root}/{split} 存在孤儿 label: {orphan_stems[0]}"
                    )
                split_report["orphan_labels"] += len(orphan_stems)
                with tqdm(
                    total=len(image_index),
                    desc=f"合并 {source_component}/{split}",
                    unit="样本",
                ) as progress:
                    for stem, image_path in sorted(image_index.items()):
                        label_path = label_index.get(stem)
                        kept = removed = original = 0
                        canonical_annotations: list[_CanonicalAnnotation] = []
                        if label_path is not None:
                            for line_number, line in enumerate(
                                read_text_auto(label_path).splitlines(),
                                start=1,
                            ):
                                if not line.strip():
                                    continue
                                original += 1
                                try:
                                    old_id, tokens = yolo.parse_yolo_row(
                                        line, inventory.kind
                                    )
                                except ValueError as exc:
                                    raise ValueError(
                                        f"{label_path} 第 {line_number} 行: {exc}"
                                    ) from exc
                                final_id = source_map.get(old_id)
                                if final_id is None:
                                    removed += 1
                                    continue
                                tokens[0] = str(final_id)
                                output_line = " ".join(tokens)
                                canonical_annotations.append(
                                    _CanonicalAnnotation(
                                        class_id=final_id,
                                        geometry=tuple(
                                            float(value) for value in tokens[1:]
                                        ),
                                        line=output_line,
                                        source_path=label_path,
                                        line_number=line_number,
                                    )
                                )
                                kept += 1

                        empty_after_edit = original > 0 and kept == 0
                        if empty_after_edit and empty_image_policy == "drop":
                            split_report["dropped_samples"] += 1
                            split_report["dropped_annotations"] += original
                            progress.update()
                            continue
                        quarantined = empty_after_edit and empty_image_policy == "quarantine"
                        if quarantined:
                            split_report["quarantined_samples"] += 1
                            split_report["quarantined_annotations"] += kept
                            split_report["quarantined_removed_annotations"] += removed
                        annotation_signature = metadata_fingerprint(
                            sorted(
                                (
                                    annotation.class_id,
                                    *annotation.geometry,
                                )
                                for annotation in canonical_annotations
                            )
                        )
                        register_in: dict[str, _SeenSample] | None = None
                        if not quarantined:
                            digest = digests[image_path]
                            by_split = seen_hashes.setdefault(digest, {})
                            other = next(
                                (
                                    (previous_split, previous.image_path)
                                    for previous_split, previous in sorted(
                                        by_split.items()
                                    )
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
                                    split_report["dropped_annotations"] += original
                                    progress.update()
                                    continue
                            previous = by_split.get(split)
                            if previous is not None:
                                split_report["duplicate_samples"] += 1
                                if duplicate_policy == "error":
                                    raise ValueError(
                                        "重复图片内容: "
                                        f"{previous.image_path} <-> {image_path}"
                                    )
                                if duplicate_policy == "hash-dedupe":
                                    previous_signature = metadata_fingerprint(
                                        sorted(
                                            (
                                                annotation.class_id,
                                                *annotation.geometry,
                                            )
                                            for annotation in previous.annotations
                                        )
                                    )
                                    if previous_signature != annotation_signature:
                                        raise ValueError(
                                            "重复图片的重写后 label 不一致: "
                                            f"{previous.image_path} <-> {image_path}"
                                        )
                                    split_report["dropped_samples"] += 1
                                    split_report["dropped_annotations"] += original
                                    progress.update()
                                    continue
                                if duplicate_policy == "drop":
                                    split_report["dropped_samples"] += 1
                                    split_report["dropped_annotations"] += original
                                    progress.update()
                                    continue
                                if duplicate_policy == "merge-annotations":
                                    before_count = len(previous.annotations)
                                    merge_result = _merge_annotations(
                                        previous.annotations,
                                        canonical_annotations,
                                        kind=inventory.kind,
                                        conflict_policy=(
                                            duplicate_annotation_conflict_policy
                                        ),
                                        conflict_iou=(
                                            duplicate_annotation_conflict_iou
                                        ),
                                    )
                                    previous.annotations = merge_result.annotations
                                    should_write_label = (
                                        previous.label_written
                                        or label_path is not None
                                        or bool(previous.annotations)
                                    )
                                    if should_write_label:
                                        previous.label_target.parent.mkdir(
                                            parents=True, exist_ok=True
                                        )
                                        previous.label_target.write_text(
                                            _annotation_text(previous.annotations),
                                            encoding="utf-8",
                                        )
                                        if not previous.label_written:
                                            previous.label_written = True
                                            split_report["labels"] += 1
                                    split_report["merged_samples"] += 1
                                    split_report["merged_annotations"] += (
                                        merge_result.accepted
                                    )
                                    split_report["deduplicated_annotations"] += (
                                        merge_result.deduplicated
                                    )
                                    split_report["conflicted_annotations"] += (
                                        merge_result.conflicted
                                    )
                                    split_report["annotation_conflict_pairs"] += (
                                        merge_result.conflict_pairs
                                    )
                                    split_report["replaced_annotations"] += (
                                        merge_result.replaced
                                    )
                                    split_report["dropped_samples"] += 1
                                    split_report["dropped_annotations"] += max(
                                        0, original - merge_result.accepted
                                    )
                                    split_report["kept_annotations"] += (
                                        len(previous.annotations) - before_count
                                    )
                                    split_report["removed_annotations"] += removed
                                    progress.update()
                                    continue
                            else:
                                register_in = by_split
                        stem_pool = (
                            used_quarantine_stems if quarantined else used_output_stems
                        ).setdefault(split, set())
                        output_stem = allocate_flat_sample_stem(stem, stem_pool)
                        flattened_stem = stem.replace("\\", "/").replace("/", "__")
                        if output_stem != flattened_stem:
                            split_report["renamed_samples"] += 1
                        if quarantined:
                            base = stage / "quarantine" / split
                            image_target = (
                                base
                                / "images"
                                / f"{output_stem}{image_path.suffix.lower()}"
                            )
                            label_target = base / "labels" / f"{output_stem}.txt"
                        else:
                            image_target = (
                                stage
                                / "images"
                                / split
                                / f"{output_stem}{image_path.suffix.lower()}"
                            )
                            label_target = (
                                stage / "labels" / split / f"{output_stem}.txt"
                            )
                        common._transfer_file(image_path, image_target, image_mode)
                        if label_path is not None:
                            label_target.parent.mkdir(parents=True, exist_ok=True)
                            label_target.write_text(
                                _annotation_text(canonical_annotations),
                                encoding="utf-8",
                            )
                            if quarantined:
                                split_report["quarantined_labels"] += 1
                            else:
                                split_report["labels"] += 1
                        if not quarantined:
                            split_report["images"] += 1
                            split_report["kept_annotations"] += kept
                            split_report["removed_annotations"] += removed
                            if register_in is not None:
                                register_in[split] = _SeenSample(
                                    image_path=image_path,
                                    label_target=label_target,
                                    annotations=list(canonical_annotations),
                                    label_written=label_path is not None,
                                )
                        progress.update()

        if report["splits"]["train"]["images"] == 0:
            raise ValueError("所有 train 样本均被删除或隔离，输出无法用于训练")
        if "val" in report["splits"] and report["splits"]["val"]["images"] == 0:
            message = "所有 val 样本均被删除或隔离"
            if require_train_val:
                raise ValueError(message)
            report["warnings"].append(message)

        data: dict[str, Any] = {
            "path": str(output),
            "task": "segment" if inventory.kind == "yolo_instance" else "detect",
            "nc": len(resolved.output_names),
            "names": resolved.output_names,
        }
        for split in split_names:
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
                    "format": inventory.kind,
                    "taxonomy_mode": inventory.taxonomy_mode,
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


def choose_sources(
    search_root: Path,
    *,
    required_kind: str | None = None,
) -> list[Path] | None:
    candidates = [
        candidate
        for candidate in detect_datasets(
            search_root,
            kinds={required_kind} if required_kind else yolo.YOLO_KINDS,
        )
        if candidate.kind in yolo.YOLO_KINDS
        and (required_kind is None or candidate.kind == required_kind)
    ]
    print(f"\n在 {search_root} 中检测到 {len(candidates)} 个 YOLO 数据集:\n")
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"  {index:>2}. {KIND_LABELS.get(candidate.kind, candidate.kind):<18}"
            f"类别 {candidate.class_count:>4}，label {candidate.annotation_count:>7}  "
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
        help="输入 data.yaml 或 YOLO 数据集目录；至少重复两次",
    )
    parser.add_argument("--out", type=Path, help="最终合并数据集目录")
    parser.add_argument("--datasets", type=Path, help="交互扫描目录，默认仓库 datasets/")
    parser.add_argument("--plan", type=Path, help="读取已有完整类别操作计划 YAML")
    parser.add_argument(
        "--id-policy",
        choices=("compact", "preserve", "explicit"),
        help="覆盖计划中的类别 ID 策略；explicit 的映射取自计划 explicit_ids",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "yolo-detect", "yolo-seg"),
        default="auto",
    )
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink", "reflink"),
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
        "--empty-image-policy",
        choices=("keep", "drop", "quarantine"),
        default="keep",
    )
    parser.add_argument(
        "--duplicate-policy",
        choices=("keep", "error", "hash-dedupe", "drop", "merge-annotations"),
        default="keep",
    )
    parser.add_argument(
        "--duplicate-annotation-conflict-policy",
        choices=("error", "keep-first", "keep-last", "keep-both"),
        default="error",
        help=(
            "merge-annotations 遇到同位置异类或高重合几何时的处理策略"
        ),
    )
    parser.add_argument(
        "--duplicate-annotation-conflict-iou",
        type=float,
        default=0.95,
        metavar="FLOAT",
        help="merge-annotations 判定几何冲突的 IoU 阈值，默认 0.95",
    )
    parser.add_argument(
        "--split-leakage-policy", choices=("warn", "error", "drop"), default="warn"
    )
    parser.add_argument("--orphan-policy", choices=("error", "drop"), default="error")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--require-train-val",
        action="store_true",
        help="要求输出同时包含非空 train/val split",
    )
    parser.add_argument("--clean", action="store_true", help="清理已有输出后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只分析并输出合并计划")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="采用第一个来源及推荐类别方案，并跳过最终确认",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    forced_kind = {
        "auto": None,
        "yolo-detect": "yolo_detection",
        "yolo-seg": "yolo_instance",
    }[args.format]
    source_paths = args.src
    if not source_paths:
        source_paths = choose_sources(
            (args.datasets or auto_datasets_root()).resolve(),
            required_kind=forced_kind,
        )
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
    print("\n正在读取全部来源和类别...")
    inventory = build_inventory(
        source_paths,
        forced_kind=forced_kind,
        taxonomy_mode=args.taxonomy,
        mapping_file=args.mapping_file,
        orphan_policy=args.orphan_policy,
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
            annotation_label="重写对应 labels 类别 ID",
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
                    annotation_label="标注",
                    drop_effect="删除所有来源中的对应标注行",
                    unknown_effect="删除对应标注行",
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

    output = args.out or default_output_dir(inventory.sources[0].dataset.root, "merge-yolo")
    if args.out is None:
        raw = input(f"输出目录 [{output}]: ").strip()
        if raw:
            output = Path(raw)
    if not args.dry_run and not args.yes:
        answer = input(
            "确认一次性重写所有 labels、复制图片并生成最终数据集？[y/N]: "
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
        empty_image_policy=args.empty_image_policy,
        duplicate_policy=args.duplicate_policy,
        duplicate_annotation_conflict_policy=(
            args.duplicate_annotation_conflict_policy
        ),
        duplicate_annotation_conflict_iou=(
            args.duplicate_annotation_conflict_iou
        ),
        split_leakage_policy=args.split_leakage_policy,
        orphan_policy=args.orphan_policy,
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
