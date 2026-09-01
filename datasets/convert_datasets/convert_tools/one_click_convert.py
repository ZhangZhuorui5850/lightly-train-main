#!/usr/bin/env python3
"""
一键整理并转换数据集：
1. 先把原始数据整理成 train/val/test
2. 再输出统一的 dataset_det / dataset_cls / dataset_seg

示例:
    python one_click_convert.py data_a data_b -o converted_all
    python one_click_convert.py data_a -o converted_all --label-format yolo --seed 42
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

try:
    from . import LabelMeToYOLO as labelme_to_yolo
    from . import sync_picture
    from .dataset_transaction import staged_output, validate_output_location
    from .progress import tqdm
except ImportError:  # direct ``python one_click_convert.py`` execution
    import LabelMeToYOLO as labelme_to_yolo  # type: ignore[no-redef]
    import sync_picture  # type: ignore[no-redef]
    from dataset_transaction import staged_output, validate_output_location  # type: ignore[no-redef]
    from progress import tqdm  # type: ignore[no-redef]


def resolve_yolo_schemas(
    src_roots: list[Path], class_ref: str | None,
) -> tuple[list[str] | None, dict[int, dict[int, int]]]:
    """Resolve one canonical class list and per-source ID remapping."""
    source_names = [labelme_to_yolo.load_yolo_class_names_from_metadata(root) for root in src_roots]
    if class_ref:
        class_map = labelme_to_yolo.load_class_ref(Path(class_ref).expanduser().resolve())
        canonical = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    else:
        available = [names for names in source_names if names]
        canonical = list(available[0]) if available else None
        if len(src_roots) > 1 and len(available) != len(src_roots):
            raise ValueError("多源 YOLO 合并要求每个来源都有 data.yaml/classes.txt，或显式提供 --class-ref")
        if canonical is not None:
            expected = set(canonical)
            for root, names in zip(src_roots, source_names):
                if names is not None and set(names) != expected:
                    raise ValueError(
                        f"YOLO 类别集合冲突: {root}；请用 --class-ref 提供统一类别定义"
                    )
    if canonical is None:
        return None, {}
    canonical_ids = {name: index for index, name in enumerate(canonical)}
    remaps: dict[int, dict[int, int]] = {}
    for source_index, (root, names) in enumerate(zip(src_roots, source_names)):
        if names is None:
            if len(src_roots) > 1:
                raise ValueError(f"来源缺少 YOLO 类别 metadata: {root}")
            continue
        unknown = [name for name in names if name not in canonical_ids]
        if unknown:
            raise ValueError(f"来源含 --class-ref 未定义的类别: {root}: {unknown}")
        remaps[source_index] = {
            old_id: canonical_ids[name] for old_id, name in enumerate(names)
        }
    return canonical, remaps


def write_yolo_metadata(class_names: list[str] | None, synced_root: Path) -> None:
    if not class_names:
        return
    (synced_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")


def detect_pre_split(src_roots: list[Path], label_format: str) -> bool:
    """Only accept split directories containing at least one valid image/label pair."""
    for root in src_roots:
        if label_format not in {"labelme", "yolo"}:
            split_files = [root / "ImageSets" / f"{split}.txt" for split in ("train", "val", "test")]
            if sum(
                path.is_file() and bool(sync_picture.read_text_auto(path).strip())
                for path in split_files
            ) >= 2:
                continue
            return False
        present = 0
        for split in ("train", "val", "test"):
            images, labels = _collect_split_records(root, split, label_format)
            label_keys = {key for _, _, key in labels}
            if any(key in label_keys for _, _, _, key in images):
                present += 1
        if present < 2:
            return False
    return True


def _collect_split_records(root: Path, split: str, label_format: str):
    combined = root / split
    if combined.is_dir():
        return sync_picture.collect_files([combined], label_format)
    images, labels = sync_picture.collect_files([root], label_format)
    prefix = f"{split}/".casefold()
    return (
        [record for record in images if record[3][1].startswith(prefix)],
        [record for record in labels if record[2][1].startswith(prefix)],
    )


def _has_combined_split_layout(root: Path) -> bool:
    return sum((root / split).is_dir() for split in ("train", "val", "test")) >= 2


def merge_pre_split_sources(
    src_roots: list[Path],
    synced_root: Path,
    label_format: str,
    *,
    yolo_remaps: dict[int, dict[int, int]] | None = None,
) -> dict:
    """按样本对合并预划分来源，绝不独立处理图片和标注。"""
    import os

    label_suffix = ".json" if label_format == "labelme" else ".txt"
    stats = {
        "split_counts": {"train": 0, "val": 0, "test": 0},
        "skipped_conflicts": 0,
        "orphan_images": 0,
        "orphan_labels": 0,
    }

    for split in ("train", "val", "test"):
        (synced_root / split).mkdir(parents=True, exist_ok=True)

    used_per_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    def link_or_copy(source: Path, destination: Path) -> None:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)

    def copy_label(source: Path, destination: Path, source_index: int) -> None:
        remap = (yolo_remaps or {}).get(source_index)
        if label_format != "yolo" or not remap or all(old == new for old, new in remap.items()):
            link_or_copy(source, destination)
            return
        converted: list[str] = []
        for raw_line in sync_picture.read_text_auto(source).splitlines():
            parts = raw_line.split()
            if not parts:
                continue
            old_id = int(float(parts[0]))
            if old_id not in remap:
                raise ValueError(f"标注类别 ID {old_id} 不在来源 metadata 中: {source}")
            parts[0] = str(remap[old_id])
            converted.append(" ".join(parts))
        destination.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")

    for source_index, root in enumerate(tqdm(
        src_roots,
        desc="合并预划分来源",
        unit="目录",
        leave=False,
    )):
        for split in ("train", "val", "test"):
            image_records, label_records = _collect_split_records(root, split, label_format)
            if not image_records and not label_records:
                continue
            labels_by_key: dict[tuple[int, str], list[Path]] = {}
            for label_path, _stem, key in label_records:
                labels_by_key.setdefault(key, []).append(label_path)
            image_keys = {key for _, _, _, key in image_records}
            stats["orphan_labels"] += sum(
                len(paths) for key, paths in labels_by_key.items() if key not in image_keys
            )

            for image_path, original_stem, extension, key in image_records:
                label_paths = labels_by_key.get(key, [])
                if len(label_paths) != 1:
                    stats["orphan_images"] += 1
                    if len(label_paths) > 1:
                        stats["skipped_conflicts"] += 1
                    continue
                wanted = sync_picture.sanitize_stem(original_stem)
                if wanted in used_per_split[split]:
                    wanted = sync_picture.unique_stem(
                        f"src{source_index + 1}_{wanted}", used_per_split[split]
                    )
                image_destination = synced_root / split / f"{wanted}{extension}"
                label_destination = synced_root / split / f"{wanted}{label_suffix}"
                link_or_copy(image_path, image_destination)
                try:
                    copy_label(label_paths[0], label_destination, source_index)
                except BaseException:
                    image_destination.unlink(missing_ok=True)
                    raise
                used_per_split[split].add(wanted)
                stats["split_counts"][split] += 1

    stats["paired_total"] = sum(stats["split_counts"].values())
    return stats


def normalize_selected_tasks(task: str) -> tuple[str, ...]:
    task_key = task.strip().lower()
    if task_key == "all":
        return ("det", "cls", "seg")
    if task_key in {"det", "cls", "seg"}:
        return (task_key,)
    raise ValueError(f"不支持的任务类型: {task}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键整理原始数据并输出统一的 dataset_det / dataset_cls / dataset_seg",
    )
    parser.add_argument("sources", nargs="+", help="原始源目录，可多个")
    parser.add_argument(
        "-o",
        "--output-root",
        required=True,
        help="总输出目录，内部会生成 dataset_det、dataset_cls、dataset_seg",
    )
    parser.add_argument(
        "--task",
        choices=["det", "cls", "seg", "all"],
        default="all",
        help="输出任务类型：det/cls/seg/all",
    )
    parser.add_argument(
        "--label-format",
        choices=["auto", "labelme", "yolo"],
        default="auto",
        help="输入标注格式：auto 自动判断，labelme 为 .json，yolo 为 .txt",
    )
    parser.add_argument(
        "--seg-type",
        choices=["auto", "instance", "semantic"],
        default="auto",
        help="seg 输出类型：auto 自动判断，instance 为 YOLO polygon，semantic 为 PNG mask",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="划分 train/val/test 的随机种子",
    )
    parser.add_argument(
        "--preserve-splits",
        choices=["auto", "yes", "no"],
        default="auto",
        help=(
            "源目录已含 train/val/test 时如何处理："
            "auto=检测到则跳过 sync 直接用现有划分（默认），"
            "yes=强制跳过 sync（要求已预划分），"
            "no=始终走 sync 重新 8:1:1 划分"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅模拟整理阶段，不实际写文件",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="安全替换已有输出目录",
    )
    parser.add_argument(
        "--class-ref",
        default=None,
        help="参考类别文件（data.yaml / classes.txt），输出将使用其中的完整类别列表和 ID 映射",
    )
    return parser.parse_args(argv)


def _detect_seg_type(src_roots: list[Path]) -> str:
    """Auto-detect seg type from source directories."""
    return sync_picture.detect_seg_type(src_roots)


def _resolve_seg_type(seg_type: str, src_roots: list[Path]) -> str:
    """Resolve 'auto' seg_type to 'instance' or 'semantic'."""
    if seg_type != "auto":
        return seg_type
    detected = _detect_seg_type(src_roots)
    if detected == "semantic":
        return "semantic"
    return "instance"


def run_semantic_conversion(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    seed: int | None = None,
    dry_run: bool = False,
    class_ref: str | None = None,
    report_output_root: Path | None = None,
) -> None:
    """Convert VOC-style semantic segmentation dataset to dataset_semantic/."""
    src_roots = [Path(s).expanduser().resolve() for s in sources]
    for src in src_roots:
        if not src.is_dir():
            raise FileNotFoundError(f"源目录不存在: {src.resolve()}")

    output_root = Path(output_root).expanduser().resolve()
    target_dir = output_root / "dataset_semantic"
    reported_target = (
        report_output_root.expanduser().resolve() / "dataset_semantic"
        if report_output_root is not None
        else target_dir
    )

    print("=" * 60)
    print("  语义分割数据集转换")
    print("=" * 60)
    print(f"SOURCES      : {[str(p.resolve()) for p in src_roots]}")
    print(f"OUTPUT       : {reported_target}")
    print(f"DRY_RUN      : {dry_run}")

    # Check for VOC-style ImageSets
    has_imagesets = any((root / "ImageSets").is_dir() for root in src_roots)

    # Collect pairs to get stats
    pairs = sync_picture.collect_voc_semantic_files(src_roots)
    if not pairs:
        raise ValueError(
            "未找到图片+mask 配对。确保源目录有 JPEGImages/ + SegmentationClass/ 或 images/ + masks/"
        )

    print(f"PAIRS_FOUND  : {len(pairs)}")
    print(f"SPLIT_SOURCE : {'ImageSets/ 文件' if has_imagesets else '随机 8:1:1'}")

    if dry_run:
        print("\n[INFO] dry-run 模式，不实际写入文件。")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    stats = sync_picture.split_voc_semantic_files(src_roots, target_dir, seed=seed, dry_run=dry_run)

    # Scan unique class IDs from masks to generate class names
    import numpy as np
    from PIL import Image

    all_class_ids: set[int] = set()
    output_masks: dict[str, Path] = {}
    mask_walker = tqdm(
        (target_dir / "masks").rglob("*.png"),
        desc="扫描语义 mask 类别",
        unit="mask",
        leave=False,
    )
    for mask_file in mask_walker:
        output_masks[mask_file.stem] = mask_file
        with Image.open(mask_file) as img:
            vals = set(np.unique(np.array(img)).tolist())
            all_class_ids.update(vals)
        mask_walker.set_postfix_str(
            f"类别值={len(all_class_ids)} 当前={mask_file.name}", refresh=False
        )

    ignore_index = 255
    sorted_ids = sorted(class_id for class_id in all_class_ids if class_id != ignore_index)
    source_names = {
        root: labelme_to_yolo.load_yolo_class_names_from_metadata(root)
        for root in src_roots
    }
    if class_ref:
        target_map = labelme_to_yolo.load_class_ref(Path(class_ref).expanduser().resolve())
        target_names = [name for name, _ in sorted(target_map.items(), key=lambda item: item[1])]
    else:
        available_names = [names for names in source_names.values() if names]
        if available_names:
            if len(available_names) != len(src_roots):
                raise ValueError(
                    "多源语义数据合并时，每个来源都必须提供 data.yaml/classes.txt，"
                    "或显式提供 --class-ref"
                )
            target_names = list(available_names[0])
            expected = set(target_names)
            for root, names in source_names.items():
                if names is not None and set(names) != expected:
                    raise ValueError(f"语义类别集合冲突: {root}；请使用 --class-ref")
        else:
            target_names = [
                "background" if old_id == 0 else f"class_{old_id}"
                for old_id in sorted_ids
            ]

    if not target_names:
        raise ValueError("语义 mask 中除 ignore_index=255 外没有有效类别")
    if len(target_names) > ignore_index:
        raise ValueError("语义类别数量必须小于 255，避免与 ignore_index=255 冲突")

    target_ids = {name: index for index, name in enumerate(target_names)}
    id_remap_summary: dict[str, dict[int, int]] = {}
    for _image_path, original_mask, output_stem in tqdm(
        pairs,
        desc="重映射语义 mask",
        unit="mask",
        total=len(pairs),
    ):
        output_mask = output_masks.get(output_stem)
        if output_mask is None:
            continue
        root = None
        for candidate in src_roots:
            try:
                original_mask.relative_to(candidate)
                root = candidate
                break
            except ValueError:
                continue
        names = source_names.get(root) if root is not None else None
        with Image.open(output_mask) as image:
            source_array = np.array(image)
        observed = sorted(int(value) for value in np.unique(source_array) if int(value) != ignore_index)
        remap: dict[int, int] = {}
        for old_id in observed:
            if names is not None:
                if old_id >= len(names):
                    raise ValueError(f"mask 类别 ID {old_id} 超出 metadata 范围: {original_mask}")
                name = names[old_id]
                if name not in target_ids:
                    raise ValueError(f"源语义类别 {name!r} 不在目标类别中: {original_mask}")
                remap[old_id] = target_ids[name]
            elif class_ref:
                if old_id >= len(target_names):
                    raise ValueError(
                        f"mask 类别 ID {old_id} 无法映射到 --class-ref: {original_mask}"
                    )
                remap[old_id] = old_id
            else:
                remap[old_id] = sorted_ids.index(old_id)
        id_remap_summary[str(root or original_mask.parent)] = remap
        if any(old_id != new_id for old_id, new_id in remap.items()):
            remapped = np.full(source_array.shape, ignore_index, dtype=np.uint8)
            for old_id, new_id in remap.items():
                remapped[source_array == old_id] = new_id
            Image.fromarray(remapped).save(output_mask)

    class_names = {index: name for index, name in enumerate(target_names)}

    # Write classes.txt
    classes_path = target_dir / "classes.txt"
    with classes_path.open("w", encoding="utf-8") as f:
        for cid in range(len(class_names)):
            f.write(f"{class_names[cid]}\n")

    # Write data.yaml
    import yaml

    data_yaml = target_dir / "data.yaml"
    yaml_content: dict = {"task": "semantic_segmentation"}
    splits_present = [s for s in ("train", "val", "test") if (target_dir / "images" / s).is_dir()]
    for split in splits_present:
        yaml_content[split] = {
            "images": f"images/{split}",
            "masks": f"masks/{split}",
        }
    yaml_content["classes"] = class_names
    if ignore_index in all_class_ids:
        yaml_content["ignore_index"] = ignore_index

    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    # Print summary
    print()
    print("─" * 40)
    print("  转换完成")
    print("─" * 40)
    for split, count in stats.get("split_counts", {}).items():
        print(f"  {split:<8} {count} 对")
    print(f"  source ids: {sorted_ids}")
    print(f"  id remap : {id_remap_summary}")
    print(f"  output   : {reported_target}")
    if stats.get("errors"):
        print(f"\n  [WARNING] {len(stats['errors'])} 个错误:")
        for e in stats["errors"][:10]:
            print(f"    {e}")
    print()


def _run_conversion_impl(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    task: str = "all",
    label_format: str = "auto",
    seg_type: str = "auto",
    seed: int | None = None,
    dry_run: bool = False,
    preserve_splits: str = "auto",
    class_ref: str | None = None,
    report_output_root: Path | None = None,
) -> None:
    if label_format not in {"auto", "labelme", "yolo"}:
        raise ValueError(f"不支持的 label_format: {label_format}")
    if seg_type not in {"auto", "instance", "semantic"}:
        raise ValueError(f"不支持的 seg_type: {seg_type}")
    if preserve_splits not in {"auto", "yes", "no"}:
        raise ValueError(f"不支持的 preserve_splits: {preserve_splits}")
    sync_module = sync_picture
    convert_module = labelme_to_yolo
    selected_tasks = normalize_selected_tasks(task)

    src_roots = [Path(s).expanduser().resolve() for s in sources]
    for src in src_roots:
        if not src.is_dir():
            raise FileNotFoundError(f"源目录不存在: {src.resolve()}")

    output_root = Path(output_root).expanduser().resolve()
    reported_root = (
        report_output_root.expanduser().resolve()
        if report_output_root is not None
        else output_root
    )

    # Resolve seg type for seg tasks
    need_seg = "seg" in selected_tasks
    resolved_seg_type = _resolve_seg_type(seg_type, src_roots) if need_seg else "instance"

    # For semantic seg, we don't need label_format (JSON/TXT) at all
    need_instance_seg = need_seg and resolved_seg_type == "instance"
    need_det_or_cls = "det" in selected_tasks or "cls" in selected_tasks
    need_labelme_yolo = need_instance_seg or need_det_or_cls

    if need_labelme_yolo and label_format == "auto":
        label_format = sync_module.detect_label_format(src_roots)
        if label_format == "voc_seg":
            # VOC-style detected but user wants instance seg or det/cls
            # Fall back to unknown since there are no JSON/TXT labels
            if not need_instance_seg:
                label_format = "unknown"
            else:
                raise ValueError(
                    "源目录是 VOC 语义分割格式（SegmentationClass/），无 JSON/TXT 标注，无法转为 instance seg"
                )
        if label_format == "mixed":
            raise ValueError("自动检测到源目录同时包含 JSON 和 TXT，请显式指定 --label-format")
        if label_format == "unknown":
            if need_seg and resolved_seg_type == "semantic":
                # Semantic seg can still run, just skip det/cls
                print("[WARNING] 未检测到 JSON/TXT 标注，det/cls 任务将跳过，仅执行 semantic seg")
                need_det_or_cls = False
                need_labelme_yolo = False
            else:
                raise ValueError("未检测到可用标注文件（.json 或 .txt）")

    yolo_class_names: list[str] | None = None
    yolo_remaps: dict[int, dict[int, int]] = {}
    if need_labelme_yolo and label_format == "yolo":
        yolo_class_names, yolo_remaps = resolve_yolo_schemas(src_roots, class_ref)
    needs_yolo_remap = any(
        old_id != new_id
        for remap in yolo_remaps.values()
        for old_id, new_id in remap.items()
    )

    is_pre_split = detect_pre_split(src_roots, label_format)
    if preserve_splits == "yes" and not is_pre_split:
        raise ValueError("--preserve-splits=yes 但源目录没有至少两个有效 split")
    use_preserve = preserve_splits == "yes" or (preserve_splits == "auto" and is_pre_split)

    print("=" * 60)
    print("  一键转换开始")
    print("=" * 60)
    print(f"SOURCES      : {[str(p.resolve()) for p in src_roots]}")
    print(f"TASKS        : {', '.join(selected_tasks)}")
    if need_seg:
        print(f"SEG_TYPE     : {resolved_seg_type}")
    if need_labelme_yolo:
        print(f"LABEL_FORMAT : {label_format}")
    print(f"OUTPUT_ROOT  : {reported_root}")
    if class_ref:
        print(f"CLASS_REF    : {class_ref}")
    print(f"PRESERVE_SPL : {use_preserve} (mode={preserve_splits}, detected={is_pre_split})")
    if "det" in selected_tasks and (need_det_or_cls or not need_seg):
        print(f"DET_ROOT     : {reported_root / 'dataset_det'}")
    if "cls" in selected_tasks and (need_det_or_cls or not need_seg):
        print(f"CLS_ROOT     : {reported_root / 'dataset_cls'}")
    if need_seg:
        seg_root_name = "dataset_semantic" if resolved_seg_type == "semantic" else "dataset_seg"
        print(f"SEG_ROOT     : {reported_root / seg_root_name}")

    def _invoke_labelme_to_yolo(
        synced_root: Path,
        requested_tasks: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        component_tasks = tuple(requested_tasks or selected_tasks)
        if not component_tasks:
            return
        converter_task = component_tasks[0] if len(component_tasks) == 1 else "all"
        original_policy = convert_module.EXISTING_OUTPUT_POLICY  # type: ignore[reportAttributeAccessIssue]
        convert_module.EXISTING_OUTPUT_POLICY = "clean"  # type: ignore[reportAttributeAccessIssue]
        try:
            with tempfile.TemporaryDirectory(prefix="dataset_components_") as component_temp:
                component_root = Path(component_temp) / "converted"
                converter_args = [
                    "--source-root",
                    str(synced_root),
                    "--output-root",
                    str(component_root),
                    "--task",
                    converter_task,
                    "--source-format",
                    label_format,
                ]
                if class_ref:
                    converter_args.extend(["--class-ref", class_ref])
                convert_module.main(converter_args)
                directories = {
                    "det": "dataset_det",
                    "cls": "dataset_cls",
                    "seg": "dataset_seg",
                }
                for component_task in component_tasks:
                    directory_name = directories[component_task]
                    source_dir = component_root / directory_name
                    destination = output_root / directory_name
                    if not source_dir.is_dir():
                        raise RuntimeError(f"转换器未生成预期组件: {source_dir}")
                    if destination.exists():
                        raise FileExistsError(f"输出组件已存在: {destination}")
                    shutil.move(str(source_dir), str(destination))
        finally:
            convert_module.EXISTING_OUTPUT_POLICY = original_policy  # type: ignore[reportAttributeAccessIssue]

    # ── Semantic seg: self-contained path, skips LabelMeToYOLO ──────────
    if need_seg and resolved_seg_type == "semantic":
        print("\n[INFO] 语义分割转换路径")
        run_semantic_conversion(
            sources,
            output_root,
            seed=seed,
            dry_run=dry_run,
            class_ref=class_ref,
            report_output_root=reported_root,
        )
        # If there are also det/cls tasks, run them through the existing flow
        if need_det_or_cls:
            # Strip "seg" from task for LabelMeToYOLO
            remaining_tasks = [t for t in selected_tasks if t != "seg"]
            # Need to handle sync for det/cls
            if use_preserve:
                if dry_run:
                    print("\n[INFO] dry-run 模式已结束，未执行 det/cls 转换。")
                    return
                if (
                    len(src_roots) == 1
                    and not needs_yolo_remap
                    and _has_combined_split_layout(src_roots[0])
                ):
                    synced_root = src_roots[0]
                    _invoke_labelme_to_yolo(synced_root, remaining_tasks)
                else:
                    with tempfile.TemporaryDirectory(prefix="dataset_sync_") as temp_dir:
                        synced_root = Path(temp_dir) / "synced_source"
                        stats = merge_pre_split_sources(
                            src_roots, synced_root, label_format, yolo_remaps=yolo_remaps
                        )
                        if stats["paired_total"] == 0:
                            raise ValueError("多源合并后没有有效图片+标注配对")
                        if label_format == "yolo":
                            write_yolo_metadata(yolo_class_names, synced_root)
                        _invoke_labelme_to_yolo(synced_root, remaining_tasks)
            else:
                with tempfile.TemporaryDirectory(prefix="dataset_sync_") as temp_dir:
                    synced_root = Path(temp_dir) / "synced_source"
                    copy_stats = sync_module.copy_files(
                        src_roots, synced_root, label_format=label_format, seed=seed, dry_run=dry_run,
                        yolo_remaps=yolo_remaps,
                    )
                    if dry_run:
                        print("\n[INFO] dry-run 模式已结束。")
                        return
                    if copy_stats["paired_total"] == 0:
                        print("\n[WARNING] det/cls 整理阶段没有配对，跳过。")
                        return
                    if label_format == "yolo":
                        write_yolo_metadata(yolo_class_names, synced_root)
                    _invoke_labelme_to_yolo(synced_root, remaining_tasks)
        return

    # ── 分支 A：已预划分，跳过 sync 的随机洗牌 ─────────────────────────────
    if use_preserve:
        if dry_run:
            print("\n[INFO] 已检测到预划分，跳过 sync。dry-run 结束。")
            return
        if (
            len(src_roots) == 1
            and not needs_yolo_remap
            and _has_combined_split_layout(src_roots[0])
        ):
            synced_root = src_roots[0]
            print(f"[INFO] 单源已预划分，直接使用: {synced_root}")
            _invoke_labelme_to_yolo(synced_root)
            return
        # 多源：合并到临时目录，保留各源的 split 归属
        with tempfile.TemporaryDirectory(prefix="dataset_sync_") as temp_dir:
            synced_root = Path(temp_dir) / "synced_source"
            print(f"[INFO] 多源已预划分，合并到: {synced_root}")
            stats = merge_pre_split_sources(
                src_roots, synced_root, label_format, yolo_remaps=yolo_remaps
            )
            print(f"  合并完成: train={stats['split_counts']['train']} "
                  f"val={stats['split_counts']['val']} test={stats['split_counts']['test']} "
                  f"冲突跳过={stats['skipped_conflicts']}")
            if label_format == "yolo":
                write_yolo_metadata(yolo_class_names, synced_root)
            if stats["paired_total"] == 0:
                raise ValueError("合并阶段没有得到任何有效图片+标注配对")
            _invoke_labelme_to_yolo(synced_root)
            return

    # ── 分支 B：走原 sync 流程，重新 8:1:1 随机划分 ─────────────────────────
    with tempfile.TemporaryDirectory(prefix="dataset_sync_") as temp_dir:
        synced_root = Path(temp_dir) / "synced_source"
        print(f"TEMP_SYNCED  : {synced_root}")

        stats = sync_module.copy_files(
            src_roots,
            synced_root,
            label_format=label_format,
            seed=seed,
            dry_run=dry_run,
            yolo_remaps=yolo_remaps,
        )

        if label_format == "yolo" and not dry_run:
            write_yolo_metadata(yolo_class_names, synced_root)

        if dry_run:
            print("\n[INFO] dry-run 模式已结束，未执行后续转换。")
            return

        if stats["paired_total"] == 0:
            raise ValueError("整理阶段没有得到任何有效图片+标注配对")

        _invoke_labelme_to_yolo(synced_root)


def run_conversion(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    task: str = "all",
    label_format: str = "auto",
    seg_type: str = "auto",
    seed: int | None = None,
    dry_run: bool = False,
    preserve_splits: str = "auto",
    class_ref: str | None = None,
    clean: bool = False,
) -> None:
    """Validate inputs and publish a complete conversion atomically."""
    source_roots = [Path(source).expanduser().resolve() for source in sources]
    output = validate_output_location(Path(output_root), source_roots)
    kwargs = dict(
        task=task,
        label_format=label_format,
        seg_type=seg_type,
        seed=seed,
        dry_run=dry_run,
        preserve_splits=preserve_splits,
        class_ref=class_ref,
    )
    if dry_run:
        _run_conversion_impl(source_roots, output, **kwargs)
        return
    with staged_output(output, clean=clean) as stage:
        _run_conversion_impl(
            source_roots,
            stage,
            report_output_root=output,
            **kwargs,
        )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_conversion(
        args.sources,
        args.output_root,
        task=args.task,
        label_format=args.label_format,
        seg_type=args.seg_type,
        seed=args.seed,
        dry_run=args.dry_run,
        preserve_splits=args.preserve_splits,
        class_ref=args.class_ref,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
