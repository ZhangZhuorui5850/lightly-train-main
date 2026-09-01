"""语义分割数据集交互式类别整理。

读取 EDA 产出的 image_class_inventory CSV，交互式选择删除类 + 压缩阈值，
共现感知下采样 train，硬链接 materialize 新数据集。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from . import common as rt
from . import train_tools
from . import seg_tools
from . import interactive
from .progress import track

SPLIT_ORDER = ("train", "val", "test")


# ---------------------------------------------------------------------------
# inventory 加载
# ---------------------------------------------------------------------------

def _load_inventory(inventory_csv: Path) -> list[dict[str, Any]]:
    """加载 image_class_inventory CSV。"""
    import csv
    rows: list[dict[str, Any]] = []
    with open(inventory_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["width"] = int(row["width"])
            row["height"] = int(row["height"])
            row["total_labeled_pixels"] = int(row["total_labeled_pixels"])
            row["class_ids"] = [int(x) for x in row["class_ids"].split("|") if x] if row["class_ids"] else []
            row["per_class_pixels"] = {int(k): v for k, v in json.loads(row["per_class_pixels"]).items()}
            rows.append(row)
    return rows


def _find_inventory_csv(data_path: Path, eda_dir: Path | None) -> Path:
    """查找 inventory CSV。优先用 eda_dir，否则自动查找最近的 EDA。"""
    if eda_dir is not None:
        candidates = list(Path(eda_dir).glob("image_class_inventory_*.csv"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"在 {eda_dir} 中未找到 image_class_inventory_*.csv")

    # 自动查找：在 EDA_OUTPUT_ROOT_DIR 下找最近的 semantic-eda 目录
    eda_root = rt.EDA_OUTPUT_ROOT_DIR
    if eda_root.is_dir():
        candidates = sorted(
            eda_root.glob("*-semantic-eda"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            csvs = list(candidate.glob("image_class_inventory_*.csv"))
            if csvs:
                print(f"  [INFO] 自动找到 EDA 目录: {candidate}")
                return csvs[0]

    # 回退：现场扫描
    print("  [WARN] 未找到 EDA inventory，将现场扫描 mask（建议先跑 eda）")
    return None  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 交互
# ---------------------------------------------------------------------------

def _build_class_table(
    inventory: list[dict[str, Any]],
    classes: dict[int, Any],
    ignore_classes: set[int],
    rec: dict[str, Any] | None,
) -> tuple[list[int], dict[int, str], dict[str, int]]:
    """构建带编号类别表，返回 (sorted_class_ids, class_names, train_image_counts)。"""
    class_names: dict[int, str] = {}
    for class_id, info in classes.items():
        if isinstance(info, dict):
            class_names[class_id] = str(info.get("name", class_id))
        else:
            class_names[class_id] = str(info)

    # 统计每类 train 图片数
    train_image_counts: Counter = Counter()
    all_image_counts: Counter = Counter()
    for row in inventory:
        for cid in row["class_ids"]:
            all_image_counts[cid] += 1
            if row["split"] == "train":
                train_image_counts[cid] += 1

    sorted_class_ids = sorted(cid for cid in classes if cid not in ignore_classes)

    # 打印表
    print(f"\n{'─' * 70}")
    print(f"  {'#':>3} {'ID':>4} {'名称':<20} {'Train':>6} {'All':>6} {'标记'}")
    print(f"{'─' * 70}")
    for idx, cid in enumerate(sorted_class_ids):
        train_n = train_image_counts.get(cid, 0)
        all_n = all_image_counts.get(cid, 0)
        marks: list[str] = []
        if rec:
            if cid in rec.get("recommend_delete", []):
                marks.append("[建议删除]")
            if cid in rec.get("recommend_compress", []):
                marks.append("[建议压缩]")
        mark_str = " ".join(marks)
        print(f"  {idx + 1:>3} {cid:>4} {class_names.get(cid, str(cid)):<20} {train_n:>6} {all_n:>6} {mark_str}")
    print(f"{'─' * 70}")

    return sorted_class_ids, class_names, dict(train_image_counts)


def _parse_class_ids(input_str: str, valid_ids: set[int]) -> list[int]:
    """解析逗号分隔的类别 ID 输入。"""
    if not input_str or not input_str.strip():
        return []
    result: list[int] = []
    for part in input_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            cid = int(part)
        except ValueError:
            raise ValueError(f"无效的类别 ID: '{part}'（请输入整数）")
        if cid not in valid_ids:
            raise ValueError(f"未知的类别 ID: {cid}（可选: {sorted(valid_ids)}）")
        if cid not in result:
            result.append(cid)
    return result


def _interactive_select(
    inventory: list[dict[str, Any]],
    classes: dict[int, Any],
    ignore_classes: set[int],
    rec: dict[str, Any] | None,
) -> tuple[list[int], int]:
    """交互式选择删除类和压缩阈值。返回 (drop_class_ids, image_threshold)。"""
    sorted_ids, _, _ = _build_class_table(inventory, classes, ignore_classes, rec)

    # 默认推荐删除
    default_delete = rec.get("recommend_delete", []) if rec else []
    default_delete_str = ",".join(str(cid) for cid in default_delete)

    # 输入删除 ID
    while True:
        raw = interactive.prompt_text(
            "输入要删除的类别 ID（逗号分隔，留空不删除）",
            default=default_delete_str if default_delete_str else None,
        )
        if not raw:
            drop_ids = []
            break
        try:
            drop_ids = _parse_class_ids(raw, set(sorted_ids))
            break
        except ValueError as e:
            print(f"  [ERROR] {e}")

    # 输入压缩阈值
    default_threshold = rec.get("recommend_threshold", 0) if rec else 0
    image_threshold = interactive.prompt_int(
        "压缩阈值（train 每类最多保留图片数，0=不压缩）",
        default=default_threshold,
    )

    return drop_ids, image_threshold


# ---------------------------------------------------------------------------
# 共现感知下采样
# ---------------------------------------------------------------------------

def _cooccurrence_aware_downsample(
    train_rows: list[dict[str, Any]],
    drop_class_ids: set[int],
    image_threshold: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """共现感知下采样 train 集。返回 (保留的 train rows, manifest_info)。

    策略：
    1. 删除类后，每张图的「保留类集合」与每个保留类的 train 图片数。
    2. 标记超阈值类（excess）= train 图片数 > 阈值的保留类。
    3. 可丢池 = 保留类集合 ⊆ excess 的 train 图。
    4. 贪心丢弃：优先丢覆盖最多 excess 类、最少稀有类的图。
    5. 可丢池耗尽但仍有 excess 类超阈值 → 停止并告警。
    """
    if image_threshold <= 0:
        return train_rows, {"skipped": True, "reason": "threshold=0"}

    # Step 1: 计算删除类后每图的保留类集合
    retained_sets: list[set[int]] = []
    for row in train_rows:
        retained = set(row["class_ids"]) - drop_class_ids
        retained_sets.append(retained)

    # Step 2: 统计每个保留类的 train 图片数
    class_image_count: Counter = Counter()
    for retained in retained_sets:
        for cid in retained:
            class_image_count[cid] += 1

    # 找出超阈值类（excess）
    excess_classes = {cid for cid, cnt in class_image_count.items() if cnt > image_threshold}
    if not excess_classes:
        return train_rows, {
            "skipped": False,
            "reason": "no_excess_classes",
            "excess_classes": [],
            "dropped_count": 0,
            "residual_excess": {},
        }

    # Step 3: 构建可丢池 = 保留类集合 ⊆ excess 的图
    droppable_indices: list[int] = []
    for i, retained in enumerate(retained_sets):
        if retained and retained.issubset(excess_classes):
            droppable_indices.append(i)

    # Step 4: 贪心丢弃
    # 排序：覆盖最多 excess 类优先，覆盖最少非 excess 类（稀有类）优先
    def _sort_key(idx: int) -> tuple[int, int]:
        retained = retained_sets[idx]
        n_excess = len(retained & excess_classes)
        n_rare = len(retained - excess_classes)
        return (-n_excess, n_rare)

    droppable_indices.sort(key=_sort_key)

    keep_mask = [True] * len(train_rows)
    dropped_count = 0

    for idx in droppable_indices:
        # 检查是否所有 excess 类都已降到阈值以下
        if all(class_image_count[cid] <= image_threshold for cid in excess_classes):
            break
        # 丢弃这张图
        keep_mask[idx] = False
        dropped_count += 1
        for cid in retained_sets[idx]:
            class_image_count[cid] -= 1

    # Step 5: 检查残余超额
    residual_excess = {
        cid: class_image_count[cid]
        for cid in excess_classes
        if class_image_count[cid] > image_threshold
    }

    if residual_excess:
        print(f"  [WARN] 可丢池已耗尽，以下类仍超阈值（未牺牲稀有类）:")
        for cid, cnt in residual_excess.items():
            print(f"    class {cid}: {cnt} > {image_threshold}")

    kept_rows = [row for row, keep in zip(train_rows, keep_mask) if keep]

    return kept_rows, {
        "skipped": False,
        "excess_classes": sorted(excess_classes),
        "dropped_count": dropped_count,
        "residual_excess": residual_excess,
        "class_image_count_after": dict(class_image_count),
    }


# ---------------------------------------------------------------------------
# materialize 新数据集
# ---------------------------------------------------------------------------

def _link_or_copy(src: Path, dst: Path) -> None:
    """硬链接，跨设备回退拷贝。"""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _build_contiguous_id_mapping(
    retain_ids: list[int],
) -> dict[int, int]:
    """构建 old_id → new_id (0..K-1) 映射。"""
    return {old_id: new_id for new_id, old_id in enumerate(retain_ids)}


def _rewrite_mask_pixels(
    mask_path: Path,
    id_mapping: dict[int, int],
    ignore_source_ids: set[int],
) -> None:
    """就地改写 mask 像素值：按映射表重映射，不在映射中的像素设为 255。"""
    from PIL import Image as _Image
    import numpy as _np

    with _Image.open(mask_path) as img:
        arr = _np.array(img)

    # 构建 LUT：映射内→新 ID，源 ignore→255，其余→255
    max_val = int(arr.max()) if arr.size > 0 else 0
    lut = _np.full(max_val + 1, 255, dtype=_np.uint8)
    for old_id, new_id in id_mapping.items():
        if old_id <= max_val:
            lut[old_id] = new_id
    for sid in ignore_source_ids:
        if sid <= max_val:
            lut[sid] = 255

    new_arr = lut[_np.clip(arr, 0, max_val)]
    _Image.fromarray(new_arr).save(mask_path)


def _materialize_curated_dataset(
    *,
    source_data_path: Path,
    train_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    drop_class_ids: list[int],
    image_threshold: int,
    curate_manifest: dict[str, Any],
    export_suffix: str = "__curated",
    contiguous_ids: bool = False,
) -> Path:
    """硬链接 materialize 新数据集，返回新数据集根目录。"""
    source_data_path = source_data_path.expanduser().resolve()
    source_root = source_data_path.parent

    # 新目录名
    new_root = rt.deduplicate_path(Path(str(source_root) + export_suffix))
    new_root.mkdir(parents=True, exist_ok=True)

    # 加载源配置
    data_cfg = train_tools.load_semantic_segmentation_data_config(source_data_path)
    classes = data_cfg["classes"]
    ignore_classes = {int(item) for item in data_cfg.get("ignore_classes", set()) or set()}

    # 保留类
    retain_classes = {cid for cid in classes if cid not in set(drop_class_ids) and cid not in ignore_classes}

    # contiguous_ids 模式：构建 old→new (0..K-1) 映射
    contiguous_mapping: dict[int, int] | None = None
    if contiguous_ids:
        sorted_retain = sorted(retain_classes)
        contiguous_mapping = _build_contiguous_id_mapping(sorted_retain)
        curate_manifest["contiguous_id_mapping"] = {
            str(old): new for old, new in contiguous_mapping.items()
        }

    # 新 classes：只保留保留类（删除类省略 → 加载器自动 ignore）
    new_classes: dict[int, Any] = {}
    if contiguous_ids and contiguous_mapping is not None:
        # classes key 用新编号
        inv_mapping = {v: k for k, v in contiguous_mapping.items()}
        for new_id in range(len(inv_mapping)):
            old_id = inv_mapping[new_id]
            new_classes[new_id] = classes[old_id]
    else:
        for cid in sorted(classes.keys()):
            if cid in retain_classes:
                new_classes[cid] = classes[cid]

    # 构建 train 图片集合
    train_image_set = {row["image_path"] for row in train_rows}

    # 链接文件
    linked_train = 0
    linked_val = 0
    linked_test = 0
    total_rows = len(all_rows)

    for row in track(all_rows, label="seg-curate/链接文件", total=total_rows, unit="file"):
        split = row["split"]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        # 确定目标路径
        # 从源配置获取 split 的 images/masks 目录
        split_cfg = data_cfg.get(split, {})
        src_images_dir = Path(split_cfg.get("images", ""))
        src_masks_dir = Path(split_cfg.get("masks", ""))

        if not src_images_dir or not src_masks_dir:
            continue

        # 相对路径
        try:
            rel_image = image_path.relative_to(src_images_dir)
        except ValueError:
            rel_image = Path(image_path.name)

        # 新的目标目录
        new_split_images = new_root / split / "images"
        new_split_masks = new_root / split / "masks"
        new_split_images.mkdir(parents=True, exist_ok=True)
        new_split_masks.mkdir(parents=True, exist_ok=True)

        new_image_path = new_split_images / rel_image
        new_mask_path = (new_split_masks / rel_image).with_suffix(".png")

        # train 分割只链接选中的图
        if split == "train" and str(image_path) not in train_image_set:
            continue

        _link_or_copy(image_path, new_image_path)
        if contiguous_ids and contiguous_mapping is not None:
            # 需要改写像素，先拷贝再就地重映射
            shutil.copy2(mask_path, new_mask_path)
            _rewrite_mask_pixels(new_mask_path, contiguous_mapping, ignore_classes)
        else:
            _link_or_copy(mask_path, new_mask_path)

        if split == "train":
            linked_train += 1
        elif split == "val":
            linked_val += 1
        elif split == "test":
            linked_test += 1

    # 写新 data.yaml
    new_yaml: dict[str, Any] = {
        "path": str(new_root),
        "train": {
            "images": str(new_root / "train" / "images"),
            "masks": str(new_root / "train" / "masks"),
        },
        "val": {
            "images": str(new_root / "val" / "images"),
            "masks": str(new_root / "val" / "masks"),
        },
        "classes": new_classes,
    }
    # 如果源有 test split，也加上
    if (new_root / "test").is_dir():
        new_yaml["test"] = {
            "images": str(new_root / "test" / "images"),
            "masks": str(new_root / "test" / "masks"),
        }
    # 保留 ignore_classes
    if ignore_classes:
        new_yaml["ignore_classes"] = sorted(ignore_classes)

    import yaml
    yaml_path = new_root / "data.yaml"
    yaml_path.write_text(yaml.dump(new_yaml, allow_unicode=True, default_flow_style=False, sort_keys=False), encoding="utf-8")

    # 写 manifest
    curate_manifest["output_path"] = str(new_root)
    curate_manifest["yaml_path"] = str(yaml_path)
    curate_manifest["linked_images"] = {"train": linked_train, "val": linked_val, "test": linked_test}
    manifest_path = new_root / "curate_manifest.json"
    manifest_path.write_text(json.dumps(curate_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return new_root


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def run_semantic_curate(args: argparse.Namespace) -> None:
    """交互式语义分割类别整理。"""
    source_data_path = Path(args.data).expanduser().resolve()
    eda_dir = getattr(args, "eda_dir", None)
    eda_dir = Path(eda_dir) if eda_dir else None
    export_suffix = getattr(args, "export_suffix", "__curated")
    contiguous_ids = bool(getattr(args, "contiguous_ids", False))

    # CLI 覆盖交互默认值
    cli_drop_classes = getattr(args, "drop_classes", None)
    cli_image_threshold = getattr(args, "image_threshold", None)

    # 加载配置
    data_cfg = train_tools.load_semantic_segmentation_data_config(source_data_path)
    classes = data_cfg["classes"]
    ignore_classes = {int(item) for item in data_cfg.get("ignore_classes", set()) or set()}

    # 查找 inventory CSV
    inventory_csv = _find_inventory_csv(source_data_path, eda_dir)
    if inventory_csv is None:
        # 回退：现场扫描
        print("  [INFO] 现场扫描 mask 生成 inventory...")
        from .seg_semantic_eda import _collect_semantic_eda
        report, inventory_rows = _collect_semantic_eda(source_data_path)
        rec = report.get("recommendations")
    else:
        print(f"  [INFO] 加载 inventory: {inventory_csv}")
        inventory_rows = _load_inventory(inventory_csv)
        # 尝试加载 EDA report 获取推荐
        eda_json = inventory_csv.parent / "semantic_eda_semantic.json"
        if eda_json.exists():
            report_data = json.loads(eda_json.read_text(encoding="utf-8"))
            rec = report_data.get("recommendations")
        else:
            rec = None

    # 交互选择
    if cli_drop_classes is not None and cli_image_threshold is not None:
        # CLI 模式：跳过交互
        drop_ids = [int(x) for x in str(cli_drop_classes).split(",") if x.strip()]
        image_threshold = int(cli_image_threshold)
        print(f"\n  [CLI 模式] 删除类: {drop_ids}, 压缩阈值: {image_threshold}")
    else:
        # 交互模式
        drop_ids, image_threshold = _interactive_select(inventory_rows, classes, ignore_classes, rec)

    drop_set = set(drop_ids)
    retain_ids = [cid for cid in sorted(classes.keys()) if cid not in drop_set and cid not in ignore_classes]

    if not drop_ids and image_threshold <= 0:
        print("\n  [WARN] 无删除且无阈值，跳过导出。")
        return

    # 按 split 分组
    train_rows = [r for r in inventory_rows if r["split"] == "train"]
    val_rows = [r for r in inventory_rows if r["split"] == "val"]
    test_rows = [r for r in inventory_rows if r["split"] == "test"]

    # 共现感知下采样（仅 train）
    kept_train_rows, downsample_info = _cooccurrence_aware_downsample(
        train_rows, drop_set, image_threshold
    )

    # 构建 manifest
    curate_manifest: dict[str, Any] = {
        "source_data": str(source_data_path),
        "eda_inventory": str(inventory_csv) if inventory_csv else None,
        "drop_classes": drop_ids,
        "retain_classes": retain_ids,
        "image_threshold": image_threshold,
        "downsample": downsample_info,
        "train_images_before": len(train_rows),
        "train_images_after": len(kept_train_rows),
        "val_images": len(val_rows),
        "test_images": len(test_rows),
    }

    # 打印整理前后对照表
    print(f"\n{'─' * 70}")
    print(f"  整理前后对照")
    print(f"{'─' * 70}")
    print(f"  {'类别':<20} {'Train Before':>12} {'Train After':>12} {'操作'}")
    print(f"{'─' * 70}")

    # 统计整理前后每类 train 图片数
    before_counts: Counter = Counter()
    for row in train_rows:
        for cid in row["class_ids"]:
            if cid not in drop_set:
                before_counts[cid] += 1
    after_counts: Counter = Counter()
    for row in kept_train_rows:
        for cid in row["class_ids"]:
            if cid not in drop_set:
                after_counts[cid] += 1

    class_names: dict[int, str] = {}
    for cid, info in classes.items():
        if isinstance(info, dict):
            class_names[cid] = str(info.get("name", cid))
        else:
            class_names[cid] = str(info)

    for cid in sorted(classes.keys()):
        if cid in ignore_classes:
            continue
        name = class_names.get(cid, str(cid))
        if cid in drop_set:
            print(f"  {name:<20} {'-':>12} {'-':>12} 删除")
        else:
            before = before_counts.get(cid, 0)
            after = after_counts.get(cid, 0)
            action = "压缩" if after < before else ""
            print(f"  {name:<20} {before:>12} {after:>12} {action}")
    print(f"{'─' * 70}")

    # materialize
    print(f"\n  [INFO] 正在生成新数据集...")
    new_root = _materialize_curated_dataset(
        source_data_path=source_data_path,
        train_rows=kept_train_rows,
        all_rows=inventory_rows,
        drop_class_ids=drop_ids,
        image_threshold=image_threshold,
        curate_manifest=curate_manifest,
        export_suffix=export_suffix,
        contiguous_ids=contiguous_ids,
    )

    # 输出
    print(f"\n{'=' * 63}")
    print(f"  整理完成！")
    print(f"{'=' * 63}")
    print(f"\n  新数据集: {new_root}")
    print(f"  - data.yaml: {new_root / 'data.yaml'}")
    print(f"  - manifest: {new_root / 'curate_manifest.json'}")
    print(f"  - Train 图片: {len(kept_train_rows)} (原 {len(train_rows)})")
    print(f"  - 删除类: {drop_ids if drop_ids else '无'}")
    print(f"  - 压缩阈值: {image_threshold if image_threshold > 0 else '不压缩'}")

    if downsample_info.get("residual_excess"):
        print(f"\n  [WARN] 残余超额（可丢池耗尽，未牺牲稀有类）:")
        for cid, cnt in downsample_info["residual_excess"].items():
            print(f"    class {cid}: {cnt} > {image_threshold}")

    print()
