#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from .progress import tqdm
    from .text_encoding import read_text_auto
except ImportError:
    from progress import tqdm  # type: ignore[no-redef]
    from text_encoding import read_text_auto  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="递归检查目录中的 JSON 标注，自动识别 LabelMe / COCO 并给出风险判断。",
    )
    parser.add_argument("source", help="待检查的目录或单个 JSON 文件")
    parser.add_argument("--limit", type=int, default=20, help="每类样例最多显示多少条")
    return parser.parse_args()


def iter_json_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(tqdm(
        source.rglob("*.json"),
        desc="索引 JSON 文件",
        unit="file",
        leave=False,
    ))


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(read_text_auto(path))
    except Exception:
        return None


def detect_json_format(data: Any) -> str:
    if isinstance(data, dict) and isinstance(data.get("shapes"), list):
        return "labelme"
    if (
        isinstance(data, dict)
        and isinstance(data.get("images"), list)
        and isinstance(data.get("annotations"), list)
        and isinstance(data.get("categories"), list)
    ):
        return "coco"
    return "unknown"


def infer_labelme_shape_type(shape: dict[str, Any]) -> str:
    shape_type = str(shape.get("shape_type", "")).strip().lower()
    if shape_type:
        return shape_type
    points = shape.get("points", [])
    if len(points) in {2, 4}:
        return "rectangle"
    if len(points) >= 3:
        return "polygon"
    return ""


def is_coco_polygon_segmentation(segmentation: Any) -> bool:
    return (
        isinstance(segmentation, list)
        and len(segmentation) > 0
        and all(isinstance(poly, list) and len(poly) >= 6 for poly in segmentation)
    )


def is_coco_rle_segmentation(segmentation: Any) -> bool:
    return isinstance(segmentation, dict) and "counts" in segmentation and "size" in segmentation


def print_examples(title: str, rows: list[str], limit: int) -> None:
    print(f"\n{title}: {len(rows)}")
    for row in rows[:limit]:
        print(f"  - {row}")
    if len(rows) > limit:
        print(f"  ... 仅显示前 {limit} 条")


def analyze_labelme(source: Path, json_items: list[tuple[Path, dict[str, Any]]], limit: int) -> None:
    json_count = 0
    shape_count = 0
    raw_type_counter: Counter[str] = Counter()
    inferred_type_counter: Counter[str] = Counter()
    point_count_counter: Counter[int] = Counter()
    label_counter: Counter[str] = Counter()

    missing_shape_type: list[str] = []
    polygon_with_4_points: list[str] = []
    rectangle_with_4_points: list[str] = []
    inferred_rectangle_without_shape_type: list[str] = []
    unsupported_shape_type: list[str] = []
    empty_label: list[str] = []
    per_dir_counter: Counter[str] = Counter()

    for json_file, data in json_items:
        json_count += 1
        rel_json = json_file.relative_to(source) if source.is_dir() else Path(json_file.name)
        per_dir_counter[str(rel_json.parent)] += 1
        shapes = data.get("shapes", [])

        for idx, shape in enumerate(shapes, start=1):
            raw_shape_type = str(shape.get("shape_type", "")).strip().lower()
            points = shape.get("points", [])
            point_count = len(points)
            inferred = infer_labelme_shape_type(shape)
            label = str(shape.get("label", "")).strip()

            shape_count += 1
            raw_type_counter[raw_shape_type or "(missing)"] += 1
            inferred_type_counter[inferred or "(unknown)"] += 1
            point_count_counter[point_count] += 1
            if label:
                label_counter[label] += 1

            sample_desc = (
                f"{rel_json} | shape#{idx} | label={label or '(empty)'} | "
                f"shape_type={raw_shape_type or '(missing)'} | points={point_count} | "
                f"infer={inferred or '(unknown)'}"
            )

            if not raw_shape_type:
                missing_shape_type.append(sample_desc)
            if raw_shape_type == "polygon" and point_count == 4:
                polygon_with_4_points.append(sample_desc)
            if raw_shape_type == "rectangle" and point_count == 4:
                rectangle_with_4_points.append(sample_desc)
            if not raw_shape_type and inferred == "rectangle":
                inferred_rectangle_without_shape_type.append(sample_desc)
            if raw_shape_type not in {"", "rectangle", "polygon"}:
                unsupported_shape_type.append(sample_desc)
            if not label:
                empty_label.append(sample_desc)

    print("=" * 72)
    print("LabelMe 标注检查报告")
    print("=" * 72)
    print(f"检查路径: {source}")
    print(f"识别格式: LabelMe / 每图一个 JSON")
    print(f"JSON 文件数: {json_count}")
    print(f"shape 总数: {shape_count}")

    print("\n[1] shape_type 原始字段统计")
    for name, count in raw_type_counter.most_common():
        print(f"  {name:<18} {count}")

    print("\n[2] 按当前转换脚本推断后的类型统计")
    for name, count in inferred_type_counter.most_common():
        print(f"  {name:<18} {count}")

    print("\n[3] points 数量统计")
    for count, num in point_count_counter.most_common():
        print(f"  points={count:<10} {num}")

    print("\n[4] 标签统计（前 20）")
    for label, count in label_counter.most_common(20):
        print(f"  {label:<28} {count}")

    print("\n[5] 子目录分布（前 20）")
    for dirname, count in per_dir_counter.most_common(20):
        print(f"  {dirname:<40} {count}")

    print_examples("[6] 缺失 shape_type", missing_shape_type, limit)
    print_examples("[7] 明确写 polygon 但只有 4 个点", polygon_with_4_points, limit)
    print_examples("[8] 明确写 rectangle 且有 4 个点", rectangle_with_4_points, limit)
    print_examples("[9] 缺失 shape_type 且会被兜底判成 rectangle", inferred_rectangle_without_shape_type, limit)
    print_examples("[10] 非 rectangle/polygon 的 shape_type", unsupported_shape_type, limit)
    print_examples("[11] 空标签", empty_label, limit)

    print("\n[12] 结论建议")
    if polygon_with_4_points:
        print("  - 存在大量 4 点 polygon。这种数据外观看起来像矩形，但当前规则会按 segmentation 处理。")
    if inferred_rectangle_without_shape_type:
        print("  - 存在缺失 shape_type 且会被兜底判成 rectangle 的标注，det 里可能混入非预期框。")
    if unsupported_shape_type:
        print("  - 存在非 rectangle/polygon 的 shape_type，当前转换脚本会跳过或报不支持。")
    if empty_label:
        print("  - 存在空标签，转换时会被跳过。")
    if not any([polygon_with_4_points, inferred_rectangle_without_shape_type, unsupported_shape_type, empty_label]):
        print("  - 当前 LabelMe 数据整体比较规整，明显混判风险不高。")
    print("  - LabelMe 下 det/seg 分流的优先依据是 shape_type，其次才是 points 数量兜底。")


def analyze_coco(source: Path, json_items: list[tuple[Path, dict[str, Any]]], limit: int) -> None:
    json_count = 0
    image_count = 0
    annotation_count = 0
    category_count = 0
    category_name_counter: Counter[str] = Counter()
    category_id_counter: Counter[int] = Counter()
    bbox_counter = 0
    polygon_counter = 0
    rle_counter = 0
    empty_segmentation_counter = 0
    invalid_bbox_counter = 0

    category_id_to_name_global: dict[int, str] = {}
    per_json_ann_counter: Counter[str] = Counter()
    unknown_category: list[str] = []
    polygon_with_bbox: list[str] = []
    segmentation_without_bbox: list[str] = []
    invalid_bbox_examples: list[str] = []

    for json_file, data in json_items:
        json_count += 1
        rel_json = json_file.relative_to(source) if source.is_dir() else Path(json_file.name)
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        image_count += len(images)
        annotation_count += len(annotations)
        category_count += len(categories)
        per_json_ann_counter[str(rel_json)] = len(annotations)

        category_id_to_name = {
            int(cat["id"]): str(cat.get("name", f"id_{cat['id']}"))
            for cat in categories
            if isinstance(cat, dict) and "id" in cat
        }
        category_id_to_name_global.update(category_id_to_name)

        for cat_id, cat_name in category_id_to_name.items():
            category_name_counter[cat_name] += 1

        for idx, ann in enumerate(annotations, start=1):
            cat_id = ann.get("category_id")
            if isinstance(cat_id, int):
                category_id_counter[cat_id] += 1
            cat_name = category_id_to_name_global.get(cat_id, f"id_{cat_id}")

            bbox = ann.get("bbox")
            segmentation = ann.get("segmentation")

            has_bbox = isinstance(bbox, list) and len(bbox) == 4
            has_polygon = is_coco_polygon_segmentation(segmentation)
            has_rle = is_coco_rle_segmentation(segmentation)

            if has_bbox:
                bbox_counter += 1
                try:
                    _, _, w, h = [float(v) for v in bbox]
                    if w <= 0 or h <= 0:
                        invalid_bbox_counter += 1
                        invalid_bbox_examples.append(
                            f"{rel_json} | ann#{idx} | category={cat_name} | bbox={bbox}"
                        )
                except Exception:
                    invalid_bbox_counter += 1
                    invalid_bbox_examples.append(
                        f"{rel_json} | ann#{idx} | category={cat_name} | bbox={bbox}"
                    )
            if has_polygon:
                polygon_counter += 1
                polygon_with_bbox.append(
                    f"{rel_json} | ann#{idx} | category={cat_name} | polygon_points={sum(len(poly) // 2 for poly in segmentation)} | has_bbox={has_bbox}"
                )
            if has_rle:
                rle_counter += 1
            if segmentation in (None, [], {}):
                empty_segmentation_counter += 1
            if (has_polygon or has_rle) and not has_bbox:
                segmentation_without_bbox.append(
                    f"{rel_json} | ann#{idx} | category={cat_name} | segmentation_without_bbox"
                )
            if cat_id not in category_id_to_name_global:
                unknown_category.append(
                    f"{rel_json} | ann#{idx} | category_id={cat_id}"
                )

    print("=" * 72)
    print("COCO 标注检查报告")
    print("=" * 72)
    print(f"检查路径: {source}")
    print(f"识别格式: COCO / annotations JSON")
    print(f"JSON 文件数: {json_count}")
    print(f"image 总数: {image_count}")
    print(f"annotation 总数: {annotation_count}")
    print(f"category 条目总数: {category_count}")

    print("\n[1] 标签统计（按 annotation 次数，前 20）")
    category_ann_counter = Counter(
        {category_id_to_name_global.get(cat_id, f'id_{cat_id}'): count for cat_id, count in category_id_counter.items()}
    )
    for name, count in category_ann_counter.most_common(20):
        print(f"  {name:<28} {count}")

    print("\n[2] 标注结构统计")
    print(f"  具有 bbox 的 annotation           {bbox_counter}")
    print(f"  具有 polygon segmentation 的 annotation {polygon_counter}")
    print(f"  具有 RLE segmentation 的 annotation     {rle_counter}")
    print(f"  segmentation 为空/缺失               {empty_segmentation_counter}")
    print(f"  非法 bbox（宽高<=0 或格式异常）      {invalid_bbox_counter}")

    print("\n[3] 每个 annotations JSON 的 annotation 数（前 20）")
    for name, count in per_json_ann_counter.most_common(20):
        print(f"  {name:<48} {count}")

    print_examples("[4] polygon segmentation 样例", polygon_with_bbox, limit)
    print_examples("[5] segmentation 存在但没有 bbox", segmentation_without_bbox, limit)
    print_examples("[6] category_id 找不到 category 定义", unknown_category, limit)
    print_examples("[7] 非法 bbox 样例", invalid_bbox_examples, limit)

    print("\n[8] 结论建议")
    if polygon_counter > 0:
        print("  - 数据里包含 segmentation，多半不是纯检测集。若只做 det，需要先确认是否保留 bbox、忽略 segmentation。")
    if polygon_counter > 0 and bbox_counter > 0:
        print("  - 同时存在 bbox 和 segmentation，这通常是标准 COCO 检测/实例分割混合标注，不是转换错误。")
    if segmentation_without_bbox:
        print("  - 存在 segmentation 但没有 bbox 的 annotation，某些检测流程可能无法直接使用。")
    if invalid_bbox_counter > 0:
        print("  - 存在非法 bbox，训练或转换时可能报错。")
    if unknown_category:
        print("  - 存在 category_id 未定义的问题，建议先修标注文件。")
    if not any([polygon_counter, segmentation_without_bbox, invalid_bbox_counter, unknown_category]):
        print("  - 当前 COCO 数据整体比较规整，明显结构问题不多。")
    print("  - COCO 里 rectangle/polygon 不是靠 shape_type 区分，而是看 annotation 的 bbox 和 segmentation 字段。")


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"路径不存在: {source}")

    json_files = iter_json_files(source)
    if not json_files:
        raise FileNotFoundError(f"未发现 JSON 文件: {source}")

    labelme_items: list[tuple[Path, dict[str, Any]]] = []
    coco_items: list[tuple[Path, dict[str, Any]]] = []
    unknown_items: list[str] = []
    parse_failed: list[str] = []

    for json_file in tqdm(
        json_files,
        desc="解析 JSON 标注",
        unit="file",
        total=len(json_files),
    ):
        data = load_json(json_file)
        if not isinstance(data, dict):
            parse_failed.append(str(json_file))
            continue
        fmt = detect_json_format(data)
        if fmt == "labelme":
            labelme_items.append((json_file, data))
        elif fmt == "coco":
            coco_items.append((json_file, data))
        else:
            unknown_items.append(str(json_file))

    print("=" * 72)
    print("JSON 标注格式探测")
    print("=" * 72)
    print(f"检查路径: {source}")
    print(f"递归发现 JSON: {len(json_files)}")
    print(f"LabelMe JSON: {len(labelme_items)}")
    print(f"COCO JSON:    {len(coco_items)}")
    print(f"未知结构:     {len(unknown_items)}")
    print(f"解析失败:     {len(parse_failed)}")

    if unknown_items:
        print_examples("[A] 未知结构 JSON", unknown_items, args.limit)
    if parse_failed:
        print_examples("[B] 解析失败 JSON", parse_failed, args.limit)

    if labelme_items and coco_items:
        print("\n[提示] 同一个目录下同时发现 LabelMe 和 COCO，两类报告都会输出。")

    if labelme_items:
        print()
        analyze_labelme(source, labelme_items, args.limit)

    if coco_items:
        print()
        analyze_coco(source, coco_items, args.limit)

    if not labelme_items and not coco_items:
        print("\n没有发现可识别的 LabelMe 或 COCO 标注结构。")


if __name__ == "__main__":
    main()
