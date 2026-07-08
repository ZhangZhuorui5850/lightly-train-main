"""目标尺寸均衡补充工具（size-supplement）的核心逻辑。

以一个已有的（通常按类别均衡导出的）检测数据集为基底，从源全量库增量
挑选“小/中/大目标丰富”的图加入，使按框数量计的尺寸占比逼近用户目标，
产出一套全新数据集（不修改原数据），并写出报告。

详见设计文档 docs/superpowers/specs/2026-05-29-size-supplement-design.md。
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import common as rt
from .det_export import (
    _bucket_box_area,
    _open_image_size,
    build_export_dataset_eda,
    render_export_dataset_eda_markdown,
)
from .det_shared import ExportImageCandidate, collect_source_image_infos
from .progress import track

BUCKET_NAMES = ("tiny", "small", "medium", "large")
RATIO_BUCKET_NAMES = ("small", "medium", "large")


def _empty_buckets() -> dict[str, int]:
    return {name: 0 for name in BUCKET_NAMES}


def remap_lines_by_name(
    label_lines: Sequence[str],
    *,
    source_id_to_name: dict[int, str],
    name_to_id: dict[str, int],
    allow_new_classes: bool = False,
) -> tuple[list[str], dict[str, int], set[str]]:
    """把源数据集的 YOLO 标签行按「类别名字」重映射到目标 id 空间。

    基底导出时 id 被重排过，所以只能靠名字对齐。name 不在 name_to_id 中时：
    allow_new_classes=False 丢弃该框（默认，不新增类别）；True 则追加新类别。
    返回 (重映射后的行, 可能扩展过的 name_to_id, 本批次新增的类别名集合)。
    """
    name_to_id = dict(name_to_id)
    added_names: set[str] = set()
    remapped: list[str] = []
    for line in label_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            source_id = int(float(parts[0]))
        except ValueError:
            continue
        name = source_id_to_name.get(source_id, str(source_id))
        if name in name_to_id:
            new_id = name_to_id[name]
        elif allow_new_classes:
            new_id = (max(name_to_id.values()) + 1) if name_to_id else 0
            name_to_id[name] = new_id
            added_names.add(name)
        else:
            continue
        remapped.append(" ".join([str(new_id), *parts[1:]]))
    return remapped, name_to_id, added_names


class ImageSizeCache:
    """图片像素尺寸缓存，按 绝对路径+mtime 命中，落盘为 JSON，避免重复 PIL 打开。

    size_reader 默认用 det_export._open_image_size，可在测试中注入。
    """

    def __init__(
        self,
        cache_path: Path,
        *,
        size_reader: Callable[[Path], tuple[int, int]] = _open_image_size,
    ) -> None:
        self.cache_path = Path(cache_path)
        self._size_reader = size_reader
        self._entries: dict[str, list[float]] = {}
        self._dirty = False
        if self.cache_path.exists():
            try:
                payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    self._entries = {
                        str(key): list(value)
                        for key, value in payload.items()
                        if isinstance(value, (list, tuple)) and len(value) == 3
                    }
            except Exception:
                self._entries = {}

    def get(self, image_path: Path) -> tuple[int, int]:
        image_path = Path(image_path)
        key = str(image_path.resolve())
        try:
            mtime = image_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        cached = self._entries.get(key)
        if cached is not None and cached[2] == mtime:
            return int(cached[0]), int(cached[1])
        width, height = self._size_reader(image_path)
        self._entries[key] = [int(width), int(height), mtime]
        self._dirty = True
        return int(width), int(height)

    def save(self) -> None:
        if not self._dirty and self.cache_path.exists():
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._entries, ensure_ascii=False),
            encoding="utf-8",
        )
        self._dirty = False


def bucket_label_lines(
    label_lines: Sequence[str],
    *,
    width: int,
    height: int,
    with_class_counts: bool = False,
):
    """统计一组 YOLO 标签行的小/中/大框数（按像素面积分桶）。

    label_lines 为 "cls cx cy w h"（归一化）。malformed 行被忽略。
    with_class_counts=True 时额外返回 {class_id: 框数}。
    """
    buckets = _empty_buckets()
    class_box_counts: dict[int, int] = {}
    for line in label_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            box_width = max(float(parts[3]), 0.0) * width
            box_height = max(float(parts[4]), 0.0) * height
        except ValueError:
            continue
        bucket = _bucket_box_area(box_width * box_height)
        buckets[bucket] += 1
        class_box_counts[class_id] = class_box_counts.get(class_id, 0) + 1
    if with_class_counts:
        return buckets, class_box_counts
    return buckets


def parse_size_ratio(text: str) -> dict[str, float]:
    """把 "33/33/33" / "30 30 40" / "1:1:2" 解析成归一化的 small/medium/large 占比。"""
    parts = [piece for piece in re.split(r"[\s/:,]+", text.strip()) if piece]
    if len(parts) != 3:
        raise ValueError(f"目标比例需要 3 个值（小/中/大），收到 {len(parts)} 个：{text!r}")
    try:
        values = [float(piece) for piece in parts]
    except ValueError as exc:
        raise ValueError(f"目标比例必须是数字：{text!r}") from exc
    if any(value < 0 for value in values):
        raise ValueError(f"目标比例不能为负：{text!r}")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"目标比例之和必须大于 0：{text!r}")
    return {name: value / total for name, value in zip(RATIO_BUCKET_NAMES, values)}


def _bucket_props(buckets: dict[str, int]) -> dict[str, float]:
    total = sum(buckets.get(name, 0) for name in RATIO_BUCKET_NAMES)
    if total <= 0:
        return {name: 0.0 for name in RATIO_BUCKET_NAMES}
    return {name: buckets.get(name, 0) / total for name in RATIO_BUCKET_NAMES}


@dataclass(frozen=True)
class SupplementCandidate:
    """一张可被补充进来的源图的尺寸/类别贡献摘要。

    key 用于去重与稳定排序（通常是源图相对路径）。
    """

    key: str
    buckets: dict[str, int]
    class_box_counts: dict[int, int] = field(default_factory=dict)

    @property
    def total_boxes(self) -> int:
        return sum(self.buckets.get(name, 0) for name in BUCKET_NAMES)


def _class_coverage_bonus(
    candidate: SupplementCandidate,
    class_counts: dict[int, int],
) -> float:
    """候选图对“低于平均框数的弱势类别”的贡献度（软兼顾类别用）。"""
    if not class_counts or not candidate.class_box_counts:
        return 0.0
    mean = sum(class_counts.values()) / len(class_counts)
    bonus = 0.0
    for class_id, box_count in candidate.class_box_counts.items():
        if class_counts.get(class_id, 0) < mean:
            bonus += box_count
    return bonus


def select_supplement_candidates(
    *,
    base_buckets: dict[str, int],
    candidates: Sequence[SupplementCandidate],
    target_ratio: dict[str, float],
    num_to_add: int,
    base_class_box_counts: dict[int, int] | None = None,
    class_bonus_weight: float = 0.0,
    over_penalty: float = 1.0,
    batch_size: int = 200,
    max_add: int | None = None,
    stop_when_satisfied: bool = True,
    tolerance: float = 0.02,
) -> tuple[list[SupplementCandidate], dict[str, Any]]:
    """贪心赤字驱动地从候选池里挑图，使尺寸占比逼近 target_ratio。

    num_to_add 是「至少补多少张」的下限（floor）；达到下限后，如果亏空桶离目标
    仍超过 tolerance 且还有「能改善比例」的候选，就继续往上加、可超过 num_to_add，
    直到达标 / 没有能再改善的候选 / 触及 max_add / 源池耗尽。

    每批开始按当前最缺哪个尺寸桶重算赤字权重打分；缺小目标就优先选小目标多的图，
    对超标桶给惩罚；同等贡献下按 class_bonus_weight 软性偏好能补弱势类别的图。
    返回 (选中列表, 统计摘要)。
    """
    current = {name: int(base_buckets.get(name, 0)) for name in BUCKET_NAMES}
    class_counts: dict[int, int] = dict(base_class_box_counts or {})
    remaining = list(candidates)
    floor = min(max(int(num_to_add), 0), len(remaining))
    hard_cap = len(remaining) if max_add is None else min(max(int(max_add), 0), len(remaining))
    hard_cap = max(hard_cap, floor)
    effective_batch = max(1, int(batch_size))

    def _deficits_met(props: dict[str, float]) -> bool:
        return all(target_ratio.get(name, 0.0) - props.get(name, 0.0) <= tolerance for name in BUCKET_NAMES)

    selected: list[SupplementCandidate] = []
    while remaining and len(selected) < hard_cap:
        in_floor = len(selected) < floor
        props = _bucket_props(current)
        if not in_floor and stop_when_satisfied and _deficits_met(props):
            break

        deficit = {name: max(0.0, target_ratio.get(name, 0.0) - props.get(name, 0.0)) for name in BUCKET_NAMES}
        over = {name: max(0.0, props.get(name, 0.0) - target_ratio.get(name, 0.0)) for name in BUCKET_NAMES}

        def _score(candidate: SupplementCandidate) -> float:
            gain = sum(deficit[name] * candidate.buckets.get(name, 0) for name in BUCKET_NAMES)
            penalty = over_penalty * sum(over[name] * candidate.buckets.get(name, 0) for name in BUCKET_NAMES)
            score = gain - penalty
            if class_bonus_weight:
                score += class_bonus_weight * _class_coverage_bonus(candidate, class_counts)
            return score

        ranked = sorted(remaining, key=lambda c: (-_score(c), c.key))
        added_keys: set[str] = set()
        for candidate in ranked:
            if len(selected) >= hard_cap or len(added_keys) >= effective_batch:
                break
            if not in_floor and _score(candidate) <= 0:
                break  # 没有能再改善比例的候选了
            for name in BUCKET_NAMES:
                current[name] += candidate.buckets.get(name, 0)
            for class_id, box_count in candidate.class_box_counts.items():
                class_counts[class_id] = class_counts.get(class_id, 0) + box_count
            selected.append(candidate)
            added_keys.add(candidate.key)
            if in_floor and len(selected) >= floor:
                break  # 填满下限后回到外层，进入增长阶段重新评估
            if not in_floor and stop_when_satisfied and _deficits_met(_bucket_props(current)):
                break
        if not added_keys:
            break
        remaining = [candidate for candidate in remaining if candidate.key not in added_keys]

    achieved_total = sum(current.values())
    achieved_ratio = _bucket_props(current)
    satisfied = _deficits_met(achieved_ratio)
    bucket_shortfall = {
        name: max(0, int(round(target_ratio.get(name, 0.0) * achieved_total)) - current[name])
        for name in BUCKET_NAMES
    }
    summary: dict[str, Any] = {
        "requested_add": int(num_to_add),
        "selected_add": len(selected),
        "satisfied": satisfied,
        "pool_exhausted": (not remaining) and not satisfied,
        "base_buckets": {name: int(base_buckets.get(name, 0)) for name in BUCKET_NAMES},
        "achieved_buckets": dict(current),
        "achieved_ratio": achieved_ratio,
        "target_ratio": dict(target_ratio),
        "tolerance": tolerance,
        "bucket_shortfall": bucket_shortfall,
    }
    return selected, summary


def allocate_supplements_to_splits(total: int, split_ratio: str = "8:1:1") -> dict[str, int]:
    """把 total 张补充图按 split_ratio（默认 8:1:1）分到 train/val/test，整数且和为 total。"""
    parts = [float(piece) for piece in re.split(r"[:/\s,]+", split_ratio.strip()) if piece]
    if len(parts) != 3 or sum(parts) <= 0:
        raise ValueError(f"split_ratio 需要 3 个正值，收到：{split_ratio!r}")
    weight_sum = sum(parts)
    raw = [total * part / weight_sum for part in parts]
    floors = [int(math.floor(value)) for value in raw]
    remainder = total - sum(floors)
    order = sorted(range(3), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in range(remainder):
        floors[order[i]] += 1
    return {"train": floors[0], "val": floors[1], "test": floors[2]}


def build_output_dir_name(
    *,
    base_name: str,
    achieved_ratio: dict[str, float],
    total_images: int,
) -> str:
    """生成体现“做了什么”的新数据集目录名，如 <base>__szsup_s33m30l37_n15000。"""
    small = int(round(achieved_ratio.get("small", 0.0) * 100))
    medium = int(round(achieved_ratio.get("medium", 0.0) * 100))
    large = int(round(achieved_ratio.get("large", 0.0) * 100))
    return f"{base_name}__szsup_s{small}m{medium}l{large}_n{total_images}"


def infer_source_dataset_name(base_name: str) -> str:
    """从基底目录名推断默认源数据集目录名，去掉导出后缀。

    dataset_det_A_10000 -> dataset_det；dataset_det_A__szsup_..._n15000 -> dataset_det。
    """
    name = re.sub(r"__szsup_.*$", "", base_name)
    name = re.sub(r"_A(_\d+)?$", "", name)
    return name


def summarize_size(data_yaml: Path) -> dict[str, Any]:
    """快速汇总一个检测数据集的图数与小/中/大尺寸框数及占比（用于交互预览）。"""
    rt.import_runtime_dependencies()
    cfg = rt.load_data_config(Path(data_yaml))
    root = Path(cfg["_root_dir"])
    infos_by_split, _ = collect_source_image_infos(cfg, root)
    image_count = sum(len(infos) for infos in infos_by_split.values())
    cache = ImageSizeCache(root / ".imgsize_cache.json")
    buckets, class_counts = _scan_buckets(infos_by_split, cache)
    cache.save()
    return {
        "image_count": image_count,
        "size_buckets": buckets,
        "size_ratio": _bucket_props(buckets),
        "class_box_counts": class_counts,
    }


def _scan_buckets(infos_by_split, cache: ImageSizeCache):
    """汇总一个数据集的 小/中/大 框数与每类框数。"""
    buckets = _empty_buckets()
    class_counts: dict[int, int] = {}
    for infos in infos_by_split.values():
        for info in infos:
            width, height = cache.get(info.src_image_path)
            image_buckets, image_classes = bucket_label_lines(
                info.label_lines, width=width, height=height, with_class_counts=True
            )
            for name in BUCKET_NAMES:
                buckets[name] += image_buckets[name]
            for class_id, count in image_classes.items():
                class_counts[class_id] = class_counts.get(class_id, 0) + count
    return buckets, class_counts


def _copy_info_to_split(info, dst_root: Path, split: str, used: set[str], lines: Sequence[str] | None = None) -> None:
    name = info.rel_path.name
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = name
    counter = 1
    while candidate in used:
        candidate = f"{stem}__{counter}{suffix}"
        counter += 1
    used.add(candidate)
    dst_image = dst_root / "images" / split / candidate
    dst_label = dst_root / "labels" / split / f"{Path(candidate).stem}.txt"
    dst_image.parent.mkdir(parents=True, exist_ok=True)
    dst_label.parent.mkdir(parents=True, exist_ok=True)
    label_lines = list(info.label_lines if lines is None else lines)
    shutil.copy2(info.src_image_path, dst_image)
    dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")


def _names_in_order(names: dict[int, str]) -> list[str]:
    return [names[key] for key in sorted(names)]


def _generate_standard_eda(out_root: Path, base_names: dict[int, str], source_data_path: Path) -> dict[str, str]:
    """复用 det_export 的 EDA：扫描产出数据集，写 export_dataset_eda_<tag>.json/.md。"""
    out_cfg = rt.load_data_config(out_root / "data.yaml")
    infos_by_split, _ = collect_source_image_infos(out_cfg, out_root)
    candidates_by_split: dict[str, list[ExportImageCandidate]] = {}
    for split, infos in infos_by_split.items():
        candidates_by_split[split] = [
            ExportImageCandidate(
                split_name=split,
                rel_split_image_dir=info.rel_split_image_dir,
                rel_split_label_dir=info.rel_split_label_dir,
                rel_path=info.rel_path,
                src_image_path=info.src_image_path,
                src_label_path=info.src_label_path,
                filtered_lines=info.label_lines,
                class_box_counts=info.class_box_counts,
            )
            for info in infos
        ]
    class_id_mapping = {class_id: class_id for class_id in base_names}
    eda = build_export_dataset_eda(
        selected_candidates_by_split=candidates_by_split,
        kept_names=base_names,
        class_id_mapping=class_id_mapping,
    )
    markdown = render_export_dataset_eda_markdown(
        export_dataset_eda=eda,
        export_root=out_root,
        source_data_path=source_data_path,
        report_path=None,
    )
    tag = rt.dataset_tag_from_dir(out_root)
    json_path = out_root / f"export_dataset_eda_{tag}.json"
    md_path = out_root / f"export_dataset_eda_{tag}.md"
    json_path.write_text(json.dumps(eda, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _render_supplement_markdown(report: dict[str, Any]) -> str:
    base = report["base_size_ratio"]
    final = report["final_size_ratio"]
    base_counts = report["base_size_buckets"]
    final_counts = report["final_size_buckets"]
    lines = [
        "# 目标尺寸均衡补充报告",
        "",
        "## 1. 基本信息",
        f"- 基底数据集：`{report['base_root']}`",
        f"- 源数据池：`{report['source_root']}`",
        f"- 产出数据集：`{report['output_root']}`",
        f"- 生成时间：{report['generated_at']}",
        f"- 目标比例（小/中/大）：{report['target_ratio']['small']:.3f} / "
        f"{report['target_ratio']['medium']:.3f} / {report['target_ratio']['large']:.3f}",
        f"- 目标总图数：{report['target_total_images']}，实际总图数：{report['final_total_images']}",
        f"- 基底图数：{report['base_total_images']}，补充图数：{report['supplemented_images']}",
        f"- 源池去重剔除（已在基底中）：{report['deduplicated_from_source']} 张",
        f"- 源池可用候选：{report['source_candidate_images']} 张",
        "",
        "## 2. 尺寸占比 before / after（按框数量）",
        "",
        "| 尺寸 | 基底框数 | 基底占比 | 最终框数 | 最终占比 | 目标占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in BUCKET_NAMES:
        lines.append(
            f"| {name} | {base_counts.get(name, 0)} | {base.get(name, 0.0):.3f} | "
            f"{final_counts.get(name, 0)} | {final.get(name, 0.0):.3f} | {report['target_ratio'].get(name, 0.0):.3f} |"
        )
    lines += [
        "",
        "## 3. 补充图的 split 分配（8:1:1）",
        "",
        "| split | 补充图数 |",
        "|---|---:|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {report['supplement_split_allocation'][split]} |")
    lines += ["", "## 4. 各类别框数 delta（最终 - 基底）", "", "| 类别 | 基底框数 | 最终框数 | delta |", "|---|---:|---:|---:|"]
    for class_id in sorted(report["class_box_counts_final"], key=int):
        base_c = report["class_box_counts_base"].get(class_id, 0)
        final_c = report["class_box_counts_final"][class_id]
        lines.append(f"| {report['class_names'].get(class_id, class_id)} | {base_c} | {final_c} | {final_c - base_c:+d} |")
    if report.get("used_fallback"):
        added = report.get("added_class_names", [])
        lines += [
            "",
            "## 5. 自动降级：允许新类别",
            "",
            f"- base-only 过滤后候选不足（不足 {report['requested_supplement']} 张），自动启用 allow_new_classes。",
            f"- 新增类别 {len(added)} 个：{', '.join(added[:20])}{'...' if len(added) > 20 else ''}。",
        ]
    elif report.get("non_base_only_images", 0) > 0:
        lines += [
            "",
            "## 5. 类别过滤情况",
            "",
            f"- 源池中有 {report['non_base_only_images']} 张图仅含非基底类别，已跳过（未新增类别）。",
        ]
    if report["pool_exhausted"]:
        section = "## 6" if report.get("used_fallback") or report.get("non_base_only_images", 0) > 0 else "## 5"
        lines += [
            "",
            f"{section}. 供给不足提醒",
            "",
            f"- 源池候选不足以补满目标图数，目标补充 {report['requested_supplement']} 张，实际只补了 "
            f"{report['supplemented_images']} 张。",
        ]
        shortfall = report["bucket_shortfall"]
        if any(shortfall[name] > 0 for name in BUCKET_NAMES):
            gaps = ", ".join(f"{name} 还差 {shortfall[name]} 框" for name in BUCKET_NAMES if shortfall[name] > 0)
            lines.append(f"- 距离目标比例仍有缺口：{gaps}。")
    lines.append("")
    return "\n".join(lines)


def run_size_supplement(
    *,
    base_data_yaml: Path,
    source_data_yaml: Path,
    target_ratio: dict[str, float],
    target_total_images: int,
    split_ratio: str = "8:1:1",
    allow_new_classes: bool = False,
    class_bonus_weight: float = 0.25,
    over_penalty: float = 1.0,
    batch_size: int = 200,
    tolerance: float = 0.02,
) -> Path:
    """以基底数据集为底，从源池增量补充，使尺寸占比逼近 target_ratio，产出新数据集。

    类别按「名字」对齐（基底导出时 id 被重排过）。allow_new_classes=False（默认）只保留
    名字在基底里的框、其余丢弃，绝不新增类别；True 则把源池新类别追加进来。

    target_total_images 是「至少补到这么多图」的下限；若补满后尺寸比例仍未达标且源池还有
    能改善比例的图，会继续往上加、可超过该数字。不修改基底与源池；返回新数据集根目录。
    """
    rt.import_runtime_dependencies()
    base_data_yaml = Path(base_data_yaml)
    source_data_yaml = Path(source_data_yaml)
    base_cfg = rt.load_data_config(base_data_yaml)
    source_cfg = rt.load_data_config(source_data_yaml)
    base_root = Path(base_cfg["_root_dir"])
    source_root = Path(source_cfg["_root_dir"])

    base_names = rt.normalize_names(base_cfg.get("names"))
    source_names = rt.normalize_names(source_cfg.get("names"))

    # 按名字对齐：基底类别名 -> 基底 id。新类别在 fallback 阶段按需追加。
    name_to_id = {name: class_id for class_id, name in base_names.items()}
    # 源池中所有类别名（含非基底的），供 fallback 使用。
    all_source_class_names = set(source_names.values())

    base_infos_by_split, _ = collect_source_image_infos(base_cfg, base_root)
    source_infos_by_split, _ = collect_source_image_infos(source_cfg, source_root)
    base_total_images = sum(len(infos) for infos in base_infos_by_split.values())

    num_to_add = int(target_total_images) - base_total_images
    if num_to_add <= 0:
        raise ValueError(
            f"目标总图数 ({target_total_images}) 必须大于基底图数 ({base_total_images})；"
            "本工具只做增量补充。"
        )

    cache = ImageSizeCache(source_root / ".imgsize_cache.json")
    base_buckets, base_class_counts = _scan_buckets(base_infos_by_split, cache)

    base_stems = {
        info.rel_path.stem
        for infos in base_infos_by_split.values()
        for info in infos
    }
    # --- Pass 1: base-only（不新增类别）---
    candidates: list[SupplementCandidate] = []
    candidate_info_by_key: dict[str, Any] = {}
    candidate_lines_by_key: dict[str, list[str]] = {}
    non_base_images: list[tuple[str, Any, list[str]]] = []  # (key, info, label_lines)
    deduplicated = 0
    dropped_no_base_boxes = 0
    for split, infos in source_infos_by_split.items():
        for info in track(infos, label=f"det/size-supp 扫描 {split}", total=len(infos), unit="img"):
            if info.rel_path.stem in base_stems:
                deduplicated += 1
                continue
            remapped_lines, _, _ = remap_lines_by_name(
                info.label_lines,
                source_id_to_name=source_names,
                name_to_id=name_to_id,
                allow_new_classes=False,
            )
            if not remapped_lines:
                # 这张图没有任何 base 类的框 → 暂存，供 fallback 使用
                non_base_images.append(
                    (f"{split}/{info.rel_path.as_posix()}", info, list(info.label_lines))
                )
                continue
            width, height = cache.get(info.src_image_path)
            buckets, class_counts = bucket_label_lines(
                remapped_lines, width=width, height=height, with_class_counts=True
            )
            key = f"{split}/{info.rel_path.as_posix()}"
            candidates.append(
                SupplementCandidate(key=key, buckets=buckets, class_box_counts=class_counts)
            )
            candidate_info_by_key[key] = info
            candidate_lines_by_key[key] = remapped_lines

    # --- Pass 2（fallback）: base-only 候选不足时，允许新类别 ---
    used_fallback = False
    if len(candidates) < num_to_add and non_base_images:
        if allow_new_classes:
            used_fallback = True
            # 把源池中非基底类别追加到 name_to_id 映射
            next_id = (max(name_to_id.values()) + 1) if name_to_id else 0
            for name in sorted(all_source_class_names - set(name_to_id)):
                name_to_id[name] = next_id
                next_id += 1
            for key, info, raw_lines in non_base_images:
                remapped_lines, _, _ = remap_lines_by_name(
                    raw_lines,
                    source_id_to_name=source_names,
                    name_to_id=name_to_id,
                    allow_new_classes=True,
                )
                if not remapped_lines:
                    dropped_no_base_boxes += 1
                    continue
                width, height = cache.get(info.src_image_path)
                buckets, class_counts = bucket_label_lines(
                    remapped_lines, width=width, height=height, with_class_counts=True
                )
                candidates.append(
                    SupplementCandidate(key=key, buckets=buckets, class_box_counts=class_counts)
                )
                candidate_info_by_key[key] = info
                candidate_lines_by_key[key] = remapped_lines
        else:
            dropped_no_base_boxes += len(non_base_images)
    else:
        dropped_no_base_boxes += len(non_base_images)

    # final_names 在 fallback 之后计算，确保包含可能追加的新类别。
    final_names = {class_id: name for name, class_id in name_to_id.items()}

    cache.save()

    if not candidates:
        raise ValueError(
            "源池去重并按基底类别过滤后，没有可补充的候选图。"
            + (f" 有 {len(non_base_images)} 张图仅含非基底类别，"
               "可设置 allow_new_classes=True 尝试。" if non_base_images else "")
        )

    selected, selection_summary = select_supplement_candidates(
        base_buckets=base_buckets,
        candidates=candidates,
        target_ratio=target_ratio,
        num_to_add=num_to_add,
        base_class_box_counts=base_class_counts,
        class_bonus_weight=class_bonus_weight,
        over_penalty=over_penalty,
        batch_size=batch_size,
        tolerance=tolerance,
    )

    split_alloc = allocate_supplements_to_splits(len(selected), split_ratio=split_ratio)
    supplement_keys_by_split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    cursor = 0
    for split in ("train", "val", "test"):
        take = split_alloc[split]
        for candidate in selected[cursor : cursor + take]:
            supplement_keys_by_split[split].append(candidate.key)
        cursor += take

    final_total_images = base_total_images + len(selected)
    achieved_ratio = selection_summary["achieved_ratio"]
    output_name = build_output_dir_name(
        base_name=base_root.name,
        achieved_ratio=achieved_ratio,
        total_images=final_total_images,
    )
    final_root = rt.deduplicate_path(base_root.parent / output_name)
    temp_root = rt.deduplicate_path(base_root.parent / f"{base_root.name}__supplement_tmp")
    temp_root.mkdir(parents=True, exist_ok=True)

    renamed = False
    try:
        used_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
        for split, infos in base_infos_by_split.items():
            for info in infos:
                _copy_info_to_split(info, temp_root, split, used_by_split.setdefault(split, set()))
        for split, keys in supplement_keys_by_split.items():
            for key in keys:
                _copy_info_to_split(
                    candidate_info_by_key[key],
                    temp_root,
                    split,
                    used_by_split.setdefault(split, set()),
                    lines=candidate_lines_by_key[key],
                )

        rt.dump_yaml(
            temp_root / "data.yaml",
            {
                "path": str(final_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "task": base_cfg.get("task", "detect"),
                "nc": len(final_names),
                "names": final_names,
            },
        )
        (temp_root / "classes.txt").write_text(
            "\n".join(_names_in_order(final_names)) + "\n", encoding="utf-8"
        )

        supplement_class_counts: dict[int, int] = {}
        for candidate in selected:
            for class_id, count in candidate.class_box_counts.items():
                supplement_class_counts[class_id] = supplement_class_counts.get(class_id, 0) + count
        final_class_counts = dict(base_class_counts)
        for class_id, count in supplement_class_counts.items():
            final_class_counts[class_id] = final_class_counts.get(class_id, 0) + count

        report: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_root": str(base_root),
            "source_root": str(source_root),
            "output_root": str(final_root),
            "base_data_yaml": str(base_data_yaml),
            "source_data_yaml": str(source_data_yaml),
            "target_ratio": dict(target_ratio),
            "target_total_images": int(target_total_images),
            "final_total_images": final_total_images,
            "base_total_images": base_total_images,
            "supplemented_images": len(selected),
            "requested_supplement": num_to_add,
            "exceeded_target_total": final_total_images > int(target_total_images),
            "satisfied": selection_summary["satisfied"],
            "source_candidate_images": len(candidates),
            "deduplicated_from_source": deduplicated,
            "dropped_no_base_boxes": dropped_no_base_boxes,
            "non_base_only_images": len(non_base_images),
            "used_fallback": used_fallback,
            "allow_new_classes": allow_new_classes,
            "added_class_names": sorted(set(final_names.values()) - set(base_names.values())),
            "split_ratio": split_ratio,
            "supplement_split_allocation": split_alloc,
            "base_size_buckets": dict(base_buckets),
            "final_size_buckets": selection_summary["achieved_buckets"],
            "base_size_ratio": _bucket_props(base_buckets),
            "final_size_ratio": achieved_ratio,
            "bucket_shortfall": selection_summary["bucket_shortfall"],
            "pool_exhausted": selection_summary["pool_exhausted"],
            "class_names": {str(class_id): name for class_id, name in final_names.items()},
            "class_box_counts_base": {str(k): v for k, v in base_class_counts.items()},
            "class_box_counts_final": {str(k): v for k, v in final_class_counts.items()},
            "selection_params": {
                "class_bonus_weight": class_bonus_weight,
                "over_penalty": over_penalty,
                "batch_size": batch_size,
                "tolerance": tolerance,
            },
        }
        (temp_root / "supplement_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (temp_root / "supplement_report.md").write_text(
            _render_supplement_markdown(report), encoding="utf-8"
        )

        temp_root.rename(final_root)
        renamed = True
    finally:
        if not renamed and temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)

    try:
        _generate_standard_eda(final_root, final_names, base_data_yaml)
    except Exception as exc:  # EDA 是附加产物，失败不影响主产出
        (final_root / "export_dataset_eda_ERROR.txt").write_text(
            f"标准 EDA 生成失败：{exc!r}\n", encoding="utf-8"
        )

    return final_root
