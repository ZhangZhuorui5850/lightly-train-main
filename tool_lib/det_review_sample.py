"""数据集质检抽样（det-review-sample）的核心逻辑。

实现 plan0603.md 中 Step 0~Step 4 的完整流程：
- 扫描数据集 + 一致性校验
- 几何 / 层级问题检测（dup / conflict / bad_box / dense）
- 每类配额计算 + 贪心 set-cover 选图
- 导出可在 X-AnyLabeling 复核的子集（LabelMe JSON + YOLO + 报告）

Step 3（模型推理错漏标）不在本模块内，留待 CLI `--with-model` 扩展。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import common as rt
from .det_shared import read_yolo_label_lines
from .progress import track

# ── 默认阈值 ──────────────────────────────────────────────
IOU_DUP = 0.9
CENTER_DIST_PX = 5
SIZE_RATIO = 0.95
IOU_CONFLICT = 0.5
MIN_BOX_PX = 4
MAX_AREA_RATIO = 0.95
DENSE_TOP_N = 20
DEFAULT_K = 3
DEFAULT_ALPHA = 3.0
DEFAULT_CAP = 10
DEFAULT_SEED = 42


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class BoxAnnotation:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    flags: set[str] = field(default_factory=set)


@dataclass
class ImageReviewInfo:
    rel_path: Path
    split: str
    src_image_path: Path
    src_label_path: Path
    boxes: list[BoxAnnotation]
    image_width: int = 0
    image_height: int = 0
    image_flags: set[str] = field(default_factory=set)
    # 模型分析相关字段
    model_flags: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pred_json_path: Path | None = None

    @property
    def class_ids(self) -> set[int]:
        return {box.class_id for box in self.boxes}

    @property
    def has_problem(self) -> bool:
        return bool(self.image_flags) or bool(self.model_flags) or any(box.flags for box in self.boxes)

    @property
    def problem_score(self) -> float:
        # 几何问题分数
        geometry_score = float(
            len(self.image_flags) + sum(len(box.flags) for box in self.boxes)
        )
        # 模型问题分数（权重更高）
        model_score = 0.0
        model_weights = {
            "missing": 2.0,    # 漏标权重高
            "swapped": 2.5,    # 错标权重最高
            "false_pos": 1.5,  # 误检权重中
            "loc": 1.0,        # 框定位差权重低
            "low_conf": 0.5,   # 低置信度权重最低
        }
        for flag_type, items in self.model_flags.items():
            model_score += len(items) * model_weights.get(flag_type, 1.0)
        return geometry_score + model_score


@dataclass
class ClassGroup:
    group_name: str
    relation: str
    class_ids: frozenset[int]


@dataclass
class HealthCheckReport:
    total_images: int = 0
    total_boxes: int = 0
    nc: int = 0
    id_out_of_bounds: list[tuple[str, int]] = field(default_factory=list)
    zero_instance_classes: list[int] = field(default_factory=list)
    class_image_counts: dict[int, int] = field(default_factory=dict)
    class_box_counts: dict[int, int] = field(default_factory=dict)
    mismatched_pairs: list[str] = field(default_factory=list)


# ── 图尺寸缓存 ───────────────────────────────────────────


class ImageSizeCache:
    """JSON 持久化的 图片尺寸缓存，key=abs_path，value=[w, h, mtime]。"""

    def __init__(
        self,
        cache_path: Path,
        *,
        size_reader: Callable[[Path], tuple[int, int]] | None = None,
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
                        str(k): list(v)
                        for k, v in payload.items()
                        if isinstance(v, (list, tuple)) and len(v) == 3
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
        if self._size_reader is not None:
            width, height = self._size_reader(image_path)
        else:
            width, height = _pil_image_size(image_path)
        self._entries[key] = [int(width), int(height), mtime]
        self._dirty = True
        return int(width), int(height)

    def save(self) -> None:
        if not self._dirty and self.cache_path.exists():
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._entries, ensure_ascii=False), encoding="utf-8"
        )
        self._dirty = False


def _pil_image_size(image_path: Path) -> tuple[int, int]:
    if rt.Image is None:
        rt.ensure_plot_dependencies()
    if rt.Image is None:
        raise ModuleNotFoundError("Pillow not available")
    with rt.Image.open(image_path) as img:
        w, h = img.size
    return int(w), int(h)


# ── Step 0: 扫描 ─────────────────────────────────────────


def scan_dataset(
    data_path: Path,
    *,
    cache: ImageSizeCache | None = None,
) -> tuple[list[ImageReviewInfo], dict[int, str], int, HealthCheckReport]:
    """扫描 data.yaml + labels/*.txt，返回 (infos, names, nc, report)。"""
    cfg = rt.load_data_config(Path(data_path))
    names = rt.normalize_names(cfg.get("names"))
    nc = int(cfg.get("nc", len(names)))

    report = HealthCheckReport(nc=nc)
    infos: list[ImageReviewInfo] = []
    class_img_counter: dict[int, int] = {}
    class_box_counter: dict[int, int] = {}

    for split in ("train", "val", "test"):
        if not cfg.get(split):
            continue
        try:
            img_dir, lbl_dir, _ = rt.resolve_dataset_split_paths(cfg, split)
        except (FileNotFoundError, ValueError):
            continue
        if not img_dir.exists():
            continue

        for rel_img in track(
            rt.file_helpers.list_image_filenames_from_dir(image_dir=img_dir),
            label=f"det/review 校验 {split}", unit="img",
        ):
            rel_path = Path(rel_img)
            src_img = img_dir / rel_path
            src_lbl = (lbl_dir / rel_path.with_suffix(".txt")) if lbl_dir else None

            if src_lbl is None or not src_lbl.exists():
                report.mismatched_pairs.append(f"{split}/{rel_path}")
                continue

            lines, _ = read_yolo_label_lines(src_lbl)
            boxes: list[BoxAnnotation] = []
            for line in lines:
                parts = line.split()
                cid = int(float(parts[0]))
                if cid >= nc or cid < 0:
                    report.id_out_of_bounds.append((f"{split}/{rel_path}", cid))
                boxes.append(
                    BoxAnnotation(
                        class_id=cid,
                        cx=float(parts[1]),
                        cy=float(parts[2]),
                        w=float(parts[3]),
                        h=float(parts[4]),
                    )
                )

            w, h = (0, 0)
            if cache is not None:
                w, h = cache.get(src_img)
            else:
                try:
                    w, h = _pil_image_size(src_img)
                except Exception:
                    pass

            info = ImageReviewInfo(
                rel_path=rel_path,
                split=split,
                src_image_path=src_img,
                src_label_path=src_lbl,
                boxes=boxes,
                image_width=w,
                image_height=h,
            )
            infos.append(info)

            seen_classes: set[int] = set()
            for box in boxes:
                class_box_counter[box.class_id] = class_box_counter.get(box.class_id, 0) + 1
                seen_classes.add(box.class_id)
            for cid in seen_classes:
                class_img_counter[cid] = class_img_counter.get(cid, 0) + 1

    report.total_images = len(infos)
    report.total_boxes = sum(class_box_counter.values())
    report.class_image_counts = class_img_counter
    report.class_box_counts = class_box_counter
    for cid in range(nc):
        if cid not in class_img_counter:
            report.zero_instance_classes.append(cid)

    return infos, names, nc, report


# ── Step 1: class_groups ──────────────────────────────────


def load_class_groups(
    groups_path: Path,
    names: dict[int, str],
    section: str = "coco80",
) -> list[ClassGroup]:
    """解析 class_groups.yaml 中指定 section，返回 ClassGroup 列表。"""
    import yaml as _yaml

    text = groups_path.read_text(encoding="utf-8")
    raw = _yaml.safe_load(text)
    if not isinstance(raw, dict):
        return []

    section_data = raw.get(section)
    if not isinstance(section_data, list):
        return []

    name_to_id = {v: k for k, v in names.items()}
    groups: list[ClassGroup] = []
    for entry in section_data:
        if not isinstance(entry, dict):
            continue
        gname = entry.get("group", "")
        relation = entry.get("relation", "sibling")
        class_names = entry.get("classes", [])
        resolved = frozenset(
            name_to_id[n] for n in class_names if n in name_to_id
        )
        if len(resolved) >= 2:
            groups.append(
                ClassGroup(group_name=gname, relation=relation, class_ids=resolved)
            )
    return groups


# ── Step 2: 几何 + 层级 flag ──────────────────────────────


def _norm_box_xyxy(box: BoxAnnotation, w: int, h: int) -> tuple[float, float, float, float]:
    x1 = (box.cx - box.w / 2) * w
    y1 = (box.cy - box.h / 2) * h
    x2 = (box.cx + box.w / 2) * w
    y2 = (box.cy + box.h / 2) * h
    return x1, y1, x2, y2


def _xyxy_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_center_dist_px(
    a: BoxAnnotation, b: BoxAnnotation, w: int, h: int
) -> float:
    dx = (a.cx - b.cx) * w
    dy = (a.cy - b.cy) * h
    return math.sqrt(dx * dx + dy * dy)


def _box_wh_ratio(a: BoxAnnotation, b: BoxAnnotation) -> float:
    # compare w and h ratios
    if a.w <= 0 or b.w <= 0 or a.h <= 0 or b.h <= 0:
        return 0.0
    rw = min(a.w, b.w) / max(a.w, b.w)
    rh = min(a.h, b.h) / max(a.h, b.h)
    return min(rw, rh)


def detect_geometry_flags(
    infos: list[ImageReviewInfo],
    class_groups: list[ClassGroup] | None = None,
    *,
    iou_dup: float = IOU_DUP,
    center_dist_px: float = CENTER_DIST_PX,
    size_ratio_thresh: float = SIZE_RATIO,
    iou_conflict: float = IOU_CONFLICT,
    min_box_px: float = MIN_BOX_PX,
    max_area_ratio: float = MAX_AREA_RATIO,
    dense_top_n: int | None = None,
) -> None:
    """就地给每框/每图打 flag（dup / conflict / bad_box / dense）。"""
    if dense_top_n is None:
        dense_top_n = max(DENSE_TOP_N, len(infos) // 100)

    # 建立 sibling 关系索引：{(cid_a, cid_b) sorted → group_name}
    sibling_pairs: dict[tuple[int, int], str] = {}
    if class_groups:
        for grp in class_groups:
            if grp.relation != "sibling":
                continue
            ids = sorted(grp.class_ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    key = (ids[i], ids[j])
                    sibling_pairs[key] = grp.group_name

    # 记录每图框数用于 dense 选 top-N
    box_counts: list[tuple[int, int]] = []  # (idx, count)

    for idx, info in enumerate(infos):
        w = info.image_width
        h = info.image_height
        if w <= 0 or h <= 0:
            try:
                w, h = _pil_image_size(info.src_image_path)
                info.image_width = w
                info.image_height = h
            except Exception:
                pass

        n_boxes = len(info.boxes)
        box_counts.append((idx, n_boxes))

        # 预计算 xyxy
        xyxys: list[tuple[float, float, float, float]] = []
        for box in info.boxes:
            xyxys.append(_norm_box_xyxy(box, w, h))

        # ── bad_box ──
        for i, box in enumerate(info.boxes):
            bw_px = box.w * w
            bh_px = box.h * h
            area_px = bw_px * bh_px
            img_area = w * h

            # 极小
            if bw_px < min_box_px or bh_px < min_box_px:
                box.flags.add("bad_box:tiny")
            # 整图
            if img_area > 0 and area_px / img_area > max_area_ratio:
                box.flags.add("bad_box:full_image")
            # degenerate (wh≈0)
            if box.w <= 0 or box.h <= 0:
                box.flags.add("bad_box:degenerate")
            # out of bounds
            x1, y1, x2, y2 = xyxys[i]
            if x1 < -1 or y1 < -1 or x2 > w + 1 or y2 > h + 1:
                box.flags.add("bad_box:out_of_bounds")

        # ── dup ──
        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                a, b = info.boxes[i], info.boxes[j]
                iou = _xyxy_iou(xyxys[i], xyxys[j])
                if iou < iou_dup:
                    continue
                cd = _box_center_dist_px(a, b, w, h)
                sr = _box_wh_ratio(a, b)
                if cd <= center_dist_px and sr >= size_ratio_thresh:
                    flag = "dup"
                    a.flags.add(flag)
                    b.flags.add(flag)

        # ── conflict ──
        if sibling_pairs:
            for i in range(n_boxes):
                for j in range(i + 1, n_boxes):
                    a, b = info.boxes[i], info.boxes[j]
                    iou = _xyxy_iou(xyxys[i], xyxys[j])
                    if iou < iou_conflict:
                        continue
                    key = (min(a.class_id, b.class_id), max(a.class_id, b.class_id))
                    gname = sibling_pairs.get(key)
                    if gname:
                        flag = f"conflict:{gname}"
                        a.flags.add(flag)
                        b.flags.add(flag)

    # ── dense ──
    box_counts.sort(key=lambda x: x[1], reverse=True)
    for idx, _ in box_counts[:dense_top_n]:
        infos[idx].image_flags.add("dense")


# ── Step 1: 配额计算 ──────────────────────────────────────


def compute_class_quotas(
    names: dict[int, str],
    class_image_counts: dict[int, int],
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    cap: int = DEFAULT_CAP,
) -> dict[int, int]:
    """每类配额 quota = clamp(k, k + round(alpha * log10(img_count)), cap)。"""
    quotas: dict[int, int] = {}
    for cid in names:
        n = class_image_counts.get(cid, 0)
        if n <= 0:
            quotas[cid] = 0
        elif n < k:
            quotas[cid] = n  # 稀有类：有几张拿几张
        else:
            quotas[cid] = max(k, min(cap, k + round(alpha * math.log10(n))))
    return quotas


# ── Step 1: 贪心 set-cover 选图 ───────────────────────────


def _size_bucket(w: float, h: float) -> str:
    area = w * h
    if area < 32 * 32:
        return "tiny"
    if area < 96 * 96:
        return "small"
    if area < 288 * 288:
        return "medium"
    return "large"


def _position_hash(info: ImageReviewInfo) -> int:
    h = hashlib.md5(str(info.rel_path).encode()).hexdigest()
    return int(h[:8], 16)


def select_review_subset(
    infos: list[ImageReviewInfo],
    quotas: dict[int, int],
    seed: int = DEFAULT_SEED,
    *,
    min_only: bool = False,
    max_images: int | None = None,
    problems_only: bool = False,
) -> list[ImageReviewInfo]:
    """三阶段选图：
    Phase 1 - 最小覆盖：贪心 set-cover，优先选问题图，每类 ≥ min(quota, 1)
    Phase 2 - 问题替换：用未选中的问题图替换覆盖集合中的正常图
    Phase 3 - 额外问题图：继续选问题图，受 max_images 限制
    """
    import random

    rng = random.Random(seed)

    # min_only 模式下每类只需 1 张
    target: dict[int, int] = {}
    for cid, q in quotas.items():
        if q <= 0:
            target[cid] = 0
        elif min_only:
            target[cid] = 1
        else:
            target[cid] = q

    remaining: dict[int, int] = dict(target)
    selected_set: set[int] = set()

    problem_indices = [i for i, info in enumerate(infos) if info.has_problem]
    rng.shuffle(problem_indices)

    # ── Phase 1: 最小覆盖 ──
    while any(v > 0 for v in remaining.values()):
        best_idx = -1
        best_score = (-1, -1, 0)

        for idx in range(len(infos)):
            if idx in selected_set:
                continue
            info = infos[idx]
            new_cov = sum(1 for cid in info.class_ids if remaining.get(cid, 0) > 0)
            if new_cov <= 0:
                continue
            is_prob = 1 if info.has_problem else 0
            score = (new_cov, is_prob, -info.problem_score)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected_set.add(best_idx)
        for cid in infos[best_idx].class_ids:
            if remaining.get(cid, 0) > 0:
                remaining[cid] -= 1

    phase1_problem_used = {i for i in selected_set if infos[i].has_problem}

    # ── Phase 2: 问题替换 ──
    swap_candidates = [i for i in problem_indices if i not in selected_set]
    swap_candidates.sort(key=lambda i: -infos[i].problem_score)

    normal_in_sel = [i for i in selected_set if not infos[i].has_problem]

    for ni in normal_in_sel:
        n_classes = infos[ni].class_ids
        for ci in swap_candidates:
            c_classes = infos[ci].class_ids
            if n_classes.issubset(c_classes) and infos[ci].problem_score > 0:
                selected_set.discard(ni)
                selected_set.add(ci)
                swap_candidates.remove(ci)
                break

    # ── Phase 3: 额外问题图 ──
    if not problems_only:
        remaining_probs = [i for i in swap_candidates]
        extras = 0
        for pi in remaining_probs:
            if max_images is not None and len(selected_set) >= max_images:
                break
            selected_set.add(pi)
            extras += 1

    return [infos[i] for i in sorted(selected_set)]
# ── Step 4: 导出 ──────────────────────────────────────────


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _yolo_to_labelme_points(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> list[list[float]]:
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return [[x1, y1], [x2, y2]]


def _build_labelme_json(
    info: ImageReviewInfo,
    names: dict[int, str],
    embed_images: bool = True,
) -> dict[str, Any]:
    shapes: list[dict[str, Any]] = []
    for box in info.boxes:
        label = names.get(box.class_id, f"class_{box.class_id}")
        points = _yolo_to_labelme_points(
            box.cx, box.cy, box.w, box.h, info.image_width, info.image_height
        )
        shapes.append(
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": ";".join(sorted(box.flags)) if box.flags else "",
                "shape_type": "rectangle",
                "flags": {},
                "mask": None,
            }
        )

    img_w = info.image_width
    img_h = info.image_height
    # imageData: 默认不嵌入，X-AnyLabeling 可从 imagePath 读图
    if embed_images:
        import base64
        raw = info.src_image_path.read_bytes()
        image_data = base64.b64encode(raw).decode("utf-8")
    else:
        image_data = None

    image_flags_dict = {f: True for f in sorted(info.image_flags)}

    return {
        "version": "5.5.0",
        "flags": image_flags_dict,
        "shapes": shapes,
        "imagePath": info.rel_path.name,
        "imageData": image_data,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def export_review_subset(
    selected: list[ImageReviewInfo],
    out_dir: Path,
    names: dict[int, str],
    report: HealthCheckReport,
    quotas: dict[int, int],
    groups_src: Path | None = None,
    embed_images: bool = True,
) -> Path:
    """导出 review_subset 目录。返回 out_dir。"""
    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    
    for d in (img_out, lbl_out):
        d.mkdir(parents=True, exist_ok=True)

    # 实际选中每类计数
    actual_counts: dict[int, int] = {}

    for info in selected:
        dst_img = img_out / info.rel_path.name
        dst_lbl = lbl_out / info.rel_path.with_suffix(".txt").name
        dst_json = img_out / info.rel_path.with_suffix(".json").name
        _link_or_copy(info.src_image_path, dst_img)
        _link_or_copy(info.src_label_path, dst_lbl)

        lm_json = _build_labelme_json(info, names, embed_images=embed_images)
        dst_json.write_text(
            json.dumps(lm_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        for cid in info.class_ids:
            actual_counts[cid] = actual_counts.get(cid, 0) + 1

    # classes.txt
    (out_dir / "classes.txt").write_text(
        "\n".join(names[cid] for cid in sorted(names)) + "\n", encoding="utf-8"
    )

    # 复制 class_groups.yaml
    if groups_src and groups_src.exists():
        shutil.copy2(groups_src, out_dir / "class_groups.yaml")

    # 报告
    _write_reports(out_dir, names, report, quotas, actual_counts, selected)

    return out_dir


def _write_reports(
    out_dir: Path,
    names: dict[int, str],
    report: HealthCheckReport,
    quotas: dict[int, int],
    actual_counts: dict[int, int],
    selected: list[ImageReviewInfo],
) -> None:
    # ── qc_report.md ──
    lines: list[str] = []
    lines.append("# 数据集质检抽样报告")
    lines.append("")
    lines.append(f"- 扫描图数：{report.total_images}")
    lines.append(f"- 扫描框数：{report.total_boxes}")
    lines.append(f"- 类别数 (nc)：{report.nc}")
    lines.append(f"- 选中图数：{len(selected)}")

    n_problem = sum(1 for info in selected if info.has_problem)
    lines.append(f"- 其中问题图：{n_problem}")
    lines.append("")

    if report.id_out_of_bounds:
        lines.append("## ID 越界告警")
        lines.append("")
        for fname, cid in report.id_out_of_bounds[:50]:
            lines.append(f"- `{fname}` 中 class_id={cid} >= nc")
        lines.append("")

    if report.zero_instance_classes:
        lines.append("## 零实例类")
        lines.append("")
        for cid in report.zero_instance_classes:
            lines.append(f"- {cid}: {names.get(cid, '?')}")
        lines.append("")

    if report.mismatched_pairs:
        lines.append("## 图/标签不匹配")
        lines.append("")
        for fname in report.mismatched_pairs[:30]:
            lines.append(f"- `{fname}`")
        lines.append("")

    lines.append("## 每类覆盖")
    lines.append("")
    lines.append("| class_id | 类名 | 全量图数 | 配额 | 选中图数 | 达标 |")
    lines.append("|---|---|---:|---:|---:|:---:|")
    for cid in sorted(names):
        total = report.class_image_counts.get(cid, 0)
        quota = quotas.get(cid, 0)
        actual = actual_counts.get(cid, 0)
        ok = "OK" if actual >= quota or total < quota else "不足"
        if total == 0:
            ok = "零实例"
        lines.append(f"| {cid} | {names[cid]} | {total} | {quota} | {actual} | {ok} |")
    lines.append("")

    # flag 统计
    flag_counter: dict[str, int] = {}
    flag_examples: dict[str, list[str]] = {}
    for info in selected:
        for f in info.image_flags:
            flag_counter[f] = flag_counter.get(f, 0) + 1
            flag_examples.setdefault(f, []).append(str(info.rel_path))
        for box in info.boxes:
            for f in box.flags:
                flag_counter[f] = flag_counter.get(f, 0) + 1
                flag_examples.setdefault(f, []).append(
                    f"{info.rel_path} [cls={box.class_id}]"
                )

    if flag_counter:
        lines.append("## Flag 命中统计")
        lines.append("")
        lines.append("| Flag | 命中数 | 示例 |")
        lines.append("|---|---:|---|")
        for flag, count in sorted(flag_counter.items(), key=lambda x: -x[1]):
            examples = flag_examples.get(flag, [])[:3]
            lines.append(f"| {flag} | {count} | {', '.join(examples)} |")
        lines.append("")

    (out_dir / "qc_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ── qc_report.csv ──
    csv_path = out_dir / "qc_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "total_images", "quota", "selected", "status"])
        for cid in sorted(names):
            total = report.class_image_counts.get(cid, 0)
            quota = quotas.get(cid, 0)
            actual = actual_counts.get(cid, 0)
            status = "ok" if actual >= quota or total < quota else "shortfall"
            if total == 0:
                status = "zero"
            writer.writerow([cid, names[cid], total, quota, actual, status])


# ── 顶层便捷函数 ──────────────────────────────────────────


def run_review_pipeline(
    data_path: Path,
    out_dir: Path,
    *,
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    cap: int = DEFAULT_CAP,
    groups_path: Path | None = None,
    groups_section: str = "coco80",
    cache_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    min_only: bool = False,
    max_images: int | None = None,
    problems_only: bool = False,
    embed_images: bool = True,
    iou_dup: float = IOU_DUP,
    iou_conflict: float = IOU_CONFLICT,
    min_box_px: float = MIN_BOX_PX,
    dense_top_n: int | None = None,
) -> Path:
    """完整运行 Step 0~4，返回输出目录。"""
    rt.import_runtime_dependencies()

    # 缓存
    cache = None
    if cache_path:
        cache = ImageSizeCache(Path(cache_path))

    # Step 0: 扫描
    infos, names, nc, report = scan_dataset(data_path, cache=cache)
    if cache:
        cache.save()

    print(f"[Step 0] 扫描完成: {report.total_images} 张图, {report.total_boxes} 个框, nc={nc}")
    if report.id_out_of_bounds:
        print(f"  ⚠ ID 越界 {len(report.id_out_of_bounds)} 处")
    if report.zero_instance_classes:
        print(f"  ⚠ 零实例类 {len(report.zero_instance_classes)} 个")

    # Step 2: 几何/层级 flag
    class_groups = None
    if groups_path and groups_path.exists():
        class_groups = load_class_groups(groups_path, names, section=groups_section)
        print(f"[Step 2] 加载 {len(class_groups)} 个 class_groups ({groups_section})")
    detect_geometry_flags(
        infos, class_groups,
        iou_dup=iou_dup,
        iou_conflict=iou_conflict,
        min_box_px=min_box_px,
        dense_top_n=dense_top_n,
    )
    n_problem = sum(1 for info in infos if info.has_problem)
    print(f"[Step 2] 问题图: {n_problem}")

    # Step 1: 配额 + 选图
    quotas = compute_class_quotas(names, report.class_image_counts, k=k, alpha=alpha, cap=cap)
    selected = select_review_subset(
        infos, quotas, seed=seed,
        min_only=min_only, max_images=max_images, problems_only=problems_only,
    )
    print(f"[Step 1] 选中 {len(selected)} 张图")

    # Step 4: 导出
    out = export_review_subset(
        selected, out_dir, names, report, quotas,
        groups_src=groups_path, embed_images=embed_images,
    )
    print(f"[Step 4] 导出完成: {out}")
    return out


def integrate_model_analysis(
    infos: list[ImageReviewInfo],
    model_results: list[Any],
    *,
    outlier_classes: set[int] | None = None,
) -> int:
    """将模型分析结果集成到 ImageReviewInfo 中。

    Args:
        infos: 图片信息列表
        model_results: det_problem_export.ImageAnalysisResult 列表
        outlier_classes: 劣质类别ID集合

    Returns:
        有问题图片数量
    """
    # 建立图片路径到ImageReviewInfo的索引
    info_by_stem: dict[str, ImageReviewInfo] = {}
    for info in infos:
        info_by_stem[info.src_image_path.stem] = info

    problem_count = 0

    for result in model_results:
        stem = result.image_path.stem if result.image_path else None
        if stem is None:
            continue

        info = info_by_stem.get(stem)
        if info is None:
            continue

        # 合并模型flag
        info.model_flags = result.flags
        info.pred_json_path = None  # 可以在这里设置

        # 添加劣质类别flag
        if outlier_classes:
            for box in info.boxes:
                if box.class_id in outlier_classes:
                    box.flags.add("outlier_class")

        if info.has_problem:
            problem_count += 1

    return problem_count


def run_review_pipeline_with_model(
    data_path: Path,
    out_dir: Path,
    experiment_dir: Path | None = None,
    infer_output_dir: Path | None = None,
    report_path: Path | None = None,
    # 几何检测参数
    enable_geometry: bool = True,
    # 模型分析参数
    enable_model_analysis: bool = True,
    match_iou_threshold: float = 0.5,
    low_conf_threshold: float = 0.3,
    # 选图参数
    k: int = DEFAULT_K,
    alpha: float = DEFAULT_ALPHA,
    cap: int = DEFAULT_CAP,
    groups_path: Path | None = None,
    groups_section: str = "coco80",
    cache_path: Path | None = None,
    seed: int = DEFAULT_SEED,
    min_only: bool = False,
    max_images: int | None = None,
    problems_only: bool = False,
    embed_images: bool = True,
    iou_dup: float = IOU_DUP,
    iou_conflict: float = IOU_CONFLICT,
    min_box_px: float = MIN_BOX_PX,
    dense_top_n: int | None = None,
    # 劣质类别参数
    enable_outlier_class: bool = True,
    outlier_class_ap: float = 0.1,
    outlier_class_gt: int = 5,
    # 可视化参数
    enable_visualization: bool = False,
) -> Path:
    """完整运行 Step 0~4（含模型分析），返回输出目录。"""
    from .det_problem_export import (
        analyze_dataset_with_model,
        build_problem_report,
    )

    rt.import_runtime_dependencies()

    # 缓存
    cache = None
    if cache_path:
        cache = ImageSizeCache(Path(cache_path))

    # Step 0: 扫描
    infos, names, nc, report = scan_dataset(data_path, cache=cache)
    if cache:
        cache.save()

    print(f"[Step 0] 扫描完成: {report.total_images} 张图, {report.total_boxes} 个框, nc={nc}")
    if report.id_out_of_bounds:
        print(f"  ID 越界 {len(report.id_out_of_bounds)} 处")
    if report.zero_instance_classes:
        print(f"  零实例类 {len(report.zero_instance_classes)} 个")

    # Step 2: 几何/层级 flag
    if enable_geometry:
        class_groups = None
        if groups_path and groups_path.exists():
            class_groups = load_class_groups(groups_path, names, section=groups_section)
            print(f"[Step 2] 加载 {len(class_groups)} 个 class_groups ({groups_section})")
        detect_geometry_flags(
            infos, class_groups,
            iou_dup=iou_dup,
            iou_conflict=iou_conflict,
            min_box_px=min_box_px,
            dense_top_n=dense_top_n,
        )
        n_geometry_problem = sum(1 for info in infos if info.image_flags or any(box.flags for box in info.boxes))
        print(f"[Step 2] 几何问题图: {n_geometry_problem}")

    # Step 3: 模型分析
    model_summary = None
    analysis_info = {}
    outlier_classes: set[int] = set()

    if enable_model_analysis and infer_output_dir:
        print(f"[Step 3] 模型分析中...")
        model_results, model_summary, analysis_info = analyze_dataset_with_model(
            source_data_yaml=data_path,
            infer_output_dir=infer_output_dir,
            report_path=report_path,
            match_iou_threshold=match_iou_threshold,
            low_conf_threshold=low_conf_threshold,
            outlier_class_ap=outlier_class_ap,
            outlier_class_gt=outlier_class_gt,
        )

        # 获取劣质类别
        if enable_outlier_class:
            outlier_classes = set(analysis_info.get("outlier_classes", []))

        # 集成模型分析结果
        n_model_problem = integrate_model_analysis(
            infos, model_results,
            outlier_classes=outlier_classes if enable_outlier_class else None,
        )
        print(f"[Step 3] 模型问题图: {n_model_problem}")

        # 打印模型分析汇总
        if model_summary:
            print(f"  missing: {model_summary.missing_count}, swapped: {model_summary.swapped_count}, "
                  f"false_pos: {model_summary.false_pos_count}, loc: {model_summary.loc_count}")

    # Step 1: 配额 + 选图
    quotas = compute_class_quotas(names, report.class_image_counts, k=k, alpha=alpha, cap=cap)
    selected = select_review_subset(
        infos, quotas, seed=seed,
        min_only=min_only, max_images=max_images, problems_only=problems_only,
    )
    print(f"[Step 1] 选中 {len(selected)} 张图")

    # 统计问题图
    n_problem_selected = sum(1 for info in selected if info.has_problem)
    print(f"  问题图: {n_problem_selected}, 覆盖图: {len(selected) - n_problem_selected}")

    # Step 4: 导出
    out = export_review_subset(
        selected, out_dir, names, report, quotas,
        groups_src=groups_path, embed_images=embed_images,
    )

    # 生成模型分析报告
    if model_summary and analysis_info:
        report_text = build_problem_report(
            model_results if enable_model_analysis else [],
            model_summary,
            names,
            analysis_info,
        )
        (out / "model_analysis_report.md").write_text(report_text, encoding="utf-8")
        print(f"[Step 4] 模型分析报告: {out / 'model_analysis_report.md'}")

    print(f"[Step 4] 导出完成: {out}")
    return out
