"""检测数据集优化。

训练后分析：找出需要合并或删除的类别，用户确认后生成新数据集。
原始图片和原始 manifest 不会被修改。

流程：
1. 从 test_report.json 计算 per-class F1 / AP / sample count，识别劣质类别
2. 从推理输出目录（save_json=True）扫描预测 JSON + GT label，构建混淆矩阵
3. 找出强单向/双向混淆 > threshold 的类别对，建议合并
4. 交互确认：合并/删除/保留
5. 写出新数据集目录（原目录不动）
"""

from __future__ import annotations

import json
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import common as rt
from .det_shared import collect_source_image_infos, safe_class_name

OPTIMIZE_SUFFIX = "_AA"
DEFAULT_CONFUSION_THRESHOLD = 0.15

# 预览导出配置：仅供用户下载判断，不影响最终新数据集
# - 跳过 train，只导出 val + test，下载量小
# - 每个候选类别/类别对的图片上限，避免重复样本太多
PREVIEW_SKIP_SPLITS = frozenset({"train"})
PREVIEW_MAX_IMAGES_PER_MERGE = 20
PREVIEW_MAX_IMAGES_PER_DROP = 50

# 预览阶段 tier 化抽样：
#   T1 黄金 — GT↔pred IoU ≥ T1 阈值，类别错配，最强信号。
#   T2 次优 — IoU 在 [T2_min, T1) 之间类别错配；或 GT 有 A、pred 有 B 但没配上。
#   T3 对照 — 仅含 GT A 或 B，无另一类痕迹；给用户看类别真实样貌。
# T1 优先填满到 max_images_per_merge；不足 PREVIEW_MIN_PER_CANDIDATE 时用 T2 再 T3 兜底。
PREVIEW_MATCH_IOU_THRESHOLD = 0.5  # 兼容旧字段名，等同于 T1 阈值
PREVIEW_TIER_T1_IOU = 0.5
PREVIEW_TIER_T2_IOU = 0.3
PREVIEW_MIN_PER_CANDIDATE = 5
# 合并后去重：同 new_id 组内 IoU ≥ 此值即视为重复框，保留面积大的那个。
MERGE_DEDUP_IOU = 0.7


# ---------------------------------------------------------------------------
# 1. 分析
# ---------------------------------------------------------------------------

def _compute_f1(precision: float, recall: float) -> float:
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def _merge_per_class_ap(
    report_jsons: list[Path],
) -> dict[str, Any]:
    """合并多份 test_report.json 的 per_class_ap。

    gt / pred / tp 跨报告求和后重算 P / R / F1，AP 取平均。
    返回合并后的 per_class_ap dict。
    """
    merged: dict[str, dict[str, float]] = {}

    for report_path in report_jsons:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        per_class = payload.get("per_class_ap") or {}
        for cid_str, info in per_class.items():
            if not isinstance(info, dict):
                continue
            gt = int(info.get("gt", 0) or 0)
            pred = int(info.get("pred", 0) or 0)
            tp = int(info.get("tp", 0) or 0)
            ap = float(info.get("ap", 0.0) or 0.0)
            if cid_str not in merged:
                merged[cid_str] = {
                    "name": str(info.get("name", cid_str)),
                    "gt": 0, "pred": 0, "tp": 0,
                    "ap_sum": 0.0, "ap_count": 0,
                }
            m = merged[cid_str]
            m["gt"] += gt
            m["pred"] += pred
            m["tp"] += tp
            m["ap_sum"] += ap
            m["ap_count"] += 1

    result: dict[str, Any] = {}
    for cid_str, m in merged.items():
        gt, pred, tp = m["gt"], m["pred"], m["tp"]
        precision = tp / pred if pred > 0 else 0.0
        recall = tp / gt if gt > 0 else 0.0
        result[cid_str] = {
            "name": m["name"],
            "gt": gt,
            "pred": pred,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "ap": m["ap_sum"] / m["ap_count"] if m["ap_count"] > 0 else 0.0,
        }
    return result


def analyze_class_quality(per_class_ap: dict[str, Any]) -> list[dict[str, Any]]:
    """从 test_report.per_class_ap 计算每类质量指标和风险等级。"""
    results: list[dict[str, Any]] = []
    for class_id_str, info in per_class_ap.items():
        if not isinstance(info, dict):
            continue
        class_id = int(class_id_str)
        name = str(info.get("name", class_id_str))
        ap = float(info.get("ap", 0.0) or 0.0)
        gt = int(info.get("gt", 0) or 0)
        pred = int(info.get("pred", 0) or 0)
        tp = int(info.get("tp", 0) or 0)
        precision = float(info.get("precision", 0.0) or 0.0)
        recall = float(info.get("recall", 0.0) or 0.0)
        f1 = _compute_f1(precision, recall)
        fp = pred - tp
        fp_rate = fp / pred if pred > 0 else 0.0

        # 风险分：低质量指标加权求和，越高越差
        # GT<5 时 AP/F1 方差极大不可信，单独处理
        risk = 0.0
        if gt == 0:
            risk = 1.0
            confidence = "none"
        elif gt < 5:
            # 小样本：AP/F1 不可靠，仅用 GT 数判断，样本太少本身就是风险
            risk = 0.6 + 0.4 * (1.0 - gt / 5)  # 基础风险 0.6~1.0
            if pred == 0:
                risk = min(risk + 0.2, 1.0)
            confidence = "low"
        else:
            if gt < 10:
                risk += 0.4 * (1.0 - gt / 10)
            if ap < 0.3:
                risk += 0.3 * (1.0 - ap / 0.3)
            if f1 < 0.4:
                risk += 0.2 * (1.0 - f1 / 0.4)
            if fp_rate > 0.5:
                risk += 0.1 * min(fp_rate, 1.0)
            confidence = "normal"

        if gt == 0:
            risk_level = "极高"
        elif risk >= 0.6:
            risk_level = "高"
        elif risk >= 0.3:
            risk_level = "中"
        else:
            risk_level = "低"

        results.append({
            "class_id": class_id,
            "name": name,
            "ap": ap,
            "gt": gt,
            "pred": pred,
            "tp": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "fp_rate": fp_rate,
            "risk": risk,
            "risk_level": risk_level,
            "confidence": confidence,
        })
    results.sort(key=lambda x: (-x["risk"], x["class_id"]))
    return results


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_merged_overlapping_boxes(
    entries: list[dict[str, Any]], iou_threshold: float
) -> tuple[list[int], int]:
    """对同 new_id 组做贪心 IoU 去重，仅当组内含至少一个 came_from_merge 项时启用。

    每项 entries[i] 需含:
      - new_id:      int
      - came_from_merge: bool
      - bbox:        (cx, cy, w, h)，YOLO 归一化坐标（IoU 在归一化坐标下不变）

    保留面积大的框；返回 (保留下来的索引按原顺序排列, 被丢弃的数量)。
    """
    by_new_id: dict[int, list[int]] = defaultdict(list)
    for idx, e in enumerate(entries):
        by_new_id[e["new_id"]].append(idx)

    removed: set[int] = set()
    for idxs in by_new_id.values():
        if len(idxs) < 2:
            continue
        if not any(entries[i]["came_from_merge"] for i in idxs):
            continue
        order = sorted(
            idxs,
            key=lambda i: entries[i]["bbox"][2] * entries[i]["bbox"][3],
            reverse=True,
        )
        kept: list[int] = []
        for i in order:
            cx_i, cy_i, w_i, h_i = entries[i]["bbox"]
            box_i = [cx_i - w_i / 2, cy_i - h_i / 2, cx_i + w_i / 2, cy_i + h_i / 2]
            drop_me = False
            for k in kept:
                cx_k, cy_k, w_k, h_k = entries[k]["bbox"]
                box_k = [cx_k - w_k / 2, cy_k - h_k / 2, cx_k + w_k / 2, cy_k + h_k / 2]
                if _box_iou(box_i, box_k) >= iou_threshold:
                    drop_me = True
                    break
            if drop_me:
                removed.add(i)
            else:
                kept.append(i)

    keep_idx = [i for i in range(len(entries)) if i not in removed]
    return keep_idx, len(removed)


def _hungarian_match(cost_matrix: list[list[float]]) -> list[tuple[int, int]]:
    """Kuhn-Munkres (Hungarian) algorithm for minimum-cost bipartite matching.

    cost_matrix[i][j] is the cost of matching row i to column j.
    Use float('inf') to forbid a pairing.
    Returns list of (row, col) matched pairs (excludes inf-cost pairs).

    Implementation: O(n^3) primal method, sufficient for n < 200.
    """
    n_rows = len(cost_matrix)
    if n_rows == 0:
        return []
    n_cols = len(cost_matrix[0])

    # Find a large finite value for padding (avoids inf arithmetic issues)
    max_finite = 0.0
    for row in cost_matrix:
        for val in row:
            if val != float('inf') and abs(val) > max_finite:
                max_finite = abs(val)
    BIG = max_finite * 100 + 1000.0

    n = max(n_rows, n_cols)
    C: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            if i < n_rows and j < n_cols:
                val = cost_matrix[i][j]
                row.append(BIG if val == float('inf') else val)
            else:
                row.append(BIG)
        C.append(row)

    # u[i] = dual variable for row i, v[j] = dual variable for col j
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)    # p[j] = row matched to column j
    way = [0] * (n + 1)  # shortest-path tree

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        min_v = [float('inf')] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = C[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < min_v[j]:
                    min_v[j] = cur
                    way[j] = j0
                if min_v[j] < delta:
                    delta = min_v[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_v[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    # Build result, excluding padded entries and original inf entries
    result: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        i = p[j]
        if i >= 1 and i <= n_rows and j <= n_cols:
            if cost_matrix[i - 1][j - 1] != float('inf'):
                result.append((i - 1, j - 1))
    return result


def build_confusion_matrix(
    infer_output_dir: Path,
    source_data_yaml: Path,
    class_names: dict[int, str],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any] | None:
    """从 infer 输出目录的预测 JSON + GT 标签构建 N×N 混淆矩阵。

    预测 JSON 位于 <infer_output_dir>/_tempfile/json/，
    每个文件格式为 {image_path, width, height, predictions: [{class_id, bbox_xyxy, ...}]}。

    返回 None 表示找不到预测 JSON 文件。
    """
    infer_output_dir = infer_output_dir.expanduser().resolve()
    json_dir = infer_output_dir / rt.INFER_TEMP_DIRNAME / "json"
    if not json_dir.exists():
        # 尝试直接在 infer_output_dir 下搜索（用户可能指定了非标准路径）
        json_dir = infer_output_dir

    pred_json_files = [
        f for f in json_dir.rglob("*.json")
        if f.name not in {"run_meta.json", "metrics_summary.json"}
        and "report" not in f.name
        and "summary" not in f.name
        and "manifest" not in f.name
    ]
    if not pred_json_files:
        return None

    all_class_ids = sorted(class_names.keys())
    if not all_class_ids:
        return None
    n = len(all_class_ids)
    class_id_to_idx = {cid: idx for idx, cid in enumerate(all_class_ids)}

    # matrix[gt_idx][pred_idx] = 混淆次数
    matrix: list[list[int]] = [[0] * n for _ in range(n)]
    total_gt_by_class: dict[int, int] = defaultdict(int)
    processed = 0

    for pred_json_path in pred_json_files:
        try:
            payload = json.loads(pred_json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "predictions" not in payload:
            continue

        image_path_str = str(payload.get("image_path", ""))
        width = int(payload.get("width", 1) or 1)
        height = int(payload.get("height", 1) or 1)
        predictions = payload.get("predictions", [])

        # 从 image_path 推导 GT label 路径（images/ → labels/，图片后缀 → .txt）
        image_path = Path(image_path_str)
        label_path: Path | None = None
        parts = list(image_path.parts)
        for i, part in enumerate(parts):
            if part == "images":
                label_parts = parts.copy()
                label_parts[i] = "labels"
                candidate = Path(*label_parts).with_suffix(".txt")
                if candidate.exists():
                    label_path = candidate
                break

        if label_path is None or not label_path.exists():
            continue

        # 解析 GT 标签（YOLO 归一化格式 → 像素 xyxy）
        gt_boxes: list[tuple[int, list[float]]] = []
        try:
            for line in label_path.read_text(encoding="utf-8").strip().splitlines():
                parts_l = line.strip().split()
                if len(parts_l) != 5:
                    continue
                cid = int(float(parts_l[0]))
                if cid not in class_id_to_idx:
                    continue
                cx = float(parts_l[1]) * width
                cy = float(parts_l[2]) * height
                bw = float(parts_l[3]) * width
                bh = float(parts_l[4]) * height
                gt_boxes.append((cid, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]))
                total_gt_by_class[cid] += 1
        except Exception:
            continue

        if not gt_boxes:
            processed += 1
            continue

        # 解析预测框（带置信度）
        pred_boxes: list[tuple[int, list[float], float]] = []
        for pred in predictions:
            cid = int(pred.get("class_id", -1))
            bbox = pred.get("bbox_xyxy", [])
            score = float(pred.get("score", 1.0))
            if cid in class_id_to_idx and len(bbox) == 4:
                pred_boxes.append((cid, [float(v) for v in bbox], score))

        if not pred_boxes:
            processed += 1
            continue

        # 匈牙利匹配：score = IoU × 置信度
        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        cost: list[list[float]] = [[float('inf')] * n_pred for _ in range(n_gt)]
        for gi, (_, gt_box) in enumerate(gt_boxes):
            for pi, (_, pred_box, pred_score) in enumerate(pred_boxes):
                iou = _box_iou(gt_box, pred_box)
                if iou >= iou_threshold:
                    cost[gi][pi] = -(iou * pred_score)

        matches = _hungarian_match(cost)
        for gi, pi in matches:
            if cost[gi][pi] == float('inf'):
                continue
            gt_cid = gt_boxes[gi][0]
            pred_cid = pred_boxes[pi][0]
            gt_idx = class_id_to_idx[gt_cid]
            pred_idx = class_id_to_idx[pred_cid]
            matrix[gt_idx][pred_idx] += 1

        processed += 1

    if processed == 0:
        return None

    return {
        "class_ids": all_class_ids,
        "class_names": {str(cid): class_names[cid] for cid in all_class_ids},
        "matrix": matrix,
        "total_gt_by_class": dict(total_gt_by_class),
        "processed_images": processed,
    }


MIN_CONFUSION_COUNT_FLOOR = 3


def _adaptive_min_confusion_count(gt_count: int) -> int:
    """自适应最小混淆次数门槛：max(绝对下限, GT数的2%)。"""
    return max(MIN_CONFUSION_COUNT_FLOOR, int(gt_count * 0.02))


def find_merge_candidates(
    confusion_matrix: dict[str, Any],
    *,
    threshold: float = DEFAULT_CONFUSION_THRESHOLD,
) -> list[dict[str, Any]]:
    """找出强单向或双向混淆率 >= threshold 的类别对，建议合并。"""
    class_ids: list[int] = confusion_matrix["class_ids"]
    matrix: list[list[int]] = confusion_matrix["matrix"]
    total_gt: dict[int, int] = confusion_matrix["total_gt_by_class"]
    class_names = {int(k): v for k, v in confusion_matrix["class_names"].items()}

    candidates: list[dict[str, Any]] = []

    for i, cid_a in enumerate(class_ids):
        gt_a = total_gt.get(cid_a, 0)
        if gt_a == 0:
            continue
        for j in range(i + 1, len(class_ids)):
            cid_b = class_ids[j]
            gt_b = total_gt.get(cid_b, 0)
            if gt_b == 0:
                continue

            count_a_to_b = matrix[i][j]
            count_b_to_a = matrix[j][i]

            # 自适应绝对数量门槛：小样本高混淆率是假信号
            min_count_a = _adaptive_min_confusion_count(gt_a)
            min_count_b = _adaptive_min_confusion_count(gt_b)
            if count_a_to_b < min_count_a and count_b_to_a < min_count_b:
                continue

            rate_a_to_b = count_a_to_b / gt_a
            rate_b_to_a = count_b_to_a / gt_b

            if max(rate_a_to_b, rate_b_to_a) < threshold:
                continue

            confusion_type = "双向" if min(rate_a_to_b, rate_b_to_a) >= threshold else "单向"
            if confusion_type == "单向" and rate_b_to_a > rate_a_to_b:
                source_id, target_id = cid_b, cid_a
                source_name = class_names.get(cid_b, str(cid_b))
                target_name = class_names.get(cid_a, str(cid_a))
                source_rate, reverse_rate = rate_b_to_a, rate_a_to_b
                source_count, reverse_count = matrix[j][i], matrix[i][j]
                source_gt, target_gt = gt_b, gt_a
            else:
                source_id, target_id = cid_a, cid_b
                source_name = class_names.get(cid_a, str(cid_a))
                target_name = class_names.get(cid_b, str(cid_b))
                source_rate, reverse_rate = rate_a_to_b, rate_b_to_a
                source_count, reverse_count = matrix[i][j], matrix[j][i]
                source_gt, target_gt = gt_a, gt_b

            candidates.append({
                "class_id_a": source_id,
                "class_id_b": target_id,
                "name_a": source_name,
                "name_b": target_name,
                "rate_a_to_b": source_rate,
                "rate_b_to_a": reverse_rate,
                "count_a_to_b": source_count,
                "count_b_to_a": reverse_count,
                "gt_a": source_gt,
                "gt_b": target_gt,
                "confusion_type": confusion_type,
                "primary_direction": f"{source_name}→{target_name}",
                "score": max(source_rate, reverse_rate),
            })

    candidates.sort(key=lambda x: (-x["score"], -(x["rate_a_to_b"] + x["rate_b_to_a"])))
    return candidates


# ---------------------------------------------------------------------------
# 2. 展示
# ---------------------------------------------------------------------------

def _trunc(name: str, width: int) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def print_quality_table(class_quality: list[dict[str, Any]]) -> None:
    risky = [q for q in class_quality if q["risk_level"] in {"极高", "高", "中"}]
    if not risky:
        print("  所有类别质量正常，无明显劣质类别。")
        return
    hdr = f"  {'类别':<22} {'GT':>6} {'AP':>6} {'P':>6} {'R':>6} {'F1':>6} {'风险':>4} {'可信':>4}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for q in risky:
        conf = q.get("confidence", "normal")
        conf_mark = "低" if conf in {"low", "none"} else ""
        print(
            f"  {_trunc(q['name'], 22):<22} {q['gt']:>6}"
            f" {q['ap']:>6.3f} {q['precision']:>6.3f}"
            f" {q['recall']:>6.3f} {q['f1']:>6.3f} {q['risk_level']:>4} {conf_mark:>4}"
        )


def print_confusion_table(merge_candidates: list[dict[str, Any]]) -> None:
    if not merge_candidates:
        print("  未发现强混淆类别对。")
        return
    hdr = f"  {'类别A':<18} {'类别B':<18} {'A→B':>7} {'B→A':>7} {'A→B次':>6} {'B→A次':>6} {'类型':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in merge_candidates:
        print(
            f"  {_trunc(c['name_a'], 18):<18}"
            f" {_trunc(c['name_b'], 18):<18}"
            f" {c['rate_a_to_b']:>7.1%} {c['rate_b_to_a']:>7.1%}"
            f" {c.get('count_a_to_b', 0):>6} {c.get('count_b_to_a', 0):>6}"
            f" {c.get('confusion_type', '双向'):>6}"
        )


# ---------------------------------------------------------------------------
# 3. 交互确认
# ---------------------------------------------------------------------------

def _read_line(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        raise SystemExit("\n输入结束，已取消执行。")


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = _read_line(f"  {prompt} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1"}:
            return True
        if raw in {"n", "no", "0"}:
            return False
        print("  请输入 y 或 n。")


def _prompt_merge_name_choice(
    *,
    name_a: str,
    name_b: str,
    gt_a: int,
    gt_b: int,
) -> str:
    total = max(gt_a + gt_b, 1)
    default_index = 1 if gt_a >= gt_b else 2
    options = {
        "1": name_a,
        "2": name_b,
    }
    print("  合并后类别名二选一：")
    print(f"    1. {name_a}  GT={gt_a}  占比={gt_a / total:.1%}")
    print(f"    2. {name_b}  GT={gt_b}  占比={gt_b / total:.1%}")
    while True:
        raw = _read_line(f"  请选择合并后类别名 [{default_index}]: ").strip()
        if not raw:
            return options[str(default_index)]
        if raw in options:
            return options[raw]
        if raw == name_a:
            return name_a
        if raw == name_b:
            return name_b
        print("  请输入 1 或 2。")


def interactive_collect_decisions(
    class_quality: list[dict[str, Any]],
    merge_candidates: list[dict[str, Any]],
    class_names: dict[int, str],
) -> dict[str, Any] | None:
    """交互确认合并/删除操作，返回 decisions dict 或 None（用户取消）。"""
    print("\n" + "=" * 62)
    print("  数据集优化 - 合并与删除确认")
    print("=" * 62)

    merge_decisions: list[dict[str, Any]] = []
    merged_ids: set[int] = set()

    if merge_candidates:
        print("\n[合并候选] 以下类别对存在强混淆，建议合并或复核标注口径：\n")
        print_confusion_table(merge_candidates)
        for cand in merge_candidates:
            cid_a, cid_b = cand["class_id_a"], cand["class_id_b"]
            if cid_a in merged_ids or cid_b in merged_ids:
                continue
            name_a, name_b = cand["name_a"], cand["name_b"]
            confusion_type = cand.get("confusion_type", "双向")
            print(
                f"\n  合并 [{name_a}] + [{name_b}]"
                f"  ({confusion_type}，A→B: {cand['rate_a_to_b']:.1%}, B→A: {cand['rate_b_to_a']:.1%})"
            )
            if _prompt_yes_no("是否合并", False):
                merged_name = _prompt_merge_name_choice(
                    name_a=name_a,
                    name_b=name_b,
                    gt_a=int(cand.get("gt_a", 0) or 0),
                    gt_b=int(cand.get("gt_b", 0) or 0),
                )
                merge_decisions.append({
                    "class_ids": [cid_a, cid_b],
                    "merged_name": merged_name,
                    "reason": (
                        f"{confusion_type}混淆：{name_a}→{name_b} {cand['rate_a_to_b']:.1%}，"
                        f"{name_b}→{name_a} {cand['rate_b_to_a']:.1%}"
                    ),
                })
                merged_ids.update([cid_a, cid_b])
    else:
        print("\n[合并候选] 未发现强混淆类别对。")

    drop_decisions: list[dict[str, Any]] = []
    dropped_ids: set[int] = set()

    risky = [
        q for q in class_quality
        if q["risk_level"] in {"极高", "高", "中"} and q["class_id"] not in merged_ids
    ]
    if risky:
        print("\n[劣质类别] 以下类别存在质量问题，建议删除或复核：\n")
        print_quality_table(risky)
        for q in risky:
            cid = q["class_id"]
            name = q["name"]
            print(
                f"\n  [{name}]  AP={q['ap']:.3f}  F1={q['f1']:.3f}"
                f"  GT={q['gt']}  风险={q['risk_level']}"
            )
            if _prompt_yes_no("删除该类别", False):
                drop_decisions.append({
                    "class_id": cid,
                    "name": name,
                    "reason": (
                        f"劣质类别：AP={q['ap']:.3f} F1={q['f1']:.3f}"
                        f" GT数={q['gt']} 风险={q['risk_level']}"
                    ),
                    "metrics": {
                        k: q[k]
                        for k in ("ap", "gt", "pred", "tp", "precision", "recall", "f1", "risk")
                    },
                })
                dropped_ids.add(cid)
    else:
        print("\n[劣质类别] 无需处理的劣质类别。")

    if not merge_decisions and not drop_decisions:
        print("\n未选择任何合并或删除操作，退出优化流程。")
        return None

    print("\n" + "=" * 62)
    print("  操作摘要")
    print("=" * 62)
    for d in merge_decisions:
        names_str = " + ".join(class_names.get(cid, str(cid)) for cid in d["class_ids"])
        print(f"  合并: {names_str}  →  {d['merged_name']}")
    for d in drop_decisions:
        print(f"  删除: {d['name']}")

    if not _prompt_yes_no("\n确认按以上操作生成新数据集", True):
        print("  已取消。")
        return None

    return {"merge_decisions": merge_decisions, "drop_decisions": drop_decisions}


# ---------------------------------------------------------------------------
# 4. 执行
# ---------------------------------------------------------------------------

def _build_new_class_mapping(
    original_class_names: dict[int, str],
    merge_decisions: list[dict[str, Any]],
    drop_decisions: list[dict[str, Any]],
) -> tuple[dict[int, int | None], dict[int, str]]:
    """构建 old_class_id → new_class_id（None 表示删除）的映射。

    返回:
        id_map:    {old_id: new_id | None}
        new_names: {new_id: name}
    """
    dropped_ids = {d["class_id"] for d in drop_decisions}

    # 每个 merge_group 是 (class_ids_list, merged_name)
    merge_groups: list[tuple[list[int], str]] = []
    merged_ids: set[int] = set()
    for md in merge_decisions:
        merge_groups.append((md["class_ids"], md["merged_name"]))
        merged_ids.update(md["class_ids"])

    # 未受影响的类保留自身名称，作为单元素组
    for cid in sorted(original_class_names.keys()):
        if cid not in merged_ids and cid not in dropped_ids:
            merge_groups.append(([cid], original_class_names[cid]))

    # 按每组最小原始 ID 排序，保持 ID 顺序稳定
    merge_groups.sort(key=lambda g: min(g[0]))

    id_map: dict[int, int | None] = {cid: None for cid in dropped_ids}
    new_names: dict[int, str] = {}
    new_id = 0
    for group_ids, group_name in merge_groups:
        for cid in group_ids:
            id_map[cid] = new_id
        new_names[new_id] = group_name
        new_id += 1

    return id_map, new_names


def apply_optimize_decisions(
    decisions: dict[str, Any],
    source_data_yaml: Path,
    *,
    output_suffix: str = OPTIMIZE_SUFFIX,
) -> tuple[Path, dict[int, str], int, int]:
    """执行合并/删除并写出新数据集，返回 (新数据集根目录, 新类别名称, 复制图数, 去重框数)。

    不修改原始数据集任何文件。合并后会在同 new_id 组内做 IoU 去重，
    阈值 MERGE_DEDUP_IOU；保留面积更大的框。
    """
    source_cfg = rt.load_data_config(source_data_yaml)
    source_root = Path(source_cfg["_root_dir"])
    original_class_names = rt.normalize_names(source_cfg.get("names"))

    id_map, new_names = _build_new_class_mapping(
        original_class_names,
        decisions["merge_decisions"],
        decisions["drop_decisions"],
    )

    # 构建反向映射：old_id → 操作标签（用于 review 分组）
    dropped_ids = {d["class_id"] for d in decisions["drop_decisions"]}
    merge_groups: dict[int, dict[str, Any]] = {}  # old_id → merge_decision
    for md in decisions["merge_decisions"]:
        for cid in md["class_ids"]:
            merge_groups[cid] = md

    source_infos_by_split, export_split_paths = collect_source_image_infos(source_cfg, source_root)

    normalized_output_stem = _build_optimize_output_stem(
        source_root_name=source_root.name,
        output_suffix=output_suffix,
    )

    # 临时目录，全部写完再重命名
    temp_root = rt.deduplicate_path(source_root.parent / f"{normalized_output_stem}__tmp")
    temp_root.mkdir(parents=True, exist_ok=True)

    total_images = sum(len(infos) for infos in source_infos_by_split.values())
    copied = 0
    dedup_total = 0  # 合并后被 IoU 去重的框数（累计）
    # review 数据：按操作标签分组，每组记录 (split, rel_path, src_image_path, original_lines, new_lines)
    review_by_op: dict[str, list[dict[str, Any]]] = {}
    print(f"\n[det/optimize] 写出新数据集 ...")
    print(f"  source={source_root}")
    print(f"  temp={temp_root}")
    print(f"  total_images={total_images}")

    for split_name, infos in source_infos_by_split.items():
        if not infos:
            continue
        dst_image_dir = temp_root / "images" / split_name
        dst_label_dir = temp_root / "labels" / split_name
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for info in infos:
            dst_image = dst_image_dir / info.rel_path
            dst_label = dst_label_dir / info.rel_path.with_suffix(".txt")
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            dst_label.parent.mkdir(parents=True, exist_ok=True)

            # 重写标签：跳过被删类，合并类改 class_id；后续做 IoU 去重
            entries: list[dict[str, Any]] = []
            affected_ops: set[str] = set()  # 该图片涉及的操作标签
            for line in info.label_lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    old_id = int(float(parts[0]))
                except ValueError:
                    continue
                new_id = id_map.get(old_id)
                if new_id is None:
                    # 删除操作
                    if old_id in dropped_ids:
                        drop_name = original_class_names.get(old_id, str(old_id))
                        affected_ops.add(f"drop_{drop_name}")
                    continue
                came_from_merge = False
                if new_id != old_id:
                    md = merge_groups.get(old_id)
                    if md is not None:
                        came_from_merge = True
                        names_str = "+".join(
                            original_class_names.get(c, str(c)) for c in md["class_ids"]
                        )
                        affected_ops.add(f"merge_{names_str}→{md['merged_name']}")
                try:
                    cx = float(parts[1]); cy = float(parts[2])
                    bw = float(parts[3]); bh = float(parts[4])
                except ValueError:
                    continue
                entries.append({
                    "new_id": new_id,
                    "came_from_merge": came_from_merge,
                    "bbox": (cx, cy, bw, bh),
                    "coords_str": " ".join(parts[1:]),
                })

            keep_idx, removed_here = _dedup_merged_overlapping_boxes(entries, MERGE_DEDUP_IOU)
            if removed_here > 0:
                dedup_total += removed_here
                affected_ops.add("dedup_overlap")
            remapped = [
                f"{entries[i]['new_id']} {entries[i]['coords_str']}"
                for i in keep_idx
            ]

            # 只有被修改的图片才复制并记录 review
            if affected_ops:
                shutil.copy2(info.src_image_path, dst_image)
                dst_label.write_text(
                    "\n".join(remapped) + ("\n" if remapped else ""),
                    encoding="utf-8",
                )
                for op_label in affected_ops:
                    review_by_op.setdefault(op_label, []).append({
                        "split": split_name,
                        "rel_path": info.rel_path,
                        "src_image_path": info.src_image_path,
                        "original_lines": info.label_lines,
                        "new_lines": tuple(remapped),
                    })
            else:
                # 未受影响的图片也复制（保持数据集完整）
                shutil.copy2(info.src_image_path, dst_image)
                dst_label.write_text(
                    "\n".join(remapped) + ("\n" if remapped else ""),
                    encoding="utf-8",
                )
            copied += 1

    # 重命名为带图片数的最终目录
    final_root = rt.deduplicate_path(source_root.parent / f"{normalized_output_stem}_{copied}")
    temp_root.rename(final_root)

    print(f"  final_root={final_root}")
    print(f"  copied_images={copied}")
    print(f"  new_class_count={len(new_names)}")
    if dedup_total > 0:
        print(f"  dedup_overlap_boxes={dedup_total} (IoU≥{MERGE_DEDUP_IOU}，保留面积大的)")

    # 写 data.yaml 和 classes.txt
    export_cfg = {
        "path": str(final_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "task": source_cfg.get("task", "detect"),
        "nc": len(new_names),
        "names": new_names,
    }
    rt.dump_yaml(final_root / "data.yaml", export_cfg)
    (final_root / "classes.txt").write_text(
        "\n".join(new_names[i] for i in range(len(new_names))) + "\n",
        encoding="utf-8",
    )

    # 写出 review 文件夹
    if review_by_op:
        _write_optimize_review(
            output_root=final_root,
            review_by_op=review_by_op,
            original_class_names=original_class_names,
            new_class_names=new_names,
        )

    return final_root, new_names, copied, dedup_total


def _write_optimize_review(
    *,
    output_root: Path,
    review_by_op: dict[str, list[dict[str, Any]]],
    original_class_names: dict[int, str],
    new_class_names: dict[int, str],
) -> None:
    """在 output_root/optimize_review/ 下按操作分组写出被修改的图片、新标签和 diff 日志。"""
    review_root = output_root / "optimize_review"
    total_reviewed = 0

    for op_label, items in sorted(review_by_op.items()):
        op_dir = review_root / op_label
        img_dir = op_dir / "images"
        lbl_dir = op_dir / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        diff_lines: list[str] = []
        for item in items:
            rel_path: Path = item["rel_path"]
            split: str = item["split"]
            src_image_path: Path = item["src_image_path"]
            original_lines: tuple[str, ...] = item["original_lines"]
            new_lines: tuple[str, ...] = item["new_lines"]

            # 复制图片和新标签
            dst_img = img_dir / split / rel_path
            dst_lbl = lbl_dir / split / rel_path.with_suffix(".txt")
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image_path, dst_img)
            dst_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")

            # 构建 diff
            diff_lines.append(f"=== {split}/{rel_path.as_posix()} ===")
            orig_set = set(original_lines)
            new_set = set(new_lines)
            for line in original_lines:
                if line not in new_set:
                    parts = line.split()
                    old_id = int(float(parts[0])) if parts else -1
                    old_name = original_class_names.get(old_id, str(old_id))
                    diff_lines.append(f"  - {line}  ({old_name} 已删除或合并)")
            for line in new_lines:
                if line not in orig_set:
                    parts = line.split()
                    new_id = int(float(parts[0])) if parts else -1
                    new_name = new_class_names.get(new_id, str(new_id))
                    diff_lines.append(f"  + {line}  ({new_name})")
            diff_lines.append("")
            total_reviewed += 1

        # 写 diff 日志
        diff_path = op_dir / "diff.txt"
        header = [
            f"操作: {op_label}",
            f"图片数: {len(items)}",
            f"类别映射: {original_class_names} → {new_class_names}",
            "",
        ]
        diff_path.write_text("\n".join(header + diff_lines), encoding="utf-8")

    print(f"\n[det/optimize] Review 文件夹已生成:")
    print(f"  {review_root}")
    print(f"  共 {total_reviewed} 张被修改图片，{len(review_by_op)} 个操作分组")


def _find_vis_image(
    infer_output_dir: Path, original_image_path: Path
) -> Path | None:
    """在推理输出目录中查找对应的可视化图片（带推理框和 GT 框）。"""
    stem = original_image_path.stem
    ext = original_image_path.suffix or ".jpg"
    # 推理输出结构: <dir>/images/**/<class>_<stem>.<ext>
    candidates = sorted(infer_output_dir.rglob(f"*_{stem}.*"))
    for c in candidates:
        if c.suffix.lower() == ext.lower() and c.is_file():
            return c
    for c in candidates:
        if c.is_file():
            return c
    return None


def _build_pred_json_lookup(infer_output_dir: Path) -> dict[str, dict[str, Any]]:
    """构建 {image_stem: pred_payload} 查询表。"""
    json_dir = infer_output_dir / rt.INFER_TEMP_DIRNAME / "json"
    if not json_dir.exists():
        json_dir = infer_output_dir
    lookup: dict[str, dict[str, Any]] = {}
    for f in json_dir.rglob("*.json"):
        name = f.name
        if name in {"run_meta.json", "metrics_summary.json"}:
            continue
        if "report" in name or "summary" in name or "manifest" in name:
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or "predictions" not in payload:
            continue
        image_path = Path(str(payload.get("image_path", "")))
        if image_path.stem:
            lookup[image_path.stem] = payload
    return lookup


def _inference_labels_yolo(pred_payload: dict[str, Any]) -> list[str]:
    """把预测 JSON 转成 YOLO 归一化标签行（用于 X-AnyLabeling 编辑）。"""
    width = float(pred_payload.get("width", 0) or 0)
    height = float(pred_payload.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        return []
    lines: list[str] = []
    for pred in pred_payload.get("predictions", []) or []:
        cid = int(pred.get("class_id", -1))
        bbox = pred.get("bbox_xyxy", [])
        if cid < 0 or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in bbox)
        cx = (x1 + x2) / 2.0 / width
        cy = (y1 + y2) / 2.0 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _prompt_preview_export_kinds(
    *, has_merge: bool, has_drop: bool
) -> frozenset[str]:
    """询问用户预览要导出哪类候选；返回空集表示跳过预览。"""
    if has_merge and has_drop:
        print("\n  预览导出选项：")
        print("    1) 仅 merge（默认）")
        print("    2) 仅 drop")
        print("    3) 两者都导出")
        print("    0) 跳过预览，直接进入决策")
        raw = _read_line("  请选择 [0/1/2/3，回车默认 1]: ").strip()
        if raw == "0":
            return frozenset()
        if raw == "2":
            return frozenset({"drop"})
        if raw == "3":
            return frozenset({"merge", "drop"})
        return frozenset({"merge"})
    only = "merge" if has_merge else "drop"
    print(f"\n  仅检出 {only} 候选。")
    if not _prompt_yes_no(f"是否导出 {only} 预览供查看？", default=True):
        return frozenset()
    return frozenset({only})


def _zip_preview_dir(preview_dir: Path) -> Path:
    """把预览目录打包成同名 .zip，方便服务器侧下载。

    使用 ZIP_STORED（不压缩）——图片本身已经是压缩格式，再压缩省不了多少空间
    但会显著拖慢。zip 与 preview_dir 同级，文件名 <preview_dir.name>.zip。
    包内顶层保留 preview_dir.name 这一层，解压后是一个整齐的文件夹。
    """
    # 用字符串拼接而不是 with_suffix，避免数据集名带点时（如 dataset.v2）被替换掉
    zip_path = preview_dir.parent / f"{preview_dir.name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    base = preview_dir.parent
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for path in sorted(preview_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(base))
    return zip_path


def _resolve_preview_dir(source_root: Path) -> Path:
    """预览目录放在源数据集同级，命名 <dataset>_preview_<YYYYMMDD-HHMMSS>。

    例如源数据集是 datasets/wuwanPic_dataset/dataset_det，
    预览目录就放到 datasets/wuwanPic_dataset/dataset_det_preview_20260521-123456。
    打包后的 .zip 也跟它并列在同一目录下。
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    name = f"{source_root.name}_preview_{ts}"
    return rt.deduplicate_path(source_root.parent / name)


def _export_preview_for_review(
    *,
    merge_candidates: list[dict[str, Any]],
    class_quality: list[dict[str, Any]],
    source_data_yaml: Path,
    class_names: dict[int, str],
    infer_output_dir: Path | None = None,
    output_suffix: str = OPTIMIZE_SUFFIX,
    max_images_per_merge: int = PREVIEW_MAX_IMAGES_PER_MERGE,
    max_images_per_drop: int = PREVIEW_MAX_IMAGES_PER_DROP,
    skip_splits: frozenset[str] = PREVIEW_SKIP_SPLITS,
    export_kinds: frozenset[str] = frozenset({"merge", "drop"}),
) -> tuple[Path, int]:
    """在确认前导出受影响图片的预览，供用户查看后再决定。

    目录布局（扁平化，便于 X-AnyLabeling 一次性查看所有候选）：
        preview_dir/
        ├── visualizations/<候选>/<原名>__<split>.<ext>   (人眼看，分子目录)
        ├── inference_dataset/                            (X-AnyLabeling 打开此处)
        │   ├── data.yaml / classes.txt                   (全局类名)
        │   ├── images/<候选>__<原名>__<split>.<ext>      (扁平)
        │   └── labels/<同名>.txt                         (YOLO 归一化预测)
        ├── README.txt                                    (所有候选的总览)
        └── manifest.json                                 (机器读)

    采样规则：
      - 仅导出 val/test 等非训练 split（由 skip_splits 控制）。
      - 合并候选：GT↔pred 一对一最佳-IoU 匹配（IoU≥{PREVIEW_MATCH_IOU_THRESHOLD}），类别错配才算混淆现场。
      - 劣质类别：图片含该类即可。
      - 每个 merge 候选最多 max_images_per_merge 张，每个 drop 候选最多 max_images_per_drop 张。
      - export_kinds 控制只导 merge / 只导 drop / 两者都导。

    返回 (preview_dir, total_preview_images)。
    """
    source_cfg = rt.load_data_config(source_data_yaml)
    source_root = Path(source_cfg["_root_dir"])
    source_infos_by_split, _ = collect_source_image_infos(source_cfg, source_root)

    pred_lookup: dict[str, dict[str, Any]] = {}
    if infer_output_dir is not None and infer_output_dir.exists():
        pred_lookup = _build_pred_json_lookup(infer_output_dir)

    preview_dir = _resolve_preview_dir(source_root)
    preview_dir.mkdir(parents=True, exist_ok=True)

    sorted_cids = sorted(class_names.keys())
    full_names_map = {i: class_names[cid] for i, cid in enumerate(sorted_cids)}
    # 推理 label 里写的是原始 class_id，X-AnyLabeling 直接用原始 class_id 索引这里的 names
    full_names_by_original_id = {cid: class_names[cid] for cid in sorted_cids}
    max_cid = max(sorted_cids) if sorted_cids else -1

    # 统一的输出目录：所有候选共用一个 X-AnyLabeling 数据集 + 一个可视化图库
    vis_root = preview_dir / "visualizations"
    ds_root = preview_dir / "inference_dataset"
    ds_images = ds_root / "images"
    ds_labels = ds_root / "labels"
    vis_root.mkdir(parents=True, exist_ok=True)
    ds_images.mkdir(parents=True, exist_ok=True)
    ds_labels.mkdir(parents=True, exist_ok=True)

    # 全局 data.yaml / classes.txt：用原始 class_id 作为 key，与推理 label 对齐
    names_for_yaml = {
        cid: full_names_by_original_id.get(cid, str(cid))
        for cid in range(max_cid + 1)
    }
    rt.dump_yaml(
        ds_root / "data.yaml",
        {
            "path": str(ds_root),
            "train": "images",
            "val": "images",
            "test": "images",
            "task": "detect",
            "nc": max_cid + 1,
            "names": names_for_yaml,
        },
    )
    (ds_root / "classes.txt").write_text(
        "\n".join(names_for_yaml[i] for i in range(max_cid + 1)) + "\n",
        encoding="utf-8",
    )

    manifest_images: list[dict[str, Any]] = []
    vis_missing_count = 0  # 找不到 vis 图的张数（不回退原图，避免误导）

    def _safe_token(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _emit_flat(
        info: Any, split_name: str, cand_key: str, tier: str | None = None
    ) -> None:
        nonlocal vis_missing_count
        # 用 rel_path（含子目录）展平成唯一标识，避免不同子目录同名图相互覆盖
        orig_id = info.rel_path.with_suffix("").as_posix().replace("/", "__")
        orig_ext = info.rel_path.suffix
        tier_token = f"{tier}__" if tier else ""

        # 1) visualizations/<候选>/[<tier>__]<原名>__<split>.<ext>，找不到 vis 时不回退原图
        vis_src: Path | None = None
        if infer_output_dir is not None:
            vis_src = _find_vis_image(infer_output_dir, info.src_image_path)
        vis_rel: str | None = None
        if vis_src is not None:
            vis_dir = vis_root / cand_key
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_dst = vis_dir / f"{tier_token}{orig_id}__{split_name}{vis_src.suffix}"
            shutil.copy2(vis_src, vis_dst)
            vis_rel = vis_dst.relative_to(preview_dir).as_posix()
        else:
            vis_missing_count += 1

        # 2) inference_dataset/images & labels，扁平化、文件名带候选+tier 前缀
        stem = f"{cand_key}__{tier_token}{orig_id}__{split_name}"
        img_dst = ds_images / f"{stem}{orig_ext}"
        lbl_dst = ds_labels / f"{stem}.txt"
        shutil.copy2(info.src_image_path, img_dst)
        pred_payload = pred_lookup.get(info.src_image_path.stem)
        lines = _inference_labels_yolo(pred_payload) if pred_payload is not None else []
        lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        manifest_images.append({
            "candidate": cand_key,
            "tier": tier,
            "split": split_name,
            "image": img_dst.name,
            "label": lbl_dst.name,
            "visualization": vis_rel,
            "original_path": str(info.src_image_path),
        })

    def _label_cids(info: Any) -> set[int]:
        cids: set[int] = set()
        for line in info.label_lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cids.add(int(float(parts[0])))
                except ValueError:
                    pass
        return cids

    match_cache: dict[str, tuple[set[int], set[int], list[tuple[int, int, float]]]] = {}

    def _per_image_match_details(
        info: Any,
    ) -> tuple[set[int], set[int], list[tuple[int, int, float]]]:
        """返回 (gt_cids, pred_cids, matches)；按图缓存，跨候选共享。

        matches 是 [(gt_cid, pred_cid, iou), ...]，IoU ≥ PREVIEW_TIER_T2_IOU；
        贪心一对一匹配，pred 按 score 降序优先。
        """
        cache_key = str(info.src_image_path)
        cached = match_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _compute_match_details(info)
        match_cache[cache_key] = result
        return result

    def _compute_match_details(
        info: Any,
    ) -> tuple[set[int], set[int], list[tuple[int, int, float]]]:
        gt_cids: set[int] = set()
        pred_cids: set[int] = set()
        matches: list[tuple[int, int, float]] = []
        payload = pred_lookup.get(info.src_image_path.stem)
        # 即便没有 pred，也仍然解析 GT 类，便于 T3 兜底
        for line in info.label_lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                gt_cids.add(int(float(parts[0])))
            except ValueError:
                continue
        if not payload:
            return gt_cids, pred_cids, matches
        width = float(payload.get("width", 0) or 0)
        height = float(payload.get("height", 0) or 0)
        if width <= 0 or height <= 0:
            return gt_cids, pred_cids, matches

        gt_boxes: list[tuple[int, list[float]]] = []
        for line in info.label_lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(float(parts[0]))
                cx = float(parts[1]) * width
                cy = float(parts[2]) * height
                bw = float(parts[3]) * width
                bh = float(parts[4]) * height
            except ValueError:
                continue
            gt_boxes.append(
                (cid, [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            )

        preds: list[tuple[int, list[float], float]] = []
        for pred in payload.get("predictions", []) or []:
            try:
                cid = int(pred.get("class_id", -1))
            except (TypeError, ValueError):
                continue
            bbox = pred.get("bbox_xyxy", [])
            if cid < 0 or len(bbox) != 4:
                continue
            score = float(pred.get("score", 1.0))
            pred_cids.add(cid)
            preds.append((cid, [float(v) for v in bbox], score))

        if not gt_boxes or not preds:
            return gt_cids, pred_cids, matches

        matched_gt: set[int] = set()
        for pred_cid, pred_box, _score in sorted(preds, key=lambda p: p[2], reverse=True):
            best_iou = 0.0
            best_gi = -1
            for gi, (_, gt_box) in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _box_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_gi >= 0 and best_iou >= PREVIEW_TIER_T2_IOU:
                matched_gt.add(best_gi)
                matches.append((gt_boxes[best_gi][0], pred_cid, best_iou))
        return gt_cids, pred_cids, matches

    def _classify_image_tier(info: Any, cid_a: int, cid_b: int) -> str | None:
        """对单张图返回该候选下的 tier，或 None 表示不收。"""
        gt_cids, pred_cids, matches = _per_image_match_details(info)
        target_pairs = {(cid_a, cid_b), (cid_b, cid_a)}
        relevant_ious = [iou for (g, p, iou) in matches if (g, p) in target_pairs]
        if any(iou >= PREVIEW_TIER_T1_IOU for iou in relevant_ious):
            return "T1"
        if relevant_ious:  # 已在 [T2_IOU, T1_IOU) 范围
            return "T2"
        has_a, has_b = cid_a in gt_cids, cid_b in gt_cids
        has_pred_a, has_pred_b = cid_a in pred_cids, cid_b in pred_cids
        if (has_a and has_pred_b) or (has_b and has_pred_a):
            return "T2"
        if has_a or has_b:
            return "T3"
        return None

    # 全数据集类别人口（含 train），供用户对照"导出几张 vs 实际有多少"
    class_image_count: dict[int, int] = defaultdict(int)
    class_box_count: dict[int, int] = defaultdict(int)
    for _infos in source_infos_by_split.values():
        for _info in _infos:
            _seen: set[int] = set()
            for _line in _info.label_lines:
                _parts = _line.strip().split()
                if len(_parts) != 5:
                    continue
                try:
                    _cid = int(float(_parts[0]))
                except ValueError:
                    continue
                class_box_count[_cid] += 1
                if _cid not in _seen:
                    class_image_count[_cid] += 1
                    _seen.add(_cid)

    total = 0
    candidate_stats: list[dict[str, Any]] = []

    # 合并候选：tier 分层抽样——T1 黄金 → T2 次优 → T3 对照，凑量但保信号
    if "merge" in export_kinds and merge_candidates:
        print(f"  [merge] 导出 {len(merge_candidates)} 个候选 ...")
    if "merge" in export_kinds:
        for cand_idx, cand in enumerate(merge_candidates, 1):
            cid_a, cid_b = cand["class_id_a"], cand["class_id_b"]
            name_a = cand.get("name_a", class_names.get(cid_a, str(cid_a)))
            name_b = cand.get("name_b", class_names.get(cid_b, str(cid_b)))
            confusion_type = cand.get("confusion_type", "")
            cand_key = f"merge_{_safe_token(name_a)}+{_safe_token(name_b)}"

            # 先按 tier 归桶（在非 skip split 上扫一遍）
            buckets: dict[str, list[tuple[Any, str]]] = {"T1": [], "T2": [], "T3": []}
            for split_name, infos in source_infos_by_split.items():
                if split_name in skip_splits:
                    continue
                for info in infos:
                    tier = _classify_image_tier(info, cid_a, cid_b)
                    if tier is not None:
                        buckets[tier].append((info, split_name))

            # 优先级凑量：T1 先填满至 max；不足 MIN 时 T2 补，再不够 T3 兜
            chosen: list[tuple[Any, str, str]] = []
            for info, split_name in buckets["T1"]:
                if len(chosen) >= max_images_per_merge:
                    break
                chosen.append((info, split_name, "T1"))
            if len(chosen) < PREVIEW_MIN_PER_CANDIDATE:
                for info, split_name in buckets["T2"]:
                    if len(chosen) >= PREVIEW_MIN_PER_CANDIDATE:
                        break
                    chosen.append((info, split_name, "T2"))
            if len(chosen) < PREVIEW_MIN_PER_CANDIDATE:
                for info, split_name in buckets["T3"]:
                    if len(chosen) >= PREVIEW_MIN_PER_CANDIDATE:
                        break
                    chosen.append((info, split_name, "T3"))

            for info, split_name, tier in chosen:
                _emit_flat(info, split_name, cand_key, tier=tier)
            tier_counts = {"T1": 0, "T2": 0, "T3": 0}
            for _, _, t in chosen:
                tier_counts[t] += 1
            count = len(chosen)

            stats_a = {"images": class_image_count.get(cid_a, 0), "boxes": class_box_count.get(cid_a, 0)}
            stats_b = {"images": class_image_count.get(cid_b, 0), "boxes": class_box_count.get(cid_b, 0)}

            candidate_stats.append({
                "kind": "merge",
                "key": cand_key,
                "images": count,
                "tier_counts": tier_counts,
                "tier_pool_sizes": {k: len(v) for k, v in buckets.items()},
                "class_stats_a": stats_a,
                "class_stats_b": stats_b,
                "name_a": name_a, "name_b": name_b,
                "cid_a": cid_a, "cid_b": cid_b,
                "confusion_type": confusion_type,
                "rate_a_to_b": cand.get("rate_a_to_b", 0),
                "rate_b_to_a": cand.get("rate_b_to_a", 0),
                "count_a_to_b": cand.get("count_a_to_b", 0),
                "count_b_to_a": cand.get("count_b_to_a", 0),
                "gt_a": cand.get("gt_a", 0),
                "gt_b": cand.get("gt_b", 0),
            })
            total += count
            print(
                f"    [{cand_idx}/{len(merge_candidates)}] {cand_key}: {count} 张 "
                f"(T1={tier_counts['T1']}/T2={tier_counts['T2']}/T3={tier_counts['T3']})  "
                f"{name_a}={stats_a['images']}图/{stats_a['boxes']}框  "
                f"{name_b}={stats_b['images']}图/{stats_b['boxes']}框"
            )

    # 劣质类别：图片含该类即可
    if "drop" in export_kinds:
        risky = [q for q in class_quality if q["risk_level"] in {"极高", "高", "中"}]
        merged_class_ids: set[int] = set()
        for cand in merge_candidates:
            merged_class_ids.add(cand["class_id_a"])
            merged_class_ids.add(cand["class_id_b"])

        drop_total = sum(1 for q in risky if q["class_id"] not in merged_class_ids)
        if drop_total:
            print(f"  [drop] 导出 {drop_total} 个候选 ...")
        drop_idx = 0
        for q in risky:
            cid = q["class_id"]
            if cid in merged_class_ids:
                continue
            drop_idx += 1
            name = q["name"]
            cand_key = f"drop_{_safe_token(name)}"

            count = 0
            for split_name, infos in source_infos_by_split.items():
                if split_name in skip_splits:
                    continue
                if count >= max_images_per_drop:
                    break
                for info in infos:
                    if count >= max_images_per_drop:
                        break
                    if cid not in _label_cids(info):
                        continue
                    _emit_flat(info, split_name, cand_key)
                    count += 1

            stats_drop = {"images": class_image_count.get(cid, 0), "boxes": class_box_count.get(cid, 0)}
            candidate_stats.append({
                "kind": "drop",
                "key": cand_key,
                "images": count,
                "class_stats": stats_drop,
                "name": name, "cid": cid,
                "risk_level": q["risk_level"],
                "confidence": q.get("confidence", "normal"),
                "ap": q.get("ap", 0.0),
                "f1": q.get("f1", 0.0),
                "gt": q.get("gt", 0),
                "pred": q.get("pred", 0),
            })
            total += count
            print(
                f"    [{drop_idx}/{drop_total}] {cand_key}: {count} 张  "
                f"{name}={stats_drop['images']}图/{stats_drop['boxes']}框"
            )

    # 全局 README：所有候选汇总在一份
    readme_lines: list[str] = [
        "# 数据集优化预览",
        "",
        f"skip_splits: {sorted(skip_splits)}",
        f"预览混淆判定 IoU 阈值: {PREVIEW_MATCH_IOU_THRESHOLD}",
        f"图片上限：merge={max_images_per_merge}, drop={max_images_per_drop}",
        f"总图片数: {total}",
        f"未找到可视化图: {vis_missing_count} 张（manifest 中 visualization 为 null）"
        if vis_missing_count else f"未找到可视化图: 0",
        "",
        "目录说明:",
        "  visualizations/<候选>/   GT+pred 可视化，用文件管理器直接看",
        "  inference_dataset/       X-AnyLabeling 打开此目录即可看完所有候选",
        "    data.yaml / classes.txt  全局类名（原始 class_id）",
        "    images/<候选>__<原名>__<split>.<ext>",
        "    labels/<同名>.txt        YOLO 归一化预测结果",
        "  manifest.json            机器读，列出每张图归属哪个候选",
        "",
    ]
    merge_stats = [c for c in candidate_stats if c["kind"] == "merge"]
    drop_stats = [c for c in candidate_stats if c["kind"] == "drop"]
    if merge_stats:
        readme_lines += [
            f"## 合并候选 ({len(merge_stats)} 组)",
            "",
            "样本分层（按可信度从高到低）：",
            "  T1 黄金 — GT框与pred框 IoU≥0.5 且类别错配，最强混淆信号",
            "  T2 次优 — IoU 在 [0.3, 0.5) 错配，或 GT 有一类、pred 有另一类但没配上",
            "  T3 对照 — 图中只含其中一类、无另一类痕迹，给你看类别真实样貌做对照",
            f"  策略：先填 T1 至上限 {max_images_per_merge}，不足 {PREVIEW_MIN_PER_CANDIDATE} 张时用 T2 再 T3 兜底",
            "",
        ]
        for c in merge_stats:
            pool = c.get("tier_pool_sizes", {})
            readme_lines += [
                f"[{c['key']}]  共 {c['images']} 张  类型: {c['confusion_type']}混淆",
                f"  样本构成: T1黄金={c['tier_counts']['T1']}  T2次优={c['tier_counts']['T2']}  T3对照={c['tier_counts']['T3']}  "
                f"(候选池: T1={pool.get('T1', 0)} / T2={pool.get('T2', 0)} / T3={pool.get('T3', 0)})",
                f"  类别概况: {c['name_a']} (ID={c['cid_a']}) — {c['class_stats_a']['images']} 张图 / {c['class_stats_a']['boxes']} 框",
                f"            {c['name_b']} (ID={c['cid_b']}) — {c['class_stats_b']['images']} 张图 / {c['class_stats_b']['boxes']} 框",
                f"  混淆数据: {c['name_a']}→{c['name_b']}: {c['rate_a_to_b']:.1%} ({c['count_a_to_b']} 次 / GT={c['gt_a']})",
                f"            {c['name_b']}→{c['name_a']}: {c['rate_b_to_a']:.1%} ({c['count_b_to_a']} 次 / GT={c['gt_b']})",
                "",
            ]
    if drop_stats:
        readme_lines += [f"## 劣质类别候选 ({len(drop_stats)} 个)", ""]
        for c in drop_stats:
            cs = c.get("class_stats", {"images": 0, "boxes": 0})
            readme_lines += [
                f"[{c['key']}]  {c['images']} 张样本  "
                f"类别 {c['name']} 全集: {cs['images']} 图 / {cs['boxes']} 框",
                f"  风险={c['risk_level']}  可信度={c['confidence']}  "
                f"AP={c['ap']:.3f}  F1={c['f1']:.3f}  GT={c['gt']}  Pred={c['pred']}",
                "",
            ]
    (preview_dir / "README.txt").write_text("\n".join(readme_lines), encoding="utf-8")

    # manifest.json：机器读
    manifest = {
        "preview_dir": str(preview_dir),
        "skip_splits": sorted(skip_splits),
        "match_iou_threshold": PREVIEW_MATCH_IOU_THRESHOLD,
        "max_images_per_merge": max_images_per_merge,
        "max_images_per_drop": max_images_per_drop,
        "total_images": total,
        "visualizations_missing": vis_missing_count,
        "candidates": candidate_stats,
        "images": manifest_images,
    }
    (preview_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if vis_missing_count:
        print(
            f"  ⚠ 未找到 {vis_missing_count} 张图的可视化（visualizations/ 中缺失），"
            f"manifest 中标记 visualization=null"
        )

    return preview_dir, total


def _build_optimize_output_stem(*, source_root_name: str, output_suffix: str) -> str:
    normalized_suffix = output_suffix.strip() or OPTIMIZE_SUFFIX
    suffix_body = normalized_suffix.lstrip("_")
    if not suffix_body:
        suffix_body = OPTIMIZE_SUFFIX.lstrip("_")

    base_name = source_root_name
    if suffix_body.startswith("A") and base_name.endswith("_A"):
        base_name = base_name[:-2]

    return f"{base_name}_{suffix_body}"


# ---------------------------------------------------------------------------
# 5. 报告
# ---------------------------------------------------------------------------

def write_optimize_report(
    output_root: Path,
    *,
    source_data_yaml: Path,
    decisions: dict[str, Any],
    class_quality: list[dict[str, Any]],
    merge_candidates: list[dict[str, Any]],
    original_class_names: dict[int, str],
    new_class_names: dict[int, str],
    confusion_available: bool,
    dedup_total: int = 0,
) -> Path:
    """写出 optimize_summary.json 和 optimize_summary.md。"""
    merge_decisions = decisions["merge_decisions"]
    drop_decisions = decisions["drop_decisions"]

    summary = {
        "source_data": str(source_data_yaml),
        "confusion_matrix_available": confusion_available,
        "original_class_count": len(original_class_names),
        "original_class_names": {str(k): v for k, v in original_class_names.items()},
        "new_class_count": len(new_class_names),
        "new_class_names": {str(k): v for k, v in new_class_names.items()},
        "merge_decisions": merge_decisions,
        "drop_decisions": drop_decisions,
        "class_quality": class_quality,
        "merge_candidates_analyzed": merge_candidates,
        "dedup_overlap_boxes": dedup_total,
        "dedup_iou_threshold": MERGE_DEDUP_IOU,
    }
    json_path = output_root / "optimize_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Markdown 报告
    lines: list[str] = [
        "# 数据集优化报告",
        "",
        f"**源数据集**: `{source_data_yaml}`",
        "",
    ]

    if merge_decisions:
        lines += ["## 合并操作", "", "| 原类别 | 合并后 | 原因 |", "|--------|--------|------|"]
        for d in merge_decisions:
            names_str = " + ".join(
                original_class_names.get(cid, str(cid)) for cid in d["class_ids"]
            )
            lines.append(f"| {names_str} | {d['merged_name']} | {d['reason']} |")
        lines.append("")
        if dedup_total > 0:
            lines += [
                f"> **自动去重重叠框**: {dedup_total} 个 "
                f"(同 new_id 组内 IoU≥{MERGE_DEDUP_IOU}，保留面积大的)",
                "",
            ]

    if drop_decisions:
        lines += [
            "## 删除操作",
            "",
            "| 类别 | AP | F1 | GT数 | 原因 |",
            "|------|----|----|------|------|",
        ]
        for d in drop_decisions:
            m = d.get("metrics", {})
            lines.append(
                f"| {d['name']} | {m.get('ap', 0):.3f}"
                f" | {m.get('f1', 0):.3f}"
                f" | {m.get('gt', 0)}"
                f" | {d['reason']} |"
            )
        lines.append("")

    lines += [
        "## 新数据集类别",
        "",
        f"共 {len(new_class_names)} 个类别：",
        "",
    ]
    for i, name in sorted(new_class_names.items()):
        lines.append(f"- {i}: {name}")
    lines.append("")

    if not confusion_available:
        lines += [
            "> **注意**: 未找到推理输出中的预测 JSON 文件，无法计算混淆矩阵。",
            "> 建议推理时使用 `--save-json` 选项，再次运行优化可获得更准确的合并候选分析。",
            "",
        ]

    md_path = output_root / "optimize_summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# 6. 归档
# ---------------------------------------------------------------------------

def _archive_optimize_report(
    *,
    optimize_experiment_dir: Path,
    source_data_yaml: Path,
    copied_count: int,
    report_json_path: Path | None,
    summary_json: Path,
    summary_md: Path,
) -> Path:
    """将优化报告归档到 <experiment_dir>/optimize/<date>-<dataset>-<run>-<N>img/。

    复制 test_report.json、optimize_summary.json、optimize_summary.md，返回归档目录。
    """
    date_tag = rt.date_tag_now()
    # 数据集名：data.yaml 上两级目录（e.g. datasets/wuwanPic_dataset/dataset_det → wuwanPic_dataset）
    dataset_name = source_data_yaml.parent.parent.name
    run_name = optimize_experiment_dir.name
    folder_name = f"{date_tag}-{dataset_name}-{run_name}-{copied_count}img"

    archive_dir = optimize_experiment_dir / "optimize" / folder_name
    archive_dir.mkdir(parents=True, exist_ok=True)

    if report_json_path is not None and report_json_path.exists():
        shutil.copy2(report_json_path, archive_dir / "test_report.json")
    if summary_json.exists():
        shutil.copy2(summary_json, archive_dir / summary_json.name)
    if summary_md.exists():
        shutil.copy2(summary_md, archive_dir / summary_md.name)

    print(f"\n[det/optimize] 优化报告已归档至:")
    print(f"  {archive_dir}")
    return archive_dir


def _cleanup_auto_infer_temp_dir(auto_infer_temp_dir: Path | None) -> None:
    if auto_infer_temp_dir is not None and auto_infer_temp_dir.exists():
        print(f"\n[det/optimize] 清理临时推理目录 ...")
        print(f"  {auto_infer_temp_dir}")
        shutil.rmtree(auto_infer_temp_dir)
        print("  已删除。")


def _cleanup_preview_dir(preview_dir: Path | None) -> None:
    if preview_dir is not None and preview_dir.exists():
        shutil.rmtree(preview_dir)


# ---------------------------------------------------------------------------
# 7. 入口
# ---------------------------------------------------------------------------

def run_optimize(args) -> None:
    source_data_yaml: Path = Path(args.source_data_yaml).expanduser().resolve()
    report_json_path: Path | None = getattr(args, "report_json", None)
    report_jsons: list[Path] | None = getattr(args, "report_jsons", None)
    infer_output_dir: Path | None = getattr(args, "infer_output_dir", None)
    confusion_threshold: float = float(getattr(args, "confusion_threshold", DEFAULT_CONFUSION_THRESHOLD))

    # 1. 加载 per_class_ap（支持多报告合并）
    per_class_ap: dict[str, Any] = {}
    if report_jsons and len(report_jsons) > 1:
        valid_reports = [p for p in report_jsons if p.exists()]
        if valid_reports:
            per_class_ap = _merge_per_class_ap(valid_reports)
            print(f"[det/optimize] 已合并 {len(valid_reports)} 份报告的类别指标。")
    elif report_json_path is not None and report_json_path.exists():
        try:
            report_payload = json.loads(report_json_path.read_text(encoding="utf-8"))
            per_class_ap = report_payload.get("per_class_ap") or {}
        except Exception as exc:
            print(f"[det/optimize] 警告：无法读取 test_report: {exc}")

    if not per_class_ap:
        print("[det/optimize] 未提供或无法读取 test_report.json，将仅根据标签统计分析。")

    # 2. 加载源数据集类别名
    data_cfg = rt.load_data_config(source_data_yaml)
    class_names = rt.normalize_names(data_cfg.get("names"))
    for class_id_str, info in per_class_ap.items():
        if isinstance(info, dict) and "name" in info:
            cid = int(class_id_str)
            if cid not in class_names:
                class_names[cid] = str(info["name"])

    # 3. 质量分析
    class_quality = analyze_class_quality(per_class_ap) if per_class_ap else []

    # 4. 混淆矩阵（可选）
    confusion_matrix = None
    merge_candidates: list[dict[str, Any]] = []
    confusion_available = False

    if infer_output_dir is not None and infer_output_dir.exists():
        print(f"\n[det/optimize] 构建混淆矩阵 ... (dir={infer_output_dir})")
        confusion_matrix = build_confusion_matrix(
            infer_output_dir=infer_output_dir,
            source_data_yaml=source_data_yaml,
            class_names=class_names,
        )
        if confusion_matrix is not None:
            confusion_available = True
            merge_candidates = find_merge_candidates(confusion_matrix, threshold=confusion_threshold)
            print(
                f"  处理图片={confusion_matrix['processed_images']}"
                f"  合并候选={len(merge_candidates)}"
            )
        else:
            print("  未找到预测 JSON，跳过混淆矩阵。")

    # 5. 打印分析
    print("\n" + "=" * 62)
    print("  数据集优化分析")
    print("=" * 62)
    print("\n[劣质类别分析]")
    if class_quality:
        print_quality_table(class_quality)
    else:
        print("  无 test_report 数据，跳过。")
    print("\n[合并候选分析]")
    if confusion_available:
        print_confusion_table(merge_candidates)
    else:
        print("  未提供推理输出目录（infer_output_dir）或未找到预测 JSON，跳过混淆矩阵。")
        print("  提示：推理时加 --save-json，再选择 infer_output_dir 可启用此功能。")

    optimize_experiment_dir: Path | None = getattr(args, "optimize_experiment_dir", None)
    auto_infer_temp_dir: Path | None = getattr(args, "auto_infer_temp_dir", None)
    if optimize_experiment_dir is not None:
        optimize_experiment_dir = optimize_experiment_dir.expanduser().resolve()

    # 6. 导出预览：让用户先看到受影响的图片再做决定
    preview_dir: Path | None = None
    has_merge = bool(merge_candidates)
    has_drop = any(q["risk_level"] in {"极高", "高", "中"} for q in class_quality)

    try:
        if has_merge or has_drop:
            export_kinds = _prompt_preview_export_kinds(has_merge=has_merge, has_drop=has_drop)
            if export_kinds:
                preview_dir, preview_count = _export_preview_for_review(
                    merge_candidates=merge_candidates,
                    class_quality=class_quality,
                    source_data_yaml=source_data_yaml,
                    class_names=class_names,
                    infer_output_dir=infer_output_dir if infer_output_dir is not None and infer_output_dir.exists() else None,
                    export_kinds=export_kinds,
                )
                print(f"\n  ★ 预览已导出（{preview_count} 张受影响图片）: {preview_dir}")
                zip_path = _zip_preview_dir(preview_dir)
                zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                print(f"    ★ 已打包: {zip_path}  ({zip_size_mb:.1f} MB，可直接 scp 下载)")
                print("    请查看图片和标签，然后决定是否执行操作。\n")

        # 7. 交互确认
        decisions = interactive_collect_decisions(class_quality, merge_candidates, class_names)
        if decisions is None:
            _cleanup_auto_infer_temp_dir(auto_infer_temp_dir)
            return

        # 8. 写出新数据集
        output_root, new_class_names, copied_count, dedup_total = apply_optimize_decisions(
            decisions=decisions,
            source_data_yaml=source_data_yaml,
        )

        # 9. 写优化报告到新数据集目录
        original_class_names = rt.normalize_names(data_cfg.get("names"))
        md_path = write_optimize_report(
            output_root,
            source_data_yaml=source_data_yaml,
            decisions=decisions,
            class_quality=class_quality,
            merge_candidates=merge_candidates,
            original_class_names=original_class_names,
            new_class_names=new_class_names,
            confusion_available=confusion_available,
            dedup_total=dedup_total,
        )

        # 10. 若是 auto-infer 流程，将报告归档到 optimize/ 文件夹，然后清理临时推理目录
        opt_report_dir: Path | None = None
        if optimize_experiment_dir is not None:
            opt_report_dir = _archive_optimize_report(
                optimize_experiment_dir=optimize_experiment_dir,
                source_data_yaml=source_data_yaml,
                copied_count=copied_count,
                report_json_path=report_json_path,
                summary_json=output_root / "optimize_summary.json",
                summary_md=md_path,
            )

        _cleanup_auto_infer_temp_dir(auto_infer_temp_dir)
    finally:
        _cleanup_preview_dir(preview_dir)

    print("\n[det/optimize] 完成！")
    print(f"  新数据集 : {output_root}")
    if opt_report_dir is not None:
        print(f"  优化报告 : {opt_report_dir}")
    else:
        print(f"  优化报告 : {md_path}")
    print(f"  新类别数 : {len(new_class_names)}")
    print(f"  合并操作 : {len(decisions['merge_decisions'])} 组")
    print(f"  删除操作 : {len(decisions['drop_decisions'])} 类")
