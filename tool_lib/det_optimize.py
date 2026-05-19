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
PREVIEW_MAX_IMAGES_PER_CATEGORY = 50


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
) -> tuple[Path, dict[int, str], int]:
    """执行合并/删除并写出新数据集，返回 (新数据集根目录, 新类别名称)。

    不修改原始数据集任何文件。
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

            # 重写标签：跳过被删类，合并类改 class_id
            remapped: list[str] = []
            affected_ops: set[str] = set()  # 该图片涉及的操作标签
            for line in info.label_lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_id = int(float(parts[0]))
                new_id = id_map.get(old_id)
                if new_id is None:
                    # 删除操作
                    if old_id in dropped_ids:
                        drop_name = original_class_names.get(old_id, str(old_id))
                        affected_ops.add(f"drop_{drop_name}")
                    continue
                if new_id != old_id:
                    # 合并操作
                    md = merge_groups.get(old_id)
                    if md is not None:
                        names_str = "+".join(
                            original_class_names.get(c, str(c)) for c in md["class_ids"]
                        )
                        affected_ops.add(f"merge_{names_str}→{md['merged_name']}")
                remapped.append(f"{new_id} {' '.join(parts[1:])}")

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

    return final_root, new_names, copied


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


def _export_preview_for_review(
    *,
    merge_candidates: list[dict[str, Any]],
    class_quality: list[dict[str, Any]],
    source_data_yaml: Path,
    class_names: dict[int, str],
    infer_output_dir: Path | None = None,
    output_suffix: str = OPTIMIZE_SUFFIX,
    max_images_per_category: int = PREVIEW_MAX_IMAGES_PER_CATEGORY,
    skip_splits: frozenset[str] = PREVIEW_SKIP_SPLITS,
) -> tuple[Path, int]:
    """在确认前导出受影响图片的预览，供用户查看后再决定。

    为压缩下载体积、只保留能让用户判断的样本，遵循三条规则：
      - 仅导出 val/test 等非训练 split（由 skip_splits 控制）。
      - 合并候选只收同时含两类的图片；劣质类别只收含该类的图片。
      - 每个候选最多 max_images_per_category 张。

    每个候选目录有两个子目录：
      - with_gt/         可视化图：同时画 GT + 推理框，便于人眼判断。
      - inference_only/  原图 + 仅含推理结果的 YOLO label，便于在 X-AnyLabeling 中查看。

    返回 (preview_dir, total_preview_images)。
    """
    source_cfg = rt.load_data_config(source_data_yaml)
    source_root = Path(source_cfg["_root_dir"])
    source_infos_by_split, _ = collect_source_image_infos(source_cfg, source_root)

    pred_lookup: dict[str, dict[str, Any]] = {}
    if infer_output_dir is not None and infer_output_dir.exists():
        pred_lookup = _build_pred_json_lookup(infer_output_dir)

    preview_dir = rt.deduplicate_path(source_root.parent / f"_preview_optimize{id(source_root):x}")
    preview_dir.mkdir(parents=True, exist_ok=True)

    sorted_cids = sorted(class_names.keys())
    full_names_map = {i: class_names[cid] for i, cid in enumerate(sorted_cids)}
    # 推理 label 里写的是原始 class_id，X-AnyLabeling 直接用原始 class_id 索引这里的 names
    full_names_by_original_id = {cid: class_names[cid] for cid in sorted_cids}
    max_cid = max(sorted_cids) if sorted_cids else -1

    def _write_inference_only_yaml(subfolder: Path) -> None:
        # 用原始 class_id 作为 key，与推理 label 中的 class_id 对齐
        names_for_yaml = {
            cid: full_names_by_original_id.get(cid, str(cid))
            for cid in range(max_cid + 1)
        }
        cfg = {
            "path": str(subfolder),
            "train": "images",
            "val": "images",
            "test": "images",
            "task": "detect",
            "nc": max_cid + 1,
            "names": names_for_yaml,
        }
        rt.dump_yaml(subfolder / "data.yaml", cfg)
        (subfolder / "classes.txt").write_text(
            "\n".join(names_for_yaml[i] for i in range(max_cid + 1)) + "\n",
            encoding="utf-8",
        )

    def _emit_pair(
        info: Any,
        split_name: str,
        with_gt_dir: Path,
        inference_only_dir: Path,
    ) -> None:
        # with_gt：复制带 GT+推理框的可视化图，找不到回退原图
        with_gt_img = with_gt_dir / "images" / split_name / info.rel_path
        with_gt_img.parent.mkdir(parents=True, exist_ok=True)
        vis: Path | None = None
        if infer_output_dir is not None:
            vis = _find_vis_image(infer_output_dir, info.src_image_path)
        shutil.copy2(vis if vis is not None else info.src_image_path, with_gt_img)

        # inference_only：原图 + 仅推理结果的 YOLO label
        infer_img = inference_only_dir / "images" / split_name / info.rel_path
        infer_lbl = inference_only_dir / "labels" / split_name / info.rel_path.with_suffix(".txt")
        infer_img.parent.mkdir(parents=True, exist_ok=True)
        infer_lbl.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(info.src_image_path, infer_img)
        pred_payload = pred_lookup.get(info.src_image_path.stem)
        lines = _inference_labels_yolo(pred_payload) if pred_payload is not None else []
        infer_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

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

    total = 0

    # 合并候选：图片必须同时含 cid_a 和 cid_b
    for cand in merge_candidates:
        cid_a, cid_b = cand["class_id_a"], cand["class_id_b"]
        name_a = cand.get("name_a", class_names.get(cid_a, str(cid_a)))
        name_b = cand.get("name_b", class_names.get(cid_b, str(cid_b)))
        confusion_type = cand.get("confusion_type", "")

        op_dir = preview_dir / f"merge_{name_a}+{name_b}"
        with_gt_dir = op_dir / "with_gt"
        inference_only_dir = op_dir / "inference_only"
        with_gt_dir.mkdir(parents=True, exist_ok=True)
        inference_only_dir.mkdir(parents=True, exist_ok=True)
        _write_inference_only_yaml(inference_only_dir)

        count = 0
        for split_name, infos in source_infos_by_split.items():
            if split_name in skip_splits:
                continue
            if count >= max_images_per_category:
                break
            for info in infos:
                if count >= max_images_per_category:
                    break
                cids = _label_cids(info)
                if cid_a not in cids or cid_b not in cids:
                    continue
                _emit_pair(info, split_name, with_gt_dir, inference_only_dir)
                count += 1

        (op_dir / "README.txt").write_text(
            f"类别: {name_a} (ID={cid_a}) + {name_b} (ID={cid_b})\n"
            f"类型: {confusion_type}混淆\n"
            f"图片数: {count} (上限 {max_images_per_category}，仅含同时出现两类的图)\n"
            f"跳过 split: {sorted(skip_splits)}\n\n"
            f"with_gt/         同时画了 GT 和推理框，用于判断是否合并\n"
            f"inference_only/  原图 + 仅含推理结果的 YOLO label，可在 X-AnyLabeling 中查看\n\n"
            f"{name_a}→{name_b}: {cand.get('rate_a_to_b', 0):.1%} "
            f"({cand.get('count_a_to_b', 0)} 次 / GT={cand.get('gt_a', 0)})\n"
            f"{name_b}→{name_a}: {cand.get('rate_b_to_a', 0):.1%} "
            f"({cand.get('count_b_to_a', 0)} 次 / GT={cand.get('gt_b', 0)})\n",
            encoding="utf-8",
        )
        total += count

    # 劣质类别：图片含该类即可
    risky = [q for q in class_quality if q["risk_level"] in {"极高", "高", "中"}]
    merged_class_ids: set[int] = set()
    for cand in merge_candidates:
        merged_class_ids.add(cand["class_id_a"])
        merged_class_ids.add(cand["class_id_b"])

    for q in risky:
        cid = q["class_id"]
        if cid in merged_class_ids:
            continue
        name = q["name"]
        op_dir = preview_dir / f"drop_{name}"
        with_gt_dir = op_dir / "with_gt"
        inference_only_dir = op_dir / "inference_only"
        with_gt_dir.mkdir(parents=True, exist_ok=True)
        inference_only_dir.mkdir(parents=True, exist_ok=True)
        _write_inference_only_yaml(inference_only_dir)

        count = 0
        for split_name, infos in source_infos_by_split.items():
            if split_name in skip_splits:
                continue
            if count >= max_images_per_category:
                break
            for info in infos:
                if count >= max_images_per_category:
                    break
                if cid not in _label_cids(info):
                    continue
                _emit_pair(info, split_name, with_gt_dir, inference_only_dir)
                count += 1

        (op_dir / "README.txt").write_text(
            f"类别: {name} (ID={cid})\n"
            f"风险等级: {q['risk_level']}  可信度: {q.get('confidence', 'normal')}\n"
            f"AP={q['ap']:.3f}  F1={q['f1']:.3f}  GT={q['gt']}  Pred={q['pred']}\n"
            f"图片数: {count} (上限 {max_images_per_category})\n"
            f"跳过 split: {sorted(skip_splits)}\n\n"
            f"with_gt/         同时画了 GT 和推理框，用于判断是否删除\n"
            f"inference_only/  原图 + 仅含推理结果的 YOLO label，可在 X-AnyLabeling 中查看\n",
            encoding="utf-8",
        )
        total += count

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
    if merge_candidates or any(
        q["risk_level"] in {"极高", "高", "中"} for q in class_quality
    ):
        preview_dir, preview_count = _export_preview_for_review(
            merge_candidates=merge_candidates,
            class_quality=class_quality,
            source_data_yaml=source_data_yaml,
            class_names=class_names,
            infer_output_dir=infer_output_dir if infer_output_dir is not None and infer_output_dir.exists() else None,
        )
        print(f"\n  ★ 预览已导出（{preview_count} 张受影响图片）: {preview_dir}")
        print("    请在文件管理器中查看图片和标签，然后决定是否执行操作。\n")

    # 7. 交互确认
    decisions = interactive_collect_decisions(class_quality, merge_candidates, class_names)
    if decisions is None:
        _cleanup_preview_dir(preview_dir)
        _cleanup_auto_infer_temp_dir(auto_infer_temp_dir)
        return

    _cleanup_preview_dir(preview_dir)

    # 8. 写出新数据集
    output_root, new_class_names, copied_count = apply_optimize_decisions(
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

    print("\n[det/optimize] 完成！")
    print(f"  新数据集 : {output_root}")
    if opt_report_dir is not None:
        print(f"  优化报告 : {opt_report_dir}")
    else:
        print(f"  优化报告 : {md_path}")
    print(f"  新类别数 : {len(new_class_names)}")
    print(f"  合并操作 : {len(decisions['merge_decisions'])} 组")
    print(f"  删除操作 : {len(decisions['drop_decisions'])} 类")
