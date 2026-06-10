"""模型辅助的质检抽样（Step 3 实现）。

基于已有的推理结果（test_report.json + 推理输出JSON），精准识别：
- 漏标 (missing): GT有，但模型在IoU≥0.5下无匹配Pred
- 错标 (swapped): GT↔Pred匹配，但类别不一致
- 误检 (false_pos): Pred有，但与所有GT IoU<0.5
- 框定位差 (loc): GT↔Pred匹配，IoU∈[0.3, 0.5)
- 低置信度 (low_conf): Pred score < threshold

复用 det_optimize.py 中的分析能力（混淆矩阵、类别质量分析）。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import common as rt


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    """计算两个框的IoU"""
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
    """匈牙利算法求解二分图最小代价匹配"""
    n_rows = len(cost_matrix)
    if n_rows == 0:
        return []
    n_cols = len(cost_matrix[0])

    # 找一个大有限值用于填充
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

    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)

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

    result: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        i = p[j]
        if i >= 1 and i <= n_rows and j <= n_cols:
            if cost_matrix[i - 1][j - 1] != float('inf'):
                result.append((i - 1, j - 1))
    return result


def analyze_class_quality(per_class_ap: dict[str, Any]) -> list[dict[str, Any]]:
    """从test_report.per_class_ap计算每类质量指标和风险等级"""
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
        denom = precision + recall
        f1 = 2 * precision * recall / denom if denom > 0 else 0.0
        fp = pred - tp
        fp_rate = fp / pred if pred > 0 else 0.0

        # 风险分
        risk = 0.0
        if gt == 0:
            risk = 1.0
            confidence = "none"
        elif gt < 5:
            risk = 0.6 + 0.4 * (1.0 - gt / 5)
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


# ── 默认阈值 ──────────────────────────────────────────────
MATCH_IOU_THRESHOLD = 0.5  # 匹配IoU阈值
LOC_IOU_THRESHOLD = 0.3    # 框定位差IoU阈值
LOW_CONF_THRESHOLD = 0.3   # 低置信度阈值
OUTLIER_CLASS_AP = 0.1     # 劣质类别AP阈值
OUTLIER_CLASS_GT = 5       # 劣质类别GT数阈值


# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class PredBox:
    """预测框"""
    class_id: int
    bbox_xyxy: list[float]  # [x1, y1, x2, y2]
    score: float


@dataclass
class GTBox:
    """真实标注框"""
    class_id: int
    bbox_xyxy: list[float]  # [x1, y1, x2, y2]


@dataclass
class MatchResult:
    """GT↔Pred匹配结果"""
    gt_idx: int
    pred_idx: int
    iou: float
    gt_class_id: int
    pred_class_id: int
    class_match: bool


@dataclass
class ImageAnalysisResult:
    """单图分析结果"""
    image_path: Path
    gt_boxes: list[GTBox]
    pred_boxes: list[PredBox]
    matches: list[MatchResult]
    missing_gts: list[int]      # GT未匹配到Pred的索引
    false_pos_preds: list[int]  # Pred未匹配到GT的索引
    swapped_matches: list[MatchResult]  # 类别不一致的匹配
    loc_matches: list[MatchResult]      # 框定位差的匹配
    low_conf_preds: list[int]           # 低置信度Pred的索引
    flags: dict[str, list[dict[str, Any]]]  # 按flag类型分组的问题


@dataclass
class ModelAnalysisSummary:
    """模型分析汇总"""
    total_images: int = 0
    total_gt_boxes: int = 0
    total_pred_boxes: int = 0
    missing_count: int = 0
    swapped_count: int = 0
    false_pos_count: int = 0
    loc_count: int = 0
    low_conf_count: int = 0
    outlier_class_count: int = 0
    images_with_problems: int = 0
    # 按类别统计
    class_missing: dict[int, int] = field(default_factory=dict)
    class_swapped: dict[int, int] = field(default_factory=dict)
    class_false_pos: dict[int, int] = field(default_factory=dict)


# ── 推理结果加载 ──────────────────────────────────────────


def discover_infer_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    """从实验目录发现推理结果。

    扫描 run_meta.json，返回该实验下的所有推理结果列表。
    """
    experiment_dir = experiment_dir.expanduser().resolve()
    runs: list[dict[str, Any]] = []

    for run_meta_path in experiment_dir.rglob("run_meta.json"):
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(run_meta, dict):
            continue
        if run_meta.get("task") != "det" or run_meta.get("action") != "infer":
            continue

        # 检查是否属于当前实验
        paths_payload = run_meta.get("paths", {})
        recorded_experiment_dir = paths_payload.get("experiment_dir")
        if recorded_experiment_dir:
            recorded_path = Path(recorded_experiment_dir).expanduser().resolve()
            if recorded_path != experiment_dir:
                continue

        output_dir = paths_payload.get("output_dir")
        if output_dir:
            output_dir = Path(output_dir).expanduser().resolve()
        else:
            output_dir = run_meta_path.parent

        # 查找report_path
        report_path = None
        report_path_str = paths_payload.get("report_path")
        if report_path_str:
            report_path = Path(report_path_str).expanduser().resolve()
            if not report_path.exists():
                report_path = None

        # 自动查找test_report.json
        if report_path is None:
            for pattern in ["*test_report.json", "test_report.json"]:
                candidates = sorted(output_dir.glob(pattern))
                if candidates:
                    report_path = candidates[-1]
                    break

        split = run_meta.get("split", "unknown")
        created_at = run_meta.get("created_at")

        runs.append({
            "output_dir": output_dir,
            "split": split,
            "report_path": report_path,
            "run_meta_path": run_meta_path,
            "created_at": created_at,
            "run_meta": run_meta,
        })

    # 按时间排序，最新的在前
    runs.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return runs


def load_report(report_path: Path) -> dict[str, Any] | None:
    """加载test_report.json"""
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_pred_json(pred_json_path: Path) -> dict[str, Any] | None:
    """加载单张图的预测JSON"""
    if not pred_json_path.exists():
        return None
    try:
        return json.loads(pred_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_pred_json_for_image(
    infer_output_dir: Path,
    image_path: Path,
) -> Path | None:
    """在推理输出目录中查找对应图片的预测JSON"""
    temp_dir = rt.infer_temp_dir(infer_output_dir)
    json_dir = temp_dir / "json"
    if not json_dir.exists():
        json_dir = infer_output_dir

    # 尝试多种匹配方式
    stem = image_path.stem
    candidates = list(json_dir.rglob(f"{stem}.json"))
    if candidates:
        return candidates[0]

    # 尝试按相对路径匹配
    for json_path in json_dir.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                pred_image_path = payload.get("image_path", "")
                if Path(pred_image_path).stem == stem:
                    return json_path
        except Exception:
            continue

    return None


# ── 逐图分析 ──────────────────────────────────────────────


def parse_gt_label(label_path: Path, width: int, height: int) -> list[GTBox]:
    """解析YOLO格式的GT标签"""
    gt_boxes = []
    if not label_path.exists():
        return gt_boxes

    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            cx = float(parts[1]) * width
            cy = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            gt_boxes.append(GTBox(class_id=class_id, bbox_xyxy=[x1, y1, x2, y2]))
        except (ValueError, IndexError):
            continue

    return gt_boxes


def parse_pred_boxes(pred_payload: dict[str, Any]) -> list[PredBox]:
    """解析预测JSON中的预测框"""
    pred_boxes = []
    for pred in pred_payload.get("predictions", []):
        try:
            class_id = int(pred.get("class_id", -1))
            bbox = pred.get("bbox_xyxy", [])
            score = float(pred.get("score", 0.0))
            if class_id >= 0 and len(bbox) == 4:
                pred_boxes.append(PredBox(
                    class_id=class_id,
                    bbox_xyxy=[float(v) for v in bbox],
                    score=score,
                ))
        except (ValueError, TypeError):
            continue
    return pred_boxes


def analyze_single_image(
    gt_boxes: list[GTBox],
    pred_boxes: list[PredBox],
    match_iou_threshold: float = MATCH_IOU_THRESHOLD,
    loc_iou_threshold: float = LOC_IOU_THRESHOLD,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
) -> ImageAnalysisResult:
    """分析单张图的GT↔Pred匹配情况"""

    # 构建代价矩阵：-score（因为匈牙利算法求最小代价）
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    if n_gt == 0 and n_pred == 0:
        return ImageAnalysisResult(
            image_path=Path(),
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            matches=[],
            missing_gts=[],
            false_pos_preds=list(range(n_pred)),
            swapped_matches=[],
            loc_matches=[],
            low_conf_preds=[],
            flags={},
        )

    # 计算IoU矩阵
    cost_matrix = [[float('inf')] * n_pred for _ in range(n_gt)]
    iou_matrix = [[0.0] * n_pred for _ in range(n_gt)]

    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = _box_iou(gt.bbox_xyxy, pred.bbox_xyxy)
            iou_matrix[gi][pi] = iou
            if iou >= match_iou_threshold:
                # 代价 = -score * iou（匹配质量）
                cost_matrix[gi][pi] = -(pred.score * iou)

    # 匈牙利匹配
    raw_matches = _hungarian_match(cost_matrix)

    # 解析匹配结果
    matches: list[MatchResult] = []
    matched_gt_indices: set[int] = set()
    matched_pred_indices: set[int] = set()

    for gi, pi in raw_matches:
        if cost_matrix[gi][pi] == float('inf'):
            continue
        iou = iou_matrix[gi][pi]
        gt = gt_boxes[gi]
        pred = pred_boxes[pi]
        matches.append(MatchResult(
            gt_idx=gi,
            pred_idx=pi,
            iou=iou,
            gt_class_id=gt.class_id,
            pred_class_id=pred.class_id,
            class_match=(gt.class_id == pred.class_id),
        ))
        matched_gt_indices.add(gi)
        matched_pred_indices.add(pi)

    # 分类问题
    missing_gts = [i for i in range(n_gt) if i not in matched_gt_indices]
    false_pos_preds = [i for i in range(n_pred) if i not in matched_pred_indices]
    swapped_matches = [m for m in matches if not m.class_match]
    loc_matches = [m for m in matches if m.class_match and m.iou < loc_iou_threshold]
    low_conf_preds = [i for i, p in enumerate(pred_boxes) if p.score < low_conf_threshold]

    # 构建flag字典
    flags: dict[str, list[dict[str, Any]]] = {}

    for gi in missing_gts:
        gt = gt_boxes[gi]
        flags.setdefault("missing", []).append({
            "type": "missing",
            "gt_class_id": gt.class_id,
            "gt_bbox": gt.bbox_xyxy,
        })

    for pi in false_pos_preds:
        pred = pred_boxes[pi]
        flags.setdefault("false_pos", []).append({
            "type": "false_pos",
            "pred_class_id": pred.class_id,
            "pred_bbox": pred.bbox_xyxy,
            "pred_score": pred.score,
        })

    for m in swapped_matches:
        flags.setdefault("swapped", []).append({
            "type": "swapped",
            "gt_class_id": m.gt_class_id,
            "pred_class_id": m.pred_class_id,
            "gt_bbox": gt_boxes[m.gt_idx].bbox_xyxy,
            "pred_bbox": pred_boxes[m.pred_idx].bbox_xyxy,
            "iou": m.iou,
        })

    for m in loc_matches:
        flags.setdefault("loc", []).append({
            "type": "loc",
            "class_id": m.gt_class_id,
            "gt_bbox": gt_boxes[m.gt_idx].bbox_xyxy,
            "pred_bbox": pred_boxes[m.pred_idx].bbox_xyxy,
            "iou": m.iou,
        })

    for pi in low_conf_preds:
        pred = pred_boxes[pi]
        flags.setdefault("low_conf", []).append({
            "type": "low_conf",
            "pred_class_id": pred.class_id,
            "pred_score": pred.score,
        })

    return ImageAnalysisResult(
        image_path=Path(),
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        matches=matches,
        missing_gts=missing_gts,
        false_pos_preds=false_pos_preds,
        swapped_matches=swapped_matches,
        loc_matches=loc_matches,
        low_conf_preds=low_conf_preds,
        flags=flags,
    )


# ── 批量分析 ──────────────────────────────────────────────


def analyze_dataset_with_model(
    source_data_yaml: Path,
    infer_output_dir: Path,
    report_path: Path | None = None,
    *,
    match_iou_threshold: float = MATCH_IOU_THRESHOLD,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
    outlier_class_ap: float = OUTLIER_CLASS_AP,
    outlier_class_gt: int = OUTLIER_CLASS_GT,
    progress_callback: Any = None,
) -> tuple[list[ImageAnalysisResult], ModelAnalysisSummary, dict[str, Any]]:
    """批量分析数据集，返回每图分析结果、汇总、类别质量分析"""

    source_cfg = rt.load_data_config(source_data_yaml)
    source_root = Path(source_cfg["_root_dir"])
    class_names = rt.normalize_names(source_cfg.get("names"))

    # 加载报告
    report_payload = None
    if report_path and report_path.exists():
        report_payload = load_report(report_path)

    per_class_ap = {}
    if report_payload:
        per_class_ap = report_payload.get("per_class_ap", {})

    # 类别质量分析
    class_quality = analyze_class_quality(per_class_ap) if per_class_ap else []

    # 构建预测JSON查找表
    temp_dir = rt.infer_temp_dir(infer_output_dir)
    json_dir = temp_dir / "json"
    if not json_dir.exists():
        json_dir = infer_output_dir

    pred_lookup: dict[str, Path] = {}
    for json_path in json_dir.rglob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "predictions" in payload:
                image_path = Path(str(payload.get("image_path", "")))
                if image_path.stem:
                    pred_lookup[image_path.stem] = json_path
        except Exception:
            continue

    # 扫描数据集
    source_infos_by_split, _ = collect_source_image_infos(source_cfg, source_root)

    results: list[ImageAnalysisResult] = []
    summary = ModelAnalysisSummary()

    total_images = sum(len(infos) for infos in source_infos_by_split.values())
    processed = 0

    for split_name, infos in source_infos_by_split.items():
        for info in infos:
            processed += 1
            if progress_callback and processed % 100 == 0:
                progress_callback(processed, total_images, f"分析中: {info.rel_path}")

            # 查找对应的预测JSON
            pred_json_path = pred_lookup.get(info.src_image_path.stem)
            if pred_json_path is None:
                continue

            pred_payload = load_pred_json(pred_json_path)
            if pred_payload is None:
                continue

            # 获取图片尺寸
            width = int(pred_payload.get("width", 0))
            height = int(pred_payload.get("height", 0))
            if width <= 0 or height <= 0:
                continue

            # 解析GT和Pred
            gt_boxes = parse_gt_label(info.src_label_path, width, height)
            pred_boxes = parse_pred_boxes(pred_payload)

            # 分析
            analysis = analyze_single_image(
                gt_boxes, pred_boxes,
                match_iou_threshold=match_iou_threshold,
                low_conf_threshold=low_conf_threshold,
            )
            analysis.image_path = info.src_image_path
            results.append(analysis)

            # 更新汇总
            summary.total_gt_boxes += len(gt_boxes)
            summary.total_pred_boxes += len(pred_boxes)
            summary.missing_count += len(analysis.missing_gts)
            summary.swapped_count += len(analysis.swapped_matches)
            summary.false_pos_count += len(analysis.false_pos_preds)
            summary.loc_count += len(analysis.loc_matches)
            summary.low_conf_count += len(analysis.low_conf_preds)

            if analysis.flags:
                summary.images_with_problems += 1

            # 按类别统计
            for gi in analysis.missing_gts:
                cid = gt_boxes[gi].class_id
                summary.class_missing[cid] = summary.class_missing.get(cid, 0) + 1

            for m in analysis.swapped_matches:
                cid = m.gt_class_id
                summary.class_swapped[cid] = summary.class_swapped.get(cid, 0) + 1

            for pi in analysis.false_pos_preds:
                cid = pred_boxes[pi].class_id
                summary.class_false_pos[cid] = summary.class_false_pos.get(cid, 0) + 1

    summary.total_images = len(results)

    # 统计劣质类别
    outlier_classes = []
    for q in class_quality:
        if q["ap"] < outlier_class_ap or q["gt"] < outlier_class_gt:
            outlier_classes.append(q["class_id"])
    summary.outlier_class_count = len(outlier_classes)

    return results, summary, {
        "class_quality": class_quality,
        "outlier_classes": outlier_classes,
        "per_class_ap": per_class_ap,
    }


# ── 辅助函数 ──────────────────────────────────────────────


def collect_source_image_infos(source_cfg: dict, source_root: Path):
    """从det_shared导入，避免循环引用"""
    from .det_shared import collect_source_image_infos as _collect
    return _collect(source_cfg, source_root)


def build_problem_report(
    results: list[ImageAnalysisResult],
    summary: ModelAnalysisSummary,
    class_names: dict[int, str],
    analysis_info: dict[str, Any],
) -> str:
    """生成问题分析报告"""
    lines = [
        "# 模型辅助质检报告",
        "",
        "## 一、总体统计",
        "",
        f"- 分析图片数: {summary.total_images}",
        f"- GT框总数: {summary.total_gt_boxes}",
        f"- Pred框总数: {summary.total_pred_boxes}",
        f"- 有问题图片数: {summary.images_with_problems} ({summary.images_with_problems/max(summary.total_images,1)*100:.1f}%)",
        "",
        "## 二、问题类型统计",
        "",
        "| 问题类型 | 数量 | 说明 |",
        "|---------|------|------|",
        f"| missing (漏标) | {summary.missing_count} | GT有，但模型未检测到 |",
        f"| swapped (错标) | {summary.swapped_count} | GT↔Pred匹配，但类别不一致 |",
        f"| false_pos (误检) | {summary.false_pos_count} | Pred有，但与所有GT不匹配 |",
        f"| loc (框定位差) | {summary.loc_count} | GT↔Pred匹配，但IoU偏低 |",
        f"| low_conf (低置信度) | {summary.low_conf_count} | Pred score < 阈值 |",
        f"| outlier_class (劣质类别) | {summary.outlier_class_count} | AP极低或GT极少 |",
        "",
        "## 三、劣质类别",
        "",
    ]

    class_quality = analysis_info.get("class_quality", [])
    if class_quality:
        lines.append("| 类别 | AP | GT | 风险等级 |")
        lines.append("|------|-----|-----|---------|")
        for q in class_quality[:20]:  # 只显示前20个
            lines.append(f"| {q['name']} | {q['ap']:.3f} | {q['gt']} | {q['risk_level']} |")
    else:
        lines.append("无test_report数据，跳过类别质量分析。")

    lines.extend([
        "",
        "## 四、问题最多的类别 Top 10",
        "",
        "### 漏标 (missing)",
        "",
    ])

    if summary.class_missing:
        sorted_missing = sorted(summary.class_missing.items(), key=lambda x: -x[1])[:10]
        lines.append("| 类别 | 漏标次数 |")
        lines.append("|------|---------|")
        for cid, count in sorted_missing:
            name = class_names.get(cid, f"class_{cid}")
            lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "### 错标 (swapped)",
        "",
    ])

    if summary.class_swapped:
        sorted_swapped = sorted(summary.class_swapped.items(), key=lambda x: -x[1])[:10]
        lines.append("| 类别 | 错标次数 |")
        lines.append("|------|---------|")
        for cid, count in sorted_swapped:
            name = class_names.get(cid, f"class_{cid}")
            lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "### 误检 (false_pos)",
        "",
    ])

    if summary.class_false_pos:
        sorted_fp = sorted(summary.class_false_pos.items(), key=lambda x: -x[1])[:10]
        lines.append("| 类别 | 误检次数 |")
        lines.append("|------|---------|")
        for cid, count in sorted_fp:
            name = class_names.get(cid, f"class_{cid}")
            lines.append(f"| {name} | {count} |")

    return "\n".join(lines) + "\n"
