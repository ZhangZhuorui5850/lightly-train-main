from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import lightly_train

# ============================================================
# 配置区
# ============================================================
MODEL_PATH = "out/my_experiment_det/exported_models/exported_best.pt"
DATA_ROOT = Path("datasets/dataset_yolo")

# 若留空，脚本自动优先寻找：
#   1) DATA_ROOT/images/test 与 DATA_ROOT/labels/test
#   2) DATA_ROOT/test/images 与 DATA_ROOT/test/labels
TEST_IMAGES: Optional[Path] = None
TEST_LABELS: Optional[Path] = None

# 类别名优先从 classes.txt 读取；若没有，再尝试 data.yaml
CLASSES_FILE: Optional[Path] = None
DATA_YAML: Optional[Path] = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.5
DEBUG = False
PRINT_EVERY = 20
MAX_IMAGES: Optional[int] = None
ALLOW_MISSING_LABEL_TXT = True
SKIP_BAD_IMAGE = True
SKIP_PREDICT_ERROR = True
SAVE_REPORT_JSON = True
REPORT_PATH = Path("out/my_experiment_det/test_report.json")
# ============================================================


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def resolve_test_dirs(data_root: Path, test_images: Optional[Path], test_labels: Optional[Path]) -> Tuple[Path, Path]:
    if test_images is not None and test_labels is not None:
        return test_images, test_labels

    candidates = [
        (data_root / "images" / "test", data_root / "labels" / "test"),
        (data_root / "test" / "images", data_root / "test" / "labels"),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.exists() and lbl_dir.exists():
            return img_dir, lbl_dir

    # 给出最合理的默认值，便于报错信息清晰
    return candidates[0]


def resolve_classes_file(data_root: Path, classes_file: Optional[Path], data_yaml: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    if classes_file is None:
        p = data_root / "classes.txt"
        if p.exists():
            classes_file = p
    if data_yaml is None:
        p = data_root / "data.yaml"
        if p.exists():
            data_yaml = p
    return classes_file, data_yaml


def load_class_names(classes_file: Optional[Path], data_yaml: Optional[Path]) -> Dict[int, str]:
    if classes_file and classes_file.exists():
        lines = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {i: name for i, name in enumerate(lines)}

    if data_yaml and data_yaml.exists():
        # 先尝试 PyYAML
        try:
            import yaml  # type: ignore
            data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
            names = data.get("names", []) if isinstance(data, dict) else []
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
            if isinstance(names, list):
                return {i: str(v) for i, v in enumerate(names)}
        except Exception:
            pass

        # 再做一个轻量级兜底解析
        text = data_yaml.read_text(encoding="utf-8")
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("names:"):
                raw = s.split(":", 1)[1].strip()
                if raw.startswith("[") and raw.endswith("]"):
                    try:
                        arr = json.loads(raw.replace("'", '"'))
                        if isinstance(arr, list):
                            return {i: str(v) for i, v in enumerate(arr)}
                    except Exception:
                        pass
    return {}


# =========================
# 坐标 / IoU
# =========================
def xywhn_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> List[float]:
    xc *= img_w
    yc *= img_h
    w *= img_w
    h *= img_h
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def box_iou(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# =========================
# 图片尺寸读取
# =========================
def get_image_size(path: Path) -> Tuple[int, int]:
    """优先 PIL；失败后回退到 JPEG/PNG/WEBP 头解析。"""
    try:
        from PIL import Image  # type: ignore
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass

    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return _get_jpeg_size(path)
    if suffix == ".png":
        return _get_png_size(path)
    if suffix == ".webp":
        return _get_webp_size(path)
    raise ValueError(f"不支持的图片格式: {suffix} | {path}")


def _get_jpeg_size(path: Path) -> Tuple[int, int]:
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"不是 JPEG: {path}")
    i = 2
    while i < len(data):
        while i < len(data) and data[i] != 0xFF:
            i += 1
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            h = int.from_bytes(data[i + 3:i + 5], "big")
            w = int.from_bytes(data[i + 5:i + 7], "big")
            return w, h
        block_len = int.from_bytes(data[i:i + 2], "big")
        i += block_len
    raise ValueError(f"JPEG 尺寸解析失败: {path}")


def _get_png_size(path: Path) -> Tuple[int, int]:
    data = path.read_bytes()[:24]
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"不是 PNG: {path}")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def _get_webp_size(path: Path) -> Tuple[int, int]:
    data = path.read_bytes()[:64]
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError(f"不是 WEBP: {path}")
    chunk = data[12:16]
    if chunk == b"VP8X":
        w = 1 + int.from_bytes(data[24:27], "little")
        h = 1 + int.from_bytes(data[27:30], "little")
        return w, h
    if chunk == b"VP8L":
        b0, b1, b2, b3 = data[21], data[22], data[23], data[24]
        w = 1 + (((b1 & 0x3F) << 8) | b0)
        h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return w, h
    raise ValueError(f"WEBP 尺寸解析失败，建议安装 Pillow: {path}")


# =========================
# GT / 预测读取
# =========================
def load_gt_txt(label_path: Path, img_w: int, img_h: int) -> Tuple[List[dict], Dict[str, int]]:
    stats = {
        "missing_txt": 0,
        "empty_txt": 0,
        "bad_lines": 0,
    }
    gts: List[dict] = []

    if not label_path.exists():
        stats["missing_txt"] = 1
        return gts, stats

    text = label_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        stats["empty_txt"] = 1
        return gts, stats

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            stats["bad_lines"] += 1
            continue
        try:
            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            gts.append({
                "class_id": cls_id,
                "bbox": xywhn_to_xyxy(xc, yc, w, h, img_w, img_h),
            })
        except Exception:
            stats["bad_lines"] += 1
    return gts, stats


def to_list(x) -> List:
    if x is None:
        return []
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        try:
            x = x.numpy()
        except Exception:
            pass
    if hasattr(x, "tolist"):
        try:
            return x.tolist()
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def extract_predictions(results) -> Tuple[List[int], List[List[float]], List[float]]:
    """兼容 lightly_train.predict() 常见返回形式。"""
    if isinstance(results, dict):
        labels = to_list(results.get("labels"))
        bboxes = to_list(results.get("bboxes"))
        scores = to_list(results.get("scores"))
        return [int(x) for x in labels], [[float(v) for v in box] for box in bboxes], [float(x) for x in scores]

    # 兜底：对象属性
    labels = to_list(getattr(results, "labels", []))
    bboxes = to_list(getattr(results, "bboxes", []))
    scores = to_list(getattr(results, "scores", []))
    return [int(x) for x in labels], [[float(v) for v in box] for box in bboxes], [float(x) for x in scores]


# =========================
# 指标计算
# =========================
def compute_ap(pred_pairs: List[Tuple[float, int]], total_gt: int) -> float:
    if total_gt == 0:
        return 0.0

    pred_pairs = sorted(pred_pairs, key=lambda x: x[0], reverse=True)
    tp = 0
    fp = 0
    precisions: List[float] = []
    recalls: List[float] = []

    for score, is_tp in pred_pairs:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_gt)

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(len(mrec) - 1):
        if mrec[i + 1] != mrec[i]:
            ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
    return ap


def class_name(class_names: Dict[int, str], cid: int) -> str:
    return class_names.get(cid, f"class_{cid}")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# =========================
# 主流程
# =========================
def main() -> None:
    test_images, test_labels = resolve_test_dirs(DATA_ROOT, TEST_IMAGES, TEST_LABELS)
    classes_file, data_yaml = resolve_classes_file(DATA_ROOT, CLASSES_FILE, DATA_YAML)
    class_names = load_class_names(classes_file, data_yaml)

    print(f"MODEL_PATH   = {MODEL_PATH}", flush=True)
    print(f"DATA_ROOT    = {DATA_ROOT}", flush=True)
    print(f"TEST_IMAGES  = {test_images} | exists={test_images.exists()}", flush=True)
    print(f"TEST_LABELS  = {test_labels} | exists={test_labels.exists()}", flush=True)
    print(f"CLASSES_FILE = {classes_file} | exists={classes_file.exists() if classes_file else False}", flush=True)
    print(f"DATA_YAML    = {data_yaml} | exists={data_yaml.exists() if data_yaml else False}", flush=True)

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"模型不存在: {MODEL_PATH}")
    if not test_images.exists():
        raise FileNotFoundError(f"测试图片目录不存在: {test_images}")
    if not test_labels.exists() and not ALLOW_MISSING_LABEL_TXT:
        raise FileNotFoundError(f"测试标签目录不存在: {test_labels}")

    image_paths = sorted(p for p in test_images.iterdir() if is_image_file(p))
    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    print(f"num_test_images = {len(image_paths)}", flush=True)
    if not image_paths:
        raise RuntimeError(f"测试目录里没有图片: {test_images}")

    print("loading model...", flush=True)
    model = lightly_train.load_model(MODEL_PATH)
    print("model loaded", flush=True)

    class_data: Dict[int, Dict[str, List]] = {}
    total_pred = 0
    total_gt = 0
    total_tp = 0
    infer_time_sum = 0.0

    dataset_stats = {
        "missing_txt_images": 0,
        "empty_txt_images": 0,
        "bad_label_lines": 0,
        "bad_images": 0,
        "predict_errors": 0,
        "processed_images": 0,
        "skipped_images": 0,
        "negative_images": 0,
    }
    bad_cases: List[dict] = []

    for idx, img_path in enumerate(image_paths, 1):
        if idx == 1 or idx % PRINT_EVERY == 0 or DEBUG:
            print(f"[{idx}/{len(image_paths)}] {img_path.name}", flush=True)

        try:
            img_w, img_h = get_image_size(img_path)
        except Exception as e:
            dataset_stats["bad_images"] += 1
            dataset_stats["skipped_images"] += 1
            bad_cases.append({"image": img_path.name, "stage": "get_image_size", "error": str(e)})
            print(f"  [WARN] 读取图片尺寸失败，跳过: {img_path.name} | {e}", flush=True)
            if SKIP_BAD_IMAGE:
                continue
            raise

        gt_path = test_labels / f"{img_path.stem}.txt"
        gts, gt_stats = load_gt_txt(gt_path, img_w, img_h)
        dataset_stats["missing_txt_images"] += gt_stats["missing_txt"]
        dataset_stats["empty_txt_images"] += gt_stats["empty_txt"]
        dataset_stats["bad_label_lines"] += gt_stats["bad_lines"]

        if gt_stats["missing_txt"] and not ALLOW_MISSING_LABEL_TXT:
            raise FileNotFoundError(f"缺少标签 txt: {gt_path}")

        if not gts:
            dataset_stats["negative_images"] += 1

        for gt in gts:
            cid = gt["class_id"]
            class_data.setdefault(cid, {"gt": 0, "pairs": []})
            class_data[cid]["gt"] += 1
            total_gt += 1

        try:
            t0 = time.perf_counter()
            results = model.predict(str(img_path), threshold=CONF_THRESHOLD)
            infer_time_sum += time.perf_counter() - t0
        except Exception as e:
            dataset_stats["predict_errors"] += 1
            dataset_stats["skipped_images"] += 1
            bad_cases.append({"image": img_path.name, "stage": "predict", "error": str(e)})
            print(f"  [WARN] 推理失败，跳过: {img_path.name} | {e}", flush=True)
            if SKIP_PREDICT_ERROR:
                continue
            raise

        if DEBUG:
            print(results, flush=True)

        labels, bboxes, scores = extract_predictions(results)
        preds = [
            {"class_id": int(c), "bbox": [float(x) for x in b], "score": float(s)}
            for c, b, s in zip(labels, bboxes, scores)
            if float(s) >= CONF_THRESHOLD
        ]
        preds.sort(key=lambda x: x["score"], reverse=True)

        gt_by_class: Dict[int, List[Tuple[int, dict]]] = {}
        for i, gt in enumerate(gts):
            gt_by_class.setdefault(gt["class_id"], []).append((i, gt))

        matched_gt: set = set()
        image_tp = 0
        for pred in preds:
            total_pred += 1
            cid = pred["class_id"]
            class_data.setdefault(cid, {"gt": 0, "pairs": []})

            best_iou = 0.0
            best_idx = -1
            for gt_i, gt in gt_by_class.get(cid, []):
                if gt_i in matched_gt:
                    continue
                iou = box_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gt_i

            is_tp = best_idx >= 0 and best_iou >= IOU_THRESHOLD
            if is_tp:
                matched_gt.add(best_idx)
                total_tp += 1
                image_tp += 1
            class_data[cid]["pairs"].append((pred["score"], 1 if is_tp else 0))

        dataset_stats["processed_images"] += 1
        if DEBUG:
            print(f"  gt={len(gts)} pred={len(preds)} tp={image_tp}", flush=True)

    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    ap_per_class: Dict[int, float] = {}
    for cid, info in sorted(class_data.items()):
        ap_per_class[cid] = compute_ap(info["pairs"], info["gt"])

    map50 = sum(ap_per_class.values()) / len(ap_per_class) if ap_per_class else 0.0
    avg_infer_ms = (infer_time_sum / max(dataset_stats["processed_images"], 1)) * 1000.0

    print("\n========== TEST METRICS ==========")
    print(f"num_images         : {len(image_paths)}")
    print(f"processed_images   : {dataset_stats['processed_images']}")
    print(f"skipped_images     : {dataset_stats['skipped_images']}")
    print(f"num_gt_boxes       : {total_gt}")
    print(f"num_preds          : {total_pred}")
    print(f"precision@0.5      : {precision:.4f}")
    print(f"recall@0.5         : {recall:.4f}")
    print(f"mAP@0.5            : {map50:.4f}")
    print(f"avg_infer_time_ms  : {avg_infer_ms:.2f}")

    print("\n========== DATASET CHECK ==========")
    print(f"missing_txt_images : {dataset_stats['missing_txt_images']}")
    print(f"empty_txt_images   : {dataset_stats['empty_txt_images']}")
    print(f"bad_label_lines    : {dataset_stats['bad_label_lines']}")
    print(f"bad_images         : {dataset_stats['bad_images']}")
    print(f"predict_errors     : {dataset_stats['predict_errors']}")
    print(f"negative_images    : {dataset_stats['negative_images']}")

    if ap_per_class:
        print("\n========== PER-CLASS AP ==========")
        for cid, ap in sorted(ap_per_class.items()):
            n_gt = class_data[cid]["gt"]
            n_pred = len(class_data[cid]["pairs"])
            print(f"{cid:3d} | {class_name(class_names, cid):<20} | AP={ap:.4f} | gt={n_gt:<5d} | pred={n_pred:<5d}")

    if bad_cases:
        print("\n========== BAD CASES (TOP 20) ==========")
        for item in bad_cases[:20]:
            print(f"[{item['stage']}] {item['image']} | {item['error']}")
        if len(bad_cases) > 20:
            print(f"... total {len(bad_cases)} bad cases")

    if SAVE_REPORT_JSON:
        ensure_parent(REPORT_PATH)
        report = {
            "config": {
                "model_path": MODEL_PATH,
                "data_root": str(DATA_ROOT),
                "test_images": str(test_images),
                "test_labels": str(test_labels),
                "conf_threshold": CONF_THRESHOLD,
                "iou_threshold": IOU_THRESHOLD,
            },
            "summary": {
                "num_images": len(image_paths),
                "processed_images": dataset_stats["processed_images"],
                "skipped_images": dataset_stats["skipped_images"],
                "num_gt_boxes": total_gt,
                "num_preds": total_pred,
                "precision_05": precision,
                "recall_05": recall,
                "map_05": map50,
                "avg_infer_time_ms": avg_infer_ms,
            },
            "dataset_check": dataset_stats,
            "class_names": {str(k): v for k, v in class_names.items()},
            "per_class_ap": {
                str(cid): {
                    "name": class_name(class_names, cid),
                    "ap": ap,
                    "gt": class_data[cid]["gt"],
                    "pred": len(class_data[cid]["pairs"]),
                }
                for cid, ap in sorted(ap_per_class.items())
            },
            "bad_cases": bad_cases,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport_json        : {REPORT_PATH}")


if __name__ == "__main__":
    main()
