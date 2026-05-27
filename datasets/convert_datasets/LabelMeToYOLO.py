from __future__ import annotations

"""
LabelMe -> YOLO / CLS 转换脚本（同时输出检测 + 分类 + 分割三份数据）
====================================================================
目录约定（脚本放在 datasets/ 目录下运行）：

  datasets/
  ├── LabelMeToYOLO_v4.py
  ├── moxingxunlian/              ← 原始 LabelMe 数据（只读）
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── dataset_det/                ← 检测格式输出（TARGET_DET）
  │   ├── images/train|val|test/  ← 图片（两份共用同一份，硬链接）
  │   ├── labels/train|val|test/  ← class_id cx cy w h
  │   ├── classes.txt
  │   └── data.yaml
  ├── dataset_cls/                ← 分类格式输出（TARGET_CLS）
  │   ├── train|val|test/class_name/*.jpg
  │   ├── classes.txt
  │   └── data.yaml
  └── dataset_seg/                ← 分割格式输出（TARGET_SEG）
      ├── images/train|val|test/  ← 同一张图片（硬链接或复制）
      ├── labels/train|val|test/  ← class_id x1 y1 x2 y2 ...（归一化）
      ├── classes.txt
      └── data.yaml

shape_type 处理规则：
  检测（det）：
    rectangle → class_id cx cy w h
  分割（seg）：
    polygon   → 所有点归一化   → class_id x1 y1 x2 y2 ...
  分类（cls）：
    rectangle → 按矩形框裁剪目标图块，保存到 split/class_name/

当前版本变更：
  - 同时输出 det / cls / seg 三份数据
  - rectangle 只进入 det / cls
  - polygon 只进入 seg
"""

import base64
import argparse
import ast
import io
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================
# 配置区  ── 按需修改
# ============================================================
SOURCE_ROOT = Path("moxingxunlian")   # 原始数据根目录（只读）
OUTPUT_ROOT = Path("converted_dataset")
TARGET_DET  = OUTPUT_ROOT / "dataset_det"     # 检测格式输出目录
TARGET_CLS  = OUTPUT_ROOT / "dataset_cls"     # 分类格式输出目录
TARGET_SEG  = OUTPUT_ROOT / "dataset_seg"     # 分割格式输出目录

SPLITS = ["train", "val", "test"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 固定类别映射（留空则自动从 JSON 收集）
FIXED_CLASS_MAP: Dict[str, int] = {}

# 是否递归扫描子目录
RECURSIVE_SCAN = True

# 输出目录已有旧数据时的策略: "ask" / "clean" / "keep"
EXISTING_OUTPUT_POLICY = "ask"

# 转换完成后是否做图-txt 配对完整性校验
INTEGRITY_CHECK = True

# True  → 无 JSON 的图片仍复制并生成空 txt（作为负样本）
# False → 无 JSON 的图片直接跳过（推荐）
CREATE_EMPTY_TXT_FOR_IMAGE_WITHOUT_JSON = False

# shapes 为空的 JSON 是否生成空 txt
ALLOW_EMPTY_SHAPES_JSON = True
PROGRESS_EVERY = 100
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 LabelMe 或已存在的 YOLO 数据整理为统一 det/cls/seg 输出目录",
    )
    parser.add_argument(
        "--source-root",
        default=str(SOURCE_ROOT),
        help="输入数据根目录，内部默认包含 train/val/test",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="统一输出根目录，内部会生成 dataset_det、dataset_cls 和 dataset_seg",
    )
    parser.add_argument(
        "--task",
        choices=["det", "cls", "seg", "all"],
        default="all",
        help="输出任务类型：det/cls/seg/all",
    )
    parser.add_argument(
        "--source-format",
        choices=["auto", "labelme", "yolo"],
        default="auto",
        help="输入格式：auto 自动判断，labelme 为 JSON，yolo 为 TXT",
    )
    parser.add_argument(
        "--class-ref",
        default=None,
        help="参考类别文件（data.yaml / classes.txt），输出将使用其中的完整类别列表和 ID 映射",
    )
    return parser.parse_args()


def normalize_selected_tasks(task: str) -> set[str]:
    task_key = task.strip().lower()
    if task_key == "all":
        return {"det", "cls", "seg"}
    if task_key in {"det", "cls", "seg"}:
        return {task_key}
    raise ValueError(f"不支持的 task: {task}")


# ============================================================
# 工具函数
# ============================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def log_progress(prefix: str, current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % PROGRESS_EVERY == 0:
        print(f"  [INFO] {prefix}: {current}/{total}")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def handle_existing_output(target_root: Path) -> None:
    if not target_root.exists():
        return
    existing_entries = list(target_root.iterdir())
    if not existing_entries:
        return

    policy = EXISTING_OUTPUT_POLICY
    if policy == "ask":
        ans = input(
            f"\n[!] 输出目录已存在旧数据: {target_root.resolve()}\n"
            f"    输入 y 清空后重新转换，输入 n 保留旧文件直接覆盖 [y/n]: "
        ).strip().lower()
        policy = "clean" if ans == "y" else "keep"

    if policy == "clean":
        for p in existing_entries:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        print(f"[INFO] 已清空旧输出目录: {target_root}")
    else:
        print(f"[INFO] 保留旧文件，同名文件将被覆盖: {target_root}")


def _empty_stats() -> dict:
    return {
        "images_copied": 0,
        "images_skipped_no_json": 0,
        "json_found": 0,
        "json_converted": 0,
        "cls_crops": 0,
        "cls_images_skipped": 0,
        "empty_labels_det": 0,
        "empty_labels_seg": 0,
        "json_without_image": 0,
        "skipped_json": 0,
        "skipped_shapes_det": 0,
        "skipped_shapes_seg": 0,
        "_skipped_detail": [],
    }


def detect_source_format(source_root: Path, splits: List[str]) -> str:
    has_json = False
    has_txt = False
    for split in splits:
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        for path in split_dir.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".json":
                has_json = True
            elif suffix == ".txt":
                has_txt = True
            if has_json and has_txt:
                return "mixed"
    if has_json:
        return "labelme"
    if has_txt:
        return "yolo"
    return "unknown"


# ============================================================
# 图片尺寸读取
# ============================================================
def get_image_size(data: dict, json_path: Path) -> Tuple[int, int]:
    w, h = data.get("imageWidth"), data.get("imageHeight")
    if w and h:
        return int(w), int(h)

    b64 = data.get("imageData")
    if b64:
        size = _read_size_from_base64(b64)
        if size:
            return size

    img = find_image_for_json(data, json_path)
    if img:
        size = _read_image_size(img)
        if size:
            return size

    raise RuntimeError(
        f"无法获取图片尺寸：缺少 imageWidth/imageHeight，"
        f"imageData 解码失败，找不到图片文件 | {json_path}"
    )


def _read_size_from_base64(b64_str: str) -> Optional[Tuple[int, int]]:
    try:
        return _read_size_from_bytes(base64.b64decode(b64_str))
    except Exception:
        return None


def _read_image_size(img_path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(img_path) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    try:
        return _read_size_from_bytes(img_path.read_bytes())
    except Exception:
        return None


def _read_size_from_bytes(data: bytes) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image  # type: ignore
        with Image.open(io.BytesIO(data)) as im:
            return int(im.width), int(im.height)
    except Exception:
        pass
    # PNG
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')
    # JPEG
    if data[:2] == b'\xff\xd8':
        i = 2
        while i < len(data):
            while i < len(data) and data[i] != 0xFF:
                i += 1
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]; i += 1
            if marker in {0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF}:
                h = int.from_bytes(data[i+3:i+5], 'big')
                w = int.from_bytes(data[i+5:i+7], 'big')
                return w, h
            block_len = int.from_bytes(data[i:i+2], 'big')
            i += block_len
    # WebP
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        chunk = data[12:16]
        if chunk == b'VP8X':
            return (1 + int.from_bytes(data[24:27], 'little'),
                    1 + int.from_bytes(data[27:30], 'little'))
        if chunk == b'VP8L':
            b0,b1,b2,b3 = data[21],data[22],data[23],data[24]
            w = 1 + (((b1 & 0x3F) << 8) | b0)
            h = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
            return w, h
    return None


# ============================================================
# shape_type 推断
# ============================================================
def _infer_shape_type(shape: dict) -> str:
    st = str(shape.get("shape_type", "")).strip().lower()
    if st:
        return st
    pts = shape.get("points", [])
    if len(pts) in {2, 4}:
        return "rectangle"
    if len(pts) >= 3:
        return "polygon"
    return ""


# ============================================================
# 检测格式转换：bbox → class_id cx cy w h
# ============================================================
def _clip_bbox(x_min, y_min, x_max, y_max, img_w, img_h):
    x_min = max(0.0, min(float(x_min), img_w))
    y_min = max(0.0, min(float(y_min), img_h))
    x_max = max(0.0, min(float(x_max), img_w))
    y_max = max(0.0, min(float(y_max), img_h))
    return x_min, y_min, x_max, y_max


def shape_to_bbox(shape: dict, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    st = _infer_shape_type(shape)
    pts = shape.get("points", [])

    if st != "rectangle":
        raise ValueError(f"不支持的 shape_type: '{st}'")

    if len(pts) == 2:
        (x1, y1), (x2, y2) = pts
        return _clip_bbox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), img_w, img_h)
    if len(pts) == 4:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return _clip_bbox(min(xs), min(ys), max(xs), max(ys), img_w, img_h)
    raise ValueError(f"rectangle 有 {len(pts)} 个点，不支持")


def bbox_to_yolo(x_min, y_min, x_max, y_max,
                 img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x_min, y_min, x_max, y_max = _clip_bbox(x_min, y_min, x_max, y_max, img_w, img_h)
    bw, bh = x_max - x_min, y_max - y_min
    if bw <= 0 or bh <= 0:
        raise ValueError(f"bbox 宽/高 <= 0: bw={bw}, bh={bh}")
    return (
        (x_min + x_max) / 2.0 / img_w,
        (y_min + y_max) / 2.0 / img_h,
        bw / img_w,
        bh / img_h,
    )


def shape_to_det(shape: dict, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    返回 (cx, cy, w, h) 归一化。
    rectangle → bbox。
    """
    x_min, y_min, x_max, y_max = shape_to_bbox(shape, img_w, img_h)
    return bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)


# ============================================================
# 分割格式转换：points → class_id x1 y1 x2 y2 ...
# ============================================================
def shape_to_seg(shape: dict, img_w: int, img_h: int) -> List[float]:
    """
    返回归一化点列表 [x1, y1, x2, y2, ...]。
    polygon   → 原始点顺序归一化，clip 到 [0,1]。
    最少 3 个点（6 个坐标值）。
    """
    st  = _infer_shape_type(shape)
    pts = shape.get("points", [])

    if st == "polygon":
        if len(pts) < 3:
            raise ValueError(f"polygon 点数太少: {len(pts)}")
        coords = []
        for x, y in pts:
            xn = max(0.0, min(float(x), img_w)) / img_w
            yn = max(0.0, min(float(y), img_h)) / img_h
            coords.extend([xn, yn])
        return coords

    raise ValueError(f"不支持的 shape_type: '{st}'")


# ============================================================
# 文件定位 / stem 映射
# ============================================================
def unique_stem(path: Path, src_dir: Path) -> str:
    rel   = path.relative_to(src_dir)
    parts = list(rel.parent.parts) + [rel.stem]
    return "__".join(parts) if len(parts) > 1 else rel.stem


def find_image_for_json(data: dict, json_path: Path) -> Optional[Path]:
    stem, parent = json_path.stem, json_path.parent
    raw = data.get("imagePath", "")
    if raw:
        raw_p = Path(raw)
        for c in [raw_p, parent / raw_p, parent / raw_p.name]:
            if c.exists() and is_image_file(c):
                return c
    for ext in IMAGE_EXTS:
        c = parent / f"{stem}{ext}"
        if c.exists() and is_image_file(c):
            return c
    return None


def build_image_index(src_dir: Path):
    pattern     = "**/*" if RECURSIVE_SCAN else "*"
    image_paths = [p for p in sorted(src_dir.glob(pattern)) if is_image_file(p)]
    image_to_stem:  Dict[Path, str] = {}
    relkey_to_stem: Dict[str, str]  = {}
    for img in image_paths:
        stem = unique_stem(img, src_dir)
        image_to_stem[img.resolve()] = stem
        relkey_to_stem[
            str(img.relative_to(src_dir).with_suffix("")).replace("\\", "/")
        ] = stem
    return image_paths, image_to_stem, relkey_to_stem


def build_annotated_image_set(src_dir: Path) -> set:
    annotated: set = set()
    pattern = "**/*.json" if RECURSIVE_SCAN else "*.json"
    for jf in src_dir.glob(pattern):
        try:
            data = load_json(jf)
        except Exception:
            continue
        img = find_image_for_json(data, jf)
        if img:
            annotated.add(img.resolve())
    return annotated


# ============================================================
# 类别收集 / 配置文件写入
# ============================================================
def normalize_label(label: str) -> str:
    return str(label).strip()


def load_class_ref(ref_path: Path) -> Dict[str, int]:
    """从参考文件（data.yaml / dataset.yaml / classes.txt）加载完整类别映射。"""
    text = ref_path.read_text(encoding="utf-8")
    names = None
    # classes.txt 格式：每行一个类别名
    if ref_path.suffix == ".txt" or "names:" not in text:
        names = [line.strip() for line in text.splitlines() if line.strip()]
        if names and any(":" in n for n in names):
            # 可能是 YAML，回退到 YAML 解析
            names = None
    # YAML 格式
    if not names:
        names = parse_names_from_yaml_text(text)
    if not names:
        print(f"[ERROR] 无法从参考文件解析类别: {ref_path}")
        sys.exit(1)
    print(f"[INFO] 从参考文件加载 {len(names)} 个类别: {ref_path}")
    return {str(name): idx for idx, name in enumerate(names)}


def collect_classes(source_root: Path, splits: List[str]) -> Dict[str, int]:
    if FIXED_CLASS_MAP:
        print("[INFO] 使用固定 FIXED_CLASS_MAP")
        return FIXED_CLASS_MAP.copy()

    # 优先从源目录的 data.yaml / classes.txt 读取类别名
    metadata_names = load_yolo_class_names_from_metadata(source_root)
    if metadata_names:
        print(f"[INFO] 从 metadata 文件读取 {len(metadata_names)} 个类别")
        return {str(name): idx for idx, name in enumerate(metadata_names)}

    class_names: set = set()
    for split in splits:
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        pattern = "**/*.json" if RECURSIVE_SCAN else "*.json"
        for jf in split_dir.glob(pattern):
            try:
                data = load_json(jf)
                for shape in data.get("shapes", []):
                    label = normalize_label(shape.get("label", ""))
                    if label:
                        class_names.add(label)
            except Exception as e:
                print(f"  [WARN] 类别收集跳过: {jf} | {e}")

    def sort_key(x: str):
        try:    return (0, int(x))
        except: return (1, x)

    return {name: idx for idx, name in enumerate(sorted(class_names, key=sort_key))}


def write_classes_file(class_map: Dict[str, int], out_path: Path) -> None:
    inverse = {v: k for k, v in class_map.items()}
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(inverse)):
            f.write(f"{inverse[i]}\n")
    print(f"[INFO] classes.txt -> {out_path}")


def write_data_yaml(class_map: Dict[str, int], target_root: Path, task: str) -> None:
    """task: 'detect' 或 'segment'"""
    inverse = {v: k for k, v in class_map.items()}
    lines = [
        f"path: {target_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"task: {task}",
        f"nc: {len(inverse)}",
        "names:",
    ]
    for i in range(len(inverse)):
        lines.append(f'  {i}: "{inverse[i]}"')
    out = target_root / "data.yaml"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] data.yaml -> {out}")


def write_cls_data_yaml(class_map: Dict[str, int], target_root: Path) -> None:
    inverse = {v: k for k, v in class_map.items()}
    lines = [
        f"path: {target_root.resolve()}",
        "train: train",
        "val: val",
        "test: test",
        "",
        "task: classify",
        f"nc: {len(inverse)}",
        "names:",
    ]
    for i in range(len(inverse)):
        lines.append(f'  {i}: "{inverse[i]}"')
    out = target_root / "data.yaml"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] data.yaml -> {out}")


def _format_float(v: float) -> str:
    return f"{v:.6f}"


def _clip01(v: float) -> float:
    return max(0.0, min(float(v), 1.0))


def load_yolo_class_names_from_metadata(source_root: Path) -> Optional[List[str]]:
    candidates = [
        source_root / "data.yaml",
        source_root / "dataset.yaml",
        source_root / "classes.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.name == "classes.txt":
            names = [
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if names:
                return names
            continue

        text = path.read_text(encoding="utf-8")
        names = parse_names_from_yaml_text(text)
        if names:
            return names
    return None


def parse_names_from_yaml_text(text: str) -> Optional[List[str]]:
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue
        after = stripped[len("names:"):].strip()
        if not after:
            break
        # Try quoted format first: names: ["a", "b"] or names: {0: "a", 1: "b"}
        try:
            parsed = ast.literal_eval(after)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            if isinstance(parsed, dict):
                return [str(parsed[k]) for k in sorted(parsed)]
        except Exception:
            pass
        # Try unquoted list: names: [a, b, c]
        if after.startswith("[") and after.endswith("]"):
            items = [s.strip().strip("\"'") for s in after[1:-1].split(",") if s.strip()]
            if items:
                return items
        # Try unquoted dict: names: {0: a, 1: b}
        if after.startswith("{") and after.endswith("}"):
            try:
                pairs = {}
                for pair in after[1:-1].split(","):
                    pair = pair.strip()
                    if ":" in pair:
                        k, v = pair.split(":", 1)
                        pairs[int(k.strip())] = v.strip().strip("\"'")
                if pairs:
                    return [pairs[i] for i in sorted(pairs)]
            except Exception:
                pass
        break

    # Block-style dict parsing (each entry on its own line)
    collecting = False
    items: Dict[int, str] = {}
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("names:"):
            collecting = True
            continue
        if not collecting:
            continue
        if not stripped:
            continue
        match = re.match(r"^(\d+)\s*:\s*[\"']?(.*?)[\"']?\s*$", stripped)
        if match:
            items[int(match.group(1))] = match.group(2)
            continue
        if items:
            break
    if items:
        return [items[i] for i in sorted(items)]
    return None


def collect_classes_from_yolo(source_root: Path, splits: List[str]) -> Dict[str, int]:
    if FIXED_CLASS_MAP:
        print("[INFO] 使用固定 FIXED_CLASS_MAP")
        return FIXED_CLASS_MAP.copy()

    metadata_names = load_yolo_class_names_from_metadata(source_root)
    if metadata_names:
        return {str(name): idx for idx, name in enumerate(metadata_names)}

    class_ids: set[int] = set()
    for split in splits:
        split_dir = source_root / split
        if not split_dir.exists():
            continue
        for txt in sorted(split_dir.rglob("*.txt")):
            for raw_line in txt.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                first = line.split()[0]
                try:
                    class_ids.add(int(float(first)))
                except ValueError:
                    continue
    return {str(cid): cid for cid in sorted(class_ids)}


def parse_yolo_label_line(line: str) -> Tuple[int, str, List[float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"字段数不足: {len(parts)}")
    try:
        cid = int(float(parts[0]))
        values = [float(x) for x in parts[1:]]
    except ValueError as e:
        raise ValueError(f"存在非数字字段: {e}") from e

    if len(values) == 4:
        return cid, "detect", values
    if len(values) >= 6 and len(values) % 2 == 0:
        return cid, "segment", values
    raise ValueError(f"不支持的 YOLO 行格式: {line}")


def save_crop_image(
    image_path: Path,
    bbox: Tuple[float, float, float, float],
    output_path: Path,
) -> None:
    from PIL import Image  # type: ignore

    x_min, y_min, x_max, y_max = bbox
    left = int(round(x_min))
    top = int(round(y_min))
    right = int(round(x_max))
    bottom = int(round(y_max))
    if right <= left or bottom <= top:
        raise ValueError("裁剪框宽/高 <= 0")

    with Image.open(image_path) as im:
        crop = im.crop((left, top, right, bottom))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path)


def convert_split_from_yolo(
    split: str,
    source_root: Path,
    target_det: Path,
    target_cls: Path,
    target_seg: Path,
    class_map: Dict[str, int],
    selected_tasks: set[str],
) -> dict:
    src_dir = source_root / split

    dst_img_det = target_det / "images" / split if "det" in selected_tasks else None
    dst_lbl_det = target_det / "labels" / split if "det" in selected_tasks else None
    dst_img_cls = target_cls / split if "cls" in selected_tasks else None
    dst_img_seg = target_seg / "images" / split if "seg" in selected_tasks else None
    dst_lbl_seg = target_seg / "labels" / split if "seg" in selected_tasks else None

    for d in [dst_img_det, dst_lbl_det, dst_img_cls, dst_img_seg, dst_lbl_seg]:
        if d is not None:
            safe_mkdir(d)

    if not src_dir.exists():
        print(f"  [WARN] split 目录不存在，跳过: {src_dir}")
        return _empty_stats()

    stats = _empty_stats()
    skipped_detail: List[dict] = []

    image_paths, image_to_stem, relkey_to_stem = build_image_index(src_dir)
    relkey_to_image = {
        str(img.relative_to(src_dir).with_suffix("")).replace("\\", "/"): img
        for img in image_paths
    }
    txt_files = sorted(src_dir.rglob("*.txt"))
    stats["json_found"] = len(txt_files)

    txt_index: Dict[str, Path] = {}
    for txt in txt_files:
        relkey = str(txt.relative_to(src_dir).with_suffix("")).replace("\\", "/")
        txt_index[relkey] = txt

    annotated_keys = set(txt_index.keys())
    annotated_set = {
        img.resolve()
        for img in image_paths
        if str(img.relative_to(src_dir).with_suffix("")).replace("\\", "/") in annotated_keys
    }

    no_label_images = [
        str(img.relative_to(src_dir))
        for img in image_paths
        if img.resolve() not in annotated_set
    ]
    if no_label_images:
        print(f"  [WARN] {len(no_label_images)} 张图无 TXT，跳过：")
        for name in no_label_images[:20]:
            print(f"    - {name}")
        if len(no_label_images) > 20:
            print(f"    ... 共 {len(no_label_images)} 张，仅显示前 20 张")
    stats["images_skipped_no_json"] = len(no_label_images)

    annotated_image_paths = [img for img in image_paths if img.resolve() in annotated_set]
    print(
        f"  [INFO] 有标注图片 {len(annotated_image_paths)} 张，"
        f"标注文件 {len(txt_files)} 个，开始复制与转换 ..."
    )

    total_annotated_images = len(annotated_image_paths)
    for img_index, img in enumerate(annotated_image_paths, start=1):
        relkey = str(img.relative_to(src_dir).with_suffix("")).replace("\\", "/")
        stem = image_to_stem[img.resolve()]
        dst_name = stem + img.suffix.lower()
        copied = False
        if dst_img_det is not None:
            shutil.copy2(img, dst_img_det / dst_name)
            copied = True
        if dst_img_seg is not None:
            shutil.copy2(img, dst_img_seg / dst_name)
            copied = True
        if copied:
            stats["images_copied"] += 1
        log_progress("复制图片", img_index, total_annotated_images)

    total_txt_files = len(txt_files)
    for txt_index, txt_file in enumerate(txt_files, start=1):
        relkey = str(txt_file.relative_to(src_dir).with_suffix("")).replace("\\", "/")
        out_stem = relkey_to_stem.get(relkey, unique_stem(txt_file, src_dir))
        matched_image = relkey_to_image.get(relkey)
        if "cls" in selected_tasks and matched_image is not None:
            size = _read_image_size(matched_image)
            if size is None:
                skipped_detail.append({"file": txt_file.name, "shape": None, "reason": "cls 缺少图片尺寸"})
                stats["cls_images_skipped"] += 1
                matched_image = None
                img_w = img_h = None
            else:
                img_w, img_h = size
        else:
            img_w = img_h = None

        det_lines: List[str] = []
        seg_lines: List[str] = []
        cls_crop_index = 0

        try:
            raw_lines = txt_file.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            reason = f"{SKIP_JSON_ERROR}: {e}"
            print(f"  [SKIP] {txt_file.name} | {reason}")
            skipped_detail.append({"file": txt_file.name, "shape": None, "reason": reason})
            stats["skipped_json"] += 1
            continue

        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                cid, label_type, values = parse_yolo_label_line(line)
            except Exception as e:
                reason = f"{SKIP_CONVERT_ERROR}(yolo): {e}"
                skipped_detail.append({"file": txt_file.name, "shape": "yolo", "reason": reason})
                if "det" in selected_tasks:
                    stats["skipped_shapes_det"] += 1
                if "seg" in selected_tasks:
                    stats["skipped_shapes_seg"] += 1
                continue

            if label_type == "detect":
                if "det" in selected_tasks:
                    det_lines.append(
                        f"{cid} {' '.join(_format_float(_clip01(v)) for v in values)}"
                    )
                if "cls" in selected_tasks and matched_image is not None and img_w is not None and img_h is not None:
                    try:
                        class_name = class_map[cid]
                        x_min = (_clip01(values[0] - values[2] / 2.0)) * img_w
                        y_min = (_clip01(values[1] - values[3] / 2.0)) * img_h
                        x_max = (_clip01(values[0] + values[2] / 2.0)) * img_w
                        y_max = (_clip01(values[1] + values[3] / 2.0)) * img_h
                        crop_name = f"{out_stem}__{cls_crop_index:04d}{matched_image.suffix.lower()}"
                        save_crop_image(
                            matched_image,
                            (x_min, y_min, x_max, y_max),
                            dst_img_cls / class_name / crop_name,  # type: ignore[operator]
                        )
                        cls_crop_index += 1
                        stats["cls_crops"] += 1
                    except Exception as e:
                        reason = f"{SKIP_CONVERT_ERROR}(cls): {e}"
                        skipped_detail.append({"file": txt_file.name, "shape": "detect", "reason": reason})
                        stats["cls_images_skipped"] += 1
                elif "cls" in selected_tasks:
                    skipped_detail.append({"file": txt_file.name, "shape": "detect", "reason": "cls 缺少源图片"})
                    stats["cls_images_skipped"] += 1
            elif "seg" in selected_tasks:
                seg_coords = [_clip01(v) for v in values]
                seg_lines.append(
                    f"{cid} {' '.join(_format_float(v) for v in seg_coords)}"
                )

        if dst_lbl_det is not None:
            det_txt = dst_lbl_det / f"{out_stem}.txt"
            det_txt.write_text("\n".join(det_lines), encoding="utf-8")
            if not det_lines:
                stats["empty_labels_det"] += 1

        if dst_lbl_seg is not None:
            seg_txt = dst_lbl_seg / f"{out_stem}.txt"
            seg_txt.write_text("\n".join(seg_lines), encoding="utf-8")
            if not seg_lines:
                stats["empty_labels_seg"] += 1

        stats["json_converted"] += 1
        log_progress("转换标注", txt_index, total_txt_files)

    if CREATE_EMPTY_TXT_FOR_IMAGE_WITHOUT_JSON:
        for _, out_stem in image_to_stem.items():
            for lbl_dir in [dst_lbl_det, dst_lbl_seg]:
                if lbl_dir is None:
                    continue
                txt = lbl_dir / f"{out_stem}.txt"
                if not txt.exists():
                    txt.write_text("", encoding="utf-8")

    stats["_skipped_detail"] = skipped_detail
    return stats


# ============================================================
# 单 split 转换（同时写 det + seg 两份 labels，图片只复制一次）
# ============================================================
SKIP_EMPTY_LABEL   = "空 label"
SKIP_UNKNOWN_LABEL = "label 不在 class_map"
SKIP_SHAPE_TYPE    = "不支持的 shape_type"
SKIP_CONVERT_ERROR = "坐标转换出错"
SKIP_JSON_ERROR    = "json 解析 / 尺寸错误"


def convert_split(
    split: str,
    source_root: Path,
    target_det: Path,
    target_cls: Path,
    target_seg: Path,
    class_map: Dict[str, int],
    selected_tasks: set[str],
) -> dict:
    src_dir = source_root / split

    dst_img_det = target_det / "images" / split if "det" in selected_tasks else None
    dst_lbl_det = target_det / "labels" / split if "det" in selected_tasks else None
    dst_img_cls = target_cls / split if "cls" in selected_tasks else None
    dst_img_seg = target_seg / "images" / split if "seg" in selected_tasks else None
    dst_lbl_seg = target_seg / "labels" / split if "seg" in selected_tasks else None

    for d in [dst_img_det, dst_lbl_det, dst_img_cls, dst_img_seg, dst_lbl_seg]:
        if d is not None:
            safe_mkdir(d)

    if not src_dir.exists():
        print(f"  [WARN] split 目录不存在，跳过: {src_dir}")
        return _empty_stats()

    stats = _empty_stats()
    skipped_detail: List[dict] = []

    # ── 建立索引 ──────────────────────────────────────────────
    image_paths, image_to_stem, relkey_to_stem = build_image_index(src_dir)
    json_pattern = "**/*.json" if RECURSIVE_SCAN else "*.json"
    json_files   = sorted(src_dir.glob(json_pattern))
    stats["json_found"] = len(json_files)

    # ── 找出有标注的图片集合 ──────────────────────────────────
    print(f"  [INFO] 扫描有效标注图片 ...")
    annotated_set = build_annotated_image_set(src_dir)

    # ── 统计无标注图片 ────────────────────────────────────────
    no_json_images = [
        str(img.relative_to(src_dir))
        for img in image_paths
        if img.resolve() not in annotated_set
    ]
    if no_json_images:
        print(f"  [WARN] {len(no_json_images)} 张图无 JSON，跳过：")
        for name in no_json_images[:20]:
            print(f"    - {name}")
        if len(no_json_images) > 20:
            print(f"    ... 共 {len(no_json_images)} 张，仅显示前 20 张")
    stats["images_skipped_no_json"] = len(no_json_images)

    annotated_image_paths = [img for img in image_paths if img.resolve() in annotated_set]
    print(
        f"  [INFO] 有标注图片 {len(annotated_image_paths)} 张，"
        f"JSON 标注 {len(json_files)} 个，开始复制与转换 ..."
    )

    # ── 复制图片（有标注才复制，det/seg 各一份）───────────────
    total_annotated_images = len(annotated_image_paths)
    for img_index, img in enumerate(annotated_image_paths, start=1):
        stem     = image_to_stem[img.resolve()]
        dst_name = stem + img.suffix.lower()
        copied = False
        if dst_img_det is not None:
            shutil.copy2(img, dst_img_det / dst_name)
            copied = True
        if dst_img_seg is not None:
            shutil.copy2(img, dst_img_seg / dst_name)
            copied = True
        if copied:
            stats["images_copied"] += 1
        log_progress("复制图片", img_index, total_annotated_images)

    # ── 转换标注 → det txt + seg txt ─────────────────────────
    total_json_files = len(json_files)
    for json_index, json_file in enumerate(json_files, start=1):
        try:
            data         = load_json(json_file)
            img_w, img_h = get_image_size(data, json_file)
            shapes       = data.get("shapes", [])
        except Exception as e:
            reason = f"{SKIP_JSON_ERROR}: {e}"
            print(f"  [SKIP] {json_file.name} | {reason}")
            skipped_detail.append({"file": json_file.name, "shape": None, "reason": reason})
            stats["skipped_json"] += 1
            continue

        matched_image = find_image_for_json(data, json_file)
        if matched_image and matched_image.resolve() in image_to_stem:
            out_stem = image_to_stem[matched_image.resolve()]
        else:
            relkey   = str(json_file.relative_to(src_dir).with_suffix("")).replace("\\", "/")
            out_stem = relkey_to_stem.get(relkey, unique_stem(json_file, src_dir))
            if not matched_image:
                stats["json_without_image"] += 1

        det_lines: List[str] = []
        seg_lines: List[str] = []
        cls_crop_index = 0

        for shape in shapes:
            label      = normalize_label(shape.get("label", ""))
            shape_type = _infer_shape_type(shape)

            if not label:
                skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                       "reason": SKIP_EMPTY_LABEL})
                if "det" in selected_tasks:
                    stats["skipped_shapes_det"] += 1
                if "seg" in selected_tasks:
                    stats["skipped_shapes_seg"] += 1
                continue

            if label not in class_map:
                reason = f"{SKIP_UNKNOWN_LABEL}: '{label}'"
                skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                       "reason": reason})
                if "det" in selected_tasks:
                    stats["skipped_shapes_det"] += 1
                if "seg" in selected_tasks:
                    stats["skipped_shapes_seg"] += 1
                continue

            cid = class_map[label]

            if shape_type == "rectangle":
                bbox = None
                try:
                    bbox = shape_to_bbox(shape, img_w, img_h)
                except Exception as e:
                    if "det" in selected_tasks:
                        reason = f"{SKIP_SHAPE_TYPE}/{SKIP_CONVERT_ERROR}(det): {e}"
                        skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                               "reason": reason})
                        stats["skipped_shapes_det"] += 1
                    if "cls" in selected_tasks:
                        reason = f"{SKIP_CONVERT_ERROR}(cls): {e}"
                        skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                               "reason": reason})
                        stats["cls_images_skipped"] += 1
                if bbox is not None and "det" in selected_tasks:
                    try:
                        cx, cy, w, h = bbox_to_yolo(*bbox, img_w, img_h)
                        det_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    except Exception as e:
                        reason = f"{SKIP_SHAPE_TYPE}/{SKIP_CONVERT_ERROR}(det): {e}"
                        skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                               "reason": reason})
                        stats["skipped_shapes_det"] += 1

                if "cls" in selected_tasks and bbox is not None and matched_image and matched_image.exists():
                    try:
                        crop_name = f"{out_stem}__{cls_crop_index:04d}{matched_image.suffix.lower()}"
                        save_crop_image(
                            matched_image,
                            bbox,
                            dst_img_cls / label / crop_name,  # type: ignore[operator]
                        )
                        cls_crop_index += 1
                        stats["cls_crops"] += 1
                    except Exception as e:
                        reason = f"{SKIP_CONVERT_ERROR}(cls): {e}"
                        skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                               "reason": reason})
                        stats["cls_images_skipped"] += 1
                elif "cls" in selected_tasks and matched_image is None:
                    skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                           "reason": "cls 缺少源图片"})
                    stats["cls_images_skipped"] += 1

            elif shape_type == "polygon" and "seg" in selected_tasks:
                try:
                    coords = shape_to_seg(shape, img_w, img_h)
                    coord_str = " ".join(f"{v:.6f}" for v in coords)
                    seg_lines.append(f"{cid} {coord_str}")
                except Exception as e:
                    reason = f"{SKIP_SHAPE_TYPE}/{SKIP_CONVERT_ERROR}(seg): {e}"
                    skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                           "reason": reason})
                    stats["skipped_shapes_seg"] += 1
            else:
                reason = f"{SKIP_SHAPE_TYPE}: '{shape_type}'"
                skipped_detail.append({"file": json_file.name, "shape": shape_type,
                                       "reason": reason})
                if "det" in selected_tasks:
                    stats["skipped_shapes_det"] += 1
                if "seg" in selected_tasks:
                    stats["skipped_shapes_seg"] += 1

        if dst_lbl_det is not None:
            det_txt = dst_lbl_det / f"{out_stem}.txt"
            det_txt.write_text("\n".join(det_lines), encoding="utf-8")
            if not det_lines:
                stats["empty_labels_det"] += 1

        if dst_lbl_seg is not None:
            seg_txt = dst_lbl_seg / f"{out_stem}.txt"
            seg_txt.write_text("\n".join(seg_lines), encoding="utf-8")
            if not seg_lines:
                stats["empty_labels_seg"] += 1

        stats["json_converted"] += 1
        log_progress("转换标注", json_index, total_json_files)

    # ── 无标注图片补空 txt（可选）────────────────────────────
    if CREATE_EMPTY_TXT_FOR_IMAGE_WITHOUT_JSON:
        for img_resolved, out_stem in image_to_stem.items():
            for lbl_dir in [dst_lbl_det, dst_lbl_seg]:
                if lbl_dir is None:
                    continue
                txt = lbl_dir / f"{out_stem}.txt"
                if not txt.exists():
                    txt.write_text("", encoding="utf-8")

    stats["_skipped_detail"] = skipped_detail
    return stats


# ============================================================
# 完整性校验
# ============================================================
def integrity_check(target_root: Path, splits: List[str], label: str) -> bool:
    print(f"\n── 完整性检查 [{label}] ────────────────────────────────")
    all_ok = True
    for split in splits:
        img_dir = target_root / "images" / split
        lbl_dir = target_root / "labels" / split
        if not img_dir.exists():
            continue

        missing_txt = [
            img.name for img in sorted(img_dir.iterdir())
            if is_image_file(img) and not (lbl_dir / f"{img.stem}.txt").exists()
        ]
        missing_img = []
        if lbl_dir.exists():
            missing_img = [
                txt.name for txt in sorted(lbl_dir.glob("*.txt"))
                if not any((img_dir / f"{txt.stem}{ext}").exists() for ext in IMAGE_EXTS)
            ]

        if missing_txt or missing_img:
            all_ok = False
            if missing_txt:
                print(f"  [{split}] ⚠️  {len(missing_txt)} 张图没有对应 txt")
                for n in missing_txt[:10]: print(f"    - {n}")
            if missing_img:
                print(f"  [{split}] ⚠️  {len(missing_img)} 个 txt 没有对应图片")
                for n in missing_img[:10]: print(f"    - {n}")
        else:
            cnt = sum(1 for f in img_dir.iterdir() if is_image_file(f))
            print(f"  [{split}] ✓  {cnt} 张图均有对应 txt")
    return all_ok


# ============================================================
# 报告输出
# ============================================================
def print_report(all_stats: dict) -> None:
    print("\n" + "=" * 60)
    print("  转换报告")
    print("=" * 60)

    for split, stats in all_stats.items():
        print(f"\n【{split}】")
        print(f"  图片复制（有标注）:        {stats['images_copied']}")
        print(f"  图片跳过（无标注）:        {stats['images_skipped_no_json']}")
        print(f"  标注文件总数:             {stats['json_found']}")
        print(f"  标注成功转换:             {stats['json_converted']}")
        print(f"  空 txt（det）:             {stats['empty_labels_det']}")
        print(f"  空 txt（seg）:             {stats['empty_labels_seg']}")
        print(f"  分类裁剪输出:             {stats['cls_crops']}")
        print(f"  分类裁剪跳过:             {stats['cls_images_skipped']}")
        print(f"  标注无图片:               {stats['json_without_image']}")
        print(f"  标注解析失败:             {stats['skipped_json']}")
        print(f"  shape 跳过（det）:         {stats['skipped_shapes_det']}")
        print(f"  shape 跳过（seg）:         {stats['skipped_shapes_seg']}")

        details = stats.get("_skipped_detail", [])
        if details:
            print("\n  ── 跳过明细（按原因聚合）──")
            for reason, cnt in Counter(d["reason"] for d in details).most_common():
                print(f"    {cnt:4d}x  {reason}")


def integrity_check_cls(target_root: Path, splits: List[str]) -> bool:
    print(f"\n── 完整性检查 [cls] ────────────────────────────────")
    all_ok = True
    for split in splits:
        split_dir = target_root / split
        if not split_dir.exists():
            continue
        image_count = 0
        class_count = 0
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_count += 1
            count = sum(1 for f in class_dir.iterdir() if is_image_file(f))
            image_count += count
        if image_count == 0:
            all_ok = False
            print(f"  [{split}] ⚠️  未发现任何分类裁剪图")
        else:
            print(f"  [{split}] ✓  {image_count} 张分类图，覆盖 {class_count} 个类别目录")
    return all_ok


# ============================================================
# 主函数
# ============================================================
def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    selected_tasks = normalize_selected_tasks(args.task)

    if args.class_ref:
        ref_path = Path(args.class_ref)
        if not ref_path.exists():
            print(f"[ERROR] --class-ref 文件不存在: {ref_path}")
            sys.exit(1)
        global FIXED_CLASS_MAP
        FIXED_CLASS_MAP = load_class_ref(ref_path)
    target_det = output_root / "dataset_det"
    target_cls = output_root / "dataset_cls"
    target_seg = output_root / "dataset_seg"

    if args.source_format == "auto":
        source_format = detect_source_format(source_root, SPLITS)
    else:
        source_format = args.source_format

    print("=" * 60)
    print("  数据集转换（LabelMe / YOLO -> 统一 YOLO 输出）")
    print("=" * 60)
    print(f"SOURCE_ROOT : {source_root.resolve()}")
    print(f"OUTPUT_ROOT : {output_root.resolve()}")
    print(f"TASKS       : {sorted(selected_tasks)}")
    if "det" in selected_tasks:
        print(f"TARGET_DET  : {target_det.resolve()}")
    if "cls" in selected_tasks:
        print(f"TARGET_CLS  : {target_cls.resolve()}")
    if "seg" in selected_tasks:
        print(f"TARGET_SEG  : {target_seg.resolve()}")
    print(f"SPLITS      : {SPLITS}")
    print(f"RECURSIVE   : {RECURSIVE_SCAN}")
    print(f"SOURCE_FMT  : {source_format}")

    if not source_root.exists():
        print(f"\n[ERROR] SOURCE_ROOT 不存在: {source_root.resolve()}")
        sys.exit(1)
    if source_format == "mixed":
        print("\n[ERROR] 自动检测到输入同时包含 JSON 与 TXT，无法自动判定。")
        print("        请显式传入 --source-format labelme 或 --source-format yolo")
        sys.exit(1)
    if source_format == "unknown":
        print("\n[ERROR] 未检测到可用标注文件（.json 或 .txt）。")
        sys.exit(1)

    safe_mkdir(output_root)
    selected_targets = []
    if "det" in selected_tasks:
        selected_targets.append(("det", target_det))
    if "cls" in selected_tasks:
        selected_targets.append(("cls", target_cls))
    if "seg" in selected_tasks:
        selected_targets.append(("seg", target_seg))

    for task_name, target in selected_targets:
        safe_mkdir(target)
        handle_existing_output(target)
        if task_name == "cls":
            for split in SPLITS:
                safe_mkdir(target / split)
        else:
            safe_mkdir(target / "images")
            safe_mkdir(target / "labels")

    if source_format == "labelme":
        class_map = collect_classes(source_root, SPLITS)
    else:
        class_map = collect_classes_from_yolo(source_root, SPLITS)
    if not class_map:
        print("\n[ERROR] 没有收集到任何类别！")
        print(f"  SOURCE_ROOT: {source_root.resolve()}")
        sys.exit(1)

    print(f"\n[INFO] 类别映射（共 {len(class_map)} 类）：")
    for k, v in sorted(class_map.items(), key=lambda x: x[1]):
        print(f"  {v:3d}  {k}")

    if "det" in selected_tasks:
        write_classes_file(class_map, target_det / "classes.txt")
        write_data_yaml(class_map, target_det, task="detect")

    if "cls" in selected_tasks:
        write_classes_file(class_map, target_cls / "classes.txt")
        write_cls_data_yaml(class_map, target_cls)

    if "seg" in selected_tasks:
        write_classes_file(class_map, target_seg / "classes.txt")
        write_data_yaml(class_map, target_seg, task="segment")

    all_stats: dict = {}
    for split in SPLITS:
        print(f"\n{'─'*60}")
        print(f"[INFO] 处理 split: {split} ...")
        if source_format == "labelme":
            all_stats[split] = convert_split(
                split, source_root, target_det, target_cls, target_seg, class_map, selected_tasks
            )
        else:
            all_stats[split] = convert_split_from_yolo(
                split,
                source_root,
                target_det,
                target_cls,
                target_seg,
                {cid: name for name, cid in class_map.items()},
                selected_tasks,
            )

    print_report(all_stats)

    if INTEGRITY_CHECK:
        if "det" in selected_tasks:
            integrity_check(target_det, SPLITS, "det")
        if "cls" in selected_tasks:
            integrity_check_cls(target_cls, SPLITS)
        if "seg" in selected_tasks:
            integrity_check(target_seg, SPLITS, "seg")

    if "det" in selected_tasks:
        print(f"\n检测输出: {target_det.resolve()}")
    if "cls" in selected_tasks:
        print(f"分类输出: {target_cls.resolve()}")
    if "seg" in selected_tasks:
        print(f"分割输出: {target_seg.resolve()}")
    print("\n转换完成 ✓  原数据集未被修改。")


if __name__ == "__main__":
    main()
