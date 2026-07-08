"""infer / eval 结果的复用与安全覆盖。

统一解决一个反复出现的痛点：再次推理/评估时目录非空就报
"Output directory is not empty"，每次都要手动加 overwrite，且覆盖时
旧文件和新文件混在一起。

设计（"科学一点"）：
- 每次 infer/eval 结束都会写 run_meta.json（已有），里面记录了本次运行的
  指纹：checkpoint / data / split / 阈值 等。run_meta 在流程末尾才写，
  所以它存在即代表上一次跑完了（completeness 代理）。
- 再次运行时只做"初步检测"，不再询问用户：
    - run_meta 指纹与当前请求一致、且所需产物齐全 → 直接复用，跳过推理/评估。
    - 否则（缺数据 / 配置不一致 / 没跑完）→ 自动重新生成所需数据。
- 覆盖时按 clean 语义清掉旧产物，但保留 _tempfile / _temp 这些缓存目录
  （缓存的预测 JSON、训练曲线等"画图的东西"）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import common as rt


# 决策结果
RUN = "run"            # 缺数据/不一致 → 自动重新生成
REUSE = "reuse"        # 复用已有结果，跳过


def _disp(path: Path) -> str:
    """尽量相对 out/ 或仓库根展示路径，失败则用绝对路径。"""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(rt.ROOT_DIR))
    except (ValueError, OSError):
        return str(path)


def _read_run_meta(output_dir: Path) -> dict[str, Any] | None:
    meta_path = Path(output_dir) / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _norm_path(value: Any) -> str | None:
    if value in (None, "", "None"):
        return None
    try:
        return str(Path(str(value)).expanduser().resolve())
    except OSError:
        return str(value)


def _norm_split(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return sorted(str(v) for v in value)
    return [str(value)]


def _norm_num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def make_fingerprint(
    *,
    task: str,
    action: str,
    checkpoint: Any,
    data: Any,
    split: Any,
    threshold: Any,
    input_mode: Any = None,
    image: Any = None,
    image_dir: Any = None,
    seg_train_type: Any = None,
) -> dict[str, Any]:
    """构造当前请求的指纹（与 run_meta 字段对齐后可比较）。"""
    return {
        "task": str(task),
        "action": str(action),
        "checkpoint": _norm_path(checkpoint),
        "data": _norm_path(data),
        "split": _norm_split(split),
        "threshold": _norm_num(threshold),
        "input_mode": str(input_mode) if input_mode is not None else None,
        "image": _norm_path(image),
        "image_dir": _norm_path(image_dir),
        "seg_train_type": str(seg_train_type) if seg_train_type is not None else None,
    }


def fingerprint_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """从 run_meta.json 还原成同结构指纹。det 与 seg 的 schema 略有差异，这里统一兼容。"""
    paths = meta.get("paths", {}) or {}
    settings = meta.get("settings", {}) or {}
    # 阈值：det 用 score_threshold，seg 用 threshold。
    threshold = settings.get("threshold", settings.get("score_threshold"))
    # 数据配置：det 用 data_yaml，seg 用 data。
    data = paths.get("data", paths.get("data_yaml"))
    return {
        "task": meta.get("task"),
        "action": meta.get("action"),
        "checkpoint": _norm_path(paths.get("checkpoint_path") or paths.get("checkpoint")),
        "data": _norm_path(data),
        "split": _norm_split(meta.get("split")),
        "threshold": _norm_num(threshold),
        "input_mode": meta.get("input_mode"),
        "image": _norm_path(paths.get("image")),
        "image_dir": _norm_path(paths.get("image_dir")),
        "seg_train_type": meta.get("seg_train_type"),
    }


# 核心指纹：两套 schema（det/seg）都可靠记录，恒比较。
_CORE_KEYS = ("checkpoint", "data", "split", "threshold", "seg_train_type")
# 辅助指纹：det 记录、seg 旧 meta 不记录。仅当 meta 里有值时才比较，
# 这样不会因 seg meta 缺这些字段而误判"不一致"。
_AUX_KEYS = ("input_mode", "image", "image_dir")


def fingerprint_diff(current: dict[str, Any], saved: dict[str, Any]) -> list[str]:
    """返回不一致的关键字段名列表；空列表表示可复用。"""
    diffs = []
    for key in _CORE_KEYS:
        if current.get(key) != saved.get(key):
            diffs.append(key)
    for key in _AUX_KEYS:
        if saved.get(key) is not None and current.get(key) != saved.get(key):
            diffs.append(key)
    return diffs


def results_present(output_dir: Path, required: tuple[str, ...] | list[str]) -> bool:
    """初步检测：required 列的相对路径是否都已存在（目录需非空）。"""
    output_dir = Path(output_dir)
    for rel in required:
        path = output_dir / rel
        if not path.exists():
            return False
        if path.is_dir() and not any(path.iterdir()):
            return False
    return True


def precheck(
    args: Any,
    output_dir: Path,
    fingerprint: dict[str, Any],
    *,
    action_label: str,
    required: tuple[str, ...] | list[str] = (),
) -> str:
    """infer/eval 父进程的初步复用检测（不询问用户）。

    - 已有结果且配置一致、所需产物齐全 → REUSE（直接复用，跳过推理/评估）。
    - 否则（缺数据 / 配置不一致 / 没跑完）→ RUN：自动把 args.overwrite 置 True，
      让随后的 prepare_output_dir(..., clean=True) 清掉旧产物（保留 _tempfile/_temp
      缓存）后重新生成所需数据。

    分片子进程由调用方负责跳过，不进此函数。
    """
    output_dir = Path(output_dir)
    meta = _read_run_meta(output_dir)
    reusable = (
        meta is not None
        and not fingerprint_diff(fingerprint, fingerprint_from_meta(meta))
        and results_present(output_dir, required)
    )
    if reusable:
        print(f"[{action_label}] ✓ 已有所需结果且配置一致，直接复用：{_disp(output_dir)}")
        return REUSE
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"[{action_label}] 已有结果缺数据/配置不一致，自动重新生成。")
    args.overwrite = True
    return RUN
