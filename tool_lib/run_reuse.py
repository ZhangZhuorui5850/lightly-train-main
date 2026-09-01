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

import hashlib
import json
from pathlib import Path
from typing import Any

from . import common as rt
from .file_index import find_files
from .progress import track


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
        return data if isinstance(data, dict) and data.get("complete") is True else None
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


def _path_signature(value: Any) -> dict[str, Any] | None:
    normalized = _norm_path(value)
    if normalized is None:
        return None
    path = Path(normalized)
    try:
        stat = path.stat()
    except OSError:
        return {"path": normalized, "missing": True}
    return {
        "path": normalized,
        "kind": "dir" if path.is_dir() else "file",
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _dataset_signature(value: Any) -> dict[str, Any] | None:
    """Return a metadata digest for a dataset config and its effective root.

    Inference reuse must become stale when an image, label, mask, or manifest is
    added, removed, or edited.  Hashing the full image payload would make every
    precheck unnecessarily expensive, so image files contribute path/size/mtime
    metadata while small text annotations and configs also contribute content.
    """
    normalized = _norm_path(value)
    if normalized is None:
        return None
    config_path = Path(normalized)
    if not config_path.exists():
        return _path_signature(value)

    root = config_path if config_path.is_dir() else config_path.parent
    if config_path.is_file() and config_path.suffix.casefold() in {".yaml", ".yml"}:
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(config, dict):
                root = rt.dataset_adapter.resolve_dataset_root(config_path, config)
        except Exception:  # 配置读取失败时仍可对配置所在目录生成保守指纹
            root = config_path.parent

    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    latest_mtime_ns = 0
    try:
        paths = sorted(
            find_files(
                [root],
                label="索引数据集指纹",
                patterns=["*"],
            ),
            key=lambda path: path.as_posix().casefold(),
        )
    except OSError:
        return {"path": normalized, "root": str(root), "unreadable": True}

    for path in track(
        paths,
        label="计算数据集指纹",
        total=len(paths),
        unit="file",
    ):
        try:
            stat = path.stat()
            relative = path.relative_to(root).as_posix()
        except (OSError, ValueError):
            continue
        file_count += 1
        total_size += int(stat.st_size)
        latest_mtime_ns = max(latest_mtime_ns, int(stat.st_mtime_ns))
        digest.update(relative.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
        digest.update(b"\0")
        if path.suffix.casefold() in {".txt", ".json", ".yaml", ".yml", ".csv"}:
            try:
                digest.update(path.read_bytes())
            except OSError:
                digest.update(b"<unreadable>")
        digest.update(b"\n")

    return {
        "path": normalized,
        "root": str(root.resolve()),
        "file_count": file_count,
        "total_size": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "metadata_sha256": digest.hexdigest(),
    }


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
    options: dict[str, Any] | None = None,
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
        "source_signatures": {
            "checkpoint": _path_signature(checkpoint),
            "data": _dataset_signature(data),
            "image": _path_signature(image),
            "image_dir": _dataset_signature(image_dir),
        },
        "options": options or {},
    }


def fingerprint_from_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """从 run_meta.json 还原成同结构指纹。det 与 seg 的 schema 略有差异，这里统一兼容。"""
    saved_fingerprint = meta.get("fingerprint")
    if isinstance(saved_fingerprint, dict):
        return saved_fingerprint
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
        "source_signatures": None,
        "options": None,
    }


# 核心指纹：两套 schema（det/seg）都可靠记录，恒比较。
_CORE_KEYS = (
    "checkpoint", "data", "split", "threshold", "seg_train_type",
    "source_signatures", "options",
)
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
