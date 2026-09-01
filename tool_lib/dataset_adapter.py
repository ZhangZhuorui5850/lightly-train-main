"""Bridge launcher components to the shared conversion dataset detector.

The detector lives with the conversion tools.  This adapter keeps launcher imports
portable and gives training/inference code one stable API for discovery and paths.
"""

from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONVERT_TOOLS_DIR = (
    PROJECT_ROOT / "datasets" / "convert_datasets" / "convert_tools"
)


@lru_cache(maxsize=1)
def _shared_modules() -> tuple[ModuleType, ModuleType]:
    if not CONVERT_TOOLS_DIR.is_dir():
        raise RuntimeError(f"数据集检测组件目录不存在: {CONVERT_TOOLS_DIR}")
    tools_path = str(CONVERT_TOOLS_DIR)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    detector = importlib.import_module("dataset_detector")
    discovery = importlib.import_module("dataset_discovery")
    return detector, discovery


def discover_dataset_configs(
    search_root: Path,
    *,
    task: str | None = None,
    include_empty: bool = False,
    include_unknown: bool = False,
    count_limit: int | None = 200,
    show_progress: bool = True,
) -> list[Path]:
    """Return detected dataset config files in detector order."""
    detector, _ = _shared_modules()
    task_kinds = {
        "det": {"yolo_detection"},
        "seg": {"yolo_instance", "semantic_mask"},
        "instance": {"yolo_instance"},
        "semantic": {"semantic_mask"},
        "cls": {"image_classification"},
    }
    kinds: Iterable[str] | None = task_kinds.get(task or "")
    candidates = detector.detect_datasets(
        Path(search_root),
        kinds=kinds,
        include_empty=include_empty,
        include_unknown=include_unknown,
        show_progress=show_progress,
        count_limit=count_limit,
    )
    return [
        (candidate.config_path or candidate.path).resolve()
        for candidate in candidates
        if candidate.config_path is not None or task == "cls"
    ]


def resolve_split_samples(
    config_path: Path,
    config: dict[str, Any],
    split: str,
    *,
    annotation: str = "labels",
) -> list[tuple[Path, Path, Path]]:
    """Resolve the exact configured samples for one split.

    Each tuple contains ``(image_path, annotation_path, relative_path)``.  The
    file-level representation preserves manifests and multi-directory splits.
    """
    _, discovery = _shared_modules()
    config_path = Path(config_path).expanduser().resolve()
    root = discovery.dataset_root_from_config(config_path, config)
    return discovery.split_sample_files(
        root,
        config,
        split,
        annotation=annotation,
    )


def resolve_dataset_root(config_path: Path, config: dict[str, Any]) -> Path:
    """Resolve a YAML dataset root with stale-path recovery."""
    _, discovery = _shared_modules()
    return discovery.dataset_root_from_config(
        Path(config_path).expanduser().resolve(), config
    )


def resolve_split_paths(
    config_path: Path,
    config: dict[str, Any],
    split: str,
    *,
    annotation: str = "labels",
) -> tuple[Path | None, Path | None]:
    """Resolve one split for images-first and split-first layouts."""
    _, discovery = _shared_modules()
    config_path = Path(config_path).expanduser().resolve()
    root = discovery.dataset_root_from_config(config_path, config)
    image_dirs = discovery.split_image_dirs(
        root,
        config,
        selected_splits=(split,),
    )
    annotation_dirs = discovery.split_annotation_dirs(
        root,
        config,
        image_dirs,
        annotation=annotation,
        include_missing=True,
        selected_splits=(split,),
    )
    return image_dirs.get(split), annotation_dirs.get(split)


def path_diagnostics(config_path: Path, config: dict[str, Any]) -> list[str]:
    _, discovery = _shared_modules()
    return discovery.dataset_path_diagnostics(
        Path(config_path).expanduser().resolve(), config
    )
