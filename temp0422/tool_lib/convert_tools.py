"""数据集一键转换工具。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from . import common as rt

CONVERT_DATASETS_ROOT = rt.ROOT_DIR / "datasets" / "convert_datasets"
ONE_CLICK_CONVERT_PATH = CONVERT_DATASETS_ROOT / "one_click_convert.py"


def list_convert_source_dirs() -> list[Path]:
    if not CONVERT_DATASETS_ROOT.exists():
        return []
    candidates = [
        path.resolve()
        for path in CONVERT_DATASETS_ROOT.iterdir()
        if path.is_dir() and path.name != "__pycache__" and not path.name.startswith(".")
    ]
    candidates.sort(key=lambda path: path.name.lower())
    return candidates


def resolve_convert_source_dir(source_dir: str | Path) -> Path:
    source_path = Path(source_dir).expanduser()
    candidates = []
    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append((Path.cwd() / source_path).resolve())
        candidates.append((CONVERT_DATASETS_ROOT / source_path).resolve())
        candidates.append((rt.ROOT_DIR / source_path).resolve())

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Convert source directory does not exist: {source_dir}")


def load_one_click_convert_module():
    spec = importlib.util.spec_from_file_location("one_click_convert_module", ONE_CLICK_CONVERT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {ONE_CLICK_CONVERT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_convert(args) -> None:
    module = load_one_click_convert_module()
    source_dir = resolve_convert_source_dir(args.source_dir)
    output_root = Path(args.output_root).expanduser().resolve()
    module.run_conversion(
        [source_dir],
        output_root,
        label_format=args.label_format,
        seed=args.seed,
        dry_run=args.dry_run,
    )
