from __future__ import annotations

import base64
import importlib.util
import sys
from pathlib import Path

from tool_lib import convert_tools
from tool_lib.interactive import parse_cli_args


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2pG1EAAAAASUVORK5CYII="
)


def _load_labelme_to_yolo_module():
    module_path = Path(__file__).resolve().parents[1] / "datasets" / "convert_datasets" / "LabelMeToYOLO.py"
    spec = importlib.util.spec_from_file_location("labelme_to_yolo_test_module", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_minimal_yolo_source(root: Path) -> Path:
    source_root = root / "source"
    train_dir = source_root / "train"
    train_dir.mkdir(parents=True)
    (source_root / "classes.txt").write_text("cat\n", encoding="utf-8")
    (train_dir / "sample.png").write_bytes(PNG_1X1)
    (train_dir / "sample.txt").write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    return source_root


def test_parse_cli_args_accepts_single_convert_task() -> None:
    args = parse_cli_args(["convert", "demo_set", "--task", "det"])

    assert args.tool_task == "data"
    assert args.tool_action == "convert"
    assert args.task == "det"


def test_run_convert_passes_selected_task_to_one_click_module(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class DummyModule:
        @staticmethod
        def run_conversion(sources, output_root, **kwargs):
            calls["sources"] = sources
            calls["output_root"] = output_root
            calls["kwargs"] = kwargs

    monkeypatch.setattr(convert_tools, "load_one_click_convert_module", lambda: DummyModule())
    monkeypatch.setattr(convert_tools, "resolve_convert_source_dir", lambda source_dir: tmp_path / "source")

    args = parse_cli_args(["convert", "demo_set", "--task", "seg"])
    convert_tools.run_convert(args)

    assert calls["kwargs"]["task"] == "seg"


def test_labelme_to_yolo_creates_only_selected_dataset(tmp_path: Path, monkeypatch) -> None:
    module = _load_labelme_to_yolo_module()
    source_root = _build_minimal_yolo_source(tmp_path)
    output_root = tmp_path / "converted"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "LabelMeToYOLO.py",
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--source-format",
            "yolo",
            "--task",
            "det",
        ],
    )

    module.main()

    assert (output_root / "dataset_det").exists()
    assert not (output_root / "dataset_cls").exists()
    assert not (output_root / "dataset_seg").exists()
