from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tool_lib import common as rt
from tool_lib import det_shared, seg_shared, train_tools
from tool_lib import interactive


def _write_det_dataset(root: Path, *, layout: str, config_name: str = "data.yaml") -> Path:
    root.mkdir(parents=True)
    split_value = "images/train" if layout == "images_first" else "train"
    config_path = root / config_name
    config_path.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": split_value,
                "task": "detect",
                "names": {0: "part"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    if layout == "images_first":
        image = root / "images/train/a.jpg"
        label = root / "labels/train/a.txt"
    else:
        image = root / "train/images/a.jpg"
        label = root / "train/labels/a.txt"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return config_path


@pytest.mark.parametrize("layout", ("images_first", "split_first"))
def test_launcher_resolves_shared_dataset_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: str,
) -> None:
    config_path = _write_det_dataset(tmp_path / "dataset", layout=layout)
    monkeypatch.setattr(rt, "yaml", yaml)

    config = rt.load_data_config(config_path)
    image_dir, label_dir, names = rt.resolve_dataset_split_paths(config, "train")

    expected_image_dir = (
        config_path.parent / ("images/train" if layout == "images_first" else "train/images")
    )
    expected_label_dir = (
        config_path.parent / ("labels/train" if layout == "images_first" else "train/labels")
    )
    assert image_dir == expected_image_dir.resolve()
    assert label_dir == expected_label_dir.resolve()
    assert names == {0: "part"}


def test_launcher_discovery_uses_content_and_recovers_stale_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets_root = tmp_path / "datasets"
    config_path = _write_det_dataset(
        datasets_root / "arbitrary_name",
        layout="split_first",
        config_name="project.yml",
    )
    stale_root = tmp_path / "old_location"
    stale_root.mkdir()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["path"] = str(stale_root)
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(rt, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(rt, "yaml", yaml)

    candidates = interactive.list_dataset_yaml_candidates("det")
    loaded = rt.load_data_config(config_path)

    assert candidates == [config_path.resolve()]
    assert loaded["_root_dir"] == config_path.parent.resolve()


def test_launcher_discovery_includes_configured_external_dataset_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    external = tmp_path / "mounted_data" / "external_det"
    config_path = _write_det_dataset(external, layout="images_first")
    (project / "datasets").mkdir(parents=True)
    monkeypatch.setattr(rt, "ROOT_DIR", project)
    monkeypatch.setattr(rt, "DATASET_DIR", external)
    monkeypatch.setattr(rt, "SEG_DATASET_DIR", project / "missing_seg")
    monkeypatch.setattr(rt, "SEMANTIC_SEG_DATASET_DIR", project / "missing_semantic")

    candidates = interactive.list_dataset_yaml_candidates("det")

    assert candidates == [config_path.resolve()]


def test_launcher_discovery_scans_every_configured_search_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    first = tmp_path / "mounted_a"
    second = tmp_path / "mounted_b"
    first_config = _write_det_dataset(first / "alpha", layout="images_first")
    second_config = _write_det_dataset(second / "beta", layout="split_first")
    (project / "datasets").mkdir(parents=True)
    monkeypatch.setattr(rt, "ROOT_DIR", project)
    monkeypatch.setattr(rt, "DATASET_SEARCH_ROOTS", [first, second])
    monkeypatch.setattr(rt, "DATASET_DIR", project / "missing_det")
    monkeypatch.setattr(rt, "SEG_DATASET_DIR", project / "missing_seg")
    monkeypatch.setattr(rt, "SEMANTIC_SEG_DATASET_DIR", project / "missing_semantic")

    candidates = interactive.list_dataset_yaml_candidates("det")

    assert set(candidates) == {first_config.resolve(), second_config.resolve()}


def test_launcher_dataset_keyword_filter_supports_chinese_paths(tmp_path: Path) -> None:
    candidates = [tmp_path / "军事数据" / "data.yaml", tmp_path / "工业缺陷" / "data.yaml"]

    assert interactive.filter_dirs_by_keyword(candidates, "军事 数据") == [candidates[0]]


def test_launcher_discovery_keeps_declared_empty_dataset_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    dataset = project / "datasets" / "waiting_for_images"
    dataset.mkdir(parents=True)
    config_path = dataset / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"path": ".", "train": "images/train", "task": "detect", "names": ["part"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rt, "ROOT_DIR", project)
    monkeypatch.setattr(rt, "DATASET_SEARCH_ROOTS", [])
    monkeypatch.setattr(rt, "DATASET_DIR", project / "missing_det")
    monkeypatch.setattr(rt, "SEG_DATASET_DIR", project / "missing_seg")
    monkeypatch.setattr(rt, "SEMANTIC_SEG_DATASET_DIR", project / "missing_semantic")

    assert interactive.list_dataset_yaml_candidates("det") == [config_path.resolve()]


def test_launcher_discovers_raw_classification_imagefolder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    root = project / "datasets" / "classification"
    for split in ("train", "val"):
        for class_name in ("cat", "dog"):
            image = root / split / class_name / f"{class_name}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"image")
    monkeypatch.setattr(rt, "ROOT_DIR", project)
    monkeypatch.setattr(rt, "DATASET_SEARCH_ROOTS", [])
    monkeypatch.setattr(rt, "DATASET_DIR", project / "missing_det")
    monkeypatch.setattr(rt, "SEG_DATASET_DIR", project / "missing_seg")
    monkeypatch.setattr(rt, "SEMANTIC_SEG_DATASET_DIR", project / "missing_semantic")

    candidates = interactive.list_dataset_yaml_candidates("cls")
    data = train_tools.load_classification_directory_data_config(root)

    assert candidates == [root.resolve()]
    assert data["classes"] == {0: "cat", 1: "dog"}
    assert train_tools._count_train_images(root) == 2


def test_launcher_uses_exact_manifest_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "manifest_det"
    for stem in ("selected", "extra"):
        image = root / f"images/train/{stem}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label = root / f"labels/train/{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "train.txt").write_text("images/train/selected.jpg\n", encoding="utf-8")
    config_path = root / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"path": ".", "train": "train.txt", "task": "detect", "names": ["part"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rt, "yaml", yaml)

    config = rt.load_data_config(config_path)
    samples, _ = rt.list_dataset_samples(config, "train")

    assert [sample.image_path.name for sample in samples] == ["selected.jpg"]
    assert samples[0].label_path == (root / "labels/train/selected.txt").resolve()


def test_launcher_preserves_every_directory_in_multi_source_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "multi_source_det"
    for group in ("left", "right"):
        image = root / group / f"{group}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
    config_path = root / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"path": ".", "train": ["left", "right"], "task": "detect", "names": ["part"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rt, "yaml", yaml)

    config = rt.load_data_config(config_path)
    samples, _ = rt.list_dataset_samples(config, "train")

    assert {sample.image_path.name for sample in samples} == {"left.jpg", "right.jpg"}
    assert {sample.relative_path.as_posix() for sample in samples} == {
        "left/left.jpg",
        "right/right.jpg",
    }


def test_det_export_collection_uses_exact_manifest_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "manifest_det"
    for stem in ("selected", "extra"):
        image = root / f"images/train/{stem}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        label = root / f"labels/train/{stem}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (root / "train.txt").write_text("images/train/selected.jpg\n", encoding="utf-8")
    config_path = root / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"path": ".", "train": "train.txt", "task": "detect", "names": ["part"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rt, "yaml", yaml)
    config = rt.load_data_config(config_path)

    infos_by_split, export_paths = det_shared.collect_source_image_infos(config, root)

    assert [item.src_image_path.name for item in infos_by_split["train"]] == ["selected.jpg"]
    assert export_paths["train"] == "images/train"
    assert train_tools._count_train_images(config_path) == 1


def test_seg_export_collection_uses_every_multi_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "multi_seg"
    for group in ("left", "right"):
        image = root / group / f"{group}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
    config_path = root / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"path": ".", "train": ["left", "right"], "task": "segment", "names": ["part"]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rt, "yaml", yaml)
    config = rt.load_data_config(config_path)

    infos_by_split, export_paths = seg_shared.collect_source_image_infos(config, root)

    assert {item.src_image_path.name for item in infos_by_split["train"]} == {
        "left.jpg",
        "right.jpg",
    }
    assert export_paths["train"] == "images/train"
