from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import dataset_discovery as discovery  # noqa: E402
from text_encoding import read_text_auto  # noqa: E402


def _save(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, (3, 2)).save(path)


def test_scan_datasets_collects_generated_and_mvtec_in_one_walk(
    tmp_path: Path, monkeypatch,
) -> None:
    generated_leaf = tmp_path / "generated" / "object" / "scratch"
    _save(generated_leaf / "image/a.png")
    _save(generated_leaf / "image/b.jpg")
    _save(generated_leaf / "fg/a.png", "L")

    category = tmp_path / "mvtec" / "bottle"
    _save(category / "train/good/1.png")
    _save(category / "test/crack/2.png")
    _save(category / "ground_truth/crack/2_mask.png", "L")

    walk_calls = 0
    original_walk = discovery.os.walk

    def counted_walk(*args, **kwargs):
        nonlocal walk_calls
        walk_calls += 1
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(discovery.os, "walk", counted_walk)
    candidates = discovery.scan_datasets(tmp_path, show_progress=False)

    assert walk_calls == 1
    generated = next(item for item in candidates if item.kind == "generated_mask")
    assert generated.path == (tmp_path / "generated").resolve()
    assert (generated.image_count, generated.annotation_count) == (2, 1)
    mvtec = next(item for item in candidates if item.kind == "mvtec")
    assert mvtec.path == (tmp_path / "mvtec").resolve()
    assert (mvtec.image_count, mvtec.annotation_count, mvtec.class_count) == (2, 1, 1)


def test_scan_normalizes_gb18030_yaml_and_preserves_chinese_content(tmp_path: Path) -> None:
    root = tmp_path / "gbk_det"
    config_text = (
        "path: .\n"
        "train: images/train\n"
        "task: detect\n"
        "names:\n"
        "  0: 车辆\n"
    )
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_bytes(config_text.encode("gb18030"))
    _save(root / "images/train/a.jpg")
    (root / "labels/train").mkdir(parents=True)
    (root / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="ascii"
    )

    candidates = discovery.scan_datasets(tmp_path, show_progress=False)
    candidate = next(item for item in candidates if item.path == root.resolve())

    assert candidate.kind == "yolo_detection"
    assert discovery.load_yaml(root / "data.yaml")["names"][0] == "车辆"
    assert (root / "data.yaml").read_bytes() == config_text.encode("gb18030")
    assert read_text_auto(root / "data.yaml", rewrite=True) == config_text
    assert (root / "data.yaml").read_bytes() == config_text.encode("utf-8")


def test_text_normalizer_handles_utf16_bom(tmp_path: Path) -> None:
    path = tmp_path / "classes.txt"
    text = "背景\n裂纹\n"
    path.write_bytes(text.encode("utf-16"))

    assert read_text_auto(path) == text
    assert path.read_bytes() == text.encode("utf-16")
    assert read_text_auto(path, rewrite=True) == text
    assert path.read_bytes() == text.encode("utf-8")


def test_text_normalizer_preserves_big5_chinese(tmp_path: Path) -> None:
    path = tmp_path / "classes.txt"
    text = "台灣資料\n"
    path.write_bytes(text.encode("big5"))

    assert read_text_auto(path) == text
    assert path.read_bytes() == text.encode("big5")
    assert read_text_auto(path, rewrite=True) == text
    assert path.read_bytes() == text.encode("utf-8")


def test_text_normalizer_preserves_cp1252_punctuation(tmp_path: Path) -> None:
    path = tmp_path / "classes.txt"
    text = "“café”\n€ naïve résumé\n"
    path.write_bytes(text.encode("cp1252"))

    assert read_text_auto(path) == text
    assert read_text_auto(path, rewrite=True) == text
    assert path.read_bytes() == text.encode("utf-8")


def test_scan_accepts_yml_uppercase_and_content_identified_configs(
    tmp_path: Path,
) -> None:
    det = tmp_path / "det"
    (det / "DATA.YML").parent.mkdir(parents=True)
    (det / "DATA.YML").write_text(
        "path: .\ntrain: images/train\ntask: detect\nnames: [part]\n",
        encoding="utf-8",
    )
    _save(det / "images/train/a.jpg")
    (det / "labels/train").mkdir(parents=True)
    (det / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    seg = tmp_path / "seg"
    (seg / "project_config.yml").parent.mkdir(parents=True)
    (seg / "project_config.yml").write_text(
        "path: .\nval: images/val\nnames: [scratch]\n",
        encoding="utf-8",
    )
    _save(seg / "images/val/b.png")
    (seg / "labels/val").mkdir(parents=True)
    (seg / "labels/val/b.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8"
    )

    candidates = discovery.scan_datasets(tmp_path, show_progress=False)
    by_path = {candidate.path: candidate for candidate in candidates}

    assert by_path[det.resolve()].kind == "yolo_detection"
    assert by_path[det.resolve()].config_path == (det / "DATA.YML").resolve()
    assert by_path[seg.resolve()].kind == "yolo_instance"
    assert by_path[seg.resolve()].config_path == (
        seg / "project_config.yml"
    ).resolve()


def test_empty_label_files_do_not_exhaust_format_sample_budget(tmp_path: Path) -> None:
    root = tmp_path / "sparse_det"
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text(
        "path: .\ntrain: images/train\nnames: [part]\n",
        encoding="utf-8",
    )
    for index in range(25):
        _save(root / f"images/train/{index:02d}.jpg")
        label = root / f"labels/train/{index:02d}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "0 0.5 0.5 0.2 0.2\n" if index == 24 else "",
            encoding="utf-8",
        )

    candidate = discovery.inspect_config_dataset(
        root / "data.yaml", show_progress=False,
    )

    assert candidate.kind == "yolo_detection"
    assert candidate.annotation_count == 25


def test_scan_groups_separated_and_split_first_labelme_layouts(tmp_path: Path) -> None:
    separated = tmp_path / "separated"
    _save(separated / "images/a.jpg")
    (separated / "labels").mkdir(parents=True)
    (separated / "labels/a.json").write_text("{}", encoding="utf-8")

    split_first = tmp_path / "split_first"
    _save(split_first / "train/images/b.jpg")
    (split_first / "train/labels").mkdir(parents=True)
    (split_first / "train/labels/b.json").write_text("{}", encoding="utf-8")
    _save(split_first / "val/images/c.jpg")
    (split_first / "val/labels").mkdir(parents=True)
    (split_first / "val/labels/c.json").write_text("{}", encoding="utf-8")
    for path in [*separated.rglob("*"), separated]:
        os.utime(path, (1_000, 1_000))
    for path in [*split_first.rglob("*"), split_first]:
        os.utime(path, (2_000, 2_000))

    labelme = [
        candidate
        for candidate in discovery.scan_datasets(tmp_path, show_progress=False)
        if candidate.kind == "labelme"
    ]

    assert [(item.path, item.image_count, item.annotation_count) for item in labelme] == [
        (split_first.resolve(), 2, 2),
        (separated.resolve(), 1, 1),
    ]


def test_candidates_are_sorted_by_latest_dataset_file_mtime(tmp_path: Path) -> None:
    roots: list[Path] = []
    for name in ("older", "newer"):
        root = tmp_path / name
        roots.append(root)
        (root / "data.yaml").parent.mkdir(parents=True)
        (root / "data.yaml").write_text(
            "path: .\ntrain: images/train\ntask: detect\nnames: [part]\n",
            encoding="utf-8",
        )
        _save(root / "images/train/a.jpg")
        (root / "labels/train").mkdir(parents=True)
        (root / "labels/train/a.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
    for path in roots[0].rglob("*"):
        if path.is_file():
            os.utime(path, (1_000, 1_000))
    for path in roots[1].rglob("*"):
        if path.is_file():
            os.utime(path, (2_000, 2_000))

    candidates = discovery.scan_datasets(tmp_path, show_progress=False)

    assert [candidate.path for candidate in candidates] == [
        roots[1].resolve(),
        roots[0].resolve(),
    ]


def test_scan_falls_back_to_local_layout_and_valid_alias(tmp_path: Path) -> None:
    root = tmp_path / "roboflow_export"
    (root / "data.yaml").parent.mkdir(parents=True)
    (root / "data.yaml").write_text(
        "train: ../stale/train/images\n"
        "val: ../stale/valid/images\n"
        "test: ../stale/test/images\n"
        "names: [part]\n",
        encoding="utf-8",
    )
    for split, stem in (("train", "a"), ("valid", "b"), ("test", "c")):
        _save(root / f"{split}/images/{stem}.jpg")
        (root / f"{split}/labels").mkdir(parents=True)
        (root / f"{split}/labels/{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )

    candidate = next(
        item
        for item in discovery.scan_datasets(tmp_path, show_progress=False)
        if item.path == root.resolve()
    )

    assert candidate.kind == "yolo_detection"
    assert candidate.splits == ("train", "val", "test")
    assert candidate.image_count == 3
    assert candidate.annotation_count == 3


def test_stale_existing_empty_root_yields_to_local_images(tmp_path: Path) -> None:
    local = tmp_path / "datasets" / "moved_det"
    stale = tmp_path / "old_det"
    (stale / "images/train").mkdir(parents=True)
    (local / "data.yaml").parent.mkdir(parents=True)
    (local / "data.yaml").write_text(
        f"path: {stale}\ntrain: images/train\ntask: detect\nnames: [part]\n",
        encoding="utf-8",
    )
    _save(local / "images/train/a.jpg")
    (local / "labels/train").mkdir(parents=True)
    (local / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    candidate = next(
        item
        for item in discovery.scan_datasets(
            tmp_path / "datasets", show_progress=False, kinds={"yolo_detection"}
        )
        if item.config_path == (local / "data.yaml").resolve()
    )

    assert candidate.path == local.resolve()
    assert candidate.image_count == 1


def test_partially_stale_root_yields_to_more_complete_local_copy(tmp_path: Path) -> None:
    local = tmp_path / "datasets" / "moved_det"
    stale = tmp_path / "old_det"
    _save(stale / "images/train/old.jpg")
    for split in ("train", "val"):
        _save(local / f"images/{split}/{split}.jpg")
        label = local / f"labels/{split}/{split}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (local / "data.yaml").write_text(
        f"path: {stale}\ntrain: images/train\nval: images/val\ntask: detect\nnames: [part]\n",
        encoding="utf-8",
    )

    config = discovery.load_yaml(local / "data.yaml")

    assert discovery.dataset_root_from_config(local / "data.yaml", config) == local.resolve()


def test_split_samples_preserve_existing_tiff_mask_extension(tmp_path: Path) -> None:
    root = tmp_path / "semantic"
    _save(root / "images/train/a.jpg")
    _save(root / "masks/train/a.tif", "L")
    config = {
        "path": ".",
        "train": {"images": "images/train", "masks": "masks/train"},
        "task": "semantic_segmentation",
        "classes": {0: "background"},
    }

    samples = discovery.split_sample_files(root, config, "train", annotation="masks")

    assert samples[0][1] == (root / "masks/train/a.tif").resolve()


def test_unrelated_parent_data_yaml_does_not_hide_nested_dataset(
    tmp_path: Path,
) -> None:
    (tmp_path / "data.yaml").write_text(
        "application: dashboard\n",
        encoding="utf-8",
    )
    root = tmp_path / "exports" / "real_det"
    root.mkdir(parents=True)
    (root / "dataset.yml").write_text(
        "path: .\ntrain: train\ntask: detect\nnames: [part]\n",
        encoding="utf-8",
    )
    _save(root / "train/images/a.jpg")
    (root / "train/labels").mkdir(parents=True)
    (root / "train/labels/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    candidates = discovery.scan_datasets(tmp_path, show_progress=False)

    assert any(
        item.path == root.resolve() and item.kind == "yolo_detection"
        for item in candidates
    )


def test_configured_parent_does_not_hide_nested_dataset(tmp_path: Path) -> None:
    parent = tmp_path / "project"
    parent.mkdir()
    (parent / "data.yaml").write_text(
        "path: .\ntrain: images/train\ntask: segment\nnames: [parent]\n",
        encoding="utf-8",
    )
    _save(parent / "images/train/a.jpg")
    (parent / "labels/train").mkdir(parents=True)
    (parent / "labels/train/a.txt").write_text(
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
        encoding="utf-8",
    )

    nested = parent / "dataset_det"
    nested.mkdir()
    (nested / "data.yaml").write_text(
        "path: .\ntrain: train/images\ntask: det\nnames: [part]\n",
        encoding="utf-8",
    )
    _save(nested / "train/images/b.jpg")
    (nested / "train/labels").mkdir(parents=True)
    (nested / "train/labels/b.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    candidates = discovery.scan_datasets(
        tmp_path,
        show_progress=False,
        kinds={"yolo_detection"},
    )
    by_path = {candidate.path: candidate for candidate in candidates}

    assert by_path[nested.resolve()].kind == "yolo_detection"


def test_task_specific_scan_prunes_unconfigured_image_content(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _save(tmp_path / "raw_export/images/a/b/c/one.jpg")
    visited: list[Path] = []
    original_walk = discovery.os.walk

    def tracked_walk(*args, **kwargs):
        for current, dirnames, filenames in original_walk(*args, **kwargs):
            visited.append(Path(current))
            yield current, dirnames, filenames

    monkeypatch.setattr(discovery.os, "walk", tracked_walk)

    discovery.scan_datasets(
        tmp_path,
        show_progress=False,
        kinds={"yolo_detection"},
    )

    assert tmp_path / "raw_export/images" in visited
    assert tmp_path / "raw_export/images/a" not in visited


def test_scan_follows_symlinked_dataset_without_looping(tmp_path: Path) -> None:
    external = tmp_path / "mounted" / "real_det"
    external.mkdir(parents=True)
    (external / "data.yaml").write_text(
        "path: .\ntrain: images/train\ntask: detect\nnames: [part]\n",
        encoding="utf-8",
    )
    _save(external / "images/train/a.jpg")
    (external / "labels/train").mkdir(parents=True)
    (external / "labels/train/a.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )
    search_root = tmp_path / "datasets"
    search_root.mkdir()
    (search_root / "linked_det").symlink_to(external, target_is_directory=True)
    (external / "loop").symlink_to(search_root, target_is_directory=True)

    candidates = discovery.scan_datasets(
        search_root,
        show_progress=False,
        kinds={"yolo_detection"},
    )

    assert [item.path for item in candidates] == [external.resolve()]
