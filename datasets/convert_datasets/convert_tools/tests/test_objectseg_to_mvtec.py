# datasets/convert_datasets/convert_tools/tests/test_objectseg_to_mvtec.py
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import objectseg_to_mvtec as om  # noqa: E402

NAMES = ["锈蚀", "裂纹", "污迹"]  # cls 0,1,2


def _tri(cx: float, cy: float, r: float = 0.08):
    """一个小三角多边形(归一化坐标)。"""
    return [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]


def _write_label(p: Path, entries):
    """entries: list[(cls, [(x,y),...])];  空 list = good 图(空标签)。"""
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, pts in entries:
        coords = " ".join(f"{x:.4f} {y:.4f}" for x, y in pts)
        lines.append(f"{cls} {coords}")
    p.write_text("\n".join(lines))


def make_src(root: Path, with_images: bool = True) -> Path:
    """最小 YOLO-seg 源:data.yaml + labels/,默认也带 images/(转换器按文件名从源取原图)。

    with_images=False 时不建 images/,给需要自己控制哪些图存在的用例(如缺图跳过)用。
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(
        yaml.safe_dump({"names": NAMES}, allow_unicode=True), encoding="utf-8"
    )
    labels = [("train", "a", [(0, _tri(0.3, 0.3))]),                                  # 锈蚀
              ("train", "b", [(0, _tri(0.3, 0.3)), (1, _tri(0.7, 0.7))]),             # 锈蚀+裂纹
              ("val", "c", [(2, _tri(0.5, 0.5))]),                                    # 污迹
              ("train", "d", [])]                                                     # 空/good
    for split, stem, entries in labels:
        _write_label(root / "labels" / split / f"{stem}.txt", entries)
        if with_images:
            img_p = root / "images" / split / f"{stem}.jpg"
            img_p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_p), np.full((64, 64, 3), 128, np.uint8))
    return root


def _img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.full((64, 64, 3), 128, np.uint8))


def make_staging(root: Path) -> Path:
    """物体 管道: a,b ; 物体 阀门: c,d。"""
    for stem in ("a", "b"):
        _img(root / "管道" / f"{stem}.jpg")
    for stem in ("c", "d"):
        _img(root / "阀门" / f"{stem}.jpg")
    return root


def test_build_label_index_maps_stem_to_split(tmp_path):
    src = make_src(tmp_path / "src")
    index = om.build_label_index(src)
    assert set(index) == {"a", "b", "c", "d"}
    assert index["a"][1] == "train"
    assert index["c"][1] == "val"
    assert index["a"][0].name == "a.txt"


def test_build_label_index_raises_on_duplicate_stem(tmp_path):
    src = make_src(tmp_path / "src")
    _write_label(src / "labels" / "test" / "a.txt", [(0, _tri(0.5, 0.5))])  # 与 train/a 重名
    with pytest.raises(om.DuplicateStemError):
        om.build_label_index(src)


def test_scan_staging_lists_objects_and_images(tmp_path):
    staging = make_staging(tmp_path / "staging")
    objects, conflicts = om.scan_staging(staging)
    assert set(objects) == {"管道", "阀门"}
    assert sorted(p.stem for p in objects["管道"]) == ["a", "b"]
    assert conflicts == {}


def test_scan_staging_flags_stem_in_two_objects(tmp_path):
    staging = make_staging(tmp_path / "staging")
    _img(staging / "阀门" / "a.jpg")  # a 同时在 管道 和 阀门
    objects, conflicts = om.scan_staging(staging)
    assert "a" in conflicts
    assert set(conflicts["a"]) == {"管道", "阀门"}


def test_scan_staging_same_stem_diff_ext_in_one_object_is_not_conflict(tmp_path):
    staging = make_staging(tmp_path / "staging")
    _img(staging / "管道" / "a.png")  # 管道 里已有 a.jpg,再加同名不同扩展的 a.png
    _objects, conflicts = om.scan_staging(staging)
    assert "a" not in conflicts  # 同一物体内不算跨物体冲突


def _run(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    result = om.convert(staging, src, out, clean=True, verbose=False)
    return out, result


def test_convert_builds_object_category_structure(tmp_path):
    out, _ = _run(tmp_path)
    assert (out / "管道" / "test" / "锈蚀" / "a.png").exists()
    assert (out / "管道" / "ground_truth" / "锈蚀" / "a_mask.png").exists()
    assert (out / "管道" / "train" / "good").is_dir()
    assert (out / "阀门" / "train" / "good").is_dir()


def test_convert_duplicates_multidefect_image_into_each_defect(tmp_path):
    out, _ = _run(tmp_path)
    assert (out / "管道" / "test" / "锈蚀" / "b.png").exists()
    assert (out / "管道" / "test" / "裂纹" / "b.png").exists()
    assert (out / "管道" / "ground_truth" / "锈蚀" / "b_mask.png").exists()
    assert (out / "管道" / "ground_truth" / "裂纹" / "b_mask.png").exists()


def test_convert_mask_is_split_per_defect(tmp_path):
    out, _ = _run(tmp_path)
    m_rust = cv2.imread(str(out / "管道" / "ground_truth" / "锈蚀" / "b_mask.png"), cv2.IMREAD_GRAYSCALE)
    m_crack = cv2.imread(str(out / "管道" / "ground_truth" / "裂纹" / "b_mask.png"), cv2.IMREAD_GRAYSCALE)
    assert set(np.unique(m_rust)).issubset({0, 255})
    assert m_rust[19, 19] == 255 and m_rust[45, 45] == 0
    assert m_crack[45, 45] == 255 and m_crack[19, 19] == 0


def test_convert_empty_label_goes_to_test_good(tmp_path):
    out, _ = _run(tmp_path)
    assert (out / "阀门" / "test" / "good" / "d.png").exists()


def test_convert_writes_manifest_csv(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    om.convert(staging, src, out, clean=True, verbose=False)

    manifest = out / "object_manifest.csv"
    assert manifest.exists()
    import csv as _csv
    with manifest.open(encoding="utf-8") as stream:
        rows = list(_csv.DictReader(stream))
    by_stem = {r["stem"]: r for r in rows}
    assert by_stem["b"]["object"] == "管道"
    assert by_stem["b"]["defects"] == "锈蚀;裂纹"
    assert by_stem["c"]["orig_split"] == "val"
    assert by_stem["a"]["src_label"] == "labels/train/a.txt"


def test_convert_reads_image_from_source_not_staging(tmp_path):
    """staging 里放的图只用来选文件名;输出的图必须来自源数据集的原图。"""
    from PIL import Image
    src = make_src(tmp_path / "src")  # 源 a 是 64x64
    # staging/管道/a.jpg 故意放一张不同尺寸的“假图”(100x100),模拟放了预览图/占位
    staging = tmp_path / "staging"
    decoy = staging / "管道" / "a.jpg"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(decoy), np.full((100, 100, 3), 200, np.uint8))
    out = tmp_path / "out"
    om.convert(staging, src, out, clean=True, verbose=False)

    produced = out / "管道" / "test" / "锈蚀" / "a.png"
    assert produced.exists()
    # 若用了 staging 的 decoy 会是 100x100;来自源则是 64x64
    with Image.open(produced) as image:
        assert image.size == (64, 64)


def test_cli_runs_end_to_end(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    import subprocess
    script = TOOLS / "objectseg_to_mvtec.py"
    r = subprocess.run(
        [sys.executable, str(script),
         "--staging", str(staging), "--src", str(src), "--out", str(out), "--clean"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "管道" / "test" / "锈蚀" / "a.png").exists()
    assert (out / "object_manifest.csv").exists()


def test_convert_refuses_nonempty_out_without_clean(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old")  # 预先放一个陈旧文件
    with pytest.raises(SystemExit):
        om.convert(staging, src, out, clean=False, verbose=False)


def test_convert_clean_overwrites_existing_out(tmp_path):
    src = make_src(tmp_path / "src")
    staging = make_staging(tmp_path / "staging")
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("old")
    om.convert(staging, src, out, clean=True, verbose=False)  # 不应报错
    assert not (out / "stale.txt").exists()  # 陈旧文件被清掉
    assert (out / "object_manifest.csv").exists()


def test_convert_raises_on_out_of_range_class_id(tmp_path):
    src = make_src(tmp_path / "src")
    # 追加一个越界类别 id 的标签(names 只有 3 类:0,1,2)+ 对应源图
    _write_label(src / "labels" / "train" / "bad.txt", [(9, _tri(0.5, 0.5))])
    _img(src / "images" / "train" / "bad.jpg")   # 源里要有原图,越界检查才会被触及
    staging = tmp_path / "staging"
    _img(staging / "管道" / "bad.jpg")
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        om.convert(staging, src, out, clean=True, verbose=False)


def test_browse_writes_annotated_previews_and_new_csv_columns(tmp_path):
    import csv as _csv
    from PIL import Image
    import seg_sample_browse as sb

    src = make_src(tmp_path / "src")  # a=锈蚀, b=锈蚀+裂纹, c=污迹, d=空
    for split, stem in [("train", "a"), ("train", "b"), ("val", "c"), ("train", "d")]:
        _img(src / "images" / split / f"{stem}.jpg")
    out = tmp_path / "browse"
    n = sb.browse(src, out, limit=10)
    assert n == 4

    assert (out / "b.png").exists()
    with Image.open(src / "images" / "train" / "b.jpg") as source_image:
        base_w = source_image.width
    with Image.open(out / "b.png") as preview_image:
        assert preview_image.width > base_w

    with (out / "sample_index.csv").open(encoding="utf-8") as stream:
        rows = {row["stem"]: row for row in _csv.DictReader(stream)}
    assert rows["b"]["defects"] == "锈蚀;裂纹"
    assert rows["b"]["n_defect_classes"] == "2"
    assert rows["b"]["n_polygons"] == "2"
    assert rows["d"]["n_defect_classes"] == "0"  # 空标签


def test_browse_skips_missing_image(tmp_path):
    import seg_sample_browse as sb
    src = make_src(tmp_path / "src", with_images=False)
    _img(src / "images" / "train" / "a.jpg")  # 只给 a 配图
    out = tmp_path / "browse"
    n = sb.browse(src, out, limit=10)
    assert n == 1
    assert (out / "a.png").exists()


def test_class_color_deterministic_and_distinct():
    import seg_sample_browse as sb
    assert sb.class_color(0) == sb.class_color(0)          # 同类同色
    assert sb.class_color(0) != sb.class_color(1)          # 不同类不同色
    c = sb.class_color(2)
    assert isinstance(c, tuple) and len(c) == 3 and all(0 <= v <= 255 for v in c)


def test_class_name_handles_out_of_range():
    import seg_sample_browse as sb
    assert sb._class_name(["锈蚀", "裂纹"], 1) == "裂纹"
    assert sb._class_name(["锈蚀", "裂纹"], 9) == "未知类别9"


def test_render_preview_appends_info_panel(tmp_path):
    import numpy as np
    import seg_sample_browse as sb
    from PIL import Image

    img_path = tmp_path / "x.jpg"
    _img(img_path)  # 64x64 灰图
    polys = [
        (0, np.array([[0.2, 0.2], [0.4, 0.2], [0.3, 0.4]])),   # 锈蚀
        (1, np.array([[0.6, 0.6], [0.8, 0.6], [0.7, 0.8]])),   # 裂纹
    ]
    canvas = sb.render_preview(img_path, polys, ["锈蚀", "裂纹", "污迹"], "train")
    with Image.open(img_path) as image:
        base_w = image.width
    assert canvas.height == 64                 # 高度=原图高
    assert canvas.width > base_w               # 右边拼了信息栏
    assert canvas.mode == "RGB"


def test_wizard_run_generate_forwards(monkeypatch, tmp_path):
    import objseg_wizard as wiz
    calls = {}

    def fake_browse(src, out, limit=500, font_path=None):
        calls.update(src=src, out=out, limit=limit, font_path=font_path)
        return 7

    monkeypatch.setattr(wiz.seg_sample_browse, "browse", fake_browse)
    n = wiz.run_generate(tmp_path / "s", tmp_path / "o", limit=3)
    assert n == 7
    assert calls["limit"] == 3 and calls["src"] == tmp_path / "s"


def test_wizard_run_build_forwards(monkeypatch, tmp_path):
    import objseg_wizard as wiz
    calls = {}

    def fake_convert(staging, src, out, clean=False, verbose=True):
        calls.update(staging=staging, src=src, out=out, clean=clean)
        return {"stats": {}, "missing": [], "conflicts": {}, "manifest": []}

    monkeypatch.setattr(wiz.objectseg_to_mvtec, "convert", fake_convert)
    result = wiz.run_build(tmp_path / "stg", tmp_path / "src", tmp_path / "out", clean=True)
    assert calls["clean"] is True and calls["staging"] == tmp_path / "stg"
    assert "manifest" in result


@pytest.mark.parametrize("names", [["good", "good", "other"], ["Ｄｅｆｅｃｔ", "defect", "other"]])
def test_conflicting_defect_names_keep_individual_masks(tmp_path, names):
    src = make_src(tmp_path / "src")
    (src / "data.yaml").write_text(yaml.safe_dump({"names": names}, allow_unicode=True))
    staging = make_staging(tmp_path / "staging")
    output = tmp_path / "output"
    om.convert(staging, src, output, verbose=False)
    masks = list((output / "管道" / "ground_truth").glob("*/b_mask.png"))
    assert len(masks) == 2
    assert len({path.parent.name.casefold() for path in masks}) == 2
    assert all(path.parent.name.casefold() != "good" for path in masks)
    assert not np.array_equal(cv2.imread(str(masks[0]), 0), cv2.imread(str(masks[1]), 0))
    assert all((output / "管道" / "test" / p.parent.name / "b.png").is_file() for p in masks)


def test_normalized_object_names_keep_separate_categories(tmp_path):
    src = make_src(tmp_path / "src")
    staging = tmp_path / "staging"
    _img(staging / "Ａ" / "a.jpg")
    _img(staging / "A" / "c.jpg")
    output = tmp_path / "output"
    om.convert(staging, src, output, verbose=False)
    categories = [p for p in output.iterdir() if p.is_dir()]
    assert len(categories) == 2
    assert len({p.name.casefold() for p in categories}) == 2
