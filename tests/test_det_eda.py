from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from tool_lib.det_eda import generate_eda_report


def _create_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 0, 0)).save(path)


def test_generate_eda_report_exports_representative_bad_images(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_det"
    image_dir = dataset_root / "images" / "train"
    label_dir = dataset_root / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    _create_image(image_dir / "good.jpg")
    (label_dir / "good.txt").write_text("0 0.5 0.5 0.4 0.4\n", encoding="utf-8")

    _create_image(image_dir / "bad_coord.jpg")
    (label_dir / "bad_coord.txt").write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")

    _create_image(image_dir / "bad_invalid.jpg")
    (label_dir / "bad_invalid.txt").write_text("0 0.4 0.4 0.3 0.3\nbroken line\n", encoding="utf-8")

    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "names:",
                "  - defect",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "eda_out"
    generate_eda_report(source_data_path=data_yaml, output_dir=output_dir, overwrite=True)

    manifest_csv_path = output_dir / "bad_images" / "manifest.csv"
    manifest_json_path = output_dir / "bad_images" / "manifest.json"
    assert manifest_csv_path.exists()
    assert manifest_json_path.exists()

    with manifest_csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    exported_images = {row["image"] for row in rows}
    assert exported_images == {"bad_coord.jpg", "bad_invalid.jpg"}

    coord_row = next(row for row in rows if row["image"] == "bad_coord.jpg")
    invalid_row = next(row for row in rows if row["image"] == "bad_invalid.jpg")
    assert coord_row["bad_image_primary_issue"] == "coord_anomaly_lines"
    assert invalid_row["bad_image_primary_issue"] == "invalid_label_lines"
    assert Path(coord_row["export_image_path"]).exists()
    assert Path(coord_row["export_label_path"]).exists()
    assert Path(coord_row["export_meta_path"]).exists()

    with manifest_json_path.open(encoding="utf-8") as f:
        manifest_payload = json.load(f)
    assert manifest_payload["summary"]["candidate_count"] == 2
    assert manifest_payload["summary"]["exported_count"] == 2
    assert manifest_payload["summary"]["splits"]["train"]["issue_counts"]["coord_anomaly_lines"] == 1
    assert manifest_payload["summary"]["splits"]["train"]["issue_counts"]["invalid_label_lines"] == 1

    markdown_path = next(output_dir.glob("dataset_eda_*.md"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## 8. 问题样本导出" in markdown
    assert "bad_images" in markdown
