from pathlib import Path
import csv

import lightly_train


CLASS_ID_TO_NAME = {
    0: "Cat",
    1: "Dog",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_ID_TO_NAME.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _as_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def _as_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


if __name__ == "__main__":
    model_path = Path("out/my_experiment_cls/exported_models/exported_best.pt")
    test_dir = Path("datasets/pet_split_250/images/test")
    out_csv = Path("out/my_experiment_cls/test_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model = lightly_train.load_model(model_path.as_posix())

    image_paths = sorted(
        [
            p
            for p in test_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )
    if not image_paths:
        raise RuntimeError(f"No test images found under: {test_dir}")

    rows = []
    correct = 0

    for image_path in image_paths:
        true_label_name = image_path.parent.name
        true_label_id = CLASS_NAME_TO_ID.get(true_label_name, -1)

        pred = model.predict(image_path.as_posix())
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        if len(pred_labels) == 0:
            pred_label_id = -1
            pred_label_name = "UNKNOWN"
            pred_score = 0.0
        else:
            pred_label_id = _as_int(pred_labels[0])
            pred_label_name = CLASS_ID_TO_NAME.get(pred_label_id, str(pred_label_id))
            pred_score = _as_float(pred_scores[0])

        is_correct = int(pred_label_id == true_label_id)
        correct += is_correct
        rows.append(
            {
                "image_path": image_path.as_posix(),
                "true_label_id": true_label_id,
                "true_label_name": true_label_name,
                "pred_label_id": pred_label_id,
                "pred_label_name": pred_label_name,
                "pred_score": round(pred_score, 6),
                "is_correct": is_correct,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "true_label_id",
                "true_label_name",
                "pred_label_id",
                "pred_label_name",
                "pred_score",
                "is_correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Evaluated: {total} images")
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"CSV saved: {out_csv.as_posix()}")
