"""分类任务工具。

当前这个文件负责 cls 的非训练功能，主要包括：
- infer: 对单张图或图片目录做分类推理
- eval: 对带类别文件夹结构的测试集做评估

运行流程大致是：
1. 解析 checkpoint
2. 加载分类模型
3. 收集输入图片
4. 逐张预测
5. 输出 csv / json / summary
"""

from __future__ import annotations

import json
from pathlib import Path

from . import common as rt
from .progress import track


def run_infer(args) -> None:
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    rt.prepare_output_dir(args.output_dir, True)
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    class_names = rt.get_model_class_names(model)

    image_paths = rt.list_image_files(args.image if args.image is not None else args.image_dir)
    rt.ensure_image_samples([rt.ImageSample(image_path=p, relative_path=Path(p.name)) for p in image_paths])

    rows: list[dict[str, object]] = []
    for image_path in track(image_paths, label="cls/infer 推理", unit="img"):
        pred = model.predict(str(image_path), topk=args.topk, threshold=args.threshold)
        labels = pred["labels"]
        scores = pred["scores"]
        if len(labels) == 0:
            pred_label_id = -1
            pred_label_name = "UNKNOWN"
            pred_score = 0.0
        else:
            pred_label_id = rt.scalar_int(labels[0])
            pred_label_name = class_names.get(pred_label_id, str(pred_label_id))
            pred_score = rt.scalar_float(scores[0])
        rows.append(
            {
                "image_path": str(image_path),
                "pred_label_id": pred_label_id,
                "pred_label_name": pred_label_name,
                "pred_score": round(pred_score, 6),
            }
        )

    json_path = args.output_dir / "cls_infer_results.json"
    csv_path = args.output_dir / "cls_infer_results.csv"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rt.save_records_csv(csv_path, rows, ["image_path", "pred_label_id", "pred_label_name", "pred_score"])
    print(f"Inferred {len(rows)} images")
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to: {csv_path}")


def run_eval(args) -> None:
    checkpoint_path = rt.resolve_checkpoint_path(args.checkpoint, args.experiment_dir)
    rt.prepare_output_dir(args.output_dir, True)
    model = rt.lightly_train.load_model(model=checkpoint_path, device=rt.resolve_device(args.device))
    model.eval()
    class_names = rt.get_model_class_names(model)
    name_to_id = {name: idx for idx, name in class_names.items()}

    image_paths = rt.list_image_files(args.test_dir)
    if not image_paths:
        raise ValueError("测试目录中没有找到图片。")

    rows: list[dict[str, object]] = []
    correct = 0
    for image_path in track(image_paths, label="cls/eval 评估", unit="img"):
        true_label_name = image_path.parent.name
        true_label_id = name_to_id.get(true_label_name, -1)
        pred = model.predict(str(image_path), topk=args.topk, threshold=args.threshold)
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        if len(pred_labels) == 0:
            pred_label_id = -1
            pred_label_name = "UNKNOWN"
            pred_score = 0.0
        else:
            pred_label_id = rt.scalar_int(pred_labels[0])
            pred_label_name = class_names.get(pred_label_id, str(pred_label_id))
            pred_score = rt.scalar_float(pred_scores[0])
        is_correct = int(pred_label_id == true_label_id)
        correct += is_correct
        rows.append(
            {
                "image_path": str(image_path),
                "true_label_id": true_label_id,
                "true_label_name": true_label_name,
                "pred_label_id": pred_label_id,
                "pred_label_name": pred_label_name,
                "pred_score": round(pred_score, 6),
                "is_correct": is_correct,
            }
        )

    accuracy = correct / len(rows) if rows else 0.0
    csv_path = args.output_dir / "cls_eval_results.csv"
    summary_path = args.output_dir / "cls_eval_summary.json"
    rt.save_records_csv(
        csv_path,
        rows,
        [
            "image_path",
            "true_label_id",
            "true_label_name",
            "pred_label_id",
            "pred_label_name",
            "pred_score",
            "is_correct",
        ],
    )
    summary = {"checkpoint": str(checkpoint_path), "num_images": len(rows), "correct": correct, "accuracy": accuracy}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Evaluated: {len(rows)} images")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
