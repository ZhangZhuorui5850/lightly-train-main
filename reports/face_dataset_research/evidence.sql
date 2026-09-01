-- Audited summary rows used by the portable report.
-- Each statement is self-contained and was executed with SQLite 3 on 2026-08-27.

WITH headline_metrics(total_images, labeled_images, valid_face_boxes, under8_share) AS (
  VALUES (38203, 22094, 246476, 0.4773)
)
SELECT * FROM headline_metrics;

WITH size_bands(source, band, share, boxes, total_boxes, input_size) AS (
  VALUES
    ('DarkFace', '<4 px', 0.3169, 15971, 50393, '640×640'),
    ('DarkFace', '4–8 px', 0.3714, 18713, 50393, '640×640'),
    ('DarkFace', '8–16 px', 0.2235, 11266, 50393, '640×640'),
    ('DarkFace', '≥16 px', 0.0882, 4443, 50393, '640×640'),
    ('WIDER train', '<4 px', 0.1164, 18272, 156995, '640×640'),
    ('WIDER train', '4–8 px', 0.2932, 46027, 156995, '640×640'),
    ('WIDER train', '8–16 px', 0.2977, 46736, 156995, '640×640'),
    ('WIDER train', '≥16 px', 0.2927, 45960, 156995, '640×640')
)
SELECT * FROM size_bands;

WITH dataset_inventory(split, images, json_labels, json_boxes, missing_json, quality_note) AS (
  VALUES
    ('WIDER test', 16097, 0, 0, 16097, '仅用于官方提交，不进入本次数据集'),
    ('WIDER train', 12880, 12872, 156968, 8, '严格使用现有 JSON；清理 27 个退化框'),
    ('DarkFace', 6000, 6000, 50396, 0, '清理 3 个退化框；6,000 张全部可用'),
    ('WIDER val', 3226, 3222, 39112, 0, '4 张图清理后零有效框；清理 11 个退化框')
)
SELECT * FROM dataset_inventory;

WITH model_comparison(model, availability, params_m, coco_map, latency_ms, input, role) AS (
  VALUES
    ('picodet-s-coco', '工作区 0.14.2', 1.17, 26.7, 2.2, '416×416', '极限速度对照'),
    ('ltdetrv2-s-coco', '最新稳定版', 9.9, 50.7, 5.4, '640×640', '升级后的优先候选'),
    ('dinov3/vitt16-ltdetr-coco', '工作区 0.14.2', 10.1, 49.8, 5.4, '640×640', '现有仓库首跑'),
    ('dinov3/vitt16plus-ltdetr-coco', '工作区 0.14.2', 18.1, 52.5, 7.0, '640×640', '第二档精度对照')
)
SELECT * FROM model_comparison;
