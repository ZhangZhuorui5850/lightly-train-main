from __future__ import annotations

import sys

from tool_lib import common as rt
from tool_lib.dispatch import dispatch
from tool_lib.interactive import build_interactive_args, parse_cli_args

# ============================================================
# 统一配置区
# 常改的路径和脚本都放这里；底层模块会自动读取这些配置
# ============================================================
# 公共配置：项目通用输出目录
COMMON_SETTINGS = {
    "out_dir": "out",
    "experiment_root_dir": "out",
    "infer_output_root_dir": "out",
    "eda_output_root_dir": "out/EDA",
    "all_report_root_dir": "out/all_report",
}

# 检测配置：保留在一个地方，通过注释和空行分组
DET_SETTINGS = {
    # 基础配置
    # 这两个最常改：改数据集或实验目录时，下面大部分 det 默认路径会自动跟着变
    "det_dataset_dir": "datasets/wuwanPic_dataset/dataset_det",
    # 留空 / None / "auto" 时，会自动从 out/ 下检索最近的 det 实验目录
    "det_experiment_dir": None,
    "det_default_split": "test",

    # 阈值配置
    # det_score_threshold:
    #   推理时的置信度阈值。默认 0.3 更偏向少漏检；误检多时可再调高。
    # det_report_iou_threshold:
    #   评估 test_report.json 时，预测框和标注框的 IoU 命中阈值。
    #   指标太严格就调低，例如 0.5 -> 0.4。
    "det_score_threshold": 0.3,
    "det_report_iou_threshold": 0.5,

    # 可选覆盖项
    # 如果你不填，系统会自动推导：
    # det_data_yaml          -> <det_dataset_dir>/data.yaml
    # det_infer_image_dir    -> <det_dataset_dir>/images/<det_default_split>
    # det_infer_output_dir   -> <experiment_dir>/infer-<dataset>-<split>
    # det_eval_output_dir    -> 同 det_infer_output_dir
    # det_eval_report_path   -> <det_infer_output_dir>/<run_name>-test_report.json
    # det_export_report_json -> 同 det_eval_report_path
    #
    # 只有你确实想和自动规则不一样时，再单独打开某一项覆盖。
    # "det_data_yaml": "datasets/wuwanPic_dataset/dataset_det/data.yaml",
    # "det_infer_image_dir": "datasets/wuwanPic_dataset/dataset_det/images/test",
    # "det_infer_output_dir": "out/2026-04-14/NEU_train/infer-neu-test",
    # "det_eval_output_dir": "out/2026-04-14/NEU_train/infer-neu-test",
    # "det_eval_report_path": "out/2026-04-14/NEU_train/infer-neu-test/infer-neu-test-test_report.json",
    # "det_export_report_json": "out/2026-04-14/NEU_train/infer-neu-test/infer-neu-test-test_report.json",

    # export 默认配置
    # det_export_source_data 不填时，默认跟 det_data_yaml 一致
    # det_export_good_class_threshold:
    #   类别 AP 参考阈值。
    #   自动模式下它进入分析报告；严格模式下它参与删类。
    # det_export_min_class_images:
    #   一个类别至少要有多少张图才参与导出。
    #   自动模式下会结合 8:1:1 重划分要求做保底推导，默认会倾向保留至少能覆盖 train/val/test 的类别。
    # det_export_min_class_boxes:
    #   一个类别至少要有多少个框才参与导出。
    #   填 0 时按 det_export_balance_ratio 自动推导。
    #   高密度图片过滤后还会再复核一次，避免某个类别最后只剩几个框。
    # det_export_target_images_per_class:
    #   train 导出时，每个类别尽量靠近这个图数。
    #   填 0 时自动取可用类别图数的中位数。
    # det_export_target_total_images:
    #   导出后总图片数目标。填 0 表示由筛选结果自动决定。
    #   填入后，系统会优先围绕这个总量预算自动推导类别下限、框密度阈值和每类目标框数。
    # det_export_split_ratio:
    #   导出后重新划分的 train:val:test 比例。
    #   交互模式固定使用 8:1:1；用户只需要输入总图数，其余阈值优先走自动分析。
    # det_export_target_boxes_per_class:
    #   train 导出时，每个类别尽量靠近这个框数。
    #   填 0 时按 det_export_balance_ratio 自动推导。
    # det_export_balance_ratio:
    #   导出后 train 中最多和最少类别的框数比例上限。
    # det_export_auto_balance:
    #   导出前先做标签分布分析，并自动推导保留阈值和每类框数上限。
    # det_export_auto_relax_class_threshold:
    #   开启后，AP 阈值进入参考分析，类别保留主要看标签数量和平衡规则。
    #   关闭后，AP 阈值进入严格删类。
    # det_export_max_boxes_per_image:
    #   单张图保留类别的总框数上限。填 0 时自动模式会按总图数预算和框密度分布推导。
    # det_export_max_boxes_per_class_per_image:
    #   单张图里同一类别的框数上限。填 0 时自动模式会按总图数预算和框密度分布推导。
    # det_export_box_density_penalty:
    #   选图时对高框密度图片的惩罚强度，越大越偏向少框图片。
    #   自动模式下会以这个值为基础，按筛选压力向上调整。
    # "det_export_source_data": "datasets/wuwanPic_dataset/dataset_det/data.yaml",
    "det_export_target_total_images": 0,
    "det_export_good_class_threshold": 0.0,
    "det_export_auto_balance": True,
    "det_export_auto_relax_class_threshold": True,
    "det_export_balance_ratio": 0.0,
    "det_export_min_class_images": 0,
    "det_export_min_class_boxes": 0,
    "det_export_target_images_per_class": 0,
    "det_export_split_ratio": "8:1:1",
    "det_export_target_boxes_per_class": 0,
    "det_export_max_boxes_per_image": 0,
    "det_export_max_boxes_per_class_per_image": 0,
    "det_export_box_density_penalty": 0.0,
    "det_export_suffix": "_A",
}

# 分类配置：后面如果要统一 cls 默认路径，就加在这里
CLS_SETTINGS = {
    # 分类推理/评估默认阈值。预测太保守时可以调低。
    "cls_threshold": 0.5,
}

# 分割配置：和 det 一样，常改的路径和导出参数都放在这里
SEG_SETTINGS = {
    # 基础配置
    # 改 seg 数据集时，下面 seg_export_* 的源数据会自动跟着变
    "seg_dataset_dir": "datasets/neu_dataset/dataset_seg",

    # 阈值配置
    # seg_threshold:
    #   推理/评估默认阈值。掩码太少就调低，噪声太多就调高。
    "seg_threshold": 0.8,

    # export 默认配置（参数语义对齐 det，关键词把"框"换成"实例"）
    # seg_export_source_data 不填时，默认 <seg_dataset_dir>/data.yaml
    # seg_export_report_json 不填或填 "auto" 时，自动检索最近的 seg_eval_summary.json
    # seg_export_good_class_threshold:
    #   类别 AP 参考阈值。自动模式下进入分析报告；严格模式下参与删类。
    # seg_export_min_class_images:
    #   一个类别至少要有多少张图才参与导出。
    # seg_export_min_class_instances:
    #   一个类别至少要有多少个实例（多少条 polygon）才参与导出。
    #   填 0 时按 seg_export_balance_ratio 自动推导。
    # seg_export_target_images_per_class:
    #   train 导出时，每个类别尽量靠近这个图数。填 0 自动取中位数。
    # seg_export_target_total_images:
    #   导出后总图片数目标。填 0 表示由筛选结果自动决定；
    #   填入后系统会围绕这个总量预算自动推导类别下限、实例密度阈值和每类目标实例数。
    # seg_export_split_ratio:
    #   导出后重新划分的 train:val:test 比例（交互模式固定 8:1:1）。
    # seg_export_target_instances_per_class:
    #   train 导出时每个类别尽量靠近这个实例数。填 0 按 balance_ratio 自动推导。
    # seg_export_balance_ratio:
    #   导出后 train 中最多和最少类别的实例数比例上限。
    # seg_export_auto_balance:
    #   导出前先做标签分布分析，并自动推导保留阈值和每类实例数上限。
    # seg_export_auto_relax_class_threshold:
    #   开启后 AP 阈值仅作参考；关闭后 AP 阈值进入严格删类。
    # seg_export_max_instances_per_image:
    #   单张图保留类别的实例总数上限。填 0 自动按总图数预算和密度分布推导。
    # seg_export_max_instances_per_class_per_image:
    #   单张图里同一类别的实例数上限。填 0 自动推导。
    # seg_export_instance_density_penalty:
    #   选图时对高实例密度图片的惩罚强度，越大越偏向少实例图片。
    #   自动模式下会以这个值为基础，按筛选压力向上调整。
    # "seg_export_source_data": "datasets/neu_dataset/dataset_seg/data.yaml",
    # "seg_export_report_json": "auto",
    "seg_export_target_total_images": 0,
    "seg_export_good_class_threshold": 0.0,
    "seg_export_auto_balance": True,
    "seg_export_auto_relax_class_threshold": True,
    "seg_export_balance_ratio": 0.0,
    "seg_export_min_class_images": 0,
    "seg_export_min_class_instances": 0,
    "seg_export_target_images_per_class": 0,
    "seg_export_split_ratio": "8:1:1",
    "seg_export_target_instances_per_class": 0,
    "seg_export_max_instances_per_image": 0,
    "seg_export_max_instances_per_class_per_image": 0,
    "seg_export_instance_density_penalty": 0.0,
    "seg_export_suffix": "_A",
}

def build_user_settings() -> dict[str, object]:
    # 把分组配置合并成底层模块统一使用的一份 settings
    settings: dict[str, object] = {}
    settings.update(COMMON_SETTINGS)
    settings.update(CLS_SETTINGS)
    settings.update(DET_SETTINGS)
    settings.update(SEG_SETTINGS)
    return settings


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    rt.apply_user_settings(build_user_settings())
    args = parse_cli_args(argv) if argv else build_interactive_args()
    if args is None:
        return
    dispatch(args)


if __name__ == "__main__":
    main()
