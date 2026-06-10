"""功能分发器。

这个文件处在 launcher.py 和具体功能模块之间，职责是：
- 判断当前请求是跑现成脚本，还是跑 tool_lib 里的任务实现
- 在需要时初始化运行时依赖
- 把请求分发到 cls_tools / det_tools / seg_tools

可以把它理解成整个工具系统的路由层。
"""

from __future__ import annotations

from . import common as rt
from . import cls_tools, det_tools, seg_tools
from . import convert_tools
from . import det_optimize
from . import train_tools


def dispatch(args) -> None:
    # 训练：所有任务统一走 train_tools（不再依赖外部脚本）
    if args.tool_action == "train" and args.tool_task in {"det", "cls", "seg"}:
        train_tools.run_train(args)
        return

    if args.tool_task == "data" and args.tool_action == "convert":
        convert_tools.run_convert(args)
        return
    if args.tool_task == "det" and args.tool_action in {"report", "eda", "optimize", "review-sample"}:
        rt.import_runtime_dependencies()
    if args.tool_task == "det" and args.tool_action == "report":
        det_tools.run_report(args)
        return
    if args.tool_task == "det" and args.tool_action == "eda":
        det_tools.run_eda(args)
        return
    if args.tool_task == "det" and args.tool_action == "optimize":
        det_optimize.run_optimize(args)
        return
    if args.tool_task == "det" and args.tool_action == "review-sample":
        _dispatch_review_sample(args)
        return
    rt.import_runtime_dependencies()

    if args.tool_task == "cls":
        if args.tool_action == "infer":
            cls_tools.run_infer(args)
            return
        if args.tool_action == "eval":
            cls_tools.run_eval(args)
            return
    if args.tool_task == "det":
        if args.tool_action == "infer":
            det_tools.run_infer(args)
            return
        if args.tool_action == "export":
            det_tools.run_export(args)
            return
    if args.tool_task == "seg":
        if args.tool_action == "infer":
            seg_tools.run_infer(args)
            return
        if args.tool_action == "eval":
            seg_tools.run_eval(args)
            return
        if args.tool_action == "export":
            seg_tools.run_export(args)
            return
    raise ValueError(f"Unsupported action: {args.tool_task}/{args.tool_action}")


def _dispatch_review_sample(args) -> None:
    """分发质检抽样任务"""
    from pathlib import Path
    from .det_review_sample import run_review_pipeline_with_model

    # 如果是交互模式，先走交互流程
    if getattr(args, "interactive", False):
        from .interactive import build_interactive_review_sample_args
        args = build_interactive_review_sample_args()
        if args is None:
            return

    # 确定输出目录
    out_dir = getattr(args, "out_dir", None)
    if out_dir is None:
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path(rt.EXPERIMENT_ROOT_DIR) / f"review_subset_{timestamp}"

    # 获取参数
    data_path = Path(args.data).expanduser().resolve()
    experiment_dir = getattr(args, "experiment_dir", None)
    infer_output_dir = getattr(args, "infer_output_dir", None)
    report_path = getattr(args, "report_path", None)

    # 如果指定了experiment_dir但没有infer_output_dir，自动查找
    if experiment_dir and not infer_output_dir:
        from .det_problem_export import discover_infer_runs
        experiment_dir = Path(experiment_dir).expanduser().resolve()
        runs = discover_infer_runs(experiment_dir)
        if runs:
            infer_output_dir = runs[0].get("output_dir")
            if report_path is None:
                report_path = runs[0].get("report_path")

    # 运行质检抽样
    result_dir = run_review_pipeline_with_model(
        data_path=data_path,
        out_dir=Path(out_dir),
        experiment_dir=experiment_dir,
        infer_output_dir=infer_output_dir,
        report_path=report_path,
        # 功能开关
        enable_geometry=getattr(args, "enable_geometry", True),
        enable_model_analysis=getattr(args, "enable_model_analysis", True),
        enable_outlier_class=getattr(args, "enable_outlier_class", True),
        enable_visualization=getattr(args, "enable_visualization", False),
        # 选图参数
        k=getattr(args, "k", 3),
        alpha=getattr(args, "alpha", 3.0),
        cap=getattr(args, "cap", 10),
        max_images=getattr(args, "max_images", 500),
        min_only=getattr(args, "min_only", False),
        problems_only=getattr(args, "problems_only", False),
        # 几何检测参数
        iou_dup=getattr(args, "iou_dup", 0.9),
        iou_conflict=getattr(args, "iou_conflict", 0.5),
        min_box_px=getattr(args, "min_box_px", 4),
        dense_top_n=getattr(args, "dense_top_n", None),
        # 模型分析参数
        match_iou_threshold=getattr(args, "match_iou_threshold", 0.5),
        low_conf_threshold=getattr(args, "low_conf_threshold", 0.3),
        outlier_class_ap=getattr(args, "outlier_class_ap", 0.1),
        outlier_class_gt=getattr(args, "outlier_class_gt", 5),
        # 其他
        groups_path=getattr(args, "groups_path", None),
        groups_section=getattr(args, "groups_section", "coco80"),
        cache_path=getattr(args, "cache_path", None),
        seed=getattr(args, "seed", 42),
        embed_images=getattr(args, "embed_images", True),
    )

    print(f"\n{'=' * 63}")
    print(f"  完成！")
    print(f"{'=' * 63}")
    print(f"\n  输出目录: {result_dir}")
    print(f"\n  ── 下一步 ──")
    print(f"  1. 将 review_subset/ 拷贝到本地")
    print(f"  2. 用 X-AnyLabeling 打开 images/ 文件夹")
    print(f"  3. 按 description 字段排序，优先复核高风险 flag")
    print(f"  4. 复核完毕后可回灌训练")
    print(f"\n  详细报告: {result_dir / 'qc_report.md'}")
