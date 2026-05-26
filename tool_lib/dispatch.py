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
    if args.tool_task == "det" and args.tool_action in {"report", "eda", "optimize"}:
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
