#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集转换工具的统一启动台 (single entry point).

只需记住这一个脚本。它本身不做转换，只负责：
  1) 列出所有转换/整理工具(按流水线阶段分组，每个带一句话说明)；
  2) 交互式询问你要做什么，然后把控制权交给对应工具。

用法:
    python convert.py                 # 交互式菜单(记不住用哪个就跑这个)
    python convert.py <command> ...   # 直接运行某工具，后面参数原样转发
    python convert.py <command> -h    # 看该工具自己的参数
    python convert.py list            # 只打印菜单
    python convert.py doctor          # 检查注册表、脚本与 dry-run 契约

交互菜单支持 more 展开全部工具、b 返回菜单、q 退出。

底层工具都在 ./convert_tools/ 下，每个也能单独运行。
"""
from __future__ import annotations

import difflib
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS_DIR = HERE / "convert_tools"


@dataclass(frozen=True)
class Tool:
    """一个可分发的工具的登记项。"""

    script: str          # convert_tools/ 下的文件名
    desc: str            # 一句话说明(菜单里显示)
    stage: str           # 流水线阶段(菜单分组)
    usage: str           # 参数提示(选中后显示)
    interactive: bool    # True=直接回车即进入自身交互流程
    primary: bool = True  # False=归到"更多单步/辅助工具"，默认不展开
    read_only: bool = False
    atomic_publish: bool = False


# command -> Tool。改这里就能增删工具。
# 主流程(primary)显示在默认菜单；其余是内部分步或辅助工具(primary=False)，
# 已被主流程包含或很少直接用，默认折叠，`convert.py more` 才展开。
COMMANDS: dict[str, Tool] = {
    # ---- 主流程 ----
    "datasets": Tool(
        "dataset_detector.py",
        "快速检测并诊断 Cls/Det/Seg/Semantic/LabelMe/MVTec 数据集",
        "自动发现",
        "回车=扫描 datasets/；或 --task cls|det|seg|semantic；--diagnose <data.yaml>",
        interactive=True,
        read_only=True,
    ),
    "auto": Tool(
        "conversion_wizard.py",
        "先选转换功能，再自动检索兼容数据集并统一输出命名",
        "自动发现",
        "回车=扫描仓库 datasets/；可重复 --datasets <路径>；--list 只查看",
        interactive=True,
        atomic_publish=True,
    ),
    "yolo2semantic": Tool(
        "yoloseg_to_semantic.py",
        "YOLO Seg polygon txt → train_seg.py 使用的 PNG 语义掩码",
        "转语义分割",
        "回车=自动扫描并选择；或 --src <数据集> --out <输出目录>",
        interactive=True,
    ),
    "semantic2xany": Tool(
        "semantic_to_xanylabeling.py",
        "PNG 语义 mask → X-AnyLabeling MASK 导入映射 JSON",
        "转语义分割",
        "回车=自动扫描并选择；或 --src <dataset_semantic 或 data.yaml>",
        interactive=True,
    ),
    "merge-datasets": Tool(
        "dataset_merger.py",
        "选格式与基准 data.yaml，预览整套类别映射后一次性合并",
        "合并数据集",
        "回车=完整交互；或 --format <格式>、重复 --src 并用 --reference-source <序号>",
        interactive=True,
        atomic_publish=True,
    ),
    "edit-classes": Tool(
        "class_editor.py",
        "检查 PNG 语义/YOLO Det/YOLO Seg，交互删除、合并、重命名并重排 ID",
        "处理数据类别",
        "回车=扫描并选择；或 --src <数据集> [--out <输出目录>]",
        interactive=True,
        atomic_publish=True,
    ),
    "face-wider-prepare": Tool(
        "face_wider_prepare.py",
        "WIDER Face YOLO → 定向 Copy-Paste / 训练切片 / 组合训练集",
        "处理数据类别",
        "需 --mode copy-paste|tile|combined --out <输出目录>；可先用 --dry-run",
        interactive=False,
        atomic_publish=True,
    ),
    "edit-semantic": Tool(
        "semantic_class_editor.py",
        "仅编辑 PNG 语义数据集类别（兼容旧入口）",
        "更多",
        "回车=扫描并选择；或 --src <数据集> [--out <输出目录>]",
        interactive=True,
        primary=False,
    ),
    "oneclick": Tool(
        "one_click_convert.py",
        "整理散图/LabelMe → YOLO det/cls/seg 一步到位(内部已含 sync + labelme2yolo)",
        "整理 + 转 YOLO",
        "需 <源目录...> --output-root <输出目录>；可先用 -h 查看参数",
        interactive=False,
    ),
    "generated2seg": Tool(
        "generated2seg_interactive.py",
        "图生图返回的 <物体>/<缺陷>/{image,fg掩码} → YOLO Seg",
        "整理 + 转 YOLO",
        "回车=自动扫描并选择；或 --src <路径> [--out <路径>] [--yes]",
        interactive=True,
        atomic_publish=True,
    ),
    "mvtec2yolo": Tool(
        "mvtec_to_yolo.py",
        "标准 MVTec AD → YOLO Seg/Det（默认同时生成）",
        "整理 + 转 YOLO",
        "回车=扫描 datasets/ 并选择；或 --src <目录> [--out <目录>] [--task segment|detect|both]",
        interactive=True,
    ),
    "to-mvtec": Tool(
        "seg2mvtec_interactive.py",
        "缺陷版(legacy):扫描 *seg 数据集，每个缺陷=一个 category，交互式转 MVTec AD",
        "转 MVTec AD",
        "回车=交互扫描并选择；或加 --all --yes 转全部",
        interactive=True,
        atomic_publish=True,
    ),
    "to-mvtec-obj": Tool(
        "objectseg_to_mvtec.py",
        "物体版:按人工分好的物体文件夹 + 源seg 生成 MVTec(category=物体，内含各缺陷)",
        "转 MVTec AD",
        "需 --staging <物体文件夹根> --src <seg源> --out <输出> [--clean]",
        interactive=False,
    ),
    "obj-wizard": Tool(
        "objseg_wizard.py",
        "物体版向导(交互):分两步——①生成标注预览挑数据 ②分好后转 MVTec",
        "转 MVTec AD",
        "回车进入向导,按提示选第1步或第2步",
        interactive=True,
        atomic_publish=True,
    ),
    "det2seg": Tool(
        "mirror_det_subset_to_seg.py",
        "把挑出的 det 子集按图片名镜像成对应的 seg 子集",
        "子集对齐",
        "回车=扫描 datasets/ 并按相关性排序供选择；或 --det-subset/--seg-source 直接指定",
        interactive=True,
    ),
    "rebalance-splits": Tool(
        "rebalance_yolo_splits.py",
        "单个 labels/masks 数据集：汇总已有 train/val/test 并均衡重划分",
        "子集对齐",
        "回车=扫描并选择；或 --src <数据集> --ratios 0.8 0.1 0.1 --out <输出目录>",
        interactive=True,
    ),
    "balanced-test": Tool(
        "balanced_test_selector.py",
        "Det/Seg 长尾诊断 + 类别门槛 + split 可选的固定数量测试集挑选",
        "子集对齐",
        "回车=先选 Det/Seg 再按修改时间选数据集；或 --source <候选池> --task detect --min-class-images 80 --count 200",
        interactive=True,
    ),
    "merge-yolo": Tool(
        "yolo_dataset_merger.py",
        "仅合并 YOLO Det/Polygon Seg 数据集（兼容旧入口）",
        "更多",
        "重复 --src <数据集> 至少两次，或回车扫描多选",
        interactive=True,
        primary=False,
    ),
    # ---- 更多：单步 / 特殊输入 / 辅助(默认折叠) ----
    "coco2sync": Tool(
        "coco_to_synced.py",
        "COCO 2017 标注 → train/val/test 结构(特殊输入，之后接 oneclick)",
        "更多",
        "--help 看参数；需 COCO 根目录",
        interactive=False, primary=False,
    ),
    "sync": Tool(
        "sync_picture.py",
        "只做整理：零散源文件夹 → train/val/test(oneclick 的第 1 步)",
        "更多",
        "--help 看参数；一般需指定源目录与输出目录",
        interactive=False, primary=False,
    ),
    "labelme2yolo": Tool(
        "LabelMeToYOLO.py",
        "只做转换：LabelMe → YOLO det/cls/seg(oneclick 的第 2 步)",
        "更多",
        "--help 看参数；需 LabelMe 源目录",
        interactive=False, primary=False,
    ),
    "seg2mvtec": Tool(
        "yoloseg_to_mvtec.py",
        "只转单个 YOLO-seg 数据集(to-mvtec 的非交互内核)",
        "更多",
        "需 --src <seg目录> --out <输出目录>",
        interactive=False, primary=False,
    ),
    "sample-browse": Tool(
        "seg_sample_browse.py",
        "物体版第0步(可选):抽样摊图,帮你归纳源数据里有哪些物体",
        "更多",
        "需 --src <seg源> --out <浏览目录> [--limit 500]",
        interactive=False, primary=False,
    ),
    "make-sample": Tool(
        "make_sample_yoloseg.py",
        "造一份示例 YOLO-seg 数据(仅 demo/测试用)",
        "更多",
        "回车走默认输出；或 --out <目录> [--force]",
        interactive=True, primary=False,
    ),
    "inspect-labelme": Tool(
        "inspect_labelme_shapes.py",
        "统计 / 检查 LabelMe 标注内容(排查用)",
        "更多",
        "--help 看参数；需 LabelMe 目录",
        interactive=False, primary=False,
        read_only=True,
    ),
}

STAGE_ORDER = [
    "自动发现",
    "转语义分割",
    "合并数据集",
    "处理数据类别",
    "整理 + 转 YOLO",
    "转 MVTec AD",
    "子集对齐",
    "更多",
]


def _ordered_commands(show_all: bool) -> list[tuple[str, Tool]]:
    """按 STAGE_ORDER 稳定排序后的 (command, Tool) 列表；show_all=False 只含主流程。"""
    items: list[tuple[str, Tool]] = []
    for stage in STAGE_ORDER:
        for cmd, tool in COMMANDS.items():
            if tool.stage == stage and (show_all or tool.primary):
                items.append((cmd, tool))
    return items


def validate_registry(commands: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Return registry errors for all tools or a selected command subset."""
    errors: list[str] = []
    known_stages = set(STAGE_ORDER)
    scripts: dict[str, str] = {}
    selected = set(commands) if commands is not None else None
    for command, tool in COMMANDS.items():
        if selected is not None and command not in selected:
            continue
        if tool.stage not in known_stages:
            errors.append(f"{command}: 未登记的阶段 {tool.stage!r}")
        script = TOOLS_DIR / tool.script
        if not script.is_file():
            errors.append(f"{command}: 脚本不存在 {script}")
        elif not tool.read_only:
            source = script.read_text(encoding="utf-8", errors="ignore")
            if "--dry-run" not in source:
                errors.append(f"{command}: 写入工具缺少统一 --dry-run")
        previous = scripts.get(tool.script)
        if previous is not None:
            errors.append(f"{command}: 与 {previous} 重复登记脚本 {tool.script}")
        scripts[tool.script] = command
    return errors


def validate_cli_contracts() -> list[str]:
    """Check every registered CLI and the dry-run contract for writers."""
    errors = []
    for command, tool in COMMANDS.items():
        try:
            result = subprocess.run(
                [sys.executable, str(TOOLS_DIR / tool.script), "--help"],
                capture_output=True, text=True, timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{command}: CLI 检查失败: {exc}")
            continue
        if result.returncode != 0:
            errors.append(f"{command}: --help 执行失败 (exit={result.returncode}): {result.stderr.strip()}")
        elif not tool.read_only and "--dry-run" not in result.stdout:
            errors.append(f"{command}: CLI 帮助缺少 --dry-run")
    return errors


def _capability_text(tool: Tool) -> str:
    if tool.read_only:
        return "[只读]"
    script = TOOLS_DIR / tool.script
    try:
        source = script.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "[不可用]"
    labels = ["dry-run"] if "--dry-run" in source else []
    if tool.atomic_publish or "staged_output" in source:
        labels.append("原子替换")
    elif "write_json_atomic" in source:
        labels.append("原子写入")
    return f"[{'/'.join(labels)}]" if labels else ""


def print_menu(numbered: bool = False, show_all: bool = False) -> list[str]:
    """打印菜单；numbered=True 时给每项编号，返回编号→command 的顺序表。"""
    print((__doc__ or "").strip().splitlines()[0])
    registry_errors = validate_registry()
    if registry_errors:
        print("\n[配置错误]")
        for error in registry_errors:
            print(f"  - {error}")
    scope = "全部工具" if show_all else "常用工具"
    print(f"\n{scope}:\n")
    order: list[str] = []
    last_stage = None
    for cmd, tool in _ordered_commands(show_all):
        if tool.stage != last_stage:
            print(f"  [{tool.stage}]")
            last_stage = tool.stage
        order.append(cmd)
        prefix = f"{len(order):>2}. " if numbered else "    "
        print(f"  {prefix}{cmd:<16} {tool.desc} {_capability_text(tool)}".rstrip())
    if not show_all:
        print("\n  (输入 more 展开单步/辅助工具，详见 README.md)")
    return order


def _run(tool: Tool, args: list[str]) -> int:
    script = TOOLS_DIR / tool.script
    if not script.exists():
        print(f"脚本不存在: {script}", file=sys.stderr)
        return 1
    # 子进程运行:每个工具自己的 argparse / 交互 / 进度条都原样生效，stdin 继承。
    try:
        return_code = subprocess.call([sys.executable, str(script), *args])
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    if return_code < 0:
        print(f"工具被信号 {-return_code} 终止: {script.name}", file=sys.stderr)
        return 128 - return_code
    elif return_code > 0:
        print(f"工具执行失败(exit={return_code}): {script.name}", file=sys.stderr)
    return return_code


def interactive() -> int:
    show_all = False
    while True:
        order = print_menu(numbered=True, show_all=show_all)
        print()
        try:
            raw = input("选择要做的处理 [序号/命令，more 展开全部，q 退出]: ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n已取消。")
            return 130
        if not raw or raw.lower() in ("q", "quit", "exit"):
            return 0
        if raw.lower() in ("more", "all", "更多"):
            if show_all:
                print("\n当前已显示全部工具。")
            else:
                show_all = True
            print()
            continue
        # 支持序号或命令名
        cmd = None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(order):
                cmd = order[idx - 1]
        elif raw in COMMANDS:
            cmd = raw
        if cmd is None:
            print(f"无效选择: {raw!r}，请重新输入。")
            continue
        registry_errors = validate_registry([cmd])
        if registry_errors:
            print("\n所选工具当前不可用:", file=sys.stderr)
            for error in registry_errors:
                print(f"  - {error}", file=sys.stderr)
            continue
        tool = COMMANDS[cmd]
        print(f"\n>> {cmd} — {tool.desc}")
        print(f"   {tool.usage}")
        hint = "回车进入工具交互；b 返回" if tool.interactive else "输入参数；回车显示帮助；b 返回"
        try:
            arg_line = input(f"参数({hint}): ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n已取消。")
            return 130
        if arg_line.casefold() in {"b", "back", "返回"}:
            print()
            continue
        if arg_line.casefold() in {"q", "quit", "exit"}:
            return 0
        try:
            args = shlex.split(arg_line) if arg_line else []
        except ValueError as exc:
            print(f"参数引号不完整或转义无效: {exc}")
            continue
        print()
        if not args and not tool.interactive:
            _run(tool, ["-h"])
            print()
            continue
        result = _run(tool, args)
        if result in (0, 130) or result >= 128:
            return result
        print("\n工具执行失败，返回菜单修正参数。\n")


def _print_registry_errors(errors: list[str]) -> None:
    print("转换工具注册表存在错误:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 无参数 → 交互菜单
    if not argv:
        return interactive()
    first = argv[0]
    if first in ("-h", "--help", "help", "list", "menu"):
        print_menu()
        return 0
    if first in ("more", "all"):
        print_menu(show_all=True)
        return 0
    if first == "doctor":
        registry_errors = validate_registry()
        if not registry_errors:
            registry_errors.extend(validate_cli_contracts())
        if registry_errors:
            _print_registry_errors(registry_errors)
            return 2
        print(f"转换工具注册表检查通过：{len(COMMANDS)} 个命令可用。")
        return 0
    if first not in COMMANDS:
        print(f"未知命令: {first!r}", file=sys.stderr)
        suggestions = difflib.get_close_matches(first, COMMANDS, n=3, cutoff=0.45)
        if suggestions:
            print(f"可能的命令: {', '.join(suggestions)}", file=sys.stderr)
        print()
        print_menu()
        return 2
    registry_errors = validate_registry([first])
    if registry_errors:
        _print_registry_errors(registry_errors)
        return 2
    # 直接分发:剩余参数原样转发给目标工具。
    return _run(COMMANDS[first], argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
