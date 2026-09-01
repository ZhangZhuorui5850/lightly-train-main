# AGENTS.md — Agent 工作指导

给在本仓库工作的 AI agent（ZCode / Claude Code / Codex 等）的速查文档。
原则：**先查这里，不要重新探索**环境、入口和目录约定；本文件描述的是已验证的事实（2026-08 调研）。

本项目 = 上游 [LightlyTrain](https://github.com/lightly-ai/lightly-train) 0.14.2 的 fork（`src/`）+
大量自研训练/推理/数据工具（`launcher.py`、`tool_lib/`、`train_*.py`、`datasets/convert_datasets/`）。
用途：DINOv3 backbone 的目标检测 / 实例分割 / 语义分割 / 分类训练与数据集治理。

---

## 1. 运行环境（最重要，不要自己再找）

- **Conda 环境是 `lightlytrain`（全小写）**。运行任何 Python 前先激活：

  ```bash
  conda activate lightlytrain            # 交互 shell
  conda run -n lightlytrain python ...   # 单条命令（agent 推荐用这种）
  ```

- 环境内容（已验证）：Python 3.10.20、PyTorch 2.5.1+cu121（CUDA 可用）、lightly_train 0.14.2、pytest 9.x。
- ⚠️ conda 里还有一个**大写** `LightlyTrain` 环境，里面只有 conda-pack，是离线打包残留的空环境，**不要用**。其他环境（cvmodel、dino_yolo 等）与本项目无关。
- 本机 GPU：单卡 RTX 4060 Laptop，**8GB 显存**。训练 batch size 要保守；推理/评估用默认 `--device auto` 即可。多卡调度代码（gpu_parallel）在别的机器上才用得上。
- Makefile 里的 ruff / mdformat / pre-commit 等格式化工具**未装**在这个环境，跑不了 `make format`。

## 2. LightlyTrain 导入来源

仓库是 `src/` 布局，环境里同时存在 PyPI 上游包的普通安装和仓库 fork：

- 仓库 `src/` 是打了本地补丁的 fork：续训掉点修复（EMA updates / best 指标持久化，见
  `RESUME_FIX_NOTES.md`）、新增 `_metrics/` 模块、EOMT 与任务数据集的修改等（src 最后提交 2026-05-29）。
- `launcher.py` 通过 `tool_lib/common.py` 把仓库 `src/` 放到导入路径首位，训练、推理和评估使用 fork；
  启动时会校验模块来源。
- `train_det.py`、`train_seg.py` 等直接导入 `lightly_train` 的独立脚本仍按 Python 环境路径解析。
  需要让独立脚本使用 fork 时显式设置：

  ```bash
  PYTHONPATH=src conda run -n lightlytrain python ...   # 临时验证，环境不变
  ```

- ⚠️ `pip install -e .` 或升级/重装 lightly-train 会改变整个环境的解析方式，执行前需要用户确认。
- 只改 `tool_lib/`、`launcher.py`、`train_*.py`、`seg_copy_paste.py`、`datasets/convert_datasets/**` 的不受此影响。

## 3. 入口地图

| 任务 | 入口 |
|---|---|
| 训练 / 推理 / 评估 / EDA / 数据筛选导出 / 报告（cls、det、seg 通用） | `launcher.py` |
| 数据集格式转换（**一切转换只记这一个入口**） | `datasets/convert_datasets/convert.py` |
| 独立训练脚本（改文件顶部配置区再运行） | `train_det.py`、`train_seg.py` |
| 语义分割在线 Copy-Paste 增强 | `seg_copy_paste.py`（被 train_seg.py import） |
| 离线打包迁移 | `offline_transfer/package_offline.sh` / `install_offline.sh` |

### launcher.py — 训练与实验工具总入口

- 交互模式：`python launcher.py`（进菜单）；CLI 模式：`python launcher.py <command> [options]`；
  出错时加 `--debug` 看完整 traceback。
- CLI 子命令：`train` `infer` `eval` `export` `seg-export` `seg-eda` `seg-curate` `eda`
  `review-sample` `report` `clean` `optimize`。
- **顶部统一配置区**：`COMMON_SETTINGS` / `DET_SETTINGS` / `CLS_SETTINGS` / `SEG_SETTINGS` /
  `SAHI_SETTINGS`。改默认数据集、实验目录、阈值、导出参数都改这里，底层模块自动读取；
  不要去底层模块里硬编码路径。
- 不指定 `--experiment-dir` / `--checkpoint` 时，自动从 `out/` 检索最近的相关实验并取 best 权重。
- 所有子命令的参数表在 `LAUNCHER.md`——**写 CLI 命令前先查它，别猜参数名**。
- 推理常用示例：

  ```bash
  python launcher.py infer --data datasets/mydata/data.yaml --split test   # 数据集评测
  python launcher.py infer --image-dir path/to/images                      # 文件夹批量
  python launcher.py eval --task det --data ... --split test               # 评估+对比图
  ```

### datasets/convert_datasets/convert.py — 数据转换唯一入口

- `python convert.py` 交互菜单；`convert.py list` 打印菜单；`convert.py <command> ...` 原样转发参数。
- `convert.py doctor` 检查注册脚本和常用写入工具的 `--dry-run` 契约。
- 常用 command：`datasets`（识别/诊断数据集）、`auto`（扫描出可做的转换）、`oneclick`、
  `yolo2semantic`、`semantic2xany`、`mvtec2yolo`、`det2seg`、`generated2seg`、`labelme2yolo`、
  `coco2sync`、`seg2mvtec`。
- 实现在 `convert_tools/`（约 30 个脚本，可单独运行，也被 convert.py 调起）。
- 转换实现统一位于 `datasets/convert_datasets/convert_tools/`，launcher 通过
  `tool_lib/dataset_adapter.py` 复用数据集识别与路径解析能力。
- 安全约定（agent 要遵守）：先 `--dry-run`；输出目录必须与源目录完全分离；非空输出默认拒绝，
  原子替换需 `--clean` / `--overwrite` / `--force`；主流程先写 staging 校验后再发布。

### 训练

- `train_det.py` / `train_seg.py` 是“改文件顶部常量再运行”的脚本（`out`、`model`、`data`、
  `steps`、`batch_size` 等），跑前先改配置而不是加参数。
- `train_seg.py` 必须在调用 train 前 `import seg_copy_paste` 并 `seg_copy_paste.enable(...)`
  （开关 `COPY_PASTE` 在文件顶部）。
- `launcher.py train` 走 `tool_lib/train_tools.py` 直接调 lightly_train API
  （det / cls / seg_instance / seg_semantic），不依赖上面两个脚本。
- backbone 预训练权重在 `weights/`（DINOv3 vits16 / vitl16 / vits16plus 的 .pth）。

## 4. 代码结构速查

```
launcher.py                总入口：统一配置区 + main()
tool_lib/
  interactive.py           菜单交互 + CLI 参数解析（所有 subparsers 在这，约 3000 行）
  dispatch.py              路由层：按 task × action 分发
  train_tools.py           训练实现（直接调 lightly_train API）
  cls_tools/det_tools/seg_tools.py   各任务推理、评估、报告外壳
  det_infer/det_export/det_eda/det_report/det_analysis/...   det 各功能核心
  seg_eda/seg_export/seg_semantic_curate/...                 seg 各功能核心
  dataset_adapter.py       统一数据集检测与路径解析（复用 convert_tools）
  gpu_parallel.py          det/seg 共用多卡调度内核（探测、负载过滤、分片、子进程编排）
  common.py                全局配置读取、运行时依赖延迟导入、通用函数
  progress.py              多卡进度条
datasets/convert_datasets/ 数据转换：convert.py + convert_tools/
src/lightly_train/         上游库 fork 源码（见 §2 陷阱）
tests/                     上游测试 + 本项目测试（test_launcher_cli.py、test_gpu_parallel.py、
                           test_seg_parallel.py、test_seg_copy_paste_cache.py、test_det_infer.py 等）
tool_lib/tests/            工具层测试（导出/选图逻辑）
out/                       一切输出的根（已 gitignore，不要提交、不要随意清理）
```

轻量功能（eda / report / convert）延迟导入 torch，没 GPU 环境也能跑；训练和 infer/eval 需要 torch。

## 5. 目录与产物约定

- `out/`：实验 `out/<日期或任务>/<name>/`，内含 `train.log`、`checkpoints/`、`exported_models/`
  （含 `exported_best.pt`）、`infer/`、`eval/`；EDA 报告在 `out/EDA/`；全量对比报告在 `out/all_report/`。
- `datasets/`：数据集不入库；只有 `convert_tools/` 的白名单脚本入库（.gitignore 里是白名单规则，
  新增转换脚本要同步加白名单，否则会被忽略）。
- `weights/`：预训练 backbone 权重。`reports/`：专项分析 markdown。`temp/`：临时杂物。
- `offline_transfer/`：离线打包（打包对象就是 `lightlytrain` conda env + 代码 + 权重）。
- `docs/source/`：上游 LightlyTrain 的 sphinx 文档；上游用法说明在 `README.md`。
- `backup_resume_fix/`：续训修复的前后快照；`RESUME_FIX_NOTES.md`：修复说明。

## 6. 测试

```bash
conda run -n lightlytrain python -m pytest tests/test_launcher_cli.py -x   # 单文件
conda run -n lightlytrain python -m pytest tool_lib/tests -x               # 工具层
PYTHONPATH=src conda run -n lightlytrain python -m pytest tests -x         # fork 上游测试
```

- pytest 配置在 pyproject.toml（`python_files = tests/*.py tests/**/*.py`），无全局 pythonpath 注入。
  经过 `launcher.py` / `tool_lib.common` 的用例会优先使用仓库 `src/`；直接导入
  `lightly_train` 的用例遵循当前 Python 环境解析顺序。
- 改 `tool_lib/` / `launcher.py` / `convert_tools/` 逻辑后，跑对应的测试文件；
  上游库自身的大套测试很慢，按需选择。

## 7. 本项目新增的常用环境变量

| 变量 | 作用 |
|---|---|
| `LIGHTLY_DATASET_SEARCH_ROOTS=/mnt/a:/mnt/b` | 临时扩展数据集扫描根 |
| `LIGHTLY_PROGRESS_LAYOUT=compact` | 单行紧凑进度布局 |
| `LIGHTLY_PROGRESS_MAX_CARD_BARS=<n>` | 多卡进度条上限（默认 8） |
| `LIGHTLY_PROGRESS_SNAPSHOT_INTERVAL=<秒>` | 批处理日志快照间隔 |
| `LIGHTLY_GPU_MIN_FREE_MIB` / `LIGHTLY_GPU_MAX_UTILIZATION` | 选卡时的负载过滤 |
| `LIGHTLY_NUM_SHARDS` / `LIGHTLY_SHARD_INDEX` | 手动分片 |

`LIGHTLY_TRAIN_*` 开头的是上游库变量，定义在 `src/lightly_train/_env.py`。

## 8. Git 与沟通约定

- 直接在 `master` 分支工作；commit message 常用中文，一事一提交。
- 用户用中文交流；自研文档（LAUNCHER.md、convert README、本文件）都用中文，新文档保持中文。
- 上游代码（`src/lightly_train/`）能不动就不动；本项目功能集中在 `tool_lib/` 和
  `datasets/convert_datasets/`。确实要改上游时，参考 §2 的生效方式并与用户确认同步策略。

## 9. 深入阅读（按需）

- `LAUNCHER.md` — launcher 全部子命令与参数表
- `datasets/convert_datasets/README.md` — 转换工具全景、流水线图、各脚本说明
- `RESUME_FIX_NOTES.md` — 续训掉点修复的根因与改动清单
- `offline_transfer/OFFLINE_TRANSFER_GUIDE_CN.md` — 离线打包迁移手册
- `docs/plan0603.md`、`reports/*.md` — 历史计划与分析
