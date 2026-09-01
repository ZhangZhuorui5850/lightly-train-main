# lightly-train 代码打包与迁移方案分析

> 调研日期：2026-09-01　工作分支：`master`　环境：`conda lightlytrain`（Python 3.10.20 / torch 2.5.1+cu121）
> 所有体积与文件数均为实测值（在 WSL 内执行 `du` / `git ls-files` 得到）。

---

## 0. 一句话结论

"不含 data 和 out，只传框架代码"这个方向**是对的，且收益极大**（52G → 8.3M）。
但方案有一个**致命前提**：本项目当前 **git 索引远远落后于磁盘**，`git clone` / `git archive`
会丢掉大量最新代码。迁移必须**按文件系统打包**（或先把改动全部提交）。

---

## 1. 体积事实（实测）

| 目录 / 文件 | 体积 | 性质 | 是否迁移 |
|---|---:|---|---|
| `datasets/` | **52G** | 数据（含 `coco_synced` 20G、`aeroscapes` 790M） | ❌ |
| `out/` | **1.9G** | 实验产物（checkpoint / 推理结果 / 报告） | ❌ |
| `.git/` | 185M | 版本历史 | ⚠️ 可选 |
| `weights/` | 1.4G | DINOv3 预训练权重（vitl16 1.2G / vits16plus 110M / vits16 83M） | ⚠️ 可选 |
| `tool_lib/*.ttc` | 47M | 微软雅黑字体（报告绘图中文渲染用） | ⚠️ 可选 |
| `docs/` | 12M | 上游 LightlyTrain sphinx 文档 | ⚠️ 可选 |
| `src/` | **3.3M** | 上游库 fork（353 文件） | ✅ |
| `tool_lib/` | **1.3M** | 自研工具层（42 文件，不含字体） | ✅ |
| `tests/` | **1.3M** | 测试（164 文件） | ✅ |
| `datasets/convert_datasets/convert_tools/` | **1.1M** | 数据转换工具（56 .py） | ✅ |
| 顶层脚本 + 文档 + 其他 | ~1.3M | launcher.py / train_*.py / *.md / reports / temp 等 | ✅（temp 可剔） |

**分档总览（`du` 统计值，含 4KB 块对齐开销）**

| 口径 | du 值 | 说明 |
|---|---:|---|
| **A 核心代码** | **8.3M** | 推荐口径：src + tool_lib + tests + convert_tools + 顶层脚本 |
| B = A + docs | 20M | 加 12M 上游文档 |
| C = B + 中文字体 | 66M | 加 47M `.ttc` |
| D = C + weights | 1.4G | 加 1.4G 预训练权重 |
| E = D + .git | ~1.6G | 全量 |

> **口径修正**：上表是 `du` 的磁盘占用（按 4KB 块计），A 档 681 个文件里多数只有几 KB，
> 块对齐开销被显著放大。**文件表观总大小实际只有 5.5 MB**（`tar -tzvf` 求和实测）。
> 实际产物见 §9。

> 注：`.pth` 权重本身已是高度压缩的二进制，gzip 几乎无收益。若 D 档要打包，建议
> `tar` 不加 `-z`（或换 `zstd`），否则白白多花几分钟 CPU。

---

## 2. ⚠️ 首要风险：git 索引与磁盘严重不一致

这是整个方案里最容易被忽略、代价最高的一点。

```
当前 master：已修改 56 个文件，未跟踪 60 个文件（共 116 项未入库）
```

**未被 git 记录的关键代码（部分列举）：**

| 文件 | 状态 | 影响 |
|---|---|---|
| `AGENTS.md` | 未跟踪 | 项目速查文档 |
| `offline_transfer/` | 未跟踪 | 打包/安装脚本本体 |
| `train_face_wider.py` | 未跟踪 | 训练入口之一 |
| `tool_lib/dataset_adapter.py` | 未跟踪 | 数据集识别核心 |
| `tool_lib/file_index.py` | 未跟踪 | |
| `tool_lib/seg_eda.py`、`seg_instance_eda.py` | 未跟踪 | |
| `tool_lib/training_run.py` | 未跟踪 | `train_seg.py` 直接 import 它 |
| `tool_lib/artifact_transaction.py` | 未跟踪 | |
| `tests/` 下 13 个测试文件 | 未跟踪 | 含 `test_launcher_cli.py` |
| `src/lightly_train/_data/file_helpers.py` | 已修改 | fork 的本地补丁 |
| `tool_lib/convert_tools.py` | **已删除** | 工作区已删，git 里还在 |

**`convert_tools` 的情况尤其严重：**

```
datasets/convert_datasets/convert_tools/
    已跟踪： 20 个文件
    未跟踪： 34 个文件   ← 62% 的转换脚本根本没进过 git
```

未跟踪的包括 `dataset_detector.py`、`dataset_merger.py`、`coco_to_synced.py`、
`face_wider_prepare.py`、`yolo_class_editor.py`、`semantic_to_xanylabeling.py` 等主力脚本，
以及 19 个测试文件。

**成因**：`.gitignore` 用的是**白名单**机制（`datasets/*` 全忽略，再逐条 `!...` 放行）。
白名单本身是生效的（文件显示为 `??` 未跟踪而非被忽略），但这些文件从未 `git add`。

**结论**：
- ❌ 不要靠 `git clone` / `git archive` / `git checkout` 迁移，会丢掉约 62% 的转换脚本和全部新增模块。
- ✅ 要么**按文件列表打包**（推荐，见 §4），要么**先 `git add -A && git commit` 再走 git。

---

## 3. 如何界定"lightly-train 相关代码"范围

不要按"名字里有没有 lightly-train"来筛——顶层目录名就叫这个，等于没筛。建议按**依赖层级**分三层，
判定标准是：**迁移后能否在裸机上跑通 `python launcher.py` 和 `python convert.py`**。

### L1 框架本体（必须）
- `src/lightly_train/` —— 上游 0.14.2 的 fork，含本地补丁：
  - 续训掉点修复（EMA `updates` → registered buffer；`best_agg_metric_values` 持久化），见 `RESUME_FIX_NOTES.md`
  - 新增 `_metrics/` 模块
  - EOMT 与任务数据集改动
  - 最近一次改动：`19deb8d6 (2026-05-29)`；当前工作区另有 1 个未提交修改

### L2 自研层（必须）
- `launcher.py`（统一配置区 + CLI）
- `tool_lib/`（训练/推理/评估/EDA/报告/多卡调度）
- `train_det.py` / `train_seg.py` / `train_face_wider.py` / `seg_copy_paste.py`
- `datasets/convert_datasets/convert.py` + `convert_tools/`（数据转换唯一入口）
- `class_groups.yaml`、`pyproject.toml`

### L3 支撑与可选
- `tests/`（164 文件，1.3M）—— **建议带上**，这是迁移后唯一能自证没打包错的手段
- `weights/`、`docs/`、`tool_lib/*.ttc`、`offline_transfer/`、`reports/` —— 按需
- 可剔除：`temp/`（临时）、`backup_resume_fix/`（修复前后快照，纯历史）、
  `.github/`、`dev_tools/`、`docker/`、`licences/`（上游工程文件）

### 明确剔除
- `datasets/` **除** `convert_datasets/convert.py` 和 `convert_datasets/convert_tools/` 外全部剔除
- `out/`（已被 .gitignore 忽略）
- `.git/`（185M，除非需要保留历史）
- 所有 `__pycache__/` 与 `*.pyc`

> 补充：已提交进 git 的两个样例数据集 `TLPD`（106M）和 `NEU-DET811kuozeng1bei`（65M）
> 位于 `datasets/convert_datasets/` 下。实测全仓库代码中**无任何引用**（grep `TLPD|NEU-DET` 无命中），
> 属于历史遗留，随 `datasets/` 一起剔除即可，不影响功能。

---

## 4. 打包清单与命令

采用与仓库现有 `offline_transfer/package_offline.sh` 相同的 **`find` + `tar --no-recursion`**
模式（该模式在 `tar` 中对 `--exclude` 的匹配歧义免疫，已在本项目验证过）。

> ⚠️ 不要写成 `tar -czf x.tar.gz --exclude='datasets/*' 项目 项目/datasets/convert_datasets/convert_tools`。
> GNU tar 对"显式列出的命令行参数是否仍受 `--exclude` 约束"存在版本差异，容易静默漏文件。

### 4.1 A 档：核心代码 8.3M（推荐）

```bash
# 在项目父目录执行（这里是 /home/zzr）
cd /home/zzr
PROJ=lightly-train-main
DATE=$(date +%Y%m%d)

{
  find "$PROJ" -type f \
    ! -path "$PROJ/datasets/*" \
    ! -path "$PROJ/out/*" \
    ! -path "$PROJ/.git/*" \
    ! -path "$PROJ/docs/*" \
    ! -path "$PROJ/weights/*" \
    ! -path "$PROJ/temp/*" \
    ! -path "$PROJ/backup_resume_fix/*" \
    ! -path "*/__pycache__/*" \
    ! -name "*.pyc" \
    -print0
  find "$PROJ/datasets/convert_datasets/convert_tools" -type f \
    ! -path "*/__pycache__/*" ! -name "*.pyc" -print0
  printf '%s\0' "$PROJ/datasets/convert_datasets/convert.py"
} | tar --null --no-recursion --files-from=- \
       --create --gzip --file "${PROJ}-code-${DATE}.tar.gz"

sha256sum "${PROJ}-code-${DATE}.tar.gz" > "${PROJ}-code-${DATE}.sha256"
```

若**需要**权重 / docs / 字体，从上面的 `! -path ...` 排除项里删掉对应行即可
（`weights` 那档建议去掉 `--gzip`）。

### 4.2 交付物自检（打包后立刻做）

```bash
# 文件数应 ≈ 700（核心代码 708 个文件）
tar -tzf "${PROJ}-code-${DATE}.tar.gz" | wc -l

# 必须存在的关键路径
tar -tzf "${PROJ}-code-${DATE}.tar.gz" | grep -E \
  'launcher.py$|src/lightly_train/__init__.py$|tool_lib/training_run.py$|tool_lib/dataset_adapter.py$|datasets/convert_datasets/convert.py$|convert_tools/dataset_detector.py$'

# 必须不存在的路径（漏了说明排除失效）
tar -tzf "${PROJ}-code-${DATE}.tar.gz" | grep -cE '/out/|/\.git/|/TLPD/|/NEU-DET|__pycache__|\.pyc$'
# ↑ 期望输出 0
```

### 4.3 关于现有 `offline_transfer/package_offline.sh`

脚本**方向正确**（prune 掉 `out` 和 `datasets`，只回收 `convert.py` + `convert_tools`，带 SHA256），
但用于本次"只传代码"的需求有 4 处需要调整：

| 问题 | 说明 |
|---|---|
| ① 它默认连 conda 环境一起打 | 与"只传框架代码"冲突；需要拆出纯代码模式 |
| ② 代码包里含 `.git`（185M） | `find` 只 prune 了 `out` 和 `datasets`，`.git` 被一起打进去 |
| ③ 未排除 `__pycache__` / `*.pyc` | 无害但是垃圾文件 |
| ④ 硬依赖 `conda-pack` | 当前环境的 `conda-pack` 版本号显示为 **`0.0.0`**（异常，正常应为 0.7.x），**能否正常打包未经验证**，用前必须实测 |

---

## 5. 迁移后的目录结构

```
<服务器安装根>/
└── lightly-train-main/
    ├── launcher.py                 # 总入口（统一配置区在文件顶部）
    ├── seg_copy_paste.py
    ├── train_det.py  train_seg.py  train_face_wider.py
    ├── class_groups.yaml  pyproject.toml
    ├── AGENTS.md  LAUNCHER.md  README.md  RESUME_FIX_NOTES.md
    ├── src/
    │   └── lightly_train/          # fork，必须保留；launcher 强制使用它
    ├── tool_lib/                   # 自研工具层（*.ttc 可选）
    ├── tests/
    ├── offline_transfer/
    ├── datasets/                   # ← 必须存在（空目录即可），见注意事项①
    │   └── convert_datasets/
    │       ├── convert.py
    │       └── convert_tools/      # 56 个 .py + tests/
    ├── weights/                    # 可选，按需放 DINOv3 权重
    └── out/                        # 运行时自动创建
```

解压后**手工补建目录**（`--no-recursion` + `-type f` 不会打包空目录）：

```bash
mkdir -p datasets weights out
```

---

## 6. 运行环境要求

### 6.1 必需（已实测的当前版本）

| 组件 | 当前版本 | 说明 |
|---|---|---|
| Python | 3.10.20 | `pyproject.toml` 要求 `>=3.8,<3.14` |
| PyTorch | **2.5.1+cu121** | 含 CUDA 12.1 运行时 |
| torchvision | 0.20.1+cu121 | 必须与 torch 的 CUDA 版本一致 |
| torchmetrics | 1.9.0 | |
| pytorch-lightning | 2.6.1 | |
| transformers | 5.3.0 | |
| numpy | 2.2.6 | |
| albumentations | 2.0.8 | |
| 显卡 | 单卡 RTX 4060 Laptop 8GB | 服务器若更大显存，batch size 可上调 |

完整依赖见 `pyproject.toml` 的 `[project.dependencies]`（torch/torchvision/transformers/
pytorch_lightning/torchmetrics/albumentations/omegaconf/pyarrow/psutil/filelock/fsspec/
lightly/nvidia-ml-py/tensorboard/tqdm/eval-type-backport）。
可选 extras 按需：`ultralytics`、`onnx`、`onnxruntime`、`onnxslim`、`rfdetr`、
`super-gradients`、`timm`、`mlflow`、`wandb`、`dicom`。

### 6.2 驱动与 CUDA

- 服务器 NVIDIA 驱动需**支持 CUDA 12.1**（Linux 驱动 ≥ 530.x）。
- 若服务器是较新的卡（H100 / A800 / L40S 等），cu121 通常仍可运行，
  但如需最佳性能或用到新特性，应重新装匹配 CUDA 版本的 torch，
  并**回归测试**——本项目用了 EOMT / DINOv3，对算子版本较敏感。

### 6.3 lightly_train 的导入机制（本项目最特殊的一点）

环境里同时存在：
- `site-packages/lightly_train/`（PyPI 上游 0.14.2）
- 仓库 `src/lightly_train/`（打了本地补丁的 fork）

`tool_lib/common.py::_prioritize_repo_source()` 会把 `src` 插到 `sys.path` 最前面，
并且在加载时**硬校验**：若 `lightly_train` 是从别处导入的，直接抛 `RuntimeError`。

**迁移含义：**
- ✅ 只要 `src/` 目录在，fork 自动生效，**不需要** `pip install -e .`
  （AGENTS.md 也明确要求：改安装方式前需用户确认）
- ⚠️ 但 site-packages 里的上游包**仍要装**，且版本建议锁定 **0.14.2**——fork 是基于 0.14.2 改的，
  若服务器装了别的版本，可能出现 API 不匹配
- ⚠️ `train_det.py` / `train_seg.py` 这类**独立脚本不走 launcher**，不会自动注入 `src`。
  需要时显式 `PYTHONPATH=src python ...`（`train_seg.py` 已自带 `sys.path.insert`，但 `train_det.py` 需要确认）

---

## 7. 迁移后验证清单

按顺序执行，任一步失败就停：

```bash
cd lightly-train-main
source <env>/bin/activate        # 或 conda activate lightlytrain

# 1) torch 与 CUDA
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
#    期望：2.5.1+cu121 12.1 True

# 2) fork 是否真的生效（最关键）
python -c "import sys;sys.path.insert(0,'src');import lightly_train as lt;print(lt.__file__);print(lt.__version__)"
#    期望：.../lightly-train-main/src/lightly_train/__init__.py    且   0.14.2
#    若打印 site-packages 路径 → 后续所有训练都会静默走上游代码，必须修好

# 3) launcher 可加载
python launcher.py --help

# 4) 转换工具自检（convert.py doctor 延迟导入 torch，无 GPU 也能跑）
python datasets/convert_datasets/convert.py doctor

# 5) 测试
conda run -n lightlytrain python -m pytest tests/test_launcher_cli.py -x
conda run -n lightlytrain python -m pytest tool_lib/tests -x
```

---

## 8. 注意事项与坑

| # | 事项 | 说明 |
|---|---|---|
| ① | **`datasets/` 空目录必须存在** | `launcher.py` 的 `COMMON_SETTINGS.dataset_search_roots = ["datasets"]`，目录不存在会让数据集自动发现失败。可用 `LIGHTLY_DATASET_SEARCH_ROOTS=/data1:/data2` 指向仓库外的真实数据位置 |
| ② | **`out/` 缺失是预期行为** | launcher 会自动从 `out/` 检索最近实验和 best 权重，空目录时相关功能无效属正常 |
| ③ | **数据要单独迁移** | 真实数据在服务器另配；`train_seg.py` / `train_det.py` 顶部常量区的 `DATA_YAML`、`OUT`、`BACKBONE_WEIGHTS` 需按服务器路径改 |
| ④ | 好消息：仓库**无硬编码绝对路径** | 已 grep `launcher.py`、`tool_lib/*.py`、`train_*.py`、`convert.py`，未见 `/home/zzr` 或 `/mnt/` 硬编码 |
| ⑤ | **中文字体** | `tool_lib/*.ttc`（47M）用于报告/EDA 图上的中文标注。服务器上若没有中文字体，绘出来的中文会变方框。可在服务器装 `fonts-wqy-zenhei` 并改字体路径，省掉 47M |
| ⑥ | `.gitignore` 白名单机制 | 以后新增 `convert_tools/` 下的脚本，必须同步往 `.gitignore` 加 `!` 放行条目，否则文件永远不会进 git——这正是当前 34 个未跟踪文件的成因之一 |
| ⑦ | 换行符 | 当前在 Windows（Git Bash / WSL 混合）操作。`.gitattributes` 只声明了 `*.ipynb linguist-vendored`，**没有 `* text=auto eol=lf`**，若从 Windows 侧打包可能带入 CRLF。建议打包前 `file launcher.py` 抽查，或在 `.gitattributes` 补 eol 规则 |
| ⑧ | 只传代码 = 不带 conda 环境 | 若服务器**离线**，`pip install` 走不通，需要连环境一起打包。此时用 `offline_transfer/package_offline.sh`，但**先实测 conda-pack**（见 §4.3 问题④） |
| ⑨ | 校验文件完整性 | 包跨机器传输后先 `sha256sum -c` 再解压，1.4G 级别的文件传输中途损坏并不罕见 |
| ⑩ | 传输方式 | A 档压缩后仅 1.2M，`scp` 足够；1.4G（含权重）建议 `rsync -P` 或 `tar` 直传，`.pth` 不再压缩 |

---

## 9. 打包演练实测结果（2026-09-01 已执行）

方案不是纸面推演，已在本机完整跑通一遍：打包 → 解压到 `/tmp` → 全量验证。
脚本：`temp/package_and_verify.sh`。

### 产物

```
/home/zzr/lightly-train-main-code-20260901.tar.gz     1.20 MB
sha256: 7095c4b2212edfbfb93ac44692b90859a14562f453070947d293c46ae8b4850b

包内条目数：681
内容表观大小：5.5 MB（压缩比 4.6:1）
```

### 自检结果

| 检查项 | 结果 |
|---|---|
| 关键文件存在性（launcher.py / `src/lightly_train/__init__.py` / `tool_lib/training_run.py` / `tool_lib/dataset_adapter.py` / `convert.py` / `convert_tools/dataset_detector.py` / `train_face_wider.py` / AGENTS.md） | ✅ 全部命中 |
| 违禁条目（`out/`、`.git/`、`TLPD`、`NEU-DET`、`__pycache__`、`*.pyc`、`*.ttc`、`.pytest_cache`） | ✅ 0 |
| 解压后 `torch` / CUDA | ✅ `2.5.1+cu121`，`available True` |
| **fork 来源校验** | ✅ `<项目>/src/lightly_train/__init__.py`，version `0.14.2` |
| `python launcher.py --help` | ✅ 12 个子命令全部列出 |
| `convert.py doctor` | ✅ 「转换工具注册表检查通过：25 个命令可用」 |
| `pytest tests/test_launcher_cli.py tool_lib/tests` | ✅ **93 passed, 1 skipped**（7.44s） |

### 演练中暴露的两个问题（已修正）

1. **首版脚本漏了 `*.ttc` 排除项** → 47M 中文字体被打进包里，产物 31M。
   加上 `! -name "*.ttc"` 后降到 1.20M。**这说明打包后必须跑 `tar -tzvf` 看最大文件，
   光看自检的"关键文件存在"发现不了体积异常。**
2. **`.pytest_cache` 需要显式排除** —— 项目根目录有该缓存目录，会被 `find` 一并收走。

### 关于中文字体（`.ttc`，47M）

A 档默认**不含**字体。后果：在服务器上跑 `seg-eda` / `report` 生成的图片里，中文标签会变方框。
两个选择：

```bash
# 选项 1：服务器上装开源中文字体（省 47M）
apt-get install -y fonts-wqy-zenhei     # 然后改 tool_lib 里的字体路径

# 选项 2：把字体加进包（改一个排除项即可）
#   在打包命令里删掉  ! -name "*.ttc"  那一行
```

---

## 10. 建议的执行顺序

1. **先提交或先打包**（二选一）
   - 推荐：先 `git add -A && git commit -m "chore: 迁移前提交全部工作区改动"`，
     让 git 与磁盘对齐——这本身就是一件事一提交的好习惯，且之后两种迁移方式都安全
   - 或者：跳过提交，直接按 §4.1 打文件包
2. 按 A 档打包（8.3M），执行 §4.2 自检
3. 传输 + `sha256sum -c` 校验
4. 解压，`mkdir -p datasets weights out`
5. 按 §7 逐条验证，**第 2 步（fork 来源检查）不通过就不要开始训练**
6. 按需单独迁移数据集与权重
