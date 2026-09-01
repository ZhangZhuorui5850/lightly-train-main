# SCNet 新服务器初始化与 LightlyTrain 训练环境评估

评估日期：2026-09-01  
评估方式：SCNet 页面信息、`ssh scnet` 只读检查、PyTorch/DTK 临时环境测试  
评估范围：网络、存储、加速卡、Python 环境、Conda、tmux、Cityscapes 下载和 LightlyTrain 迁移

## 1. 结论

这台实例具备运行深度学习训练的硬件基础：2 张 K100_AI 加速卡，每张 64GB 显存，平台提供 DTK 25.04.2 版 PyTorch 2.5.1。当前需要完成三项准备工作：安装 tmux、下载 Cityscapes、为 LightlyTrain 建立继承 DTK PyTorch 的 Python 虚拟环境。

推荐采用以下环境结构：

```text
SCNet 实例
├── 平台系统 Python
│   ├── DTK PyTorch 2.5.1
│   └── torchvision 0.20.1
├── /root/private_data/bupt/envs/
│   ├── cityscapes-download/       数据下载专用 venv
│   └── lightlytrain-dtk/          训练专用 venv，继承系统 DTK 包
├── /root/private_data/bupt/datasets/cityscapes/
└── /root/private_data/bupt/lightly-train-main/
```

Conda 当前暂缓安装。平台的 DTK PyTorch 位于系统 Python，普通 Conda 环境会形成独立 Python 包目录，并隐藏这套厂商适配包。训练环境优先使用 `venv --system-site-packages`；厂商提供 Conda 版 DTK 安装源后，可重新评估 Conda 方案。

## 2. 已验证的服务器状态

| 项目 | 实测结果 | 判断 |
|---|---|---|
| 操作系统 | Ubuntu 22.04，容器化实例 | 适合当前项目 |
| Python | 3.10.12 | 满足 LightlyTrain `>=3.8,<3.14` |
| PyTorch | 2.5.1 DTK 定制版 | 版本满足项目要求 |
| torchvision | 0.20.1 DTK 定制版 | 与 PyTorch 配套 |
| 加速卡 | 2 × K100_AI，单卡 64GB | 硬件资源充足 |
| HIP/DTK | HIP 6.3.25405、DTK 25.04.2 | 需要加载 DTK 环境脚本 |
| CPU/内存 | 页面配额 14 核、238GB | 以页面计费配额为准 |
| 私人存储 | `/root/private_data`，450GB，可写 | 推荐保存代码、数据和环境 |
| 公共存储 | `/root/public_data`，只读 | 多用户共享数据区 |
| Git/编译工具 | Git、Git LFS、gcc、g++、make、cmake 已安装 | 无需重复安装 |
| 下载工具 | curl、wget、unzip、rsync 已安装 | 无需重复安装 |
| Conda | 命令缺失，`/opt/conda` 仅有残留目录 | 当前不可用 |
| tmux | 缺失 | 建议安装 |
| Slurm | `sbatch/srun/squeue` 缺失 | 当前实例采用直接运行方式 |
| LightlyTrain | 缺失 | 同步项目后建立环境 |

## 3. 网络代理解决什么

服务器所在计算网络采用受控外网出口。实测现象如下：

1. DNS 可以解析 Cityscapes 域名。
2. 当前 shell 直接访问 Cityscapes、PyPI、GitHub 返回网络不可达。
3. 临时加载平台提供的 HTTP/HTTPS 代理后，这三个网站均返回 HTTP 200。

代理的数据路径是：

```text
下载程序 → SCNet 平台代理 → 公共互联网网站
```

代理只服务于联网命令，例如：

- `pip install`
- `csDownload`
- `curl`、`wget`
- GitHub 克隆或拉取
- 首次在线下载模型权重

本地数据训练、离线推理和已经缓存的权重无需网络代理。代理变量只在当前 shell 中生效，新开 SSH 或 tmux 窗口后需要重新加载。

推荐加载方法：

```bash
set -a
source <(
  grep -E '^(http_proxy|https_proxy|ftp_proxy)=' \
    /root/private_data/.ai_user_info/ai_proxy
)
set +a
```

这段命令完成三件事：

1. `grep` 只提取代理文件中的三个变量赋值行。
2. `set -a` 将变量导出给 pip、curl 和 Python requests。
3. `set +a` 恢复 shell 默认行为。

## 4. DTK 环境脚本解决什么

K100_AI 使用 DTK/HIP 软件栈。PyTorch 运行时需要找到 DTK 的动态链接库，例如 `libgalaxyhip.so.5`。

未加载 DTK 环境时，实测导入 PyTorch 报错：

```text
ImportError: libgalaxyhip.so.5: cannot open shared object file
```

加载平台脚本：

```bash
source /opt/dtk-25.04.2/env.sh
```

该脚本会设置 `PATH`、`LD_LIBRARY_PATH` 和 DTK/HIP 相关变量。加载后实测结果：

```text
torch 2.5.1
HIP 6.3.25405
torch.cuda.is_available() = True
torch.cuda.device_count() = 2
0 K100_AI 64GB
1 K100_AI 64GB
```

这里继续使用 `torch.cuda.*` API，这是 DTK PyTorch 为上层 PyTorch 程序提供的兼容接口。每个训练 shell 或 tmux 会话都需要加载一次 DTK 环境。

网络代理与 DTK 环境属于两条独立链路：

| 环境设置 | 作用对象 | 使用时机 |
|---|---|---|
| HTTP/HTTPS 代理 | 外网访问 | 下载数据、安装包、拉取代码和权重 |
| DTK `env.sh` | 加速卡运行库 | 导入 PyTorch、训练、推理和评估 |

## 5. Conda 评估

本地工作站使用 `lightlytrain` Conda 环境，里面是 CUDA 12.1/NVIDIA 版 PyTorch。SCNet 提供 K100_AI 和 DTK/HIP 版 PyTorch，两套环境面向不同加速平台。

将本地 Conda 环境直接打包迁移到 SCNet，会携带 CUDA/NVIDIA 依赖，并与 K100_AI 的 DTK 运行库产生兼容风险。

新建普通 Conda 环境也会使用自己的 Python 和 `site-packages`，平台预装的 `/usr/local/lib/python3.10/dist-packages/torch` 默认不会进入这个环境。要在 Conda 中训练，需要获得匹配 DTK 25.04.2 的 PyTorch 安装方式、wheel 或厂商 channel。

当前推荐训练环境：

```bash
source /opt/dtk-25.04.2/env.sh

python3 -m venv \
  --system-site-packages \
  /root/private_data/bupt/envs/lightlytrain-dtk

source \
  /root/private_data/bupt/envs/lightlytrain-dtk/bin/activate
```

`--system-site-packages` 让虚拟环境继承平台已经适配好的 DTK PyTorch，同时把后续 LightlyTrain 依赖安装在个人目录中。

安装项目依赖时需要保护现有 `torch` 和 `torchvision`。项目同步完成后，应先审查 pip 的依赖解析结果，再执行安装。项目的 `xformers`、NVIDIA NVML 监控以及可能涉及 CUDA 专用算子的功能需要单独验证。

## 6. tmux 是否必装

tmux 的优先级是“强烈推荐”。系统已有 `nohup`，两者都能让任务在 SSH 断开后继续运行。

| 场景 | tmux | nohup |
|---|---|---|
| SSH 断开后继续运行 | 支持 | 支持 |
| 重新进入实时界面 | 支持 | 通过日志查看 |
| 交互式观察训练 | 方便 | 较弱 |
| 多个命名任务 | 方便 | 需要手动管理 PID/日志 |
| 平台实例关机或释放 | 任务停止 | 任务停止 |

安装命令：

```bash
apt-get update
apt-get install -y tmux
```

`apt-get` 访问软件源时需要先加载网络代理。tmux 安装在当前容器系统层，实例重建后可能需要重新安装。

训练会话示例：

```bash
tmux new -s lightlytrain
source /opt/dtk-25.04.2/env.sh
source /root/private_data/bupt/envs/lightlytrain-dtk/bin/activate
```

## 7. 最小安装清单

### 当前建议安装

1. `tmux`：保护下载和训练免受 SSH 断线影响。
2. `cityscapesScripts`：负责 Cityscapes 登录下载、续传和 MD5 校验。
3. LightlyTrain 项目依赖：同步代码和建立训练 venv 后安装。

### 当前暂缓

1. Conda：等待 DTK 适配方式明确。
2. aria2：`csDownload --resume` 已覆盖当前续传需求。
3. CUDA Toolkit、NVIDIA 驱动：该实例使用 DTK/K100_AI。
4. xformers：属于平台相关二进制依赖，完成基础训练验证后评估。
5. Docker：当前已经运行在平台容器内。

## 8. 推荐执行顺序

### 阶段 A：基础工具和数据

1. SSH 登录并加载网络代理。
2. 安装 tmux。
3. 在 `/root/private_data/bupt/envs/cityscapes-download` 建立下载 venv。
4. 安装 `cityscapesScripts`。
5. 下载并解压 Cityscapes 到 `/root/private_data/bupt/datasets/cityscapes`。

### 阶段 B：代码和训练环境

1. 将仓库同步到 `/root/private_data/bupt/lightly-train-main`。
2. 加载 `/opt/dtk-25.04.2/env.sh`。
3. 创建 `lightlytrain-dtk` venv，并启用 `--system-site-packages`。
4. 安装项目缺失依赖，同时保留 DTK 版 torch/torchvision。
5. 验证项目导入来源指向仓库 `src/`。

### 阶段 C：兼容性验证

1. PyTorch 单卡矩阵运算测试。
2. PyTorch 双卡通信测试。
3. LightlyTrain 最小模型构建测试。
4. Cityscapes 单样本数据读取测试。
5. 1～5 step 的语义分割训练测试。
6. 显存、日志、checkpoint 和续训验证。

完成阶段 C 后，再开始正式长时间训练。这个顺序可以尽早暴露 DTK 与 LightlyTrain 的算子、监控和多卡兼容问题，控制无效计费时间。

## 9. 当前风险

1. LightlyTrain fork 主要在 CUDA/NVIDIA 环境验证，K100_AI/DTK 兼容性尚未完成项目级测试。
2. `nvidia-ml-py` 属于项目基础依赖，K100_AI 环境缺少 NVIDIA NVML；GPU 监控路径需要验证或降级。
3. DINOv3/EoMT 可能调用对 HIP/DTK 支持程度不同的 PyTorch 算子。
4. 双卡训练需要验证 DTK 的通信后端和 LightlyTrain 启动方式。
5. 平台计费为 7.2 元/小时，兼容性测试应采用最小数据和最少 step。
6. `/root/bupt` 位于容器 overlay；代码、数据、环境和 checkpoint 应落在 `/root/private_data/bupt`。

## 10. 最终建议

采用“平台 DTK PyTorch + 继承系统包的 venv + tmux + 持久化私人目录”的方案。网络代理仅在联网阶段加载，DTK 环境在每次训练会话加载。Conda 等到厂商适配安装源明确后再引入。

下一步应先完成阶段 A：安装 tmux，并在个人持久化目录下载 Cityscapes。服务器修改操作应在用户确认后执行。
