# seg eval/infer 多卡并行与升级（对齐 det）设计

日期：2026-06-17
状态：已批准设计，待写实现计划

## 背景与目标

`det infer`（[tool_lib/det_infer.py](../../../tool_lib/det_infer.py)）已是工业级推理框架：自动多卡分片、负载感知选卡、SAHI 切片、dry-run、run_meta、跨 shard 指标合并。

`seg`（[tool_lib/seg_tools.py](../../../tool_lib/seg_tools.py)）的 `run_eval` / `run_semantic_eval` / `run_infer` 仍是**纯串行单卡**：一张图一次 `predict`，`device="auto"` 直接甩给底层（一般落 `cuda:0`，无负载感知），无计时、无 dry-run、无 run_meta、无容错。

目标：把 seg 的 **eval 与 infer** 升级到与 det 对齐——多卡分片、负载感知选卡、吞吐优化、工程一致性——且对外产物（summary/可视化）与现状保持**逐位兼容**。

## 硬约束（决定方案边界）

1. **`predict()` 不支持 batch**：所有 task_model 的 `predict(image)` 是单图签名；transform 里有 `TODO: enable once predict_batch is implemented`。因此提速只能靠**多卡分片**（与 det 一致），不能靠加大 batch。
2. **`predict()` 已 `@torch.no_grad()` 且内部 `eval()`**（见 `dinov3_eomt_semantic_segmentation/task_model.py:263`）。所以"包 inference_mode/关梯度"这层无收益。
3. 上游 `predict_task.py` 有一条 Fabric 多卡管线，但只产 mask、不算指标，且 `batch_size` 仍锁 1。**本次不采用**。

## 方案总览

四部分：①抽共享 GPU 调度模块；②跨 shard 指标合并（语义=混淆矩阵相加，实例=RLE 序列化）；③seg eval/infer 并行入口 + 负载感知单卡；④吞吐优化（预取 + 语义 LUT）+ 工程对齐（计时/run_meta/dry-run/容错）。

## 1. 共享调度模块 `tool_lib/gpu_parallel.py`

**先做独立的纯重构提交**（det 行为零变化、绿测试），再叠加 seg。

从 `det_infer.py` 抽出以下"纯调度"内核到新模块：

- `GPU_HIGH_MEMORY_RATIO_THRESHOLD`
- `query_gpu_inventory()` — `nvidia-smi` 查 index/显存/利用率，按负载排序，尊重 `CUDA_VISIBLE_DEVICES`
- `filter_high_memory_gpus()`
- `format_gpu_summary()`
- `select_fallback_single_gpu()`
- `filter_indices_for_shard(count, shard_index, num_shards)` — 把现有 `filter_samples_for_shard` 的取模分片泛化成"与样本类型无关的索引选择"，det 与 seg 共用
- `run_sharded_subprocesses(jobs)` — 统一"按 GPU pin `CUDA_VISIBLE_DEVICES` 起子进程 + `wait` + 收集失败 shard"循环；`jobs` 为 `(shard_index, gpu_index, command)` 列表，返回失败 shard 列表

**留在各任务模块**（不抽）：子进程命令构造（det 产 det 报表、seg 产 seg summary，参数面不同）、结果合并逻辑（指标语义不同）。

`det_infer.py` 改为从 `gpu_parallel` import 上述符号，替换 call site；`filter_samples_for_shard` 改为薄封装调用 `filter_indices_for_shard`，保持现有签名与行为。

## 2. 跨 shard 指标合并

### 2.1 语义分割（精确、紧凑）

每个 shard 在自己的图子集上累积 `K×K` 混淆矩阵（复用 `_compute_semantic_iou` 之前的 `confusion` 累加，`seg_tools.py:319-336`），写出 shard 文件（混淆矩阵 + 行记录 + 计时）。父进程**逐元素求和**所有 shard 的混淆矩阵，再算 mIoU/Pixel Accuracy。与串行**逐位一致**。

多 split 语义评估：当前 `run_semantic_eval` 顺序循环 split（`seg_tools.py:375`）。并行模式像 det `run_parallel_all_infer` 那样，把"所有 split 的所有图"平铺到所有卡，子进程各自按 split 分桶累积混淆矩阵，父进程按 split 合并。

### 2.2 实例分割（RLE 序列化）

mAP 是全局排序指标，**不能对各 shard 的 mAP 取平均**（错误）。父进程必须看到全部 `(pred, gt)` 后喂**单个** `InstanceSegmentationTaskMetric`（`seg_tools.py:239-266`）。

掩码 `H×W bool` 直接存会爆，采用 **RLE 游程编码**：

- `rle_encode(mask_bool) -> {"size": [H, W], "counts": [int, ...]}`：列优先（COCO 约定）游程，首段为 0 像素的游程长度，交替计数。
- `rle_decode(rle) -> mask_bool`：还原 `H×W` bool。
- 单元测试覆盖往返等价（含全 0、全 1、空尺寸、非方形）。

每个 shard 写出 `shard_result.json`：每张图的 `pred_labels` / `pred_scores` / `pred_masks_rle` 与 `gt_labels` / `gt_masks_rle`，外加 `class_names`、`images_with_labels`、计时。父进程读所有 shard，解码 RLE，按图调用 `update_metric`（`seg_tools.py:257`），最后 `compute_aggregated_values()`，产出与串行一致的 `seg_eval_summary.json`。

## 3. seg eval/infer 并行入口

### 3.1 eval

`run_eval` / `run_semantic_eval` 前面加并行调度层 `run_parallel_seg_eval(args)`，进入条件与 det 对齐：

- `device == "auto"`、非 shard 子进程、非 dry-run；
- `query_gpu_inventory` + `filter_high_memory_gpus` 后 ≥2 张空闲卡，且样本数 ≥ shard 数。

满足则按 `eligible_gpus` 数起子进程：`launcher.py eval --task seg ... --shard-index i --num-shards n --output-dir <shard_tmp> --skip-important-artifacts`，pin `CUDA_VISIBLE_DEVICES`。子进程是"shard 模式"，跑现有串行逻辑但**写 shard 结果而非最终 summary**。父进程合并 → 写最终 summary（语义/实例各自的文件名不变）。

不满足条件 → 退回单卡，但用 `select_fallback_single_gpu` 选最空的卡。

### 3.2 infer

`run_infer`（`seg_tools.py:207`）加并行层，按图索引分片；**无需合并指标**——子进程各写可视化/JSON 到 shard 目录，父进程把各 shard 的 `images/` 汇集到最终目录（复用 det 的 `copy_tree_contents` 思路）。seg infer 复用 `infer` parser，已有 shard 旗标，直接接上。

### 3.3 CLI 旗标

`eval` parser（`interactive.py:2689`）新增（全部 `argparse.SUPPRESS`，与 `infer` parser 对齐）：
`--shard-index` / `--num-shards` / `--selected-splits` / `--multi-output-root` / `--skip-important-artifacts` / `--dry-run`。

## 4. 吞吐优化（P2）

### 4.1 预取重叠

后台线程（单 worker 生产者队列）预读+解码下一张图（eval 同时预载 GT 掩码/标签）。`predict()` 接受 PILImage，把预解码的 PIL 图直接传入，使 CPU 解码与 GPU 前向重叠。对 eval 与 infer 的串行内循环均适用。

### 4.2 语义掩码 LUT

`_load_semantic_mask`（`seg_tools.py:114`）当前对 `类别 × 标签` 做整图 `np.all` 扫描，O(类别×标签) 次全图遍历。改为查找表：

- 单通道标签：构建 `lut[max_label+1] = internal_id`（缺省 -100），一次索引完成映射。
- RGB 元组标签：把每像素 `(r,g,b)` 打包成 `r<<16 | g<<8 | b` 的 int，对打包后的标签建 dict→internal，向量化映射。

单测验证 LUT 版与旧版在多类别/RGB 标签/ignore 类别下输出逐元素一致。

## 5. 工程对齐（P3）

- **计时**：内循环累加 `infer_time_sum_ms`，summary 写 `avg_infer_time_ms`。
- **run_meta.json**：记录 checkpoint、data、split、num_images、device、产物路径、设置（对齐 det `write_run_meta`）。
- **--dry-run**：打印计划（设备模式、split、样本数、计划产物），不执行。
- **容错**：每图 `predict` 包 `try/except`，失败记一条 warning 并计入"失败图数"，不中断整轮；summary 记录失败计数。
- summary schema 在不破坏现有字段前提下，新增上述字段。

## 测试策略（TDD）

纯函数单测（不需 GPU/模型）：

- `filter_indices_for_shard`：取模分片覆盖、边界（shard 数 > 样本数）。
- `rle_encode`/`rle_decode`：往返等价，含全 0/全 1/空/非方形。
- 混淆矩阵合并：多 shard 求和 == 单进程结果。
- 语义 LUT：与旧 `_load_semantic_mask` 逐元素一致。
- `select_fallback_single_gpu`：给定 mock GPU 清单选最空卡；空清单返回 None。
- shard 结果序列化/反序列化往返。

集成（mock GPU 清单 + 1-shard 退化路径）：

- `num_shards=1` 时并行入口产物与串行 `run_eval`/`run_semantic_eval` 逐位一致。
- mock `query_gpu_inventory` 返回 <2 卡时走单卡退化。

## 兼容性与风险

- det 抽取为独立提交，det 现有测试必须全绿后才叠加 seg。
- 所有并行仅在 `device=auto` + 多卡时启用；指定 `--device` 或单卡环境行为不变（除单卡改为负载感知选卡）。
- 子进程通过 `launcher.py eval/infer` 自调用，与 det 完全相同的进程模型。

## 非目标

- 不引入 batched predict（上游未实现 `predict_batch`）。
- 不接 Fabric `predict_task` 管线。
- 不改 seg export / eda / curate。
- 不改训练。
