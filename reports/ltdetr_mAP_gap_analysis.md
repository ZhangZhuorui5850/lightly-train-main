# LT-DETR 训练结果差异分析报告

**日期**：2026-05-27
**对比对象**：自有训练脚本（`train_det.py` 系）vs `train_det_ltdetr_baseline.py`
**结论一句话**：两者最终 mAP 差距（0.70 vs 0.75）**不是学习率问题**，而是**按 epoch 对齐导致优化更新步数相差约 3.3 倍**，DETR 类模型因此欠训。

---

## 1. 现象

| 项目 | 我方 | Baseline |
|---|---|---|
| GPU 卡数 | 4 | 6 |
| 总 batch size | 96 | 24 |
| `model_args` 里设的 lr | 5e-5 | 1e-4 |
| 总 steps | 26,000 | 85,000 |
| 对应 epoch | ~250 | ~250 |
| 最终 mAP | **0.70** | **0.75** |

前提：同一份数据、同一份 backbone 权重、同一模型。两条 loss 曲线均「看似收敛」。

---

## 2. 排除项：学习率不是原因

库对学习率做了**自动 sqrt 缩放**，`model_args["lr"]` 只是基准值，并非最终送入优化器的值。

代码（`src/lightly_train/_task_models/dinov3_ltdetr_object_detection/train_model.py:404`）：

```python
lr = self.model_args.lr * math.sqrt(
    global_batch_size / self.model_args.default_batch_size   # default_batch_size = 16
)
```

即：**实际 lr = 设定 lr × √(总 batch / 16)**。

代入数字：

| | 设定 lr | 总 batch | 实际 lr |
|---|---|---|---|
| 我方 | 5e-5 | 96 | 5e-5 × √(96/16) = **1.2247e-4** |
| Baseline | 1e-4 | 24 | 1e-4 × √(24/16) = **1.2247e-4** |

两者经各自 batch 缩放后**实际学习率精确相等**。故学习率不是差距来源。

---

## 3. 根因：按 epoch 对齐 → 更新步数相差 3.3 倍

`steps` 参数是**整个训练的总优化步数（总参数更新次数）**，不是每个 epoch 的步数。

- 我方：batch 96，250 epoch → **26,000 次**权重更新
- Baseline：batch 24，250 epoch → **85,000 次**权重更新
- 比值 85000 / 26000 ≈ 3.3，与 batch 比 96/24 = 4 同量级（差异来自 epoch 数略不同）

**关键误区**：相同 epoch ≠ 相同训练量。batch 大 4 倍，每个 epoch 喂入的图片数相同，但**权重更新次数少约 4 倍**。两者看过的图片数几乎一致（都约 250 轮），但我方只用了 baseline 约 1/3 的梯度更新。

DETR / LT-DETR 对**更新次数**极其敏感（匈牙利匹配 + decoder query 收敛慢），不是「看够图片数」即可。因此我方曲线虽看似收敛，实为**欠训**，0.70 vs 0.75 基本由此 3.3 倍更新差距造成。

### 加剧因素（均按 step 计，非按图片计）

- **EMA**：默认 `use_ema_model=True`，导出的是 EMA 权重，momentum 0.9999。26,000 步平滑出的 EMA 远不如 85,000 步充分。
- **Warmup**：`lr_warmup_steps` 固定约 2000 步。我方 2000/26000 ≈ 7.7% 处于 warmup，baseline 仅约 2.4%，有效训练步数占比更低。

---

## 4. 次要差异（影响小，已排除为主因）

- **卡数 4 vs 6**：有效 lr 已对齐；SyncBN 统计基于全局 batch，影响有限。
- **data 传参方式**（路径字符串 vs dict）、`num_workers`、stdout hook、绘图与指标收集：只影响日志/速度/产物，不动训练数值。
- **precision**：两边均为 `bf16-mixed`（库默认相同）。
- **大 batch 泛化间隙**：即便步数对齐，batch 96 vs 24 仍可能让我方低几个点（梯度噪声小、隐式正则弱），是叠加效应而非主因。

---

## 5. 建议（公平对比 / 追回精度）

按推荐度排序：

1. **按 step 对齐（最公平）**：将我方 `steps` 提至与 baseline 同量级（~85,000）。代价是 batch 96 下相当于约 1000 epoch（约 4 倍图片量），算力成本高但可比性最佳。
2. **将总 batch 降至 24**：相同 250 epoch 自然得到 ~85,000 步，更新次数对齐，并回到 LT-DETR 常用小 batch 区间，同时规避大 batch 泛化间隙。
3. **折中**：`steps` 提到 5 万左右并适当延长训练，通常可追回大部分精度。

> 注意：两脚本均未显式设随机种子，DDP 卡数不同 + dataloader shuffle 会带来 run-to-run 抖动，单次结果不宜下死结论。

---

## 6. 结论

- 学习率不是原因：sqrt 自动缩放后两者实际 lr 相等（均 1.2247e-4）。
- 真因：用 **epoch 对齐**而非 **step（更新次数）对齐**，大 batch 使我方更新次数缩水约 3.3 倍，导致 DETR 欠训；EMA、warmup 的按步计特性进一步放大该劣势。
- 对策：以 step（更新次数）为对齐基准，或将 batch 降到与 baseline 相同量级。
