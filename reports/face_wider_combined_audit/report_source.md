# WIDER FACE combined 训练收益下降调研

## 技术结论：优先修正数据监督与尺度分布

调研日期：2026-09-10。两组模型均按用户确认的 DINOv3 ViT-S/16 分析。目标是解释 combined + 关闭 Random Zoom Out 后收益有限，并确定下一次训练最有价值的改动。

本次找到两个有直接数据证据的机制：**切片产生边界标签缺失风险；训练样本向较大、较容易的人脸迁移，极小脸的单位训练曝光下降。** 它们能够解释部分提升被抵消的现象，具体贡献需要消融实验确定。当前证据支持优先处理这两项。

实际结果：两组都开启 SAHI 时，mAP 从 0.28347 降到 0.27406，AP_small 从 0.16077 降到 0.15006。80,000 step 的曲线已经进入平台期，继续增加训练时长的优先级低。关闭 Zoom Out 的参数传递正确；冻结 backbone 是双方共有的建模条件，应放在数据对照之后研究。

本报告由本地全量标签与 manifest 审计、图像字节校验、训练/评估源码检查、用户提供的服务器结果截图，以及公开论文共同支持。因果结论采用“候选机制”表述；单次实验的约 1 个百分点差距仍需多随机种子确认。

## 1. 指标口径与实测结果

AP/mAP 范围为 0–1；差值以百分点表示。mAP 为 IoU 0.50–0.95 的平均 AP。AP_small/medium/large 沿用当前 COCO 风格评估器的原图像素面积分组。后文“640 等效短边”定义为 640×min(YOLO归一化宽,归一化高)，描述模型输入尺度；两种尺寸口径各自独立。

| 指标 | 原始模型 SAHI | combined 全图 | combined SAHI | SAHI 对齐后的差值（百分点） |
|---|---:|---:|---:|---:|
| mAP | 0.28347 | 0.27175 | 0.27406 | −0.94 |
| AP50 | 0.54029 | 0.52188 | 0.52885 | −1.14 |
| AP75 | 0.26447 | 0.25318 | 0.25301 | −1.15 |
| AP_small | 0.16077 | 0.14657 | 0.15006 | −1.07 |
| AP_medium | 0.59008 | 0.58315 | 0.58047 | −0.96 |
| AP_large | 0.72532 | 0.72715 | 0.72171 | −0.36 |
| AR_small | 0.20330 | 0.19165 | 0.19393 | −0.94 |

来源：本次会话中三组 metrics_summary.json 截图手工转录。共同测试图片数量 1,615；两组 SAHI 的 score threshold=0.3、overlap=0.2、slice_size=null。旧服务器的准确代码版本与隐式默认值尚待归档。combined 自身 SAHI 带来的 mAP 增量仅 0.23 个百分点，AP_small 增量约 0.35 个百分点。

图中的 validation mAP 约在前 16,000 step 后明显趋缓，小脸仍缓慢改善，中大脸略有回落。这个观察支持平台期判断；缺少逐次验证原始数据，报告保留截图级精度。

## 2. combined 是本项目生成的 WIDER FACE 派生训练集

WIDER FACE 官方数据含 32,203 张图、393,703 张脸，强调尺度、姿态与遮挡变化，并提供官方划分及难度评估。[WIDER FACE 官方项目](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)

当前 combined 的来源由 preparation_summary.json 和 augmentation_manifest.jsonl 明确记录：从本地 face_yolo_wider 的 train 生成 Copy-Paste 与 tile，保留原图。它属于本项目派生数据版本。该版本包含 YOLO 框，blur、occlusion、invalid 等官方属性的恢复需要原始标注；当前内部 test 和 COCO 风格指标应作为本项目评估结果解读。

全量审计结果：

| 子集 | 图片数 | 框数 | 640等效短边中位数 |
|---|---:|---:|---:|
| 原始 train | 12,883 | 156,219 | 9.37 px |
| 新增 tile | 6,442 | 135,455 | 17.69 px |
| 新增 CP 图片（含原有框） | 1,466 | 6,890 | 38.75 px |
| combined train | 20,791 | 298,564 | 14.05 px |
| test | 1,615 | 22,866 | 9.37 px |

CP 真正新增的粘贴脸为 2,931 个，其余为接收图已有框。Combined 中 tile 图片占 30.98%，CP 图片占 7.05%，原始全图占 61.97%。6,442 张 tile 来自 3,454 张源图，新增数据复用了既有人脸与场景。

本地 val/test 的文件名集合、图片字节与标签字节均与 original 完全一致（逐文件 SHA-256 比较）。manifest 中全部 donor、recipient、tile source 均归属本地 train；原始 split 之间同名图片交集为 0。该检查确认已记录的派生来源遵守划分，跨不同文件名的内容重复和跨服务器历史划分仍需独立核验。

来源：audit_results.json；输入目录 datasets/face_detect/face_yolo_wider 与 face_yolo_wider_combined_v1。本次读取全部 298,564 个 combined train 框，检查类别/边界之外的基础数值有效性，并使用 tile 重建核验几何对应关系。基础非有限值及非正宽高计数为 0。

## 3. 最具体的质量风险：裁剪后可见的人脸被删除标签

生成器 make_tile 在原框保留面积低于 50% 时跳过该框，图像依然保留该区域。这会产生“可见局部人脸只有图像监督、缺少对应正样本框”的风险；检测损失会把某些对应预测归入背景。

全量重建 6,442 个裁剪窗口发现：

- 2,405 张 tile 存在被过滤的可见原框，占 tile 的 37.33%，占 combined train 的 11.57%。
- 共 3,915 个原框在窗口内仍有正面积交集、被低于 50% 的规则移除。
- 其中 1,241 个满足“仍保留至少 25% 原框面积，且可见区域 640 等效短边至少 8 px”。这是一组优先人工复核候选。
- 另有 3,785 个保留标注的框发生部分裁剪。对于保留框，当前目标框对应裁后边界；原图完整框提供另一种几何监督。

**3,915 是几何候选数，具体可辨认程度需要逐图判断。** 大部分边缘碎片可能属于合理过滤；把全部候选算作明确漏标会高估问题。已人工查看 tile_006379_wider_train_19_Couple_Couple_19_693.jpg：底部可见额头和双眼的较大半张脸，其原框保留 45.36%，在标签中被过滤；可见区域短边约 160.29 个640等效像素。该例证明候选中确实存在语义上仍可辨认的人脸。

源码位置：datasets/convert_datasets/convert_tools/face_wider_prepare.py 的 make_tile，约 405–411 行。audit_results.json 保存了 8 个可复核的大面积例子及 source/crop/box 坐标；verify_tiles.py 将相同过滤规则重建为 YOLO 框，与实际标签逐项核对。

解释力度：**质量风险已确认；对 AP 下降的贡献待实验。** 它还可能影响中大脸，与本次中大脸同向下降相容。

建议在新派生版本中采用统一边界规则：对于仍有明显可辨认人脸的候选，重新选取窗口、保留经过审阅的裁后框，或使用训练端确实支持的 ignore 区域。当前 YOLO 路径优先重新选窗口，避免引入新的 ignore 协议。单纯提高面积保留阈值会扩大被删除的可见区域，应先定义完整的图像与标签处理策略。

## 4. 最强分布线索：极小脸曝光降低，较大脸监督增加

以下比例以各子集全部框为分母，尺寸为640等效短边，统计发生在在线增强之前。

| 短边 | 原始 train | combined train | test |
|---|---:|---:|---:|
| 小于8 px | 40.49% | 24.61% | 40.82% |
| 8–16 px | 29.28% | 31.85% | 31.24% |
| 16–32 px | 18.07% | 28.05% | 17.88% |
| 至少32 px | 12.17% | 15.49% | 10.05% |

放大切片达成了设计目标，同时改变了训练任务的难度分布。test 仍有约四成框处于最困难的 <8 px 区间，combined 则把更多监督投入 8–32 px。tile 的 <8 px 比例约 7.55%；CP 粘贴目标是12–28 px。

按均匀图片采样、相同有效 batch 和 step，原始 <8 px 框的每图均值为 63,246/12,883=4.91，combined 为73,468/20,791=3.53，**每次图片采样得到的极小脸框期望减少约28.0%**。这比单看框总量更贴近优化器收到的数据。目标检测损失存在框数归一化、随机裁剪、匹配等机制，因此该数值描述输入曝光，实际梯度权重需要运行时统计。

超过100张脸的训练图一直是235张，它们的图片采样概率由1.82%降至1.13%。切片把密集场景拆成较容易的局部视野，而全图高密度监督的占比降低。

解释力度：**分布迁移和曝光变化已确认；对性能的净作用待消融。** 多尺度采样可改善小脸学习，但它需要围绕测试尺度分布设计。PyramidBox 同时强调小脸上下文、低层特征与 data-anchor-sampling；该研究支持尺度分布值得控制。[PyramidBox 论文](https://arxiv.org/abs/1803.07737)

## 5. 关闭 Zoom Out 与放大切片叠加，可能削弱尺度覆盖

train_face_wider.py 明确传入 random_zoom_out=None，底层仅在参数非空时构造 RandomZoomOut，因此开关生效。默认训练仍有 probability=0.8 的 RandomIoUCrop、480–800 的 ScaleJitter 和翻转/颜色增强。固定 image_size=640 时，多尺度采样继续执行。

关闭 Zoom Out 移除了随机缩小来源；新增 tile 放大脸；在线 RandomIoUCrop 进一步改变局部视野。三者叠加使尺度覆盖和上下文发生变化。这提供了另一条可能抵消收益的机制。关闭 Zoom Out 对某些极小脸可能有利，对跨尺度泛化也可能带来代价，净作用由单独对照决定。

当前训练量为80,000×32=2,560,000次图片曝光，约123.1次 combined 数据遍历；自动梯度累积为1。学习率基础值按sqrt(32/16)缩放到约7.07e-5，随后进入调度器。前次对话中基于20,000 step的曝光不足推测已撤回。

DINOv3 官方展示了冻结特征在密集任务上的能力，冻结训练本身属于受支持方案。本次双方模型按用户确认一致，冻结 backbone 只能作为当前系统的适配限制候选。[DINOv3 官方实现](https://github.com/facebookresearch/dinov3)

## 6. Copy-Paste 的新增监督有限，外观合成仍有风险

本版本1466张CP图只新增2931个人脸，约占combined全部框的0.98%。它选择原先缺少小脸的接收图，能够改变图片级覆盖率；其直接数量贡献有限。

实现采用框裁剪、10%上下文扩展、椭圆羽化mask、局部亮度匹配。这样生成的脸可能缺少头部/身体上下文，并且边缘/姿态与背景存在差异。供体源自已有train，真实场景多样性的增量有限。代码的目标尺寸为12–28 px线性均匀采样。

Simple Copy-Paste 的论文在 COCO/LVIS 实例分割上展示了收益，其真实实例mask和任务条件与本项目的椭圆人脸粘贴存在差异。因此论文能够支持开展实验，收益幅度需要本地验证。[Simple Copy-Paste 论文](https://arxiv.org/abs/2012.07177)

解释力度：**样本占比与实现方式已确认，合成域差异是次级假设。** 首轮把CP分支单独去掉可检验它的净贡献，优先级低于tile边界规则与训练尺度分布。

## 7. SAHI 的收益受过滤、合并和100框评估上限影响

本地 predict_sahi 按模型 image_size 切图，同时推理全图。默认tile NMS IoU=0.3，全图/局部一致性阈值=0.1；在合并前应用用户score threshold，本次为0.3。相邻密集人脸可能在合并时竞争；若新模型的小脸置信度偏低，0.3也可能截断收益。

这与普通全图路径存在口径差异：普通评估使用threshold=0.0获取全量预测计算mAP；SAHI使用真实阈值过滤后计算mAP。两组SAHI都用0.3时仍可作相同部署设置下的比较。SAHI开关前后的数值同时混合了尺度、合并与阈值影响。

当前mAP默认maxDets为[1,10,100]。本地test有38张图超过100个GT，总计有5290个GT超出每图100框容量，占全部GT的23.13%。假设每张图的前100框都完全正确，整体逐框召回上限也只有约76.87%；该上限用于说明容量约束，具体COCO插值AP及分尺寸AR另有计算规则。两次评估共享这个限制，主要影响可观察提升空间。

训练模型的query数为300，与评估的100框限制各自独立。应同时保留标准100框结果和明确标记的maxDets=300诊断结果；官方WIDER easy/medium/hard AP需要官方协议。

SAHI论文主要在VisDrone/xView和相应检测器上验证收益。本项目LTDETR自带的切片/合并实现需要独立调参。[SAHI论文](https://arxiv.org/abs/2202.06934)

## 8. 最小验证顺序：先质量，再增强消融

1. **修复验证优先：** 对可见框过滤候选进行人工抽查，围绕原始图/裁剪图/标签配对，制定边界规则。生成独立v2数据，保留v1作对照。先dry-run，staging校验后发布。使用固定20,000 step预算比较原v1 tile和边界修正版tile，唯一变化为裁剪规则。
2. **完成2×2对照：** A原图+Zoom Out开启；B原图+关闭；C原图+修正版tile+开启；D原图+修正版tile+关闭。统一vits16、权重、冻结状态、batch32、学习率、step、seed和验证集。首轮20,000 step与现有平台拐点匹配，属于筛选预算。随后对胜出和基线以完整预算及至少3个seed复验。
3. **按证据调整混合权重：** 在tile有效时对比当前约31%与约15%的tile采样占比，检查<8 px全图召回及密集场景召回。CP作为独立追加分支最后验证。解冻backbone、固定高分辨率放在这些实验之后。

每次记录：全图mAP、AP_small、<8/8–16/16–32/≥32 px召回、GT>100图召回；SAHI保存实际切片尺寸、阈值和合并参数。best按相同val指标选择，test留给最终比较。SAHI阈值0.05/0.1/0.3及合并设置的小范围扫描在val上完成，再锁定参数评估test。

成功标准：修正版在多seed中稳定提高小脸AP/召回，且中大脸指标保持可接受范围。以配对图片bootstrap评估测试集不确定性，以多seed评估训练波动；两者分别记录。

## 9. 可复现方法、边界与仍待回答的问题

运行：`conda run -n lightlytrain python reports/face_wider_combined_audit/audit.py`，再运行同目录verify_tiles.py。audit_results.json保存全量统计与边界样本，tile_verification.json保存几何重建结果。主检查使用Pillow读取源图尺寸，NumPy计算裁剪交集及640等效短边，SHA-256比对验证/测试图片和标签。全部操作为读取数据与写入reports目录。

现有基础检查覆盖框格式的有限数值与正宽高；tile几何重建进一步验证裁剪公式和实际标签一致性。语义漏标、原始官方invalid属性、CP图像真实性、跨文件名重复和旧基线train与当前test的重叠属于后续质量审查项。test在两服务器上的完整哈希需要服务器执行同样检查才能最终确认。

本次接受双方vits16的明确说明。旧训练数据版本、旧步数/学习率/随机种子缺少完整记录，因此结果支持“这次方案总体略降”，各增强的单独贡献保留待验证。截图中的曲线支持平台期判断，原始event/逐图预测和checkpoint缺少本地访问，因此本次提供代码/数据机制诊断，未实施训练复验。

公开研究用于解释机制与设计验证；单项研究的收益属于其任务和实验条件。此次数据派生风险主要来自项目生成规则，WIDER FACE公开来源与官方标注质量需要分别讨论。

本次修正此前判断：800输入讨论与当前640实验无关；训练量按80,000 step、batch32计算；冻结backbone和Zoom Out关闭各自属于待消融的建模条件。最先要解决的是可见人脸的边界监督，以及极小脸在combined中的曝光下降。

## 10. 证据索引

- 用户截图：原始SAHI、combined全图、combined SAHI的metrics_summary与run_meta，以及80,000 step训练曲线；模型型号依据用户明确确认。
- 本地数据：datasets/face_detect/face_yolo_wider；datasets/face_detect/face_yolo_wider_combined_v1；augmentation_manifest.jsonl；preparation_summary.json。
- 生成实现：datasets/convert_datasets/convert_tools/face_wider_prepare.py（make_tile、generate_tiles、generate_copy_paste、paste_patch）。
- 训练实现：train_face_wider.py；src/lightly_train/_task_models/dinov3_ltdetr_object_detection/transforms.py、train_model.py、task_model.py。
- 评估实现：tool_lib/det_infer.py；src/lightly_train/_metrics/mean_average_precision.py。
- 外部原始来源：[WIDER FACE](https://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)、[PyramidBox](https://arxiv.org/abs/1803.07737)、[Simple Copy-Paste](https://arxiv.org/abs/2012.07177)、[SAHI](https://arxiv.org/abs/2202.06934)、[DINOv3](https://github.com/facebookresearch/dinov3)。检索日期2026-09-10。
