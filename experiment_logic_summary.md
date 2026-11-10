# Night3R 实验部分逻辑梳理

## 1. 总体思路

当前实验部分不是简单罗列主结果和消融实验，而是按照以下证据链展开：

> Night3R 是否有效 -> 极低光下是否仍然有效 -> 提升是否真正来自对齐事件 -> BRG 为什么采用当前设计 -> 异构监督训练是否合理。

实验依次回答以下问题：

1. Night3R 在真实低光数据上能否超过现有 RGB-only 和 event-only 方法？
2. 当 RGB 信息进一步减少时，Night3R 是否仍能保持稳定？
3. 性能提升是否真正来自同步事件，而不是额外参数或事件值分布？
4. 为什么使用 BRG，而不是 Cross-Attention、Concat-MLP 或直接替换 RGB token？
5. Event density、occupancy、polarity balance 和 source calibration 分别起什么作用？
6. 不同数据来源具有不同监督时，动态损失组合是否合理？

## 2. 实验协议与公平性

### 2.1 数据集用途

- MVSEC、DSEC 和 DL3DV 用于训练。
- MVSEC 和 DSEC 具有当前协议下可用的深度 GT，因此用于定量深度评测。
- DL3DV 不作为 metric-depth 测试集，只用于增加 RGB 场景、低光外观和 Teacher point-map supervision 的多样性。
- DL3DV 序列首先通过 Practical-RIFE 进行时间插帧，再通过 DVS-Voltmeter 合成与高帧率 RGB 对齐的事件流。
- MVSEC 和 DSEC 的事件评测使用真实同步事件，而不是 DL3DV 的合成事件。

### 2.2 训练样本与监督

每个训练样本包含低光 RGB pair 和对应的 event voxel pair。根据数据来源，还可以提供：

- 注册的正常曝光 RGB reference pair；
- Metric depth；
- 对应监督需要的元数据；
- 有效事件支持。

当前精简后的核心训练监督包括：

- Point-Map Distillation Loss；
- Metric Depth Loss；
- Gate Regularization Loss。

动态 loss normalization 是损失组合方式，不是独立的训练 loss。

### 2.3 评测公平性

- 所有方法使用相同的 128 对相邻测试帧。
- MVSEC 和 DSEC 使用相同的 pair ordering 和有效深度 mask。
- 极低光实验中，所有 RGB 方法接收完全相同的 darkened RGB。
- Night3R 在不同曝光下始终接收原始同步事件。
- EventVGGT 作为不依赖 RGB 曝光的 event-only 参考。
- 跨模型比较采用 per-pair median alignment，因为不同 foundation model 的输出尺度定义不同。
- Real/Zero/Shuffle 和 Night3R 内部消融同时报告 raw 与 median-aligned 指标，因为它们共享模型和输出尺度定义。

## 3. 整体性能：真实低光结果

### 对应内容

- `Comparison With State-of-the-Art Methods`
- Table 0：`tables/table0_main_results`

### 研究问题

Night3R 在真实低光 MVSEC 和 DSEC 上，能否超过 RGB-only 与 event-only 方法？

### 对比方法

- MASt3R：RGB-only geometry foundation model；
- Dark3R：低光 RGB reconstruction model；
- VGGT：RGB-only geometry model；
- EventVGGT：event-only geometry model；
- Night3R：RGB-event reconstruction model。

### 主要结论

- Night3R 在 MVSEC 和 DSEC 上取得最好的整体 median-aligned 结果。
- RGB-only 方法受到低光图像信息缺失的限制。
- EventVGGT 在 MVSEC 上具有竞争力，但在 DSEC 上明显退化，说明 event-only geometry 的跨数据集稳定性有限。
- Night3R 保留 RGB geometry prior，同时使用事件补充低光下缺失的观测信息，因此具有更稳定的跨数据集表现。

这一部分回答最基本的问题：Night3R 是否有效。

## 4. 主要鲁棒性结果：极低光压力测试

### 对应内容

- `Controlled extreme-low-light evaluation`
- Table 5：`tables/table5_extreme_low_light`

### 研究问题

随着 RGB 曝光持续下降，Night3R 是否比 RGB-only 方法更加稳定？

### 实验控制

- 使用相同测试 RGB pair。
- 将 RGB 曝光控制为 `0.20`、`0.10` 和 `0.05`。
- 所有 RGB 方法接收相同 darkened RGB。
- Night3R 的事件输入保持不变，并继续使用原始同步事件。
- EventVGGT 作为 exposure-independent event-only reference。

### 主要结论

- RGB-only 方法随着曝光下降持续退化。
- 最严重退化出现在 `e=0.05`。
- Night3R 在 MVSEC 上跨曝光保持相对稳定。
- Night3R 在 DSEC 最低曝光下仍保持较好的深度精度。
- RGB 越暗，Night3R 与 RGB-only 方法的差距越明显，说明同步事件能够在 appearance information 严重减少时提供补充几何信息。

这一实验属于论文的主要结果，而不是普通 stress-test 消融，因为极低光稳定性正是 Night3R 的核心应用价值。

## 5. 事件贡献归因：Real、Zero 和 Shuffle

### 对应内容

- `Controlled Analysis of Aligned Event Measurements`
- Table 3：`tables/table3_event_input_control`

### 研究问题

Night3R 的提升是否真正来自有意义且正确对齐的事件信息？

### 控制变量

以下内容保持完全不变：

- Night3R checkpoint；
- RGB 输入；
- 测试 pair；
- GT 和有效 mask；
- 评测代码与输出尺度定义。

实验只改变 event tensor：

- **Real**：使用原始同步事件；
- **Zero**：将 event tensor 全部置零；
- **Shuffle**：保留所有事件值，但破坏 RGB-event 对齐及事件通道语义。

### Shuffle 的具体实现

对第 `n` 个评测 pair：

- 沿高度方向循环移动约 `H/3`；
- 沿宽度方向循环移动约 `W/5`；
- 沿 event-channel 维循环移动 `1 + n mod (C-1)`，其中 `C=2B`；
- 该操作完全确定，不依赖随机种子。

Shuffle 保持：

- Event tensor shape；
- 所有事件数值；
- 全局事件值分布；
- 全局 event density；
- 全局 occupancy；
- 非零事件总量。

Shuffle 破坏：

- Event 与 RGB 的空间对齐；
- Event 与场景边缘和运动结构的对应；
- 原始 temporal-bin 顺序；
- 原始 polarity-channel 对应关系。

### 主要结论

- Real 优于 Zero，说明模型确实使用了事件分支。
- Real 优于 Shuffle，说明提升依赖正确对齐的事件结构。
- 性能提升不能仅由额外输入通道、额外参数或事件 marginal distribution 解释。

这一部分将整体性能提升归因到同步事件本身，而不只是说明两者存在相关性。

## 6. 融合方式与 BRG 设计

### 对应内容

- `Analysis of Fusion and Reliability Modeling`
- Table 1：`tables/table1_fusion_BRG.tex`

### 研究问题

为什么使用 BRG，而不是其他 RGB-event fusion operator？

### 对比内容

- Bounded Reliability Gate；
- Concat-MLP；
- Scalar-gated Cross-Attention；
- Additive Projection；
- Direct RGB Replacement；
- 不同 adapter optimization scope。

### 主要结论

- Scalar-gated Cross-Attention 和 Additive Projection 在两个数据集上均出现退化。
- Direct RGB Replacement 可以改善部分 MVSEC 指标，但会严重损害 DSEC，说明覆盖 RGB representation 会造成明显跨域问题。
- 仅优化 patch embedding 和 fusion layers 不如完整 Event Adapter 优化。
- Concat-MLP 与完整 BRG 数值接近，因此不能声称 BRG 在所有学习式融合方式上都有巨大优势。
- BRG 的优势是跨数据集表现更均衡，并且具有明确的结构属性：
  - 保留预训练 RGB pathway；
  - 以 residual 方式注入事件；
  - 使用显式有界且可检查的 reliability coefficient。

因此，BRG 的主要价值是竞争性精度、跨域稳定性和可解释事件调制的组合，而不是在单一指标上形成巨大差距。

## 7. Reliability Statistics 与跨来源校准

### 对应内容

- `Event statistics and cross-source calibration`
- Table 2：`tables/table2_event_statistics_source_calibration`

### 研究问题

Event density、occupancy、polarity balance 和 source calibration 分别起什么作用？

### 分析内容

- Event density；
- Spatial occupancy；
- Polarity balance；
- Source-statistic normalization；
- Source-dependent bias。

### 主要结论

- 各个事件统计量之间的差距较小。
- 部分简化版本可能在单个指标上略好。
- 现有结果不支持“每个统计量都独立带来显著提升”的强结论。
- 完整统计量在 MVSEC 和 DSEC 之间提供更均衡的结果。
- Source-statistic normalization 带来较小但一致的跨数据集收益。
- Source bias 主要调节 MVSEC 和 DSEC 之间的性能 trade-off。

因此，reliability estimator 更适合解释为跨事件来源的校准机制，而不是一个能够独立产生巨大深度收益的模块。

## 8. 异构监督与动态损失分析

### 对应内容

- `Analysis of Training With Heterogeneous Supervision`
- Table 4：`tables/table4_dynamic_loss_supervision.tex`

### 研究问题

不同训练来源具有不同标注时，Night3R 如何稳定地组合这些监督？

### 分析内容

- Point-map consistency；
- Metric-depth supervision；
- Active-loss normalization；
- Source-dependent loss weights；
- Static plain loss 对照。

### 主要结论

- 移除 metric-depth supervision 会造成最明显的性能下降。
- Metric depth 是最主要的尺度敏感几何约束。
- Teacher point-map consistency 主要改善 MVSEC，在 DSEC 上接近中性。
- Point-map consistency 应被视为 complementary privileged supervision，而不是 metric depth 的替代品。
- Active-loss normalization 的直接数值收益较小，但可以避免不同样本启用不同 loss 时改变整体目标尺度。
- Unit source weights 和 static plain loss 对 DSEC 的影响更明显，跨数据集结果更不均衡。

这一部分说明 source-aware objective 是处理 heterogeneous supervision 的实用机制，同时明确 metric-depth supervision 才是主导训练信号。

## 9. 当前实验部分的最终结论

当前实验支持三项主要结论：

1. Night3R 在真实低光 MVSEC 和 DSEC 上取得更好的整体深度结果，并在 RGB 持续变暗时保持更稳定的性能。
2. Real/Zero/Shuffle 控制实验证明，性能提升来自正确对齐的事件测量，而不是额外参数、输入通道或事件边缘分布。
3. BRG 和 reliability modeling 的价值主要体现在保留 RGB geometry prior、限制事件残差、提高跨数据集稳定性以及校准不同来源的事件分布。

同时，实验也保持了必要的结论边界：

- 不声称 BRG 在所有指标上大幅超过 Concat-MLP；
- 不声称每个 reliability statistic 都独立产生显著收益；
- 不使用缺少有效深度 GT 的 DL3DV 进行 metric-depth 定量结论；
- 将 metric-depth supervision 明确识别为最主要的训练信号。

## 10. 需要保持一致的论文口径

后续更新论文和表格时，需要持续检查以下一致性：

- Method、Table 4 和最终 checkpoint 使用相同的 loss 配置；
- Table 0、Table 3 和 Table 5 使用同一个最终 Night3R checkpoint；
- 同一数据集在不同表格中的 Night3R 指标采用相同评测协议；
- Cross-model comparison 使用 median-aligned 指标；
- 同 checkpoint 控制实验同时报告 raw 和 median-aligned 指标；
- Shuffle 始终表示确定性的空间循环移动和 event-channel 循环移动，而不是随机像素置乱；
- DL3DV 只作为训练数据，不作为当前 metric-depth 测试集。
