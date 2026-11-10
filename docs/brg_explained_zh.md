# Night3R BRG: 统计依据与可学习部分

## 先给结论

BRG（Bounded Reliability Gate，受限可靠性门）**不是纯统计规则**。它由三件事组成：

1. 从原始 event voxel 确定性提取统计量，作为事件是否可信的证据；
2. 用一个可学习的门控网络，将统计量转换为每个 token 的注入强度；
3. 用另一个可学习的投影，将 event token 转换为要加到 RGB token 上的残差内容。

因此，统计量回答的是“这个 patch 的事件活动长什么样”；学习到的参数回答的是“这种活动是否值得用、用多少、具体补充什么信息”。

BRG 的目标是**修正** RGB token，而不是让 event 分支替代 RGB 几何主路径。

## 一次 BRG 融合发生了什么

对同一个空间 patch，BRG 接收：

- `R_i`：低照度 RGB encoder 输出的 RGB token；
- `E_i`：event encoder 输出的 event token；
- `s_i`：从原始 event voxel 计算的统计量。

其主路径可以简化为：

```text
raw event voxel V
  |-- 固定统计提取 ------------------> s_i = [density, occupancy, polarity_balance]
  |                                       |
  |                                       +--> 可学习的统计归一化和 gate MLP --> g_i
  |
  +--> 可学习的 event encoder --> E_i --> 可学习的 event projection --> Delta E_i

R'_i = R_i + a * g_i * Delta E_i
```

其中：

- `g_i` 是单个标量 token gate，表示这个位置能注入多少事件残差；
- `Delta E_i` 是与 token 维度相同的向量，表示注入的事件内容；
- `a` 是全局事件残差尺度；
- `R'_i` 才送入原 MASt3R 的 cross-view decoder。

实际实现还对 `g_i` 和 `a` 设置了上限，并在无事件区域强制令 `g_i = 0`。

## 固定的统计部分

统计从**未经 event encoder 归一化改变幅值的原始 voxel**计算；它们本身不含可训练参数。

对每个像素，先计算各通道事件幅值总和：

```text
A(x, y) = sum_c |V(c, x, y)|
```

再得到三个统计图，并池化到与 token 网格相同的空间尺寸：

| 统计量 | 公式/含义 | 是否可学习 |
|---|---|---|
| `density` | `log(1 + A)`，事件活动总量 | 否 |
| `occupancy` | `1[A > 0]`，该位置是否存在事件 | 否 |
| `polarity_balance` | `|A_plus - A_minus| / (A_plus + A_minus + eps)`，正负极性是否严重失衡 | 否 |

其中 `occupancy` 还承担硬安全约束：若一个 token 对应区域没有事件，最终 gate 一定为零。这个规则不需要、也不能靠训练绕过。

统计提取实现见 `model/event_fusion.py` 中的 `compute_event_stats`。

## 哪些地方可学习

| 模块 | 输入与输出 | 学习什么 | 作用 |
|---|---|---|---|
| event encoder | voxel -> `E_i` | 如何从时间 bin、极性与空间结构提取事件特征 | 提供事件内容表示 |
| `EventStatNorm` 仿射项 | 固定统计 -> 标准化统计 | 每一维统计的缩放与偏置 | 缓解不同数据源/事件率分布差异 |
| `token_gate` | 标准化统计 -> gate logit | 哪种统计组合该信任、信任程度多大 | 决定“用多少” |
| `source_bias`（可选） | 数据来源 ID -> gate 偏置 | 各数据来源整体更保守或更积极的先验 | 处理不同传感器/数据构建方式 |
| `event_proj` | `E_i` -> `Delta E_i` | 事件特征该怎样转为 RGB token 残差 | 决定“注入什么” |
| `event_scale_logit` | 标量 -> `a` | 全局事件残差预算 | 限制事件整体影响力 |
| `channel_mlp`（可选） | RGB/event/stat -> 逐通道调制 | 每一 token 通道的细粒度调制 | 默认主路径关闭，避免任意改写 RGB token |

这里最容易混淆的点是：`token_gate` 的核心输入是统计量，而 `event_proj` 的输入是 event token。也就是说：

- **统计量控制量**：是否注入、注入强度；
- **event token 控制内容**：被注入的残差到底是什么。

## 门控公式与“受限”是什么意思

省略可选模块后，主路径等价于：

```text
s_hat_i   = EventStatNorm(s_i)
logit_i   = token_gate(s_hat_i) + source_bias
g_i       = gate_cap * sigmoid(bound(logit_i) / temperature) * 1[occupancy_i > 0]
Delta E_i = event_proj(LayerNorm(E_i))
a         = event_scale_max * sigmoid(event_scale_logit)
R'_i      = R_i + a * g_i * Delta E_i
```

以下项目是固定的设计约束或训练日程，而不是由数据学习出的规则：

- `occupancy` 掩码：无事件时 gate 必为零；
- `bound(logit)`：用 `tanh` 限制 gate logit，避免饱和和极端梯度；
- `gate_cap`：训练早期较小，随后按配置逐步放宽；
- `event_scale_max`：即使 `event_scale_logit` 增大，事件残差也有最大预算；
- 输出形式：主路径使用残差式 `RGB + residual`，不替换 RGB token。

因此，“bounded”不表示没有学习；它表示学习到的 gate 和残差不能无界地破坏预训练 RGB/MASt3R 的几何表示。

## 它靠什么学会可靠性

训练中没有“这个 event token 可靠/不可靠”的人工标签。BRG 的参数通过下游几何目标反向传播学习：当某类事件注入能降低点图蒸馏误差、深度误差或其他激活的几何损失时，相关参数得到正向更新；反之则被压低。

同时使用 gate regularization 防止走捷径：

- 限制平均 gate，防止所有位置长期开启；
- 惩罚低事件区域的 gate；
- 惩罚过大的 gate logit；
- 若启用逐通道调制，惩罚偏离恒等映射的幅度。

所以 gate 的语义是“**对最终三维重建是否有帮助的可学习代理**”，不是一个经过标注训练的真实物理置信度。

## 三个直观例子

### 1. 完全没有事件

`occupancy = 0`，所以 `g_i = 0`。无论 gate MLP 其他参数如何，输出都是：

```text
R'_i = R_i
```

模型回退到原 RGB 几何路径。

### 2. 有事件，但统计像热点或不稳定闪烁

`density` 高并不自动意味着 gate 高。`token_gate` 会根据三种统计的组合学习这种模式是否常常改善几何；配置中的上界和正则进一步限制它的影响。BRG 不能保证检测出所有坏事件，但它避免把“事件多”硬编码为“必然可靠”。

### 3. 有事件且确实帮助低照度匹配

gate MLP 给出非零 `g_i`，`event_proj` 输出有用的 `Delta E_i`，两者乘上受限全局尺度后作为小残差补给 RGB token。后续 cross-view decoder 使用这个融合 token 改善对应关系和点图。

## 它和三阶段训练的关系

BRG 属于训练代码中的 `fusion_adapter`。三个阶段并不改变 BRG 的统计公式或融合公式，只改变哪些参数允许更新：

| 阶段 | BRG 状态 | 其他可训练范围 |
|---|---|---|
| Stage 1 | BRG 的可学习部分已训练 | event encoder 仅开放 `patch_embed`；RGB/decoder/head 冻结 |
| Stage 2 | BRG 继续训练 | 完整 event encoder、cross-view decoder、下游 head 开放；RGB encoder 仍冻结 |
| Stage 3 | BRG 继续训练 | 在 Stage 2 基础上，仅额外开放 RGB encoder 末端若干 block，学习率很小 |

因此 Stage 1 的重点正是先让 BRG 学会以低风险方式工作；后两阶段让事件特征和下游几何逐步共同适配。

## 当前主配置中应关注的开关

以 `config/train_night3r_event_reliability_dsec_holdout_v1_8gpu.yaml` 为例：

- `fusion_type: brg`：启用 BRG；
- `stat_dim: 3`：使用上面的三个统计量；
- `event_stat_learnable_affine: true`：统计归一化有可学习仿射参数；
- `use_source_bias: true`：不同来源可有可学习 gate 偏置；
- `use_channel_modulation: false`：不启用可选的逐通道改写；
- `event_scale_init: 0.035`、`event_scale_max: 0.12`：事件残差从很小的预算开始且始终有上限；
- `gate_cap_final: 0.28`：token gate 的可用上限也被限制；
- `use_gate_regularization: true`：训练中约束 gate 不要滥用。

不同实验 YAML 会改变这些超参数或启用可选模块；解释具体 checkpoint 时，应以该 checkpoint 保存的 `resolved_event_train_config.json` 为准。

## 调试时看什么

BRG 会输出可记录的诊断量。优先检查：

| 指标 | 异常信号 |
|---|---|
| `event_scale` | 接近零：事件几乎没参与；长期接近最大值：事件预算可能过激 |
| `gate_mean` / `gate_max` | 均值很高：可能到处注入；几乎为零：门控未学到使用事件 |
| `low_event_gate_mean` | 明显非零：低事件区域泄漏，需要检查 mask 或正则 |
| `residual_ratio` | 太大：事件残差可能覆盖 RGB token；太小：事件贡献可能不可见 |
| `gate_logit_abs` | 过大：gate 接近饱和，学习信号变差 |

相关实现：

- 统计提取：`model/event_fusion.py:82-135`
- BRG 参数定义：`model/event_fusion.py:376-570`
- BRG 前向和融合公式：`model/event_fusion.py:618-744`
- Gate 正则与诊断：`model/event_fusion.py:746-835`
- 三阶段解冻策略：`training/event_teacher_student.py:21-152`
