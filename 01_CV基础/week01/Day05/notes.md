Day05 学习成果总结：Transformer Block 与 Transformer Encoder

学习主题：Transformer Block 与 Transformer Encoder
核心目标：理解 Self-Attention 为什么不是完整 Transformer、掌握 Residual / LayerNorm / FFN 的作用，能够完成 Transformer Block 的 Shape 推导、手写简化实现，并开始阅读 PyTorch TransformerEncoderLayer 源码。

1. 今天真正掌握的知识
1.1 Self-Attention ≠ 完整 Transformer Block

已经理解：

Self-Attention 主要负责 Token 与 Token 之间的信息交互，而完整 Transformer Block 还需要 FFN、Residual Connection 和 LayerNorm。

核心职责：

模块	核心作用
Multi-Head Self-Attention	Token ↔ Token 信息交互
FFN	单个 Token 内部的非线性特征变换
Residual Connection	保留原始信息 + 提供直接梯度路径
LayerNorm	稳定中间表示，帮助网络优化

已经能够从“模型能力”角度分析删除某个模块后的影响，而不是简单背诵模块名称。

1.2 Residual Connection

掌握基本形式：

$$ Y=X+F(X) $$

理解了 Residual 的三个重要作用：

保留输入中的原始信息；
学习相对于输入的“变化量/残差”；
提供直接的梯度传播路径。

梯度关系：

$$ \frac{\partial Y}{\partial X} = 1+\frac{\partial F(X)}{\partial X} $$

已经理解：

Residual 并不意味着“不改变输入”，而是允许网络在保留输入的基础上学习需要进行的修改。

同时知道不能简单说成：

“Residual 可以防止梯度消失。”

更准确的说法是：

Residual 提供直接的梯度路径，有助于深层网络的优化和梯度传播。

1.3 LayerNorm

已经掌握 Transformer 中 LayerNorm 的基本计算方式。

对于：

X.shape = [B, N, D]

LayerNorm 在 Transformer 中通常针对每个 Token 的 D 个特征计算：

$$ \mu=\frac{1}{D}\sum_i x_i $$ $$ \sigma^2=\frac{1}{D}\sum_i(x_i-\mu)^2 $$ $$ \hat{x}_i= \frac{x_i-\mu} {\sqrt{\sigma^2+\epsilon}} $$

然后：

$$ y_i=\gamma_i\hat{x}_i+\beta_i $$

已经理解：

LayerNorm 不改变 Tensor Shape；
nn.LayerNorm(D) 中的参数对应特征维度；
gamma 和 beta 的 Shape 是 [D]；
每个特征维度拥有独立的可学习缩放和偏移参数；
所有 Token 共享同一组 gamma/beta。

同时能够区分：

LayerNorm 是针对单个样本/Token 的特征维度进行统计，而不是简单理解成“对 Batch 维度进行归一化”。

1.4 FFN

掌握 Transformer 中典型 FFN：

$$ FFN(x)=W_2\sigma(W_1x+b_1)+b_2 $$

典型结构：

D
↓
D_ff
↓
Activation
↓
D

例如：

512 → 2048 → 512

已经理解升维的意义：

在更高维的特征空间中进行非线性变换，从而获得更加丰富的特征表达，再投影回原始维度。

同时明确：

Linear 本身是线性变换；
真正引入非线性的是 GELU/ReLU 等 Activation；
FFN 不进行 Token ↔ Token 信息交互；
所有 Token 使用同一套 FFN 参数；
FFN 最终恢复到 D，便于与 Residual Connection 相加。
1.5 Transformer Block 的完整结构

已经能够理解简化的 Post-LN Transformer Block：

$$ X_1=LN(X+MHA(X)) $$ $$ Y=LN(X_1+FFN(X_1)) $$

对应：

X
↓
MHA
↓
Residual
↓
LayerNorm
↓
FFN
↓
Residual
↓
LayerNorm
↓
Output

已经能够独立完成完整 Shape 推导。

例如：

X              [2, 4, 512]
MHA            [2, 4, 512]
Residual       [2, 4, 512]
LayerNorm      [2, 4, 512]
Linear1        [2, 4, 2048]
GELU           [2, 4, 2048]
Linear2        [2, 4, 512]
Residual2      [2, 4, 512]
LayerNorm      [2, 4, 512]
Output         [2, 4, 512]
1.6 MHA 的 Shape 推导进一步巩固

对于：

X = [B, N, 768]
H = 12
D_head = 64

已经能够正确推导：

[B, N, 768]
      ↓
[B, N, 12, 64]
      ↓ transpose
[B, 12, N, 64]
      ↓
QKᵀ
[B, 12, N, N]
      ↓
Attention × V
[B, 12, N, 64]
      ↓
Concat
[B, N, 768]

并进一步完成：

[B, N, 768]
      ↓
Linear1
[B, N, 3072]
      ↓
GELU
[B, N, 3072]
      ↓
Linear2
[B, N, 768]

已经纠正了之前对 Head Split 最终维度排列的小问题：

[B, N, H, D_head]
        ↓ transpose
[B, H, N, D_head]
1.7 Pre-LN 与 Post-LN

今天第一次接触并通过 PyTorch 源码确认了 Pre-LN / Post-LN。

Post-LN：

MHA → Residual → LN
FFN → Residual → LN

即：

$$ X_1=LN(X+MHA(X)) $$ $$ Y=LN(X_1+FFN(X_1)) $$

Pre-LN：

LN → MHA → Residual
LN → FFN → Residual

即：

$$ X_1=X+MHA(LN(X)) $$ $$ Y=X_1+FFN(LN(X_1)) $$

已经知道 PyTorch TransformerEncoderLayer 可以通过：

norm_first

控制使用 Pre-LN 还是 Post-LN。

默认：

norm_first=False

因此今天创建的 TransformerEncoderLayer 默认采用 Post-LN。

1.8 开始具备 Transformer 源码阅读能力

今天不仅看了模块结构，还实际阅读了：

nn.TransformerEncoderLayer.forward

以及：

nn.TransformerEncoderLayer._sa_block
nn.TransformerEncoderLayer._ff_block

已经能够将源码与理论对应起来。

例如 _sa_block：

self.self_attn(
    x,
    x,
    x,
    ...
)

理解为：

Q = x
K = x
V = x

因此是 Self-Attention。

同时理解：

return self.dropout1(x)

对应 Attention 输出后的 Dropout。

_ff_block：

x = self.linear2(
    self.dropout(
        self.activation(
            self.linear1(x)
        )
    )
)
return self.dropout2(x)

已经能够对应：

Linear1
↓
Activation
↓
Dropout
↓
Linear2
↓
Dropout
2. 我回答中暴露出的薄弱点
2.1 Residual 的理解曾经偏向“保留信息”

早期回答主要强调：

“因为 X 里面会有很多重要的信息。”

虽然正确，但不够深入。

经过今天的学习，现在已经补充到：

Residual
├── 保留原始信息
├── 学习残差变化
└── 提供直接梯度路径
后续要求

不要只从“信息保留”解释 Residual，要同时想到：

信息路径 + 学习目标 + 梯度路径。

2.2 LayerNorm 的表述需要更加严谨

曾经出现过：

“BatchNorm 是侧重 Batch 维度。”

这个表述过于粗糙。

更准确的是：

BatchNorm 利用 Mini-Batch 的统计量进行归一化；LayerNorm 针对单个样本/Token 的指定特征维度计算统计量。

以后不要死记：

BN = B
LN = D

而要根据 Tensor Shape 和 normalized dimensions 判断。

2.3 曾经把 Linear 的 Shape 行为部分理解成 Broadcasting

在 FFN 实验中曾认为：

[B,N,D] 输入 Linear 后保持 [B,N] 可能涉及 broadcasting。

后来已经纠正。

正确理解：

nn.Linear 对输入最后一个维度执行线性映射，前面的维度作为 batch-like dimensions 保持不变，并不是依靠 broadcasting 实现。

这是后续阅读 PyTorch 代码时需要继续注意的点。

2.4 Head Split 的维度排列曾出现小错误

曾经写过：

[B,N,H,D_head]

作为最终 Attention 输入 Shape。

后来已经理解：

[B,N,H,D_head]
        ↓ transpose
[B,H,N,D_head]

后者才是进行标准多头 Attention 计算时使用的排列。

目前已经能够正确回答。

2.5 Pre-LN / Post-LN 还需要进一步理解“为什么”

目前已经知道：

Post-LN:
MHA → Add → LN

Pre-LN:
LN → MHA → Add

但还没有深入理解：

为什么 Pre-LN 往往更有利于深层 Transformer 的训练稳定性？

这一部分暂时不影响进入 Day6，但后续深入 Transformer 时需要补充。

3. 今天完成的 Python 代码 / 实验
3.1 手写 LayerNorm

完成：

import torch

x = torch.randn(2, 4, 8)

mean = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True, correction=False)

eps = 1e-5
x_norm = (x - mean) / torch.sqrt(var + eps)

得到：

[2, 4, 8]

验证了 LayerNorm 的基本计算过程。

3.2 手动加入 gamma / beta

完成：

gamma = torch.ones(8)
beta = torch.zeros(8)

y = gamma * x_norm + beta

理解了：

gamma.shape = [8]
beta.shape  = [8]

以及其与 [2,4,8] Tensor 的广播关系。

3.3 手写 LayerNorm 与 PyTorch 对照

完成：

ln = torch.nn.LayerNorm(8)

y_torch = ln(x)

diff = (y_manual - y_torch).abs().max()

得到：

tensor(5.6505e-05, ...)

理解了：

手写实现与 PyTorch 基本一致，小量误差来自浮点计算及底层计算顺序，而不是公式错误。

3.4 FFN 实验

完成：

[2,4,512]
↓
Linear(512,2048)
↓
[2,4,2048]
↓
GELU
↓
[2,4,2048]
↓
Linear(2048,512)
↓
[2,4,512]

并观察了：

linear1.weight.shape = [2048,512]
linear2.weight.shape = [512,2048]

理解了 nn.Linear(in_features, out_features) 的 Weight Shape：

[out_features, in_features]
3.5 手写简化版 Transformer Block

完成了：

MHA Output
↓
Residual
↓
LayerNorm
↓
FFN
↓
Residual
↓
LayerNorm

并验证了最终：

Output.shape = [2,4,512]
3.6 Transformer Block Shape 独立推导

已经能够不看代码独立完成：

X              [2,4,512]
MHA            [2,4,512]
Residual       [2,4,512]
LayerNorm      [2,4,512]
Linear1        [2,4,2048]
GELU           [2,4,2048]
Linear2        [2,4,512]
Residual2      [2,4,512]
LayerNorm      [2,4,512]
Output         [2,4,512]
3.7 PyTorch TransformerEncoderLayer 源码阅读

完成：

layer = nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    activation="gelu",
    batch_first=True
)

print(layer)

并观察到：

self_attn
linear1
linear2
norm1
norm2
dropout
dropout1
dropout2

进一步阅读：

inspect.getsource(nn.TransformerEncoderLayer.forward)

以及：

inspect.getsource(nn.TransformerEncoderLayer._sa_block)
inspect.getsource(nn.TransformerEncoderLayer._ff_block)

这是今天最重要的一项源码实践。

4. 还存在的问题
需要继续巩固
Pre-LN vs Post-LN 的训练稳定性原理
目前理解“是什么”，还没有深入理解“为什么”。
LayerNorm 与 BatchNorm 的精确定义
已经基本掌握，但需要避免过度简化。
PyTorch Linear 的底层 Shape 机制
已经纠正 Broadcasting 误解，需要在后续代码中继续强化。
Dropout 的底层机制
已经知道其正则化作用，但还没有深入到训练/推理行为及数学细节。
完整 Transformer Encoder
今天主要理解了单个 TransformerEncoderLayer。
还需要进一步理解多个 Block 堆叠形成 Encoder 后，信息如何逐层传播。
5. 复试可能被问到的问题
基础问题
Q1. Self-Attention 和 Transformer Block 有什么区别？

核心回答：

Self-Attention 主要负责 Token 间的信息交互，而 Transformer Block 还包含 FFN、Residual Connection 和 LayerNorm，使模型能够进行 Token 内部的非线性特征变换，并改善信息和梯度传播以及训练稳定性。

Q2. 为什么 Transformer 中需要 FFN？

Self-Attention 主要负责 Token 间的信息交互，FFN 则对每个 Token 独立进行非线性特征变换，使模型能够进一步加工和提升特征表示能力。

Q3. 为什么 FFN 要先升维再降维？

通过升维将特征映射到更大的中间空间，在更丰富的特征空间中进行非线性变换，然后再投影回模型原始维度。

Q4. Residual Connection 有什么作用？

一方面保留输入信息，另一方面提供直接的梯度传播路径，同时使网络学习相对于输入的残差变化，有助于深层网络优化。

Q5. LayerNorm 为什么适合 Transformer？

Transformer 中不同 Token 可以独立进行 LayerNorm，不依赖 Batch 的统计量，因此对于序列长度和 Batch Size 的变化更加灵活，同时能够稳定中间特征分布并帮助优化。

Q6. LayerNorm 是在哪个维度进行的？

对于：

[B,N,D]

通常对每个 Token 的 D 个特征进行归一化。

Q7. Pre-LN 和 Post-LN 有什么区别？

Post-LN 在 Residual Add 后进行 LayerNorm；Pre-LN 在 Attention/FFN 子层之前进行 LayerNorm。

Q8. 为什么 Transformer Block 中有两个 LayerNorm？

因为一个 Transformer Block 包含 Self-Attention 和 FFN 两个主要子层，每个子层都有对应的归一化操作。

Q9. 如果去掉 FFN，会发生什么？

Token 之间仍然可以通过 Self-Attention 进行信息交互，但模型失去了对每个 Token 进行进一步非线性特征变换的能力，整体表达能力会受到限制。

Q10. 为什么 MHA 的输出可以和输入做 Residual？

因为：

输入：[B,N,D]
MHA输出：[B,N,D]

二者 Shape 相同，因此可以进行逐元素相加。

6. Day6 学习前必须复习的内容

不需要重新学习整个 Day5，只需要复习以下内容。

必背但不是死记
① Transformer Block 四个核心组件
MHA
→ Token 间信息交互

FFN
→ Token 内部非线性特征变换

Residual
→ 信息保留 + 直接梯度路径

LayerNorm
→ 稳定中间表示 + 帮助优化
② Residual
$$ Y=X+F(X) $$ $$ \frac{\partial Y}{\partial X} = 1+\frac{\partial F(X)}{\partial X} $$
③ LayerNorm

对于：

[B,N,D]

记住：

Transformer 中通常对每个 Token 的 D 个特征计算均值和方差。

④ FFN
$$ FFN(x)=W_2\sigma(W_1x+b_1)+b_2 $$

Shape：

[B,N,D]
→
[B,N,D_ff]
→
[B,N,D]
⑤ MHA Shape
[B,N,D]
→ [B,H,N,D_head]
→ [B,H,N,N]
→ [B,H,N,D_head]
→ [B,N,D]

重点记住：

H × D_head = D
⑥ Pre-LN / Post-LN
Post-LN:
MHA → Residual → LN
FFN → Residual → LN

Pre-LN:
LN → MHA → Residual
LN → FFN → Residual
7. Day5 是否达到进入 Day6 的标准？
结论：达到，而且可以正常进入 Day6。
理由

你已经满足进入 Day6 的几个关键条件：

① 能解释 Transformer Block，而不是只会背结构

你能够回答：

为什么需要 FFN？

为什么需要 Residual？

删除 FFN 后模型还能做什么？

LayerNorm 为什么在 D 维度？

这些问题说明你已经理解组件的功能和相互关系。

② 能独立完成 Tensor Shape 推导

你已经能够独立完成：

[B,N,D]
→
[B,H,N,D_head]
→
[B,H,N,N]
→
[B,N,D]
→
[B,N,D_ff]
→
[B,N,D]

这是继续学习 ViT、DETR、CLIP 等模型的重要基础。

③ 已经完成从理论到代码的闭环

今天不是单纯听理论，而是完成了：

理论
 ↓
数学公式
 ↓
Tensor Shape
 ↓
手写 LayerNorm
 ↓
PyTorch 对照
 ↓
手写 FFN
 ↓
手写 Transformer Block
 ↓
阅读 PyTorch 源码

这已经符合你当前学习路线要求的：

理解 → 实现 → 验证 → 阅读源码

④ 还有薄弱点，但不足以阻碍 Day6

目前剩余的问题主要是：

Pre-LN 深层优化原理；
Dropout 细节；
LayerNorm 更严格的定义；
Transformer Encoder 多层堆叠后的进一步理解。

这些属于深化问题，而不是 Day6 的前置阻塞问题。

因此没有必要为了“100%掌握 Day5”而停下来。

Day5 最终能力定位
Day5 前：

知道 Attention
知道 Self-Attention
知道 MHA

        ↓

Day5 学习

Residual
LayerNorm
FFN
Transformer Block
Tensor Shape
PyTorch 实现
PyTorch 源码

        ↓

Day5 后：

能够解释 Transformer Block
能够推导 Block Shape
能够手写简化 Block
能够阅读 TransformerEncoderLayer
能够区分 Pre-LN / Post-LN
能够分析删除模块后的模型能力

因此：

Day5：通过，可以进入 Day6。

Day6 预告

下一阶段进入：

Day6｜Positional Encoding：Transformer 为什么需要位置信息？

核心问题：

Self-Attention 能够看到所有 Token，但它为什么天然不知道 Token 的顺序？

然后逐步进入：

Self-Attention
      ↓
缺少位置信息
      ↓
Positional Encoding
      ↓
Sinusoidal Positional Encoding
      ↓
Learnable Position Embedding
      ↓
ViT 中的位置编码
      ↓
视觉 Transformer 为什么同样需要位置？

这一步会正式把你目前的 Transformer 基础连接到后面的 ViT / DETR / Vision-Language / Open-Vocabulary Detection。