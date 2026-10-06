# Day07｜Transformer 综合理解、Tensor Shape、PyTorch 实现与消融实验

> 学习阶段：CV → Transformer 基础  
> Day：07  
> 核心主题：Transformer Block 综合理解 + Tensor Shape 追踪 + PyTorch 手写实现 + FFN 消融实验  
> 学习目标：能够从 `[B, N, D]` 出发，完整解释一个 Transformer Block 的数据流、各模块作用以及实验现象。

---

# 1. 今天真正掌握的知识

## 1.1 Attention、Self-Attention 与 Multi-Head Attention

已经能够区分：

### Attention

Attention 中的：

- Query（Q）
- Key（K）
- Value（V）

不一定来自同一个输入。

因此 Attention 可以用于不同信息源之间的信息交互。

### Self-Attention

Self-Attention 是 Attention 的特殊情况：

```text
Q、K、V 都来自同一个输入 X

但需要注意：

Q = XW_Q
K = XW_K
V = XW_V

虽然三者来源都是 X，但经过的是不同的线性投影。
```

## 1.2 Multi-Head Attention

理解了 Multi-Head Attention 的核心思想：

```text
输入 X
[B, N, D]
      ↓
Q/K/V
[B, N, D]
      ↓
拆分多个 Head
[B, H, N, d_head]
```

其中：

```text
d_head = D / H
```

不同 Head 可以在不同的特征子空间中学习不同的 Token 关系。

最终：

```text
多个 Head
[B, H, N, d_head]
      ↓
Concat
[B, N, D]
      ↓
W^O
[B, N, D]
```

已经理解：

`W^O` 不只是为了“把维度变回 D”，因为多个 Head Concat 后本身已经是 D 维。更重要的是通过 `W^O` 对不同 Head 的信息进行进一步混合。

---

# 2. Transformer 中最重要的 Tensor Shape

已经能够独立追踪 Transformer Block 中的主要 Tensor Shape。

设：

```text
B = Batch Size
N = Token 数量
D = Embedding Dimension
H = Head 数量
d_head = D / H
D_ff = FFN 隐藏层维度
```

完整流程：

```text
X
[B, N, D]

Q/K/V
[B, N, D]

Split Heads
[B, H, N, d_head]

QK^T
[B, H, N, N]

Softmax
[B, H, N, N]

Attention @ V
[B, H, N, d_head]

Transpose + Concat
[B, N, D]

W^O
[B, N, D]

Residual + LayerNorm
[B, N, D]

FFN Linear1
[B, N, D_ff]

GELU
[B, N, D_ff]

FFN Linear2
[B, N, D]

Residual + LayerNorm
[B, N, D]
```

---

# 3. QK^T 的理解

已经理解：

```text
QK^T
```

得到：

```text
[B, H, N, N]
```

其中两个 N 的含义不同：

```text
第一个 N：Query Token
第二个 N：Key Token
```

因此：

```text
scores[h, i, j]
```

表示：

```text
第 i 个 Query Token 对第 j 个 Key Token 的匹配程度。
```

这意味着 Self-Attention 可以让一个 Token 与序列中的其他 Token 建立信息联系。

---

# 4. Scaling 与 Softmax

已经理解 Scaled Dot-Product Attention：

$$
Attention(Q,K,V) = Softmax \left( \frac{QK^T}{\sqrt{d_{head}}} \right)V
$$

注意：

除以 `sqrt(d_head)`

而不是：

`sqrt(D)`

原因是 Q/K 每个 Head 的维度是：

```text
d_head = D / H
```

所以点积的尺度与 `d_head` 有关。

Softmax 的维度

使用：

```text
softmax(scores, dim=-1)
```

因为最后一个维度对应：

```text
Key Token
```

对于每一个 Query Token：

```text
attention weights
```

需要在所有 Key Token 上归一化，使其权重和为：

```text
1
```

实验中验证了 Attention 权重某一行的和为：

```text
1.
```

---

# 5. Transformer Block 中 MHA 与 FFN 的功能分工

今天已经形成了比较清晰的认识：

```text
MHA
↓
Token ↔ Token
信息交互
```

而：

```text
FFN
↓
单个 Token 内部
特征维度上的非线性变换
```

可以总结为：

```text
MHA 负责跨 Token 的信息混合，FFN 负责 Token 内部的特征变换。
```

两者作用互补：

```text
MHA
建立 Token 之间的信息联系
        ↓
FFN
对聚合后的 Token 表示进行进一步非线性变换
```

因此 Transformer Block 同时具备：

```text
Token 间的信息交互
+
Token 内部的非线性特征变换
```

---

# 6. FFN 的结构

Transformer 中 FFN 的典型结构：

```text
[B, N, D]
    ↓
Linear(D → D_ff)
    ↓
[B, N, D_ff]
    ↓
GELU
    ↓
[B, N, D_ff]
    ↓
Linear(D_ff → D)
    ↓
[B, N, D]
```

已经理解：

- FFN 是逐 Token 独立计算的；
- FFN 不直接负责 Token 与 Token 之间的信息交互；
- FFN 主要改变每个 Token 内部的特征表示；
- 最终必须回到 D 维，因为需要与 Residual 分支进行逐元素相加。

---

# 7. Residual Connection

已经理解 Residual Connection 的核心作用：

```text
X + F(X)
```

可以理解为：

```text
网络不是完全重新学习一个新的表示，而是在原始表示基础上学习一个修改量 / 增量。
```

Residual 还提供了直接的信息和梯度路径，有利于深层网络训练。

Transformer Block 中存在两处 Residual：

```text
X + MHA(X)

X1 + FFN(X1)
```

---

# 8. LayerNorm

今天已经掌握 LayerNorm 在 Transformer Block 中的基本位置和作用。

对于今天实现的 Post-LN：

$$
X_1 = LN(X + MHA(X))
$$

$$
X_2 = LN(X_1 + FFN(X_1))
$$

需要注意：

不能简单地说 LayerNorm “一定防止梯度爆炸/消失”。

更准确的理解是：

```text
LayerNorm 对中间表示进行归一化，有助于稳定网络训练。
```

---

# 9. Post-LN 与 Pre-LN

已经能够区分两种结构。

## Post-LN

```text
X
 ↓
MHA
 ↓
Residual
 ↓
LayerNorm
```

数学形式：

$$
X_1 = LN(X + MHA(X))
$$

FFN：

$$
X_2 = LN(X_1 + FFN(X_1))
$$

## Pre-LN

```text
X
 ↓
LayerNorm
 ↓
MHA
 ↓
Residual
```

数学形式：

$$
X_1 = X + MHA(LN(X))
$$

$$
X_2 = X_1 + FFN(LN(X_1))
$$

今天已经能够识别两者结构差异，但：

```text
Pre-LN 为什么在现代 Transformer 中经常被采用，以及它与训练稳定性的更深入关系，还没有进行系统学习。
```

该问题留待后续 Transformer 深入阶段。

---

# 10. Positional Encoding

今天继续巩固了位置编码。

已经理解：

标准 Self-Attention 本身并不天然知道 Token 的顺序。

因此需要额外加入位置编码：

$$
X' = X + P
$$

其中：

```text
X: [B, N, D]
P: [N, D]
X': [B, N, D]
```

位置编码不是简单地把：

```text
position = 0, 1, 2, ...
```

直接作为一个标量输入。

而是：

```text
将每个位置编码成一个 D 维向量。
```

然后与 Token Embedding 相加。

由于 P 没有 Batch 维度，在实际计算中可以通过 broadcasting 与 `[B,N,D]` 的 X 对齐。

---

# 11. 完整 Transformer Block 数据流

今天已经能够独立口述完整数据流：

```text
输入 X
[B, N, D]
      ↓
Q/K/V
[B, N, D]
      ↓
Split Heads
[B, H, N, d_head]
      ↓
QK^T
[B, H, N, N]
      ↓
Scaling / sqrt(d_head)
[B, H, N, N]
      ↓
Softmax(dim=-1)
[B, H, N, N]
      ↓
Attention @ V
[B, H, N, d_head]
      ↓
Transpose + Concat
[B, N, D]
      ↓
W^O
[B, N, D]
      ↓
Residual
[B, N, D]
      ↓
LayerNorm
[B, N, D]
      ↓
Linear1
[B, N, D_ff]
      ↓
GELU
[B, N, D_ff]
      ↓
Linear2
[B, N, D]
      ↓
Residual
[B, N, D]
      ↓
LayerNorm
[B, N, D]
      ↓
Output
[B, N, D]
```

目前能够正确解释该流程中的主要 Tensor Shape 和模块作用。

---

# 12. 今天完成的 Python 代码 / 实验

## 12.1 手写 Multi-Head Attention

使用 PyTorch 手动实现了：

```text
Q/K/V
↓
Split Heads
↓
QK^T
↓
Scaling
↓
Softmax
↓
Attention @ V
↓
Concat
↓
W^O
```

核心参数：

```text
B, N, D = 2, 5, 8
H = 2
d_head = D // H
```

得到：

```text
Q:              [2, 2, 5, 4]
K:              [2, 2, 5, 4]
V:              [2, 2, 5, 4]

scores:         [2, 2, 5, 5]
attn_weights:   [2, 2, 5, 5]

output_head:    [2, 2, 5, 4]
concat:         [2, 5, 8]
final:          [2, 5, 8]
```

并验证：

```text
Attention 权重某一行的和 = 1
```

说明 Softmax 归一化维度理解正确。

---

# 13. Mini Transformer Block 实现

使用 PyTorch 实现了一个 Post-LN Transformer Block：

```text
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
```

参数：

```text
B = 2
N = 5
D = 8
D_ff = 16
```

验证了：

```text
X + MHA(X)
[B,5,8]

Linear1
[B,5,16]

GELU
[B,5,16]

Linear2
[B,5,8]

X1 + FFN(X1)
[B,5,8]
```

---

# 14. FFN 消融实验

今天完成了一个简单的 FFN Ablation Experiment。

## 实验任务

Copy Task：

```text
Input = Target
```

例如：

```text
Input:
[2, 7, 6, 4, 6]

Target:
[2, 7, 6, 4, 6]
```

模型：

```text
Embedding
↓
Transformer Block
↓
Linear Classifier
```

比较：

```text
Baseline：包含 FFN
Ablation：去掉 FFN
```

## 实验设置

```text
B = 32
N = 5
D = 16
H = 4
D_ff = 32
VOCAB_SIZE = 10
Optimizer = Adam
Learning Rate = 1e-3
Epochs = 300
```

---

# 15. FFN 消融实验结果

## Baseline：包含 FFN

```text
Epoch 50:  1.3039
Epoch 100: 0.5034
Epoch 150: 0.2219
Epoch 200: 0.1262
Epoch 250: 0.0814
Epoch 300: 0.0567

Final Accuracy = 1.0
```

## Ablation：去掉 FFN

```text
Epoch 50:  1.6275
Epoch 100: 0.8348
Epoch 150: 0.3665
Epoch 200: 0.1727
Epoch 250: 0.0994
Epoch 300: 0.0654

Final Accuracy = 1.0
```

---

# 16. 实验结论

目前能够支持的结论：

```text
在本实验设置下，去掉 FFN 后模型仍然可以完成 Copy Task，并最终达到 100% 的训练集 Accuracy；但是相比包含 FFN 的模型，训练过程中 Loss 下降更慢，并且在相同训练轮数下最终 Loss 略高。
```

因此可以推测：

```text
FFN 可能有助于模型在该任务上的优化和特征变换。
```

但是目前不能得出：

```text
FFN 是模型完成任务的必要条件
```

或者：

```text
FFN 一定能够提高模型泛化能力
```

因为实验设计还存在明显限制。

---

# 17. 今天暴露出的实验思维问题

今天最大的进步不是某个公式，而是开始意识到：

```text
实验结果 ≠ 实验结论。
```

目前实验存在：

- 数据集规模非常小；
- Train/Test 没有分离；
- 使用的是非常简单的 Copy Task；
- 只有一次实验；
- 没有进行多随机种子重复实验；
- 当前只观察了训练 Loss 和训练 Accuracy；
- 无法判断模型的泛化能力。

因此实验结果的说服力有限。

---

# 18. 今天回答中暴露出的薄弱点

## 18.1 Pre-LN / Post-LN 的深入理解还不够

已经能够正确写出：

Post-LN:

```text
X1 = LN(X + MHA(X))

X2 = LN(X1 + FFN(X1))
```

以及：

Pre-LN:

```text
X1 = X + MHA(LN(X))

X2 = X1 + FFN(LN(X1))
```

但是目前主要停留在：

```text
“结构上知道区别”。
```

还没有深入理解：

- 为什么 Pre-LN 更容易训练；
- 梯度路径有什么区别；
- 为什么深层 Transformer 中经常采用 Pre-LN；
- Pre-LN 和 Post-LN 的训练稳定性差异。

## 18.2 对 FFN “为什么能够增强特征表达”的数学理解还可以继续加强

目前能够正确说：

```text
FFN 对 Token 内部进行非线性特征变换。
```

但还需要进一步理解：

$$
x \rightarrow W_1x+b_1 \rightarrow GELU \rightarrow W_2x+b_2
$$

为什么：

```text
D → D_ff → D
```

能够提升表示能力。

特别需要避免简单理解为：

```text
“因为 D_ff 更大，所以信息更多。”
```

真正关键的是：

```text
高维中间表示 + 非线性激活 + 两层线性映射共同构成了更强的逐 Token 非线性变换。
```

## 18.3 Attention 与 FFN 的“作用对象”已经掌握，但还需要进一步理解二者如何协同

目前能够回答：

```text
MHA → Token 间
FFN → Token 内
```

下一阶段需要进一步理解：

```text
MHA 从其他 Token 聚合什么信息？
        ↓
FFN 如何处理聚合后的信息？
        ↓
为什么不断重复 MHA + FFN 可以逐层构建更复杂表示？
```

---

# 19. 还存在的问题

当前主要问题：

## 问题 1：FFN 消融实验还不够有说服力

下一步如果继续研究 FFN，可以：

```text
扩大训练集
↓
划分 Train / Test
↓
增加任务难度
↓
比较 Test Accuracy
↓
使用多个随机种子
↓
统计平均值和标准差
```

## 问题 2：还没有深入 Transformer Encoder 的堆叠机制

目前已经理解：

```text
一个 Transformer Block
```

但需要进一步理解：

```text
Block
↓
Block
↓
Block
↓
...
↓
Encoder
```

以及：

```text
为什么多个 Block 堆叠后能够逐层提取更加复杂的表示。
```

## 问题 3：还没有进入 Vision Transformer

目前 Transformer 的输入主要还是抽象的：

```text
Token Sequence
```

还需要进一步解决：

```text
图像本身不是 Token，为什么可以输入 Transformer？
```

这将自然连接到：

```text
Image
↓
Patch
↓
Patch Embedding
↓
Visual Tokens
↓
Positional Embedding
↓
Transformer Encoder
↓
Vision Transformer
```

这也是 Day8 的重要衔接点。

---

# 20. 复试可能被问到的问题

## 基础问题

### Q1：Self-Attention 和普通 Attention 有什么区别？

回答核心：

```text
Self-Attention 中 Q、K、V 都来自同一个输入，而普通 Attention 中 Q、K、V 可以来自不同信息源。
```

### Q2：为什么 Multi-Head Attention 要使用多个 Head？

核心：

```text
将特征划分到不同子空间，使不同 Head 可以学习不同类型的 Token 关系和特征交互。
```

### Q3：为什么 Attention 要除以 sqrt(d_head)？

核心：

```text
随着 d_head 增大，QKᵀ 的数值尺度可能变大，使 Softmax 进入过于尖锐的区域，从而影响梯度和训练稳定性。因此使用 sqrt(d_head) 进行缩放。
```

### Q4：为什么 Softmax 使用 dim=-1？

核心：

```text
最后一维对应 Key Token，需要对每一个 Query Token 在所有 Key Token 上进行归一化，使其注意力权重之和为 1。
```

### Q5：为什么 FFN 的输入输出维度都必须是 D？

核心：

```text
因为 Transformer Block 使用 Residual Connection，需要输入和变换后的结果具有相同形状，才能进行逐元素相加。
```

### Q6：MHA 和 FFN 分别解决什么问题？

核心：

```text
MHA 负责 Token 间的信息交互；FFN 负责 Token 内部的非线性特征变换。
```

### Q7：如果去掉 FFN 会发生什么？

不能简单回答：

```text
“模型无法工作。”
```

更准确：

```text
去掉 FFN 后仍然可以通过 Attention 进行 Token 间的信息交互，但失去了 FFN 提供的逐 Token 非线性特征变换能力。具体性能影响需要通过实验验证。
```

### Q8：为什么 Transformer 需要 Positional Encoding？

核心：

```text
Self-Attention 本身对 Token 的排列顺序没有显式的位置感知能力，因此需要额外加入位置编码，使模型能够利用 Token 的位置信息。
```

### Q9：为什么 Residual Connection 有用？

核心：

```text
它保留原始表示，同时让子模块学习对原表示的修改量，并提供直接的信息和梯度路径，有利于深层网络训练。
```

### Q10：为什么 LayerNorm 放在这里？

需要能够根据：

```text
Pre-LN
Post-LN
```

分别解释，而不是只背一个固定结构。

---

# 21. 明天 Day8 学习前必须复习的内容

进入 Day8 前，建议重点复习以下内容。

## 必须熟练

### ① MHA 完整公式

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
Attention(Q,K,V) = Softmax \left( \frac{QK^T}{\sqrt{d_{head}}} \right)V
$$

### ② Tensor Shape

必须能够快速回答：

```text
X                  [B,N,D]

Q/K/V              [B,N,D]

Split Heads        [B,H,N,d_head]

QK^T               [B,H,N,N]

Attention Weight   [B,H,N,N]

Attention @ V      [B,H,N,d_head]

Concat             [B,N,D]

Wo                 [B,N,D]

FFN Linear1        [B,N,D_ff]

FFN Linear2        [B,N,D]
```

### ③ MHA 和 FFN 的功能分工

必须能够脱离公式回答：

```text
MHA：
Token ↔ Token
跨 Token 信息交互

FFN：
Token 内部
特征维度非线性变换
```

### ④ Transformer Block

能够完整画出：

```text
X
↓
MHA
↓
Residual + LN
↓
FFN
↓
Residual + LN
↓
Output
```

并能够解释每一步：

```text
是什么？
为什么需要？
解决什么问题？
如果去掉会怎样？
```

### ⑤ Positional Encoding

重点复习：

- 为什么需要 PE？
- PE 的 Shape 是什么？
- 为什么可以和 X 相加？
- 为什么 PE 不是一个简单的 position scalar？

### ⑥ Ablation Experiment

复习今天的实验：

```text
Independent Variable：
是否使用 FFN

Baseline：
包含 FFN

Ablation：
不包含 FFN

Metrics：
Loss
Training Accuracy
```

当前结论：

```text
去掉 FFN 后仍然可以拟合 Copy Task，
但收敛速度较慢。
```

实验局限：

```text
数据少
无 Test Set
任务简单
无法判断泛化能力
```

---

# 22. Day7 是否达到进入 Day8 的标准？

结论：✅ 达到

我认为你已经达到进入 Day8 的标准。

原因不是因为你“背完了 Transformer 公式”，而是因为今天已经完成了三个层面的能力验证。

## 第一层：理论理解

你已经能够解释：

- Attention
- Self-Attention
- MHA
- FFN
- Residual
- LayerNorm
- Positional Encoding
- Transformer Block

并且能够说明它们之间的关系。

## 第二层：代码与 Tensor Shape

你已经亲自完成：

```text
手写 MHA
↓
验证 Tensor Shape
↓
验证 Attention 权重
↓
实现 Transformer Block
```

并且能够独立解释：

```text
[B,N,D]
→
[B,H,N,d_head]
→
[B,H,N,N]
→
[B,H,N,d_head]
→
[B,N,D]
```

这说明你已经不是只停留在概念层面。

## 第三层：实验与科研思维

今天尤其重要的一点是完成了：

```text
Baseline
vs
Ablation
```

并且你没有因为：

```text
Accuracy = 100%
```

就直接得出：

```text
“FFN 没有作用。”
```

相反，你主动指出：

- 训练集太小
- 没有 Train/Test 划分
- 无法判断泛化能力
- 实验说服力不足

这说明你开始具备：

```text
控制变量 → 观察指标 → 分析结果 → 识别实验局限 → 避免过度归因
```

的基本科研思维。

---

# 23. Day7 最终评价

```text
理论理解       █████████░  90%
Tensor Shape   ██████████ 100%
代码实现       █████████░  90%
实验能力       ████████░░  80%
科研分析       ████████░░  80%
Pre/Post-LN    ██████░░░░  60%
```

其中 Pre-LN / Post-LN 的深入部分暂时没有必要阻塞后续学习。

---

# 24. Day8 的衔接方向

Day7 的最终结果可以概括为：

```text
Transformer
    ↓
已经理解 Encoder Block 内部怎么工作
    ↓
下一个问题：
Transformer 如何处理图像？
    ↓
Image
    ↓
Patch
    ↓
Patch Embedding
    ↓
Visual Tokens
    ↓
Positional Embedding
    ↓
Transformer Encoder
    ↓
Vision Transformer (ViT)
```

因此 Day8 可以自然进入：

```text
Vision Transformer：为什么图像可以被转换成 Token，以及 Transformer 如何开始处理视觉信息。
```

这一步是后续：

```text
ViT
 ↓
Vision Transformer
 ↓
Vision-Language Model
 ↓
Open-Vocabulary Detection
 ↓
Multimodal Learning
```

的重要基础。

---

# Day07 总结

今天真正完成的不是“又学了几个 Transformer 公式”，而是完成了从：

```text
理解 Transformer
        ↓
理解 Tensor Shape
        ↓
自己写 MHA
        ↓
自己写 Transformer Block
        ↓
设计 FFN Ablation
        ↓
运行实验
        ↓
分析 Loss / Accuracy
        ↓
识别实验局限
        ↓
避免错误归因
```

的完整闭环。

Day7 达到进入 Day8 的标准。

下一阶段重点从：

```text
“Transformer 如何处理 Token？”
```

转向：

```text
“图像如何变成 Token，并进入 Transformer？”
```

这将正式连接 Transformer 与 Computer Vision。