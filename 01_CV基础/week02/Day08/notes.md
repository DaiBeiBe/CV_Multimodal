# Day08｜Vision Transformer（ViT）：从图像 Patch 到视觉 Token

## 一、今日学习主题

今天完成了从 Transformer 到 Vision Transformer（ViT）的关键过渡：

```text
图像
↓
Patch Partition
↓
Patch Embedding
↓
Visual Token Sequence
↓
Position Embedding
↓
CLS Token
↓
Transformer Encoder
↓
CLS Representation
↓
Classification Head
```

核心目标：

> 理解 Transformer 为什么可以用于图像，以及 ViT 如何将 `[B,C,H,W]` 的图像转换成 Transformer 可以处理的 `[B,N,D]` 视觉 Token 序列。

---

# 二、今天真正掌握的知识

## 2.1 图像为什么需要转换成 Token Sequence

已经理解：

Transformer 的标准序列输入可以抽象为：

```text
[B, N, D]
```

其中：

* `B`：Batch Size
* `N`：Sequence Length / Token 数量
* `D`：Embedding Dimension

原始图像：

```text
[B, 3, H, W]
```

不能直接按照标准 Transformer 的序列形式处理，因此 ViT 首先将图像划分为 Patch，再将 Patch 转换成视觉 Token。

核心转换：

```text
[B, C, H, W]
        ↓
[B, N, P×P×C]
        ↓
[B, N, D]
```

已经理解 ViT 的本质之一：

> **ViT 的核心不是重新设计 Transformer，而是先把图像转换成 Transformer 能够处理的 Token 序列。**

---

## 2.2 Patch Partition

已经掌握 Patch 数量计算：

$$
N=\frac{H}{P}\times\frac{W}{P}
$$

例如：

```text
224×224 图像
P=16

224/16 = 14

N = 14×14 = 196
```

对于 `32×32` 图像：

```text
P=8  → 4×4   → 16 patches
P=4  → 8×8   → 64 patches
P=2  → 16×16 → 256 patches
```

---

## 2.3 Patch Embedding

已经能够区分：

```text
Patch Partition
Flatten
Linear Projection
```

理解了 Patch Embedding 并不是简单的：

> “切块 + 拉平”。

完整过程：

```text
Patch
↓
Flatten
↓
Linear Projection
↓
Visual Token
```

例如：

```text
16×16×3 = 768
```

经过：

```text
Linear(768, 512)
```

得到：

```text
512-dimensional visual token
```

已经理解：

* Flatten：改变表示形式，将空间 Patch 转换成向量；
* Linear Projection：将原始 Patch 向量映射到统一的 embedding 空间；
* Embedding Dimension `D` 是模型设计选择，并不是 Transformer 固定要求。

### 注意

今天发现一个需要继续保持严谨的表述：

> Linear Projection 不应该直接等同于“高层语义提取”。

更准确地说：

> Linear Projection 首先完成从原始像素空间到模型 embedding 空间的映射；更高层的语义表示主要由后续 Transformer Encoder 多层特征交互逐渐形成。

---

## 2.4 Visual Token Sequence

已经能够独立推导：

```text
输入：
[B,3,224,224]

P=16
N=196
D=512
```

得到：

```text
[B,196,16,16,3]
        ↓
[B,196,768]
        ↓
[B,196,512]
```

已经能够通过 Tensor Shape 理解：

> 每一个 Patch 最终对应一个 D 维 Visual Token。

---

## 2.5 Position Embedding

已经理解为什么 ViT 必须加入位置信息。

Self-Attention 本身没有 CNN 那样显式的局部空间结构，也没有天然的二维空间位置信息。

例如：

```text
Patch A = 猫头
Patch B = 猫身体
```

如果不知道它们在图像中的位置，模型无法仅通过 Token 内容完整判断：

```text
猫头在猫身体上方
```

因此需要：

```text
Visual Token
+
Position Embedding
```

常见形状：

```text
Visual Token:
[B,N,D]

Position Embedding:
[1,N,D]

相加：
[B,N,D]
```

已经理解 `[1,N,D]` 可以通过 broadcasting 加到 `[B,N,D]`。

同时理解：

> Position Embedding 不一定是显式的 `(x,y)` 坐标，而可以是针对序列位置学习得到的向量。

---

## 2.6 CLS Token

已经理解 CLS Token 的作用。

初始：

```text
[Patch1, Patch2, ..., PatchN]
```

加入：

```text
[CLS, Patch1, Patch2, ..., PatchN]
```

因此：

```text
[B,N,D]
↓
[B,N+1,D]
```

CLS 参数：

```python
self.cls_token = nn.Parameter(
    torch.zeros(1, 1, embed_dim)
)
```

使用：

```python
cls_tokens = self.cls_token.expand(B, -1, -1)
```

然后：

```python
x = torch.cat((cls_tokens, x), dim=1)
```

得到：

```text
[B,N+1,D]
```

已经理解：

> CLS 并不是天然就包含整张图像的信息，而是在多层 Self-Attention 中不断与所有 Patch Token 交互，逐渐形成全局图像表示。

最终：

```python
cls_out = x[:, 0]
```

得到：

```text
[B,D]
```

再输入分类头。

---

## 2.7 Broadcasting 与 Expand + Cat 的区别

今天重点纠正并掌握了一个容易混淆的地方。

### Position Embedding

```text
[B,N,D]
+
[1,N,D]

↓ Broadcasting

[B,N,D]
```

Token 数量不变。

### CLS Token

```text
[1,1,D]
↓ expand
[B,1,D]
↓ cat(dim=1)
[B,N+1,D]
```

Token 数量增加 1。

因此：

> Broadcasting 主要用于逐元素运算中的维度对齐，而 CLS Token 是通过 `expand + cat` 增加一个新的 Token。

---

## 2.8 Transformer Encoder 在 ViT 中的作用

已经能够将之前 Day5～Day7 学习的 Transformer Block 与 ViT 联系起来。

MiniViT 中使用：

```text
LayerNorm
↓
Multi-Head Self-Attention
↓
Residual
↓
LayerNorm
↓
MLP
↓
Residual
```

并堆叠多个 Transformer Encoder Block。

今天实现的 MiniViT：

```text
Patch Embedding
↓
CLS + Position Embedding
↓
Transformer Encoder × depth
↓
LayerNorm
↓
CLS
↓
Classification Head
```

---

## 2.9 CNN 与 ViT 的区别

已经理解：

### CNN

具有显式的局部连接和局部性归纳偏置：

```text
局部区域
↓
卷积
↓
逐渐扩大感受野
↓
全局信息
```

### ViT

通过 Self-Attention 让不同 Patch 之间直接建立关系：

```text
Patch A ↔ Patch B
Patch A ↔ Patch C
Patch B ↔ Patch C
...
```

因此远距离 Patch 可以直接进行信息交互。

同时认识到：

> ViT 并不是“近处和远处的关系权重天然相同”，而是缺少 CNN 那种显式的局部距离归纳偏置。

---

# 三、Patch Size 与计算复杂度

这是今天非常重要的实验与理论结论。

## 3.1 Patch Size 越小

```text
Patch Size ↓
↓
Patch 数量 N ↑
↓
空间细节 ↑
↓
Token 数量 ↑
↓
Attention 计算和显存 ↑↑
```

Patch 数量：

$$
N=\left(\frac{H}{P}\right)^2
$$

Self-Attention 主要复杂度：

$$
O(N^2D)
$$

---

## 3.2 32×32 图像实验

实际验证：

```text
P=8:
N=16
Token=17
Attention=17×17

P=4:
N=64
Token=65
Attention=65×65

P=2:
N=256
Token=257
Attention=257×257
```

Attention 元素数量：

```text
P=8 → 289
P=4 → 4225
P=2 → 66049
```

已经通过代码实验验证了理论。

---

## 3.3 为什么 Token 增长 16 倍，Attention 却增长约 256 倍？

已经掌握：

```text
P=8 → P=2

Patch 数：
16 → 256
= 16倍

Attention：
16² → 256²
≈ 256倍
```

原因：

$$
QK^T
$$

产生：

$$
[N,N]
$$

大小的 Attention Matrix。

因此 Attention 具有：

$$
N^2
$$

级别的增长。

---

# 四、今天完成的 Python 代码

今天独立完成并运行了一个完整的 MiniViT。

## 4.1 PatchEmbedding

实现：

```python
class PatchEmbedding(nn.Module):
    ...
```

完成：

```text
[B,C,H,W]
↓
[B,N,P,P,C]
↓
[B,N,P²C]
↓
[B,N,D]
```

掌握了：

* `reshape`
* `permute`
* `flatten`
* `nn.Linear`

---

## 4.2 MultiHeadSelfAttention

实现：

```python
class MultiHeadSelfAttention(nn.Module):
    ...
```

完成：

```text
[B,N,D]
↓
QKV
↓
[B,heads,N,head_dim]
↓
QKᵀ
↓
[B,heads,N,N]
↓
Attention
↓
[B,N,D]
```

---

## 4.3 MLP

实现：

```python
class MLP(nn.Module):
    ...
```

结构：

```text
Linear
↓
GELU
↓
Linear
```

---

## 4.4 TransformerEncoderBlock

实现：

```python
class TransformerEncoderBlock(nn.Module):
    ...
```

采用 Pre-LN 结构：

```text
x + Attention(LN(x))
x + MLP(LN(x))
```

---

## 4.5 MiniViT

实现完整：

```python
class MiniViT(nn.Module):
    ...
```

完整数据流：

```text
Image
↓
PatchEmbedding
↓
CLS Token
↓
Position Embedding
↓
Transformer Encoder
↓
LayerNorm
↓
CLS
↓
Classification Head
```

成功运行：

```text
Input:
[2,3,32,32]

Output:
[2,10]
```

---

# 五、今天完成的实验

## 实验 1：Token 数量

```text
P=8  → 16 patches → 17 tokens
P=4  → 64 patches → 65 tokens
P=2  → 256 patches → 257 tokens
```

---

## 实验 2：Attention Shape

实际运行得到：

```text
P=8 → [2,4,17,17]
P=4 → [2,4,65,65]
P=2 → [2,4,257,257]
```

说明：

```text
Attention Shape =
[B, num_heads, N, N]
```

---

## 实验 3：Attention Matrix 元素数量

```text
P=8 → 17²  = 289
P=4 → 65²  = 4225
P=2 → 257² = 66049
```

验证了 Self-Attention 的二次复杂度。

---

## 实验 4：参数量

实际得到：

```text
P=8 → 214,218
P=4 → 208,074
P=2 → 218,058
```

重要结论：

> Patch 数量大幅增加，并不意味着模型参数量按照同样倍数增加。

需要区分：

```text
模型参数量
vs
Attention 计算量
vs
运行时显存
```

Transformer 主体参数主要与：

* `embed_dim`
* `num_heads`
* `depth`
* MLP hidden dimension

相关。

而 Attention 的计算和中间激活则受到 `N²` 强烈影响。

---

# 六、今天回答中暴露出的薄弱点

## 6.1 Attention 矩阵 Shape 曾出现错误

最终验收第 6 题中曾回答：

```text
Q = [2,8,197,64]
Kᵀ = [2,8,64,197]

错误写成：
Attention = [2,8,64,64]
```

正确应该是：

```text
[2,8,197,64]
×
[2,8,64,197]

=
[2,8,197,197]
```

必须牢牢记住：

$$
[B,H,N,d]\times[B,H,d,N]
=
[B,H,N,N]
$$

其中 `d` 是 Head 内部特征维度，最终会被矩阵乘法消掉。

这是 Day9 前必须重点复习的内容。

---

## 6.2 Linear Projection 与高层语义不能直接画等号

曾表述为：

> Linear Projection 完成从底层像素到高层语义的映射。

更严谨的说法：

> Linear Projection 将 Patch 的原始像素向量映射到统一的 embedding 空间；高层语义主要通过后续 Transformer Encoder 的多层特征交互逐渐形成。

---

## 6.3 科研实验中的因果归因还需要加强

在实验设计题中曾出现：

> 如果 P=4 AP 下降，可能是 GPU/硬件无法支持。

这个归因不成立。

如果实验已经成功训练并获得 AP：

```text
P=4 → AP下降
```

不能直接归因于 GPU。

应该考虑：

* Patch 太小导致冗余/噪声信息增加；
* 模型容量不足；
* 超参数没有适配；
* 数据集本身不需要更细粒度；
* 训练策略不适合；
* 更高 Token 数量没有转化成有效信息。

科研实验应该遵循：

```text
观察结果
↓
提出多个假设
↓
设计额外实验
↓
排除可能原因
↓
形成结论
```

而不是看到结果后立即进行单一归因。

---

# 七、目前还存在的问题

## 7.1 需要继续巩固 Attention Shape

尤其要熟练掌握：

```text
Q:
[B,H,N,d]

K:
[B,H,N,d]

Kᵀ:
[B,H,d,N]

QKᵀ:
[B,H,N,N]
```

这是后续学习 DETR、Cross-Attention、Vision-Language Model 时非常重要的基础。

---

## 7.2 需要进一步理解 Token 数量与显存的关系

目前已经理解：

$$
N^2
$$

会导致 Attention 成本增加。

下一阶段需要进一步认识：

> 实际 GPU 显存不仅包含 Attention Matrix，还包括 Q/K/V、中间激活、梯度、优化器状态等。

---

## 7.3 对 ViT 的局限性还需要继续深入

目前主要理解了：

* Token 数量；
* Attention 计算；
* Patch Size。

后续还需要理解：

* ViT 对数据规模的依赖；
* CNN 的 inductive bias；
* 为什么 ViT 在小数据集上可能不如 CNN；
* Hierarchical Vision Transformer；
* 为什么目标检测不能简单照搬 Image Classification ViT。

这些内容会自然连接到后续 DETR / Swin / OVD。

---

# 八、复试可能被问到的问题

## 基础问题

### 1.

什么是 Vision Transformer？

### 2.

为什么 ViT 要把图片切成 Patch？

### 3.

Patch Embedding 和普通 Embedding 有什么区别？

### 4.

为什么 ViT 需要 Position Embedding？

### 5.

CLS Token 是干什么的？

### 6.

为什么 CLS 可以表示整张图像？

---

## Tensor Shape 问题

### 7.

224×224 图片，Patch Size=16，有多少个 Patch？

答案：

$$
14\times14=196
$$

### 8.

加入 CLS 后有多少 Token？

$$
197
$$

### 9.

如果：

```text
B=2
N=197
D=512
heads=8
```

那么：

```text
Q.shape = [2,8,197,64]
K.shape = [2,8,197,64]
QKᵀ.shape = [2,8,197,197]
```

---

## 深入问题

### 10.

为什么 Patch Size 越小，计算量增长很快？

核心：

$$
N\propto1/P^2
$$

而：

$$
Attention\sim O(N^2D)
$$

---

### 11.

ViT 和 CNN 的主要区别是什么？

重点：

```text
CNN：
局部连接 + 强局部归纳偏置

ViT：
Self-Attention + 全局 Token 交互
```

---

### 12.

Patch Size 越小是不是一定越好？

不是。

需要权衡：

```text
空间细节
vs
计算量
vs
显存
vs
有效信息
```

---

### 13.

如果 Patch Size 从 16 改成 8，Token 数量增加多少？

4 倍。

Attention 主要计算规模约增加：

16 倍。

---

### 14.

如果一个小目标只占很小的区域，Patch Size 太大会有什么问题？

一个 Patch 可能包含：

```text
目标 + 背景
```

导致目标空间信息被压缩。

---

### 15.

如果 Patch Size 很小但 AP 反而下降，应该如何分析？

不能直接归因。

应该：

```text
提出多个假设
↓
控制变量
↓
设计消融实验
↓
验证假设
↓
分析原因
```

---

# 九、Day9 学习前必须复习的内容

Day9 开始之前，建议重点复习以下内容。

## 第一优先级：必须熟练

### 1. ViT 完整数据流

```text
[B,C,H,W]
↓
Patch
↓
[B,N,P²C]
↓
Linear
↓
[B,N,D]
↓
CLS
↓
[B,N+1,D]
↓
Position Embedding
↓
[B,N+1,D]
↓
Transformer Encoder
↓
[B,N+1,D]
↓
CLS
↓
[B,D]
↓
Head
↓
[B,num_classes]
```

---

### 2. Multi-Head Attention Shape

必须做到不查资料直接写：

```text
[B,N,D]
↓
QKV
↓
[B,H,N,d]
↓
QKᵀ
↓
[B,H,N,N]
↓
Attention @ V
↓
[B,H,N,d]
↓
[B,N,D]
```

---

### 3. Patch Size 与 Token 数量

牢记：

$$
N=\frac{H}{P}\frac{W}{P}
$$

以及：

$$
Attention\sim O(N^2D)
$$

---

## 第二优先级：理解

### 4. CLS 为什么可以表示全局图像

因为：

```text
CLS
↓
Self-Attention
↓
与所有 Patch Token 交互
↓
多层信息融合
↓
Global Representation
```

---

### 5. Position Embedding 为什么存在

因为 Transformer 本身没有 CNN 那样的显式空间结构，需要额外注入位置信息。

---

### 6. CNN vs ViT

重点理解：

```text
CNN：
局部性 / 平移等归纳偏置 / 层级感受野

ViT：
Token 化 / Self-Attention / 全局交互
```

---

# 十、Day8 是否达到进入 Day9 的标准？

## 结论：✅ 达到，可以进入 Day9。

理由如下。

### ① 理论链路已经完整

你已经能够独立解释：

```text
Image
→ Patch
→ Patch Embedding
→ Position Embedding
→ CLS
→ Transformer
→ Classification
```

不是停留在概念记忆，而是能够解释每一步为什么存在。

---

### ② 已经能够自己写 MiniViT

你独立完成了：

```text
PatchEmbedding
MultiHeadSelfAttention
MLP
TransformerEncoderBlock
MiniViT
```

并成功运行：

```text
[2,3,32,32]
→
[2,10]
```

这说明理论已经开始转化成代码能力。

---

### ③ 已经进行了真正的控制变量实验

你没有停留在：

> “Patch 越小计算量越大。”

而是实际比较：

```text
P=8
P=4
P=2
```

并观察：

* Token 数量；
* Attention Shape；
* Attention Matrix 元素；
* 参数量。

这是从“学习模型”向“做实验”迈出的重要一步。

---

### ④ 仍存在的小问题不足以阻止进入 Day9

目前主要问题：

```text
Attention Matrix Shape 曾出现一次错误
实验结果的因果归因还需要加强
```

这两个问题属于**可以边学习边强化的问题**，并不是 ViT 核心结构没有掌握。

特别是你在最后复试模拟题中，已经能够比较完整地用自己的语言解释 ViT，这说明整体知识结构已经形成。

---

# Day8 最终能力定位

经过今天的学习，你已经从：

> **“知道 Transformer 可以用于视觉”**

进入到了：

> **“能够解释、实现并实验验证一个最小 Vision Transformer。”**

目前能力链已经变成：

```text
CNN
↓
Attention
↓
Self-Attention
↓
Multi-Head Attention
↓
Transformer Block
↓
Transformer Encoder
↓
Position Encoding
↓
Vision Transformer
↓
Patch / Token / CLS
↓
Tensor Shape
↓
PyTorch Implementation
↓
实验设计
```

这条链路已经可以继续向下一阶段延伸。

**Day8 状态：✅ 完成**

**进入 Day9：✅ 允许**

**Day9 前重点：Attention Shape + ViT 数据流 + Patch Size/复杂度 + 科研实验归因。**
