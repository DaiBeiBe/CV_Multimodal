# Day09｜ViT 深入理解：Patch Size、Attention 计算与视觉 Inductive Bias

> 学习路线：CV → CNN → 目标检测 → Transformer → ViT → DETR → Open-Vocabulary Detection → Vision-Language → 多模态

---

## 1. 今日真正掌握的知识

### 1.1 Patch Size 与 Token 数量

已经掌握 ViT 图像 Token 化之后最核心的数量关系：

$$
N=\frac{HW}{P^2}
$$

其中：

* \(H,W\)：输入图像空间尺寸
* \(P\)：Patch Size
* \(N\)：Patch Token 数量

例如：

$$
224\times224,\quad P=16
$$

得到：

$$
N=\frac{224\times224}{16^2}=196
$$

当：

$$
P:16\rightarrow8
$$

由于 Patch 面积缩小为原来的 \(1/4\)，所以：

$$
N\rightarrow4N
$$

已经理解：

> Patch Size 越小，并不意味着 ViT 一定越好。

Patch Size 减小：

* 优点：保留更加细粒度的空间信息；
* 缺点：Token 数量增加；
* 进一步导致 Self-Attention 的计算和显存开销快速增加。

因此 Patch Size 本质上存在：

> **空间细节 / 表示能力 / 计算成本之间的 Trade-off。**

---

### 1.2 Self-Attention Tensor Shape

已经掌握：

$$
X\in\mathbb{R}^{B\times N\times D}
$$

经过线性投影：

$$
Q,K,V\in\mathbb{R}^{B\times N\times D}
$$

计算：

$$
QK^T
$$

得到：

$$
\boxed{[B,N,N]}
$$

已经能够理解为什么 Attention Matrix 是 \(N\times N\)：

> 每一个 Query Token 都需要与所有 Key Token 计算相关性。

因此：

* 行：Query Token
* 列：Key Token
* 每个位置：一个 Query-Key 之间的相关性分数

---

### 1.3 Attention Matrix 的元素数量 ≠ Attention 计算复杂度

这是今天非常重要的一个知识点，而且已经通过自测得到确认。

Attention Matrix：

$$
QK^T\in[B,N,N]
$$

因此矩阵本身的元素数量为：

$$
\boxed{BN^2}
$$

而不是：

$$
BN^2D
$$

但是计算每一个 Attention Score 时，需要进行一个 \(D\) 维向量的点积：

$$
q_i\cdot k_j
=
\sum_{d=1}^{D}q_{id}k_{jd}
$$

所以计算 \(QK^T\) 的主要复杂度为：

$$
\boxed{O(BN^2D)}
$$

忽略 Batch 后：

$$
\boxed{O(N^2D)}
$$

已经能够明确区分：

| 概念          | 含义                              |
| ----------- | ------------------------------- |
| \(N^2\)     | Attention Matrix 的空间规模/元素数量     |
| \(N^2D\)    | 计算 Attention Score 的主要计算复杂度     |
| \([B,N,N]\) | Attention Matrix 的 Tensor Shape |

这是 Day9 最重要的数学理解之一。

---

### 1.4 Patch Size 对 Attention 成本的影响

已经掌握：

$$
N=\frac{HW}{P^2}
$$

因此：

$$
P\downarrow
\Rightarrow N\uparrow
$$

而 Self-Attention：

$$
O(N^2D)
$$

所以：

$$
P\downarrow
\Rightarrow N^2\uparrow\uparrow
\Rightarrow Attention\ Cost\uparrow\uparrow
$$

例如：

$$
P:16\rightarrow8
$$

则：

$$
N\rightarrow4N
$$

因此：

$$
N^2\rightarrow16N^2
$$

在 \(D\) 不变时：

$$
O(N^2D)\rightarrow16O(N^2D)
$$

已经能够正确回答：

> Token 数量增加 4 倍，但 Attention Matrix 的元素数量和主要 Attention 计算量增加约 16 倍。

---

### 1.5 高分辨率图像为什么对 ViT 特别昂贵

已经建立完整因果链：

$$
H,W\uparrow
$$

↓

$$
N=\frac{HW}{P^2}\uparrow
$$

↓

$$
Attention\ Matrix\ [B,N,N]
$$

↓

$$
N^2\uparrow
$$

↓

$$
Self\text{-}Attention\ Cost\sim O(N^2D)\uparrow
$$

因此，当图像分辨率增加时，ViT 的 Self-Attention 成本会快速增长。

特别是当图像的长和宽同时扩大时：

$$
H,W\rightarrow2H,2W
$$

则：

$$
N\rightarrow4N
$$

进而：

$$
N^2\rightarrow16N^2
$$

这解释了为什么高分辨率视觉任务会给标准 ViT 的全局 Self-Attention 带来很大的计算压力。

---

### 1.6 CNN 与 ViT 的视觉 Inductive Bias

已经理解 Inductive Bias 的基本概念：

> Inductive Bias 是模型结构或学习算法在训练前对数据规律所做的假设。

CNN 具有较强的显式视觉归纳偏置，主要包括：

#### Locality

卷积核只处理局部区域：

> 邻近像素之间存在重要的局部关系。

#### Weight Sharing

同一个卷积核可以在图像不同位置使用：

> 相同类型的局部模式可以出现在不同空间位置。

因此 CNN 在模型结构层面已经编码了比较强的视觉先验。

---

### 1.7 ViT 的视觉 Inductive Bias 与数据依赖

已经理解一个重要的准确表述：

> ViT 不是“完全没有 Inductive Bias”，而是相比 CNN，其显式的视觉归纳偏置更弱。

因此 ViT 更依赖：

* 大规模数据；
* 大规模预训练；
* 数据驱动地学习视觉规律。

但不能理解为：

> “CNN 有 Bias，所以不需要数据。”

正确理解是：

> CNN 和 ViT 都需要数据，只是 CNN 的结构先验更强，因此在有限数据条件下通常能够获得更多结构上的帮助；ViT 则需要更多地从数据中学习视觉规律。

---

### 1.8 标准 ViT 分类与目标检测的区别

已经理解：

标准 ViT 分类流程：

```text
Image
  ↓
Patch Embedding
  ↓
Patch Tokens
  ↓
CLS Token
  ↓
Position Embedding
  ↓
Transformer Encoder
  ↓
CLS Representation
  ↓
Classification Head
  ↓
Image-level Class
```

标准分类模型最终关注的是：

> 整张图像属于什么类别？

而目标检测需要：

> 图像中有哪些目标？每个目标在哪里？属于什么类别？

因此检测输出需要多个：

$$
(Class,\ Bounding\ Box)
$$

标准 ViT 分类结构的问题不是：

> “ViT 天生不能做目标检测。”

而是：

> **标准 ViT 分类结构的输出形式是 Image-level Prediction，并没有直接设计成 Multiple Object + Bounding Box Prediction。**

这为下一阶段学习 DETR 建立了直接动机。

---

### 1.9 Patch Token 不等于 Object Token

今天在 Day9 末尾提前接触到了一个非常重要的 Day10 过渡问题。

不能简单认为：

> 一个 Patch Token 对应一个目标。

因为：

* 一个目标可能覆盖很多 Patch；
* 一个 Patch 可能包含目标的一部分；
* 多个目标可能同时涉及大量 Patch；
* Patch Token 本质上表示图像空间区域，而不是明确的 Object Instance。

因此：

$$
Patch\ Token\neq Object\ Token
$$

这也是为什么后续 DETR 不会简单地：

> “让每个 Patch Token 负责一个目标。”

而是引入新的机制解决 Object-level Prediction。

---

## 2. 回答中暴露出的薄弱点

### 2.1 Tensor Shape 的数学解释还可以更严谨

在自测第 4 题中回答：

> “因为矩阵乘法就是这样。”

结论是正确的，但表达没有展示出完整的线性代数推导。

应该能够主动说出：

$$
[B,N,D]\times[B,D,N]
\rightarrow[B,N,N]
$$

核心原因是矩阵乘法：

$$
[m,k]\times[k,n]\rightarrow[m,n]
$$

因此：

$$
[N,D]\times[D,N]\rightarrow[N,N]
$$

后续阅读 Transformer 源码时，需要做到：

> 看到矩阵乘法代码，就能主动推导 Tensor Shape，而不是只记忆最终 Shape。

---

### 2.2 对 Inductive Bias 的表述曾出现概念混淆

自测第 8 题中出现了：

> “在训练前就已经有这部分特征，不属于视觉先验。”

这是错误的。

正确理解：

CNN 在训练之前并不知道具体的“边缘”“猫耳朵”等语义特征。

CNN 提前编码的是：

* Locality；
* Weight Sharing；
* 局部空间结构等假设。

所以：

> **CNN 预先规定的是视觉规律的结构假设，而不是预先学会具体视觉特征。**

这是目前 Day9 最需要修正的概念点。

---

### 2.3 “CLS 只能分类一次”的表述需要更精确

可以说：

> 标准 ViT 使用一个 CLS Token，并通过一个 Classification Head 输出一个 Image-level Prediction。

但不能把它理解成：

> “一个 Token 天生只能产生一个结果。”

实际上输出形式由：

> Representation + Prediction Head + Task Design

共同决定。

这一点在后续理解 Detection Head、Object Query、Decoder 时非常重要。

---

### 2.4 Attention 复杂度的表达要避免简写歧义

自测中使用过：

> “NND”

实际想表达的是：

$$
N^2D
$$

后续建议统一写：

$$
\boxed{O(N^2D)}
$$

避免把 \(NND\) 写法带入代码或论文笔记。

---

## 3. 今天完成的 Python 代码 / 实验

今天使用并验证了 Day8 MiniViT 代码，并围绕 Patch Size 对 ViT 的影响完成了实验。

### 3.1 PatchEmbedding

核心流程：

```text
Image
 ↓
Patch Partition
 ↓
Flatten
 ↓
Linear Projection
 ↓
Patch Tokens
```

代码中：

```python
num_patches = (img_size // patch_size) ** 2
patch_dim = in_channels * patch_size * patch_size
```

能够正确建立：

$$
P\rightarrow N
$$

的关系。

---

### 3.2 Multi-Head Self-Attention

已经验证：

```text
Q, K, V
 ↓
Q @ Kᵀ
 ↓
Attention Matrix
 ↓
Softmax
 ↓
Attention @ V
```

重点理解了：

```text
Q/K/V
[B, N, D]
```

经过多头拆分后：

```text
[B, h, N, Dh]
```

最终：

```text
Attention
[B, h, N, N]
```

---

### 3.3 实验一：Patch Size → Token 数量

使用不同 Patch Size：

```text
P = 8
P = 4
P = 2
```

验证：

$$
N=\left(\frac{H}{P}\right)^2
$$

并观察到：

```text
Patch Size ↓
Token 数量 ↑
```

---

### 3.4 实验二：不同 Patch Size 下 MiniViT 前向传播

验证不同 Patch Size 下模型仍可以完成完整 Forward：

```text
Image
→ PatchEmbedding
→ CLS
→ Position Embedding
→ Transformer Encoder
→ Classification Head
```

说明 Patch Size 改变后，整个 Token Pipeline 仍然能够工作。

---

### 3.5 实验三：Attention Matrix 大小

计算：

$$
N_{total}=N_{patch}+1
$$

其中 `+1` 来自 CLS Token。

例如 `32×32` 图像：

| Patch Size | Patch Tokens | Total Tokens |       Attention Matrix |
| ---------- | -----------: | -----------: | ---------------------: |
| 8          |           16 |           17 |     \(17\times17=289\) |
| 4          |           64 |           65 |    \(65\times65=4225\) |
| 2          |          256 |          257 | \(257\times257=66049\) |

验证了：

> Patch Size 减小后，Attention Matrix 的规模增长非常快。

同时注意到：

> 实际 ViT 包含 CLS Token，因此严格计算时使用 \(N_{total}=N_{patch}+1\)。

---

### 3.6 实验四：参数量与 Patch Size

还进行了不同 Patch Size 下 MiniViT 参数量的实验。

这一实验帮助区分：

> **Attention 的计算量随 Token 数量呈 \(N^2\) 增长，不等于模型参数量也一定呈 \(N^2\) 增长。**

这是一个重要的概念区分。

---

## 4. 还存在的问题

### 必须继续巩固

1. 从线性代数规则直接推导 Transformer Tensor Shape；
2. 严格区分：

   * Tensor Shape；
   * Tensor 元素数量；
   * FLOPs / 计算复杂度；
   * 参数量；
3. 更准确地理解 Inductive Bias；
4. 更准确地区分 Patch Token、Image-level Representation 和 Object-level Representation；
5. 理解标准 ViT 分类结构为什么不能直接完成多目标检测。

### 下一阶段需要解决

进入 DETR 后，需要进一步理解：

```text
Image
 ↓
Backbone
 ↓
Visual Features
 ↓
Transformer
 ↓
Object Queries
 ↓
Object-level Predictions
 ↓
Class + Bounding Box
```

尤其需要解决：

> **Transformer 如何从“处理 Token”进一步变成“预测多个 Object”？**

---

## 5. 复试可能被问到的问题

### 基础问题

1. ViT 为什么要把图片划分成 Patch？
2. Patch Size 是什么？
3. 为什么 Patch Size 越小，Token 数量越多？
4. Token 数量公式是什么？
5. 为什么 Self-Attention 在高分辨率图像上计算成本很高？
6. Self-Attention 的主要计算复杂度是多少？
7. Attention Matrix 的 Shape 是什么？
8. 为什么是 \(N\times N\)？
9. \(N^2\) 和 \(N^2D\) 有什么区别？

### CNN vs ViT

10. 什么是 Inductive Bias？
11. CNN 为什么具有较强的视觉 Inductive Bias？
12. CNN 的 Locality 是什么意思？
13. Weight Sharing 为什么是视觉先验？
14. 为什么 ViT 通常更加依赖大规模数据？
15. ViT 是不是完全没有 Inductive Bias？

### ViT 与 Detection

16. 标准 ViT 为什么主要用于图像分类？
17. CLS Token 的作用是什么？
18. 为什么标准 ViT 分类不能直接输出多个 Bounding Box？
19. ViT 能不能用于目标检测？
20. Patch Token 和 Object Token 有什么区别？

### 深入追问

21. 如果 Patch Size 从 16 改成 8，Token 数量变化多少？
22. Attention Matrix 大小变化多少？
23. QKᵀ 的 Shape 为什么是 `[B,N,N]`？
24. 如果 `D=768，heads=12`，每个 Head 的维度是多少？
25. `[B,h,N,Dh] × [B,h,Dh,N]` 的结果是什么？
26. 为什么 Attention Matrix 元素数量是 \(N^2\)，但计算复杂度是 \(N^2D\)？
27. 如果图像分辨率扩大两倍，Self-Attention 成本大概变化多少？

---

## 6. Day10 学习前必须复习的内容

Day10 不需要重新学习整个 Transformer，只需要重点复习下面这些内容。

### 6.1 必须熟练掌握

$$
\boxed{N=\frac{HW}{P^2}}
$$

以及：

$$
\boxed{Q,K,V\in[B,N,D]}
$$

$$
\boxed{QK^T\in[B,N,N]}
$$

$$
\boxed{Attention\ Cost\approx O(N^2D)}
$$

---

### 6.2 必须能够口头解释

看到：

```text
Patch Size ↓
```

能够马上说出：

```text
Patch Size ↓
→ Patch 数量 ↑
→ Token 数量 N ↑
→ Attention Matrix N×N ↑
→ Attention 计算和显存开销 ↑↑
```

---

### 6.3 复习 CNN vs ViT Inductive Bias

必须能够解释：

```text
CNN
├── Locality
├── Weight Sharing
└── 强视觉先验

ViT
├── 显式视觉先验较弱
└── 更依赖大规模数据 / 预训练
```

并牢记：

> ViT 不是没有 Inductive Bias。

---

### 6.4 复习 ViT 分类输出

必须能够解释：

```text
Patch Tokens
 ↓
Transformer Encoder
 ↓
CLS Token
 ↓
Classification Head
 ↓
Image-level Prediction
```

以及：

```text
Detection
 ↓
Multiple Objects
 ↓
Class + Bounding Box
```

---

### 6.5 带着一个问题进入 Day10

不要提前背 DETR 公式，只需要思考：

> **如果一个 CLS Token 只能形成图像级表示，那么 Transformer 如何才能预测一张图片中的多个不同目标？**

这就是 Day10 的核心起点。

---

## 7. Day9 是否达到进入 Day10 的标准？

# ✅ 达到，Day9 验收通过。

### 判断依据

#### ① Patch 与 Token 数量

能够独立使用：

$$
N=\frac{HW}{P^2}
$$

进行计算。

**已掌握。**

---

#### ② Self-Attention Tensor Shape

能够正确回答：

$$
[B,N,D]\rightarrow[B,N,N]
$$

并且在 Multi-Head Attention 中能够正确推导：

$$
[B,h,N,D_h]
\rightarrow
[B,h,N,N]
$$

**已掌握。**

---

#### ③ Attention 复杂度

已经能够正确区分：

$$
\text{Matrix Elements}=O(N^2)
$$

和：

$$
\text{Attention Computation}=O(N^2D)
$$

这是进入后续 Transformer / Detection 学习必须具备的基础。

**已掌握。**

---

#### ④ Patch Size 与计算成本

能够独立解释：

$$
P\downarrow
\rightarrow N\uparrow
\rightarrow N^2\uparrow
\rightarrow O(N^2D)\uparrow
$$

**已掌握。**

---

#### ⑤ CNN vs ViT

能够理解：

* CNN 的 Locality；
* Weight Sharing；
* CNN 较强的视觉 Inductive Bias；
* ViT 相对较弱的显式视觉先验；
* ViT 对大规模数据 / 预训练的依赖。

虽然 Inductive Bias 的定义在回答中出现过一次概念表述错误，但经过纠正后已经明确理解。

**达到进入 Day10 的要求。**

---

#### ⑥ ViT → Detection 的过渡

已经意识到：

> 标准 ViT 分类输出不适合直接进行多目标检测。

同时能够理解：

> Patch Token ≠ Object Token。

这非常重要，因为 Day10 DETR 正是从这个问题出发。

**达到进入 Day10 的要求。**

---

## Day9 最终能力评价

| 能力                                  | 当前状态          |
| ----------------------------------- | ------------- |
| Patch Size / Token 数量               | 🟢 掌握         |
| \(N=HW/P^2\)                        | 🟢 掌握         |
| Q/K/V Shape                         | 🟢 掌握         |
| Attention Matrix Shape              | 🟢 掌握         |
| \(N^2\) vs \(N^2D\)                 | 🟢 掌握         |
| Self-Attention 复杂度                  | 🟢 掌握         |
| CNN Inductive Bias                  | 🟢 掌握，需保持概念严谨 |
| ViT 数据依赖                            | 🟢 掌握         |
| ViT 分类输出                            | 🟢 掌握         |
| Patch Token vs Object Token         | 🟢 初步掌握       |
| Transformer 源码 Tensor Shape 推导      | 🟡 需要继续训练     |
| Detection 中 Object-level Prediction | 🟡 即将进入 Day10 |

---

# Day9 → Day10 知识衔接

到 Day9 为止，你已经解决了：

> **ViT 如何把图像变成 Token，以及 Self-Attention 为什么在视觉任务中昂贵。**

接下来 Day10 要解决一个更关键的问题：

> **Transformer 不只是能处理图像 Token，它能不能直接参与目标检测？**

最终进入：

```text
CNN / ViT
   ↓
Visual Feature
   ↓
Transformer
   ↓
Object Query
   ↓
多个 Object Prediction
   ↓
Class + Bounding Box
```

正式进入：

# Day10｜DETR：从 ViT 到 Transformer Object Detection

核心学习顺序：

**为什么需要 DETR → Detection as Set Prediction → Object Query → Transformer Decoder → Hungarian Matching → Classification + Bounding Box Loss → DETR 完整数据流 → 官方代码阅读。**
