# Day03｜Attention：从局部信息到动态全局信息聚合

> 学习主题：从 CNN 的局部感受野与长距离依赖过渡到 Attention  
> 学习目标：理解 Attention 的核心思想、Q/K/V、Scaled Dot-Product Attention，并使用 PyTorch 手写简化版 Attention。  
> Day3 状态：**通过 ✅**

---

# 1. 今天真正掌握的知识

## 1.1 CNN 与 Attention 的关系

Day2 学习了 CNN 的一个重要特点：

> CNN 通过局部连接逐层传播信息，随着网络深度增加，感受野逐渐扩大，从而建立远距离依赖。

但这种远距离信息交互具有一个特点：

- 每一层主要进行局部信息交互；
- 远距离位置之间通常需要经过多层传播；
- 中间需要经过多次特征变换。

Attention 提供了另一种思路：

> 一个位置可以根据当前输入中的相关性，直接选择并聚合其他位置的信息。

因此：

```text
CNN
局部连接
    ↓
逐层传播
    ↓
逐渐建立远距离依赖


Attention
计算位置之间的相关性
    ↓
动态分配权重
    ↓
聚合相关位置的信息


需要注意：

> Attention 的核心并不是简单的“全局信息”，而是**基于相关性进行动态信息选择与聚合**。

---

## 1.2 Attention 的核心思想：动态信息聚合

假设有四个特征：

```text
A B C D
```

现在需要更新 A。

如果模型计算得到：

```text
A → B：0.1
A → C：0.7
A → D：0.1
A → A：0.1
```

那么新的 A 可以表示为：

$$
A'=0.1A+0.1B+0.7C+0.1D
$$

这说明：

> C 对新的 A 的贡献最大。

这里需要注意一个容易混淆的地方：

不能简单说：

> “新的 A 就是最像 C。”

更加准确的说法是：

> **C 对新的 A 的信息贡献最大。**

因为最终的输出是多个 Value 加权聚合后的结果，而不是简单复制某一个输入。

---

# 2. Q / K / V

Attention 使用三个核心表示：

* Query（Q）
* Key（K）
* Value（V）

可以用一个直观的方式理解：

```text
Q：我想找什么？

K：我有什么，可以用来匹配？

V：如果你关注我，我真正提供什么信息？
```

---

## 2.1 Query

Query 表示当前特征的“查询需求”。

例如：

> 当前这个位置想知道哪些其他位置与自己相关。

---

## 2.2 Key

Key 用于与 Query 进行匹配。

通过：

$$
QK^T
$$

可以计算 Query 与各个 Key 之间的匹配程度。

---

## 2.3 Value

Value 表示真正用于最终信息聚合的内容。

Attention 首先通过 Q/K 决定：

> 应该关注谁？

然后通过 Attention Weight 决定：

> 从这些位置提取多少信息？

最后对 V 进行加权求和。

因此：

```text
Q/K
↓
决定“关注谁、关注多少”
↓
Attention Weight
↓
对 V 进行加权
↓
新的特征表示
```

---

## 2.4 一个需要注意的细节

V 并不是一个“完全固定、与模型无关的信息”。

在实际 Transformer 中，Q/K/V 通常都是输入特征经过可学习的线性投影得到的：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

因此更准确的理解是：

> V 是当前位置经过 Value 投影后，用于提供给其他位置聚合的信息表示。

---

# 3. QKᵀ：计算 Query-Key 相关性

假设：

$$
Q,K\in R^{N\times D}
$$

那么：

$$
K^T\in R^{D\times N}
$$

因此：

$$
QK^T
$$

的形状为：

$$
(N\times D)(D\times N)
=
N\times N
$$

得到：

$$
S=QK^T
$$

其中：

$$
S_{ij}=Q_iK_j^T
$$

表示：

> 第 i 个 Query 与第 j 个 Key 的匹配程度。

---

## 3.1 实验中的例子

实验设置：

$$
Q=K=
\begin{bmatrix}
1&0&0&0\\
0&1&0&0\\
0&0&1&0
\end{bmatrix}
$$

因此：

```text
Q.shape = [3, 4]
K.shape = [3, 4]
```

计算：

```python
scores = Q @ K.transpose(-2, -1)
```

得到：

```text
tensor([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])
```

说明：

* Q1 与 K1 最匹配；
* Q2 与 K2 最匹配；
* Q3 与 K3 最匹配。

---

# 4. Scaling：为什么除以 sqrt(d_k)

Scaled Dot-Product Attention 使用：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

其中：

$$
d_k
$$

表示 Key 的特征维度。

实验中：

$$
d_k=4
$$

所以：

$$
\sqrt{d_k}=2
$$

原始分数：

$$
[1,0,0]
$$

经过 Scaling：

$$
[0.5,0,0]
$$

---

## 4.1 为什么需要 Scaling？

如果特征维度较大：

$$
QK^T
$$

的数值可能变得比较大。

Softmax 输入过大时容易出现：

> Softmax 输出过于尖锐，接近 one-hot。

例如：

```text
[100, 1, 0]
```

Softmax 后可能几乎全部集中在第一个位置。

这会导致梯度变得不理想。

因此通过：

$$
\frac{1}{\sqrt{d_k}}
$$

控制分数的尺度，使训练更加稳定。

---

# 5. Softmax：将相关性转成 Attention Weight

Scaled Score：

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

不能直接用于加权求和。

需要经过：

$$
A=Softmax(S)
$$

得到 Attention Weight。

Softmax：

$$
Softmax(x_i)
=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
$$

---

## 5.1 Softmax 的维度

本次 PyTorch 实验中：

```python
attention_weights = torch.softmax(scores, dim=-1)
```

使用：

```python
dim=-1
```

是因为：

```text
每一行
=
一个 Query 对所有 Key 的关注分布
```

例如：

```text
       K1      K2      K3
Q1   0.4519  0.2741  0.2741
Q2   0.2741  0.4519  0.2741
Q3   0.2741  0.2741  0.4519
```

第一行：

$$
[0.4519,0.2741,0.2741]
$$

表示：

> Query 1 对 Key 1、Key 2、Key 3 的注意力权重。

并且：

$$
0.4519+0.2741+0.2741\approx1
$$

所以每一个 Query 对所有 Key 的权重构成一个归一化分布。

---

# 6. Attention 最终计算

完整公式：

$$
\boxed{
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

可以拆成：

```text
Q、K、V
   ↓
QKᵀ
   ↓
相关性分数
   ↓
除以 √d_k
   ↓
Softmax
   ↓
Attention Weight
   ↓
Attention Weight × V
   ↓
新的特征表示
```

---

# 7. Tensor Shape 总结

假设：

$$
Q,K,V\in R^{N\times D}
$$

其中：

* N：特征/token 的数量；
* D：特征维度。

那么：

### 第一步

$$
QK^T
$$

形状：

$$
(N\times D)(D\times N)
=
N\times N
$$

---

### 第二步

$$
Softmax(QK^T)
$$

仍然是：

$$
N\times N
$$

---

### 第三步

$$
AttentionWeight\times V
$$

即：

$$
(N\times N)(N\times D)
=
N\times D
$$

所以最终：

$$
\boxed{
N\times D
}
$$

也就是说：

> Attention 输入和输出的 token 数量、特征维度可以保持一致，但每一个位置的表示已经融合了其他位置的信息。

---

# 8. 今天完成的 Python / PyTorch 实验

## 8.1 构造 Q/K/V

使用 PyTorch 构造：

```python
import math
import torch

Q = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float32,
)

K = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float32,
)

V = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float32,
)
```

Tensor Shape：

```text
Q: [3, 4]
K: [3, 4]
V: [3, 4]
```

---

## 8.2 计算 QKᵀ

```python
scores = Q @ K.transpose(-2, -1)
```

得到：

```text
tensor([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.]
])
```

Shape：

```text
[3, 3]
```

---

## 8.3 Scaling

```python
scores = scores / math.sqrt(K.size(-1))
```

因为：

```text
d_k = 4
sqrt(d_k) = 2
```

所以：

```text
tensor([
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5]
])
```

---

## 8.4 Softmax

```python
attention_weights = torch.softmax(scores, dim=-1)
```

得到：

```text
tensor([
    [0.4519, 0.2741, 0.2741],
    [0.2741, 0.4519, 0.2741],
    [0.2741, 0.2741, 0.4519]
])
```

---

## 8.5 Attention 加权求和

```python
output = attention_weights @ V

print(output)
print(output.shape)
```

得到：

```text
tensor([
    [0.4519, 0.2741, 0.2741, 0.0000],
    [0.2741, 0.4519, 0.2741, 0.0000],
    [0.2741, 0.2741, 0.4519, 0.0000]
])
```

Shape：

```text
torch.Size([3, 4])
```

验证：

$$
(3\times3)(3\times4)
=
3\times4
$$

---

# 9. Attention 动态性实验

为了验证 Attention 并不是固定规则，而是会随着输入变化而变化，进行了动态性实验。

---

## 9.1 修改前

第一个 Query：

$$
Q_1=[1,0,0,0]
$$

与：

$$
K_1=[1,0,0,0]
$$

最匹配。

因此：

```text
Q1 → K1
```

第 1 行 Attention Weight：

```text
[0.4519, 0.2741, 0.2741]
```

主要关注 Key 1。

---

## 9.2 修改后

将：

```python
Q[0] = torch.tensor([0, 0, 1, 0])
K[2] = torch.tensor([0, 0, 1, 0])
```

此时：

$$
Q_1K_3^T
$$

变大。

得到：

```text
scores:

[
    [0.0, 0.0, 0.5],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5]
]
```

对应第 1 行：

```text
[0.2741, 0.2741, 0.4519]
```

此时：

```text
Q1 → K3
```

---

## 9.3 实验结论

实验验证：

```text
输入特征发生变化
        ↓
Q/K 之间的匹配关系发生变化
        ↓
QKᵀ发生变化
        ↓
Softmax 后 Attention Weight 发生变化
        ↓
最终信息聚合结果发生变化
```

因此：

> **Attention 权重是由当前输入特征之间的关系动态计算出来的，而不是固定的。**

这是今天实验最重要的结论之一。

---

# 10. 今天回答中暴露出的薄弱点

整体来看，今天没有出现影响 Day4 学习的核心概念错误，但存在几个需要持续注意的表达问题。

---

## 10.1 不要把“贡献最大”等同于“最终表示最像它”

之前对于：

```text
A' = 0.1A + 0.1B + 0.7C + 0.1D
```

你曾表达过：

> 新的 A 信息最相关于原始 C。

这个表达略显绝对。

更准确：

> **C 对新的 A 的信息贡献最大。**

原因是：

$$
A'
$$

是多个 Value 加权求和的结果，并不意味着最终表示一定“最像 C”。

---

## 10.2 “V 是位置本身的信息”需要进一步精确化

你已经理解 V 是真正参与信息聚合的内容，这是正确的。

但更严格地说：

> V 通常也是输入特征经过可学习线性映射得到的表示。

即：

$$
V=XW_V
$$

所以不能把 V 理解为完全固定、不经过变换的原始特征。

---

## 10.3 不要把 Attention 简化成“全局注意力”

目前已经能够正确理解：

> Attention 的重点是**根据相关性动态选择和聚合信息**。

后续学习 Transformer 时，需要继续保持这个理解。

“全局”只是 Attention 能够建立远距离信息交互的一个重要特点，而不是 Attention 的完整定义。

---

## 10.4 Scaling 的理解还需要在后续数学学习中进一步深化

目前已经能够正确解释：

> Scaling 可以控制分数尺度，避免 Softmax 过度饱和。

这是合格的。

后续如果深入 Transformer，需要进一步理解：

$$
Var(QK^T)
$$

为什么会随着：

$$
d_k
$$

增大，以及为什么除以：

$$
\sqrt{d_k}
$$

能够进行尺度控制。

Day3 暂时不要求深入推导。

---

# 11. 目前还存在的问题

## 11.1 尚未深入 Self-Attention

目前已经理解一般 Attention 的 Q/K/V 计算过程，但还没有正式展开：

$$
X\rightarrow Q,K,V
$$

以及：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

因此下一阶段需要理解：

> 为什么一个输入序列可以自己产生 Q、K、V，并完成自己与自己的信息交互。

---

## 11.2 尚未学习 Multi-Head Attention

目前只实现了：

> 单头 Scaled Dot-Product Attention。

还没有学习：

```text
一个 Attention
      ↓
多个 Head
      ↓
不同 Head 学习不同关系
      ↓
Concat
      ↓
Linear
```

这是 Day4 的重点。

---

## 11.3 尚未进入 Transformer Encoder

目前还没有系统学习：

* Multi-Head Attention；
* Residual Connection；
* LayerNorm；
* Feed Forward Network；
* Position Encoding / Positional Embedding；
* Transformer Encoder。

这些内容将在后续逐步建立。

---

# 12. 复试可能被问到的问题

## 基础问题

### Q1：Attention 是什么？

建议回答思路：

> Attention 根据 Query 和 Key 的相关性计算不同位置的重要程度，然后利用这些权重对 Value 进行加权聚合，从而生成新的特征表示。

---

### Q2：为什么 CNN 已经能够获得较大的感受野，还需要 Attention？

回答：

> CNN 可以通过增加网络深度逐渐建立长距离依赖，但远距离信息通常需要经过多层局部传播。Attention 可以根据特征之间的相关性直接进行信息交互，并动态决定应该关注哪些位置。

---

### Q3：Q、K、V 分别是什么？

回答：

> Query 表示当前特征希望查询什么信息，Key 用于与 Query 进行匹配，Value 是被关注位置真正提供并参与最终加权聚合的信息。

---

### Q4：为什么计算 QKᵀ？

回答：

> QKᵀ通过点积计算每一个 Query 与所有 Key 之间的匹配程度，从而得到后续 Attention Weight 的基础分数。

---

### Q5：为什么要除以 sqrt(d_k)？

回答：

> 当特征维度较大时，QKᵀ的数值尺度可能变大，使 Softmax 进入过度饱和区域，因此通过除以 sqrt(d_k) 控制分数尺度，使训练更加稳定。

---

### Q6：Softmax 为什么通常使用 dim=-1？

回答：

> 因为对于每一个 Query，需要在所有 Key 上计算一个归一化的注意力权重分布，所以应该沿 Key 对应的维度进行 Softmax。

---

### Q7：Attention 最终为什么还要乘 V？

回答：

> Q/K主要用于计算相关性和确定权重，而 V 才是实际被聚合的信息。因此需要使用 Attention Weight 对 V 进行加权求和，得到新的特征表示。

---

### Q8：Attention Matrix 的 shape 为什么是 N×N？

回答：

> N 个 Query 分别与 N 个 Key 进行两两匹配，因此会得到 N×N 的相关性矩阵。

---

### Q9：如果 Q、K、V 都是 N×D，最终输出为什么还是 N×D？

回答：

$$
QK^T:
(N\times D)(D\times N)=N\times N
$$

然后：

$$
(N\times N)(N\times D)=N\times D
$$

所以最终输出仍然是 N×D。

---

### Q10：Attention 为什么被称为动态的？

回答：

> Attention Weight 不是固定参数，而是根据当前输入特征计算 Q/K 的匹配程度，再经过 Softmax 动态得到。因此输入发生变化时，Attention Weight 也会发生变化。

---

# 13. Day3 核心知识结构

```text
Day2：CNN
│
├── 局部连接
├── 感受野
└── 长距离依赖
        │
        ↓
为什么不能直接根据相关性进行信息交互？
        │
        ↓
Day3：Attention
│
├── 动态信息聚合
│
├── Query
│
├── Key
│
├── Value
│
├── QKᵀ
│
├── Scaling
│
├── Softmax
│
└── Attention Weight × V
        │
        ↓
新的特征表示
        │
        ↓
Self-Attention
        │
        ↓
Multi-Head Attention
        │
        ↓
Transformer
```

---

# 14. 明天 Day4 学习前必须复习的内容

Day4 开始之前，不需要重新学习整个 Day1～Day3。

重点复习以下内容。

## 必须掌握 1：Attention 核心思想

能够自己解释：

> Attention 是根据特征之间的相关性动态分配权重，并对 Value 进行加权聚合。

---

## 必须掌握 2：Q/K/V

牢记：

```text
Q：Query，查询需求

K：Key，用于匹配

V：Value，真正提供给其他位置聚合的信息
```

---

## 必须掌握 3：完整公式

$$
\boxed{
Attention(Q,K,V)
=
Softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
}
$$

能够解释公式中的每一部分，而不是只背公式。

---

## 必须掌握 4：Tensor Shape

记住：

$$
Q,K,V\in R^{N\times D}
$$

那么：

$$
QK^T\in R^{N\times N}
$$

最终：

$$
Attention(Q,K,V)\in R^{N\times D}
$$

---

## 必须掌握 5：Softmax 维度

对于：

```python
scores.shape = [N, N]
```

通常：

```python
torch.softmax(scores, dim=-1)
```

表示：

> 对每一个 Query，在所有 Key 上进行归一化。

---

## 必须掌握 6：Attention 的动态性

能够解释今天的实验：

```text
Q1 原来与 K1 最匹配
        ↓
修改 Q1/K3
        ↓
Q1 与 K3 最匹配
        ↓
Attention Weight 改变
        ↓
Q1 从关注 K1 转向关注 K3
```

这说明 Attention 是：

> **输入依赖、动态计算的。**

---

# 15. Day3 是否达到进入 Day4 的标准？

## 结论：达到，Day3 通过 ✅

判断依据如下：

| Day3 目标                     | 完成情况 |
| --------------------------- | ---- |
| 理解 Attention 为什么出现          | ✅    |
| 理解 Attention 与 CNN 长距离依赖的关系 | ✅    |
| 理解动态信息聚合                    | ✅    |
| 理解 Q/K/V                    | ✅    |
| 理解 QKᵀ                      | ✅    |
| 理解 Scaling                  | ✅    |
| 理解 Softmax                  | ✅    |
| 理解 Softmax 的维度              | ✅    |
| 理解 Weighted Sum             | ✅    |
| 掌握 Tensor Shape             | ✅    |
| 独立手写简化 Attention            | ✅    |
| 完成动态 Attention 实验           | ✅    |
| 能解释实验结果                     | ✅    |

---

## 通过理由

Day3 并不是仅仅完成了：

```text
看公式
↓
记公式
↓
运行代码
```

而是完成了完整闭环：

```text
CNN 长距离依赖问题
        ↓
为什么需要 Attention
        ↓
动态相关性
        ↓
Q / K / V
        ↓
QKᵀ
        ↓
Scaling
        ↓
Softmax
        ↓
Attention Weight
        ↓
Weighted Sum
        ↓
新的特征表示
        ↓
PyTorch 手写实现
        ↓
修改输入
        ↓
观察 Attention Weight 动态变化
        ↓
理解 Attention 的动态性
```

尤其是最后的动态性实验，验证了：

> **Attention 并不是固定地关注某个位置，而是根据当前输入特征之间的关系动态决定关注对象。**

因此已经满足进入 Day4 的核心前置条件。

---

# 16. Day3 最终一句话总结

> **Attention 的核心不是简单地“看全局”，而是根据 Query 与 Key 的相关性动态计算注意力权重，再利用这些权重对 Value 进行信息聚合，从而让每个位置获得与当前任务更相关的其他位置的信息。**

---

# 17. Day4 学习入口

Day4 不需要重新从 Attention 开始。

下一步直接从：

$$
X
$$

开始回答一个核心问题：

> **如果 Q、K、V 都来自同一个输入 X，那么一个序列如何完成自己与自己的信息交互？**

由此进入：

```text
Self-Attention
    ↓
Multi-Head Attention
    ↓
多个 Head 为什么有意义？
    ↓
Concat
    ↓
Linear
    ↓
Transformer Encoder
```

Day4 的目标不是重新学习 Attention，而是：

> **在已经掌握单头 Attention 的基础上，理解 Transformer 是如何把 Attention 组织成完整模块的。**

```
```
