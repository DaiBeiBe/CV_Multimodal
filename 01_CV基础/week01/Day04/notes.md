# Day4｜Self-Attention 与 Multi-Head Attention

> 学习主题：Self-Attention 与 Multi-Head Attention
> 学习路线：Attention → Self-Attention → Multi-Head Attention → 手写实现 → PyTorch 实现对照
> Day4 状态：**✅ 已完成，达到进入 Day5 的标准**

---

## 1. 今天真正掌握的知识

### 1.1 Attention 与 Self-Attention

已经理解二者的核心区别：

* 一般 Attention 中，Q、K、V 可以来自不同的信息源；
* Self-Attention 中，Q、K、V 都来源于同一个输入序列 X；
* 但 Q、K、V 并不是直接等于 X，而是经过不同的线性映射得到：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

“Self”强调的是：

> **信息交互发生在同一个输入序列内部。**

而不是简单理解成：

> Q = K = V = X

---

### 1.2 Q / K / V 的作用

已经能够从功能角度理解三者：

* **Query：我想找什么？**
* **Key：我有什么可以被匹配？**
* **Value：匹配之后我真正提供什么信息？**

因此：

* Q 和 K 决定“应该关注谁”；
* V 决定“从被关注对象那里获取什么信息”。

也理解了为什么不能简单令：

$$
Q=K=V=X
$$

因为这样会失去 Q/K/V 三种不同角色的独立可学习映射能力。

---

### 1.3 Self-Attention 完整计算过程

已经掌握 Self-Attention 的完整计算链：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

$$
A=\operatorname{softmax}(S)
$$

$$
O=AV
$$

对于：

$$
X\in R^{B\times N\times D}
$$

有：

$$
Q\in R^{B\times N\times d_k}
$$

$$
K\in R^{B\times N\times d_k}
$$

$$
V\in R^{B\times N\times d_v}
$$

$$
QK^T\in R^{B\times N\times N}
$$

$$
O\in R^{B\times N\times d_v}
$$

---

### 1.4 `[N, N]` Attention Matrix

已经理解：

$$
QK^T
$$

得到 `[B, N, N]`，而不是 `[B, N, D]`。

原因：

$$
[B,N,d_k]\times[B,d_k,N]
=
[B,N,N]
$$

其中：

* 第一个 N：Query token；
* 第二个 N：Key token。

因此：

$$
A_{ij}
$$

表示：

> 第 i 个 Query 对第 j 个 Key 的注意力权重。

需要注意：

> 不能严格地把 \(q_i^Tk_j\) 称为原始 \(x_i\) 与 \(x_j\) 的相关性。

更准确的说法是：

> **Q 表示与 K 表示之间的匹配程度。**

---

### 1.5 Softmax 的维度

理解了为什么使用：

```python
torch.softmax(scores, dim=-1)
```

因为 Attention Score：

$$
S\in R^{B\times N\times N}
$$

最后一个维度对应 Key token。

对于每一个 Query，需要在所有 Key 上形成一个概率分布：

$$
\sum_j A_{ij}=1
$$

因此应该沿最后一个维度进行 softmax，而不是对整个 `[N,N]` 矩阵统一归一化。

---

### 1.6 Scaled Dot-Product Attention

理解了为什么需要：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

核心原因：

> 随着 \(d_k\) 增大，点积的数值尺度可能变大，使 softmax 输入的 logits 过大，从而导致 softmax 饱和、梯度变小、训练不稳定。

因此：

$$
\sqrt{d_k}
$$

用于控制点积结果的尺度。

---

### 1.7 Multi-Head Attention

已经理解 Multi-Head Attention 不是简单地：

> “把 Attention 重复很多遍”。

而是：

> **不同 Head 使用不同的可学习 Q/K/V 映射，在不同表示子空间中学习不同的信息交互模式。**

对于：

$$
D=512,\quad H=8
$$

有：

$$
d_{head}=\frac{D}{H}=64
$$

不同 Head 可以学习不同的信息关系，最终再进行融合。

---

### 1.8 Multi-Head Attention Tensor Shape

已经能够独立推导完整 Shape 流程。

对于：

$$
B=2,\quad N=16,\quad D=512,\quad H=8
$$

有：

$$
d_{head}=64
$$

Projection 后：

$$
Q,K,V:[2,16,512]
$$

Split Heads：

$$
[2,16,8,64]
$$

Transpose：

$$
[2,8,16,64]
$$

计算 Attention Score：

$$
QK^T:[2,8,16,16]
$$

Attention × V：

$$
[2,8,16,64]
$$

Merge Heads：

$$
[2,16,512]
$$

这一部分已经掌握较为扎实。

---

### 1.9 为什么需要 Output Projection \(W_O\)

理解了：

> Concat 只是把不同 Head 的输出放到一起，而 \(W_O\) 负责进一步学习不同 Head 信息之间如何组合。

即：

$$
\text{Concat}(head_1,\dots,head_H)W_O
$$

因此：

* Concat：组织不同 Head 的信息；
* \(W_O\)：重新混合、融合不同 Head 的信息。

已经理解 \(W_O\) 并不是单纯为了改变维度，而具有**跨 Head 信息融合**的作用。

---

## 2. 今天回答中暴露出的薄弱点

### 2.1 曾经把 \(q_i^Tk_j\) 简化成 \(x_i,x_j\) 的相关性

早期回答中出现过类似：

> “QKᵀ可以得到 x1 和 x2、x3 的相关性”

这个说法方向没有错，但不够严谨。

应该改成：

> \(q_i^Tk_j\) 表示第 i 个 Query 表示和第 j 个 Key 表示之间的匹配程度。

因为：

$$
q_i=x_iW_Q
$$

$$
k_j=x_jW_K
$$

所以实际比较的是经过不同投影后的表示。

---

### 2.2 第 3 题完整公式链没有全部写出

能够正确写出：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

以及各自 Shape，但自测时没有完整写出：

$$
QK^T
\rightarrow
\frac{QK^T}{\sqrt{d_k}}
\rightarrow
softmax
\rightarrow
AV
$$

后续需要训练自己做到：

> **看到 Self-Attention，能够直接从输入写到输出。**

---

### 2.3 `softmax(dim=-1)` 的理解还需要更加严格

已经知道“对每个 token 相对于其他 token 的权重归一化”，但还需要进一步熟练使用：

> `[B,N,N]` 中，倒数第二维是 Query，最后一维是 Key，因此对最后一个 N 做 softmax。

这是以后阅读 Transformer 源码时非常重要的细节。

---

### 2.4 MHA 实验中曾出现代码级错误

第一次手写 MHA 时出现过：

* Q/K/V 直接使用随机 Tensor，而不是从 X 经过投影得到；
* 曾经错误地直接对 Attention Weight 进行 reshape；
* `Wv` / `Wq` 曾出现变量使用错误；
* V 曾错误使用 `Wq`。

这些问题最终都被自己发现并修正。

这说明目前：

> **原理已经基本掌握，但代码熟练度还需要继续提高。**

尤其要加强：

```text
reshape
transpose
matmul
head_output
concat
output projection
```

之间的顺序意识。

---

### 2.5 第 10 题没有完整写出

自测第 10 题选择在脑中过了一遍，没有完整展开。

虽然结合之前的代码实验可以判断已经掌握，但以后面对导师提问时，最好能够完整口述：

$$
CNN Feature
\rightarrow
Token
\rightarrow
Q/K/V
\rightarrow
Split Heads
\rightarrow
QK^T
\rightarrow
Softmax
\rightarrow
Attention\times V
\rightarrow
Concat
\rightarrow
W_O
$$

尤其需要能够解释：

$$
[196,196]
$$

表示：

> 196 个视觉 Token 之间的两两注意力关系。

---

## 3. 今天完成的 Python 代码 / 实验

### 3.1 实验一：手写 Self-Attention

使用 PyTorch 从零实现了 Self-Attention，没有调用 `nn.MultiheadAttention`。

核心流程：

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv

attention_score = Q @ K.transpose(-2, -1)

scores = attention_score / math.sqrt(D)

attention_weights = torch.softmax(scores, dim=-1)

output = attention_weights @ V
```

实际验证：

```text
X                  [2,4,8]
Wq/Wk/Wv           [8,8]
Q/K/V              [2,4,8]
attention_score    [2,4,4]
scores             [2,4,4]
attention_weights  [2,4,4]
output             [2,4,8]
```

并进一步检查了：

$$
output[0,0]
=
\sum_j attention\_weights[0,0,j]V[0,j]
$$

真正理解了 Attention 的加权聚合过程。

---

### 3.2 实验二：手写 Multi-Head Attention

从 Self-Attention 进一步实现了 MHA：

```python
Q = X @ Wq
K = X @ Wk
V = X @ Wv

Q = Q.reshape(B,N,H,d_head).transpose(-3,-2)
K = K.reshape(B,N,H,d_head).transpose(-3,-2)
V = V.reshape(B,N,H,d_head).transpose(-3,-2)

scores = Q @ K.transpose(-2,-1)

scores = scores / math.sqrt(Q.size(-1))

attention_weights = torch.softmax(scores, dim=-1)

head_output = attention_weights @ V

output = head_output.transpose(-3,-2).reshape(B,N,D)
```

实际 Shape：

```text
X              [2,4,8]
Q/K/V          [2,2,4,4]
scores         [2,2,4,4]
weights        [2,2,4,4]
head_output    [2,2,4,4]
output         [2,4,8]
```

---

### 3.3 实验三：PyTorch `MultiheadAttention`

使用：

```python
mha = torch.nn.MultiheadAttention(
    embed_dim=D,
    num_heads=H,
    batch_first=True
)

attn_output, attn_weights = mha(X,X,X)
```

理解了默认：

```text
attn_weights: [B,N,N]
```

是多个 Head 平均后的 Attention Weight。

设置：

```python
average_attn_weights=False
```

后得到：

```text
[B,H,N,N]
```

即每个 Head 独立的 Attention Weight。

---

### 3.4 实验四：拆解 PyTorch MHA 内部参数

进一步检查了：

```python
mha.in_proj_weight
mha.in_proj_bias
mha.out_proj.weight
mha.out_proj.bias
```

理解了：

```text
in_proj_weight
        ↓
[WQ
 WK
 WV]
```

其中：

$$
in\_proj\_weight\in R^{3D\times D}
$$

对于：

$$
D=8
$$

其 Shape 为：

```text
[24,8]
```

并成功拆分：

```python
Wq = mha.in_proj_weight[:D]
Wk = mha.in_proj_weight[D:2*D]
Wv = mha.in_proj_weight[2*D:]
```

同时理解了 PyTorch Linear 中：

$$
Y=XW^T+b
$$

因此手写实现使用：

```python
Q = X @ Wq.T + bq
```

---

### 3.5 实验五：手写 MHA 与 PyTorch 官方实现数值对齐

最终将手写实现加入：

$$
W_O
$$

以及 bias 后，与：

```python
mha(X,X,X)
```

进行数值比较。

结果：

```text
True
tensor(5.9605e-08)
```

说明：

> **手写 Multi-Head Attention 与 PyTorch `nn.MultiheadAttention` 在浮点误差范围内完全一致。**

这是 Day4 最重要的实验结果之一。

---

## 4. 还存在的问题

### 理论方面

1. 需要进一步熟练完整 Self-Attention 数学链；
2. 需要更加严格地区分：

   * Query；
   * Key；
   * Value；
   * 原始输入 X；
3. 需要继续巩固 `softmax(dim=-1)` 的维度意义；
4. 需要进一步理解 \(d_k\) 增大为什么会导致点积方差增大；
5. 需要继续熟悉 `[B,H,N,N]` 的实际含义。

### 代码方面

目前已经能够手写 MHA，但还不是完全“肌肉记忆”。

后续需要加强：

```text
reshape
→ transpose
→ QKᵀ
→ softmax
→ Attention × V
→ transpose
→ reshape
→ WO
```

尤其要避免：

* 维度写错；
* transpose 维度写错；
* WQ/WK/WV 混用；
* 在还没有 `Attention × V` 的情况下直接 reshape；
* 忘记 Output Projection。

### 知识边界

Day4 暂时**没有系统学习**：

* Residual Connection；
* LayerNorm；
* FFN；
* Transformer Encoder Block；
* Transformer Decoder；
* Positional Encoding；
* ViT；
* DETR。

这些不属于 Day4 的核心任务，不需要现在补齐。

---

## 5. 复试可能被问到的问题

### 基础问题

1. 什么是 Self-Attention？
2. Self-Attention 和普通 Attention 有什么区别？
3. 为什么需要 Q、K、V？
4. 为什么不能直接令 Q=K=V=X？
5. Q、K、V 分别代表什么？
6. 为什么 QKᵀ 得到的是 `[N,N]`？
7. `[N,N]` Attention Matrix 中每个元素代表什么？
8. 为什么 Softmax 要在最后一个维度进行？
9. 为什么要除以 \(\sqrt{d_k}\)？
10. 如果不进行 Scaling 会发生什么？

### MHA 问题

11. 为什么需要 Multi-Head Attention？
12. Multi-Head Attention 相比单头 Attention 有什么优势？
13. 为什么不同 Head 可以学习不同的信息？
14. 为什么：

$$
d_{head}=\frac{D}{H}
$$

15. 为什么 Attention Score 的 Shape 是：

$$
[B,H,N,N]
$$

16. 为什么最后需要 Concatenate？
17. Concatenate 后为什么还需要 \(W_O\)？
18. 如果去掉 \(W_O\)，模型还能不能运行？
19. 去掉 \(W_O\) 后会损失什么？
20. MHA 和多个独立 Attention 简单并列有什么区别？

### CNN → Attention 问题

21. CNN Feature Map 为什么可以转换成 Token？
22. `[B,196,512]` 中 196 是什么意思？
23. 如果 14×14 Feature Map Flatten 后得到 196 个 Token，每个 Token 表示什么？
24. `[196,196]` Attention Matrix 表示什么？
25. CNN 与 Self-Attention 在信息交互机制上有什么区别？
26. 为什么 Self-Attention 能够建立远距离 Token 之间的直接信息交互？
27. 为什么这对于视觉任务有意义？

### 源码问题

28. PyTorch `MultiheadAttention` 中 `in_proj_weight` 为什么是 `[3D,D]`？
29. 为什么手写代码中需要 `Wq.T`？
30. `average_attn_weights=False` 有什么作用？
31. `reshape` 和 `transpose` 在 MHA 中分别起什么作用？
32. 为什么不能直接把 Attention Weight reshape 成最终输出？

---

## 6. Day5 学习前必须复习的内容

Day5 开始前，不需要重新学习全部 Attention。

重点复习下面 6 项。

### ① Self-Attention 完整公式

必须能够脱离资料写出：

$$
Q=XW_Q
$$

$$
K=XW_K
$$

$$
V=XW_V
$$

$$
A=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)
$$

$$
O=AV
$$

---

### ② Tensor Shape

必须熟练：

$$
X:[B,N,D]
$$

$$
Q,K:[B,N,d_k]
$$

$$
V:[B,N,d_v]
$$

$$
QK^T:[B,N,N]
$$

$$
O:[B,N,d_v]
$$

---

### ③ MHA Shape

重点记忆逻辑，而不是死记数字：

$$
[B,N,D]
$$

→

$$
[B,N,H,d_{head}]
$$

→

$$
[B,H,N,d_{head}]
$$

→

$$
[B,H,N,N]
$$

→

$$
[B,H,N,d_{head}]
$$

→

$$
[B,N,H,d_{head}]
$$

→

$$
[B,N,D]
$$

其中：

$$
d_{head}=\frac DH
$$

---

### ④ Attention Matrix 的含义

重点理解：

$$
A_{ij}
$$

表示：

> Query token \(i\) 对 Key token \(j\) 的注意力权重。

不要再简单称为“两个原始 token 的相关性”。

---

### ⑤ \(W_O\) 的作用

记住：

> Concat 是把 Head 的结果放到一起；\(W_O\) 是学习如何重新组合这些 Head 的信息。

---

### ⑥ 重新看一遍自己的 MHA 代码

尤其复习：

```python
reshape
transpose
Q @ K.transpose(-2,-1)
softmax(dim=-1)
attention_weights @ V
transpose
reshape
W_O
```

不要只看代码，要能够解释每一行：

> **为什么要这样写？Shape 为什么变成这样？如果删除这一行会发生什么？**

---

## 7. Day4 是否达到进入 Day5 的标准？

# ✅ 达到

判断依据如下。

### 标准 1：概念理解

已经能够独立解释：

* Attention；
* Self-Attention；
* Q/K/V；
* Scaled Dot-Product Attention；
* Multi-Head Attention。

**通过。**

---

### 标准 2：数学理解

能够理解：

$$
QK^T
$$

为什么得到：

$$
[N,N]
$$

并理解：

$$
\frac{QK^T}{\sqrt{d_k}}
$$

为什么存在。

**通过。**

---

### 标准 3：Tensor Shape

能够独立完成：

$$
[B,N,D]
\rightarrow
[B,H,N,d_{head}]
\rightarrow
[B,H,N,N]
\rightarrow
[B,N,D]
$$

并在具体数字下正确计算。

**通过，而且掌握较扎实。**

---

### 标准 4：代码实现

已经独立完成：

* Self-Attention；
* Multi-Head Attention；
* PyTorch MHA；
* PyTorch 参数拆解；
* 手写实现与官方实现对齐。

**通过。**

---

### 标准 5：实验验证

不是停留在：

> “我的代码能运行。”

而是进一步完成：

$$
\boxed{
\text{手写实现}
\approx
\text{PyTorch 官方实现}
}
$$

最终误差：

$$
5.96\times10^{-8}
$$

属于浮点计算误差范围。

**通过。**

---

### 标准 6：科研学习能力

今天的学习已经开始形成：

> **理论 → Shape → 代码 → 官方实现 → 数值验证**

这正是以后进行论文复现时需要具备的基本能力。

---

# Day4 最终评价

**Day4：✅ 完成**

当前能力状态：

```text
Attention
    ↓
Self-Attention
    ↓
Q / K / V
    ↓
Scaled Dot-Product
    ↓
Tensor Shape
    ↓
Multi-Head Attention
    ↓
手写 MHA
    ↓
PyTorch MHA
    ↓
参数映射
    ↓
数值验证
```

这一整条链已经基本打通。

因此没有必要继续在 Day4 上堆更多 Attention 细节，可以进入 Day5。

---

## Day5 的核心问题

Day4 解决的是：

> **“Token 之间如何进行信息交互？”**

答案：

$$
\boxed{Multi\text{-}Head\ Attention}
$$

Day5 要进一步解决：

> **“完成信息交互以后，一个 Transformer Block 为什么还需要 Residual、LayerNorm 和 FFN？”**

重点进入：

$$
\boxed{
MHA
\rightarrow
Residual
\rightarrow
LayerNorm
\rightarrow
FFN
\rightarrow
Transformer\ Block
}
$$

最终目标不是背住 Transformer Block 的结构，而是理解：

> **为什么只有 Attention 还不够？为什么还需要 FFN？为什么需要残差连接？为什么需要 LayerNorm？**

这将是从“理解 Attention”真正进入“理解 Transformer”的关键一步。
