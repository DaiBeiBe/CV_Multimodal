# Day06 学习成果总结：Positional Encoding 与 Transformer 中的位置建模

## 一、今天真正掌握的知识

### 1. 为什么 Self-Attention 仍然需要 Positional Encoding

已经理解：

Self-Attention 能够建立 token 之间的全局关系，但自身没有天然的位置概念。

Self-Attention 的核心计算：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
Attention(X)= softmax\left(\frac{QK^T}{\sqrt d}\right)V
$$

如果输入中没有位置编码，那么模型主要根据 token 内容建立关系，无法天然区分：

```text
A B C
```

和

```text
C B A
```

中的位置差异。

因此需要加入 Positional Encoding：

$$
X'=X+P
$$

使 Transformer 的输入同时包含：

- token 内容信息：$X$
- 位置信息：$P$

### 2. 理解了 X + P 的含义

掌握：

$$
X' = X + P
$$

其中：

$$
X\in\mathbb{R}^{B\times N\times D}
$$

$$
P\in\mathbb{R}^{N\times D}
$$

最终：

$$
X'\in\mathbb{R}^{B\times N\times D}
$$

其中：

- $B$：batch size
- $N$：token 数量
- $D$：embedding / feature dimension

理解了为什么可以相加：

```text
X: [B, N, D]
P: [N, D]
```

PyTorch 会利用 broadcasting，将 $P$ 在 batch 维度上应用到每个样本。

更准确地说，可以理解为：

```text
P → [1, N, D]
X → [B, N, D]
```

然后进行广播相加。

### 3. 理解了 Absolute Positional Encoding

理解了：

Absolute Position 表示 token 在整个序列中的绝对位置。

例如：

```text
position 0
position 1
position 2
...
```

因此：

$$
P_0,\ P_1,\ P_2,\dots
$$

分别对应不同的位置。

同一个 token 出现在不同位置时：

$$
X_A+P_0
$$

和

$$
X_A+P_2
$$

会得到不同的表示。

因此模型可以区分：

```text
A B C
```

中的 A 和：

```text
C B A
```

中的 A。

### 4. 掌握了 Sinusoidal Positional Encoding

掌握了经典正弦位置编码公式：

$$
PE(pos,2i) = \sin\left( \frac{pos}{10000^{2i/d}} \right)
$$

$$
PE(pos,2i+1) = \cos\left( \frac{pos}{10000^{2i/d}} \right)
$$

理解了各参数含义：

- $pos$：token 的绝对位置
- $d$：位置编码维度
- $i$：频率维度索引
- $2i$：偶数维
- $2i+1$：奇数维

### 5. 理解了不同维度具有不同频率

今天重点理解了：

不同的 $i$ 对应不同的尺度。

可以写成：

$$
\lambda_i=10000^{2i/d}
$$

于是：

$$
PE(pos,2i)=\sin(pos/\lambda_i)
$$

随着 $i$ 增大，变化尺度增大，因此对应的变化更加平缓。

最终不同维度能够从不同尺度描述位置：

- 部分维度 → 更快速的变化
- 部分维度 → 更缓慢的变化

共同形成一个位置向量。

### 6. 理解了 Sinusoidal PE 的代码实现

最终正确实现：

```python
import torch
import math

N = 16
D = 64

position = torch.arange(N).unsqueeze(-1)

div_term = torch.exp(
    torch.arange(0, D, 2)
    * (-torch.log(torch.tensor(10000.0)) / D)
)

pe = torch.zeros(N, D)

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

print(pe.shape)
```

得到：

```text
torch.Size([16, 64])
```

理解：

- 16 → 16个位置
- 64 → 每个位置的64维位置编码

同时观察到：

```text
pe[0] = [0, 1, 0, 1, ...]
```

原因：

$$
\sin(0)=0
$$

$$
\cos(0)=1
$$

### 7. 通过代码验证了不同频率

观察：

```python
pe[:, 0]
```

和：

```python
pe[:, 2]
```

发现二者随位置变化的速度不同。

因此实际验证了：

不同维度使用不同频率对位置进行编码。

### 8. 理解了 PE 的 Tensor Shape

掌握：

$$
PE\in[N,D]
$$

其中：

- 行：position
- 列：dimension

因此：

```python
plt.imshow(pe)
```

中：

- 纵轴 → N / position
- 横轴 → D / dimension

这一点今天出现过一次理解方向错误，但已经纠正。

### 9. 理解了没有 PE 时 ABC 与 CBA 的实验现象

通过 Self-Attention 实验验证：

```text
ABC
```

和：

```text
CBA
```

在没有 PE 时，Attention Scores 相同：

```text
[[0.5, 0,   0],
 [0,   0.5, 0],
 [0,   0,   0.5]]
```

Attention Weights 也相同。

但是最终 Output 只是按照 token 顺序发生了相应的排列变化。

这说明：

Self-Attention 本身没有显式的位置概念。

同时理解了这里不能错误地说“模型完全不受顺序影响”，更准确的理解是：没有位置编码时，Self-Attention 对输入 token 的排列具有相应的置换性质。

### 10. 通过 X + P 验证了位置会进入 Attention

加入手工设计的：

```text
P =
[[0.1, 0.2, 0.3, 0.4],
 [0.5, 0.6, 0.7, 0.8],
 [0.9, 1.0, 1.1, 1.2]]
```

之后：

$$
X'=X+P
$$

再进行 Self-Attention。

观察到 ABC 与 CBA 的 Attention Scores 不再相同。

理解了原因：

$$
Q=X'W_Q,\quad K=X'W_K
$$

因此：

$$
QK^T
$$

已经受到位置编码影响。

在简化 $W_Q=W_K=I$ 时：

$$
QK^T=(X+P)(X+P)^T
$$

展开：

$$
XX^T+XP^T+PX^T+PP^T
$$

因此 Attention 可以同时受到：

- token 内容
- position

的影响。

### 11. 理解了“PE 告诉模型 token 在哪里，而不是 token 是什么”

今天最后的核心认识：

$$
X=\text{token content}
$$

$$
P=\text{token position}
$$

因此：

Positional Encoding 不是用来描述 token 本身是什么，而是描述 token 在序列中的位置。

例如同一个 token A：

$$
X_A+P_0
$$

和：

$$
X_A+P_2
$$

内容相同，但位置不同，因此最终表示不同。

### 12. 初步理解视觉 Transformer 中的位置问题

理解了文本与图像位置建模的区别。

文本：

```text
token 0 → token 1 → token 2 → ...
```

主要是一维序列位置。

图像：

```text
P00 P01 P02
P10 P11 P12
P20 P21 P22
```

本质上具有二维空间结构：

- 行方向
- 列方向

将图像 patch 展平为：

```text
P00 P01 P02 P10 P11 P12 ...
```

一维位置编码可以唯一标识 patch 在序列中的位置，但不能直接表达完整的二维空间关系。

例如：

```text
P00 → P01
```

和：

```text
P00 → P10
```

虽然都可以通过一维位置表示出来，但分别对应：

- 右邻居
- 下邻居

这种二维空间关系。

## 二、回答中暴露出的薄弱点

### 1. 对“不同频率”的最初理解存在偏差

你最初认为：

不同频率可能是动态训练出来的。

这个理解不准确。

今天已经明确：

经典 Sinusoidal Positional Encoding 是固定构造的，并不是通过训练动态学习频率。

需要区分：

```text
Sinusoidal PE
→ 固定计算

Learnable Positional Embedding
→ 参数可以训练
```

### 2. 初始代码中出现了 div_term 使用方向错误

最开始写成：

```python
torch.sin(position / div_term)
```

但：

```text
div_term
=
exp(-2i/D * ln10000)
=
10000^(-2i/D)
```

因此正确实现应该是：

```python
torch.sin(position * div_term)
```

这个问题已经通过代码检查和实际运行修正。

目前不构成 Day7 阻碍，但后续复习公式时需要再次注意。

### 3. 对 imshow(pe) 的轴方向一开始理解反了

你第一次回答：

横轴代表 N，纵轴代表 D。

实际应该是：

```text
pe.shape = [N, D]
```

纵轴 → N / position
横轴 → D / dimension

原因是矩阵：

```text
pe[row, column]
```

对应：

```text
pe[position, dimension]
```

这个问题已经纠正。

### 4. 对二维位置的理解最初略显绝对

你最初认为：

1D spatial information 也可以表达 2D spatial information，只是复杂一些。

后续已经修正为更加准确的理解：

一维位置可以唯一标识二维网格中的 patch，但不能显式表达二维坐标和方向关系。

这是目前需要继续保持的准确表述。

### 5. 对“为什么使用 sin/cos”的理解曾经过度集中于数值尺度

你曾经认为：

如果直接使用 position number，P 可能太大，从而压制 X。

这个观点可以作为一个尺度方面的考虑，但不是今天 sinusoidal PE 的核心解释。

今天已经进一步理解：

Sin/Cos 的重要作用在于通过不同频率的多个维度构造结构化的位置表示。

后续学习更深入的位置编码方法时，需要继续区分：

- 为什么需要位置；
- 为什么采用某一种位置表示；
- 不同位置编码方案之间解决的问题有什么区别。

## 三、今天完成的 Python 代码 / 实验

### 实验1：Sinusoidal Positional Encoding

完成：

```python
N = 16
D = 64
```

生成：

$$
PE\in[16,64]
$$

验证：

```python
pe.shape = [16,64]
```

并观察：

```python
pe[0]
pe[:,0]
pe[:,2]
```

验证不同维度具有不同变化频率。

### 实验2：ABC / CBA 无 PE 对比实验

构造：

```python
A = [1,0,0,0]
B = [0,1,0,0]
C = [0,0,1,0]
```

构造：

```text
ABC
CBA
```

然后输入 Self-Attention。

验证：

```text
无 PE
→ Attention Scores 相同
→ Attention Weights 相同
→ 输出只是 token 顺序发生对应变化
```

### 实验3：加入手工 Position Encoding

构造：

```text
P =
[[0.1,0.2,0.3,0.4],
 [0.5,0.6,0.7,0.8],
 [0.9,1.0,1.1,1.2]]
```

计算：

$$
X'=X+P
$$

分别得到：

```text
ABC + PE
CBA + PE
```

验证同一个 token 在不同位置会得到不同表示。

### 实验4：加入 PE 后重新计算 Self-Attention

将：

$$
X+P
$$

输入 Self-Attention。

观察到：

```text
ABC scores
≠
CBA scores
```

从而实验验证：

PE 进入输入之后，会影响 Q/K，从而影响 Attention Score。

### 实验5：位置编码可视化理解

分析：

```python
plt.imshow(pe)
```

理解：

- 纵轴 → position
- 横轴 → dimension

并理解 pe.T 后：

```text
[N,D]
→
[D,N]
```

即交换 position 与 dimension 两个轴。

## 四、还存在的问题

### 1. Sinusoidal PE 需要进一步熟练到“看到公式就能解释”

目前已经理解公式，但还需要做到：

$$
PE(pos,2i) = \sin(pos/10000^{2i/d})
$$

$$
PE(pos,2i+1) = \cos(pos/10000^{2i/d})
$$

看到后能够立即解释：

- 为什么偶数维使用 sin；
- 为什么奇数维使用 cos；
- $i$ 在控制什么；
- 不同维度为什么有不同频率；
- 为什么随着维度增加变化更慢。

### 2. 还需要进一步区分不同 Positional Encoding 方法

Day6 已经建立了基础，但后续仍需要逐渐区分：

- Sinusoidal PE
- Learnable PE
- 其他位置建模方法

尤其需要理解它们的：

- 来源
- 是否可学习
- 优点
- 局限
- 使用场景

### 3. 二维视觉位置编码目前属于“理解层面”

已经理解：

```text
1D position
vs
2D spatial position
```

但还没有深入到具体视觉 Transformer 中不同二维位置编码的代码实现。

这属于后续 ViT / DETR / Vision Transformer 学习阶段继续深入的内容。

### 4. Transformer 完整数据流还需要再串一次

虽然今天已经理解：

```text
X + P
→ Q/K/V
→ Attention
```

但 Day7 开始前还应该能够完整说出：

```text
Input
→ Token/Embedding
→ Positional Encoding
→ Multi-Head Self-Attention
→ Residual
→ LayerNorm
→ FFN
→ Residual
→ LayerNorm
```

以及每一个模块为什么存在。

## 五、复试可能被问到的问题

### 基础问题

**为什么 Transformer 需要 Positional Encoding？**

核心回答：

Self-Attention 可以建立 token 之间的全局关系，但本身没有显式的位置概念，因此需要 Positional Encoding 为 token 提供位置信息。

**Positional Encoding 和 token embedding 有什么区别？**

Token embedding 表示 token 的内容，Positional Encoding 表示 token 的位置。

**为什么可以直接把 X 和 P 相加？**

因为它们在 token 数量和特征维度上对应，X 为 `[B,N,D]`，P 为 `[N,D]`，通过 broadcasting 可以扩展到 batch 维度。

**Sinusoidal PE 的公式是什么？**

$$
PE(pos,2i)= \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

$$
PE(pos,2i+1)= \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

**为什么不同维度使用不同频率？**

让不同维度从不同尺度描述位置信息，一部分维度变化较快，一部分维度变化较慢，共同形成位置表示。

### 深入问题

**没有 PE 时，ABC 和 CBA 有什么区别？**

需要注意：

Attention 本身没有显式位置概念，因此不能从输入中获得“第几个 token”这种位置信息；对于相应的 token 排列，Self-Attention 会表现出相应的置换性质。

**PE 是如何影响 Attention 的？**

$$
X'=X+P
$$

然后：

$$
Q=X'W_Q,\quad K=X'W_K
$$

因此：

$$
QK^T
$$

会同时受到内容和位置表示的影响。

**为什么图像中的位置编码比文本更复杂？**

文本主要是一维序列，而图像 patch 具有二维空间结构，需要考虑行和列方向上的空间关系。

**把图像 patch 展平后使用 1D PE 可以吗？**

可以唯一标识 patch 在展平序列中的位置，但不能直接显式表达完整的二维空间关系。

**为什么不能简单说“Attention 可以自动理解位置”？**

因为标准 Self-Attention 的计算主要来自 token 内容之间的相似性，没有显式的位置输入；位置需要通过额外的位置建模机制提供。

## 六、Day7 学习前必须复习的内容

### 第一优先级：必须掌握

#### 1. Self-Attention 的完整计算链

必须能够脱离资料解释：

$$
X \rightarrow Q,K,V \rightarrow QK^T \rightarrow \frac{QK^T}{\sqrt d} \rightarrow softmax \rightarrow Attention\ Weight \rightarrow Output
$$

并知道每一步解决什么问题。

#### 2. Transformer 为什么需要位置编码

必须能够自己完整解释：

```text
SA
↓
建立 token 间关系
↓
缺少位置
↓
加入 P
↓
X' = X + P
↓
重新计算 Q/K/V
↓
Attention 同时受到内容和位置影响
```

#### 3. [B,N,D] 和 [N,D]

必须熟练：

```text
X = [B,N,D]
P = [N,D]
X + P = [B,N,D]
```

并理解 broadcasting。

#### 4. Sinusoidal PE 公式

至少要做到：

- 偶数维 → sin
- 奇数维 → cos
- 不同 i → 不同频率

并理解代码中的：

```python
div_term
```

为什么最终使用：

```python
position * div_term
```

#### 5. 重新理解 ABC / CBA 实验

必须能回答：

为什么没有 PE 时两个序列的 Attention Score 表现出相同的结构？

以及：

加入 PE 后为什么 Attention Score 发生变化？

### 第二优先级：理解即可

复习：

```text
1D position
vs
2D visual position
```

重点记住：

唯一标识 patch ≠ 显式表达二维空间关系。

## 七、Day6 是否达到进入 Day7 的标准？

结论：达到，可以进入 Day7。 ✅

而且不是“勉强达到”，而是核心知识已经达到进入下一阶段的要求。

### 理由1：能够自己解释“为什么需要 PE”

你没有停留在：

“因为 Transformer 需要位置编码。”

而是能够解释：

Self-Attention 可以建立 token 间关系，但不能提供位置信息，因此需要通过 $X+P$ 把位置信息加入 token 表示。

这说明已经理解因果关系。

### 理由2：能够完成数学到代码的转换

你今天不仅理解：

$$
X'=X+P
$$

还亲自完成了：

```text
Sinusoidal PE
→ Tensor shape
→ Attention
→ 加 PE
→ 再计算 Attention
```

说明已经开始形成：

```text
数学公式 → Tensor → PyTorch 实现 → 实验现象
```

这一科研/工程能力链条。

### 理由3：能够用实验验证理论

尤其是：

```text
ABC vs CBA
```

实验非常关键。

你不是简单接受：

“没有 PE，Transformer 不知道位置。”

而是实际观察：

```text
无 PE
→ scores 相同结构

加入 PE
→ scores 发生变化
```

这已经符合你项目要求的：

```text
理解代码 → 修改代码 → 观察实验 → 分析原因
```

### 理由4：已经能够区分文本位置和视觉位置

你最后能够回答：

一维位置可以唯一标识 patch，但不能直接表达二维空间关系。

这说明 Day6 的最后一个视觉位置目标已经达到。

## Day6 最终评价

| 能力 | 状态 |
| --- | --- |
| 理解为什么需要 PE | ✅ 掌握 |
| 理解 X + P | ✅ 掌握 |
| Tensor Shape / Broadcasting | ✅ 掌握 |
| Absolute Position | ✅ 掌握 |
| Sinusoidal PE | ✅ 基本掌握 |
| 不同频率的意义 | ✅ 掌握 |
| PE PyTorch 实现 | ✅ 已完成 |
| ABC/CBA 无 PE 实验 | ✅ 已完成 |
| 加 PE 后 Attention 实验 | ✅ 已完成 |
| 1D vs 2D Visual Position | ✅ 掌握 |
| 数学 → 代码 → 实验联系 | ✅ 已建立 |

### 最终判断

Day6 达标，可以进入 Day7。

Day7 不需要重新从 Attention 或 Positional Encoding 开始。下一阶段应该把：

```text
CNN
→ Attention
→ Self-Attention
→ MHA
→ PE
→ Transformer Block
```

真正串成一个完整的 Transformer 数据流和模型结构，再逐步进入视觉 Transformer / ViT 等视觉场景。