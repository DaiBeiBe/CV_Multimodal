import torch
import math

B = 2
N = 4
D = 8

X = torch.randn(B, N, D)

# 你的代码：
# 1. 创建 Wq, Wk, Wv
Wq = torch.randn(D, D)
Wk = torch.randn(D, D)
Wv = torch.randn(D, D)

# 2. 得到 Q, K, V
Q = X @ Wq
K = X @ Wk
V = X @ Wv
# 3. 计算 attention score
attention_score = Q @ K.transpose(-2, -1)
# 4. 除以 sqrt(D)
scores = attention_score / math.sqrt(D)
# 5. softmax
attention_weights = torch.softmax(scores, dim=-1)
# 6. 与 V 相乘
output = attention_weights @ V
# 7. 打印每一步 shape
print("X shape:", X.shape)
print("Wq shape:", Wq.shape)
print("Wk shape:", Wk.shape)
print("Wv shape:", Wv.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("attention_score shape:", attention_score.shape)
print("scores shape:", scores.shape)
print("attention_weights shape:", attention_weights.shape)
print("output shape:", output.shape)

print(attention_weights[0])
print(attention_weights[0].sum(dim=-1))

print("attention_weights[0, 0]:", attention_weights[0, 0])
print("sum:", attention_weights[0, 0].sum())

manual_output = (
    attention_weights[0, 0, 0] * V[0, 0]
    + attention_weights[0, 0, 1] * V[0, 1]
    + attention_weights[0, 0, 2] * V[0, 2]
    + attention_weights[0, 0, 3] * V[0, 3]
)

print("manual:", manual_output)
print("torch :", output[0, 0])
