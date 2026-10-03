import math
import torch

B = 2
N = 4
D = 8
H = 2
d_head = D // H

X = torch.randn(B, N, D)

# 1. 创建 Wq, Wk, Wv
Wq = torch.randn(D, D)
Wk = torch.randn(D, D)
Wv = torch.randn(D, D)

# 2. 从 X 得到 Q/K/V
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# 3. reshape + transpose
Q = Q.reshape(B, N, H, d_head).transpose(-3, -2)
K = K.reshape(B, N, H, d_head).transpose(-3, -2)
V = V.reshape(B, N, H, d_head).transpose(-3, -2)

# 4. attention
scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(Q.size(-1))
attention_weights = torch.softmax(scores, dim=-1)

# 5. attention @ V
head_output = attention_weights @ V

# 6. transpose + reshape 回 [B,N,D]
output = head_output.transpose(-3, -2).reshape(B, N, D)

# 7. 打印每一步 shape
print("X shape:", X.shape)
print("Wq shape:", Wq.shape)
print("Wk shape:", Wk.shape)
print("Wv shape:", Wv.shape)
print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)
print("scores shape:", scores.shape)
print("attention_weights shape:", attention_weights.shape)
print("head_output shape:", head_output.shape)
print("output shape:", output.shape)


Wo = torch.randn(D, D)
final_output = output @ Wo
print("final_output shape:", final_output.shape)
