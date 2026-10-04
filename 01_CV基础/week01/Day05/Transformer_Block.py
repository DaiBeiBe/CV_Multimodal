import math
import torch
import torch.nn as nn

B = 2
N = 4
D = 512
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
# 7. 混合Dweights
Wo = torch.randn(D, D)
attn_out = output @ Wo

# Residual connection
residual = X + attn_out

# Layer Normalization
norm1 = nn.LayerNorm(D)
x1 = norm1(residual)

# FFN
D_ff = 2048

linear1 = nn.Linear(D, D_ff)
x_ff = linear1(x1)

x_ff = nn.GELU(x_ff)

linear2 = nn.Linear(D_ff, D)
ffn_out = linear2(x_ff)

# Residual Connection
residual2 = x1 + ffn_out

# LayerNorm
norm2 = nn.LayerNorm(D)
final_output = norm2(residual2)
