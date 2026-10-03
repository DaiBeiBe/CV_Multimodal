# 验证：你自己写出来的数学结构，和 PyTorch 官方实现，本质上是不是同一个东西。
import math
import torch
import torch.nn as nn

B = 2
N = 4
D = 8
H = 2
d_head = D // H

X = torch.randn(B, N, D)

mha = torch.nn.MultiheadAttention(embed_dim=D, num_heads=H, batch_first=True)

Wq = mha.in_proj_weight[:D]
Wk = mha.in_proj_weight[D : 2 * D]
Wv = mha.in_proj_weight[2 * D :]

bq = mha.in_proj_bias[:D]
bk = mha.in_proj_bias[D : 2 * D]
bv = mha.in_proj_bias[2 * D :]

Q = X @ Wq.T + bq
K = X @ Wk.T + bk
V = X @ Wv.T + bv

Q = Q.reshape(B, N, H, d_head).transpose(-3, -2)
K = K.reshape(B, N, H, d_head).transpose(-3, -2)
V = V.reshape(B, N, H, d_head).transpose(-3, -2)

scores = Q @ K.transpose(-2, -1)
scores = scores / math.sqrt(Q.size(-1))
attention_weights = torch.softmax(scores, dim=-1)

head_output = attention_weights @ V

output = head_output.transpose(-3, -2).reshape(B, N, D)

Wo = mha.out_proj.weight
bo = mha.out_proj.bias

final_output = output @ Wo.T + bo

pytorch_output, _ = mha(X, X, X, need_weights=False)

print(torch.allclose(final_output, pytorch_output, atol=1e-6))
print((final_output - pytorch_output).abs().max())
