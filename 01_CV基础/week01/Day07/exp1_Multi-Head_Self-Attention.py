# 不用 nn.MultiheadAttention，自己用 PyTorch 写一个最小 Self-Attention，并让代码实际打印每一步 shape。
import math
import torch

B, N, D = 2, 5, 8
H = 2
d_head = D // H

X = torch.randn(B, N, D)

Wq = torch.randn(D, D)
Wk = torch.randn(D, D)
Wv = torch.randn(D, D)
Wo = torch.randn(D, D)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

# TODO 1
# [B, N, D] -> [B, H, N, d_head]
Q = torch.reshape(Q, [B, N, H, d_head]).transpose(-3, -2)
K = torch.reshape(K, [B, N, H, d_head]).transpose(-3, -2)
V = torch.reshape(V, [B, N, H, d_head]).transpose(-3, -2)

# TODO 2
# Q @ K^T
scores = Q @ K.transpose(-2, -1)

# TODO 3
# scaling + softmax
attn_weights = torch.softmax(scores / math.sqrt(d_head), dim=-1)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)
print("scores:", scores.shape)
print("attn_weights:", attn_weights.shape)

print("row sum:", attn_weights[0, 0, 0].sum())

# TODO 4
# weighted sum
output_head = attn_weights @ V

# TODO 5
# [B,H,N,d_head] -> [B,N,D]
output = output_head.transpose(-3, -2).reshape(B, N, D)

# TODO 6
# output projection
output = output @ Wo

print("output_head:", output_head.shape)
print("concat:", output.shape)
print("final:", output.shape)

print(attn_weights[0, 0, 0])
