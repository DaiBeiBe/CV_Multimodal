import math
import torch
import torch.nn as nn

B, N, D = 2, 5, 8
H = 2
d_head = D // H
D_ff = 16

X = torch.randn(B, N, D)

# ===== MHA =====
Wq = torch.randn(D, D)
Wk = torch.randn(D, D)
Wv = torch.randn(D, D)
Wo = torch.randn(D, D)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

Q = Q.reshape(B, N, H, d_head).transpose(1, 2)
K = K.reshape(B, N, H, d_head).transpose(1, 2)
V = V.reshape(B, N, H, d_head).transpose(1, 2)

scores = Q @ K.transpose(-2, -1)
attn_weights = torch.softmax(scores / math.sqrt(d_head), dim=-1)

output_head = attn_weights @ V

mha_output = output_head.transpose(1, 2).reshape(B, N, D)

mha_output = mha_output @ Wo

# ===== TODO =====
# Residual + LayerNorm
ln1 = nn.LayerNorm(D)
X1 = ln1(X + mha_output)
# print(X1.shape)

# FFN
ffn_linear1 = nn.Linear(D, D_ff)
ffn_linear2 = nn.Linear(D_ff, D)
ffn_output = ffn_linear2(torch.nn.functional.gelu(ffn_linear1(X1)))

# Residual + LayerNorm
ln2 = nn.LayerNorm(D)
X2 = ln2(X1 + ffn_output)
# print("X:", X.shape)
# print("MHA output:", mha_output.shape)
# print("X1:", X1.shape)
# print("FFN output:", ffn_output.shape)
# print("X2:", X2.shape)

# print("mean |X2 - X1|:", torch.mean(torch.abs(X2 - X1)).item())

X2_no_ffn = ln2(X1)
print("X2:", X2.shape)
print("X2_no_ffn:", X2_no_ffn.shape)

print("mean difference:", torch.mean(torch.abs(X2 - X2_no_ffn)).item())
