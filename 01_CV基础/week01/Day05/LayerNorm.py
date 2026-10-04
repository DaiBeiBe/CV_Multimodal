import torch

x = torch.randn(2, 4, 8)  # [B, N, D]

mean = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True, correction=False)

eps = 1e-5

x_norm = (x - mean) / torch.sqrt(var + eps)

gamma = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
beta = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

y_manual = gamma * x_norm + beta

ln = torch.nn.LayerNorm(8)
y_torch = ln(x_norm)

diff = (y_manual - y_torch).abs().max()
print(diff)

print(ln.weight.shape)
print(ln.bias.shape)
