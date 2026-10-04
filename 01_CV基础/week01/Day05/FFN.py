import torch
import torch.nn as nn

x = torch.randn(2, 4, 512)
D = 512
D_ff = 2048

linear1 = nn.Linear(D, D_ff)
x_ff = linear1(x)

activation = nn.GELU()
x_ff = activation(x_ff)

linear2 = nn.Linear(D_ff, D)
output = linear2(x_ff)

print("Input shape:", x.shape)
print("Linear weight shape:", linear2.weight.shape)
print("Output shape:", output.shape)
