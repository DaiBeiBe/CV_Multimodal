import torch
import math

N = 16
D = 64

position = torch.arange(N).unsqueeze(-1)

div_term = torch.exp(torch.arange(0, D, 2) * (-math.log(10000.0) / D))

pe = torch.zeros(N, D)

pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)

print(pe.shape)
print(pe[0])
print(pe[1])
print(pe[:, 0])
print(pe[:, 2])

# 可视化
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.imshow(pe.T)
plt.xlabel("Dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.show()
