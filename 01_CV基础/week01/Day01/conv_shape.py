import torch
import torch.nn as nn

x = torch.randn(1, 3, 640, 640)

conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
relu1 = nn.ReLU()
conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
relu2 = nn.ReLU()

y = conv1(x)
z = relu1(y)
m = conv2(z)
n = relu2(m)

print("input: ", x.shape)
print("conv1: ", y.shape)
print("relu1: ", z.shape)
print("conv2: ", m.shape)
print("relu2: ", n.shape)

import matplotlib.pyplot as plt

feature = n[0, 0].detach().numpy()

plt.imshow(feature, cmap="gray")
plt.title("Feature Map - Channel 0")
plt.axis("off")
plt.show()
