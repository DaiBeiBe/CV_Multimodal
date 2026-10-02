import math
import torch

Q = torch.tensor(
    [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float32,
)
K = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ],
    dtype=torch.float32,
)
V = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=torch.float32)

scores = Q @ K.transpose(-2, -1)

scores = scores / math.sqrt(K.size(-1))
print("scores:", scores)

attention_weights = torch.softmax(scores, dim=-1)

output = attention_weights @ V
print(output)
print(output.shape)
