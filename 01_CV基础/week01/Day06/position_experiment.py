import math
import torch
import torch.nn as nn

A = [1, 0, 0, 0]
B = [0, 1, 0, 0]
C = [0, 0, 1, 0]

X_ABC = torch.tensor([A, B, C], dtype=torch.float32)
X_CBA = torch.tensor([C, B, A], dtype=torch.float32)
X_ABC = torch.reshape(X_ABC, [1, 3, 4])
X_CBA = torch.reshape(X_CBA, [1, 3, 4])

# print(X_ABC)
# print(X_CBA)
# print(X_ABC.shape)
# print(X_CBA.shape)


# 加入 Self-Attention
class SelfAttention(nn.Module):
    def __init__(self, d_modal):
        super().__init__()
        self.d_modal = d_modal

    def forward(self, X, mask=None):
        # 1. Q = X
        self.Q = X
        # 2. K = X
        self.K = X
        # 3. V = X
        self.V = X
        # 4. Q @ K.T
        self.scores = self.Q @ self.K.transpose(-2, -1)
        # 5. 除以 sqrt(d)
        self.scores = self.scores / math.sqrt(self.Q.size(-1))
        # 6. torch.softmax(...)
        self.attn_weights = torch.softmax(self.scores, dim=-1)
        # 7. 得到 Attention Output
        self.output = self.attn_weights @ self.V

        return self.scores, self.attn_weights, self.output


attn = SelfAttention(d_modal=4)
scores_ABC, attn_weights_ABC, output_ABC = attn(X=X_ABC)
scores_CBA, attn_weights_CBA, output_CBA = attn(X=X_CBA)

# print("Scores ABC:")
# print(scores_ABC)
# print("Attention Weights ABC:")
# print(attn_weights_ABC)
# print("Output ABC:")
# print(output_ABC)
# print("Scores CBA:")
# print(scores_CBA)
# print("Attention Weights CBA:")
# print(attn_weights_CBA)
# print("Output CBA:")
# print(output_CBA)

P = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
P = torch.tensor(P, dtype=torch.float32)

X_ABC_with_PE = X_ABC + P
X_CBA_with_PE = X_CBA + P

# print("X_ABC:")
# print(X_ABC)
# print("X_CBA:")
# print(X_CBA)
# print("X_ABC_with_PE:")
# print(X_ABC_with_PE)
# print("X_CBA_with_PE:")
# print(X_CBA_with_PE)

scores_ABC_PE, attn_weights_ABC_PE, output_ABC_PE = attn(X=X_ABC_with_PE)
scores_CBA_PE, attn_weights_CBA_PE, output_CBA_PE = attn(X=X_CBA_with_PE)

print("Scores ABC with PE:")
print(scores_ABC_PE)
print("Scores CBA with PE:")
print(scores_CBA_PE)
