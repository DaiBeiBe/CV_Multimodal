import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# 1. 实验配置
# =========================

torch.manual_seed(42)

B = 32
N = 5
D = 16
H = 4
D_ff = 32
VOCAB_SIZE = 10

d_head = D // H


# =========================
# 2. Mini Transformer Block
# =========================


class MiniTransformerBlock(nn.Module):

    def __init__(self, use_ffn=True):
        super().__init__()

        self.use_ffn = use_ffn

        # MHA
        self.Wq = nn.Linear(D, D, bias=False)
        self.Wk = nn.Linear(D, D, bias=False)
        self.Wv = nn.Linear(D, D, bias=False)
        self.Wo = nn.Linear(D, D, bias=False)

        # LayerNorm
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

        # FFN
        self.linear1 = nn.Linear(D, D_ff)
        self.linear2 = nn.Linear(D_ff, D)

    def mha(self, x):

        B, N, D = x.shape

        # Q K V
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)

        # [B,N,D]
        # ->
        # [B,H,N,d_head]

        Q = Q.reshape(B, N, H, d_head).transpose(1, 2)
        K = K.reshape(B, N, H, d_head).transpose(1, 2)
        V = V.reshape(B, N, H, d_head).transpose(1, 2)

        # Attention Score
        scores = Q @ K.transpose(-2, -1)

        scores = scores / math.sqrt(d_head)

        # Attention Weight
        attn_weights = torch.softmax(scores, dim=-1)

        # Attention Output
        output = attn_weights @ V

        # [B,H,N,d_head]
        # ->
        # [B,N,D]

        output = output.transpose(1, 2).reshape(B, N, D)

        # Output Projection
        output = self.Wo(output)

        return output

    def forward(self, x):

        # ===== MHA =====

        mha_output = self.mha(x)

        # Residual + LayerNorm

        x1 = self.ln1(x + mha_output)

        # ===== FFN =====

        if self.use_ffn:

            ffn_output = self.linear2(F.gelu(self.linear1(x1)))

        else:

            ffn_output = torch.zeros_like(x1)

        # Residual + LayerNorm

        x2 = self.ln2(x1 + ffn_output)

        return x2


# =========================
# 3. Copy Task Model
# =========================


class CopyModel(nn.Module):

    def __init__(self, use_ffn=True):

        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, D)

        self.transformer = MiniTransformerBlock(use_ffn=use_ffn)

        self.classifier = nn.Linear(D, VOCAB_SIZE)

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        logits = self.classifier(x)

        return logits


# =========================
# 4. Generate Copy Task
# =========================

x = torch.randint(0, VOCAB_SIZE, (B, N))

y = x.clone()

# print("Input:")
# print(x)

# print("\nTarget:")
# print(y)


model = CopyModel(use_ffn=True)
logits = model(x)
# print("\nLogits shape:")
# print(logits.shape)

# 计算初始 Loss
loss = F.cross_entropy(logits.reshape(B * N, VOCAB_SIZE), y.reshape(B * N))
print("Initial loss:", loss.item())

# =========================
# 5. Train Baseline
# =========================

torch.manual_seed(42)

model = CopyModel(use_ffn=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 300

for epoch in range(EPOCHS):
    model.train()

    logits = model(x)

    loss = F.cross_entropy(logits.reshape(B * N, VOCAB_SIZE), y.reshape(B * N))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], " f"Loss: {loss.item():.4f}")


# =========================
# 6. Evaluate
# =========================

model.eval()

with torch.no_grad():
    logits = model(x)
    pred = logits.argmax(dim=-1)

    accuracy = (pred == y).float().mean().item()

# print("\nBaseline final loss:", loss.item())
# print("Baseline accuracy:", accuracy)
# print("Input[0]:", x[0].tolist())
# print("Target[0]:", y[0].tolist())
# print("Pred[0]:", pred[0].tolist())


# =========================
# 7. Train FFN Ablation
# =========================

torch.manual_seed(42)

model_no_ffn = CopyModel(use_ffn=False)

optimizer_no_ffn = torch.optim.Adam(model_no_ffn.parameters(), lr=1e-3)

EPOCHS = 300

for epoch in range(EPOCHS):
    model_no_ffn.train()

    logits_no_ffn = model_no_ffn(x)

    loss_no_ffn = F.cross_entropy(
        logits_no_ffn.reshape(B * N, VOCAB_SIZE), y.reshape(B * N)
    )

    optimizer_no_ffn.zero_grad()
    loss_no_ffn.backward()
    optimizer_no_ffn.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}], " f"Loss: {loss_no_ffn.item():.4f}")


# =========================
# 8. Evaluate Ablation
# =========================

model_no_ffn.eval()

with torch.no_grad():
    logits_no_ffn = model_no_ffn(x)
    pred_no_ffn = logits_no_ffn.argmax(dim=-1)

    accuracy_no_ffn = (pred_no_ffn == y).float().mean().item()

print("\nAblation final loss:", loss_no_ffn.item())
print("Ablation accuracy:", accuracy_no_ffn)

print("Input[0]:", x[0].tolist())
print("Target[0]:", y[0].tolist())
print("Pred without FFN:", pred_no_ffn[0].tolist())
