import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Image -> Patch -> Patch Embedding"""

    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        P = self.patch_size
        # 切 patch: (B, C, H/P, P, W/P, P)
        x = x.reshape(B, C, H // P, P, W // P, P)
        # 重排为: (B, H/P, W/P, C, P, P)
        x = x.permute(0, 2, 4, 1, 3, 5)
        # 展平每个 patch: (B, N, C*P*P)
        x = x.reshape(B, self.num_patches, self.patch_dim)
        # 线性投影: (B, N, embed_dim)
        x = self.proj(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim * mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MiniViT(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
    ):
        super().__init__()

        # Image -> Patch -> Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer Encoder
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # Image -> Patch -> Patch Embedding
        x = self.patch_embed(x)  # (B, N, D)

        # CLS
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        # Position Embedding
        x = x + self.pos_embed  # (B, N+1, D)

        # Transformer Encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # CLS
        cls_out = x[:, 0]  # (B, D)

        # Classification Head
        out = self.head(cls_out)  # (B, num_classes)
        return out


if __name__ == "__main__":
    model = MiniViT(
        img_size=32,
        patch_size=2,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=4,
        num_heads=4,
    )
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    # print(y.shape)

    # print("patch tokens:", model.patch_embed(x).shape)

    # patch_tokens = model.patch_embed(x)

    # B = patch_tokens.shape[0]
    # cls_tokens = model.cls_token.expand(B, -1, -1)

    # tokens = torch.cat((cls_tokens, patch_tokens), dim=1)

    # print("with cls:", tokens.shape)
    # print("pos embed:", model.pos_embed.shape)

    # tokens = tokens + model.pos_embed

    # print("after pos:", tokens.shape)

    # for block in model.blocks:
    #     tokens = block(tokens)

    # print("after transformer:", tokens.shape)

    # tokens = model.norm(tokens)
    # print("after norm:", tokens.shape)

    # cls_out = tokens[:, 0]
    # print("cls:", cls_out.shape)

    # print("output:", model.head(cls_out).shape)

    # exp_1
    # for patch_size in [8, 4, 2]:
    #     model = MiniViT(
    #         img_size=32,
    #         patch_size=patch_size,
    #         in_channels=3,
    #         num_classes=10,
    #         embed_dim=64,
    #         depth=4,
    #         num_heads=4,
    #     )

    #     print(
    #         f"patch_size={patch_size}, "
    #         f"num_patches={model.patch_embed.num_patches}, "
    #         f"num_tokens={model.patch_embed.num_patches + 1}"
    #     )

    # exp_2
    # for patch_size in [8, 4, 2]:

    #     print("=" * 50)
    #     print(f"Patch Size = {patch_size}")

    #     model = MiniViT(
    #         img_size=32,
    #         patch_size=patch_size,
    #         in_channels=3,
    #         num_classes=10,
    #         embed_dim=64,
    #         depth=1,  # 实验时先用 1 层
    #         num_heads=4,
    #     )

    #     x = torch.randn(2, 3, 32, 32)

    #     y = model(x)

    #     print("Output shape:", y.shape)

    # exp_3
    # for patch_size in [8, 4, 2]:

    #     N = (32 // patch_size) ** 2 + 1

    #     attention_elements = N * N

    #     print(f"P={patch_size}, " f"N={N}, " f"N²={attention_elements}")

    # exp_4
    for patch_size in [8, 4, 2]:

        model = MiniViT(
            img_size=32,
            patch_size=patch_size,
            in_channels=3,
            num_classes=10,
            embed_dim=64,
            depth=4,
            num_heads=4,
        )

        params = sum(p.numel() for p in model.parameters())

        print(f"patch_size={patch_size}, " f"params={params:,}")
