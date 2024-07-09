import torch
import torch.nn as nn
from torch import Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.scale = (embed_dim // n_heads) ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor):
        """ (B, L, D) -> (B, L, D) """
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)  # (B, H, L, D/H)
        k = k.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)  # (B, H, L, D/H)
        v = v.view(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)  # (B, H, L, D/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale                      # (B, H, L, L)
        causal_mask = torch.triu(torch.ones((L, L), device=x.device), diagonal=1).bool()
        attn.masked_fill_(causal_mask[None, None, :, :], float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, D)

        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x: Tensor):
        """ (B, L, D) -> (B, L, D) """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, codebook_num: int, embed_dim: int, n_heads: int, n_layers: int, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens

        self.SOS = nn.Parameter(torch.randn(embed_dim) * 0.02)  # the embedding of <SOS> token
        self.token_emb = nn.Embedding(codebook_num, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros((1, max_tokens, embed_dim)))

        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, n_heads) for _ in range(n_layers)])
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, codebook_num))

        self.apply(self._init_weights)

    def forward(self, idx: Tensor):
        """ (B, L) -> (B, L+1, C) """
        x = self.token_emb(idx)                                 # (B, L, D)
        assert x.shape[1] <= self.max_tokens
        x = x + self.pos_emb[:, :x.shape[1], :]                 # (B, L, D)

        SOS = self.SOS[None, None, :].repeat(x.shape[0], 1, 1)  # (B, 1, D)
        x = torch.cat((SOS, x), dim=1)                          # (B, L+1, D)

        x = self.blocks(x)                                      # (B, L+1, D)
        x = self.classifier(x)                                  # (B, L+1, C)
        return x

    def sample_one_step(self, idx: Tensor, temperature: float = 1.0, topk: int = None):
        """ (B, L) -> (B, L+1) """
        logits = self(idx[:, :self.max_tokens])[:, -1, :] / temperature
        if topk is not None:
            v, _ = torch.topk(logits, min(topk, logits.shape[-1]), largest=True, sorted=True)
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, 1)
        return torch.cat((idx, idx_next), dim=1)

    def sample(self, B: int, L: int, temperature: float = 1.0, topk: int = None):
        idx = torch.empty((B, 0), dtype=torch.int64, device=self.SOS.device)
        for i in range(L):
            idx = self.sample_one_step(idx, temperature, topk)
        return idx

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def _test():
    model = Transformer(codebook_num=512, embed_dim=512, n_heads=8, n_layers=12, max_tokens=256)
    # model = Transformer(codebook_num=1024, embed_dim=1024, n_heads=16, n_layers=24, max_tokens=512)
    x = torch.randint(0, 100, (10, 50))
    y = model(x)
    print(model)
    print(y.shape)  # (10, 51, 512)
    print(sum(p.numel() for p in model.parameters()))  # 38486016


if __name__ == '__main__':
    _test()
