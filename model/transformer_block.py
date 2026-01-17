import torch.nn as nn
from .attention import CausalSelfAttention
import torch

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        super().__init__()
        head_size = d_model // n_heads

        self.attn = nn.ModuleList([
            CausalSelfAttention(d_model, head_size, max_len)
            for _ in range(n_heads)

        ])

        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out = torch.cat([h(x) for h in self.attn], dim=-1)
        x = self.ln1(x + self.proj(attn_out))
        x = self.ln2(x + self.ff(x))
        return x
    