import torch
import torch.nn as nn
from .transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len  # already exists
        self.block_size = max_len  # ← ADD THIS
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, max_len) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        B, T = x.size()

        pos = torch.arange(0, T, device=x.device)

        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.blocks(x)
        x = self.ln_f(x)

        return self.head(x)
 