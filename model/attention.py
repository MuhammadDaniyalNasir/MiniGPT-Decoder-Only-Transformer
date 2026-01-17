import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, head_size, max_len):
        super().__init__()
        self.q = nn.Linear(d_model, head_size)
        self.k = nn.Linear(d_model, head_size)
        self.v = nn.Linear(d_model, head_size)
        self.max_len = max_len

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        att = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)

        return att @ v
    