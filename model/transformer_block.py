import torch.nn as nn
from .attention import CausalSelfAttention
import torch

class TransformerBlock(nn.Module):
    """
    A single Transformer block combining multi-head attention and feed-forward.
    
    Architecture:
    1. Multi-head self-attention (parallel attention heads)
    2. Layer normalization + residual connection
    3. Feed-forward network (2 linear layers with ReLU)
    4. Layer normalization + residual connection
    
    WHY this design: Residual connections help gradient flow and allow deeper networks.
    Normalization stabilizes training. Multiple attention heads capture different features.
    """
    
    def __init__(self, d_model, n_heads, max_len, dropout=0.1):
        """
        Initialize transformer block.
        
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            max_len: Maximum sequence length
            dropout: Dropout rate (currently defined but not used)
        """
        super().__init__()
        
        # Calculate dimension for each attention head
        # WHY divide: Each head gets d_model/n_heads dimensions
        # Example: d_model=128, n_heads=4 → head_size=32 each
        head_size = d_model // n_heads

        # Create multiple attention heads
        # Each head performs attention independently, then outputs are concatenated
        # WHY multiple heads: Different heads learn different types of relationships
        self.attn = nn.ModuleList([
            CausalSelfAttention(d_model, head_size, max_len)
            for _ in range(n_heads)
        ])

        # Project concatenated attention outputs back to d_model dimension
        # Input: (B, T, d_model) from concatenated heads
        # Output: (B, T, d_model) after projection
        self.proj = nn.Linear(d_model, d_model)
        
        # Layer normalization after attention
        # Normalizes features to have mean=0, std=1
        # WHY: Stabilizes training and allows higher learning rates
        self.ln1 = nn.LayerNorm(d_model)

        # Feed-forward network (position-wise fully connected)
        # Expands to 4*d_model, applies ReLU (non-linearity), then contracts back
        # WHY 4x expansion: Provides capacity for complex transformations
        # ReLU adds non-linearity to learn complex patterns
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # Expand
            nn.ReLU(),                         # Non-linearity
            nn.Linear(4 * d_model, d_model)   # Contract back
        )
        
        # Layer normalization after feed-forward
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
        
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        
        # Multi-head attention: run each head in parallel and concatenate
        # Each head outputs (B, T, head_size), concat gives (B, T, d_model)
        attn_out = torch.cat([h(x) for h in self.attn], dim=-1)
        
        # Residual connection + projection + layer norm
        # WHY residual (x + ...): Helps gradients flow and preserves original info
        # x + proj(attn_out) combines attention output with original input
        x = self.ln1(x + self.proj(attn_out))
        
        # Feed-forward network + residual connection + layer norm
        # Another residual connection ensures information from attention is preserved
        x = self.ln2(x + self.ff(x))
        
        return x
