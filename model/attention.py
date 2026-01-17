import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    """
    Implements a single attention head with causal masking.
    
    Causal masking ensures that each token can only attend to previous tokens
    (and itself), preventing information leakage from future tokens.
    
    WHY: This is essential for autoregressive language models where we generate
    one token at a time and can't look ahead.
    """
    
    def __init__(self, d_model, head_size, max_len):
        """
        Initialize attention head.
        
        Args:
            d_model: Total embedding dimension (will be split across heads)
            head_size: Dimension of this specific attention head
            max_len: Maximum sequence length (needed for causal mask buffer)
        """
        super().__init__()
        
        # Linear projections for Query, Key, Value
        # Each projects from d_model to head_size
        self.q = nn.Linear(d_model, head_size)  # Query projection
        self.k = nn.Linear(d_model, head_size)  # Key projection
        self.v = nn.Linear(d_model, head_size)  # Value projection
        self.max_len = max_len

        # Register causal mask as a buffer (not a learnable parameter)
        # Lower triangular matrix: allows attending to current & past tokens only
        # torch.tril creates a lower triangular matrix of ones
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
        )
    
    def forward(self, x):
        """
        Compute scaled dot-product attention with causal masking.
        
        Args:
            x: Input tensor of shape (B, T, C)
               B = batch size
               T = sequence length (time steps)
               C = embedding dimension (d_model)
        
        Returns:
            Attention output of shape (B, T, head_size)
        """
        B, T, C = x.size()
        
        # Project input to Q, K, V
        q = self.q(x)  # (B, T, head_size)
        k = self.k(x)  # (B, T, head_size)
        v = self.v(x)  # (B, T, head_size)

        # Compute attention scores: (Q @ K^T) / sqrt(d_k)
        # WHY divide by sqrt(head_size): Prevents scores from becoming too large
        # and keeps gradients stable during backprop
        att = (q @ k.transpose(-2, -1)) / (C ** 0.5)  # (B, T, T)
        
        # Apply causal mask: set future positions to -inf
        # mask[:T, :T] gets the relevant portion for current sequence length
        # Where mask == 0, set to -inf (these are future tokens we can't see)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        
        # Convert scores to probabilities (0 to 1)
        # -inf values become 0 after softmax
        att = torch.softmax(att, dim=-1)  # (B, T, T)

        # Weight values by attention probabilities
        # Each token attends to all previous tokens based on learned attention weights
        return att @ v  # (B, T, head_size)
