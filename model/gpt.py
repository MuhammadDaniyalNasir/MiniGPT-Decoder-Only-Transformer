import torch
import torch.nn as nn
from .transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    """
    A miniature GPT-like model for text generation.
    
    Architecture:
    1. Token embedding: Convert token IDs to dense vectors
    2. Positional embedding: Add position information
    3. Stack of transformer blocks: Process with self-attention & feed-forward
    4. Output head: Project to vocabulary size for next token prediction
    
    HOW: Takes token IDs as input, outputs logits over vocabulary for prediction.
    """
    
    def __init__(self, vocab_size, d_model, n_layers, n_heads, max_len):
        """
        Initialize MiniGPT model.
        
        Args:
            vocab_size: Number of unique tokens in vocabulary
            d_model: Embedding dimension (hidden size)
            n_layers: Number of transformer blocks to stack
            n_heads: Number of attention heads in each block
            max_len: Maximum sequence length (context window)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_len = max_len
        self.block_size = max_len  # Alias for sequence length limit
        
        # Convert token IDs (0 to vocab_size-1) to dense vectors
        # WHY embedding: Allows model to learn semantic relationships between tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Add position information to embeddings
        # WHY positional embedding: Transformer has no inherent notion of sequence order
        # Position embedding tells model "this is token at position T"
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # Stack n_layers transformer blocks
        # Each block has attention + feed-forward
        # WHY stacking: Deeper networks can learn more complex patterns
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, max_len) for _ in range(n_layers)]
        )
        
        # Final layer normalization
        # Normalizes activations before output projection
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        # Projects d_model dimension down to vocab_size
        # Each output value is a logit (pre-softmax score) for that token
        self.head = nn.Linear(d_model, vocab_size)
        

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Batch of token IDs, shape (B, T)
               B = batch size, T = sequence length
        
        Returns:
            Logits tensor of shape (B, T, vocab_size)
            Each position has vocab_size scores for next token prediction
        """
        B, T = x.size()

        # Create position indices: [0, 1, 2, ..., T-1]
        # WHY: Need to index into positional embedding
        pos = torch.arange(0, T, device=x.device)

        # Combine token embeddings with position embeddings
        # WHY add: Both encode information; sum combines them
        # Shape: (B, T, d_model)
        x = self.token_embedding(x) + self.pos_embedding(pos)
        
        # Pass through all transformer blocks
        # Each block processes the sequence with attention and feed-forward
        x = self.blocks(x)
        
        # Final layer normalization
        x = self.ln_f(x)

        # Project to vocabulary size
        # Output logits for each position and token
        # Shape: (B, T, vocab_size)
        return self.head(x)
