import torch

def get_batch(data, block_size, batch_size):
    """
    Create a batch of training examples.
    
    Creates training pairs (x, y) where:
    - x = input sequence of block_size tokens
    - y = target sequence (x shifted right by 1)
    
    WHY shifted: Model learns to predict next token given previous tokens
    
    Args:
        data: Tensor of all token IDs, shape (total_tokens,)
        block_size: Length of each training sequence
        batch_size: Number of examples in batch
    
    Returns:
        x: Batch of inputs, shape (batch_size, block_size)
        y: Batch of targets, shape (batch_size, block_size)
        
    HOW:
    1. Randomly pick batch_size starting positions
    2. Extract block_size tokens from each position for x
    3. Extract block_size tokens shifted by 1 for y (next token prediction)
    """
    
    # Random starting positions for batch samples
    # Ensures: 0 ≤ ix < len(data) - block_size (room for full sequence)
    # Shape: (batch_size,)
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # Create input sequences
    # For each starting position, extract block_size consecutive tokens
    # x = data[i:i+block_size] for each i in ix
    # Result shape: (batch_size, block_size)
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Create target sequences (shifted by 1)
    # Model predicts: "given x[0:T], predict y[0:T]"
    # Where y[t] = x[t+1] (the next token)
    # y = data[i+1:i+block_size+1] for each i in ix
    # Result shape: (batch_size, block_size)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y