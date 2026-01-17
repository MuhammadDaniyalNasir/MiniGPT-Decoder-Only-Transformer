config = {
    # -------- DATA --------
    "batch_size": 32,       # Number of sequences per training step
                            # Higher = faster training but more memory
    
    "block_size": 64,       # Context window / sequence length
                            # Model predicts next token given last 64 tokens
                            # Higher = more context but more computation
    
    # -------- TRAINING --------
    "epochs": 3000,         # Number of training iterations
                            # Each iteration processes one batch
    
    "lr": 3e-4,             # Learning rate for optimizer
                            # Controls step size during gradient descent
                            # Too high = unstable; too low = slow convergence
    
    # -------- MODEL ARCHITECTURE --------
    "n_heads": 4,           # Number of attention heads per block
                            # Each head attends to different features
                            # Must divide d_model evenly
    
    "n_layers": 4,          # Number of transformer blocks stacked
                            # More layers = more capacity but slower/harder to train
    
    "d_model": 128,         # Model dimension / embedding size
                            # Larger = more capacity but more parameters
                            # Must be divisible by n_heads (here: 128/4 = 32)
}