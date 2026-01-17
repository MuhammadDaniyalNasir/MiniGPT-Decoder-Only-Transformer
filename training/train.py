import torch 
import torch.nn as nn
import torch.nn.functional as F
from training.dataset import get_batch

def train(model, data, config, optimizer, device):
    """
    Training loop for the model.
    
    HOW training works:
    1. Get random batch of sequences
    2. Forward pass: predict logits for next tokens
    3. Compute loss: compare predictions vs actual next tokens
    4. Backward pass: compute gradients
    5. Optimizer step: update model weights using gradients
    
    Args:
        model: MiniGPT model to train
        data: Tensor of all encoded tokens
        config: Dictionary with hyperparameters
        optimizer: Adam optimizer
        device: "cpu" or "cuda"
    """
    
    # Set model to training mode
    # WHY: Enables dropout, batch norm changes behavior in train vs eval
    model.train()

    # Training loop: run for N epochs
    for step in range(config["epochs"]):
        # Get a batch of random sequences and their targets
        # x shape: (batch_size, block_size)
        # y shape: (batch_size, block_size) - next tokens for each position
        x, y = get_batch(data, config["block_size"], config["batch_size"])
        
        # Move batch to correct device (GPU or CPU)
        x, y = x.to(device), y.to(device)

        # Forward pass: predict logits
        # logits shape: (batch_size, block_size, vocab_size)
        # For each position, we get scores for all vocabulary tokens
        logits = model(x)
        
        # Compute loss: cross-entropy between predicted logits and actual next tokens
        # Reshape for loss computation:
        # - logits: (batch_size, block_size, vocab_size) → (batch_size*block_size, vocab_size)
        # - y: (batch_size, block_size) → (batch_size*block_size,)
        # Cross-entropy measures how wrong our predictions are
        # Lower loss = better predictions
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # Flatten to (total_predictions, vocab_size)
            y.view(-1)                          # Flatten targets to (total_predictions,)
        )

        # Zero the gradients from previous step
        # WHY: Gradients accumulate by default; we want fresh gradients each step
        optimizer.zero_grad()
        
        # Backward pass: compute gradients
        # Calculates how much each parameter contributes to the loss
        loss.backward()
        
        # Optimizer step: update parameters
        # Moves parameters in direction opposite to gradients
        # Step size controlled by learning rate
        optimizer.step()

        # Print progress every 200 steps
        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
