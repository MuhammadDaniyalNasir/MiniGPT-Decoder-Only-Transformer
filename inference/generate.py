# import torch 

# @torch.no_grad()
# def generate(model, idx, max_new_tokens, device):
#     model.eval()

#     for _ in range(max_new_tokens):
#         logits, _ = model(idx)
#         probs = torch.softmax(logits[:, -1, :], dim=-1)
#         next_token = torch.multinomial(probs, num_samples=1)
#         idx = torch.cat([idx, next_token], dim=1)
#         idx_cond = idx[:, -model.block_size:]
#         logits, _ = model(idx_cond)

#     return idx
import torch

@torch.no_grad()
def generate(model, idx, max_new_tokens, device):
    """
    Generate new text tokens autoregressively.
    
    Autoregressive generation:
    1. Start with initial token(s)
    2. Predict next token based on sequence
    3. Append predicted token to sequence
    4. Repeat steps 2-3 until desired length
    
    WHY @torch.no_grad(): Disables gradient tracking (faster, uses less memory)
                          We don't need gradients during generation
    
    Args:
        model: Trained MiniGPT model
        idx: Starting token(s), shape (1, 1)
        max_new_tokens: How many tokens to generate
        device: "cpu" or "cuda"
    
    Returns:
        Generated sequence including starting token
        Shape: (1, 1+max_new_tokens)
    """
    
    # Set model to evaluation mode
    # WHY: Disables dropout, uses running stats for batch norm (if used)
    # Ensures deterministic predictions
    model.eval()

    # Generate loop: create one new token at a time
    for _ in range(max_new_tokens):
        # Crop context to model's maximum length
        # WHY: Model can only process up to block_size tokens
        # If sequence exceeds block_size, keep only the most recent block_size tokens
        # Shape after crop: (1, min(sequence_length, block_size))
        idx_cond = idx[:, -model.block_size:].to(device)

        # Forward pass: predict logits for all positions
        # logits shape: (1, T, vocab_size) where T = current sequence length
        logits = model(idx_cond)

        # Extract logits for only the last position (most recent token)
        # WHY: We only care about next token prediction
        # Shape: (1, vocab_size)
        logits = logits[:, -1, :]

        # Convert logits to probabilities
        # Higher logits → higher probabilities
        # Probabilities sum to 1 across vocabulary
        # Shape: (1, vocab_size)
        probs = torch.softmax(logits, dim=-1)

        # Sample next token from probability distribution
        # torch.multinomial: randomly pick a token
        # Higher probability tokens more likely to be selected
        # num_samples=1: pick exactly 1 token
        # Shape: (1, 1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append next token to sequence
        # Concatenate along sequence dimension (dim=1)
        # Shape grows: (1, previous_length+1)
        idx = torch.cat([idx, next_token], dim=1)

    # Return full generated sequence
    # Includes the original starting token plus all newly generated tokens
    return idx
