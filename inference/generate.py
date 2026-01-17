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
    model.eval()  # freeze model

    for _ in range(max_new_tokens):
        # Crop context window
        idx_cond = idx[:, -model.block_size:].to(device)

        # Forward pass
        logits = model(idx_cond)

        # Focus only on last time step
        logits = logits[:, -1, :]  # shape: (B, vocab_size)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)

        # Append next token to sequence
        idx = torch.cat([idx, next_token], dim=1)

    return idx
