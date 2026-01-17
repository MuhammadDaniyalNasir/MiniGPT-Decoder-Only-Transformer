import torch 
import torch.nn as nn
import torch.nn.functional as F
from training.dataset import get_batch

def train(model, data, config, optimizer, device):
    model.train()

    for step in range(config["epochs"]):
        x, y = get_batch(data, config["block_size"], config["batch_size"])
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)

        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
        