import torch
from tokenizer.char_tokenizer import CharTokenizer
from data.prepare_data import load_data, encode_text
from model.gpt import MiniGPT
from training.config import config
from training.train import train
from inference.generate import generate  # ← import generation function
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
import os



# ---------------- LOAD DATA ----------------
text = load_data("/home/danix/Machine-Learning/projects/mini-gpt/data/input.txt")
tokenizer = CharTokenizer(text)
data = encode_text(tokenizer, text)

# ---------------- INITIALIZE MODEL ----------------
model = MiniGPT(
    vocab_size = tokenizer.vocab_size,
    d_model = config["d_model"],
    n_layers = config["n_layers"],
    n_heads = config["n_heads"],
    max_len = config["block_size"]
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
if os.path.exists("mini_gpt.pth"):
    print("Loading model parameters...")
    model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))
    print("Model parameters loaded.")
else:
    print("Training model from scratch...")
    train(model, data, config, optimizer, device)
    torch.save(model.state_dict(), "mini_gpt.pth")
    print("Model saved!")
# ---------------- TRAIN MODEL ----------------
# train(model, data, config, optimizer, device)

# torch.save(model.state_dict(), "mini_gpt_model.pth")
# print("\nModel saved to mini_gpt_model.pth")

print("\n--- TRAINING FINISHED ---")
print("Text length:", len(text))
print("Token count:", len(data))
print("Vocab size:", tokenizer.vocab_size)

# ---------------- GENERATE SAMPLE TEXT ----------------
start_token = torch.zeros((1, 1), dtype=torch.long).to(device)  # start with a single token
generated = generate(model, start_token, max_new_tokens=300, device=device)

print("\n--- GENERATED TEXT ---\n")
print(tokenizer.decode(generated[0].tolist()))
