import torch
from tokenizer.char_tokenizer import CharTokenizer
from data.prepare_data import load_data, encode_text
from model.gpt import MiniGPT
from training.config import config
from training.train import train
from inference.generate import generate
import torch.optim as optim

# ============================================
# SETUP: Device configuration
# ============================================
# Use GPU if available, otherwise CPU
# GPU is much faster for deep learning
device = "cuda" if torch.cuda.is_available() else "cpu"
import os

print(f"Using device: {device}\n")


# ============================================
# STEP 1: LOAD AND TOKENIZE DATA
# ============================================
print("Loading data...")
# Read raw text file
text = load_data("/data/input.txt")

# Create tokenizer from text (builds vocabulary)
# WHY: Tokenizer extracts all unique characters and maps them to IDs
tokenizer = CharTokenizer(text)

# Convert text string to tensor of token IDs
# WHY: Model works with numbers, not strings
# data shape: (total_tokens,)
data = encode_text(tokenizer, text)

print(f"Text length: {len(text)}")
print(f"Token count: {len(data)}")
print(f"Vocab size: {tokenizer.vocab_size}\n")


# ============================================
# STEP 2: INITIALIZE MODEL
# ============================================
print("Initializing model...")
# Create MiniGPT with specified architecture
model = MiniGPT(
    vocab_size = tokenizer.vocab_size,      # Number of unique characters
    d_model = config["d_model"],             # 128: embedding dimension
    n_layers = config["n_layers"],           # 4: transformer blocks
    n_heads = config["n_heads"],             # 4: attention heads
    max_len = config["block_size"]           # 64: context window
).to(device)

# Create optimizer (Adam is generally a good choice)
# WHY AdamW: Combines momentum with adaptive learning rates
# lr: learning rate - controls step size of weight updates
optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n")


# ============================================
# STEP 3: TRAIN OR LOAD MODEL
# ============================================
if os.path.exists("mini_gpt.pth"):
    # Model checkpoint exists: load pretrained weights
    print("Loading model parameters...")
    model.load_state_dict(torch.load("mini_gpt.pth", map_location=device))
    print("Model parameters loaded.\n")
else:
    # Model checkpoint doesn't exist: train from scratch
    print("Training model from scratch...")
    # WHY from scratch: Weights are randomly initialized, need training to learn
    train(model, data, config, optimizer, device)
    
    # Save trained model weights
    # WHY save: Can load later without retraining
    torch.save(model.state_dict(), "mini_gpt.pth")
    print("Model saved!\n")


# ============================================
# STEP 4: SUMMARY
# ============================================
print("--- TRAINING FINISHED ---")
print(f"Text length: {len(text)}")
print(f"Token count: {len(data)}")
print(f"Vocab size: {tokenizer.vocab_size}")


# ============================================
# STEP 5: GENERATE SAMPLE TEXT
# ============================================
print("\nGenerating text...")
# Start with a single token (ID=0, usually first character)
# WHY: Model predicts based on context; needs at least one token to start
start_token = torch.zeros((1, 1), dtype=torch.long).to(device)

# Generate 300 new tokens from this starting token
# WHY 300: Long enough to see interesting patterns but not too long
generated = generate(model, start_token, max_new_tokens=300, device=device)

print("\n--- GENERATED TEXT ---\n")
# Convert token IDs back to readable characters
print(tokenizer.decode(generated[0].tolist()))
