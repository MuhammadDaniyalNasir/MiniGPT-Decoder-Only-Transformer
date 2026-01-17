# Mini-GPT (PyTorch)

A **from-scratch implementation of a GPT-style language model** using PyTorch.

This project is intentionally minimal and educational — designed to help you **deeply understand Transformers, attention, tokenization, training loops, and text generation**, without relying on high-level libraries like HuggingFace.

---

## 🚀 Features

- Character-level tokenizer (fully custom)
- GPT-style Transformer architecture
- Causal self-attention
- Feed-forward networks with expansion
- Training + inference pipeline
- Text generation (autoregressive)
- Model checkpoint saving/loading

---

## 📂 Project Structure

```
mini-gpt/
│
├── main.py                  # Entry point (train + generate)
├── requirements.txt
│
├── model/
│   ├── gpt.py               # MiniGPT model
│   ├── transformer_block.py # Attention + FFN block
│   └── attention.py         # Causal self-attention
│
├── tokenizer/
│   └── char_tokenizer.py    # Character-level tokenizer
│
├── training/
│   ├── train.py             # Training loop
│   ├── dataset.py           # Batch sampling
│   └── config.py            # Hyperparameters
│
├── inference/
│   └── generate.py          # Text generation logic
│
├── data/
│   └── input.txt            # Training text corpus
│
└── checkpoints/
    └── model.pt             # Saved model weights
```

---

## 📦 Installation

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

Add your training text to:

```
data/input.txt
```

Then run:

```bash
python main.py
```

The model will:
- Train from scratch **or** load saved weights if available
- Save the model to `checkpoints/model.pt`

---

## ✨ Text Generation

After training completes, the model automatically generates text.

You can control:
- Starting token / prompt
- Number of generated tokens

Generation is **autoregressive**, one token at a time.

---

## ⚙️ Configuration

Edit hyperparameters in:

```
training/config.py
```

Example:

```python
config = {
    "batch_size": 32,
    "block_size": 128,
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "lr": 3e-4,
    "max_iters": 3000
}
```

---

## 📖 Learning Goals

This project is designed to help you understand:

- How GPT-style models work internally
- Why attention + feed-forward layers are both necessary
- How tokenization affects learning
- How autoregressive generation works
- How model capacity influences language quality

---

## 🧪 Why Character-Level Tokenization?

- Extremely simple
- Transparent
- Perfect for learning

Later, this can be extended to:
- BPE / WordPiece
- SentencePiece
- Byte-level encoding

---

## 🔮 Future Improvements

- Add GELU instead of ReLU
- Add dropout
- Add validation split
- Add temperature / top-k sampling
- Add CLI arguments
- Switch to subword tokenization

---

## 🧑‍💻 Author

Built as a **learning-focused GPT implementation** by a Computer Science student exploring deep learning and language models.

---

## 📜 License

MIT License — free to use, modify, and learn from.
