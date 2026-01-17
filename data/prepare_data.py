import torch
import torch.nn as nn


def load_data(file_path):
    """
    Load data from a given file path.
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        # data = f.readlines()
        return f.read() # Reading file content as a single string
    
def encode_text(tokenizer, text):
    """
    Encode text using the provided tokenizer.
    """
    return torch.tensor(tokenizer.encode(text), dtype=torch.long) # Convert to tensor
