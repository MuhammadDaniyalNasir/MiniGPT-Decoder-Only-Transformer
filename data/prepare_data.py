import torch
import torch.nn as nn


def load_data(file_path):
    """
    Load raw text data from file.
    
    Args:
        file_path: Path to text file
    
    Returns:
        String containing entire file content
        
    WHY: Loads training data into memory for tokenization
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read entire file as single string (preserves newlines, spaces, etc.)
        return f.read()
    
def encode_text(tokenizer, text):
    """
    Convert text string to tensor of token IDs.
    
    Args:
        tokenizer: CharTokenizer object with encode method
        text: String to encode
    
    Returns:
        PyTorch tensor of dtype long (integers)
        Shape: (length_of_text,)
        
    WHY tensor: PyTorch models work with tensors; enables GPU processing
    WHY dtype=long: Token IDs are integers, long is standard for indices
    
    Example:
        Input: "hello"
        Output: tensor([4, 8, 12, 12, 15]) if vocab maps chars to these IDs
    """
    # Encode text to list of token IDs, then convert to tensor
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)
