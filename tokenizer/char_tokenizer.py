class CharTokenizer:
    """
    Character-level tokenizer.
    
    Converts between:
    - Strings (text) ↔ Token IDs (numbers the model understands)
    
    WHY character-level: Simple, works with any text, small vocabulary.
    Downside: Sequences get long since each character is a token.
    
    HOW:
    - encode: "hello" → [id_h, id_e, id_l, id_l, id_o]
    - decode: [id_h, id_e, ...] → "hello"
    """
    
    def __init__(self, text):
        """
        Initialize tokenizer from text.
        
        Args:
            text: Sample text to build vocabulary from
                  Extracts all unique characters in the text
        """
        
        # Get all unique characters and sort them
        # WHY sort: Ensures consistent ordering across runs
        chars = sorted(set(text))
        
        # stoi = "string to integer" mapping
        # Example: {'a': 0, 'b': 1, 'c': 2, ...}
        # Allows fast lookup: char → token ID
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        
        # itos = "integer to string" mapping
        # Inverse of stoi: {0: 'a', 1: 'b', 2: 'c', ...}
        # Allows fast lookup: token ID → char
        self.itos = {i: ch for ch, i in self.stoi.items()}
        
        # Number of unique characters
        # This is the vocabulary size needed for model output layer
        self.vocab_size = len(chars)

    def encode(self, s):
        """
        Convert string to list of token IDs.
        
        Args:
            s: String to encode
        
        Returns:
            List of integers representing the string
            
        Example:
            encode("hi") → [4, 8]  (if 'h'→4, 'i'→8)
        """
        return [self.stoi[c] for c in s]
    
    def decode(self, ids):
        """
        Convert list of token IDs back to string.
        
        Args:
            ids: List of token IDs
        
        Returns:
            String representation
            
        Example:
            decode([4, 8]) → "hi"
        """
        return ''.join(self.itos[i] for i in ids)