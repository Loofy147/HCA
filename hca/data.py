import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """
    A character-level dataset for training the HCA model.
    """
    def __init__(self, file_path: str, seq_len: int):
        """
        Initializes the dataset.

        Args:
            file_path (str): The path to the text file.
            seq_len (int): The length of the input sequences.
        """
        with open(file_path, 'r') as f:
            self.text = f.read()

        self.seq_len = seq_len
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.encoded_text = [self.stoi[ch] for ch in self.text]

    def __len__(self):
        return len(self.encoded_text) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.encoded_text[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
