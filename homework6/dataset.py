import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size=128):
        bos = tokenizer.token_to_id("<bos>")
        eos = tokenizer.token_to_id("<eos>")
        ids = [bos] + tokenizer.encode(text).ids + [eos]

        self.block_size = block_size
        self.inputs = []
        self.targets = []

        # Разбивка текста
        for i in range(0, len(ids) - block_size, block_size):
            x = torch.tensor(ids[i:i+block_size], dtype=torch.long)
            y = torch.tensor(ids[i+1:i+1+block_size], dtype=torch.long)
            self.inputs.append(x)
            self.targets.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
