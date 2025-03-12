import torch
from torch.utils.data import Dataset

class ESM2Dataset(Dataset):
    def __init__(self, dataframe, batch_converter):
        self.data = dataframe
        self.batch_converter = batch_converter

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the sequence and label
        sequence = self.data.iloc[idx]['sequence']
        label = self.data.iloc[idx]['label']

        # Tokenize sequence using the ESM batch converter
        _, _, tokens = self.batch_converter([(None, sequence)])
        tokens = torch.tensor(tokens, dtype=torch.long).squeeze(0)
        
        return tokens, torch.tensor(label, dtype=torch.long)