import torch
from torch.utils.data import Dataset

class ARGDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get the sequence and label
        tokens = self.dataframe.iloc[idx]['encoded']
        label = self.dataframe.iloc[idx]['label']

        tokens = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        
        return tokens, label
