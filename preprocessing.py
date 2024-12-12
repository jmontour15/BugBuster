import pandas as pd
import numpy as np
import torch


def aa_encode(sequence, aa_map):
    encoded_sequence = []
    
    for aa in sequence:
        encoded_sequence.append(aa_map[aa])

    return encoded_sequence

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch to the maximum sequence length in the batch
    and compute attention masks.

    Args:
        batch: List of tuples (tokens, label) from the dataset's __getitem__.
        
    Returns:
        Padded sequences tensor, corresponding label
    """
    # Separate tokens and labels
    tokens_list, labels_list = zip(*batch)

    # Ensure tokenized sequences are tensors
    token_tensors = [torch.tensor(tokens, dtype=torch.long) for tokens in tokens_list]

    # Debugging: print token sequence lengths
    # print("Token sequence lengths:", [len(tokens) for tokens in token_tensors])

    # Pad sequences to the max sequence length in the batch
    padded_tokens = pad_sequence(
        token_tensors,
        batch_first=True,
        padding_value=0  # Use 0 as the padding index
    )

    # Compute attention masks
    # 1 for actual tokens, 0 for padding tokens
    attention_masks = (padded_tokens != 0).long()

    # Debugging: print padded tokens and masks
    # print("Padded tokens:", padded_tokens)
    # print("Attention masks:", attention_masks)

    # Convert labels to tensor
    labels = torch.tensor(labels_list, dtype=torch.long)

    return padded_tokens, attention_masks, labels


