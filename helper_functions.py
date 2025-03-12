import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logomaker

def make_predictions(model, dataloader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for sequences, attention_masks, targets in tqdm(dataloader):
            sequences, attention_masks, targets = sequences.to(device), attention_masks.to(device), targets.to(device)
        
            # Get model predictions
            outputs = model(sequences, attention_mask=attention_masks)

            # Convert logits to class labels
            predicted = (torch.sigmoid(outputs) > 0.5).long().cpu().numpy().flatten()
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted)
            
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return y_true, y_pred

def plot_attention_maps(weights, attention_mask, tick_spacing=10):
    # Extract weights
    mlm_weights = weights["MLM_weights"]  # Shape: [batch_size, num_layers, num_heads, seq_len, seq_len]
    classification_weights = weights["task_specific_head_weights"]  # Same shape

    # Compute mean over layers and heads -> Shape: [batch_size, seq_len, seq_len]
    mlm_mean = mlm_weights.mean(dim=(1, 2))
    classification_mean = classification_weights.mean(dim=(1, 2))
    combined_mean = (mlm_mean + classification_mean) / 2

    # Move to CPU if using GPU
    mlm_mean = mlm_mean.cpu().numpy()
    classification_mean = classification_mean.cpu().numpy()
    combined_mean = combined_mean.cpu().numpy()
    attention_mask = attention_mask.cpu().numpy()  # Shape: [batch_size, seq_len]

    batch_size, seq_len = mlm_mean.shape[0], mlm_mean.shape[1]

    # Set up figure with 3 columns (MLM, Classification, Combined) and batch_size rows
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 4 * batch_size))

    for i in range(batch_size):
        token_indices = np.where(attention_mask[i] == 1)[0]  # Indices of real tokens
        num_tokens = len(token_indices)

        # Limit tick labels (every `tick_spacing` tokens)
        xtick_labels = [str(idx) if j % tick_spacing == 0 else "" for j, idx in enumerate(token_indices)]
        ytick_labels = xtick_labels  # Match y-axis ticks to x-axis ticks

        # Define a helper function for plotting
        def plot_heatmap(ax, data, title):
            sns.heatmap(data[:num_tokens, :num_tokens], 
                        ax=ax, cmap="viridis", square=True, cbar=True,
                        xticklabels=xtick_labels, yticklabels=ytick_labels)
            ax.set_title(title)
            ax.set_xlabel("Token Index")
            ax.set_ylabel("Token Index")
            ax.tick_params(axis='x', rotation=45)  # Rotate x-tick labels

        # Plot attention maps
        plot_heatmap(axes[i, 0], mlm_mean[i], f"MLM Sample {i+1}")
        plot_heatmap(axes[i, 1], classification_mean[i], f"Classification Head Sample {i+1}")
        plot_heatmap(axes[i, 2], combined_mean[i], f"Combined Sample {i+1}")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
import torch

def motif_plot(weights, attention_mask, sequences, aa_to_int=None):
    """
    This function plots motif logos based on the provided weights and sequences,
    with improved emphasis on higher weights.
    
    Parameters:
    - weights: Dictionary with keys "MLM_weights" and "task_specific_head_weights",
               MLM_weights shape: [batch_size, layers, heads, seq_len, seq_len]
               task_specific_head_weights shape: [batch_size, layers, heads, seq_len, seq_len]
    - attention_mask: Tensor of shape [batch_size, sequence_length] to filter out padded tokens
    - sequences: List of amino acid sequence strings
    - aa_to_int (optional): A dictionary to map amino acids to integers. 
    """
    # Default amino acid to integer mapping if not provided
    if aa_to_int is None:
        aa_to_int = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 
                     'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 
                     'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20}
    
    # Ensure attention mask is properly applied
    batch_size = len(sequences)
    
    # Extract weights
    mlm_weights = weights.get("MLM_weights", None)
    task_weights = weights.get("task_specific_head_weights", None)
    
    # Check if we need to move tensors to CPU and convert to numpy
    if isinstance(mlm_weights, torch.Tensor):
        mlm_weights = mlm_weights.detach().cpu().numpy()
    if isinstance(task_weights, torch.Tensor):
        task_weights = task_weights.detach().cpu().numpy()
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.detach().cpu().numpy()
    
    # Use either MLM weights, task weights, or both if available
    if mlm_weights is not None and task_weights is not None:
        # Average over layers and heads for each weight type
        mlm_mean = np.mean(mlm_weights, axis=(1, 2))  # [batch_size, seq_len, seq_len]
        task_mean = np.mean(task_weights, axis=(1, 2))  # [batch_size, seq_len, seq_len]
        # Calculate combined weights
        combined_weights = (mlm_mean + task_mean) / 2
    elif mlm_weights is not None:
        combined_weights = np.mean(mlm_weights, axis=(1, 2))
    elif task_weights is not None:
        combined_weights = np.mean(task_weights, axis=(1, 2))
    else:
        raise ValueError("No valid weights provided")
    
    # Create figure - only one row of plots
    fig, axes = plt.subplots(batch_size, 1, figsize=(15, 3 * batch_size))
    
    # Make axes iterable for single batch case
    if batch_size == 1:
        axes = [axes]
    
    # Process each sequence in the batch
    for i in range(batch_size):
        # Convert current sequence to list of amino acids
        seq = sequences[i]
        
        # Get the actual sequence length (without padding)
        if np.ndim(attention_mask) > 1:
            # For 2D attention mask [batch_size, seq_len]
            actual_length = int(np.sum(attention_mask[i]))
        else:
            # For 1D attention mask or if not provided
            actual_length = len(seq)
        
        # Ensure we don't exceed the sequence length
        actual_length = min(actual_length, len(seq))
        
        # Extract the relevant part of the attention matrix
        seq_attention = combined_weights[i, :actual_length, :actual_length]
        
        # Create a matrix to store amino acid information for the logo
        logo_matrix = np.zeros((actual_length, len(aa_to_int)))
        
        # For each position in the sequence
        for pos in range(actual_length):
            # Get the attention weights for this position
            pos_weights = seq_attention[pos, :]
            
            # Create a dictionary to aggregate weights by amino acid
            aa_weights = {aa: 0 for aa in aa_to_int.keys()}
            
            # Map attention weights to corresponding amino acids in the sequence
            for j in range(actual_length):
                if j < len(seq):
                    aa = seq[j]
                    if aa in aa_weights:
                        aa_weights[aa] += pos_weights[j]
            
            # Apply non-linear transformation to emphasize higher weights
            # Using power function with exponent > 1 to emphasize high values
            transformed_weights = {aa: weight**3 for aa, weight in aa_weights.items()}
            
            # Normalize the transformed weights to sum to 1
            total = sum(transformed_weights.values())
            if total > 0:  # Avoid division by zero
                normalized_weights = {aa: weight/total for aa, weight in transformed_weights.items()}
            else:
                normalized_weights = transformed_weights
            
            # Fill the logo matrix
            for aa, weight in normalized_weights.items():
                aa_idx = aa_to_int[aa] - 1  # Convert 1-indexed to 0-indexed
                logo_matrix[pos, aa_idx] = weight
        
        # Create a DataFrame for logomaker with amino acid labels
        logo_df = pd.DataFrame(logo_matrix, columns=list(aa_to_int.keys()))
        
        # Plot using logomaker with improved styling
        logo = logomaker.Logo(logo_df, ax=axes[i], color_scheme='chemistry')
        axes[i].set_title(f"Attention-based Motif for Sequence {i+1}", fontsize=14)
        axes[i].set_xlabel("Position", fontsize=12)
        axes[i].set_ylabel("Information Content", fontsize=12)
        
        # Add x-axis ticks for every 10 positions
        if actual_length > 10:
            tick_positions = np.arange(0, actual_length, 20)
            axes[i].set_xticks(tick_positions)
            axes[i].set_xticklabels(tick_positions)
        
    plt.tight_layout()
    plt.show()
    return fig

"""
CLAUDE STUFF
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def visualize_attention(self, batch_tokens, attention_mask, sequences=None, layer_idx=0, head_idx=0):
    """
    Visualize attention matrix from the classification head with min-max normalization.
    
    Parameters:
    - batch_tokens: Tokenized input sequences
    - attention_mask: Attention mask tensor
    - sequences: Original amino acid sequences (optional, for axis labels)
    - layer_idx: Layer index to visualize (default: 0)
    - head_idx: Head index to visualize (default: 0)
    
    Returns:
    - Matplotlib figure
    """
    self.eval()
    
    # Get token representations
    with torch.no_grad():
        results = self.ProteinMLM(batch_tokens, repr_layers=[self.num_mlm_layers])
        token_reps = results["representations"][self.num_mlm_layers]
    
    # Get attention weights from classification head
    _, attn_weights = self.TaskSpecificHead(token_reps, attention_mask, return_attention=True)
    
    # Convert to numpy if needed
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    
    batch_size = attn_weights.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, batch_size, figsize=(7 * batch_size, 6))
    if batch_size == 1:
        axes = [axes]  # Make iterable for single batch
    
    # For each sequence in batch
    for i in range(batch_size):
        # Get the length of the sequence (excluding padding)
        if isinstance(attention_mask, torch.Tensor):
            seq_len = attention_mask[i].sum().item()
        else:
            seq_len = attention_mask[i].sum()
            
        # If sequences were provided, use the actual length
        if sequences is not None and i < len(sequences):
            seq_len = min(seq_len, len(sequences[i]))
        
        # Extract attention matrix for this sequence, layer and head
        attn_matrix = attn_weights[i, layer_idx, head_idx, :seq_len, :seq_len]
        
        # Apply min-max normalization
        attn_min = attn_matrix.min()
        attn_max = attn_matrix.max()
        if attn_max > attn_min:  # Avoid division by zero
            attn_matrix_norm = (attn_matrix - attn_min) / (attn_max - attn_min)
        else:
            attn_matrix_norm = attn_matrix
        
        # Create heatmap with min-max normalized values
        sns.heatmap(
            attn_matrix_norm, 
            cmap='viridis', 
            ax=axes[i],
            vmin=0, vmax=1  # Explicitly set the range for normalized values
        )
        
        # Set labels
        axes[i].set_title(f"Sequence {i+1} - Layer {layer_idx+1}, Head {head_idx+1}\nMin: {attn_min:.4f}, Max: {attn_max:.4f}")
        axes[i].set_xlabel("Token Position")
        axes[i].set_ylabel("Token Position")
        
        # Add amino acid labels if sequences are provided
        if sequences is not None and i < len(sequences):
            seq = sequences[i][:seq_len]
            # Only add labels if sequence is not too long
            if seq_len <= 30:
                axes[i].set_xticks(np.arange(seq_len) + 0.5)
                axes[i].set_yticks(np.arange(seq_len) + 0.5)
                axes[i].set_xticklabels(list(seq))
                axes[i].set_yticklabels(list(seq))
    
    plt.tight_layout()
    return fig

def visualize_all_heads(self, batch_tokens, attention_mask, sequences=None, layer_idx=0):
    """
    Visualize attention matrices for all heads in a specific layer with min-max normalization.
    
    Parameters:
    - batch_tokens: Tokenized input sequences
    - attention_mask: Attention mask tensor
    - sequences: Original amino acid sequences (optional, for axis labels)
    - layer_idx: Layer index to visualize (default: 0)
    
    Returns:
    - Matplotlib figure
    """
    self.eval()
    
    # Get token representations
    with torch.no_grad():
        results = self.ProteinMLM(batch_tokens, repr_layers=[self.num_mlm_layers])
        token_reps = results["representations"][self.num_mlm_layers]
    
    # Get attention weights from classification head
    _, attn_weights = self.TaskSpecificHead(token_reps, attention_mask, return_attention=True)
    
    # Convert to numpy if needed
    if isinstance(attn_weights, torch.Tensor):
        attn_weights = attn_weights.detach().cpu().numpy()
    
    # Assuming we're visualizing the first sequence in the batch
    batch_idx = 0
    
    # Get number of heads
    num_heads = attn_weights.shape[2]
    
    # Get the length of the sequence (excluding padding)
    if isinstance(attention_mask, torch.Tensor):
        seq_len = attention_mask[batch_idx].sum().item()
    else:
        seq_len = attention_mask[batch_idx].sum()
        
    # If sequences were provided, use the actual length
    if sequences is not None and batch_idx < len(sequences):
        seq_len = min(seq_len, len(sequences[batch_idx]))
    
    # Create figure with subplots for each head
    nrows = 2
    ncols = (num_heads + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.flatten() if num_heads > 1 else [axes]
    
    for head_idx in range(num_heads):
        # Extract attention matrix for this head
        attn_matrix = attn_weights[batch_idx, layer_idx, head_idx, :seq_len, :seq_len]
        
        # Apply min-max normalization
        attn_min = attn_matrix.min()
        attn_max = attn_matrix.max()
        if attn_max > attn_min:  # Avoid division by zero
            attn_matrix_norm = (attn_matrix - attn_min) / (attn_max - attn_min)
        else:
            attn_matrix_norm = attn_matrix
        
        # Create heatmap with min-max normalized values
        sns.heatmap(
            attn_matrix_norm, 
            cmap='viridis', 
            ax=axes[head_idx],
            vmin=0, vmax=1  # Explicitly set the range for normalized values
        )
        
        # Set labels
        axes[head_idx].set_title(f"Head {head_idx+1} (Min: {attn_min:.4f}, Max: {attn_max:.4f})")
        
        # Only add axis labels to the leftmost and bottom plots
        if head_idx % ncols == 0:
            axes[head_idx].set_ylabel("Token Position")
        if head_idx >= (nrows - 1) * ncols:
            axes[head_idx].set_xlabel("Token Position")
    
    # If there's an odd number of heads, remove the last empty subplot
    if num_heads < nrows * ncols:
        for i in range(num_heads, nrows * ncols):
            fig.delaxes(axes[i])
    
    plt.suptitle(f"Attention Heads for Layer {layer_idx+1} (Min-Max Normalized)", y=0.98, fontsize=16)
    plt.tight_layout()
    return fig
