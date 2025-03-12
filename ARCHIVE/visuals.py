import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot_aa_distribution(data, column_name, plot_title):
    
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    # Find amino acid counts
    counts = {'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0, 'E': 0, 'G': 0, 
               'H': 0, 'I': 0, 'L': 0, 'K': 0, 'M': 0, 'F': 0, 'P': 0, 
               'O': 0, 'S': 0, 'U': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0, 
               'X': 0}

    aa_code = {1:'A', 2:'R', 3:'N', 4:'D', 5:'C', 6:'Q', 7:'E', 8:'G', 
               9:'H', 10:'I', 11:'L', 12:'K', 13:'M', 14:'F', 15:'P', 
               16:'O', 17:'S', 18:'U', 19:'T', 20:'W', 21:'Y', 22:'V', 
               23:'X'}

    seqs = data[column_name]
    for seq in seqs:
        for aa in seq:
            counts[aa_code[aa]] += 1

    # Extract keys and values
    x = list(counts.keys())
    y = list(counts.values())

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    plt.bar(x, y, color='limegreen', edgecolor='black')
    plt.title(plot_title)
    plt.xlabel("Amino Acids")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_aa_distribution_by_group(data, column_name, group_column, plot_title):
    """
    Plots relative amino acid distributions for each group in the dataset with side-by-side bars.
    
    Parameters:
    - data: DataFrame containing the sequences and group information.
    - column_name: Column name containing amino acid sequences (either encoded or characters).
    - group_column: Column name containing group labels (e.g., 'resistant' vs 'susceptible').
    - plot_title: Title for the plot.
    """
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")
    if group_column not in data.columns:
        raise ValueError(f"Column '{group_column}' not found in dataframe.")

    # Amino acid codes mapping for decoding numeric sequences
    aa_code = {1: 'A', 2: 'R', 3: 'N', 4: 'D', 5: 'C', 6: 'Q', 7: 'E', 8: 'G', 
               9: 'H', 10: 'I', 11: 'L', 12: 'K', 13: 'M', 14: 'F', 15: 'P', 
               16: 'O', 17: 'S', 18: 'U', 19: 'T', 20: 'W', 21: 'Y', 22: 'V', 
               23: 'X'}
    
    # Amino acid counts template
    counts_template = {aa: 0 for aa in aa_code.values()}

    # Group the data by the group_column
    grouped_data = data.groupby(group_column)

    # Initialize plot
    plt.figure(figsize=(14, 8))
    group_colors = ['steelblue', 'orange']  # Define colors for each group
    bar_width = 0.35  # Width of the bars
    index = range(len(aa_code))  # Indices for the amino acids

    # Plot bars for each group side by side
    for idx, (group, group_df) in enumerate(grouped_data):
        # Initialize counts for this group
        counts = counts_template.copy()
        total_aa = 0  # To calculate relative frequencies

        # Calculate amino acid counts for this group
        seqs = group_df[column_name]
        for seq in seqs:
            for aa in seq:
                # Decode numeric amino acids to their single-letter codes, if necessary
                decoded_aa = aa_code[aa] if isinstance(aa, int) else aa
                counts[decoded_aa] += 1
                total_aa += 1

        # Normalize counts to relative frequencies
        relative_counts = {aa: count / total_aa for aa, count in counts.items()}

        # Extract x and y for plotting
        x = list(relative_counts.keys())
        y = list(relative_counts.values())

        # Offset the position of the bars for each group to be next to each other
        positions = [i + idx * bar_width for i in range(len(x))]

        # Plot bars for this group
        plt.bar(positions, y, alpha=0.7, color=group_colors[idx % len(group_colors)], 
                label=f"{group}", edgecolor='black', width=bar_width)

    # Adjust x-axis to match the amino acids
    plt.xticks([i + bar_width / 2 for i in range(len(aa_code))], aa_code.values())

    # Add plot details
    plt.title(plot_title)
    plt.xlabel("Amino Acids")
    plt.ylabel("Relative Frequency")
    plt.legend(title=group_column)
    plt.tight_layout()
    plt.show()
