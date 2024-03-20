'''
# Install necessary libraries
!pip install faiss-gpu
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
!pip install torch-geometric

# Define the PyTorch and CUDA version for compatibility
import torch
TORCH_VERSION = torch.__version__.split('+')[0]  # get the PyTorch version
CUDA_VERSION = torch.version.cuda.replace('.', '')  # format the CUDA version correctly

# Install PyTorch Geometric dependencies
!pip install torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_VERSION}.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_VERSION}.html
!pip install torch-cluster -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_VERSION}.html
!pip install torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu{CUDA_VERSION}.html


'''

import pandas as pd
import numpy as np
import faiss
from itertools import combinations

df = pd.read_csv(r"C:\Users\KATT\Documents\ProteinInteraPredict\Modified_df.csv")


# Define a dictionary for amino acid tokens
amino_acid_tokens = {'-': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                     'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
                     'UNK': 21, 'SEP': 22}

# Function to tokenize a single protein sequence
def tokenize_sequence(sequence, tokenizer=amino_acid_tokens):
    # Using .get() method with 'UNK' as default value for unknown amino acids
    return [tokenizer.get(aa, tokenizer['UNK']) for aa in sequence]

# Applying the tokenization function to each protein sequence in the DataFrame
df['tokenized_sequence'] = df['Sequence'].apply(tokenize_sequence)



# Updated calculate_top_20_pairs function with validity checks for the coordinates
def calculate_top_20_pairs(coords1, coords2):
    if coords1 is None or coords2 is None or len(coords1) % 3 != 0 or len(coords2) % 3 != 0:
        return None
    try:
        coords1_np = np.array(coords1).reshape(-1, 3).astype('float32')
        coords2_np = np.array(coords2).reshape(-1, 3).astype('float32')
    except ValueError as e:
        print(f"Reshape error: {e}")
        return None
    index = faiss.IndexFlatL2(3)
    index.add(coords2_np)
    _, indices = index.search(coords1_np, 21)
    return indices[:, 1:]

# Function to mask sequences based on indices of closest amino acids
def mask_sequence(sequence, indices):
    if indices is None:  # Check if indices are None before proceeding
        return None
    masked_sequence = ['-' for _ in sequence]
    for index_set in indices:
        for idx in index_set:
            if 0 <= idx < len(sequence):
                masked_sequence[idx] = sequence[idx]
    return ''.join(masked_sequence)

# Prepare the DataFrame for pairwise comparison
# Extract 'pair_id' from 'File Name' assuming it can be identified as the first 4 characters
df['pair_id'] = df['File Name'].apply(lambda x: x[:4])  # Adjust according to your 'File Name' structure

pairs_list = []

# Generate pairwise comparisons for proteins within a group
for pair_id, group in df.groupby('pair_id'):
    if group.shape[0] > 1:  # Ensure group has more than one member for comparison
        for (idx1, row1), (idx2, row2) in combinations(group.iterrows(), 2):
            top_20_indices_1 = calculate_top_20_pairs(row1['Normalized Coordinates'], row2['Normalized Coordinates'])
            top_20_indices_2 = calculate_top_20_pairs(row2['Normalized Coordinates'], row1['Normalized Coordinates'])

            # Proceed only if valid indices were returned
            if top_20_indices_1 is not None and top_20_indices_2 is not None:
                masked_seq_1 = mask_sequence(row1['Sequence'], top_20_indices_1)
                masked_seq_2 = mask_sequence(row2['Sequence'], top_20_indices_2)

                pairs_list.append({
                  'pair_id': pair_id,
                  'File Name A': row1['File Name'],
                  'File Name B': row2['File Name'],
                  'masked_sequence_A': masked_seq_1,
                  'masked_sequence_B': masked_seq_2,
                  'coords_A': row1['Normalized Coordinates'],  # Adding initial coordinates for protein A
                  'coords_B': row2['Normalized Coordinates'],  # Adding initial coordinates for protein B
              })


# Creating a DataFrame from pairs_list to hold the results
pairs_df = pd.DataFrame(pairs_list)
