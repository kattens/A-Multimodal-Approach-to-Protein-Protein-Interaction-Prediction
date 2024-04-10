'''
THE FIRST THING IS TO SETUP YOUR ENVIROMENT WITH THE NECCESSARY LIBRARIES

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
import torch

df = pd.read_csv(r"C:\Users\KATT\Documents\ProteinInteraPredict\Modified_df.csv")
# Define a dictionary for amino acid tokens
amino_acid_tokens = {'-': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                     'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
                     'UNK': 21, 'SEP': 22} 


""" DEFINE ALL THE FUNCTIONS HERE """

#tokenizer function
def tokenize_sequence(sequence, tokenizer=amino_acid_tokens):
    # Using .get() method with 'UNK' as default value for unknown amino acids
    return [tokenizer.get(aa, tokenizer['UNK']) for aa in sequence]


#distance calculation with faiss function
def calculate_pairs_within_distance(coords1, coords2, min_dist=6, max_dist=8):
    if coords1 is None or coords2 is None:
        return None
    coords1_np = np.array(coords1).reshape(-1, 3).astype('float32')
    coords2_np = np.array(coords2).reshape(-1, 3).astype('float32')
    index = faiss.IndexFlatL2(3)
    index.add(coords2_np)
    squared_min_dist = min_dist ** 2
    squared_max_dist = max_dist ** 2
    distances, indices = index.search(coords1_np, len(coords2_np))
    filtered_indices = []
    for dist, idx in zip(distances, indices):
        mask = (dist >= squared_min_dist) & (dist <= squared_max_dist)
        filtered_indices.append(idx[mask])
    return filtered_indices

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

# Normalize coordinates 
def standard_scale(coords):
    """Scale coordinates to have mean 0 and std 1."""
    coords = np.array(coords)
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    standardized_coords = (coords - mean) / std
    return standardized_coords.tolist()



df['Sequence Tokens'] = df['Sequence'].apply(tokenize_sequence) # Applying the tokenization Function
df['pair_id'] = df['File Name'].apply(lambda x: x[:4])  # Adjust according to your 'File Name' structure
df['Normalized Coordinates'] = df['Parsed Coordinates'].apply(lambda x: standard_scale(x) if x is not None else None) # Apply normalization to each row of the DataFrame

pairs_list = []


















