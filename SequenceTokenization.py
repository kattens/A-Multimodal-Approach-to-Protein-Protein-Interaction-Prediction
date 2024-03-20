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


""" TOKENIZATION OF AMINO ACIDS"""

# Define a dictionary for amino acid tokens
amino_acid_tokens = {'-': 0,'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                     'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
                     'UNK': 21, 'SEP': 22}

# Function to tokenize a single protein sequence
def tokenize_sequence(sequence, tokenizer=amino_acid_tokens):
    # Using .get() method with 'UNK' as default value for unknown amino acids
    return [tokenizer.get(aa, tokenizer['UNK']) for aa in sequence]

# Applying the tokenization function to each protein sequence in the DataFrame
df['Sequence Tokens'] = df['Sequence'].apply(tokenize_sequence)


"""CALCULATING CLOSEST amino acids(in the distance of 6-8 A)"""

# Updated function to calculate pairs within a specified distance
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
    if indices is None:
        return None
    masked_sequence = ['-' for _ in sequence]
    for index_set in indices:
        for idx in index_set:
            if 0 <= idx < len(sequence):
                masked_sequence[idx] = sequence[idx]
    return ''.join(masked_sequence)


pairs_list = []

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

# Add embeddings and sequences for 'File Name A'
pairs_df = pairs_df.merge(df[['File Name', 'Embeddings', 'Sequence']],
                          left_on='File Name A',
                          right_on='File Name',
                          how='left',
                          suffixes=('', '_A'))

# Add embeddings and sequences for 'File Name B'
pairs_df = pairs_df.merge(df[['File Name', 'Embeddings', 'Sequence']],
                          left_on='File Name B',
                          right_on='File Name',
                          how='left',
                          suffixes=('_A', '_B'))

# At this point, 'pairs_df' has two sets of embeddings and sequences columns
pairs_df.drop(columns=['File Name_A', 'File Name_B'], inplace=True)

# Tokenize the original and masked sequences for both A and B
pairs_df['tokenized_sequence_A'] = pairs_df['Sequence_A'].apply(tokenize_sequence)
pairs_df['tokenized_sequence_B'] = pairs_df['Sequence_B'].apply(tokenize_sequence)
pairs_df['tokenized_masked_sequence_A'] = pairs_df['masked_sequence_A'].apply(tokenize_sequence)
pairs_df['tokenized_masked_sequence_B'] = pairs_df['masked_sequence_B'].apply(tokenize_sequence)

# Define a function to sum tokenized values for each protein in the pair
def sum_tokenized_sequences(seq_tokens, masked_tokens):
    # Assuming both lists are of the same length, otherwise consider padding or other handling
    return [seq + masked for seq, masked in zip(seq_tokens, masked_tokens)]

# Apply the summation function for both A and B sequences and their masked versions
pairs_df['sum_tokenized_sequence_A'] = pairs_df.apply(lambda row: sum_tokenized_sequences(row['tokenized_sequence_A'], row['tokenized_masked_sequence_A']), axis=1)
pairs_df['sum_tokenized_sequence_B'] = pairs_df.apply(lambda row: sum_tokenized_sequences(row['tokenized_sequence_B'], row['tokenized_masked_sequence_B']), axis=1)


# Simulate the encoding of embeddings directly for demonstration purposes

# Assuming these are the tokenized and summed sequences for protein A and B from the second row
input_ids_a_example = torch.tensor(pairs_df.iloc[1]['sum_tokenized_sequence_A'], dtype=torch.long)
input_ids_b_example = torch.tensor(pairs_df.iloc[1]['sum_tokenized_sequence_B'], dtype=torch.long)

# Assuming these are the embeddings for protein A and B from the second row
embeddings_a_example = torch.tensor(pairs_df.iloc[1]['Embeddings_A'], dtype=torch.float)
embeddings_b_example = torch.tensor(pairs_df.iloc[1]['Embeddings_B'], dtype=torch.float)

# Directly do the encoding embeddings into a token-like format using a placeholder approach
encoded_embeddings_a_example = torch.arange(100, 105, dtype=torch.long)  # Simulated encoded embeddings for A
encoded_embeddings_b_example = torch.arange(105, 110, dtype=torch.long)  # Simulated encoded embeddings for B

# Concatenate the tokenized sequences and simulated encoded embeddings for the example input
combined_input_example = torch.cat([input_ids_a_example, input_ids_b_example, encoded_embeddings_a_example, encoded_embeddings_b_example])

