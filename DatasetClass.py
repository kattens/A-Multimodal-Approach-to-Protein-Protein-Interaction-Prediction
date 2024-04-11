"""
This block defines a custom dataset class, `ProteinInteractionDataset`, 
for use in machine learning models that process protein sequences. It is built upon PyTorch's 
`Dataset` class and utilizes the BERT tokenizer from the `transformers` library for sequence tokenization. 
The class is initialized with a pandas DataFrame containing protein sequences, a BERT tokenizer, a maximum
 sequence length, and a masking probability for tokens. The dataset supports indexing to retrieve tokenized
   and optionally masked protein sequences, which are prepared in a format suitable for training 
   transformer models. Specifically, it includes methods to tokenize global sequences directly and to 
   both tokenize and apply dynamic masking to local sequences based on a specified probability. 
   The result is a dictionary containing input IDs, attention masks, and labels for local sequences, 
   where labels are used to indicate the original tokens that were masked (facilitating tasks like masked
     language modeling). This setup is particularly designed for tasks that require understanding the 
     interactions between protein sequences through models like BERT, which can benefit from both 
     concatenated sequence inputs and randomly masked training techniques.

"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np

class ProteinInteractionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, mask_probability=0.15):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Global sequence with random masking
        global_seq = f"[ENTITY1] {row['Sequence_A']} [SEP] [ENTITY2] {row['Sequence_B']}"
        input_ids_global, attention_mask_global, labels_global = self.random_mask_sequence(global_seq)

        # Local sequence without masking
        local_seq = f"[ENTITY1] {row['Sequence_A']} [SEP] [ENTITY2] {row['Sequence_B']}"
        input_ids_local, attention_mask_local = self.tokenize_sequence(local_seq)

        return {
            "input_ids_global": torch.tensor(input_ids_global),
            "attention_mask_global": torch.tensor(attention_mask_global),
            "labels_global": torch.tensor(labels_global),
            "input_ids_local": torch.tensor(input_ids_local),
            "attention_mask_local": torch.tensor(attention_mask_local)
        }

    def tokenize_sequence(self, sequence):
        encoded = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            return_tensors='pt',
            padding=False,
            truncation=True
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        return input_ids.numpy(), attention_mask.numpy()

    def random_mask_sequence(self, sequence):
        """Applies random masking to a sequence as in BERT's pretraining."""
        tokens = self.tokenizer.tokenize(sequence)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Create a mask array
        labels = np.full(len(input_ids), -100, dtype=int)
        mask_indices = np.random.rand(len(input_ids)) < self.mask_probability
        labels[mask_indices] = input_ids[mask_indices]
        input_ids[mask_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return input_ids, np.ones_like(input_ids), labels  # Attention mask is simply all ones



    """
    This is the collate func for handling the len of the sequences when doing batch processing
    """

def collate_fn(batch):
    # Determine the maximum length in this batch for dynamic padding
    # We calculate max length considering both global and local input IDs
    max_length = max(max(len(item['input_ids_global']), len(item['input_ids_local'])) for item in batch)
    
    # Initialize a dictionary to hold the padded versions of our batch data
    padded_batch = {}

    # Iterate over each key in the items of the batch; these keys represent different tensor types
    for key in ['input_ids_global', 'attention_mask_global', 'labels_global', 'input_ids_local', 'attention_mask_local']:
        # Create a padded version of each tensor type in the batch
        padded_vector = [
            torch.cat([item[key],  # Original tensor
                       torch.full((max_length - len(item[key]),),  # Padding to max length
                                  fill_value=0 if 'mask' in key or 'input_ids' in key else -100)])  # Padding value
            for item in batch  # For each item in the batch
        ]

        # Convert the list of tensors to a single tensor
        padded_batch[key] = torch.stack(padded_vector)

    # Return the padded batch, which now contains tensors of equal length
    return padded_batch
