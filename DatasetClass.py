import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import random
import numpy as np


class ProteinInteractionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, mask_probability=0.15):
        """
        Initializes the dataset.
        :param dataframe: Pandas DataFrame containing global and local sequences.
        :param tokenizer: Initialized BertTokenizer for sequence tokenization.
        :param max_length: Maximum sequence length for tokenization.
        :param mask_probability: Probability of masking a token in the local sequences.
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def __len__(self):
        """Returns the total number of items in the dataset."""
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Retrieves an item by index.
        :param idx: Index of the item.
        :return: A dictionary containing tokenized inputs for global sequences, and
                 tokenized and dynamically masked inputs along with labels for local sequences.
        """
        row = self.dataframe.iloc[idx]

        # Tokenize and concatenate global sequences
        global_seq = f"{row['Sequence_A']} [SEP] {row['Sequence_B']}"
        input_ids_global, attention_mask_global = self.tokenize_sequence(global_seq)

        # Dynamically mask local sequences and prepare labels
        local_seq = f"{row['masked_sequence_A']} [SEP] {row['masked_sequence_B']}"
        input_ids_local, attention_mask_local, labels_local = self.mask_and_tokenize_sequence(local_seq)

        return {
            "input_ids_global": input_ids_global,
            "attention_mask_global": attention_mask_global,
            "input_ids_local": input_ids_local,
            "attention_mask_local": attention_mask_local,
            "labels_local": labels_local,
        }

    def tokenize_sequence(self, sequence):
        """
        Tokenizes a sequence, respecting the max_length constraint.
        """
        encoded = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

    def mask_and_tokenize_sequence(self, sequence):
        """
        Masks tokens in a sequence with a specified probability and tokenizes the sequence.
        """
        tokens = self.tokenizer.tokenize(sequence)
        masked_tokens, labels = [], []

        for token in tokens:
            if random.random() < self.mask_probability:
                masked_tokens.append(self.tokenizer.mask_token)
                labels.append(self.tokenizer.convert_tokens_to_ids(token))
            else:
                masked_tokens.append(token)
                labels.append(-100)  # -100 is used to ignore these tokens in loss calculation

        # Convert list of tokens to IDs and truncate or pad as necessary
        encoded = self.tokenizer.encode_plus(
            masked_tokens,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)

        # Ensure labels are padded with -100 to match the length
        labels = labels[:self.max_length] + [-100] * (self.max_length - len(labels))

        return input_ids, attention_mask, torch.tensor(labels, dtype=torch.long)
