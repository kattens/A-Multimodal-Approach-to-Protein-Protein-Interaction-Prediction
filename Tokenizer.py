# Import required libraries
from transformers import BertTokenizer, BertModel

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define special tokens for amino acids and special tokens for entities
special_tokens = ['[ENTITY1]', '[ENTITY2]']

# Add special tokens to the tokenizer
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# Check if the special tokens were added successfully
print(f"Token '[ENTITY1]' has ID: {tokenizer.convert_tokens_to_ids('[ENTITY1]')}")
print(f"Token '[ENTITY2]' has ID: {tokenizer.convert_tokens_to_ids('[ENTITY2]')}")

# Initialize the BERT model
model = BertModel.from_pretrained('bert-base-uncased')
model.resize_token_embeddings(len(tokenizer))
print('Token embeddings resized to accommodate new tokens.')

# Helper function to convert numerical token IDs back to their textual representation
def ids_to_text(ids):
    return ' '.join(tokenizer.convert_ids_to_tokens(ids))

# Check the updated size of the tokenizer's vocabulary
print(f"Updated vocabulary size: {len(tokenizer)}")

if '[ENTITY1]' in tokenizer.get_vocab() and '[ENTITY2]' in tokenizer.get_vocab():
    print("[ENTITY1] and [ENTITY2] are in the tokenizer's vocabulary.")
else:
    print("[ENTITY1] and [ENTITY2] are NOT in the tokenizer's vocabulary.")
