import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming 'df' is your DataFrame containing the data

# Splitting the dataset into 80% training and 20% testing
train_df, test_df = train_test_split(pairs_df, test_size=0.2, random_state=42)

# Example usage, showing the first few rows of the train and test sets
print("Training set preview:")
print(train_df.head())

print("\nTest set preview:")
print(test_df.head())

# The lengths of the train and test sets
print(f"\nSize of training set: {len(train_df)} rows")
print(f"Size of test set: {len(test_df)} rows")


#the most important part to check if the class definition and data management is correctly working
from torch.utils.data import DataLoader

# Assuming ProteinInteractionDataset is implemented to handle your DataFrame structure
train_dataset = ProteinInteractionDataset(train_df, tokenizer)
test_dataset = ProteinInteractionDataset(test_df, tokenizer)

#since the model isnt runnint we reduced the batch size to half
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)


import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam

# Define your custom ProteinInteractionModel here (for context)
class ProteinInteractionModel(nn.Module):
    def __init__(self, bert_model_path):
        super(ProteinInteractionModel, self).__init__()
        # Example initialization; replace with your actual model details
        self.bert_global = BertModel.from_pretrained(bert_model_path)
        self.bert_local = BertModel.from_pretrained(bert_model_path)
        # Your model's additional layers and components here

    def forward(self, input_ids_global, attention_mask_global, input_ids_local, attention_mask_local):
        # Your forward pass definition here
        pass
        # Example forward pass; replace with your actual logic
        # return some output compatible with your loss function

# Initialization and setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
special_tokens_dict = {'additional_special_tokens': ['[ENTITY1]', '[ENTITY2]']}
tokenizer.add_special_tokens(special_tokens_dict)

# Initialize your ProteinInteractionModel with the path to a pre-trained BERT model
model = ProteinInteractionModel('bert-base-uncased')
model.to(device)

# Assuming your DataLoader, train_loader, is defined elsewhere and correctly set up
optimizer = Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
scaler = GradScaler()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids_global, attention_mask_global, input_ids_local, attention_mask_local, labels = (
            batch['input_ids_global'].to(device),
            batch['attention_mask_global'].to(device),
            batch['input_ids_local'].to(device),
            batch['attention_mask_local'].to(device),
            batch['labels_local'].to(device),
        )

        with autocast():
            prediction_scores = model(input_ids_global, attention_mask_global, input_ids_local, attention_mask_local)
            loss = loss_fn(prediction_scores.view(-1, model.config.vocab_size), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")
