import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class SequenceProcessor(nn.Module):
    def __init__(self, bert_model_path):
        super(SequenceProcessor, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)

    def forward(self, input_ids, attention_mask=None):
        # Pass sequences through BERT and return the last hidden states as features
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

class CustomAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CustomAttention, self).__init__()
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.context_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, global_features, local_features):
        # Generate keys, queries, and values for the attention mechanism
        keys = self.key_layer(global_features)
        queries = self.query_layer(local_features)
        values = self.value_layer(global_features)
        
        # Compute attention scores and weights
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_weights = self.softmax(attention_scores)
        
        # Context is weighted sum of values
        context = torch.matmul(attention_weights, values)
        processed_context = self.context_layer(context)
        return processed_context

class ProteinInteractionModel(nn.Module):
    def __init__(self, bert_model_path):
        super(ProteinInteractionModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize sequence processors for global and local sequences
        self.sequence_processor_global = SequenceProcessor(bert_model_path)
        self.sequence_processor_local = SequenceProcessor(bert_model_path)
        
        # Fetch hidden size from BERT model configuration
        hidden_size = BertModel.from_pretrained(bert_model_path).config.hidden_size

        # Attention module to integrate features from both channels
        self.custom_attention = CustomAttention(hidden_size)
        
        # MLM head for predicting masked tokens, typically using the vocab size from BERT configuration
        self.mlm_head = nn.Linear(hidden_size, BertModel.from_pretrained(bert_model_path).config.vocab_size)

        self.to(self.device)  # Move the model to the appropriate device

    def forward(self, input_ids_global, attention_mask_global, input_ids_local, attention_mask_local):
        # Process global and local sequences separately
        global_features = self.sequence_processor_global(input_ids_global, attention_mask_global)
        local_features = self.sequence_processor_local(input_ids_local, attention_mask_local)

        # Integrate features from global and local sequences using custom attention
        combined_features = self.custom_attention(global_features, local_features)

        # Output predictions for masked token positions in local sequences
        prediction_scores = self.mlm_head(combined_features)

        return prediction_scores
