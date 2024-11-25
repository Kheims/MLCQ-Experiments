import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel, AutoModelForPreTraining

@dataclass
class SimpleLSTMConfig:
    vocab_size: int = 10000  
    embedding_dim: int = 100 
    hidden_dim: int = 128    
    num_layers: int = 1      
    dropout: float = 0.0     
    max_seq_len: int = 512   
    num_classes: int = 5     

class SimpleLSTM(nn.Module):
    def __init__(self, config: SimpleLSTMConfig):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(config.hidden_dim, config.num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[:, -1, :]  
        output = self.fc(last_hidden)
        return output

class SimpleBILSTM(nn.Module):
    def __init__(self, config: SimpleLSTMConfig):
        super(SimpleBILSTM, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(config.dropout)
        

        self.fc = nn.Linear(2*config.hidden_dim, config.num_classes)


    def forward(self, x):

        embedded = self.embedding(x)
        hidden_dim = self.lstm.hidden_size
        lstm_out, _ = self.lstm(embedded)
        last_hidden = torch.cat((lstm_out[:, -1, :hidden_dim],lstm_out[:,0,hidden_dim:]), dim=1)  
        output = self.dropout(last_hidden)
        output = self.fc(output)
        return output



class SimpleBILSTMAttn(nn.Module):
    def __init__(self, config: SimpleLSTMConfig):
        super(SimpleBILSTMAttn, self).__init__()
        
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=config.dropout)

        self.attention = nn.Linear(config.hidden_dim * 2, 1)

        self.fc = nn.Linear(2*config.hidden_dim, config.num_classes)

    def attention_net(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim * 2)
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # context vector shape : ((batch_size, seq_len,1) * (batch_size, seq_len, hidden_dim*2)) 
        #                           -> (batch_size, seq_len, hidden_dim*2)  'elem wise matmul'
        #          sum over seq_len -> (batch_size, hidden_dim*2) ie context vector repr of each snippet
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector 


    def forward(self, x):

        embedded = self.embedding(x)
        hidden_dim = self.lstm.hidden_size
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out) 
        attention_output = self.attention_net(lstm_out)
        output = self.fc(attention_output)
        return output
    
class SimpleBILSTMAttnWithCodeBERT(nn.Module):
    def __init__(self, config):
        super(SimpleBILSTMAttnWithCodeBERT, self).__init__()
        
        # Load CodeBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        self.embedding_layer = self.codebert.embeddings
        
        # Set the hidden size for LSTM based on CodeBERT's embedding size
        self.hidden_dim = config.hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.codebert.config.hidden_size,
            hidden_size=self.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=config.dropout)
        
        # Attention layer and fully connected layer
        self.attention = nn.Linear(self.hidden_dim * 2, 1)
        self.fc = nn.Linear(self.hidden_dim * 2, config.num_classes)

    def attention_net(self, lstm_output):
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x, attention_mask):
        # Tokenize and encode with CodeBERT
        #inputs = self.tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True, max_length=512)
        codebert_embeddings = self.embedding_layer(x)
        
        # Extract embeddings (last hidden state) from CodeBERT
        #embeddings = codebert_output.last_hidden_state  # (batch_size, seq_len, codebert_hidden_dim)
        
        # Pass embeddings through BiLSTM
        lstm_out, _ = self.lstm(codebert_embeddings)  # (batch_size, seq_len, hidden_dim * 2)

        lstm_out = self.dropout(lstm_out) 
        # Apply attention mechanism
        attention_output = self.attention_net(lstm_out)  # (batch_size, hidden_dim * 2)
        
        # Pass through the fully connected layer for classification
        output = self.fc(attention_output)  # (batch_size, num_classes)
        
        return output

class SimpleBILSTMAttnWithCuBERT(nn.Module):
    def __init__(self, config):
        super(SimpleBILSTMAttnWithCuBERT, self).__init__()
        
        tokenizer = AutoTokenizer.from_pretrained("claudios/cubert-20210711-Java-2048")
        model = AutoModelForPreTraining.from_pretrained("claudios/cubert-20210711-Java-2048")
        self.embedding_layer = self.codebert.embeddings
        
        # Set the hidden size for LSTM based on CodeBERT's embedding size
        self.hidden_dim = config.hidden_dim
        self.lstm = nn.LSTM(
            input_size=self.codebert.config.hidden_size,
            hidden_size=self.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(p=config.dropout)
        
        # Attention layer and fully connected layer
        self.attention = nn.Linear(self.hidden_dim * 2, 1)
        self.fc = nn.Linear(self.hidden_dim * 2, config.num_classes)

    def attention_net(self, lstm_output):
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector

    def forward(self, x, attention_mask):
        # Tokenize and encode with CodeBERT
        #inputs = self.tokenizer(code_snippets, return_tensors="pt", padding=True, truncation=True, max_length=512)
        codebert_embeddings = self.embedding_layer(x)
        
        # Extract embeddings (last hidden state) from CodeBERT
        #embeddings = codebert_output.last_hidden_state  # (batch_size, seq_len, codebert_hidden_dim)
        
        # Pass embeddings through BiLSTM
        lstm_out, _ = self.lstm(codebert_embeddings)  # (batch_size, seq_len, hidden_dim * 2)

        lstm_out = self.dropout(lstm_out) 
        # Apply attention mechanism
        attention_output = self.attention_net(lstm_out)  # (batch_size, hidden_dim * 2)
        
        # Pass through the fully connected layer for classification
        output = self.fc(attention_output)  # (batch_size, num_classes)
        
        return output