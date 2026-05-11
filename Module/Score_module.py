import torch
import torch.nn as nn

class CNNRegressionModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_filters=100, kernel_sizes=[3, 4, 5], dropout=0.1):
        super(CNNRegressionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(hidden_dim, num_filters, k) for k in kernel_sizes])
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        x = embedded.permute(0, 2, 1)
        conved = [self.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        flattened = torch.cat(pooled, dim=1)
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output.squeeze(-1)


class CNNClassificationModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_filters=100, kernel_sizes=[3, 4, 5], num_classes=2, dropout=0.1):
        super(CNNClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(hidden_dim, num_filters, k) for k in kernel_sizes])
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        x = embedded.permute(0, 2, 1)
        conved = [self.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        flattened = torch.cat(pooled, dim=1)
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output


class LSTMRegressionModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_layers=3, dropout=0.1):
        super(LSTMRegressionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        flattened = lstm_out.reshape(-1, self.max_length * lstm_out.size(2))
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output.squeeze(-1)


class LSTMClassificationModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_layers=3, num_classes=2, dropout=0.1):
        super(LSTMClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        flattened = lstm_out.reshape(-1, self.max_length * lstm_out.size(2))
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output


class GRURegressionModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_layers=3, dropout=0.1):
        super(GRURegressionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        gru_out, _ = self.gru(embedded)
        flattened = gru_out.reshape(-1, self.max_length * gru_out.size(2))
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output.squeeze(-1)


class GRUClassificationModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_layers=3, num_classes=2, dropout=0.1):
        super(GRUClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        gru_out, _ = self.gru(embedded)
        flattened = gru_out.reshape(-1, self.max_length * gru_out.size(2))
        x = self.relu(self.fc1(flattened))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output


class TransformerRegressionModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_heads=8, num_layers=6, dropout=0.1):
        super(TransformerRegressionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, hidden_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = embedded + self.positional_encoding[:, :embedded.size(1), :]
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        transformer_out = self.transformer_encoder(embedded.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.transpose(0, 1).reshape(-1, self.max_length * transformer_out.size(2))
        x = self.relu(self.fc1(transformer_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output.squeeze(-1)


class TransformerClassificationModel(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_dim=128, num_heads=8, num_layers=6, num_classes=2, dropout=0.1):
        super(TransformerClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_length, hidden_dim))
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = embedded + self.positional_encoding[:, :embedded.size(1), :]
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        transformer_out = self.transformer_encoder(embedded.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.transpose(0, 1).reshape(-1, self.max_length * transformer_out.size(2))
        x = self.relu(self.fc1(transformer_out))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        output = self.fc3(x)
        return output