import torch
from torch import nn


class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size=1,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout_prob=0.3
    ):
        super().__init__()

        self.output_size = output_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(
            in_features=hidden_dim,
            out_features=output_size
        )

    def forward(self, x, hidden, cell):

        embedded = self.embedding(x)

        lstm_output, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell)
        )

        lstm_output = lstm_output[:, -1, :]

        output = self.dropout(lstm_output)
        output = self.fc(output)

        return output.squeeze(1), (hidden, cell)

    def init_hidden(self, batch_size, device):

        hidden = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim
        ).to(device)

        cell = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim
        ).to(device)

        return hidden, cell

        