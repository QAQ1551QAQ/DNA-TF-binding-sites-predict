import sys 
sys.path.append(r"../efficient_kan")
from src.efficient_kan import KAN
import torch.nn as nn
import torch
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.len_vocab, config.embed, padding_idx=config.len_vocab - 1)

        config.learning_rate = 0.001
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=100,
                      out_channels=64,
                      kernel_size=8,
                      stride=1,
                      padding=4),
            # nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=64,
                      out_channels=32,
                      kernel_size=8,
                      stride=1,
                      padding=4),
            # nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=32,
                      out_channels=16,
                      kernel_size=8,
                      stride=1,
                      padding=4),
            # nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )

        self.BiLSTM = nn.Sequential(
            nn.LSTM(input_size=25,
                    hidden_size=32,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    bias=True)
        )

        self.e_kan = KAN([64, 32, 2])

    def forward(self, x):
        # out = self.embedding(x[0]) # [batch_size, seq_len, embedding_dim]
        out = self.embedding(x).permute(0, 2, 1) # [batch_size, embedding_dim, seq_len]
        # CNN
        out = self.cnn(out) # [batch_size, seq_len, embedding_dim]
        # RNN-Att
        lstm_out,(H,C) = self.BiLSTM(out) #[batch_size, seq_len, embedding_dim]
        out = lstm_out[:, -1, :]
        # kan
        out = self.e_kan(out) 
        return out
