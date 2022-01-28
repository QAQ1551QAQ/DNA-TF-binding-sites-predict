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

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=100,out_channels=64,kernel_size=16,padding=8),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=64,out_channels=32,kernel_size=16,padding=8),
            nn.ReLU(True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=32,out_channels=16,kernel_size=16,padding=8),
            nn.ReLU(True),
            nn.MaxPool1d(2),
        )

        self.BiLSTM = nn.Sequential(
            nn.LSTM(input_size=16,hidden_size=32,num_layers=1,batch_first=True,bidirectional=True,
                    bias=True,dropout = 0.1)
        )

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Sequential(nn.Linear(32 * 2, 32),nn.ReLU(),nn.Dropout(0.2),nn.Linear(32, 2))

    def forward(self, x):
        # out = self.embedding(x[0]) # [batch_size, seq_len, embedding_dim]
        out = self.embedding(x[0]).permute(0, 2, 1) # [batch_size, embedding_dim, seq_len]
        # CNN
        out = self.cnn(out).permute(0, 2, 1) # [batch_size, seq_len, embedding_dim]
        # RNN
        lstm_out,(H,C) = self.BiLSTM(out) #[batch_size, seq_len, embedding_dim]
        out = lstm_out[:, -1, :]
        # FC
        out = self.fc(out) 
        return out
