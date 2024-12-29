import sys 
sys.path.append(r"/home/hegd/efficient_kan")
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

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=128, kernel_size=11),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=9),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
        )

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=160, kernel_size=9),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=160, out_channels=160, kernel_size=1),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=160, out_channels=160, kernel_size=5),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=160, out_channels=256, kernel_size=8),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=100, out_channels=180, kernel_size=1),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=180, out_channels=256, kernel_size=8),
            nn.ReLU(True),nn.Dropout(0.2),
            nn.MaxPool1d(2),
        )

        self.BiLSTM = nn.Sequential(
            nn.LSTM(input_size=256,hidden_size=128,num_layers=2,batch_first=True,bidirectional=True,
                    bias=True,dropout = 0.2)
        )

        self.tanh1 = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        # self.w = nn.Parameter(torch.zeros(128 * 2))
        self.w = nn.Parameter(torch.normal(0,1,(1,128 * 2))).squeeze(0).to(config.device)
        self.norm = nn.LayerNorm(256)
        self.e_kan = KAN([256, 32, 2])
        #self.fc = nn.Sequential(nn.ReLU(),nn.Dropout(0.2),nn.Linear(128 * 2, 64),
        #                        nn.ReLU(),nn.Dropout(0.2),nn.Linear(64, 2)  
        #)

    def forward(self, x):
        # out = self.embedding(x[0]) # [batch_size, seq_len, embedding_dim]
        out = self.embedding(x).permute(0, 2, 1) # [batch_size, embedding_dim, seq_len]
        # CNN
        out1 = self.cnn1(self.dropout(out)).permute(0, 2, 1) # [batch_size, seq_len, embedding_dim]
        out2 = self.cnn2(self.dropout(out)).permute(0, 2, 1)
        out3 = self.cnn3(self.dropout(out)).permute(0, 2, 1)
        out = torch.sum(torch.cat([out1,out2,out3],1), 1).unsqueeze(1)
        # RNN-Att
        lstm_out,(H,C) = self.BiLSTM(out) #[batch_size, seq_len, embedding_dim]
        out = out + lstm_out
#         M = self.tanh1(out)  # [batch_size, seq_len, embedding_dim]
#         alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1) #[batch_size, seq_len, embedding_dim]
#         att_out = out * self.dropout(alpha)
#         out = out + att_out
        out = torch.sum(self.norm(out), 1) #[batch_size, embedding_dim]
        # FC
        #out = self.fc(out) #[batch_size, num_classes]
        out = self.e_kan(out)
        return out
