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

        self.BiLSTM = nn.Sequential(
            nn.LSTM(input_size=100,hidden_size=50,num_layers=2,batch_first=True,bidirectional=True,
                    bias=True,dropout = 0.2)
        )

        self.tanh1 = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        # self.w = nn.Parameter(torch.zeros(128 * 2))
        self.w = nn.Parameter(torch.normal(0,1,(1,50 * 2))).squeeze(0).to(config.device)
        self.norm = nn.LayerNorm(100)
        self.e_kan = KAN([100, 32, 2])

    def forward(self, x):
        out = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        out = torch.sum(out, 1).unsqueeze(1)
        #print("out.shape",out.shape)
        # RNN
        lstm_out,(H,C) = self.BiLSTM(out) #[batch_size, seq_len, embedding_dim]
        out = out + lstm_out
        # Att
        M = self.tanh1(out)  # [batch_size, seq_len, embedding_dim]
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1) #[batch_size, seq_len, embedding_dim]
        att_out = out * self.dropout(alpha)
        out = out + att_out
        out = torch.sum(self.norm(out), 1) #[batch_size, embedding_dim]
        # KAN
        out = self.e_kan(out)
        return out
