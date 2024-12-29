#conding=utf-8
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.len_vocab, config.embed, padding_idx=config.len_vocab - 1)
        self.Conv1 = nn.Conv1d(in_channels=100, out_channels=32, kernel_size=8)
        self.Maxpool = nn.MaxPool1d(2)
        self.Linear2 = nn.Linear(3040, 2)

    def forward(self, x):
        x = self.embedding(x) # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1) # [batch_size, embedding_dim, seq_len]
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.Linear2(x)
        return x
