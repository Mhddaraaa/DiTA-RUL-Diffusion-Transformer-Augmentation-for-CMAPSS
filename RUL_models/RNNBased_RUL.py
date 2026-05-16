import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, h):  # h: [B, L, D]
        # energy: [B, L, 1] -> weights: [B, L, 1]
        e = self.v(torch.tanh(self.w(h)))
        a = torch.softmax(e, dim=1)
        # pooled: [B, D]
        return (a * h).sum(dim=1)
    

class RNNRULPredictor(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, bidirectional=False):
        super(RNNRULPredictor, self).__init__()

        self.in_proj = nn.Sequential(
              nn.Linear(input_size, hidden_size),
              nn.LayerNorm(hidden_size),
              nn.Dropout(0.1),
          )
        
        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=bidirectional
        )
        
        rnn_out_dim = hidden_size * (2 if bidirectional else 1)
        self.pool = AttnPool(rnn_out_dim)
        self.norm = nn.LayerNorm(rnn_out_dim)

        self.fc1 = nn.Linear(rnn_out_dim, 16)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(16, 8)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.in_proj(x)
        out, _ = self.lstm1(x)

        out = self.pool(out)            # [B, H*dir]
        out = self.norm(out)

        out = self.fc1(out)
        out = F.gelu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = F.gelu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        return out
