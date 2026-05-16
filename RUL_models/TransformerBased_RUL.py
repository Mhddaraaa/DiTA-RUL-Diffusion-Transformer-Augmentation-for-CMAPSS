import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]

class TransformerRULPredictor(nn.Module):
    def __init__(
        self,
        input_dim=17,
        seq_len=30,
        d_model=64,
        num_heads=4,
        num_layers=2,
        ff_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        self.in_ln = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        self.drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # attention pooling (learned)
        self.pool = nn.Sequential(
            nn.Linear(d_model, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, L, input_dim]
        x = self.in_proj(x)
        x = self.in_ln(x)
        x = self.pos(x)
        x = self.drop(x)

        x = self.encoder(x)  # [B, L, D]

        # attention pooling over time
        w = torch.softmax(self.pool(x).squeeze(-1), dim=1)  # [B, L]
        x = torch.sum(x * w.unsqueeze(-1), dim=1)           # [B, D]

        return self.head(x)
