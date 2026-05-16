import torch
import torch.nn as nn
import torch.nn.functional as F
    

class HybridRULPredictor(nn.Module):
    """
    True CNN→LSTM hybrid.
    Input:  x [B, L, 17]  (L = window length)
    Conv1d: operates on [B, 17, L]
    LSTM:   operates on [B, T, C] where T is downsampled time length
    Output: [B, 1]
    """
    def __init__(self, in_channels=17, hidden_size=64, dropout=0.1):
        super().__init__()

        # Use k=3 with padding=1 to preserve length before pooling
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # LSTM over time: we treat conv channels as features
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.Dropout(dropout),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # [B, C=64, T]
        x = x.permute(0, 2, 1)  # [B, T, 64] for LSTM

        out, _ = self.lstm(x)   # [B, T, hidden]
        out = out[:, -1, :]     # last time step
        return self.head(out)

