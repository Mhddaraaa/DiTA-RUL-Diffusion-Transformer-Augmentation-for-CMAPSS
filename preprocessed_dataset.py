import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedDatasetDTE(Dataset):
    """
    For DTE training ONLY.
    Returns:
      - no pairs: (x, y)
      - pairs: (x, pos_x, neg_x, y)
    x: [L, F]
    y: [1]
    """
    def __init__(self, data_path, ruls_path, return_pairs=False, indices=None):
        self.data = np.load(data_path).astype(np.float32)   # [N,L,F]
        self.ruls = np.load(ruls_path).astype(np.float32)   # [N]
        self.return_pairs = return_pairs

        if indices is not None:
            self.data = self.data[indices]
            self.ruls = self.ruls[indices]

        assert len(self.data) == len(self.ruls)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])  # [L,F]
        y = torch.tensor(self.ruls[idx], dtype=torch.float32).view(1)  # [1]

        if self.return_pairs:
            pos_idx, neg_idx = self._get_pairs(idx)
            pos_x = torch.from_numpy(self.data[pos_idx])
            neg_x = torch.from_numpy(self.data[neg_idx])
            return x, pos_x, neg_x, y

        return x, y

    def _get_pairs(self, idx):
        rul = self.ruls[idx]
        similar = np.where(np.abs(self.ruls - rul) <= 5)[0]
        dissimilar = np.where((np.abs(self.ruls - rul) > 5) & (np.abs(self.ruls - rul) <= 15))[0]

        pos_idx = int(np.random.choice(similar)) if len(similar) > 0 else int(idx)
        neg_idx = int(np.random.choice(dissimilar)) if len(dissimilar) > 0 else int(idx)
        return pos_idx, neg_idx


class PreprocessedDatasetDiffusion(Dataset):
    """
    For diffusion training.
    Returns: (x, alpha)
    x: [L,F]
    alpha: scalar float tensor
    """
    def __init__(self, data_path, alpha_path, indices=None):
        self.data = np.load(data_path).astype(np.float32)
        self.alpha = np.load(alpha_path).astype(np.float32)

        assert len(self.data) == len(self.alpha)

        if indices is not None:
            self.data = self.data[indices]
            self.alpha = self.alpha[indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])  # [L,F]
        a = torch.tensor(self.alpha[idx], dtype=torch.float32)  # scalar
        return x, a