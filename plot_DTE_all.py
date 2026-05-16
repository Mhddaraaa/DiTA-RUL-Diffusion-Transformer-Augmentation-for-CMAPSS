import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from config import config
from DTE_model.DTE_network import (
    Encoder, Decoder, TSHAE, DropBlockLatent, WaveletConvBlock, FourierBlock
)
from utils.loss import TotalLoss
from DTE_running import train_epoch, valid_epoch, get_dataset_score
from preprocessed_dataset import PreprocessedDatasetDTE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def encode_latent_from_npy(data_path, ruls_path, units_path, fd="FD001", batch_size=256):
    x_np = np.load(data_path).astype(np.float32)          # [N,L,F]
    rul_np = np.load(ruls_path).astype(np.float32)        # [N]
    unit_np = np.load(units_path).astype(np.int32)        # [N]

    assert len(x_np) == len(rul_np) == len(unit_np)

    model_path = os.path.join(config["output_dir"], f"best_vae_model_{fd}.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(config["output_dir"], "best_vae_model.pt")

    model = torch.load(model_path, map_location=device, weights_only=False).to(device).eval()

    N = x_np.shape[0]
    mus = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.from_numpy(x_np[start:end]).to(device)  # [B,L,F]

        y_hat, z, mean, log_var, x_hat = model(x)
        mus.append(mean.cpu())  # Better than Z for plot

    mu = torch.cat(mus, dim=0).numpy()  # [N, latent_dim]
    return mu, rul_np, unit_np


# ---------------- main ----------------
fds = ["FD001", "FD002", "FD003", "FD004"]

unit_map = {
    "FD001": 86,
    "FD002": 103,
    "FD003": 39,
    "FD004": 152
}
# unit_map = {
#     "FD001": 78,
#     "FD002": 193,
#     "FD003": 65,
#     "FD004": 167
# }

results = {}
global_rul_min = float("inf")
global_rul_max = float("-inf")

for fd in fds:
    prefix = f"{fd}_train_ws{config['window_size']}_step1"
    data_path  = f"output/preprocessed/preprocessed_data_{prefix}.npy"
    ruls_path  = f"output/preprocessed/preprocessed_ruls_{prefix}.npy"
    units_path = f"output/preprocessed/preprocessed_units_{prefix}.npy"

    mu, rul, unit_ids = encode_latent_from_npy(data_path, ruls_path, units_path, fd=fd)

    
    Z = (mu - mu.mean(axis=0, keepdims=True)) / (mu.std(axis=0, keepdims=True) + 1e-8)
    Z2 = PCA(n_components=2, random_state=123).fit_transform(Z)

    target_unit = unit_map.get(fd, np.random.choice(np.unique(unit_ids)))

    results[fd] = {"Z2": Z2, "rul": rul, "unit_ids": unit_ids, "target_unit": target_unit}
    global_rul_min = min(global_rul_min, rul.min())
    global_rul_max = max(global_rul_max, rul.max())

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
axes = axes.ravel()

norm = Normalize(vmin=global_rul_min, vmax=global_rul_max)
bg_cmap = plt.cm.viridis
fg_cmap = plt.cm.hot_r

sm_bg = ScalarMappable(norm=norm, cmap=bg_cmap)
sm_fg = ScalarMappable(norm=norm, cmap=fg_cmap)

for ax, fd in zip(axes, fds):
    Z2 = results[fd]["Z2"]
    rul = results[fd]["rul"]
    unit_ids = results[fd]["unit_ids"]
    target_unit = results[fd]["target_unit"]

    # optional flip for consistent visual orientation
    if fd in ("FD001"):
        x_all = -Z2[:, 0]
        y_all = -Z2[:, 1]
    elif fd in ("FD002"):
        x_all = -Z2[:, 0]
        y_all = Z2[:, 1]
    else:
        x_all =  Z2[:, 0]
        y_all =  Z2[:, 1]

    mask = (unit_ids == target_unit)

    ax.scatter(x_all, y_all, c=rul, cmap=bg_cmap, norm=norm, s=3, alpha=0.3)
    ax.scatter(
        x_all[mask], y_all[mask],
        c=rul[mask], cmap=fg_cmap, norm=norm,
        s=20, edgecolors="k", linewidths=0.3
    )
    ax.set_title(f"{fd} / Unit #{target_unit}")

for ax in axes[2:]:
    ax.set_xlabel("latent dim 1 (PCA)")
for ax in axes[::2]:
    ax.set_ylabel("latent dim 2 (PCA)")

fig.subplots_adjust(right=0.82)
fig.tight_layout(rect=[0, 0, 0.82, 1])

cax_bg = fig.add_axes([0.84, 0.15, 0.02, 0.7])
fig.colorbar(sm_bg, cax=cax_bg).set_label("RUL (cycles) – background")

cax_fg = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(sm_fg, cax=cax_fg).set_label("RUL (cycles) – highlighted unit")

plt.show()