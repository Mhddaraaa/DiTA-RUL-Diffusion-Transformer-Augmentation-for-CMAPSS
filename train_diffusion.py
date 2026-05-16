import os
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import config
from utils.loss import DiffusionLoss
from preprocessed_dataset import PreprocessedDatasetDiffusion
from DTE_model.DTE_network import (
    Encoder, Decoder, TSHAE, DropBlockLatent, WaveletConvBlock, FourierBlock
)
from Diffusion_model.Diff_network import DiffWave
from Diffusion_model.ddpm import Diffusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_diffusion(fd):
    prefix = f"{fd}_train_ws{config['window_size']}_step1"
    data_path = f"output/preprocessed/preprocessed_data_{prefix}.npy"
    alpha_path = f"output/preprocessed/preprocessed_alpha_{prefix}.npy"

    # ---- make sure this matches your saved DTE file naming ----
    dte_path = os.path.join(config["output_dir"], f"best_vae_model_{fd}.pt")
    if not os.path.exists(dte_path):
        # fallback if you saved as best_vae_model.pt (single FD training)
        dte_path = os.path.join(config["output_dir"], "best_vae_model.pt")

    os.makedirs(config["output_dir"], exist_ok=True)
    diff_save = os.path.join(config["output_dir"], f"best_diff_model_{fd}.pt")

    # split
    N = np.load(data_path, mmap_mode="r").shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    n_val = int(0.1 * N)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    ds_tr = PreprocessedDatasetDiffusion(data_path, alpha_path, indices=tr_idx)
    ds_va = PreprocessedDatasetDiffusion(data_path, alpha_path, indices=val_idx)

    dl_tr = DataLoader(ds_tr, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=config["batch_size"], shuffle=False)

    # load DTE
    dte: TSHAE = torch.load(dte_path, map_location=device, weights_only=False).to(device)
    dte.eval()

    # diffusion
    diff_model = DiffWave(config).to(device)
    diffusion = Diffusion(
        noise_steps=config["noise_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_name=config["schedule_name"],
        device=device
    )

    opt = torch.optim.Adam(diff_model.parameters(), lr=config["lr"])
    criterion = DiffusionLoss(config)

    best_val = float("inf")

    pbar = tqdm(range(config["max_epochs"]), desc="Epoch")
    for epoch in pbar:
        diff_model.train()
        tr_loss = 0.0
        nb = 0

        for x, alpha in dl_tr:
            x = x.to(device)                 # [B,L,F]
            alpha = alpha.to(device).float() # [B]

            with torch.no_grad():
                _, z, _, _, _ = dte(x)        # z: [B, latent_dim]

            t = diffusion.sample_time_steps(x.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(x, t)

            pred_noise = diff_model(x_t, t, z, alpha)
            x0_pred = diffusion.predict_x0(x_t, t, pred_noise)

            loss_x0 = F.mse_loss(x0_pred, x)
            loss = criterion(pred_noise, noise, x0_pred, x) + 0.1 * loss_x0

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += float(loss.detach())
            nb += 1

        tr_loss /= max(nb, 1)

        # val (use only diffusion MSE for checkpointing)
        diff_model.eval()
        va = 0.0
        vb = 0
        with torch.no_grad():
            for x, alpha in dl_va:
                x = x.to(device)
                alpha = alpha.to(device).float()
                _, z, _, _, _ = dte(x)

                t = diffusion.sample_time_steps(x.shape[0]).to(device)
                x_t, noise = diffusion.noise_images(x, t)
                pred_noise = diff_model(x_t, t, z, alpha)
                loss_diff = F.mse_loss(pred_noise, noise)

                va += float(loss_diff)
                vb += 1

        va /= max(vb, 1)
        pbar.set_postfix({
            "train": f"{tr_loss:.4f}",
            "val_diff": f"{va:.4f}"
        })

        if va < best_val:
            best_val = va
            torch.save(diff_model, diff_save)
            print(f"[{fd}] saved BEST diffusion: {best_val:.4f} -> {diff_save}")


if __name__ == "__main__":
    # MUST match preprocessing + DTE input
    assert config["input_size"] == 25, "config['input_size'] must be 25 if you use cols 1..25"

    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        train_diffusion(fd)