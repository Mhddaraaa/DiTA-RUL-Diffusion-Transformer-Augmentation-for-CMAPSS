import argparse
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
VALID_FDS = ("FD001", "FD002", "FD003", "FD004")


def normalize_fds(fds):
    normalized = [fd.upper() for fd in fds]
    invalid = [fd for fd in normalized if fd not in VALID_FDS]
    if invalid:
        raise ValueError(f"Unsupported FD subset(s): {', '.join(invalid)}")
    return normalized


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_diffusion(
    fd,
    cfg=None,
    preprocessed_dir=None,
    step=1,
    valid_fraction=0.1,
    seed=42,
    num_workers=0,
):
    cfg = cfg or config
    preprocessed_dir = preprocessed_dir or os.path.join(cfg["output_dir"], "preprocessed")

    prefix = f"{fd}_train_ws{cfg['window_size']}_step{step}"
    data_path = os.path.join(preprocessed_dir, f"preprocessed_data_{prefix}.npy")
    alpha_path = os.path.join(preprocessed_dir, f"preprocessed_alpha_{prefix}.npy")

    for path in (data_path, alpha_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preprocessed file: {path}")

    dte_path = os.path.join(cfg["output_dir"], f"best_vae_model_{fd}.pt")
    if not os.path.exists(dte_path):
        dte_path = os.path.join(cfg["output_dir"], "best_vae_model.pt")
    if not os.path.exists(dte_path):
        raise FileNotFoundError(f"Missing DTE checkpoint: {dte_path}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    diff_save = os.path.join(cfg["output_dir"], f"best_diff_model_{fd}.pt")

    N = np.load(data_path, mmap_mode="r").shape[0]
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(valid_fraction * N)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    ds_tr = PreprocessedDatasetDiffusion(data_path, alpha_path, indices=tr_idx)
    ds_va = PreprocessedDatasetDiffusion(data_path, alpha_path, indices=val_idx)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dte: TSHAE = torch.load(dte_path, map_location=device, weights_only=False).to(device)
    dte.eval()

    diff_model = DiffWave(cfg).to(device)
    diffusion = Diffusion(
        noise_steps=cfg["noise_steps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        schedule_name=cfg["schedule_name"],
        device=device
    )

    opt = torch.optim.Adam(diff_model.parameters(), lr=cfg["lr"])
    criterion = DiffusionLoss(cfg)

    best_val = float("inf")

    pbar = tqdm(range(cfg["max_epochs"]), desc=f"Training diffusion ({fd})")
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the diffusion model for CMAPSS FD subsets.")
    parser.add_argument("--fds", nargs="+", default=list(VALID_FDS), help="FD subsets to train.")
    parser.add_argument("--output-dir", default=config["output_dir"], help="Directory for DTE and diffusion checkpoints.")
    parser.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Directory containing preprocessed .npy files. Defaults to OUTPUT_DIR/preprocessed.",
    )
    parser.add_argument("--window-size", type=int, default=int(config["window_size"]))
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=None, help="Override config['max_epochs'].")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config['batch_size'].")
    parser.add_argument("--lr", type=float, default=None, help="Override config['lr'].")
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser


def config_from_args(args):
    cfg = dict(config)
    cfg["output_dir"] = args.output_dir
    cfg["window_size"] = args.window_size
    if args.epochs is not None:
        cfg["max_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    return cfg


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    assert cfg["input_size"] == 25, "config['input_size'] must be 25 if you use cols 1..25"
    set_seed(args.seed)

    for fd in normalize_fds(args.fds):
        train_diffusion(
            fd,
            cfg=cfg,
            preprocessed_dir=args.preprocessed_dir,
            step=args.step,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
