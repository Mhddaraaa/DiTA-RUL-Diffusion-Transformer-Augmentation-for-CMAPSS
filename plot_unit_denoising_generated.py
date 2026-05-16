import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import config
from DTE_model.DTE_network import (
    Encoder, Decoder, TSHAE, DropBlockLatent, WaveletConvBlock, FourierBlock
)  # needed for torch.load
from Diffusion_model.Diff_network import (
    DiffWave, DiffusionEmbedding, ConditionerEmbedding, DiffusionTransformerBlock
)
from Diffusion_model.ddpm import Diffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREP_DIR = os.path.join(config["output_dir"], "preprocessed")

PLOT = {
    "T24": 5, "T30": 6, "T50": 7, "P30": 10, "Nf": 11, "Nc": 12,
    "Ps30": 14, "phi": 15, "NRf": 16, "NRc": 17, "BPR": 18,
    "htBleed": 20, "W31": 23, "W32": 24,
}
PLOT_NAMES = list(PLOT.keys())
PLOT_IDXS  = [PLOT[k] for k in PLOT_NAMES]

def center_stitch(windows: np.ndarray) -> np.ndarray:
    W, L, F = windows.shape
    T = W + L - 1
    c = L // 2

    full = np.zeros((T, F), dtype=np.float32)
    for i in range(W):
        full[i + c] = windows[i, c]

    full[:c] = windows[0, :c]
    full[c + W:] = windows[-1, c+1:]
    return full

def smooth_moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    """
    x: [T, F]
    k: odd window size (3 or 5 is usually perfect here)	Too noisy → k=7, smooth → k=3
    """
    assert k % 2 == 1 and k >= 3
    pad = k // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(k, dtype=np.float32) / k

    y = np.empty_like(x, dtype=np.float32)
    for f in range(x.shape[1]):
        y[:, f] = np.convolve(xpad[:, f], kernel, mode="valid")
    return y

def load_preprocessed(fd: str, split: str):
    prefix = f"{fd}_{split}_ws{config['window_size']}_step1"
    X = np.load(os.path.join(PREP_DIR, f"preprocessed_data_{prefix}.npy")).astype(np.float32)   # [N,L,25]
    A = np.load(os.path.join(PREP_DIR, f"preprocessed_alpha_{prefix}.npy")).astype(np.float32)  # [N]
    U = np.load(os.path.join(PREP_DIR, f"preprocessed_units_{prefix}.npy")).astype(np.int32)    # [N]
    return X, A, U

def load_models(fd: str):
    dte_path  = os.path.join(config["output_dir"], f"best_vae_model_{fd}.pt")
    diff_path = os.path.join(config["output_dir"], f"best_diff_model_{fd}.pt")
    dte: TSHAE = torch.load(dte_path, map_location=device, weights_only=False).to(device).eval()
    diff: DiffWave = torch.load(diff_path, map_location=device, weights_only=False).to(device).eval()
    diffusion = Diffusion(
        noise_steps=config["noise_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_name=config["schedule_name"],
        device=device,
    )
    return dte, diff, diffusion

def plot_unit_denoise(fd="FD001", split="train", target_unit=19, t_fixed=25, batch_size=64):
    X, A, U = load_preprocessed(fd, split)
    dte, diff_model, diffusion = load_models(fd)

    print(f"plot {fd} - unit: {target_unit}")

    mask = (U == target_unit)
    X_u = X[mask]   # [W,L,25] normalized
    A_u = A[mask]   # [W]
    if len(X_u) < 5:
        raise ValueError("Too few windows.")

    W = len(X_u)
    x0pred_list = []

    for s in range(0, W, batch_size):
        e = min(W, s + batch_size)
        xb = torch.from_numpy(X_u[s:e]).to(device)                                # [B,L,F]
        # ab = torch.from_numpy(np.clip(A_u[s:e], 1e-4, 1.0)).float().to(device)    # [B]
        ab = torch.from_numpy(A_u[s:e]).float().to(device) 
        with torch.no_grad():
            _, z, _, _, _ = dte(xb)

            # fixed timestep for all in batch (diagnostic)
            t = torch.full((xb.shape[0],), int(t_fixed), dtype=torch.long, device=device)

            x_t, noise = diffusion.noise_images(xb, t)
            pred_noise = diff_model(x_t, t, z, ab)
            x0_pred = diffusion.predict_x0(x_t, t, pred_noise)  # this SHOULD align better

        x0pred_list.append(x0_pred.cpu().numpy())

    X0P = np.concatenate(x0pred_list, axis=0)  # [W,L,25]

    # build full trajectories (normalized)
    real_full = center_stitch(X_u)
    pred_full = center_stitch(X0P)

    # pred_full = smooth_moving_average(pred_full, k=7)

    real_plot = real_full[:, PLOT_IDXS]
    pred_plot = pred_full[:, PLOT_IDXS]

    print("REAL min/max:", float(real_plot.min()), float(real_plot.max()))
    print("X0P  min/max:", float(pred_plot.min()), float(pred_plot.max()))

    t_axis = np.arange(real_plot.shape[0])

    cols = 2
    rows = int(np.ceil(len(PLOT_IDXS)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.2*rows), sharex=True)
    axes = axes.ravel()

    for i, name in enumerate(PLOT_NAMES):
        ax = axes[i]
        ax.plot(t_axis, real_plot[:, i], label="Real (norm)", linewidth=1.0)
        ax.plot(t_axis, pred_plot[:, i], label=f"Sampled (t={t_fixed})", linewidth=1.0)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)
        # ax.set_ylim(0.0, 1.0)
        # ax.set_ylim(-0.2, 1.5)

    for j in range(len(PLOT_IDXS), len(axes)):
        fig.delaxes(axes[j])

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-2].set_xlabel("Time index")
    axes[-1].set_xlabel("Time index")
    fig.suptitle(f"{fd} – Unit #{target_unit}: Real vs Sampled", y=0.995)
    plt.tight_layout()

    out = os.path.join(config["output_dir"], f"{fd}_unit{target_unit}_denoise_align_t{t_fixed}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

if __name__ == "__main__":
    fds = ["FD001", "FD002", "FD003", "FD004"]
    units = [99, 46, 1, 94]
    for fd, u in zip(fds, units):
        plot_unit_denoise(fd=fd, split="train", target_unit=u, t_fixed=25)