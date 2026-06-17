import argparse
import os
import tempfile

_MPLCONFIGDIR = os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.makedirs(_MPLCONFIGDIR, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import config
from DTE_model.DTE_network import (
    Decoder,
    DropBlockLatent,
    Encoder,
    FourierBlock,
    TSHAE,
    WaveletConvBlock,
)
from Diffusion_model.Diff_network import (
    ConditionerEmbedding,
    DiffusionEmbedding,
    DiffusionTransformerBlock,
    DiffWave,
)
from Diffusion_model.ddpm import Diffusion


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_FDS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_UNITS = {
    "FD001": 99,
    "FD002": 46,
    "FD003": 1,
    "FD004": 94,
}

PLOT = {
    "T24": 5,
    "T30": 6,
    "T50": 7,
    "P30": 10,
    "Nf": 11,
    "Nc": 12,
    "Ps30": 14,
    "phi": 15,
    "NRf": 16,
    "NRc": 17,
    "BPR": 18,
    "htBleed": 20,
    "W31": 23,
    "W32": 24,
}
PLOT_NAMES = list(PLOT.keys())
PLOT_IDXS = [PLOT[k] for k in PLOT_NAMES]


def normalize_fds(fds):
    normalized = [fd.upper() for fd in fds]
    invalid = [fd for fd in normalized if fd not in VALID_FDS]
    if invalid:
        raise ValueError(f"Unsupported FD subset(s): {', '.join(invalid)}")
    return normalized


def parse_unit_map(value):
    if not value:
        return {}

    parsed = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            fd, unit = item.split("=", 1)
        elif ":" in item:
            fd, unit = item.split(":", 1)
        else:
            raise ValueError(f"Invalid unit-map item: {item}")
        parsed[fd.strip().upper()] = int(unit)
    return parsed


def choose_units(fds, units=None, unit_map_text=None):
    unit_map = dict(DEFAULT_UNITS)
    unit_map.update(parse_unit_map(unit_map_text))

    if units is not None:
        if len(units) != len(fds):
            raise ValueError("--units must contain one unit number for each FD subset.")
        unit_map.update({fd: int(unit) for fd, unit in zip(fds, units)})

    return unit_map


def center_stitch(windows: np.ndarray) -> np.ndarray:
    W, L, F = windows.shape
    T = W + L - 1
    c = L // 2

    full = np.zeros((T, F), dtype=np.float32)
    for i in range(W):
        full[i + c] = windows[i, c]

    full[:c] = windows[0, :c]
    full[c + W:] = windows[-1, c + 1:]
    return full


def smooth_moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Apply a per-feature moving average to a stitched trajectory."""
    assert k % 2 == 1 and k >= 3
    pad = k // 2
    xpad = np.pad(x, ((pad, pad), (0, 0)), mode="reflect")
    kernel = np.ones(k, dtype=np.float32) / k

    y = np.empty_like(x, dtype=np.float32)
    for f in range(x.shape[1]):
        y[:, f] = np.convolve(xpad[:, f], kernel, mode="valid")
    return y


def load_preprocessed(fd: str, split: str, preprocessed_dir=None, step: int = 1):
    preprocessed_dir = preprocessed_dir or os.path.join(config["output_dir"], "preprocessed")
    prefix = f"{fd}_{split}_ws{config['window_size']}_step{step}"
    data_path = os.path.join(preprocessed_dir, f"preprocessed_data_{prefix}.npy")
    alpha_path = os.path.join(preprocessed_dir, f"preprocessed_alpha_{prefix}.npy")
    units_path = os.path.join(preprocessed_dir, f"preprocessed_units_{prefix}.npy")

    for path in (data_path, alpha_path, units_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preprocessed file: {path}")

    X = np.load(data_path).astype(np.float32)
    A = np.load(alpha_path).astype(np.float32)
    U = np.load(units_path).astype(np.int32)
    return X, A, U


def load_models(fd: str):
    dte_path = os.path.join(config["output_dir"], f"best_vae_model_{fd}.pt")
    diff_path = os.path.join(config["output_dir"], f"best_diff_model_{fd}.pt")

    if not os.path.exists(dte_path):
        raise FileNotFoundError(f"Missing DTE checkpoint: {dte_path}")
    if not os.path.exists(diff_path):
        raise FileNotFoundError(f"Missing diffusion checkpoint: {diff_path}")

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


def plot_unit_denoise(
    fd="FD001",
    split="train",
    target_unit=19,
    t_fixed=25,
    batch_size=64,
    step=1,
    preprocessed_dir=None,
    output_dir=None,
    show=False,
):
    X, A, U = load_preprocessed(fd, split, preprocessed_dir=preprocessed_dir, step=step)
    dte, diff_model, diffusion = load_models(fd)

    print(f"Plotting {fd} unit {target_unit}")

    mask = U == target_unit
    X_u = X[mask]
    A_u = A[mask]
    if len(X_u) < 5:
        raise ValueError(f"Too few windows for {fd} unit {target_unit}.")

    x0pred_list = []
    for start in range(0, len(X_u), batch_size):
        end = min(start + batch_size, len(X_u))
        xb = torch.from_numpy(X_u[start:end]).to(device)
        ab = torch.from_numpy(A_u[start:end]).float().to(device)

        with torch.no_grad():
            _, z, _, _, _ = dte(xb)
            t = torch.full((xb.shape[0],), int(t_fixed), dtype=torch.long, device=device)
            x_t, _ = diffusion.noise_images(xb, t)
            pred_noise = diff_model(x_t, t, z, ab)
            x0_pred = diffusion.predict_x0(x_t, t, pred_noise)

        x0pred_list.append(x0_pred.cpu().numpy())

    X0P = np.concatenate(x0pred_list, axis=0)

    real_full = center_stitch(X_u)
    pred_full = center_stitch(X0P)

    real_plot = real_full[:, PLOT_IDXS]
    pred_plot = pred_full[:, PLOT_IDXS]

    print("REAL min/max:", float(real_plot.min()), float(real_plot.max()))
    print("X0P  min/max:", float(pred_plot.min()), float(pred_plot.max()))

    t_axis = np.arange(real_plot.shape[0])
    cols = 2
    rows = int(np.ceil(len(PLOT_IDXS) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.2 * rows), sharex=True)
    axes = axes.ravel()

    for i, name in enumerate(PLOT_NAMES):
        ax = axes[i]
        ax.plot(t_axis, real_plot[:, i], label="Real (norm)", linewidth=1.0)
        ax.plot(t_axis, pred_plot[:, i], label=f"Sampled (t={t_fixed})", linewidth=1.0)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(len(PLOT_IDXS), len(axes)):
        fig.delaxes(axes[j])

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-2].set_xlabel("Time index")
    axes[-1].set_xlabel("Time index")
    fig.suptitle(f"{fd} - Unit #{target_unit}: Real vs Sampled", y=0.995)
    plt.tight_layout()

    output_dir = output_dir or config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"{fd}_unit{target_unit}_denoise_align_t{t_fixed}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("Saved:", out)
    return out


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot real vs diffusion-denoised trajectories for selected units.")
    parser.add_argument("--fds", nargs="+", default=list(VALID_FDS), help="FD subsets to plot.")
    parser.add_argument("--output-dir", default=config["output_dir"], help="Directory containing model checkpoints.")
    parser.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Directory containing preprocessed .npy files. Defaults to OUTPUT_DIR/preprocessed.",
    )
    parser.add_argument("--plot-dir", default=None, help="Directory for saved PNG files. Defaults to OUTPUT_DIR.")
    parser.add_argument("--window-size", type=int, default=int(config["window_size"]))
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--t-fixed", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--units", nargs="+", type=int, default=None, help="Unit numbers, one per FD.")
    parser.add_argument("--unit-map", default=None, help="FD/unit pairs, for example FD001=99,FD002=46.")
    parser.add_argument("--show", action="store_true", help="Open each plot window after saving.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config["output_dir"] = args.output_dir
    config["window_size"] = args.window_size

    fds = normalize_fds(args.fds)
    unit_map = choose_units(fds, units=args.units, unit_map_text=args.unit_map)

    for fd in fds:
        plot_unit_denoise(
            fd=fd,
            split=args.split,
            target_unit=unit_map[fd],
            t_fixed=args.t_fixed,
            batch_size=args.batch_size,
            step=args.step,
            preprocessed_dir=args.preprocessed_dir,
            output_dir=args.plot_dir,
            show=args.show,
        )


if __name__ == "__main__":
    main()
