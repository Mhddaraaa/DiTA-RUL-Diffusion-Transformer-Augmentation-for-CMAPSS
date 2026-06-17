import argparse
import os
import tempfile

_MPLCONFIGDIR = os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
os.makedirs(_MPLCONFIGDIR, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from sklearn.decomposition import PCA

from config import config
from DTE_model.DTE_network import (
    Decoder,
    DropBlockLatent,
    Encoder,
    FourierBlock,
    TSHAE,
    WaveletConvBlock,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_FDS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_UNIT_MAP = {
    "FD001": 86,
    "FD002": 103,
    "FD003": 39,
    "FD004": 152,
}


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
        fd = fd.strip().upper()
        parsed[fd] = int(unit)
    return parsed


def choose_units(fds, units=None, unit_map_text=None):
    unit_map = dict(DEFAULT_UNIT_MAP)
    unit_map.update(parse_unit_map(unit_map_text))

    if units is not None:
        if len(units) != len(fds):
            raise ValueError("--units must contain one unit number for each FD subset.")
        unit_map.update({fd: int(unit) for fd, unit in zip(fds, units)})

    return unit_map


def default_output_path(fds):
    if tuple(fds) == VALID_FDS:
        name = "all_z_DTE.png"
    else:
        name = "all_z_DTE_" + "_".join(fds) + ".png"
    return os.path.join(config["output_dir"], name)


@torch.no_grad()
def encode_latent_from_npy(data_path, ruls_path, units_path, fd="FD001", batch_size=256):
    x_np = np.load(data_path).astype(np.float32)
    rul_np = np.load(ruls_path).astype(np.float32)
    unit_np = np.load(units_path).astype(np.int32)

    assert len(x_np) == len(rul_np) == len(unit_np)

    model_path = os.path.join(config["output_dir"], f"best_vae_model_{fd}.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(config["output_dir"], "best_vae_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing DTE checkpoint: {model_path}")

    model = torch.load(model_path, map_location=device, weights_only=False).to(device).eval()

    mus = []
    for start in range(0, x_np.shape[0], batch_size):
        end = min(start + batch_size, x_np.shape[0])
        x = torch.from_numpy(x_np[start:end]).to(device)
        _, _, mean, _, _ = model(x)
        mus.append(mean.cpu())

    mu = torch.cat(mus, dim=0).numpy()
    return mu, rul_np, unit_np


def orient_latents(fd, z2):
    if fd == "FD001":
        return -z2[:, 0], -z2[:, 1]
    if fd == "FD002":
        return -z2[:, 0], z2[:, 1]
    return z2[:, 0], z2[:, 1]


def plot_dte_latents(
    fds,
    split="train",
    step=1,
    batch_size=256,
    unit_map=None,
    preprocessed_dir=None,
    output_path=None,
    show=False,
):
    preprocessed_dir = preprocessed_dir or os.path.join(config["output_dir"], "preprocessed")
    output_path = output_path or default_output_path(fds)
    unit_map = unit_map or DEFAULT_UNIT_MAP

    results = {}
    global_rul_min = float("inf")
    global_rul_max = float("-inf")

    for fd in fds:
        prefix = f"{fd}_{split}_ws{config['window_size']}_step{step}"
        data_path = os.path.join(preprocessed_dir, f"preprocessed_data_{prefix}.npy")
        ruls_path = os.path.join(preprocessed_dir, f"preprocessed_ruls_{prefix}.npy")
        units_path = os.path.join(preprocessed_dir, f"preprocessed_units_{prefix}.npy")

        for path in (data_path, ruls_path, units_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing preprocessed file: {path}")

        mu, rul, unit_ids = encode_latent_from_npy(
            data_path,
            ruls_path,
            units_path,
            fd=fd,
            batch_size=batch_size,
        )

        z = (mu - mu.mean(axis=0, keepdims=True)) / (mu.std(axis=0, keepdims=True) + 1e-8)
        z2 = PCA(n_components=2, random_state=123).fit_transform(z)

        target_unit = unit_map.get(fd)
        if target_unit is None:
            target_unit = int(np.unique(unit_ids)[0])

        results[fd] = {"z2": z2, "rul": rul, "unit_ids": unit_ids, "target_unit": target_unit}
        global_rul_min = min(global_rul_min, float(rul.min()))
        global_rul_max = max(global_rul_max, float(rul.max()))

    n_plots = len(fds)
    cols = min(2, n_plots)
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.7 * rows), squeeze=False)
    axes = axes.ravel()

    norm = Normalize(vmin=global_rul_min, vmax=global_rul_max)
    bg_cmap = plt.cm.viridis
    fg_cmap = plt.cm.hot_r
    sm_bg = ScalarMappable(norm=norm, cmap=bg_cmap)
    sm_fg = ScalarMappable(norm=norm, cmap=fg_cmap)

    used_axes = []
    for ax, fd in zip(axes, fds):
        z2 = results[fd]["z2"]
        rul = results[fd]["rul"]
        unit_ids = results[fd]["unit_ids"]
        target_unit = results[fd]["target_unit"]
        x_all, y_all = orient_latents(fd, z2)

        mask = unit_ids == target_unit
        ax.scatter(x_all, y_all, c=rul, cmap=bg_cmap, norm=norm, s=3, alpha=0.3)
        ax.scatter(
            x_all[mask],
            y_all[mask],
            c=rul[mask],
            cmap=fg_cmap,
            norm=norm,
            s=20,
            edgecolors="k",
            linewidths=0.3,
        )
        ax.set_title(f"{fd} / Unit #{target_unit}")
        ax.set_xlabel("latent dim 1 (PCA)")
        ax.set_ylabel("latent dim 2 (PCA)")
        used_axes.append(ax)

    for ax in axes[n_plots:]:
        fig.delaxes(ax)

    if len(fds) == 4 and rows == 2 and cols == 2:
        fig.set_size_inches(12, 7)

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    cax_bg = fig.add_axes([0.84, 0.15, 0.02, 0.7])
    fig.colorbar(sm_bg, cax=cax_bg).set_label("RUL (cycles) - background")

    cax_fg = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm_fg, cax=cax_fg).set_label("RUL (cycles) - highlighted unit")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print("Saved:", output_path)
    return output_path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Plot DTE latent PCA projections for CMAPSS FD subsets.")
    parser.add_argument("--fds", nargs="+", default=list(VALID_FDS), help="FD subsets to plot.")
    parser.add_argument("--output-dir", default=config["output_dir"], help="Directory containing DTE checkpoints.")
    parser.add_argument(
        "--preprocessed-dir",
        default=None,
        help="Directory containing preprocessed .npy files. Defaults to OUTPUT_DIR/preprocessed.",
    )
    parser.add_argument("--window-size", type=int, default=int(config["window_size"]))
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--units", nargs="+", type=int, default=None, help="Unit numbers, one per FD.")
    parser.add_argument("--unit-map", default=None, help="FD/unit pairs, for example FD001=86,FD002=103.")
    parser.add_argument("--output", default=None, help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Open the plot window after saving.")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config["output_dir"] = args.output_dir
    config["window_size"] = args.window_size

    fds = normalize_fds(args.fds)
    unit_map = choose_units(fds, units=args.units, unit_map_text=args.unit_map)
    plot_dte_latents(
        fds=fds,
        split=args.split,
        step=args.step,
        batch_size=args.batch_size,
        unit_map=unit_map,
        preprocessed_dir=args.preprocessed_dir,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
