import os
import argparse
import numpy as np
import pandas as pd
import inspect
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from config import config

# Import model classes referenced by serialized checkpoints.
from DTE_model.DTE_network import (
    Encoder, Decoder, TSHAE, DropBlockLatent, WaveletConvBlock, FourierBlock
)
from Diffusion_model.Diff_network import DiffWave
from Diffusion_model.ddpm import Diffusion
from RUL_models.CNNBased_RUL import CNNRULPredictor
from RUL_models.RNNBased_RUL import RNNRULPredictor
from RUL_models.Hybrid_RUL import HybridRULPredictor
from RUL_models.TransformerBased_RUL import TransformerRULPredictor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_FDS = ("FD001", "FD002", "FD003", "FD004")

# 17-sensor indices inside the 25-d preprocessed vector
# 0=time, 1..3=settings, 4=T2, 5=T24, 6=T30, 7=T50, ...
SENSOR_IDXS_17 = np.array([1, 2, 3, 5, 6, 7, 10, 11, 12, 14, 15, 16, 17, 18, 20, 23, 24], dtype=np.int64)


def normalize_fds(fds):
    normalized = [fd.upper() for fd in fds]
    invalid = [fd for fd in normalized if fd not in VALID_FDS]
    if invalid:
        raise ValueError(f"Unsupported FD subset(s): {', '.join(invalid)}")
    return normalized


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def calculate_metrics(predictions, targets):
    """RMSE, R2, and C-MAPSS Score."""
    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(targets, dtype=np.float64)

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

    score = 0.0
    for p, t in zip(pred, true):
        d = p - t
        if d < 0:
            score += np.exp(-d / 13.0) - 1.0
        else:
            score += np.exp(d / 10.0) - 1.0

    return rmse, r2, float(score)


def paths_for_fd(fd: str, split: str, step: int = 1):
    prep_dir = os.path.join(config["output_dir"], "preprocessed")
    ws = int(config["window_size"])
    prefix = f"{fd}_{split}_ws{ws}_step{step}"

    data_path  = os.path.join(prep_dir, f"preprocessed_data_{prefix}.npy")
    ruls_path  = os.path.join(prep_dir, f"preprocessed_ruls_{prefix}.npy")
    units_path = os.path.join(prep_dir, f"preprocessed_units_{prefix}.npy")
    alpha_path = os.path.join(prep_dir, f"preprocessed_alpha_{prefix}.npy")

    for p in [data_path, ruls_path, units_path, alpha_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    return data_path, ruls_path, units_path, alpha_path, prefix


def engine_last_window_indices(units_np: np.ndarray, alpha_np: np.ndarray) -> np.ndarray:
    """One window per engine: choose the max-alpha window for each unit."""
    keep = []
    for u in np.unique(units_np):
        idxs = np.where(units_np == u)[0]
        j = idxs[np.argmax(alpha_np[idxs])]
        keep.append(int(j))
    return np.array(keep, dtype=np.int64)


class MemMapWindowRULDataset(Dataset):
    """
    Memory-mapped dataset reading X:[N,L,25], y:[N] from disk.
    Returns x:[L, 17], y:scalar

    This is intentionally NOT PreprocessedDatasetDTE:
    - DTE dataset is for DTE training and generally loads into RAM.
    - RUL training concatenates real and generated windows, so memmap keeps IO scalable.
    """
    def __init__(
        self,
        x_path: str,
        y_path: str,
        indices: np.ndarray | None = None,
        sensor_idxs=SENSOR_IDXS_17,
        rul_cap: float = 125.0
    ):
        self.X = np.load(x_path, mmap_mode="r")  # [N,L,25]
        self.y = np.load(y_path, mmap_mode="r")  # [N]

        if indices is None:
            self.indices = None
            self.N = int(self.X.shape[0])
        else:
            self.indices = indices.astype(np.int64)
            self.N = int(len(self.indices))

        self.sensor_idxs = np.asarray(sensor_idxs, dtype=np.int64)
        self.rul_cap = float(rul_cap)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        j = int(idx) if self.indices is None else int(self.indices[idx])

        # copy=True avoids "non-writable numpy" warnings and any undefined behavior
        x25 = np.array(self.X[j], dtype=np.float32, copy=True)  # [L,25]
        x17 = x25[:, self.sensor_idxs]                          # [L,17]

        y = float(self.y[j])
        if self.rul_cap > 0:
            y = min(y, self.rul_cap)

        return torch.from_numpy(x17), torch.tensor(y, dtype=torch.float32)


def load_models(fd: str):
    dte_path  = os.path.join(config["output_dir"], f"best_vae_model_{fd}.pt")
    diff_path = os.path.join(config["output_dir"], f"best_diff_model_{fd}.pt")

    if not os.path.exists(dte_path):
        raise FileNotFoundError(f"DTE not found: {dte_path}")
    if not os.path.exists(diff_path):
        raise FileNotFoundError(f"Diffusion not found: {diff_path}")

    dte: TSHAE = torch.load(dte_path, map_location=DEVICE, weights_only=False).to(DEVICE).eval()
    diff: DiffWave = torch.load(diff_path, map_location=DEVICE, weights_only=False).to(DEVICE).eval()

    diffusion = Diffusion(
        noise_steps=config["noise_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule_name=config["schedule_name"],
        device=DEVICE,
    )
    return dte, diff, diffusion


@torch.no_grad()
def build_or_load_generated_train(fd: str, batch_size: int, seed: int, clip_gen: bool = False, step: int = 1):
    """
    Build generated train windows aligned one-to-one with real train windows.
    Cache location:
      output/augmented/preprocessed_data_{prefix}_GEN.npy
      output/augmented/preprocessed_ruls_{prefix}_GEN.npy
    """
    x_tr, y_tr, _, a_tr, prefix = paths_for_fd(fd, "train", step=step)

    out_aug = os.path.join(config["output_dir"], "augmented")
    os.makedirs(out_aug, exist_ok=True)

    xg_path = os.path.join(out_aug, f"preprocessed_data_{prefix}_GEN.npy")
    yg_path = os.path.join(out_aug, f"preprocessed_ruls_{prefix}_GEN.npy")

    if os.path.exists(xg_path) and os.path.exists(yg_path):
        return xg_path, yg_path

    set_seed(seed)

    X = np.load(x_tr, mmap_mode="r")                    # [N,L,25]
    y = np.load(y_tr, mmap_mode="r").astype(np.float32) # [N]
    a = np.load(a_tr, mmap_mode="r").astype(np.float32) # [N]
    N, L, F = X.shape

    dte, diff, diffusion = load_models(fd)

    # Write with memmap to avoid large RAM spikes.
    xg_mm = np.lib.format.open_memmap(xg_path, mode="w+", dtype=np.float32, shape=(N, L, F))
    np.save(yg_path, y)  # labels identical to real windows

    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)

        xb_np = np.array(X[s:e], dtype=np.float32, copy=True)  # writable contiguous [B,L,25]
        xb = torch.from_numpy(xb_np).to(DEVICE)
        ab = torch.from_numpy(np.clip(a[s:e], 1e-4, 1.0)).float().to(DEVICE)  # [B]

        _, z, _, _, _ = dte(xb)
        xg = diffusion.sample(config=config, model=diff, conditioner=z, cycle_alpha=ab)  # [B,L,25]
        xg = xg.detach().cpu().numpy().astype(np.float32)

        if clip_gen:
            xg = np.clip(xg, 0.0, 1.0)

        xg_mm[s:e] = xg

        if (s // batch_size) % 50 == 0:
            print(f"[GEN BUILD] {fd}: {e}/{N} windows generated")

    del xg_mm
    print(f"[GEN BUILD] Saved generated train set:\n  {xg_path}\n  {yg_path}")
    return xg_path, yg_path


def forward_model(model, batch_x):
    """
    Unified contract:
      batch_x is always [B, L, 17]
    RUL models should accept [B, L, 17] directly.

    This function keeps safe fallbacks for legacy variants,
    but the default path is: model(batch_x).
    """
    try:
        return model(batch_x).view(-1)
    except RuntimeError as e:
        msg = str(e)

        # Some legacy CNN variants expect [B,1,L,17]
        if ("input.dim() = 4" in msg) or ("permute" in msg and batch_x.dim() == 3):
            return model(batch_x.unsqueeze(1)).view(-1)

        # Some legacy conv1d hybrids expect [B, 17, L] and do not permute internally.
        if ("expected input" in msg and "to have 14 channels" in msg) or ("Given groups" in msg and "channels" in msg):
            x_cf = batch_x.permute(0, 2, 1).contiguous()  # [B,17,L]
            return model(x_cf).view(-1)

        raise


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).view(-1)
        out = forward_model(model, x).view(-1)
        preds.extend(out.detach().cpu().numpy().tolist())
        trues.extend(y.detach().cpu().numpy().tolist())
    return calculate_metrics(preds, trues)


def train_one_model(model, model_name, train_loader, test_loader, out_dir,
                    epochs=50, lr=1e-4, grad_clip=None, use_huber=False):
    model = model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    # With noisy generated data, Huber is often more stable than MSE.
    criterion = nn.SmoothL1Loss(beta=10.0) if use_huber else nn.MSELoss()

    best_rmse = float("inf")
    best_path = os.path.join(out_dir, f"{model_name}_best.pth")
    
    min_rms = float('inf')
    best_r2 = 0
    retain = 0
    pbar = tqdm(range(1, epochs + 1), desc="Epoch")
    for epoch in pbar:
        model.train()
        total, nb = 0.0, 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).view(-1)

            pred = forward_model(model, x).view(-1)
            loss = criterion(pred, y)

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            total += float(loss.item())
            nb += 1

        train_loss = total / max(nb, 1)
        rmse, r2, score = evaluate(model, test_loader)
        scheduler.step(rmse)
    
        if rmse < min_rms:
            min_rms = rmse
            best_r2 = r2
            retain = 0
        else:
            retain += 1

        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "Best RMSE": f"{min_rms:.4f}",
            "Best R2": f"{best_r2:.4f}",
            f"Score": f"{score:.4f}",
            "no_improve": retain
        })

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    rmse, r2, score = evaluate(model, test_loader)
    return best_rmse, r2, score, best_path


def build_models():
    models = []
    ws = int(config["window_size"])

    models.append(("RNN-based", RNNRULPredictor(input_size=17, hidden_size=64)))
    models.append(("CNN-based",CNNRULPredictor()))
    models.append(("Hybrid",HybridRULPredictor(in_channels=17, hidden_size=64, dropout=0.1)))
    models.append(("Transformer-based",TransformerRULPredictor(input_dim=17, seq_len=ws)))

    return models


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Train RUL predictors on real plus generated CMAPSS windows.")
    ap.add_argument("--fd", type=str, default=None, help="Single FD subset. Kept for backward compatibility.")
    ap.add_argument("--fds", nargs="+", default=None, help="One or more FD subsets to train.")
    ap.add_argument("--output_dir", "--output-dir", dest="output_dir", default=config["output_dir"])
    ap.add_argument("--window_size", "--window-size", dest="window_size", type=int, default=int(config["window_size"]))
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--eval_mode", "--eval-mode", dest="eval_mode", type=str, default="engine", choices=["window", "engine"])
    ap.add_argument("--epochs", type=int, default=int(config.get("max_epochs", 50)))
    ap.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=int(config.get("batch_size", 32)))
    ap.add_argument("--lr", type=float, default=float(config.get("lr", 1e-4)))
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--grad_clip", "--grad-clip", dest="grad_clip", type=float, default=float(config.get("grad_clip", 0.0)))
    ap.add_argument("--gen_batch_size", "--gen-batch-size", dest="gen_batch_size", type=int, default=64)
    ap.add_argument("--clip_gen", "--clip-gen", dest="clip_gen", action="store_true")
    ap.add_argument("--rul_cap", "--rul-cap", dest="rul_cap", type=float, default=125.0, help="Set 0 to disable capping.")
    ap.add_argument("--use_huber", "--use-huber", dest="use_huber", action="store_true", help="Use SmoothL1Loss.")
    ap.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=0)
    return ap


def resolve_fds(args):
    if args.fds:
        return normalize_fds(args.fds)
    if args.fd:
        return normalize_fds([args.fd])
    return ["FD001"]


def train_rul_for_fd(fd, args):
    set_seed(args.seed)

    out_dir = os.path.join(config["output_dir"], "rul_models_real_plus_gen", fd, args.eval_mode)
    os.makedirs(out_dir, exist_ok=True)

    tr_data, tr_ruls, tr_units, tr_alpha, _ = paths_for_fd(fd, "train", step=args.step)
    te_data, te_ruls, te_units, te_alpha, _ = paths_for_fd(fd, "test", step=args.step)

    gen_data, gen_ruls = build_or_load_generated_train(
        fd=fd,
        batch_size=args.gen_batch_size,
        seed=args.seed,
        clip_gen=args.clip_gen,
        step=args.step,
    )

    if args.eval_mode == "engine":
        units_np = np.load(te_units).astype(np.int32)
        alpha_np = np.load(te_alpha).astype(np.float32)
        idx_keep = engine_last_window_indices(units_np, alpha_np)
    else:
        idx_keep = None

    ds_real_train = MemMapWindowRULDataset(tr_data, tr_ruls, indices=None, rul_cap=args.rul_cap)
    ds_gen_train  = MemMapWindowRULDataset(gen_data, gen_ruls, indices=None, rul_cap=args.rul_cap)
    ds_train = ConcatDataset([ds_real_train, ds_gen_train])

    ds_test = MemMapWindowRULDataset(te_data, te_ruls, indices=idx_keep, rul_cap=args.rul_cap)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"\nFD={fd} | eval_mode={args.eval_mode}")
    print(f"Real train windows: {len(ds_real_train)}")
    print(f"Generated train windows: {len(ds_gen_train)}  (one generated window per real window)")
    print(f"Total train samples (real+gen): {len(ds_train)}")
    print(f"Test samples: {len(ds_test)}")
    print(f"RUL cap: {args.rul_cap}  (0 means disabled)")
    print(f"Model input contract: [B, L, 17]\n")

    results = []
    for name, model in build_models():
        print(f"\n--- Training: {name} (real + gen) ---")
        best_rmse, r2, score, best_path = train_one_model(
            model=model,
            model_name=name.replace(" ", "_"),
            train_loader=dl_train,
            test_loader=dl_test,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            grad_clip=args.grad_clip if args.grad_clip > 0 else None,
            use_huber=args.use_huber,
        )
        results.append({
            "FD": fd,
            "EvalMode": args.eval_mode,
            "Model": name,
            "RMSE": best_rmse,
            "R2": r2,
            "Score": score,
            "BestCkpt": best_path,
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "rul_results.csv")
    df.to_csv(csv_path, index=False)

    print("\nSaved:", csv_path)
    print(df)


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    config["output_dir"] = args.output_dir
    config["window_size"] = args.window_size

    for fd in resolve_fds(args):
        train_rul_for_fd(fd, args)


if __name__ == "__main__":
    main()
