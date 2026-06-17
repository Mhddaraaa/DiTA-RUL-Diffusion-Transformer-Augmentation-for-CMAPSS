import argparse
import os
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch

from config import config
from DTE_model.DTE_network import Encoder, Decoder, TSHAE
from utils.loss import TotalLoss
from DTE_running import train_epoch, valid_epoch, get_dataset_score
from preprocessed_dataset import PreprocessedDatasetDTE


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


def model_train(config, train_loader, valid_loader, fd_tag="FD001"):
    encoder = Encoder(config)
    decoder = Decoder(config)
    model = TSHAE(config, encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_step"], gamma=config["lr_gamma"]
    )
    criterion = TotalLoss(config)

    os.makedirs(config["output_dir"], exist_ok=True)

    best_rmse = float("inf")
    history = defaultdict(list)

    best_path_fd = os.path.join(config["output_dir"], f"best_vae_model_{fd_tag}.pt")
    best_path_generic = os.path.join(config["output_dir"], "best_vae_model.pt")

    pbar = tqdm(range(config["max_epochs"]), desc=f"Training DTE ({fd_tag})")
    for epoch in pbar:
        train_epoch(config, epoch, model, optimizer, criterion, train_loader, history)
        valid_epoch(config, epoch, model, criterion, valid_loader, history)
        scheduler.step()

        valid_score, valid_rmse = get_dataset_score(config, model, valid_loader, history)

        # checkpoint by RMSE
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            torch.save(model, best_path_fd)
            torch.save(model, best_path_generic)
            print(f"[{fd_tag}] Epoch: {epoch} | saved BEST DTE | rmse={best_rmse:.4f} -> {best_path_fd}")

        pbar.set_postfix({
            "Val_RMSE": f"{valid_rmse:.4f}",
            "Val_Score":f"{valid_score:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.4f}"
        })

    final_path = os.path.join(config["output_dir"], f"final_vae_model_{fd_tag}.pt")
    torch.save(model, final_path)
    print(f"[{fd_tag}] saved FINAL DTE -> {final_path}")
    return history


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the DTE/TSHAE model for CMAPSS FD subsets.")
    parser.add_argument("--fds", nargs="+", default=list(VALID_FDS), help="FD subsets to train.")
    parser.add_argument("--output-dir", default=config["output_dir"], help="Directory for model checkpoints.")
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
    parser.add_argument("--valid-size", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-split", action="store_true", help="Shuffle before the train/validation split.")
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


def preprocessed_paths(fd, cfg, preprocessed_dir, step):
    prefix = f"{fd}_train_ws{cfg['window_size']}_step{step}"
    data_path = os.path.join(preprocessed_dir, f"preprocessed_data_{prefix}.npy")
    ruls_path = os.path.join(preprocessed_dir, f"preprocessed_ruls_{prefix}.npy")

    for path in (data_path, ruls_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing preprocessed file: {path}")

    return data_path, ruls_path


def train_fd(fd, cfg, args):
    preprocessed_dir = args.preprocessed_dir or os.path.join(cfg["output_dir"], "preprocessed")
    data_path, ruls_path = preprocessed_paths(fd, cfg, preprocessed_dir, args.step)

    data = np.load(data_path, mmap_mode="r")
    N = data.shape[0]
    idx = np.arange(N)

    # Keep the default split temporal, matching the original experiment script.
    train_idx, valid_idx = train_test_split(
        idx,
        test_size=args.valid_size,
        random_state=args.seed,
        shuffle=args.shuffle_split,
    )

    train_ds = PreprocessedDatasetDTE(
        data_path=data_path,
        ruls_path=ruls_path,
        return_pairs=True,
        indices=train_idx,
    )

    valid_ds = PreprocessedDatasetDTE(
        data_path=data_path,
        ruls_path=ruls_path,
        return_pairs=False,
        indices=valid_idx,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_train(cfg, train_loader, valid_loader, fd_tag=fd)


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    assert cfg["input_size"] == 25, "config['input_size'] must be 25 with feature_cols=1..25"
    set_seed(args.seed)

    for fd in normalize_fds(args.fds):
        train_fd(fd, cfg, args)


if __name__ == "__main__":
    main()
