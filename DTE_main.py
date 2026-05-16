import os
import logging
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
            torch.save(model, best_path_generic)  # convenience fallback
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


if __name__ == "__main__":
    assert config["input_size"] == 25, "config['input_size'] must be 25 with feature_cols=1..25"

    fds = ["FD001", "FD002", "FD003", "FD004"]
    for fd in fds:
        data_path = f"output/preprocessed/preprocessed_data_{fd}_train_ws{config['window_size']}_step1.npy"
        ruls_path = f"output/preprocessed/preprocessed_ruls_{fd}_train_ws{config['window_size']}_step1.npy"

        data = np.load(data_path, mmap_mode="r")
        N = data.shape[0]
        idx = np.arange(N)

        # temporal split (no shuffle) is fine for CMAPSS windows
        train_idx, valid_idx = train_test_split(idx, test_size=0.2, random_state=42, shuffle=False)
        
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

        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_ds, batch_size=config["batch_size"], shuffle=False)

        model_train(config, train_loader, valid_loader, fd_tag=fd)