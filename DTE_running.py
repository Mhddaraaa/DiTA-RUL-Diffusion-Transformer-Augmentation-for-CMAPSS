import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_score_fun(y, y_hat):
    score = 0.0
    y = y.cpu().numpy().ravel()
    y_hat = y_hat.cpu().numpy().ravel()
    for yi, yhi in zip(y, y_hat):
        if yi <= yhi:
            score += np.exp(-(yi - yhi) / 10.0) - 1.0
        else:
            score += np.exp((yi - yhi) / 13.0) - 1.0
    return score

def train_epoch(config, epoch, model, optimizer, criterion, train_loader, history):
    model.train()
    epoch_loss = defaultdict(list)

    for data in train_loader:
        pairs_mode = train_loader.dataset.return_pairs
        optimizer.zero_grad()

        if pairs_mode:
            x, pos_x, neg_x, true_rul = data
            x = x.to(device)
            true_rul = true_rul.to(device)
            pos_x = pos_x.to(device)
            neg_x = neg_x.to(device)

            predicted_rul, z, mean, log_var, x_hat = model(x)

            # IMPORTANT: encoder-only for pos/neg
            z_pos, _, _ = model.encoder(pos_x)
            z_neg, _, _ = model.encoder(neg_x)

            loss_dict = criterion(
                mean=mean, log_var=log_var,
                y=true_rul, y_hat=predicted_rul,
                x=x, x_hat=x_hat,
                z=z, z_pos=z_pos, z_neg=z_neg
            )
        else:
            x, true_rul = data
            x = x.to(device)
            true_rul = true_rul.to(device)

            predicted_rul, z, mean, log_var, x_hat = model(x)
            loss_dict = criterion(
                mean=mean, log_var=log_var,
                y=true_rul, y_hat=predicted_rul,
                x=x, x_hat=x_hat,
                z=z
            )

        loss = loss_dict["TotalLoss"]
        loss.backward()
        optimizer.step()

        for key in loss_dict:
            epoch_loss[key].append(loss_dict[key].item())

    for key in loss_dict:
        history["Train_" + key].append(np.mean(epoch_loss[key]))

def valid_epoch(config, epoch, model, criterion, valid_loader, history):
    model.eval()
    epoch_loss = defaultdict(list)

    for data in valid_loader:
        pairs_mode = valid_loader.dataset.return_pairs
        with torch.no_grad():
            if pairs_mode:
                x, pos_x, neg_x, y = data
                x = x.to(device)
                y = y.to(device)
                pos_x = pos_x.to(device)
                neg_x = neg_x.to(device)

                y_hat, z, mean, log_var, x_hat = model(x)
                z_pos, _, _ = model.encoder(pos_x)
                z_neg, _, _ = model.encoder(neg_x)

                loss_dict = criterion(
                    mean=mean, log_var=log_var,
                    y=y, y_hat=y_hat,
                    x=x, x_hat=x_hat,
                    z=z, z_pos=z_pos, z_neg=z_neg
                )
            else:
                x, y = data
                x = x.to(device)
                y = y.to(device)

                y_hat, z, mean, log_var, x_hat = model(x)
                loss_dict = criterion(
                    mean=mean, log_var=log_var,
                    y=y, y_hat=y_hat,
                    x=x, x_hat=x_hat,
                    z=z
                )

            for key in loss_dict:
                epoch_loss[key].append(loss_dict[key].item())

    for key in loss_dict:
        history["Val_" + key].append(np.mean(epoch_loss[key]))

def get_dataset_score(config, model, dataloader, history):
    model.eval()
    rmse_acc = 0.0
    score_acc = 0.0

    n_samples = len(dataloader.dataset)
    pairs_mode = dataloader.dataset.return_pairs

    for data in dataloader:
        with torch.no_grad():
            if pairs_mode:
                x, _, _, y = data
            else:
                x, y = data

            x = x.to(device)
            y = y.to(device)

            y_hat, *_ = model(x)

            mse_batch = nn.MSELoss(reduction="sum")(y_hat, y)
            rmse_acc += mse_batch.item()
            score_acc += cal_score_fun(y, y_hat)

    rmse = (rmse_acc / n_samples) ** 0.5
    history["Val_Score"].append(score_acc)
    history["Val_RMSE"].append(rmse)
    return score_acc, rmse