import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss:
    def __init__(self, weight: float):
        self.name = "KLLoss"
        self.weight = float(weight)

    def __call__(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # KL per sample then mean over batch
        # [B, D] -> [B] -> scalar
        kl = -0.5 * (1.0 + log_var - mean.pow(2) - torch.exp(log_var))
        return kl.sum(dim=1).mean()


class RegLoss:
    def __init__(self, weight: float):
        self.name = "RegLoss"
        self.weight = float(weight)
        self.criterion = nn.MSELoss()

    def __call__(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return self.criterion(y_hat, y)


class ReconLoss:
    """
    Recon MSE with sane scaling.
    By default we compute mean MSE over all elements.
    If you want "sum", we normalize by (L*F) so magnitudes stay comparable.
    """
    def __init__(self, weight: float, mode: str = "mean"):
        self.name = "ReconLoss"
        self.weight = float(weight)
        assert mode in ["mean", "sum_norm"], "mode must be 'mean' or 'sum_norm'"
        self.mode = mode

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return F.mse_loss(x_hat, x, reduction="mean")

        # sum over (L*F) per sample, then normalize by (L*F), then mean over batch
        B = x.size(0)
        per_elem = F.mse_loss(x_hat, x, reduction="none").view(B, -1)  # [B, L*F]
        LxF = per_elem.size(1)
        return per_elem.sum(dim=1).div(LxF).mean()


class TripletLoss:
    def __init__(self, weight: float, margin: float, p: int):
        self.name = "TripletLoss"
        self.weight = float(weight)
        self.criterion = nn.TripletMarginLoss(margin=margin, p=p)

    def __call__(self, z: torch.Tensor, z_pos: torch.Tensor, z_neg: torch.Tensor) -> torch.Tensor:
        # IMPORTANT: normalize embeddings to avoid scale cheating
        z = F.normalize(z, p=2, dim=1)
        z_pos = F.normalize(z_pos, p=2, dim=1)
        z_neg = F.normalize(z_neg, p=2, dim=1)
        return self.criterion(z, z_pos, z_neg)


def decorrelation_loss(z: torch.Tensor) -> torch.Tensor:
    """
    z: [B, D]
    Penalize off-diagonal covariance (decorrelate latent dims).
    Safe for small batches.
    """
    B = z.size(0)
    if B < 2:
        return z.new_tensor(0.0)

    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.t() @ z) / (B - 1)  # [D, D]
    off_diag = cov - torch.diag(torch.diag(cov))
    return (off_diag ** 2).mean()

class TrendLoss:
    def __init__(self, weight: float):
        self.name = "TrendLoss"
        self.weight = float(weight)

    def __call__(self, x_hat, x):
        dx_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]
        dx = x[:, 1:, :] - x[:, :-1, :]
        return F.mse_loss(dx_hat.mean(dim=1), dx.mean(dim=1))
    
class LowFreqSpectralLoss:
    def __init__(self, weight: float):
        self.name = "LowFreqSpectralLoss"
        self.weight = float(weight)

    def __call__(self, x_hat, x, keep_ratio=0.2):
        x_hat = x_hat - x_hat.mean(dim=1, keepdim=True)
        x = x - x.mean(dim=1, keepdim=True)

        X_hat = torch.fft.rfft(x_hat, dim=1)
        X = torch.fft.rfft(x, dim=1)

        K = max(2, int(X.shape[1] * keep_ratio))
        return F.mse_loss(torch.abs(X_hat[:, :K, :]), torch.abs(X[:, :K, :]))


class DiffusionLoss:
    def __init__(self, config: dict):
        self.mse = nn.MSELoss()
        self.trend = TrendLoss(config.get("Trend_weight", 0.1))
        self.spec = LowFreqSpectralLoss(config.get("Spec_weight", 0.05))
        self.keep_ratio = float(config.get("Spec_keep_ratio", 0.2))
    
    def __call__(self, pred_noise, noise, x0_pred, x):
        loss = self.mse(pred_noise, noise)
        if self.trend.weight > 0.0:
            loss = loss + self.trend.weight * self.trend(x0_pred, x)

        if self.spec.weight > 0.0:
            loss = loss + self.spec.weight * self.spec(x0_pred, x, keep_ratio=self.keep_ratio)

        return loss


class TotalLoss:
    """
    Drop-in replacement.

    Supports optional KL warmup (recommended):
      config["KLLoss_warmup_steps"] (int, default 0)
    If you pass global_step=... into criterion(...), KL weight will be scaled by
    min(1, global_step / warmup_steps).
    """
    def __init__(self, config: dict):
        self.decor_weight = float(config.get("DecorLoss_weight", 0.0))

        recon_mode = config.get("ReconLoss_mode", "mean")  # "mean" or "sum_norm"

        self.kl_warmup_steps = int(config.get("KLLoss_warmup_steps", 0))

        self.kl = KLLoss(config.get("KLLoss_weight", 1.0))
        self.reg = RegLoss(config.get("RegLoss_weight", 1.0))
        self.recon = ReconLoss(config.get("ReconLoss_weight", 0.0), mode=recon_mode)
        self.triplet = TripletLoss(
            config.get("TripletLoss_weight", 0.0),
            config.get("TripletLoss_margin", 1.0),
            config.get("TripletLoss_p", 2),
        )

    def __call__(
        self,
        mean=None, log_var=None,
        y=None, y_hat=None,
        x=None, x_hat=None,
        z=None, z_pos=None, z_neg=None,
        global_step: int = 0,
    ):
        losses = {"TotalLoss": 0.0}

        # Decor loss
        if self.decor_weight > 0.0 and isinstance(z, torch.Tensor):
            d = decorrelation_loss(z) * self.decor_weight
            losses["DecorLoss"] = d
            losses["TotalLoss"] = losses["TotalLoss"] + d

        # KL loss (with optional warmup)
        if isinstance(mean, torch.Tensor) and isinstance(log_var, torch.Tensor) and self.kl.weight > 0.0:
            kl_scale = 1.0
            if self.kl_warmup_steps > 0:
                kl_scale = min(1.0, float(global_step) / float(self.kl_warmup_steps))
            kl = self.kl(mean, log_var) * self.kl.weight * kl_scale
            losses["KLLoss"] = kl
            losses["TotalLoss"] = losses["TotalLoss"] + kl
        else:
            losses["KLLoss"] = torch.tensor(0.0, device=z.device if isinstance(z, torch.Tensor) else "cpu")

        # Regression loss
        if isinstance(y, torch.Tensor) and isinstance(y_hat, torch.Tensor) and self.reg.weight > 0.0:
            reg = self.reg(y, y_hat) * self.reg.weight
            losses["RegLoss"] = reg
            losses["TotalLoss"] = losses["TotalLoss"] + reg
        else:
            losses["RegLoss"] = torch.tensor(0.0, device=z.device if isinstance(z, torch.Tensor) else "cpu")

        # Reconstruction loss
        if isinstance(x, torch.Tensor) and isinstance(x_hat, torch.Tensor) and self.recon.weight > 0.0:
            rec = self.recon(x, x_hat) * self.recon.weight
            losses["ReconLoss"] = rec
            losses["TotalLoss"] = losses["TotalLoss"] + rec
        else:
            losses["ReconLoss"] = torch.tensor(0.0, device=z.device if isinstance(z, torch.Tensor) else "cpu")

        # Triplet loss (only when pos/neg provided)
        if (
            self.triplet.weight > 0.0
            and isinstance(z, torch.Tensor)
            and isinstance(z_pos, torch.Tensor)
            and isinstance(z_neg, torch.Tensor)
        ):
            tri = self.triplet(z, z_pos, z_neg) * self.triplet.weight
            losses["TripletLoss"] = tri
            losses["TotalLoss"] = losses["TotalLoss"] + tri
        else:
            losses["TripletLoss"] = torch.tensor(0.0, device=z.device if isinstance(z, torch.Tensor) else "cpu")

        return losses