import torch
from .base import BaseDiffusion


class Diffusion(BaseDiffusion):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, schedule_name="linear", device="cpu"):
        super().__init__(noise_steps, beta_start, beta_end, schedule_name, device)

    def predict_x0(self, x_t, t, eps_pred):
        """
        x0 = (x_t - sqrt(1-a_hat)*eps) / sqrt(a_hat)
        """
        a_hat = self.alpha_hat[t][:, None, None]  # [B,1,1]
        return (x_t - torch.sqrt(1.0 - a_hat) * eps_pred) / (torch.sqrt(a_hat) + 1e-8)

    def p_mean_variance(self, x_t, t, eps_pred, clip_x0=True, x0_min=0.0, x0_max=1.0):
        """
        Compute posterior p(x_{t-1} | x_t, x0_pred)
        """
        beta_t = self.beta[t][:, None, None]          # [B,1,1]
        alpha_t = self.alpha[t][:, None, None]        # [B,1,1]
        a_hat_t = self.alpha_hat[t][:, None, None]    # [B,1,1]

        # alpha_hat_{t-1}
        t_prev = (t - 1).clamp(min=0)
        a_hat_prev = self.alpha_hat[t_prev][:, None, None]

        x0_pred = self.predict_x0(x_t, t, eps_pred)
        if clip_x0:
            x0_pred = x0_pred.clamp(x0_min, x0_max)

        # posterior mean coefficients
        coef1 = beta_t * torch.sqrt(a_hat_prev) / (1.0 - a_hat_t + 1e-8)
        coef2 = (1.0 - a_hat_prev) * torch.sqrt(alpha_t) / (1.0 - a_hat_t + 1e-8)

        mean = coef1 * x0_pred + coef2 * x_t

        # posterior variance
        var = beta_t * (1.0 - a_hat_prev) / (1.0 - a_hat_t + 1e-8)
        var = var.clamp(min=1e-20)

        return mean, var, x0_pred

    def sample(self, config, model, conditioner, cycle_alpha, clip_x0=True):
        """
        conditioner: [B, latent_dim]
        cycle_alpha: [B]
        returns: [B, L, F] in the SAME normalized space used in training
        """
        with torch.no_grad():
            B = conditioner.shape[0]
            L = config["window_size"]
            F = config["input_size"]

            x = torch.randn(B, L, F, device=self.device)
            cycle_alpha = cycle_alpha.to(self.device).float()

            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((B,), i, device=self.device, dtype=torch.long)

                eps_pred = model(x, t, conditioner, cycle_alpha)
                mean, var, x0_pred = self.p_mean_variance(x, t, eps_pred, clip_x0=clip_x0, x0_min=0.0, x0_max=1.0)

                if i > 1:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(var) * noise
                else:
                    x = mean

            return x