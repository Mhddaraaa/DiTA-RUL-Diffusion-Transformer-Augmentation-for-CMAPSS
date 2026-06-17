import math
import torch


class BaseDiffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, schedule_name="linear", device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_name = schedule_name
        self.device = device

        self.beta = self.prepare_noise_schedule(schedule_name).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self, schedule_name="linear"):
        if schedule_name == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

        elif schedule_name == "cosine":
            def alpha_hat_fn(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

            betas = []
            max_beta = 0.999
            for i in range(self.noise_steps):
                t1 = i / self.noise_steps
                t2 = (i + 1) / self.noise_steps
                beta_t = min(1 - alpha_hat_fn(t2) / alpha_hat_fn(t1), max_beta)
                betas.append(beta_t)
            return torch.tensor(betas, dtype=torch.float32)

        elif schedule_name == "quadratic":
            return torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.noise_steps) ** 2

        elif schedule_name == "sigmoid":
            return torch.sigmoid(torch.linspace(-6, 6, self.noise_steps)) * (self.beta_end - self.beta_start) + self.beta_start

        else:
            raise ValueError(f"Unknown schedule_name: {schedule_name}")

    def noise_images(self, x, time):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[time])[:, None, None]
        sqrt_one_minus = torch.sqrt(1.0 - self.alpha_hat[time])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus * eps, eps

    def sample_time_steps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))