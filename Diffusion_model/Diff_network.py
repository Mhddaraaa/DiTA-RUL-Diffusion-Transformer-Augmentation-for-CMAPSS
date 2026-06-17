from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


class ConditionerEmbedding(nn.Module):
    def __init__(self, cond_dim, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z):
        return self.net(z)  # [B, d_model]


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(max_steps), persistent=False)
        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        return self.net(x)

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)   # [T,1]
        dims  = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0 ** (dims * 4.0 / 63.0)    # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # [T,128]
        return table


class DiffusionTransformerBlock(nn.Module):
    """
    Transformer block conditioned on timestep t and conditioning vector cond via FiLM.
    """

    def __init__(self, d_model, num_heads, ff_dim, t_dim, film_scale=1.0):
        super().__init__()
        self.t_proj = nn.Linear(t_dim, d_model)

        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        # FiLM params
        self.film_scale_layer = nn.Linear(d_model, d_model)
        self.film_shift_layer = nn.Linear(d_model, d_model)

        # Start as identity (very important)
        nn.init.zeros_(self.film_scale_layer.weight)
        nn.init.zeros_(self.film_scale_layer.bias)
        nn.init.zeros_(self.film_shift_layer.weight)
        nn.init.zeros_(self.film_shift_layer.bias)

        self.film_scale = film_scale  # small gain if you want extra safety

    def forward(self, x, t_embed, cond_embed):
        t = self.t_proj(t_embed).unsqueeze(1)  # [B,1,D]
        h = x + t

        # Stable FiLM
        s = torch.tanh(self.film_scale_layer(cond_embed)) * self.film_scale  # [B,D]
        b = self.film_shift_layer(cond_embed)                                 # [B,D]
        h = h * (1.0 + s.unsqueeze(1)) + b.unsqueeze(1)

        attn_out, _ = self.mha(h, h, h)
        h = self.norm1(h + attn_out)

        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)

        return self.out_proj(h)


class DiffWave(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seq_len = config["window_size"]
        self.fea_dim = config["input_size"]
        self.d_model = config["residual_channels"]
        self.num_heads = config.get("num_heads", 4)
        self.num_layers = config["residual_layers"]
        self.ff_dim = config.get("ff_dim", 4 * self.d_model)
        self.latent_dim = config["latent_dim"]
        self.noise_steps = config["noise_steps"]

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        # If you want extra stability, swap BN -> GroupNorm (recommended)
        norm = lambda c: nn.GroupNorm(num_groups=8, num_channels=c)
        # norm = lambda c: nn.BatchNorm1d(c)

        self.input_projection = nn.Sequential(
            Conv1d(self.fea_dim, self.d_model, kernel_size=1, bias=True),
            norm(self.d_model),
            nn.GELU(),
        )

        self.diffusion_embedding = DiffusionEmbedding(self.noise_steps)  # [B,512]

        self.z_embed = ConditionerEmbedding(self.latent_dim, self.d_model)  # [B,d_model]
        self.alpha_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    ff_dim=self.ff_dim,
                    t_dim=512,
                    film_scale=config.get("film_scale", 1.0),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.post_conv = nn.Sequential(
            Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1, bias=True),
            norm(self.d_model),
            nn.GELU(),
        )

        self.output_projection = Conv1d(self.d_model, self.fea_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.output_projection.weight)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)

    def forward(self, x, diffusion_step, conditioner_z, alpha):
        """
        x: [B, L, F]
        diffusion_step: [B]
        conditioner_z: [B, latent_dim]
        alpha: [B] in [0,1]
        """
        B, L, F = x.shape
        assert L == self.seq_len

        h = self.input_projection(x.transpose(1, 2)).transpose(1, 2)  # [B,L,d_model]

        t_embed = self.diffusion_embedding(diffusion_step)  # [B,512]
        z = self.z_embed(conditioner_z)                     # [B,d_model]
        a = self.alpha_embed(alpha.view(B, 1))              # [B,d_model]
        cond = z + a                                        # [B,d_model]

        skips = []
        for block in self.blocks:
            h = block(h, t_embed, cond)
            skips.append(h)

        h = torch.stack(skips, dim=0).sum(dim=0) / sqrt(self.num_layers)  # [B,L,d_model]
        h = self.post_conv(h.transpose(1, 2))                              # [B,d_model,L]
        out = self.output_projection(h).transpose(1, 2)                    # [B,L,F]
        return out