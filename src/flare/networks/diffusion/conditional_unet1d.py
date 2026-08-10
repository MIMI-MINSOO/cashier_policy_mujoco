"""
1D Conditional UNet for Diffusion Policy.
Faithfully adapted from LeRobot's DiffusionConditionalUnet1d.
"""

import math
import torch
import torch.nn as nn
import einops


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=5, n_groups=8,
                 use_film_scale_modulation=True):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, :self.out_channels]
            bias = cond_embed[:, self.out_channels:]
            out = scale * out + bias
        else:
            out = out + cond_embed
        out = self.conv2(out)
        return out + self.residual_conv(x)


class ConditionalUnet1d(nn.Module):
    """1D UNet with FiLM conditioning. Matches LeRobot's architecture."""

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        down_dims: list[int] = (256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        diffusion_step_embed_dim: int = 256,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim

        # in_out pairs: [(input_dim, down_dims[0]), (down_dims[0], down_dims[1]), ...]
        in_out = [(input_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))
        common = dict(cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups,
                      use_film_scale_modulation=use_film_scale_modulation)

        # Encoder
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResBlock1d(dim_in, dim_out, **common),
                ConditionalResBlock1d(dim_out, dim_out, **common),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
            ]))

        # Middle
        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResBlock1d(mid_dim, mid_dim, **common),
            ConditionalResBlock1d(mid_dim, mid_dim, **common),
        ])

        # Decoder: reverse of in_out[1:]
        self.up_modules = nn.ModuleList()
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1
            self.up_modules.append(nn.ModuleList([
                ConditionalResBlock1d(dim_in * 2, dim_out, **common),
                ConditionalResBlock1d(dim_out, dim_out, **common),
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond=None):
        x = einops.rearrange(sample, "b t d -> b d t")

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=x.device).expand(x.shape[0])
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(x.shape[0])

        timestep_embed = self.diffusion_step_encoder(timestep.float())
        if global_cond is not None:
            global_feature = torch.cat([timestep_embed, global_cond], dim=-1)
        else:
            global_feature = timestep_embed

        # Encoder
        skips = []
        for res1, res2, downsample in self.down_modules:
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            skips.append(x)
            x = downsample(x)

        # Middle
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        # Decoder
        for res1, res2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = res1(x, global_feature)
            x = res2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return einops.rearrange(x, "b d t -> b t d")
