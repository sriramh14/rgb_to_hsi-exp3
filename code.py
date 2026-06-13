"""
model_rgb2hsi_difiisr.py

A compact RGB-to-HSI adaptation of the DifIISR / ResShift idea.

Core idea:
    RGB image I_rgb -> coarse HSI S_c
    residual-shift diffusion refines S_c into final HSI S_hat

Shapes:
    rgb      : [B, 3, H, W]
    hsi_gt   : [B, C, H, W]
    coarse  : [B, C, H, W]
    x_t      : [B, C, H, W]

The diffusion forward process is:
    x_t = (1 - eta_t) * S + eta_t * S_c + kappa * sqrt(eta_t) * eps

This file is intentionally self-contained and does not depend on the original DifIISR repo.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# Utility losses for HSI reconstruction
# -------------------------------------------------------------------------

def mrae_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean relative absolute error. pred/target: [B,C,H,W]."""
    return (pred - target).abs().div(target.abs() + eps).mean()


def sam_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Spectral Angle Mapper loss in radians. pred/target: [B,C,H,W]."""
    p = pred.permute(0, 2, 3, 1)
    t = target.permute(0, 2, 3, 1)
    dot = (p * t).sum(dim=-1)
    p_norm = torch.sqrt((p * p).sum(dim=-1) + eps)
    t_norm = torch.sqrt((t * t).sum(dim=-1) + eps)
    cos = dot / (p_norm * t_norm + eps)
    cos = cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """First-order spectral derivative loss along channel/wavelength dimension."""
    dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dt = target[:, 1:, :, :] - target[:, :-1, :, :]
    return F.l1_loss(dp, dt)


def spectral_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Second-order spectral smoothness prior. Can be used at inference without GT."""
    d1 = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    d2 = d1[:, 1:, :, :] - d1[:, :-1, :, :]
    return d2.abs().mean()


def project_hsi_to_rgb(hsi: torch.Tensor, response_matrix: torch.Tensor) -> torch.Tensor:
    """
    Project HSI to RGB using a spectral response matrix.

    hsi:             [B, C, H, W]
    response_matrix: [3, C], rows should usually be non-negative and normalized
    returns:         [B, 3, H, W]
    """
    if response_matrix.dim() != 2:
        raise ValueError("response_matrix must have shape [3, C].")
    if response_matrix.shape[0] != 3:
        raise ValueError("response_matrix must have shape [3, C].")
    if response_matrix.shape[1] != hsi.shape[1]:
        raise ValueError(
            f"response_matrix has {response_matrix.shape[1]} bands, "
            f"but HSI has {hsi.shape[1]} bands."
        )
    response_matrix = response_matrix.to(device=hsi.device, dtype=hsi.dtype)
    return torch.einsum("kc,bchw->bkhw", response_matrix, hsi)


def rgb_consistency_loss(
    pred_hsi: torch.Tensor,
    rgb: torch.Tensor,
    response_matrix: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compare projected HSI RGB with observed RGB."""
    rgb_hat = project_hsi_to_rgb(pred_hsi, response_matrix)
    if loss_type == "l1":
        return F.l1_loss(rgb_hat, rgb)
    if loss_type == "mse":
        return F.mse_loss(rgb_hat, rgb)
    raise ValueError(f"Unknown loss_type: {loss_type}")


# -------------------------------------------------------------------------
# Diffusion schedule
# -------------------------------------------------------------------------

def make_eta_schedule(
    steps: int = 15,
    min_noise_level: float = 0.04,
    etas_end: float = 0.99,
    kappa: float = 2.0,
    power: float = 0.3,
) -> torch.Tensor:
    """
    Exponential eta schedule similar to ResShift/DifIISR.

    Returns sqrt_etas with shape [steps].
    """
    if steps < 2:
        raise ValueError("steps must be >= 2.")
    etas_start = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
    increaser = math.exp(math.log(etas_end / etas_start) / (steps - 1))
    power_timestep = torch.linspace(0, 1, steps).pow(power) * (steps - 1)
    sqrt_etas = torch.ones(steps).mul(increaser).pow(power_timestep).mul(etas_start)
    return sqrt_etas.float()


def extract_to_shape(values: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Gather 1D schedule values at timestep t and broadcast to x_shape.
    values: [T]
    t:      [B]
    """
    out = values.to(device=t.device, dtype=torch.float32)[t].float()
    while out.dim() < len(x_shape):
        out = out[..., None]
    return out.expand(x_shape)


# -------------------------------------------------------------------------
# Building blocks
# -------------------------------------------------------------------------

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Sinusoidal timestep embedding.
    timesteps: [B]
    returns:   [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that chooses a valid number of groups."""
    def __init__(self, channels: int):
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        super().__init__(groups, channels)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = GroupNorm32(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = GroupNorm32(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class WindowAttentionBlock(nn.Module):
    """
    Lightweight Swin-like local self-attention block.
    It applies attention inside non-overlapping spatial windows.
    """
    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        self.channels = channels
        self.window_size = window_size
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, hp, wp = x.shape
        x_windows = (
            x.view(b, c, hp // ws, ws, wp // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(-1, ws * ws, c)
        )
        y = self.norm(x_windows)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x_windows = x_windows + attn_out
        x_windows = x_windows + self.ffn(x_windows)
        x = (
            x_windows.view(b, hp // ws, wp // ws, ws, ws, c)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(b, c, hp, wp)
        )
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x


class ConvBlock(nn.Module):
    """Simple conv block for the coarse RGB-to-HSI branch."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            GroupNorm32(out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            GroupNorm32(out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoarseHSINet(nn.Module):
    """
    Deterministic RGB-to-HSI coarse reconstructor.
    This provides S_c, the same-channel condition required by residual-shift diffusion.
    """
    def __init__(self, in_channels: int = 3, out_bands: int = 31, base_channels: int = 64):
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.block1 = ConvBlock(base_channels, base_channels)
        self.block2 = ConvBlock(base_channels, base_channels)
        self.block3 = ConvBlock(base_channels, base_channels)
        self.tail = nn.Conv2d(base_channels, out_bands, 3, padding=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        h = self.head(rgb)
        h = h + self.block1(h)
        h = h + self.block2(h)
        h = h + self.block3(h)
        # For normalized reflectance data in [0,1]. Remove sigmoid if training in [-1,1].
        return torch.sigmoid(self.tail(h))


class HSIResidualDiffusionUNet(nn.Module):
    """
    U-Net denoiser/refiner.

    condition_rgb=True:
        concat [x_t, coarse_hsi, rgb] -> [B, 2*C+3, H, W]
    condition_rgb=False:
        concat [x_t, coarse_hsi]      -> [B, 2*C, H, W]
    """
    def __init__(
        self,
        bands: int = 31,
        rgb_channels: int = 3,
        base_channels: int = 64,
        time_dim: Optional[int] = None,
        condition_rgb: bool = True,
        num_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.bands = bands
        self.condition_rgb = condition_rgb
        time_dim = time_dim or base_channels * 4
        in_ch = bands + bands + (rgb_channels if condition_rgb else 0)
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(in_ch, base_channels, 3, padding=1)
        self.enc1 = ResBlock(base_channels, base_channels, time_dim, dropout)
        self.attn1 = WindowAttentionBlock(base_channels, num_heads=num_heads, window_size=window_size)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1)
        self.enc2 = ResBlock(base_channels * 2, base_channels * 2, time_dim, dropout)
        self.attn2 = WindowAttentionBlock(base_channels * 2, num_heads=num_heads, window_size=window_size)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1)
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_dim, dropout)
        self.mid_attn = WindowAttentionBlock(base_channels * 4, num_heads=num_heads, window_size=window_size)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_dim, dropout)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2, time_dim, dropout)
        self.dec_attn2 = WindowAttentionBlock(base_channels * 2, num_heads=num_heads, window_size=window_size)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1)
        self.dec1 = ResBlock(base_channels * 2, base_channels, time_dim, dropout)
        self.dec_attn1 = WindowAttentionBlock(base_channels, num_heads=num_heads, window_size=window_size)
        self.out = nn.Sequential(GroupNorm32(base_channels), nn.SiLU(), nn.Conv2d(base_channels, bands, 3, padding=1))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        coarse_hsi: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.condition_rgb:
            if rgb is None:
                raise ValueError("rgb must be provided when condition_rgb=True.")
            x = torch.cat([x_t, coarse_hsi, rgb], dim=1)
        else:
            x = torch.cat([x_t, coarse_hsi], dim=1)
        time_emb = timestep_embedding(t, self.time_mlp[0].in_features)
        time_emb = self.time_mlp(time_emb)
        h = self.stem(x)
        e1 = self.attn1(self.enc1(h, time_emb))
        h = self.down1(e1)
        e2 = self.attn2(self.enc2(h, time_emb))
        h = self.down2(e2)
        h = self.mid1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, time_emb)
        h = self.up2(h)
        if h.shape[-2:] != e2.shape[-2:]:
            h = F.interpolate(h, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, e2], dim=1)
        h = self.dec_attn2(self.dec2(h, time_emb))
        h = self.up1(h)
        if h.shape[-2:] != e1.shape[-2:]:
            h = F.interpolate(h, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, e1], dim=1)
        h = self.dec_attn1(self.dec1(h, time_emb))
        return self.out(h)


# -------------------------------------------------------------------------
# Main RGB-to-HSI DifIISR-style model
# -------------------------------------------------------------------------

class RGB2HSI_DifIISR(nn.Module):
    """
    RGB-to-HSI model adapted from DifIISR's residual-shift diffusion idea.

    predict_type:
        "xstart"   -> denoiser predicts clean HSI S directly
        "epsilon"  -> denoiser predicts diffusion noise eps
        "residual" -> denoiser predicts coarse_hsi - clean_hsi
    """
    def __init__(
        self,
        bands: int = 31,
        base_channels: int = 64,
        steps: int = 15,
        kappa: float = 2.0,
        min_noise_level: float = 0.04,
        etas_end: float = 0.99,
        schedule_power: float = 0.3,
        predict_type: str = "xstart",
        condition_rgb: bool = True,
        num_heads: int = 4,
        window_size: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if predict_type not in {"xstart", "epsilon", "residual"}:
            raise ValueError("predict_type must be one of {'xstart', 'epsilon', 'residual'}.")
        self.bands = bands
        self.steps = steps
        self.kappa = float(kappa)
        self.predict_type = predict_type
        sqrt_etas = make_eta_schedule(
            steps=steps,
            min_noise_level=min_noise_level,
            etas_end=etas_end,
            kappa=kappa,
            power=schedule_power,
        )
        etas = sqrt_etas.pow(2)
        etas_prev = torch.cat([torch.zeros(1), etas[:-1]], dim=0)
        alpha = etas - etas_prev
        posterior_variance = (kappa ** 2) * etas_prev / etas.clamp_min(1e-12) * alpha
        self.register_buffer("sqrt_etas", sqrt_etas)
        self.register_buffer("etas", etas)
        self.register_buffer("etas_prev", etas_prev)
        self.register_buffer("alpha", alpha)
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-20))
        self.coarse_net = CoarseHSINet(in_channels=3, out_bands=bands, base_channels=base_channels)
        self.denoiser = HSIResidualDiffusionUNet(
            bands=bands,
            rgb_channels=3,
            base_channels=base_channels,
            condition_rgb=condition_rgb,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )

    def q_sample(
        self,
        hsi_gt: torch.Tensor,
        coarse_hsi: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from q(x_t | hsi_gt, coarse_hsi)."""
        if noise is None:
            noise = torch.randn_like(hsi_gt)
        eta_t = extract_to_shape(self.etas, t, hsi_gt.shape).to(hsi_gt.dtype)
        sqrt_eta_t = extract_to_shape(self.sqrt_etas, t, hsi_gt.shape).to(hsi_gt.dtype)
        x_t = (1.0 - eta_t) * hsi_gt + eta_t * coarse_hsi + self.kappa * sqrt_eta_t * noise
        return x_t, noise

    def predict_xstart_from_output(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        coarse_hsi: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert denoiser output to predicted clean HSI."""
        if self.predict_type == "xstart":
            return model_output
        if self.predict_type == "residual":
            return coarse_hsi - model_output
        eta_t = extract_to_shape(self.etas, t, x_t.shape).to(x_t.dtype)
        sqrt_eta_t = extract_to_shape(self.sqrt_etas, t, x_t.shape).to(x_t.dtype)
        return (x_t - eta_t * coarse_hsi - self.kappa * sqrt_eta_t * model_output) / (1.0 - eta_t).clamp_min(1e-6)

    def target_from_noise(self, hsi_gt: torch.Tensor, coarse_hsi: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Training target for the selected prediction parameterization."""
        if self.predict_type == "xstart":
            return hsi_gt
        if self.predict_type == "residual":
            return coarse_hsi - hsi_gt
        return noise

    def forward(
        self,
        rgb: torch.Tensor,
        hsi_gt: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        response_matrix: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        coarse_hsi = self.coarse_net(rgb)
        if hsi_gt is None:
            return {"coarse_hsi": coarse_hsi}
        b = hsi_gt.shape[0]
        if t is None:
            t = torch.randint(0, self.steps, (b,), device=hsi_gt.device, dtype=torch.long)
        x_t, noise = self.q_sample(hsi_gt=hsi_gt, coarse_hsi=coarse_hsi, t=t, noise=noise)
        model_output = self.denoiser(x_t=x_t, t=t, coarse_hsi=coarse_hsi, rgb=rgb)
        pred_hsi = self.predict_xstart_from_output(model_output, x_t, coarse_hsi, t)
        out = {
            "coarse_hsi": coarse_hsi,
            "x_t": x_t,
            "t": t,
            "model_output": model_output,
            "pred_hsi": pred_hsi,
            "noise": noise,
        }
        if return_loss:
            out.update(self.compute_losses(rgb, hsi_gt, coarse_hsi, pred_hsi, model_output, noise, response_matrix, loss_weights))
        return out

    def compute_losses(
        self,
        rgb: torch.Tensor,
        hsi_gt: torch.Tensor,
        coarse_hsi: torch.Tensor,
        pred_hsi: torch.Tensor,
        model_output: torch.Tensor,
        noise: torch.Tensor,
        response_matrix: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        weights = {
            "diffusion": 1.0,
            "coarse_l1": 0.2,
            "recon_l1": 0.5,
            "mrae": 0.2,
            "sam": 0.05,
            "spectral_grad": 0.05,
            "rgb": 0.0,
        }
        if loss_weights is not None:
            weights.update(loss_weights)
        target = self.target_from_noise(hsi_gt, coarse_hsi, noise)
        loss_diff = F.mse_loss(model_output, target)
        loss_coarse = F.l1_loss(coarse_hsi, hsi_gt)
        loss_l1 = F.l1_loss(pred_hsi, hsi_gt)
        loss_mrae = mrae_loss(pred_hsi, hsi_gt)
        loss_sam = sam_loss(pred_hsi, hsi_gt)
        loss_grad = spectral_gradient_loss(pred_hsi, hsi_gt)
        loss_rgb = pred_hsi.new_zeros(())
        if response_matrix is not None and weights.get("rgb", 0.0) > 0:
            loss_rgb = rgb_consistency_loss(pred_hsi.clamp(0, 1), rgb, response_matrix)
        total = (
            weights["diffusion"] * loss_diff
            + weights["coarse_l1"] * loss_coarse
            + weights["recon_l1"] * loss_l1
            + weights["mrae"] * loss_mrae
            + weights["sam"] * loss_sam
            + weights["spectral_grad"] * loss_grad
            + weights["rgb"] * loss_rgb
        )
        return {
            "loss": total,
            "loss_diffusion": loss_diff.detach(),
            "loss_coarse_l1": loss_coarse.detach(),
            "loss_recon_l1": loss_l1.detach(),
            "loss_mrae": loss_mrae.detach(),
            "loss_sam": loss_sam.detach(),
            "loss_spectral_grad": loss_grad.detach(),
            "loss_rgb": loss_rgb.detach(),
        }

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        coarse_hsi: torch.Tensor,
        rgb: torch.Tensor,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One reverse diffusion step."""
        model_output = self.denoiser(x_t=x_t, t=t, coarse_hsi=coarse_hsi, rgb=rgb)
        pred_x0 = self.predict_xstart_from_output(model_output, x_t, coarse_hsi, t)
        if clip_denoised:
            pred_x0 = pred_x0.clamp(0, 1)
        eta_t = extract_to_shape(self.etas, t, x_t.shape).to(x_t.dtype)
        eta_prev = extract_to_shape(self.etas_prev, t, x_t.shape).to(x_t.dtype)
        alpha_t = extract_to_shape(self.alpha, t, x_t.shape).to(x_t.dtype)
        posterior_var = extract_to_shape(self.posterior_variance, t, x_t.shape).to(x_t.dtype)
        mean = (eta_prev / eta_t.clamp_min(1e-12)) * x_t + (alpha_t / eta_t.clamp_min(1e-12)) * pred_x0
        noise = torch.randn_like(x_t)
        nonzero = (t != 0).float().view(-1, *([1] * (x_t.dim() - 1)))
        x_prev = mean + nonzero * torch.sqrt(posterior_var) * noise
        return x_prev, pred_x0

    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Full reverse sampling."""
        coarse_hsi = self.coarse_net(rgb)
        if noise is None:
            noise = torch.randn_like(coarse_hsi)
        t_start = torch.full((rgb.shape[0],), self.steps - 1, device=rgb.device, dtype=torch.long)
        sqrt_eta_T = extract_to_shape(self.sqrt_etas, t_start, coarse_hsi.shape).to(coarse_hsi.dtype)
        x_t = coarse_hsi + self.kappa * sqrt_eta_T * noise
        trajectory = []
        pred_x0 = coarse_hsi
        for step in reversed(range(self.steps)):
            t = torch.full((rgb.shape[0],), step, device=rgb.device, dtype=torch.long)
            x_t, pred_x0 = self.p_sample(x_t=x_t, t=t, coarse_hsi=coarse_hsi, rgb=rgb, clip_denoised=clip_denoised)
            if return_all:
                trajectory.append(pred_x0)
        out = {"hsi": pred_x0.clamp(0, 1) if clip_denoised else pred_x0, "coarse_hsi": coarse_hsi}
        if return_all:
            out["trajectory"] = torch.stack(trajectory, dim=0)
        return out

