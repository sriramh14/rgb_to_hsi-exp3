import torch
import torch.nn as nn
from .CoarseHSINET import CoarseHSINET
from .HSIResidualDiffusionUNet import HSIResidualDiffusionUNet

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
