import torch
from models.project_hsi_to_rgb import project_hsi_to_rgb
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
