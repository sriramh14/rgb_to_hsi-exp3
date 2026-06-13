import torch
def mrae_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Mean relative absolute error. pred/target: [B,C,H,W]."""
    return (pred - target).abs().div(target.abs() + eps).mean()
