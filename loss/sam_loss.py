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
