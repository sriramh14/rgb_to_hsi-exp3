def spectral_gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """First-order spectral derivative loss along channel/wavelength dimension."""
    dp = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    dt = target[:, 1:, :, :] - target[:, :-1, :, :]
    return F.l1_loss(dp, dt)
