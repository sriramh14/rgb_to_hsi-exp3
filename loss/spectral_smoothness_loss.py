def spectral_smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    """Second-order spectral smoothness prior. Can be used at inference without GT."""
    d1 = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    d2 = d1[:, 1:, :, :] - d1[:, :-1, :, :]
    return d2.abs().mean()
