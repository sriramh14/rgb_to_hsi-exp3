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
