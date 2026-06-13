import torch
import torch.nn as nn
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
