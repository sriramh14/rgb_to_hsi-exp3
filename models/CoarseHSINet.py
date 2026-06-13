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

