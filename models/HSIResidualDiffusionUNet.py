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
