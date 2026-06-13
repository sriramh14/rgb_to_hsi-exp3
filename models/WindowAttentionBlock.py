class WindowAttentionBlock(nn.Module):
    """
    Lightweight Swin-like local self-attention block.
    It applies attention inside non-overlapping spatial windows.
    """
    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        self.channels = channels
        self.window_size = window_size
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        _, _, hp, wp = x.shape
        x_windows = (
            x.view(b, c, hp // ws, ws, wp // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(-1, ws * ws, c)
        )
        y = self.norm(x_windows)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x_windows = x_windows + attn_out
        x_windows = x_windows + self.ffn(x_windows)
        x = (
            x_windows.view(b, hp // ws, wp // ws, ws, ws, c)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(b, c, hp, wp)
        )
        if pad_h or pad_w:
            x = x[:, :, :h, :w]
        return x
