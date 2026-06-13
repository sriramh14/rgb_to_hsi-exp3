
class GroupNorm32(nn.GroupNorm):
    """GroupNorm that chooses a valid number of groups."""
    def __init__(self, channels: int):
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        super().__init__(groups, channels)
