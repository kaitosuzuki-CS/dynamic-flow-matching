import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_groups, num_heads):
        super().__init__()

        self._channels = channels
        self._num_groups = num_groups
        self._num_heads = num_heads

        self.norm = nn.GroupNorm(
            num_groups=num_groups if channels % num_groups == 0 else channels,
            num_channels=channels,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape

        _x = x.reshape(B, C, H * W)
        _x = self.norm(_x)

        _x = _x.transpose(1, 2).contiguous()
        _x, _ = self.attn(_x, _x, _x)

        _x = _x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        x = x + _x

        return x
