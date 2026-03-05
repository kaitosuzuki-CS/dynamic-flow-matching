import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        kernel_size,
        stride,
        padding,
        num_groups,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._t_emb_dim = t_emb_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._num_groups = num_groups

        self.conv1 = nn.Sequential(
            nn.GroupNorm(
                num_groups=num_groups if in_channels % num_groups == 0 else in_channels,
                num_channels=in_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.t_emb_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(in_features=t_emb_dim, out_features=out_channels)
        )

        self.conv2 = nn.Sequential(
            nn.GroupNorm(
                num_groups=(
                    num_groups if out_channels % num_groups == 0 else out_channels
                ),
                num_channels=out_channels,
            ),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.residual_input_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, t_emb):
        B, C, H, W = x.shape

        _x = self.conv1(x)
        _x = _x + self.t_emb_layer(t_emb)[:, :, None, None]
        _x = self.conv2(_x)
        x = _x + self.residual_input_conv(x)

        return x
