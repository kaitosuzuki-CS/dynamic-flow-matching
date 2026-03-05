import torch
import torch.nn as nn

from ..components import AttentionBlock, ResidualBlock


class BottleneckLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        t_emb_dim,
        kernel_size,
        stride,
        padding,
        num_groups,
        num_layers,
        num_heads,
    ):
        super().__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._t_emb_dim = t_emb_dim
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._num_groups = num_groups
        self._num_layers = num_layers
        self._num_heads = num_heads

        self.residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_groups=num_groups,
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        AttentionBlock(
                            channels=out_channels,
                            num_groups=num_groups,
                            num_heads=num_heads,
                        ),
                        ResidualBlock(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            t_emb_dim=t_emb_dim,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            num_groups=num_groups,
                        ),
                    ]
                )
                for _ in range(num_layers - 1)
            ]
        )

    def forward(self, x, t_emb):
        x = self.residual_block(x, t_emb)

        for layer in self.layers:
            x = layer[0](x)  # type: ignore
            x = layer[1](x, t_emb)  # type:ignore

        return x


class Bottleneck(nn.Module):
    def __init__(self, t_emb_dim, hps):
        super().__init__()

        self._t_emb_dim = t_emb_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                BottleneckLayer(
                    in_channels=hps.in_channels[i],
                    out_channels=hps.out_channels[i],
                    t_emb_dim=t_emb_dim,
                    kernel_size=hps.kernel_size[i],
                    stride=hps.stride[i],
                    padding=hps.padding[i],
                    num_groups=hps.num_groups,
                    num_layers=hps.num_layers,
                    num_heads=hps.num_heads,
                )
                for i in range(len(hps.in_channels))
            ]
        )

    def forward(self, x, t_emb):
        for layer in self.layers:
            x = layer(x, t_emb)

        return x
