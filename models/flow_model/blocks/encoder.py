import torch
import torch.nn as nn

from ..components import AttentionBlock, ResidualBlock


class EncoderLayer(nn.Module):
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
        downsample=False,
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
        self._downsample = downsample

        self.residual_block = ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            t_emb_dim=t_emb_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_groups=num_groups,
        )
        self.attn_block = AttentionBlock(
            channels=out_channels,
            num_groups=num_groups,
            num_heads=num_heads,
        )

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ResidualBlock(
                            in_channels=out_channels,
                            out_channels=out_channels,
                            t_emb_dim=t_emb_dim,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            num_groups=num_groups,
                        ),
                        AttentionBlock(
                            channels=out_channels,
                            num_groups=num_groups,
                            num_heads=num_heads,
                        ),
                    ]
                )
                for _ in range(num_layers - 1)
            ]
        )

        if self._downsample:
            self.downsample_conv = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            self.downsample_conv = nn.Identity()

    def forward(self, x, t_emb):
        x = self.residual_block(x, t_emb)
        x = self.attn_block(x)

        for layer in self.layers:
            x = layer[0](x, t_emb)  # type: ignore
            x = layer[1](x)  # type: ignore

        x = self.downsample_conv(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, t_emb_dim, hps):
        super().__init__()

        self._t_emb_dim = t_emb_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    in_channels=hps.in_channels[i],
                    out_channels=hps.out_channels[i],
                    t_emb_dim=t_emb_dim,
                    kernel_size=hps.kernel_size[i],
                    stride=hps.stride[i],
                    padding=hps.padding[i],
                    num_groups=hps.num_groups,
                    num_layers=hps.num_layers,
                    num_heads=hps.num_heads,
                    downsample=hps.downsample[i],
                )
                for i in range(len(hps.in_channels))
            ]
        )

    def forward(self, x, t_emb):
        skip_connections = []

        for layer in self.layers:
            skip_connections.append(x)
            x = layer(x, t_emb)

        return x, skip_connections


class Encoder(nn.Module):
    def __init__(self, im_channels, t_emb_dim, hps):
        super().__init__()

        self._im_channels = im_channels
        self._t_emb_dim = t_emb_dim
        self._hps = hps

        self.in_conv = nn.Conv2d(
            in_channels=im_channels,
            out_channels=hps.in_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.encoder_block = EncoderBlock(t_emb_dim=t_emb_dim, hps=hps)

    def forward(self, x, t_emb):
        x = self.in_conv(x)
        x, skip_connections = self.encoder_block(x, t_emb)

        return x, skip_connections
