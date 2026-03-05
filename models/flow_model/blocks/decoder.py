import torch
import torch.nn as nn

from ..components import AttentionBlock, ResidualBlock


class DecoderLayer(nn.Module):
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
        upsample=False,
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
        self._upsample = upsample

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
            channels=out_channels, num_groups=num_groups, num_heads=num_heads
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

        if self._upsample:
            self.upsample_conv = nn.ConvTranspose2d(
                in_channels=in_channels // 2,
                out_channels=in_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        else:
            self.upsample_conv = nn.Identity()

    def forward(self, x, t_emb, skip_connection):
        x = self.upsample_conv(x)

        x = torch.cat([x, skip_connection], dim=1)
        x = self.residual_block(x, t_emb)
        x = self.attn_block(x)

        for layer in self.layers:
            x = layer[0](x, t_emb)  # type: ignore
            x = layer[1](x)  # type: ignore

        return x


class DecoderBlock(nn.Module):
    def __init__(self, t_emb_dim, hps):
        super().__init__()

        self._t_emb_dim = t_emb_dim
        self._hps = hps

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    in_channels=hps.in_channels[i],
                    out_channels=hps.out_channels[i],
                    t_emb_dim=t_emb_dim,
                    kernel_size=hps.kernel_size[i],
                    stride=hps.stride[i],
                    padding=hps.padding[i],
                    num_groups=hps.num_groups,
                    num_layers=hps.num_layers,
                    num_heads=hps.num_heads,
                    upsample=hps.upsample[i],
                )
                for i in range(len(hps.in_channels))
            ]
        )

    def forward(self, x, t_emb, skip_connections):
        for layer in self.layers:
            skip_connection = skip_connections.pop()
            x = layer(x, t_emb, skip_connection)

        return x


class Decoder(nn.Module):
    def __init__(self, im_channels, t_emb_dim, hps):
        super().__init__()

        self._im_channels = im_channels
        self._t_emb_dim = t_emb_dim
        self._hps = hps

        self.decoder_block = DecoderBlock(t_emb_dim=t_emb_dim, hps=hps)
        self.output_conv = nn.Sequential(
            nn.GroupNorm(
                num_groups=(
                    hps.num_groups
                    if hps.out_channels[-1] % hps.num_groups == 0
                    else hps.out_channels[-1]
                ),
                num_channels=hps.out_channels[-1],
            ),
            nn.Conv2d(
                in_channels=hps.out_channels[-1],
                out_channels=im_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x, t_emb, skip_connections):
        x = self.decoder_block(x, t_emb, skip_connections)
        x = self.output_conv(x)

        return x
