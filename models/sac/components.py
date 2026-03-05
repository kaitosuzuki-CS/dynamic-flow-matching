import math

import torch
import torch.nn as nn


def get_t_emb(t, t_emb_dim, max_positions=10000):
    t = t * max_positions
    half_dim = t_emb_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=1)
    if t_emb_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb)[:, :1]], dim=1)

    return emb


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super().__init__()

        self._in_features = in_features
        self._out_features = out_features
        self._hidden_features = hidden_features

        self.in_layer = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.in_relu = nn.ReLU()

        self.out_layer = nn.Linear(
            in_features=hidden_features, out_features=out_features
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.in_relu(x)

        x = self.out_layer(x)

        return x


class Encoder(nn.Module):
    def __init__(self, obs_shape, hps):
        super().__init__()

        self._obs_shape = obs_shape
        self._hps = hps

        C, H, W = obs_shape
        self.in_conv = nn.Conv2d(
            in_channels=C,
            out_channels=hps.latent_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.in_relu = nn.ReLU()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=hps.latent_dim,
                        out_channels=hps.latent_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.ReLU(),
                )
                for _ in range(hps.num_layers - 1)
            ]
        )

        self.mlp = MLP(
            in_features=hps.latent_dim * (H // 2) * (W // 2),
            out_features=hps.output_dim,
            hidden_features=hps.hidden_dim,
        )
        self.output_layer = nn.Sequential(nn.LayerNorm(hps.output_dim), nn.Tanh())

    def forward(self, x):
        x = self.in_conv(x)
        x = self.in_relu(x)

        for layer in self.layers:
            x = layer(x)

        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        x = self.output_layer(x)

        return x

        return x
