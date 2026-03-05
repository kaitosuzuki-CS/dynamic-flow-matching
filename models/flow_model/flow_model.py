import math

import torch
import torch.nn as nn

from .blocks import Bottleneck, Decoder, Encoder


class FlowModel(nn.Module):
    def __init__(self, hps):
        super().__init__()

        self._hps = hps
        self.im_channels, _, _ = hps.im_shape
        self.t_emb_dim = hps.t_emb_dim

        self.t_proj = nn.Sequential(
            nn.Linear(in_features=self.t_emb_dim, out_features=self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(in_features=self.t_emb_dim, out_features=self.t_emb_dim),
        )

        self.encoder = Encoder(
            im_channels=self.im_channels, t_emb_dim=self.t_emb_dim, hps=hps.encoder
        )
        self.bottleneck = Bottleneck(t_emb_dim=self.t_emb_dim, hps=hps.bottleneck)
        self.decoder = Decoder(
            im_channels=self.im_channels, t_emb_dim=self.t_emb_dim, hps=hps.decoder
        )

    def _get_t_emb(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.t_emb_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        if self.t_emb_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb)[:, :1]], dim=1)

        return emb

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_weights_with_ckpt(self, ckpt_path, freeze=False, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)["model_state_dict"]
        missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")

        print(f"Total Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x, t):
        t_emb = self._get_t_emb(t)
        t_emb = self.t_proj(t_emb)

        x, skip_connections = self.encoder(x, t_emb)
        x = self.bottleneck(x, t_emb)
        x = self.decoder(x, t_emb, skip_connections)

        return x
