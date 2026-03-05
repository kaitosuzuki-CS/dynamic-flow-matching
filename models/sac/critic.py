import math

import torch
import torch.nn as nn

from .components import MLP, Encoder, get_t_emb


class SoftCritic(nn.Module):
    def __init__(self, obs_shape, action_shape, hps):
        super().__init__()

        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._hps = hps

        self.t_emb_dim = hps.t_emb_dim

        self.encoder = Encoder(obs_shape=obs_shape, hps=hps.encoder)
        self.mlp = MLP(
            in_features=hps.encoder.output_dim + hps.t_emb_dim + action_shape[0],
            out_features=1,
            hidden_features=hps.hidden_dim,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        print(f"Total Critic Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Critic Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_target_weights(self, online_model, freeze=False):
        self.load_state_dict(online_model.state_dict())

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        print(
            f"Total Target Critic Parameters: {sum(p.numel() for p in self.parameters())}"
        )
        print(
            f"Trainable Target Critic Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x, t, a):
        x = self.encoder(x)
        t_emb = get_t_emb(t, self.t_emb_dim)

        x = torch.cat([x, t_emb, a], dim=1)
        x = self.mlp(x)

        return x
