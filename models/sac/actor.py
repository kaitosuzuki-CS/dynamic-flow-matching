import torch
import torch.nn as nn
from torch.distributions import Normal

from .components import MLP, Encoder, get_t_emb


class Actor(nn.Module):
    def __init__(self, obs_shape, action_shape, hps):
        super().__init__()

        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._hps = hps

        self.t_emb_dim = hps.t_emb_dim

        self.encoder = Encoder(obs_shape=obs_shape, hps=hps.encoder)
        self.mlp = MLP(
            in_features=hps.encoder.output_dim + hps.t_emb_dim,
            out_features=2 * action_shape[0],
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

        print(f"Total Actor Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Actor Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def init_weights_with_ckpt(self, ckpt_path, freeze=False, device="cpu"):
        ckpt = torch.load(ckpt_path, map_location=device)["actor_state_dict"]
        missing_keys, unexpected_keys = self.load_state_dict(ckpt, strict=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        print(f"Missing Keys: {missing_keys}")
        print(f"Unexpected Keys: {unexpected_keys}")

        print(f"Total Actor Parameters: {sum(p.numel() for p in self.parameters())}")
        print(
            f"Trainable Actor Parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
        )

    def forward(self, x, t):
        x = self.encoder(x)
        t_emb = get_t_emb(t, self.t_emb_dim)

        x = torch.cat([x, t_emb], dim=1)
        x = self.mlp(x)

        mu, logvar = x.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar)

        dist = Normal(mu, std)
        raw_action = dist.rsample()
        action = torch.sigmoid(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob, mu
