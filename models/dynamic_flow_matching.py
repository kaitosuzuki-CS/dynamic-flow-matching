import os
import time
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers.optimization import get_cosine_schedule_with_warmup

from env.env import Env
from env.replay_buffer import ReplayBuffer
from models.sac import Actor, SoftCritic
from utils import save_grid, set_seeds

parent_dir = Path(__file__).resolve().parent.parent


class DynamicFlowMatching:
    def __init__(self, hps, train_hps, device):

        self._hps = hps
        self._train_hps = train_hps
        self._device = device

        self._init_hyperparameters()

        self.env = Env(train_hps.env, hps.flow_model, self._device)
        self.obs_shape, self.action_shape = self._get_env_spec()

        self.actor = Actor(self.obs_shape, self.action_shape, hps.actor)

        self.critic1 = SoftCritic(self.obs_shape, self.action_shape, hps.critic)
        self.critic2 = SoftCritic(self.obs_shape, self.action_shape, hps.critic)

        self.target_critic1 = SoftCritic(self.obs_shape, self.action_shape, hps.critic)
        self.target_critic2 = SoftCritic(self.obs_shape, self.action_shape, hps.critic)

        self.target_entropy = -torch.prod(
            torch.Tensor(self.action_shape).to(device)
        ).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

    def _init_hyperparameters(self):
        self.optimizer_hps = self._train_hps.optimizer
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)

        self.actor_lr = float(self.optimizer_hps.actor_lr)
        self.critic_lr = float(self.optimizer_hps.critic_lr)
        self.alpha_lr = float(self.optimizer_hps.alpha_lr)

        self.actor_betas = tuple(
            map(float, getattr(self.optimizer_hps, "actor_betas", (0.9, 0.999)))
        )
        self.critic_betas = tuple(
            map(float, getattr(self.optimizer_hps, "critic_betas", (0.9, 0.999)))
        )
        self.alpha_betas = tuple(
            map(float, getattr(self.optimizer_hps, "alpha_betas", (0.9, 0.999)))
        )

        self.actor_weight_decay = float(
            getattr(self.optimizer_hps, "actor_weight_decay", 0)
        )
        self.critic_weight_decay = float(
            getattr(self.optimizer_hps, "critic_weight_decay", 0)
        )

        if self.scheduler_hps is not None:
            self.actor_warmup_steps = int(self.scheduler_hps.actor_warmup_steps)
            self.critic_warmup_steps = int(self.scheduler_hps.critic_warmup_steps)
            self.alpha_warmup_steps = int(self.scheduler_hps.alpha_warmup_steps)

        self.alpha = float(self._train_hps.alpha)
        self.gamma = float(self._train_hps.gamma)
        self.tau = float(self._train_hps.tau)

        self.batch_size = int(self._train_hps.batch_size)
        self.total_timesteps = int(self._train_hps.total_timesteps)
        self.warmup_steps = int(self._train_hps.warmup_steps)
        self.micro_steps = int(self._train_hps.micro_steps)
        self.update_freq = int(self._train_hps.update_freq)

        self.capacity = int(self._train_hps.capacity)

        self.checkpoint_dir = os.path.join(
            parent_dir, str(self._train_hps.checkpoint_dir)
        )
        self.checkpoint_freq = int(self._train_hps.checkpoint_freq)

        self.seed = getattr(self._train_hps, "seed", 42)

    def _init_training_scheme(self):
        self.actor_optim = Adam(
            self.actor.parameters(),
            lr=self.actor_lr,
            betas=self.actor_betas,  # type: ignore
            weight_decay=self.actor_weight_decay,
        )
        self.critic_optim = Adam(
            chain(self.critic1.parameters(), self.critic2.parameters()),
            lr=self.critic_lr,
            betas=self.critic_betas,  # type: ignore
            weight_decay=self.critic_weight_decay,
        )
        self.alpha_optim = Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=self.alpha_betas  # type:ignore
        )

        self.actor_scheduler = None
        self.critic_scheduler = None
        self.alpha_scheduler = None
        if self.scheduler_hps is not None:
            num_training_steps = self.total_timesteps - self.warmup_steps
            self.actor_scheduler = get_cosine_schedule_with_warmup(
                self.actor_optim,
                num_warmup_steps=self.actor_warmup_steps,
                num_training_steps=num_training_steps,
            )
            self.critic_scheduler = get_cosine_schedule_with_warmup(
                self.critic_optim,
                num_warmup_steps=self.critic_warmup_steps,
                num_training_steps=num_training_steps,
            )
            self.alpha_scheduler = get_cosine_schedule_with_warmup(
                self.alpha_optim,
                num_warmup_steps=self.alpha_warmup_steps,
                num_training_steps=num_training_steps,
            )

        self.replay_buffer = ReplayBuffer(
            self.capacity, self.obs_shape, self.action_shape, self._device
        )

    def _init_weights(self):
        self.actor.init_weights()
        self.critic1.init_weights()
        self.critic2.init_weights()
        self.target_critic1.init_target_weights(self.critic1)
        self.target_critic2.init_target_weights(self.critic2)

    def _init_weights_with_ckpt(self, ckpt_path, freeze=False):
        self.actor.init_weights_with_ckpt(ckpt_path, freeze)

    def _move_to_device(self, device):
        self.actor = self.actor.to(device)
        self.critic1 = self.critic1.to(device)
        self.critic2 = self.critic2.to(device)
        self.target_critic1 = self.target_critic1.to(device)
        self.target_critic2 = self.target_critic2.to(device)

        print(f"Moved to {device}")

    def _get_env_spec(self):
        return self.env.obs_shape, self.env.action_shape

    def _soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * target_param.data + (1.0 - tau) * param.data)

    def _save_checkpoint(self, updates, num_timesteps):
        checkpoint = {
            "updates": updates,
            "num_timesteps": num_timesteps,
            "actor_state_dict": self.actor.state_dict(),
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "target_critic1_state_dict": self.target_critic1.state_dict(),
            "target_critic2_state_dict": self.target_critic2.state_dict(),
            "log_alpha": self.log_alpha.detach(),
            "actor_optimizer_state_dict": self.actor_optim.state_dict(),
            "critic_optimizer_state_dict": self.critic_optim.state_dict(),
            "alpha_optimizer_state_dict": self.alpha_optim.state_dict(),
            "actor_scheduler_state_dict": (
                self.actor_scheduler.state_dict()
                if self.actor_scheduler is not None
                else None
            ),
            "critic_scheduler_state_dict": (
                self.critic_scheduler.state_dict()
                if self.critic_scheduler is not None
                else None
            ),
            "alpha_scheduler_state_dict": (
                self.alpha_scheduler.state_dict()
                if self.alpha_scheduler is not None
                else None
            ),
        }

        torch.save(
            checkpoint, os.path.join(self.checkpoint_dir, f"checkpoint_{updates}.pt")
        )

    def batch_update(self, buffer, updates):
        batch = buffer.sample(self.batch_size)
        obs, t, action, reward, next_obs, next_t, done = batch

        mask = (~done).float().unsqueeze(1)
        reward = reward.unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor(next_obs, next_t)
            qf1_next_target, qf2_next_target = self.target_critic1(
                next_obs, next_t, next_state_action
            ), self.target_critic2(next_obs, next_t, next_state_action)
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = reward + (mask * self.gamma * min_qf_next_target)

        qf1, qf2 = self.critic1(obs, t, action), self.critic2(obs, t, action)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad(set_to_none=True)
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.actor(obs, t)
        qf1_pi, qf2_pi = self.critic1(obs, t, pi), self.critic2(obs, t, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp()

        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        if self.alpha_scheduler is not None:
            self.alpha_scheduler.step()

        if updates % self.update_freq == 0:
            self._soft_update(self.target_critic1, self.critic1, self.tau)
            self._soft_update(self.target_critic2, self.critic2, self.tau)

    def train(self):
        set_seeds(self.seed)
        self._init_weights()
        self._move_to_device(self._device)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_training_scheme()

        num_timesteps = 0
        updates = 0
        total_reward = 0
        total_mse = 0
        total_dt = 0
        while num_timesteps < self.total_timesteps:
            obs, t, _, _, _ = self.env.reset()

            for _ in range(self.micro_steps):
                if num_timesteps < self.warmup_steps:
                    action = torch.rand(
                        (self.env.batch_size, *self.action_shape), device=self._device
                    )
                else:
                    with torch.no_grad():
                        action, _, _ = self.actor(obs, t)
                        action = action.detach()

                next_obs, next_t, reward, done, info = self.env.step(action)

                self.replay_buffer.push(obs, t, action, reward, next_obs, next_t, done)
                obs = next_obs.clone().detach()
                t = next_t.clone().detach()

                num_timesteps += 1

                total_reward += info["reward"]
                total_mse += info["mse"]
                total_dt += info["avg_dt"]

                if num_timesteps >= self.warmup_steps:
                    updates += 1
                    self.batch_update(self.replay_buffer, updates)

                    if updates % self.checkpoint_freq == 0:
                        self._save_checkpoint(updates, num_timesteps)

                if num_timesteps % 1000 == 0:
                    total_reward /= 1000
                    total_mse /= 1000
                    total_dt /= 1000
                    print(
                        f"Num Timesteps: {num_timesteps}, Reward: {total_reward}, MSE: {total_mse}, Average dt: {total_dt}"
                    )
                    print(
                        f"Actor LR: {self.actor_optim.param_groups[0]['lr']}, Critic LR: {self.critic_optim.param_groups[0]['lr']}, Alpha LR: {self.alpha_optim.param_groups[0]['lr']}"
                    )
                    total_reward = 0
                    total_mse = 0
                    total_dt = 0

                if num_timesteps >= self.total_timesteps:
                    break

    def infer(self, ckpt_path, batch_size=64, results_dir="results"):
        set_seeds(self.seed)
        ckpt_path = os.path.join(parent_dir, ckpt_path)
        self._init_weights_with_ckpt(ckpt_path, freeze=True)
        self._move_to_device(self._device)
        self.actor.eval()

        flow_model = self.env.model.to(self._device)
        flow_model.eval()

        with torch.no_grad():
            xt = torch.randn((batch_size, *self.env.obs_shape), device=self._device)
            t = torch.zeros((batch_size,), device=self._device)

            generated_samples = []

            start_time = time.time()
            while True:
                action, _, _ = self.actor(xt, t)
                dt = torch.min(action, (1.0 - t).unsqueeze(1)).squeeze()
                t_next = t + dt

                v_theta = flow_model(xt, t)
                dt_view = dt.view(-1, 1, 1, 1)
                xt_next = xt + dt_view * v_theta

                done = t_next >= 1.0

                generated_samples.append(xt_next[done].cpu())

                xt = xt_next[~done]
                t = t_next[~done]

                if done.all():
                    break

            end_time = time.time()
            print(f"Inference Time: {end_time - start_time}")

        generated_samples = torch.cat(generated_samples, dim=0)

        results_dir = os.path.join(parent_dir, results_dir)
        os.makedirs(results_dir, exist_ok=True)
        save_grid(
            generated_samples, 8, os.path.join(results_dir, "generated_samples.pdf")
        )

        return generated_samples

    def evaluate(self, ckpt_path, batch_size=64, results_dir="results"):
        set_seeds(self.seed)
        ckpt_path = os.path.join(parent_dir, ckpt_path)
        self._init_weights_with_ckpt(ckpt_path, freeze=True)
        self._move_to_device(self._device)
        self.actor.eval()

        with torch.no_grad():
            obs, t, _, _, _, _, _ = self.env.reset_zero(batch_size)

            generated_samples = []
            batch_t = []
            batch_dt = []
            batch_mse = []

            while True:
                action, _, _ = self.actor(obs, t)
                next_obs, next_t, completed_samples, t, dt, mse, done = self.env.infer(
                    action, batch_size
                )

                generated_samples.append(completed_samples.cpu())
                batch_t.append(t.cpu())
                batch_dt.append(dt.cpu())
                batch_mse.append(mse.cpu())

                obs = next_obs.clone().detach()
                t = next_t.clone().detach()

                if done.all():
                    break

        batch_t = [t.unsqueeze(0) if t.dim() == 0 else t for t in batch_t]
        batch_dt = [t.unsqueeze(0) if t.dim() == 0 else t for t in batch_dt]
        batch_mse = [mse.unsqueeze(0) if mse.dim() == 0 else mse for mse in batch_mse]

        generated_samples = torch.cat(generated_samples, dim=0)
        batch_t = torch.cat(batch_t, dim=0)
        batch_dt = torch.cat(batch_dt, dim=0)
        batch_mse = torch.cat(batch_mse, dim=0)

        results_dir = os.path.join(parent_dir, results_dir)
        os.makedirs(results_dir, exist_ok=True)
        torch.save(
            {
                "generated_samples": generated_samples,
                "batch_t": batch_t,
                "batch_dt": batch_dt,
                "batch_mse": batch_mse,
            },
            os.path.join(results_dir, "evaluation_stats.pt"),
        )

        return generated_samples, batch_t, batch_dt, batch_mse
