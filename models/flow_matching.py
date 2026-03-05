import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup

from models.flow_model import FlowModel
from utils import EarlyStopping, save_grid, set_seeds

parent_dir = Path(__file__).resolve().parent.parent


class FlowMatching:
    def __init__(self, hps, train_hps, train_loader, val_loader, device):

        self._hps = hps
        self._train_hps = train_hps
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._device = device

        self._init_hyperparameters()

        self.model = FlowModel(hps)

    def _init_hyperparameters(self):
        self.optimizer_hps = self._train_hps.optimizer
        self.scheduler_hps = getattr(self._train_hps, "scheduler", None)
        self.early_stopping_hps = getattr(self._train_hps, "early_stopping", None)

        self.lr = float(self.optimizer_hps.lr)
        self.betas = tuple(
            map(float, getattr(self.optimizer_hps, "betas", (0.9, 0.999)))
        )
        self.weight_decay = float(getattr(self.optimizer_hps, "weight_decay", 0))

        if self.scheduler_hps is not None:
            self.warmup_epochs = int(self.scheduler_hps.warmup_epochs)

        if self.early_stopping_hps is not None:
            self.patience = float(self.early_stopping_hps.patience)
            self.min_delta = float(self.early_stopping_hps.min_delta)

        self.num_epochs = int(self._train_hps.num_epochs)
        self.accum_steps = int(self._train_hps.accum_steps)

        self.checkpoint_dir = os.path.join(
            parent_dir, str(self._train_hps.checkpoint_dir)
        )
        self.checkpoint_freq = int(self._train_hps.checkpoint_freq)

        self.seed = getattr(self._train_hps, "seed", 42)

    def _init_training_scheme(self):
        self.optim = Adam(
            self.model.parameters(),
            lr=self.lr,
            betas=self.betas,  # type: ignore
            weight_decay=self.weight_decay,
        )

        self.scheduler = None
        if self.scheduler_hps is not None:
            num_training_steps = self.num_epochs * np.ceil(
                len(self._train_loader) / self.accum_steps
            )
            num_warmup_steps = self.warmup_epochs * np.ceil(
                len(self._train_loader) / self.accum_steps
            )
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optim,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        self.early_stopping = None
        if self.early_stopping_hps is not None:
            self.early_stopping = EarlyStopping(
                patience=self.patience, min_delta=self.min_delta
            )

    def _init_weights(self):
        self.model.init_weights()

    def _init_weights_with_ckpt(self, ckpt_path, freeze=False):
        self.model.init_weights_with_ckpt(ckpt_path, freeze)

    def _move_to_device(self, device):
        self.model = self.model.to(device)
        print(f"Moved to {device}")

    def train(self):
        set_seeds(self.seed)
        self._init_weights()
        self._move_to_device(self._device)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._init_training_scheme()

        best_model = None
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            self.optim.zero_grad(set_to_none=True)

            total_loss = 0.0
            num_batches = 0
            for x1, _ in tqdm(self._train_loader, leave=False):
                num_batches += 1
                batch_size = x1.shape[0]

                x1 = x1.to(self._device)
                x0 = torch.randn_like(x1)
                target = x1 - x0

                t = torch.rand(batch_size, device=self._device)
                xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1

                pred = self.model(xt, t)
                loss = ((target - pred) ** 2).mean()
                loss = loss / self.accum_steps

                loss.backward()

                if num_batches % self.accum_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

                total_loss += loss.item() * self.accum_steps

            if num_batches % self.accum_steps != 0:
                self.optim.step()
                self.optim.zero_grad(set_to_none=True)

                if self.scheduler is not None:
                    self.scheduler.step()

            with torch.no_grad():
                self.model.eval()

                val_loss = 0.0
                for x1, _ in tqdm(self._val_loader, leave=False):
                    batch_size = x1.shape[0]

                    x1 = x1.to(self._device)
                    x0 = torch.randn_like(x1)
                    target = x1 - x0

                    t = torch.rand(batch_size, device=self._device)
                    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1

                    pred = self.model(xt, t)
                    loss = ((target - pred) ** 2).mean()

                    val_loss += loss.item()

            total_loss /= len(self._train_loader)
            val_loss /= len(self._val_loader)

            print(f"----Epoch {epoch}----")
            print(f"Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"LR: {self.optim.param_groups[0]['lr']:.4f}")

            if epoch % self.checkpoint_freq == 0:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optim.state_dict(),
                        "scheduler_state_dict": (
                            self.scheduler.state_dict()
                            if self.scheduler is not None
                            else None
                        ),
                        "loss": val_loss,
                    },
                    os.path.join(self.checkpoint_dir, f"checkpoint_{epoch}.pt"),
                )

            if self.early_stopping is not None:
                self.early_stopping(self.model, val_loss)
                if self.early_stopping.stop:
                    best_model = self.early_stopping.best_model
                    break

        if best_model is None:
            best_model = self.model

        torch.save(
            {"model_state_dict": best_model.state_dict()},
            os.path.join(self.checkpoint_dir, "best_model.pt"),
        )

        return best_model

    def infer(
        self,
        ckpt_path,
        batch_size=64,
        num_steps=100,
        checkpoint_steps=50,
        results_dir="results",
    ):
        set_seeds(self.seed)
        ckpt_path = os.path.join(parent_dir, ckpt_path)
        self._init_weights_with_ckpt(ckpt_path, freeze=True)
        self._move_to_device(self._device)

        self.model.eval()

        with torch.no_grad():
            t = torch.linspace(0, 1, num_steps, device=self._device)
            x = torch.randn(batch_size, *self._hps.im_shape, device=self._device)
            dt = 1.0 / num_steps

            traj = []
            start_time = time.time()
            for i, _t in tqdm(enumerate(t)):
                _t = _t.expand(batch_size)
                pred = self.model(x, _t)

                x = x + dt * pred

                if (i + 1) % checkpoint_steps == 0:
                    traj.append(x.detach().cpu())

            end_time = time.time()
            print(f"Inference Time: {end_time - start_time} seconds")

        results_dir = os.path.join(parent_dir, results_dir)
        os.makedirs(results_dir, exist_ok=True)

        save_grid(x, 8, os.path.join(results_dir, "generated_samples.pdf"))
        for i, xt in enumerate(traj):
            save_grid(
                xt, 8, os.path.join(results_dir, f"{checkpoint_steps * (i + 1)}.pdf")
            )

        return x, traj
