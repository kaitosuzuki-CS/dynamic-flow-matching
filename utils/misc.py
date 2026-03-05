import copy
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torchvision.utils import make_grid, save_image

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

parent_dir = Path(__file__).resolve().parent.parent


class HPS:
    def __init__(self, hps):
        for k, v in hps.items():
            if isinstance(v, dict):
                setattr(self, k, HPS(v))
            else:
                setattr(self, k, v)


def load_config(config_path):
    config_path = os.path.join(parent_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return HPS(config)


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

        self.best_loss = float("inf")
        self.best_model = None
        self.stop = False

    def __call__(self, model, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def plot_grid(img, nrow):
    img = torch.clamp(img, 0, 1)
    grid = make_grid(img, nrow=nrow)
    npimg = grid.cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_grid(img, nrow, save_path):
    img = torch.clamp(img, 0, 1)
    grid = make_grid(img, nrow=nrow)

    save_image(grid, save_path)


# def plot_grid(img, nrow, col_names=None):

#     img = torch.clamp(img, 0.0, 1.0)

#     n_images = img.size(0)
#     ncol = nrow
#     nrows = math.ceil(n_images / ncol)

#     fig, axes = plt.subplots(nrows, ncol, figsize=(2 * ncol, 2 * nrows))

#     if nrows == 1:
#         axes = np.expand_dims(axes, axis=0)  # type: ignore

#     for idx in range(n_images):
#         row = idx // ncol
#         col = idx % ncol

#         npimg = img[idx].cpu().numpy()
#         if npimg.shape[0] == 1:
#             axes[row, col].imshow(npimg.squeeze(), cmap="gray", vmin=0, vmax=1)  # type: ignore
#         else:
#             axes[row, col].imshow(np.transpose(npimg, (1, 2, 0)))  # type: ignore
#         axes[row, col].axis("off")  # type: ignore

#         if row == 0 and col_names is not None:
#             if col == 0:
#                 axes[row, col].set_title(col_names[col], fontsize=14, fontweight="bold")  # type: ignore
#             else:
#                 axes[row, col].set_title(col_names[col], fontsize=14)  # type: ignore

#     for idx in range(n_images, nrows * ncol):
#         row = idx // ncol
#         col = idx % ncol
#         axes[row, col].axis("off")  # type: ignore

#     fig.subplots_adjust(
#         left=0,
#         right=1,
#         top=1,
#         bottom=0,
#         wspace=0,
#         hspace=0,
#     )
#     plt.subplots_adjust(wspace=0, hspace=0)

#     plt.show()


if __name__ == "__main__":
    img = torch.randn(64, 1, 32, 32)
    grid = make_grid(img, 8)

    save_image(grid, "results.pdf")
