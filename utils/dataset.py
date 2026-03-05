import os
from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

parent_dir = Path(__file__).resolve().parent.parent


def create_dataset(hps):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    data_dir = os.path.join(parent_dir, getattr(hps, "data_dir", "data"))

    train_dataset = MNIST(root=data_dir, train=True, transform=transform, download=True)
    val_dataset = MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(hps.train_bs),
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(hps.val_bs),
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    return train_loader, val_loader
