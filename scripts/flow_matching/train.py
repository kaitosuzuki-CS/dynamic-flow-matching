import argparse

import torch

from models import FlowMatching
from utils import create_dataset, load_config

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching on MNIST")
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/flow_matching/model_config.yml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--train-config-path",
        type=str,
        default="configs/flow_matching/train_config.yml",
        help="Path to the training config file.",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    train_loader, val_loader = create_dataset(train_hps.data)  # type: ignore
    model = FlowMatching(hps, train_hps, train_loader, val_loader, device)
    model.train()
