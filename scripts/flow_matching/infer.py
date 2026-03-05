import argparse

import torch

from models import FlowMatching
from utils import create_dataset, load_config

device = "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching on MNISt")
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
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the pt/pth checkpoint file",
    )
    parser.add_argument(
        "--num-samples", type=int, default=64, help="Number of samples to generate."
    )
    parser.add_argument(
        "--num-steps", type=int, default=100, help="Number of integration steps."
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        default=50,
        help="Frequency to record the integration progress.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save the generated samples.",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path
    ckpt_path = args.ckpt_path
    num_samples = args.num_samples
    num_steps = args.num_steps
    checkpoint_steps = args.checkpoint_steps
    results_dir = args.results_dir

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    # train_loader, val_loader = create_dataset(train_hps.data)  # type: ignore
    model = FlowMatching(hps, train_hps, None, None, device)

    model.infer(ckpt_path, num_samples, num_steps, checkpoint_steps, results_dir)
