import argparse

import torch

from env import generate_dopri5_ground_truth
from utils import load_config

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
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to the pt/pth checkpoint file",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/dopri5_trajectories.pt",
        help="Path to pt/pth file to save the baseline trajectories.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples to generate the baseline trajectory for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for the generation process.",
    )
    parser.add_argument(
        "--num-eval-steps",
        type=int,
        default=100,
        help="Number of integration steps for the dopri5 solver.",
    )

    parser.add_argument

    args = parser.parse_args()
    model_config_path = args.model_config_path
    ckpt_path = args.ckpt_path
    save_path = args.save_path
    num_samples = args.num_samples
    batch_size = args.batch_size
    num_eval_steps = args.num_eval_steps

    hps = load_config(model_config_path)
    generate_dopri5_ground_truth(
        hps,
        ckpt_path,
        num_samples,
        batch_size,
        num_eval_steps,
        device,
        save_path,
    )
