import argparse

import torch

from models import DynamicFlowMatching
from utils import load_config

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
        help="Path to the pt/pth checkpoint file for the SAC policy.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=64, help="Number of samples to generate."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save the generated samples.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["infer", "evaluate"],
        default="infer",
        help="Run inference or evaluation on the model (infer/evaluate)",
    )

    args = parser.parse_args()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path
    ckpt_path = args.ckpt_path
    num_samples = args.num_samples
    results_dir = args.results_dir
    mode = args.mode

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)

    model = DynamicFlowMatching(hps, train_hps, device)
    if mode == "infer":
        model.infer(ckpt_path, num_samples, results_dir)
    elif mode == "evaluate":
        model.evaluate(ckpt_path, num_samples, results_dir)
