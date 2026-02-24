#!/usr/bin/env python3
"""CLI entry point for training the v2 recursive corridor predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

from hlp.config import Config
from hlp.neural.train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the HLP recursive corridor predictor")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory for training data")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for model checkpoints")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or mps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--num-train", type=int, default=None,
                        help="Override number of training examples")
    parser.add_argument("--num-val", type=int, default=None,
                        help="Override number of validation examples")
    parser.add_argument("--teacher-epochs", type=int, default=None,
                        help="Override Phase 1 (teacher forcing) epochs")
    parser.add_argument("--adversarial-rounds", type=int, default=None,
                        help="Override Phase 2 adversarial mining rounds")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = Config.from_yaml(config_path) if config_path.exists() else Config()

    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.num_train is not None:
        config.train.num_train = args.num_train
    if args.num_val is not None:
        config.train.num_val = args.num_val
    if args.teacher_epochs is not None:
        config.train.teacher_epochs = args.teacher_epochs
    if args.adversarial_rounds is not None:
        config.train.adversarial_rounds = args.adversarial_rounds

    config.neural.checkpoint_path = str(Path(args.checkpoint_dir) / "best.pt")

    trainer = Trainer(config, device=args.device)
    trainer.run(data_dir=args.data_dir, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
