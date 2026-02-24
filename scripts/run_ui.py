#!/usr/bin/env python3
"""Launch the Hierarchical Learned Pathfinding UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from hlp.config import Config
from ui.app import App


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the HLP visualization UI")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = Config.from_yaml(config_path) if config_path.exists() else Config()

    app = App(config)
    app.run()


if __name__ == "__main__":
    main()
