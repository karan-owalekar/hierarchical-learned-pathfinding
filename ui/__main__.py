"""Entry point for `python -m ui`."""

import argparse
from pathlib import Path

from hlp.config import Config
from ui.app import App


def main() -> None:
    parser = argparse.ArgumentParser(description="HLP Pathfinding Visualizer")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = Config.from_yaml(config_path) if config_path.exists() else Config()
    App(config).run()


if __name__ == "__main__":
    main()
