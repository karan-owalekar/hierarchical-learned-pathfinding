from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class GridConfig:
    height: int = 256
    width: int = 256
    obstacle_density: float = 0.2
    seed: Optional[int] = None


@dataclass
class BlockConfig:
    block_size: int = 16


@dataclass
class NeuralConfig:
    d: int = 64
    max_levels: int = 12
    grid_resolution: int = 8
    checkpoint_path: str = "checkpoints/best.pt"


@dataclass
class TrainConfig:
    num_train: int = 50000
    num_val: int = 5000
    batch_size: int = 64
    min_path_distance: int = 10
    teacher_epochs: int = 30
    lr_teacher: float = 1e-3
    adversarial_rounds: int = 5
    adversarial_queries: int = 10000
    lr_adversarial: float = 1e-4
    pos_weight: float = 10.0
    early_stop_patience: int = 5


@dataclass
class InferenceConfig:
    mode: str = "hybrid"
    activation_threshold: float = 0.3
    verify_optimality: bool = False


@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        cfg = Config()
        for section_name, section_cls in [
            ("grid", GridConfig),
            ("block", BlockConfig),
            ("neural", NeuralConfig),
            ("train", TrainConfig),
            ("inference", InferenceConfig),
        ]:
            if section_name in raw:
                setattr(cfg, section_name, section_cls(**raw[section_name]))
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
