"""Dataset generation for the QuadTreeConvNet corridor predictor.

Flat examples contain an 8×8 downsampled obstacle density map, normalized
source/goal positions, level index, and 4-element activation label.
Recursive examples store full (grid, source, goal, path) queries for
end-to-end validation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from hlp.grid import Cell, Grid, bfs_shortest_path, generate_grid
from hlp.neural.model import GRID_RES, downsample_block


# ---------------------------------------------------------------------------
# Flat per-region dataset (Phase 1 teacher forcing)
# ---------------------------------------------------------------------------

class FlatDataset(Dataset):
    """Pre-extracted per-node examples with 8×8 grids and positions."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.samples: list[str] = []
        if self.data_dir.exists():
            self.samples = sorted(f.stem for f in self.data_dir.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        data = np.load(self.data_dir / f"{self.samples[idx]}.npz")
        return {
            "grid_8x8": torch.from_numpy(data["grid_8x8"].astype(np.float32)),
            "positions": torch.from_numpy(data["positions"].astype(np.float32)),
            "level": torch.tensor(int(data["level"]), dtype=torch.long),
            "activation": torch.from_numpy(data["activation"].astype(np.float32)),
        }


# ---------------------------------------------------------------------------
# Recursive dataset (end-to-end validation)
# ---------------------------------------------------------------------------

class RecursiveDataset(Dataset):
    """Full (grid, source, goal, path) queries for end-to-end validation."""

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)
        self.samples: list[str] = []
        if self.data_dir.exists():
            self.samples = sorted(f.stem for f in self.data_dir.glob("*.npz"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        data = np.load(self.data_dir / f"{self.samples[idx]}.npz")
        return {
            "grid": data["grid"],
            "source": data["source"],
            "goal": data["goal"],
            "path": data["path"],
        }


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _get_quad_idx(r: int, c: int, mr: int, mc: int) -> int:
    return (0 if r < mr else 2) + (0 if c < mc else 1)


# ---------------------------------------------------------------------------
# Flat label extraction from one (grid, path, source, goal) example
# ---------------------------------------------------------------------------

def extract_flat_labels(
    padded: np.ndarray,
    path: list[Cell],
    source: Cell,
    goal: Cell,
    padded_size: int,
    grid_resolution: int = GRID_RES,
) -> list[dict[str, np.ndarray]]:
    """Walk quadtree top-down and extract one training example per active node.

    Each example contains an 8×8 downsampled obstacle density map of the
    current block, normalized source/goal positions relative to the block,
    the hierarchy level, and which of the 4 child quadrants are active.
    """
    examples: list[dict[str, np.ndarray]] = []

    def process(
        r0: int, r1: int, c0: int, c1: int,
        path_segment: list[Cell],
        level: int,
    ) -> None:
        h, w = r1 - r0, c1 - c0
        if h < 2 or w < 2:
            return

        block = padded[r0:r1, c0:c1]
        grid_8x8 = downsample_block(block, grid_resolution)

        positions = np.array([
            (source.row - r0) / h,
            (source.col - c0) / w,
            (goal.row - r0) / h,
            (goal.col - c0) / w,
        ], dtype=np.float32)

        mr = (r0 + r1) // 2
        mc = (c0 + c1) // 2
        quads = [(r0, mr, c0, mc), (r0, mr, mc, c1),
                 (mr, r1, c0, mc), (mr, r1, mc, c1)]

        activation = np.zeros(4, dtype=np.float32)
        quad_segments: list[list[Cell]] = [[] for _ in range(4)]

        for cell in path_segment:
            qi = _get_quad_idx(cell.row, cell.col, mr, mc)
            activation[qi] = 1.0
            quad_segments[qi].append(cell)

        examples.append({
            "grid_8x8": grid_8x8,
            "positions": positions,
            "level": np.array(level, dtype=np.int64),
            "activation": activation,
        })

        for qi in range(4):
            if not quad_segments[qi]:
                continue
            qr0, qr1, qc0, qc1 = quads[qi]
            process(qr0, qr1, qc0, qc1, quad_segments[qi], level + 1)

    process(0, padded_size, 0, padded_size, path, 0)
    return examples


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_flat_dataset(
    output_dir: str,
    num_examples: int = 50000,
    grid_sizes: Optional[list[int]] = None,
    densities: Optional[list[float]] = None,
    queries_per_grid: int = 20,
    min_path_distance: int = 10,
    grid_resolution: int = GRID_RES,
    seed: int = 42,
) -> None:
    """Generate flat per-node training examples using the BFS oracle."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if grid_sizes is None:
        grid_sizes = [32, 64, 128, 256]
    if densities is None:
        densities = [0.1, 0.2, 0.3]

    rng = np.random.RandomState(seed)
    generated = 0
    pbar = tqdm(total=num_examples, desc="Generating flat data", unit="ex")

    while generated < num_examples:
        gs = int(rng.choice(grid_sizes))
        dens = float(rng.choice(densities))
        grid_seed = int(rng.randint(0, 2**31))

        grid = generate_grid(gs, gs, dens, seed=grid_seed, ensure_connected=False)
        free_cells = list(zip(*np.where(grid.data == 0)))
        if len(free_cells) < 2:
            continue

        num_levels = max(int(math.ceil(math.log2(max(gs, 2)))), 1)
        padded_size = 1 << num_levels
        padded = np.ones((padded_size, padded_size), dtype=np.uint8)
        padded[:gs, :gs] = grid.data

        for _ in range(min(queries_per_grid, num_examples - generated)):
            si, gi = rng.choice(len(free_cells), size=2, replace=False)
            source = Cell(int(free_cells[si][0]), int(free_cells[si][1]))
            goal = Cell(int(free_cells[gi][0]), int(free_cells[gi][1]))

            if abs(source.row - goal.row) + abs(source.col - goal.col) < min_path_distance:
                continue

            result = bfs_shortest_path(grid, source, goal)
            if result is None:
                continue

            path, _ = result
            region_examples = extract_flat_labels(
                padded, path, source, goal, padded_size,
                grid_resolution=grid_resolution,
            )

            prev = generated
            for ex in region_examples:
                np.savez_compressed(out / f"{generated:07d}", **ex)
                generated += 1
                if generated >= num_examples:
                    break
            pbar.update(generated - prev)

            if generated >= num_examples:
                break

    pbar.close()
    print(f"Flat dataset complete: {generated} examples in {output_dir}")


def generate_recursive_dataset(
    output_dir: str,
    num_queries: int = 10000,
    grid_sizes: Optional[list[int]] = None,
    densities: Optional[list[float]] = None,
    min_path_distance: int = 10,
    seed: int = 42,
) -> None:
    """Generate full (grid, source, goal, path) queries for end-to-end validation."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if grid_sizes is None:
        grid_sizes = [32, 64, 128, 256]
    if densities is None:
        densities = [0.1, 0.2, 0.3]

    rng = np.random.RandomState(seed)
    generated = 0
    pbar = tqdm(total=num_queries, desc="Generating recursive data", unit="query")

    while generated < num_queries:
        gs = int(rng.choice(grid_sizes))
        dens = float(rng.choice(densities))
        grid_seed = int(rng.randint(0, 2**31))

        grid = generate_grid(gs, gs, dens, seed=grid_seed, ensure_connected=False)
        free_cells = list(zip(*np.where(grid.data == 0)))
        if len(free_cells) < 2:
            continue

        si, gi = rng.choice(len(free_cells), size=2, replace=False)
        source = Cell(int(free_cells[si][0]), int(free_cells[si][1]))
        goal = Cell(int(free_cells[gi][0]), int(free_cells[gi][1]))

        if abs(source.row - goal.row) + abs(source.col - goal.col) < min_path_distance:
            continue

        result = bfs_shortest_path(grid, source, goal)
        if result is None:
            continue

        path, _ = result
        path_arr = np.array([[c.row, c.col] for c in path], dtype=np.int32)

        np.savez_compressed(
            out / f"{generated:07d}",
            grid=grid.data,
            source=np.array([source.row, source.col]),
            goal=np.array([goal.row, goal.col]),
            path=path_arr,
        )
        generated += 1
        pbar.update(1)

    pbar.close()
    print(f"Recursive dataset complete: {generated} queries in {output_dir}")
