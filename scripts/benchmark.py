#!/usr/bin/env python3
"""Benchmark all pathfinding methods across grid sizes and map types."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from hlp.config import Config
from hlp.grid import Cell, Grid, bfs_shortest_path, generate_grid
from hlp.neural.model import QuadTreeConvNet
from hlp.pipeline import run_hybrid, run_matrix_only, run_neural_only
from baselines.astar import astar
from baselines.dijkstra import dijkstra
from ui.map_generators import MAP_TYPE_GENERATORS, MAP_TYPE_KEYS, MAP_TYPE_NAMES

GRID_SIZES = [32, 64, 128, 256, 512]
NUM_TRIALS = 10


def _generate_grid(
    size: int, trial: int, density: float, map_type: Optional[str] = None,
) -> tuple[Grid, Cell, Cell]:
    """Generate a grid + start/goal for one benchmark trial."""
    seed = trial * 1000 + size

    if map_type and map_type != "random_scatter":
        gen_fn = MAP_TYPE_GENERATORS.get(map_type)
        if gen_fn is not None:
            try:
                return gen_fn(size, size, density=density, seed=seed)
            except TypeError:
                return gen_fn(size, size, seed=seed)

    grid = generate_grid(size, size, density, seed=seed)
    free = list(zip(*np.where(grid.data == 0)))
    if len(free) < 2:
        return grid, Cell(0, 0), Cell(size - 1, size - 1)

    rng = np.random.RandomState(trial)
    si, gi = rng.choice(len(free), size=2, replace=False)
    source = Cell(int(free[si][0]), int(free[si][1]))
    goal = Cell(int(free[gi][0]), int(free[gi][1]))
    return grid, source, goal


def _run_one_size(
    size: int,
    methods: list[str],
    args: argparse.Namespace,
    config: Config,
    model: Optional[QuadTreeConvNet],
    map_type: Optional[str] = None,
) -> list[dict]:
    """Run all methods x trials for one grid size, with per-method progress bars."""
    size_results: list[dict] = []

    for method_name in methods:
        times: list[float] = []
        costs: list[float] = []
        corridor_ratios: list[float] = []
        optimal_count = 0
        total_valid = 0

        pbar = tqdm(
            range(args.trials),
            desc=f"  {method_name:>14}",
            leave=False,
            unit="trial",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for trial in pbar:
            grid, source, goal = _generate_grid(size, trial, args.density, map_type)
            free = list(zip(*np.where(grid.data == 0)))
            if len(free) < 2:
                continue

            bfs_result = bfs_shortest_path(grid, source, goal)
            if bfs_result is None:
                continue
            bfs_cost = bfs_result[1]
            total_valid += 1

            t0 = time.perf_counter()
            result_cost: float = float("inf")
            corridor_ratio = -1.0
            try:
                if method_name == "BFS":
                    res_bfs = bfs_shortest_path(grid, source, goal)
                    result_cost = res_bfs[1] if res_bfs else float("inf")
                elif method_name == "A*":
                    res = astar(grid, source, goal)
                    result_cost = res.cost
                elif method_name == "Dijkstra":
                    res = dijkstra(grid, source, goal)
                    result_cost = res.cost
                elif method_name == "Matrix Only":
                    res = run_matrix_only(grid, source, goal, config, verbose=False)
                    result_cost = res.cost
                elif method_name == "Neural Only" and model is not None:
                    res = run_neural_only(grid, source, goal, model, config, verbose=False)
                    result_cost = res.cost
                    corridor_ratio = res.corridor_size / max(size * size, 1)
                elif method_name == "Hybrid" and model is not None:
                    res = run_hybrid(grid, source, goal, model, config, verbose=False)
                    result_cost = res.cost
                    if res.total_blocks > 0:
                        corridor_ratio = res.corridor_size / res.total_blocks
            except Exception:
                continue
            elapsed = (time.perf_counter() - t0) * 1000

            times.append(elapsed)
            costs.append(result_cost)
            if corridor_ratio >= 0:
                corridor_ratios.append(corridor_ratio)
            if abs(result_cost - bfs_cost) < 0.5:
                optimal_count += 1

        pbar.close()

        if total_valid == 0:
            continue

        avg_time = np.mean(times) if times else 0.0
        avg_cost = np.mean(costs) if costs else 0.0
        opt_rate = optimal_count / total_valid
        if corridor_ratios:
            corr_str = f"{np.mean(corridor_ratios):>9.1%}"
        else:
            corr_str = f"{'---':>10}"

        size_results.append({
            "grid_size": size,
            "method": method_name,
            "avg_time_ms": round(avg_time, 2),
            "avg_cost": round(avg_cost, 1),
            "optimality_rate": round(opt_rate, 4),
            "valid_trials": total_valid,
            "map_type": map_type or "random_scatter",
            "_display": (
                f"{size:>6} | {method_name:>14} | {avg_time:>10.2f} | "
                f"{avg_cost:>10.1f} | {opt_rate:>7.1%} | {corr_str}"
            ),
        })

    return size_results


def _load_models(config: Config) -> dict[str, QuadTreeConvNet]:
    """Load specialist models + fallback default."""
    nc = config.neural
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(nc.checkpoint_path).parent
    models: dict[str, QuadTreeConvNet] = {}

    def _try_load(path: Path) -> Optional[QuadTreeConvNet]:
        if not path.exists():
            return None
        try:
            m = QuadTreeConvNet(
                d=nc.d, max_levels=nc.max_levels,
                grid_resolution=nc.grid_resolution,
            ).to(device)
            m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            m.eval()
            return m
        except Exception:
            return None

    for key in MAP_TYPE_KEYS:
        m = _try_load(ckpt_dir / f"best_{key}.pt")
        if m is not None:
            models[key] = m

    fallback = _try_load(Path(nc.checkpoint_path))
    if fallback is not None:
        models["_default"] = fallback

    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pathfinding methods")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=GRID_SIZES)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--output", type=str, default="results/benchmark.csv")
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--all", action="store_true",
                        help="Include slow methods: BFS, Dijkstra, Matrix Only")
    parser.add_argument(
        "--map-type", type=str, choices=MAP_TYPE_KEYS,
        help="Benchmark only this map type (default: all with available models)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = Config.from_yaml(config_path) if config_path.exists() else Config()

    models = _load_models(config)
    if not models:
        print("No model checkpoints found -- only classical methods will be benchmarked")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    base_methods = ["A*"]
    if args.all:
        base_methods = ["BFS", "A*", "Dijkstra", "Matrix Only"]

    map_types = [args.map_type] if args.map_type else MAP_TYPE_KEYS

    header = (f"{'Size':>6} | {'Method':>14} | {'Avg ms':>10} | "
              f"{'Avg Cost':>10} | {'Optimal':>8} | {'Corridor':>10}")

    for mt in map_types:
        mt_name = MAP_TYPE_NAMES[MAP_TYPE_KEYS.index(mt)]
        model = models.get(mt) or models.get("_default")

        methods = list(base_methods)
        if model is not None:
            methods.extend(["Neural Only", "Hybrid"])

        print(f"\n{'=' * 60}")
        print(f"  Map type: {mt_name}")
        if model is None:
            print("  (no specialist or fallback model)")
        print(f"{'=' * 60}")

        for size in args.grid_sizes:
            size_results = _run_one_size(size, methods, args, config, model, mt)

            print(header)
            print("-" * len(header))
            for r in size_results:
                print(r["_display"])
            print()

            all_results.extend(size_results)

    csv_fields = ["grid_size", "method", "avg_time_ms", "avg_cost",
                  "optimality_rate", "valid_trials", "map_type"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
