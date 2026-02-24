#!/usr/bin/env python3
"""Benchmark all pathfinding methods across grid sizes."""

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

GRID_SIZES = [32, 64, 128, 256, 512]
NUM_TRIALS = 10


def _run_one_size(
    size: int,
    methods: list[str],
    args: argparse.Namespace,
    config: Config,
    model: Optional[QuadTreeConvNet],
) -> list[dict]:
    """Run all methods × trials for one grid size, with per-method progress bars."""
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
            grid = generate_grid(size, size, args.density, seed=trial * 1000 + size)
            free = list(zip(*np.where(grid.data == 0)))
            if len(free) < 2:
                continue

            rng = np.random.RandomState(trial)
            si, gi = rng.choice(len(free), size=2, replace=False)
            source = Cell(int(free[si][0]), int(free[si][1]))
            goal = Cell(int(free[gi][0]), int(free[gi][1]))

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
        corr_str = ""
        if corridor_ratios:
            corr_str = f"{np.mean(corridor_ratios):>9.1%}"
        else:
            corr_str = f"{'—':>10}"

        size_results.append({
            "grid_size": size,
            "method": method_name,
            "avg_time_ms": round(avg_time, 2),
            "avg_cost": round(avg_cost, 1),
            "optimality_rate": round(opt_rate, 4),
            "valid_trials": total_valid,
            "_display": (
                f"{size:>6} | {method_name:>14} | {avg_time:>10.2f} | "
                f"{avg_cost:>10.1f} | {opt_rate:>7.1%} | {corr_str}"
            ),
        })

    return size_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pathfinding methods")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=GRID_SIZES)
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
    parser.add_argument("--output", type=str, default="results/benchmark.csv")
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--all", action="store_true",
                        help="Include slow methods: BFS, Dijkstra, Matrix Only")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = Config.from_yaml(config_path) if config_path.exists() else Config()

    model = _load_model(config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    methods = ["A*"]
    if model is not None:
        methods.extend(["Neural Only", "Hybrid"])
    if args.all:
        methods = ["BFS", "A*", "Dijkstra", "Matrix Only"]
        if model is not None:
            methods.extend(["Neural Only", "Hybrid"])

    header = (f"{'Size':>6} | {'Method':>14} | {'Avg ms':>10} | "
              f"{'Avg Cost':>10} | {'Optimal':>8} | {'Corridor':>10}")

    for size in args.grid_sizes:
        size_results = _run_one_size(size, methods, args, config, model)

        print(header)
        print("-" * len(header))
        for r in size_results:
            print(r["_display"])
        print()

        all_results.extend(size_results)

    csv_fields = ["grid_size", "method", "avg_time_ms", "avg_cost",
                  "optimality_rate", "valid_trials"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Results saved to {args.output}")


def _load_model(config: Config) -> Optional[QuadTreeConvNet]:
    ckpt = Path(config.neural.checkpoint_path)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt} — skipping neural methods")
        return None
    try:
        nc = config.neural
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = QuadTreeConvNet(
            d=nc.d,
            max_levels=nc.max_levels,
            grid_resolution=nc.grid_resolution,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


if __name__ == "__main__":
    main()
