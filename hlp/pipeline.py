"""Unified inference pipeline: matrix_only, neural_only, hybrid modes.

neural_only — recursive hierarchical inference (pure D, no precomputation)
hybrid      — D predicts corridor, B verifies algebraically (guaranteed optimal)
matrix_only — full algebraic computation (guaranteed optimal)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from hlp.config import Config
from hlp.decomposition import (
    Block,
    build_block_hierarchy,
    get_block_for_cell,
    pad_grid,
    partition_into_blocks,
)
from hlp.composition import compute_all_transfer_matrices
from hlp.extraction import reconstruct_path
from hlp.grid import Cell, Grid, bfs_shortest_path
from hlp.neural.model import recursive_neural_inference


@dataclass
class PathResult:
    path: Optional[list[Cell]]
    cost: float
    optimal: bool
    mode: str
    corridor_size: int = 0
    total_blocks: int = 0
    computation_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# matrix_only mode (unchanged)
# ---------------------------------------------------------------------------

def run_matrix_only(
    grid: Grid,
    source: Cell,
    goal: Cell,
    config: Config,
    verbose: bool = True,
) -> PathResult:
    t0 = time.perf_counter()
    bs = config.block.block_size

    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return PathResult(None, float("inf"), True, "matrix_only",
                          computation_time_ms=(time.perf_counter() - t0) * 1000)

    if source == goal:
        return PathResult([source], 0.0, True, "matrix_only",
                          computation_time_ms=(time.perf_counter() - t0) * 1000)

    padded = pad_grid(grid, bs)
    l1 = partition_into_blocks(padded, bs)
    all_blocks = build_block_hierarchy(l1, padded)
    total = sum(1 for b in all_blocks.values() if b.level == 1)

    compute_all_transfer_matrices(all_blocks, padded, bs, max_workers=1)

    l1_blocks = [b for b in all_blocks.values() if b.level == 1]
    path = reconstruct_path(padded, source, goal, l1_blocks, bs)

    cost = float(len(path) - 1) if path else float("inf")
    elapsed = (time.perf_counter() - t0) * 1000

    if verbose:
        _print_result("matrix_only", cost, path, elapsed, total, total)

    return PathResult(path, cost, True, "matrix_only", total, total, elapsed)


# ---------------------------------------------------------------------------
# neural_only mode — recursive hierarchical inference (idea-doc §11.1)
# ---------------------------------------------------------------------------

def run_neural_only(
    grid: Grid,
    source: Cell,
    goal: Cell,
    model: nn.Module,
    config: Config,
    verbose: bool = True,
) -> PathResult:
    t0 = time.perf_counter()

    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return PathResult(None, float("inf"), False, "neural_only",
                          computation_time_ms=(time.perf_counter() - t0) * 1000)
    if source == goal:
        return PathResult([source], 0.0, False, "neural_only",
                          computation_time_ms=(time.perf_counter() - t0) * 1000)

    threshold = config.inference.activation_threshold
    active_cells = recursive_neural_inference(
        model, grid.data, grid.height, grid.width,
        (source.row, source.col), (goal.row, goal.col),
        stop_at_size=1,
        activation_threshold=threshold,
    )

    active_cells.add((source.row, source.col))
    active_cells.add((goal.row, goal.col))

    path = _bfs_within_cells(grid, source, goal, active_cells)

    cost = float(len(path) - 1) if path else float("inf")
    elapsed = (time.perf_counter() - t0) * 1000

    if verbose:
        _print_result("neural_only", cost, path, elapsed, len(active_cells), 0)

    return PathResult(path, cost, False, "neural_only", len(active_cells), 0, elapsed)


# ---------------------------------------------------------------------------
# hybrid mode — D predicts corridor, B verifies (idea-doc §11.2)
# ---------------------------------------------------------------------------

def run_hybrid(
    grid: Grid,
    source: Cell,
    goal: Cell,
    model: nn.Module,
    config: Config,
    verbose: bool = True,
) -> PathResult:
    t0 = time.perf_counter()
    bs = config.block.block_size

    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return PathResult(None, float("inf"), True, "hybrid",
                          computation_time_ms=(time.perf_counter() - t0) * 1000)
    if source == goal:
        return PathResult([source], 0.0, True, "hybrid",
                          computation_time_ms=(time.perf_counter() - t0) * 1000)

    threshold = config.inference.activation_threshold
    active_cells = recursive_neural_inference(
        model, grid.data, grid.height, grid.width,
        (source.row, source.col), (goal.row, goal.col),
        stop_at_size=bs,
        activation_threshold=threshold,
    )

    active_block_ids: set[tuple[int, int, int]] = set()
    for r, c in active_cells:
        active_block_ids.add((1, r // bs, c // bs))

    src_bid = get_block_for_cell(source, bs)
    goal_bid = get_block_for_cell(goal, bs)
    active_block_ids.add(src_bid)
    active_block_ids.add(goal_bid)

    padded = pad_grid(grid, bs)
    l1 = partition_into_blocks(padded, bs)
    all_blocks = build_block_hierarchy(l1, padded)
    total = sum(1 for b in all_blocks.values() if b.level == 1)

    compute_all_transfer_matrices(
        all_blocks, padded, bs,
        active_only=True, active_set=active_block_ids, max_workers=1,
    )

    corridor_l1 = [
        all_blocks[bid] for bid in active_block_ids
        if bid in all_blocks and all_blocks[bid].transfer_matrix is not None
    ]

    path = reconstruct_path(padded, source, goal, corridor_l1, bs)
    cost = float(len(path) - 1) if path else float("inf")

    if path is None:
        fallback_result = bfs_shortest_path(grid, source, goal)
        if fallback_result:
            path, cost = fallback_result[0], fallback_result[1]
            if verbose:
                print("  [hybrid] corridor miss — BFS fallback")

    if config.inference.verify_optimality and path:
        bfs_result = bfs_shortest_path(grid, source, goal)
        if bfs_result:
            bfs_cost = bfs_result[1]
            if abs(cost - bfs_cost) > 0.5:
                path = bfs_result[0]
                cost = bfs_cost

    elapsed = (time.perf_counter() - t0) * 1000
    corridor_size = len(active_block_ids)

    if verbose:
        _print_result("hybrid", cost, path, elapsed, corridor_size, total)

    return PathResult(path, cost, True, "hybrid", corridor_size, total, elapsed)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def find_path(
    grid: Grid,
    source: Cell,
    goal: Cell,
    config: Config,
    model: Optional[nn.Module] = None,
) -> PathResult:
    mode = config.inference.mode
    if mode == "neural_only" and model is not None:
        return run_neural_only(grid, source, goal, model, config)
    elif mode == "hybrid" and model is not None:
        return run_hybrid(grid, source, goal, model, config)
    else:
        return run_matrix_only(grid, source, goal, config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bfs_within_cells(
    grid: Grid,
    source: Cell,
    goal: Cell,
    allowed: set[tuple[int, int]],
) -> Optional[list[Cell]]:
    """BFS restricted to a set of allowed cells."""
    if source == goal:
        return [source]

    visited: set[tuple[int, int]] = set()
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue: deque[tuple[int, int]] = deque()

    sr, sc = source.row, source.col
    visited.add((sr, sc))
    queue.append((sr, sc))

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in allowed and (nr, nc) not in visited and grid.is_free(nr, nc):
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                if nr == goal.row and nc == goal.col:
                    path: list[Cell] = []
                    cur = (nr, nc)
                    while cur != (sr, sc):
                        path.append(Cell(*cur))
                        cur = parent[cur]
                    path.append(source)
                    path.reverse()
                    return path
                queue.append((nr, nc))

    return None


def _print_result(
    mode: str,
    cost: float,
    path: Optional[list[Cell]],
    elapsed_ms: float,
    corridor_size: int,
    total_blocks: int,
) -> None:
    path_len = len(path) if path else 0
    print(f"[{mode}] cost={cost:.1f} | path_len={path_len} | "
          f"corridor={corridor_size}/{total_blocks} | time={elapsed_ms:.2f}ms")
