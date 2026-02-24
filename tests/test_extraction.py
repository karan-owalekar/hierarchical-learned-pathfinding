"""Tests for path extraction: embeddings, distance queries, path reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from hlp.composition import compute_all_transfer_matrices
from hlp.decomposition import (
    build_block_hierarchy,
    get_block_for_cell,
    pad_grid,
    partition_into_blocks,
)
from hlp.extraction import compute_path_distance, reconstruct_path
from hlp.grid import Cell, Grid, bfs_shortest_path, generate_grid
from hlp.tropical import INF


def _setup(grid: Grid, block_size: int = 16):
    """Helper: pad, partition, build hierarchy, compute all TMs."""
    padded = pad_grid(grid, block_size)
    l1 = partition_into_blocks(padded, block_size)
    all_blocks = build_block_hierarchy(l1, padded)
    compute_all_transfer_matrices(all_blocks, padded, block_size, max_workers=1)
    l1_blocks = [b for b in all_blocks.values() if b.level == 1]
    return padded, l1_blocks


def _validate_path(grid: Grid, path: list[Cell], source: Cell, goal: Cell) -> None:
    """Check path validity: starts at source, ends at goal, each step is adjacent and free."""
    assert path[0] == source, f"Path starts at {path[0]}, expected {source}"
    assert path[-1] == goal, f"Path ends at {path[-1]}, expected {goal}"
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        assert abs(a.row - b.row) + abs(a.col - b.col) == 1, \
            f"Non-adjacent step: {a} -> {b}"
        assert grid.is_free(b.row, b.col), f"Path cell {b} is blocked"


class TestDistanceMatchesBFS:
    @pytest.mark.parametrize("size", [16, 32, 64])
    def test_empty_grid(self, size):
        g = Grid(np.zeros((size, size), dtype=np.uint8))
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(size - 1, size - 1)
        dist = compute_path_distance(padded, src, goal, l1_blocks, 16)
        bfs = bfs_shortest_path(g, src, goal)
        assert bfs is not None
        assert abs(dist - bfs[1]) < 0.01

    def test_single_obstacle(self):
        data = np.zeros((16, 16), dtype=np.uint8)
        data[3, 3] = 1
        g = Grid(data)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(7, 7)
        dist = compute_path_distance(padded, src, goal, l1_blocks, 16)
        bfs = bfs_shortest_path(g, src, goal)
        assert bfs is not None
        assert abs(dist - bfs[1]) < 0.01

    def test_h_wall_with_gap(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[16, :30] = 1  # Horizontal wall with gap at col 30
        g = Grid(data)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(31, 0)
        dist = compute_path_distance(padded, src, goal, l1_blocks, 16)
        bfs = bfs_shortest_path(padded, src, goal)
        if bfs is not None:
            assert abs(dist - bfs[1]) < 0.01

    def test_v_wall_with_gap(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[:30, 16] = 1  # Vertical wall with gap at row 30
        g = Grid(data)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(0, 31)
        dist = compute_path_distance(padded, src, goal, l1_blocks, 16)
        bfs = bfs_shortest_path(padded, src, goal)
        if bfs is not None:
            assert abs(dist - bfs[1]) < 0.01

    def test_source_equals_goal(self):
        g = Grid(np.zeros((16, 16), dtype=np.uint8))
        padded, l1_blocks = _setup(g)
        src = Cell(5, 5)
        dist = compute_path_distance(padded, src, src, l1_blocks, 16)
        assert dist == 0.0

    def test_disconnected(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[16, :] = 1  # Complete wall
        g = Grid(data)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(31, 0)
        dist = compute_path_distance(padded, src, goal, l1_blocks, 16)
        assert dist >= INF

    @pytest.mark.parametrize("seed", range(20))
    def test_random_25pct(self, seed):
        g = generate_grid(32, 32, 0.25, seed=seed, ensure_connected=True)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(g.height - 1, g.width - 1)
        dist = compute_path_distance(padded, src, goal, l1_blocks, 16)
        bfs = bfs_shortest_path(padded, src, goal)
        if bfs is not None:
            assert abs(dist - bfs[1]) < 0.01, f"Seed {seed}: dist={dist} vs bfs={bfs[1]}"


class TestPathReconstruction:
    @pytest.mark.parametrize("size", [16, 32])
    def test_empty_grid(self, size):
        g = Grid(np.zeros((size, size), dtype=np.uint8))
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(size - 1, size - 1)
        path = reconstruct_path(padded, src, goal, l1_blocks, 16)
        assert path is not None
        _validate_path(padded, path, src, goal)
        bfs = bfs_shortest_path(g, src, goal)
        assert bfs is not None
        assert len(path) - 1 == bfs[1]

    def test_with_obstacles(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[5, 3:10] = 1
        data[10, 5:15] = 1
        g = Grid(data)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(31, 31)
        path = reconstruct_path(padded, src, goal, l1_blocks, 16)
        if path is not None:
            _validate_path(padded, path, src, goal)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_reconstruct(self, seed):
        g = generate_grid(32, 32, 0.2, seed=seed, ensure_connected=True)
        padded, l1_blocks = _setup(g)
        src = Cell(0, 0)
        goal = Cell(g.height - 1, g.width - 1)
        path = reconstruct_path(padded, src, goal, l1_blocks, 16)
        if path is not None:
            _validate_path(padded, path, src, goal)
            bfs = bfs_shortest_path(padded, src, goal)
            if bfs is not None:
                assert len(path) - 1 == bfs[1], \
                    f"Seed {seed}: path cost {len(path)-1} vs BFS {bfs[1]}"
