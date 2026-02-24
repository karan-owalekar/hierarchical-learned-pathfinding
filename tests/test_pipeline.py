"""End-to-end tests for the inference pipeline: matrix_only matches BFS."""

from __future__ import annotations

import numpy as np
import pytest

from hlp.config import Config
from hlp.grid import Cell, Grid, bfs_shortest_path, generate_grid
from hlp.pipeline import run_matrix_only


class TestMatrixOnlyMatchesBFS:
    @pytest.mark.parametrize("size", [16, 32, 64])
    def test_empty_grid(self, size):
        g = Grid(np.zeros((size, size), dtype=np.uint8))
        cfg = Config()
        cfg.block.block_size = 16

        src = Cell(0, 0)
        goal = Cell(size - 1, size - 1)
        result = run_matrix_only(g, src, goal, cfg)
        bfs = bfs_shortest_path(g, src, goal)

        assert bfs is not None
        assert result.path is not None
        assert abs(result.cost - bfs[1]) < 0.01

    def test_single_obstacle(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[8, 8] = 1
        g = Grid(data)
        cfg = Config()
        cfg.block.block_size = 16

        src = Cell(0, 0)
        goal = Cell(31, 31)
        result = run_matrix_only(g, src, goal, cfg)
        bfs = bfs_shortest_path(g, src, goal)

        assert bfs is not None
        assert result.path is not None
        assert abs(result.cost - bfs[1]) < 0.01

    def test_h_wall_with_gap(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[16, :30] = 1
        g = Grid(data)
        cfg = Config()
        cfg.block.block_size = 16

        src = Cell(0, 0)
        goal = Cell(31, 0)
        result = run_matrix_only(g, src, goal, cfg)
        bfs = bfs_shortest_path(g, src, goal)

        if bfs is not None:
            assert result.path is not None
            assert abs(result.cost - bfs[1]) < 0.01

    def test_no_path(self):
        data = np.zeros((32, 32), dtype=np.uint8)
        data[16, :] = 1  # Complete wall
        g = Grid(data)
        cfg = Config()
        cfg.block.block_size = 16

        result = run_matrix_only(g, Cell(0, 0), Cell(31, 0), cfg)
        assert result.path is None or result.cost >= float("inf")

    def test_source_equals_goal(self):
        g = Grid(np.zeros((32, 32), dtype=np.uint8))
        cfg = Config()
        cfg.block.block_size = 16

        result = run_matrix_only(g, Cell(5, 5), Cell(5, 5), cfg)
        assert result.cost == 0.0
        assert result.path == [Cell(5, 5)]

    def test_obstacle_at_source(self):
        data = np.zeros((16, 16), dtype=np.uint8)
        data[0, 0] = 1
        g = Grid(data)
        cfg = Config()
        cfg.block.block_size = 16

        result = run_matrix_only(g, Cell(0, 0), Cell(15, 15), cfg)
        assert result.path is None

    @pytest.mark.parametrize("seed", range(10))
    def test_random_25pct(self, seed):
        g = generate_grid(32, 32, 0.25, seed=seed, ensure_connected=True)
        cfg = Config()
        cfg.block.block_size = 16

        src = Cell(0, 0)
        goal = Cell(g.height - 1, g.width - 1)
        result = run_matrix_only(g, src, goal, cfg)
        bfs = bfs_shortest_path(g, src, goal)

        if bfs is not None:
            assert result.path is not None, f"Seed {seed}: pipeline found no path but BFS did"
            assert abs(result.cost - bfs[1]) < 0.01, \
                f"Seed {seed}: pipeline cost {result.cost} vs BFS {bfs[1]}"

    def test_narrow_corridor(self):
        """4-row tall, 32-column wide grid — tests minimal-height blocks."""
        data = np.zeros((4, 32), dtype=np.uint8)
        g = Grid(data)
        cfg = Config()
        cfg.block.block_size = 4

        src = Cell(0, 0)
        goal = Cell(3, 31)
        result = run_matrix_only(g, src, goal, cfg)
        bfs = bfs_shortest_path(g, src, goal)

        assert bfs is not None
        assert result.path is not None
        assert abs(result.cost - bfs[1]) < 0.01

    def test_optimal_flag(self):
        g = Grid(np.zeros((16, 16), dtype=np.uint8))
        cfg = Config()
        cfg.block.block_size = 16
        result = run_matrix_only(g, Cell(0, 0), Cell(15, 15), cfg)
        assert result.optimal is True
        assert result.mode == "matrix_only"
