"""Tests for Grid class, BFS, grid generation, and map generators."""

from __future__ import annotations

import numpy as np
import pytest

from hlp.grid import (
    Cell,
    Grid,
    bfs_all_distances_within_block,
    bfs_path_within_block,
    bfs_shortest_path,
    bfs_within_block,
    generate_grid,
)
from ui.map_generators import (
    generate_dfs_maze,
    generate_random,
    generate_recursive_division,
    generate_rooms,
    generate_spiral,
)


class TestGrid:
    def test_is_free(self):
        data = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        g = Grid(data)
        assert g.is_free(0, 0)
        assert not g.is_free(0, 1)
        assert g.is_free(1, 1)

    def test_in_bounds(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        assert g.in_bounds(0, 0)
        assert g.in_bounds(3, 3)
        assert not g.in_bounds(-1, 0)
        assert not g.in_bounds(4, 0)

    def test_neighbors_center(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        nbrs = g.neighbors(2, 2)
        assert len(nbrs) == 4
        assert set(nbrs) == {(1, 2), (3, 2), (2, 1), (2, 3)}

    def test_neighbors_corner(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        nbrs = g.neighbors(0, 0)
        assert set(nbrs) == {(1, 0), (0, 1)}

    def test_neighbors_with_obstacles(self):
        data = np.zeros((3, 3), dtype=np.uint8)
        data[0, 1] = 1
        data[1, 0] = 1
        g = Grid(data)
        nbrs = g.neighbors(0, 0)
        assert nbrs == []


class TestBFS:
    def test_empty_grid_manhattan(self):
        for size in [4, 8, 16, 32]:
            g = Grid(np.zeros((size, size), dtype=np.uint8))
            src = Cell(0, 0)
            goal = Cell(size - 1, size - 1)
            result = bfs_shortest_path(g, src, goal)
            assert result is not None
            path, cost = result
            assert cost == 2 * (size - 1)
            assert path[0] == src
            assert path[-1] == goal
            assert len(path) == int(cost) + 1

    def test_single_obstacle_detour(self):
        data = np.zeros((8, 8), dtype=np.uint8)
        data[3, 3] = 1
        g = Grid(data)
        result = bfs_shortest_path(g, Cell(0, 0), Cell(6, 6))
        assert result is not None
        _, cost = result
        assert cost == 12  # Manhattan = 12, obstacle at (3,3) is on diagonal

    def test_no_path(self):
        data = np.zeros((8, 8), dtype=np.uint8)
        data[4, :] = 1  # Complete wall
        g = Grid(data)
        result = bfs_shortest_path(g, Cell(0, 0), Cell(7, 7))
        assert result is None

    def test_source_equals_goal(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        result = bfs_shortest_path(g, Cell(2, 2), Cell(2, 2))
        assert result is not None
        path, cost = result
        assert cost == 0.0
        assert path == [Cell(2, 2)]

    def test_obstacle_at_source(self):
        data = np.zeros((4, 4), dtype=np.uint8)
        data[0, 0] = 1
        g = Grid(data)
        result = bfs_shortest_path(g, Cell(0, 0), Cell(3, 3))
        assert result is None

    def test_adjacent_source_goal(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        result = bfs_shortest_path(g, Cell(1, 1), Cell(1, 2))
        assert result is not None
        path, cost = result
        assert cost == 1.0
        assert len(path) == 2


class TestBFSWithinBlock:
    def test_within_block_matches_full(self):
        g = Grid(np.zeros((16, 16), dtype=np.uint8))
        src = Cell(2, 2)
        goal = Cell(5, 5)
        full = bfs_shortest_path(g, src, goal)
        block = bfs_within_block(g, src, goal, 0, 16, 0, 16)
        assert full is not None
        assert block is not None
        assert block == full[1]

    def test_restricted_bounds(self):
        g = Grid(np.zeros((16, 16), dtype=np.uint8))
        src = Cell(2, 2)
        goal = Cell(2, 14)
        # Within full grid, direct path
        full = bfs_shortest_path(g, src, goal)
        assert full is not None
        # Within a restricted block, goal is outside
        block = bfs_within_block(g, src, goal, 0, 8, 0, 8)
        assert block is None

    def test_all_distances(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        src = Cell(0, 0)
        dists = bfs_all_distances_within_block(g, src, 0, 4, 0, 4)
        assert dists[Cell(0, 0)] == 0.0
        assert dists[Cell(3, 3)] == 6.0
        assert dists[Cell(0, 3)] == 3.0

    def test_path_within_block(self):
        g = Grid(np.zeros((8, 8), dtype=np.uint8))
        path = bfs_path_within_block(g, Cell(1, 1), Cell(3, 3), 0, 8, 0, 8)
        assert path is not None
        assert path[0] == Cell(1, 1)
        assert path[-1] == Cell(3, 3)
        assert len(path) == 5


class TestGridGeneration:
    def test_generate_grid_basic(self):
        g = generate_grid(32, 32, 0.2, seed=42)
        assert g.height == 32
        assert g.width == 32
        assert g.is_free(0, 0)
        assert g.is_free(31, 31)

    def test_zero_density(self):
        g = generate_grid(16, 16, 0.0, seed=1)
        assert g.data.sum() == 0

    def test_connected(self):
        g = generate_grid(32, 32, 0.25, seed=42, ensure_connected=True)
        result = bfs_shortest_path(g, Cell(0, 0), Cell(31, 31))
        assert result is not None


class TestMapGenerators:
    @pytest.mark.parametrize("gen_fn", [
        generate_random, generate_dfs_maze, generate_spiral,
        generate_recursive_division, generate_rooms,
    ])
    def test_valid_output(self, gen_fn):
        grid, start, goal = gen_fn(32, 32, seed=42)
        assert isinstance(grid, Grid)
        assert grid.height == 32
        assert grid.width == 32
        assert grid.is_free(start.row, start.col)
        assert grid.is_free(goal.row, goal.col)
        assert start != goal

    @pytest.mark.parametrize("gen_fn", [
        generate_random, generate_dfs_maze, generate_spiral,
        generate_recursive_division, generate_rooms,
    ])
    def test_path_exists(self, gen_fn):
        grid, start, goal = gen_fn(32, 32, seed=42)
        result = bfs_shortest_path(grid, start, goal)
        assert result is not None, f"No path in {gen_fn.__name__} grid"
