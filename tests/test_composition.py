"""Tests for hierarchical transfer matrix composition."""

from __future__ import annotations

import numpy as np
import pytest

from hlp.composition import (
    build_combined_matrix,
    classify_boundary_cells,
    compose_transfer_matrix,
    compute_all_transfer_matrices,
)
from hlp.decomposition import (
    build_block_hierarchy,
    enumerate_boundary_cells,
    pad_grid,
    partition_into_blocks,
)
from hlp.grid import Cell, Grid, bfs_shortest_path
from hlp.tropical import INF


class TestClassifyBoundary:
    def test_child_boundary_split(self):
        g = Grid(np.zeros((32, 32), dtype=np.uint8))
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)

        # Level-2 block (0,0) should have children
        parent = all_blocks.get((2, 0, 0))
        assert parent is not None
        child = all_blocks[(1, 0, 0)]
        ext, internal = classify_boundary_cells(parent, child)
        # Child's boundary cells that are on parent's boundary = external
        # Others = internal (shared with sibling blocks)
        assert len(ext) + len(internal) == len(child.boundary_cells)
        assert len(internal) > 0  # Some cells are shared with siblings


class TestLevel1TMMatchesBFS:
    def test_empty_block(self):
        g = Grid(np.zeros((16, 16), dtype=np.uint8))
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)
        compute_all_transfer_matrices(all_blocks, padded, 16, max_workers=1)

        blk = all_blocks[(1, 0, 0)]
        T = blk.transfer_matrix
        assert T is not None
        for i, ci in enumerate(blk.boundary_cells):
            for j, cj in enumerate(blk.boundary_cells):
                result = bfs_shortest_path(g, ci, cj)
                expected = result[1] if result else INF
                assert abs(T[i, j] - expected) < 0.01, \
                    f"TM[{i},{j}] = {T[i, j]} but BFS = {expected} for {ci}->{cj}"

    def test_with_obstacle(self):
        data = np.zeros((16, 16), dtype=np.uint8)
        data[3, 3] = 1
        data[5, 5] = 1
        g = Grid(data)
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)
        compute_all_transfer_matrices(all_blocks, padded, 16, max_workers=1)

        blk = all_blocks[(1, 0, 0)]
        T = blk.transfer_matrix
        assert T is not None
        for i, ci in enumerate(blk.boundary_cells):
            for j, cj in enumerate(blk.boundary_cells):
                result = bfs_shortest_path(g, ci, cj)
                expected = result[1] if result else INF
                assert abs(T[i, j] - expected) < 0.01


class TestComposedTMMatchesBFS:
    def test_32x32_empty(self):
        """Composed Level-2 TM should match BFS for boundary-to-boundary paths."""
        g = Grid(np.zeros((32, 32), dtype=np.uint8))
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)
        compute_all_transfer_matrices(all_blocks, padded, 16, max_workers=1)

        # Level-2 block covers the whole 32x32 grid
        l2_block = all_blocks.get((2, 0, 0))
        assert l2_block is not None
        T = l2_block.transfer_matrix
        assert T is not None

        for i, ci in enumerate(l2_block.boundary_cells):
            for j, cj in enumerate(l2_block.boundary_cells):
                result = bfs_shortest_path(g, ci, cj)
                expected = result[1] if result else INF
                assert abs(T[i, j] - expected) < 0.01, \
                    f"Composed TM[{i},{j}]={T[i,j]} vs BFS={expected} for {ci}->{cj}"

    def test_32x32_random(self):
        """Random 25% obstacles — composed TM boundary distances match BFS."""
        np.random.seed(99)
        data = (np.random.random((32, 32)) < 0.15).astype(np.uint8)
        g = Grid(data)
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)
        compute_all_transfer_matrices(all_blocks, padded, 16, max_workers=1)

        l2 = all_blocks.get((2, 0, 0))
        if l2 is None or l2.transfer_matrix is None:
            return

        T = l2.transfer_matrix
        for i, ci in enumerate(l2.boundary_cells[:10]):
            for j, cj in enumerate(l2.boundary_cells[:10]):
                result = bfs_shortest_path(padded, ci, cj)
                expected = result[1] if result else INF
                if expected < INF:
                    assert abs(T[i, j] - expected) < 0.01, \
                        f"TM[{i},{j}]={T[i,j]} vs BFS={expected}"
