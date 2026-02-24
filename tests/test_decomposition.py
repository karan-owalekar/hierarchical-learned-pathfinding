"""Tests for block decomposition, boundary enumeration, and hierarchy tree."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hlp.decomposition import (
    Block,
    build_block_hierarchy,
    enumerate_boundary_cells,
    get_block_for_cell,
    pad_grid,
    partition_into_blocks,
)
from hlp.grid import Cell, Grid


class TestBoundaryEnumeration:
    def test_4x4_empty(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        cells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        # 4x4 block: top=4, right=2, bottom=4, left=2 = 12
        assert len(cells) == 12

    def test_clockwise_order(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        cells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        # Top: (0,0), (0,1), (0,2), (0,3)
        assert cells[0] == Cell(0, 0)
        assert cells[1] == Cell(0, 1)
        assert cells[2] == Cell(0, 2)
        assert cells[3] == Cell(0, 3)
        # Right: (1,3), (2,3)
        assert cells[4] == Cell(1, 3)
        assert cells[5] == Cell(2, 3)
        # Bottom reversed: (3,3), (3,2), (3,1), (3,0)
        assert cells[6] == Cell(3, 3)
        assert cells[7] == Cell(3, 2)
        assert cells[8] == Cell(3, 1)
        assert cells[9] == Cell(3, 0)
        # Left reversed: (2,0), (1,0)
        assert cells[10] == Cell(2, 0)
        assert cells[11] == Cell(1, 0)

    def test_all_on_boundary(self):
        g = Grid(np.zeros((8, 8), dtype=np.uint8))
        cells = enumerate_boundary_cells(g, 0, 8, 0, 8)
        for c in cells:
            assert c.row == 0 or c.row == 7 or c.col == 0 or c.col == 7

    def test_skips_blocked(self):
        data = np.zeros((4, 4), dtype=np.uint8)
        data[0, 1] = 1  # Block one boundary cell
        g = Grid(data)
        cells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        assert Cell(0, 1) not in cells
        assert len(cells) == 11

    def test_no_duplicates(self):
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        cells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        assert len(cells) == len(set(cells))

    def test_1x1(self):
        g = Grid(np.zeros((1, 1), dtype=np.uint8))
        cells = enumerate_boundary_cells(g, 0, 1, 0, 1)
        assert cells == [Cell(0, 0)]

    def test_1xn(self):
        g = Grid(np.zeros((1, 4), dtype=np.uint8))
        cells = enumerate_boundary_cells(g, 0, 1, 0, 4)
        assert len(cells) == 4


class TestPadGrid:
    def test_already_padded(self):
        g = Grid(np.zeros((16, 16), dtype=np.uint8))
        p = pad_grid(g, 16)
        assert p.height == 16
        assert p.width == 16

    def test_needs_padding(self):
        g = Grid(np.zeros((20, 20), dtype=np.uint8))
        p = pad_grid(g, 16)
        assert p.height >= 20
        assert p.width >= 20
        assert p.height % 16 == 0
        assert p.width % 16 == 0
        # Padded cells should be blocked
        assert p.data[20, 0] == 1


class TestPartition:
    def test_covers_full_grid(self):
        g = Grid(np.zeros((32, 32), dtype=np.uint8))
        blocks = partition_into_blocks(g, 16)
        assert len(blocks) == 2
        assert len(blocks[0]) == 2

        covered = set()
        for row in blocks:
            for blk in row:
                for r in range(blk.row_start, blk.row_end):
                    for c in range(blk.col_start, blk.col_end):
                        covered.add((r, c))
        assert len(covered) == 32 * 32

    def test_no_overlap(self):
        g = Grid(np.zeros((32, 32), dtype=np.uint8))
        blocks = partition_into_blocks(g, 16)
        all_cells: list[tuple[int, int]] = []
        for row in blocks:
            for blk in row:
                for r in range(blk.row_start, blk.row_end):
                    for c in range(blk.col_start, blk.col_end):
                        all_cells.append((r, c))
        assert len(all_cells) == len(set(all_cells))


class TestHierarchy:
    def test_level_count(self):
        g = Grid(np.zeros((64, 64), dtype=np.uint8))
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)
        max_level = max(bid[0] for bid in all_blocks)
        num_br = len(l1)
        num_bc = len(l1[0])
        expected = 1 + math.ceil(math.log2(max(num_br, num_bc)))
        assert max_level == expected

    def test_children_exist(self):
        g = Grid(np.zeros((32, 32), dtype=np.uint8))
        padded = pad_grid(g, 16)
        l1 = partition_into_blocks(padded, 16)
        all_blocks = build_block_hierarchy(l1, padded)
        for bid, blk in all_blocks.items():
            if blk.level > 1:
                assert blk.children is not None
                assert len(blk.children) > 0

    def test_get_block_for_cell(self):
        assert get_block_for_cell(Cell(0, 0), 16) == (1, 0, 0)
        assert get_block_for_cell(Cell(15, 15), 16) == (1, 0, 0)
        assert get_block_for_cell(Cell(16, 0), 16) == (1, 1, 0)
        assert get_block_for_cell(Cell(0, 16), 16) == (1, 0, 1)
