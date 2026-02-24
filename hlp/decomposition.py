"""Block decomposition: Level-1 partitioning, boundary enumeration, hierarchy tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from hlp.grid import Cell, Grid


@dataclass
class Block:
    block_id: tuple[int, int, int]           # (level, block_row, block_col)
    level: int
    row_start: int
    row_end: int                             # exclusive
    col_start: int
    col_end: int                             # exclusive
    boundary_cells: list[Cell] = field(default_factory=list)
    boundary_cell_to_index: dict[Cell, int] = field(default_factory=dict)
    transfer_matrix: Optional[np.ndarray] = None
    children: Optional[list[Block]] = None
    is_active: bool = True


# ---------------------------------------------------------------------------
# Boundary cell enumeration (clockwise)
# ---------------------------------------------------------------------------

def enumerate_boundary_cells(
    grid: Grid,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> list[Cell]:
    """
    Clockwise traversal: top (L→R), right (T→B), bottom (R→L), left (B→T).
    Corner cells belong to the first edge that encounters them; subsequent
    edges skip already-visited corners.
    """
    cells: list[Cell] = []
    if row_end <= row_start or col_end <= col_start:
        return cells

    h = row_end - row_start
    w = col_end - col_start

    if h == 1 and w == 1:
        if grid.is_free(row_start, col_start):
            cells.append(Cell(row_start, col_start))
        return cells

    if h == 1:
        for c in range(col_start, col_end):
            if grid.is_free(row_start, c):
                cells.append(Cell(row_start, c))
        return cells

    if w == 1:
        for r in range(row_start, row_end):
            if grid.is_free(r, col_start):
                cells.append(Cell(r, col_start))
        return cells

    # Top edge: left to right
    for c in range(col_start, col_end):
        if grid.is_free(row_start, c):
            cells.append(Cell(row_start, c))

    # Right edge: top+1 to bottom-1
    for r in range(row_start + 1, row_end - 1):
        if grid.is_free(r, col_end - 1):
            cells.append(Cell(r, col_end - 1))

    # Bottom edge: right to left
    for c in range(col_end - 1, col_start - 1, -1):
        if grid.is_free(row_end - 1, c):
            cells.append(Cell(row_end - 1, c))

    # Left edge: bottom-1 to top+1
    for r in range(row_end - 2, row_start, -1):
        if grid.is_free(r, col_start):
            cells.append(Cell(r, col_start))

    return cells


# ---------------------------------------------------------------------------
# Grid padding
# ---------------------------------------------------------------------------

def pad_grid(grid: Grid, block_size: int) -> Grid:
    h, w = grid.height, grid.width
    bh = math.ceil(h / block_size)
    bw = math.ceil(w / block_size)
    padded_bh = 1 << math.ceil(math.log2(max(bh, 1)))
    padded_bw = 1 << math.ceil(math.log2(max(bw, 1)))
    padded_h = padded_bh * block_size
    padded_w = padded_bw * block_size
    if padded_h == h and padded_w == w:
        return grid
    padded = np.ones((padded_h, padded_w), dtype=np.uint8)
    padded[:h, :w] = grid.data
    return Grid(padded)


# ---------------------------------------------------------------------------
# Level-1 block partitioning
# ---------------------------------------------------------------------------

def partition_into_blocks(grid: Grid, block_size: int) -> list[list[Block]]:
    num_br = math.ceil(grid.height / block_size)
    num_bc = math.ceil(grid.width / block_size)

    blocks: list[list[Block]] = []
    for br in range(num_br):
        row: list[Block] = []
        for bc in range(num_bc):
            rs = br * block_size
            re = min(rs + block_size, grid.height)
            cs = bc * block_size
            ce = min(cs + block_size, grid.width)
            bcells = enumerate_boundary_cells(grid, rs, re, cs, ce)
            blk = Block(
                block_id=(1, br, bc),
                level=1,
                row_start=rs,
                row_end=re,
                col_start=cs,
                col_end=ce,
                boundary_cells=bcells,
                boundary_cell_to_index={c: i for i, c in enumerate(bcells)},
            )
            row.append(blk)
        blocks.append(row)
    return blocks


# ---------------------------------------------------------------------------
# Hierarchical block tree
# ---------------------------------------------------------------------------

def build_block_hierarchy(
    level1_blocks: list[list[Block]],
    grid: Grid,
) -> dict[tuple[int, int, int], Block]:
    num_br = len(level1_blocks)
    num_bc = len(level1_blocks[0]) if num_br > 0 else 0

    all_blocks: dict[tuple[int, int, int], Block] = {}
    for br, row in enumerate(level1_blocks):
        for bc, blk in enumerate(row):
            all_blocks[(1, br, bc)] = blk

    if num_br == 0 or num_bc == 0:
        return all_blocks

    max_level = 1 + math.ceil(math.log2(max(num_br, num_bc)))

    prev_rows = num_br
    prev_cols = num_bc

    for level in range(2, max_level + 1):
        curr_rows = math.ceil(prev_rows / 2)
        curr_cols = math.ceil(prev_cols / 2)

        for br in range(curr_rows):
            for bc in range(curr_cols):
                children_ids = [
                    (level - 1, 2 * br, 2 * bc),
                    (level - 1, 2 * br, 2 * bc + 1),
                    (level - 1, 2 * br + 1, 2 * bc),
                    (level - 1, 2 * br + 1, 2 * bc + 1),
                ]
                children = [all_blocks[cid] for cid in children_ids if cid in all_blocks]
                if not children:
                    continue

                rs = min(c.row_start for c in children)
                re = max(c.row_end for c in children)
                cs = min(c.col_start for c in children)
                ce = max(c.col_end for c in children)

                bcells = enumerate_boundary_cells(grid, rs, re, cs, ce)
                blk = Block(
                    block_id=(level, br, bc),
                    level=level,
                    row_start=rs,
                    row_end=re,
                    col_start=cs,
                    col_end=ce,
                    boundary_cells=bcells,
                    boundary_cell_to_index={c: i for i, c in enumerate(bcells)},
                    children=children,
                )
                all_blocks[(level, br, bc)] = blk

        prev_rows = curr_rows
        prev_cols = curr_cols

    return all_blocks


def get_block_for_cell(
    cell: Cell,
    block_size: int,
) -> tuple[int, int, int]:
    return (1, cell.row // block_size, cell.col // block_size)
