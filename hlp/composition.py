"""Hierarchical transfer matrix composition.

Composes child-block transfer matrices into parent transfer matrices using
a combined distance matrix + tropical Floyd-Warshall to eliminate internal
boundary cells.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

from hlp.decomposition import Block
from hlp.grid import Cell, Grid
from hlp.tropical import (
    INF,
    compute_level1_transfer_matrix,
    floyd_warshall,
)


# ---------------------------------------------------------------------------
# Boundary classification
# ---------------------------------------------------------------------------

def classify_boundary_cells(
    parent: Block,
    child: Block,
) -> tuple[list[int], list[int]]:
    """Return (external_indices, internal_indices) for a child's boundary cells."""
    parent_set = {(c.row, c.col) for c in parent.boundary_cells}
    external: list[int] = []
    internal: list[int] = []
    for i, cell in enumerate(child.boundary_cells):
        if (cell.row, cell.col) in parent_set:
            external.append(i)
        else:
            internal.append(i)
    return external, internal


# ---------------------------------------------------------------------------
# Combined distance matrix over all children
# ---------------------------------------------------------------------------

def build_combined_matrix(
    children: list[Block],
    grid: Grid,
    parent: Block,
) -> tuple[np.ndarray, list[Cell], list[int]]:
    """
    Build a single distance matrix over ALL boundary cells of ALL children.

    Returns:
        M:             (N x N) combined distance matrix
        all_cells:     ordered list of all boundary cells (length N)
        external_mask: indices into all_cells that are on the parent boundary
    """
    all_cells: list[Cell] = []
    offsets: list[int] = []
    for child in children:
        offsets.append(len(all_cells))
        all_cells.extend(child.boundary_cells)

    N = len(all_cells)
    M = np.full((N, N), INF, dtype=np.float64)

    # Fill within-child entries from their transfer matrices
    for ci, child in enumerate(children):
        n = len(child.boundary_cells)
        off = offsets[ci]
        if child.transfer_matrix is not None and child.transfer_matrix.size > 0:
            M[off: off + n, off: off + n] = child.transfer_matrix

    # Fill cross-child entries: adjacent cells across block boundaries cost 1
    cell_to_global: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(all_cells):
        cell_to_global[(cell.row, cell.col)] = idx

    for idx, cell in enumerate(all_cells):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell.row + dr, cell.col + dc
            neighbor_idx = cell_to_global.get((nr, nc))
            if neighbor_idx is not None and neighbor_idx != idx:
                # Only set crossing cost between DIFFERENT children
                child_of_idx = _child_index_of(idx, offsets, children)
                child_of_nbr = _child_index_of(neighbor_idx, offsets, children)
                if child_of_idx != child_of_nbr:
                    M[idx, neighbor_idx] = 1.0

    # Identify external indices
    parent_set = {(c.row, c.col) for c in parent.boundary_cells}
    external_mask = [i for i, c in enumerate(all_cells) if (c.row, c.col) in parent_set]

    return M, all_cells, external_mask


def _child_index_of(global_idx: int, offsets: list[int], children: list[Block]) -> int:
    for ci in range(len(children) - 1, -1, -1):
        if global_idx >= offsets[ci]:
            return ci
    return 0


# ---------------------------------------------------------------------------
# Compose transfer matrices via Floyd-Warshall
# ---------------------------------------------------------------------------

def compose_transfer_matrix(
    parent: Block,
    children: list[Block],
    grid: Grid,
) -> np.ndarray:
    """Compose children's TMs into a parent TM by Floyd-Warshall elimination."""
    M, all_cells, external_mask = build_combined_matrix(children, grid, parent)

    if M.size == 0:
        n = len(parent.boundary_cells)
        return np.full((n, n), INF, dtype=np.float64)

    floyd_warshall(M)

    n_ext = len(external_mask)
    T_parent = np.full((n_ext, n_ext), INF, dtype=np.float64)
    for i, gi in enumerate(external_mask):
        for j, gj in enumerate(external_mask):
            T_parent[i, j] = M[gi, gj]

    # Re-order to match parent's boundary cell ordering
    parent_order = {(c.row, c.col): i for i, c in enumerate(parent.boundary_cells)}
    ext_cells = [all_cells[gi] for gi in external_mask]

    n_parent = len(parent.boundary_cells)
    T_ordered = np.full((n_parent, n_parent), INF, dtype=np.float64)
    for i, ci in enumerate(ext_cells):
        pi = parent_order.get((ci.row, ci.col))
        if pi is None:
            continue
        for j, cj in enumerate(ext_cells):
            pj = parent_order.get((cj.row, cj.col))
            if pj is None:
                continue
            T_ordered[pi, pj] = T_parent[i, j]

    return T_ordered


# ---------------------------------------------------------------------------
# Full bottom-up computation
# ---------------------------------------------------------------------------

def _compute_single_l1_tm(args: tuple) -> tuple[tuple[int, int, int], np.ndarray]:
    """Worker for parallel Level-1 TM computation (must be top-level for pickling)."""
    grid_data, block_id, bcells_raw, rs, re, cs, ce = args
    grid = Grid(grid_data)
    bcells = [Cell(r, c) for r, c in bcells_raw]
    tm = compute_level1_transfer_matrix(grid, bcells, rs, re, cs, ce)
    return block_id, tm


def compute_all_transfer_matrices(
    all_blocks: dict[tuple[int, int, int], Block],
    grid: Grid,
    block_size: int,
    active_only: bool = False,
    active_set: Optional[set[tuple[int, int, int]]] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Compute transfer matrices bottom-up: Level-1 via BFS, higher via composition."""

    max_level = max(bid[0] for bid in all_blocks) if all_blocks else 0

    # --- Level 1: parallel BFS ---
    l1_args = []
    for bid, blk in all_blocks.items():
        if blk.level != 1:
            continue
        if active_only and active_set and bid not in active_set:
            continue
        bcells_raw = [(c.row, c.col) for c in blk.boundary_cells]
        l1_args.append((grid.data, bid, bcells_raw, blk.row_start, blk.row_end, blk.col_start, blk.col_end))

    if max_workers != 1 and len(l1_args) > 4:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for bid, tm in pool.map(_compute_single_l1_tm, l1_args):
                all_blocks[bid].transfer_matrix = tm
    else:
        for args in l1_args:
            bid, tm = _compute_single_l1_tm(args)
            all_blocks[bid].transfer_matrix = tm

    # --- Higher levels: composition ---
    for level in range(2, max_level + 1):
        for bid, blk in all_blocks.items():
            if blk.level != level or blk.children is None:
                continue
            if active_only and active_set and bid not in active_set:
                continue

            children_ready = all(
                c.transfer_matrix is not None for c in blk.children
            )
            if not children_ready:
                continue

            blk.transfer_matrix = compose_transfer_matrix(blk, blk.children, grid)
