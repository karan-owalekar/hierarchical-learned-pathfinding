"""Path extraction: entry/exit embeddings, corridor transfer matrix, distance query,
and full cell-by-cell path reconstruction."""

from __future__ import annotations

from typing import Optional

import numpy as np

from hlp.decomposition import Block, get_block_for_cell
from hlp.grid import (
    Cell,
    Grid,
    bfs_all_distances_within_block,
    bfs_path_within_block,
    bfs_shortest_path,
)
from hlp.tropical import INF, floyd_warshall


# ---------------------------------------------------------------------------
# Entry / exit embeddings
# ---------------------------------------------------------------------------

def compute_entry_embedding(
    grid: Grid,
    source: Cell,
    block: Block,
) -> np.ndarray:
    """BFS from source to all boundary cells of its block → distance vector."""
    n = len(block.boundary_cells)
    embed = np.full(n, INF, dtype=np.float64)
    if not grid.is_free(source.row, source.col):
        return embed

    dists = bfs_all_distances_within_block(
        grid, source,
        block.row_start, block.row_end,
        block.col_start, block.col_end,
    )
    for i, bc in enumerate(block.boundary_cells):
        if bc in dists:
            embed[i] = dists[bc]
    return embed


def compute_exit_embedding(
    grid: Grid,
    goal: Cell,
    block: Block,
) -> np.ndarray:
    """BFS from goal to all boundary cells of its block → distance vector."""
    return compute_entry_embedding(grid, goal, block)


# ---------------------------------------------------------------------------
# Corridor transfer matrix (Floyd-Warshall over corridor blocks)
# ---------------------------------------------------------------------------

def build_corridor_transfer_matrix(
    corridor_blocks: list[Block],
    grid: Grid,
) -> tuple[np.ndarray, list[Cell]]:
    """
    Build an all-pairs distance matrix over boundary cells of all corridor
    blocks, using Floyd-Warshall.

    Returns:
        M:         (N x N) distance matrix
        all_cells: ordered list of all boundary cells across corridor blocks
    """
    all_cells: list[Cell] = []
    offsets: list[int] = []

    for blk in corridor_blocks:
        offsets.append(len(all_cells))
        all_cells.extend(blk.boundary_cells)

    N = len(all_cells)
    if N == 0:
        return np.empty((0, 0), dtype=np.float64), []

    M = np.full((N, N), INF, dtype=np.float64)

    for ci, blk in enumerate(corridor_blocks):
        n = len(blk.boundary_cells)
        off = offsets[ci]
        if blk.transfer_matrix is not None and blk.transfer_matrix.size > 0:
            M[off: off + n, off: off + n] = blk.transfer_matrix

    cell_to_global: dict[tuple[int, int], int] = {}
    for idx, cell in enumerate(all_cells):
        cell_to_global[(cell.row, cell.col)] = idx

    for idx, cell in enumerate(all_cells):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = cell.row + dr, cell.col + dc
            nbr_idx = cell_to_global.get((nr, nc))
            if nbr_idx is not None and nbr_idx != idx:
                ci_a = _block_index_of(idx, offsets)
                ci_b = _block_index_of(nbr_idx, offsets)
                if ci_a != ci_b:
                    M[idx, nbr_idx] = 1.0

    floyd_warshall(M)
    return M, all_cells


def _block_index_of(global_idx: int, offsets: list[int]) -> int:
    for ci in range(len(offsets) - 1, -1, -1):
        if global_idx >= offsets[ci]:
            return ci
    return 0


# ---------------------------------------------------------------------------
# Distance query
# ---------------------------------------------------------------------------

def query_distance(
    entry_embed: np.ndarray,
    corridor_tm: np.ndarray,
    exit_embed: np.ndarray,
    src_boundary_indices: list[int],
    goal_boundary_indices: list[int],
) -> tuple[float, int, int]:
    """
    distance = min_{i in src_boundary, j in goal_boundary}
                   (entry[i_local] + T[i_global][j_global] + exit[j_local])

    Returns (distance, best_i_global, best_j_global).
    """
    best = INF
    best_i = -1
    best_j = -1

    for i_local, i_global in enumerate(src_boundary_indices):
        ei = entry_embed[i_local]
        if ei >= INF:
            continue
        for j_local, j_global in enumerate(goal_boundary_indices):
            ej = exit_embed[j_local]
            if ej >= INF:
                continue
            val = ei + corridor_tm[i_global, j_global] + ej
            if val < best:
                best = val
                best_i = i_global
                best_j = j_global

    return best, best_i, best_j


# ---------------------------------------------------------------------------
# Full path distance (single-function convenience)
# ---------------------------------------------------------------------------

def compute_path_distance(
    grid: Grid,
    source: Cell,
    goal: Cell,
    corridor_blocks: list[Block],
    block_size: int,
) -> float:
    """One-shot: compute shortest-path distance through the corridor."""
    src_bid = get_block_for_cell(source, block_size)
    goal_bid = get_block_for_cell(goal, block_size)

    src_block: Optional[Block] = None
    goal_block: Optional[Block] = None
    for blk in corridor_blocks:
        if blk.block_id == src_bid:
            src_block = blk
        if blk.block_id == goal_bid:
            goal_block = blk

    if src_block is None or goal_block is None:
        return INF

    # Same block shortcut
    if src_block is goal_block:
        dists = bfs_all_distances_within_block(
            grid, source,
            src_block.row_start, src_block.row_end,
            src_block.col_start, src_block.col_end,
        )
        return dists.get(goal, INF)

    entry = compute_entry_embedding(grid, source, src_block)
    exit_ = compute_exit_embedding(grid, goal, goal_block)

    M, all_cells = build_corridor_transfer_matrix(corridor_blocks, grid)
    if M.size == 0:
        return INF

    cell_to_global = {(c.row, c.col): i for i, c in enumerate(all_cells)}

    src_indices = [cell_to_global[(c.row, c.col)]
                   for c in src_block.boundary_cells
                   if (c.row, c.col) in cell_to_global]
    goal_indices = [cell_to_global[(c.row, c.col)]
                    for c in goal_block.boundary_cells
                    if (c.row, c.col) in cell_to_global]

    dist, _, _ = query_distance(entry, M, exit_, src_indices, goal_indices)
    return dist


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def reconstruct_path(
    grid: Grid,
    source: Cell,
    goal: Cell,
    corridor_blocks: list[Block],
    block_size: int,
) -> Optional[list[Cell]]:
    """Reconstruct the full cell-by-cell shortest path through the corridor.

    Uses BFS restricted to cells within corridor blocks for robust reconstruction.
    """
    src_bid = get_block_for_cell(source, block_size)
    goal_bid = get_block_for_cell(goal, block_size)

    src_block: Optional[Block] = None
    goal_block: Optional[Block] = None
    for blk in corridor_blocks:
        if blk.block_id == src_bid:
            src_block = blk
        if blk.block_id == goal_bid:
            goal_block = blk

    if src_block is None or goal_block is None:
        return None

    # Same block shortcut
    if src_block is goal_block:
        return bfs_path_within_block(
            grid, source, goal,
            src_block.row_start, src_block.row_end,
            src_block.col_start, src_block.col_end,
        )

    # Build corridor cell set and run BFS restricted to these cells
    corridor_cells = set()
    for blk in corridor_blocks:
        for r in range(blk.row_start, blk.row_end):
            for c in range(blk.col_start, blk.col_end):
                if grid.is_free(r, c):
                    corridor_cells.add((r, c))

    return _bfs_corridor(grid, source, goal, corridor_cells)


def _bfs_corridor(
    grid: Grid,
    source: Cell,
    goal: Cell,
    allowed_cells: set[tuple[int, int]],
) -> Optional[list[Cell]]:
    """BFS restricted to a set of allowed cells."""
    from collections import deque

    if (source.row, source.col) not in allowed_cells:
        allowed_cells.add((source.row, source.col))
    if (goal.row, goal.col) not in allowed_cells:
        allowed_cells.add((goal.row, goal.col))

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
            if (nr, nc) in allowed_cells and (nr, nc) not in visited and grid.is_free(nr, nc):
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
