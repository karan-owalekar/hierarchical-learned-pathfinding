from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class Cell:
    row: int
    col: int


class Grid:
    __slots__ = ("data", "height", "width")

    def __init__(self, data: np.ndarray) -> None:
        self.data = data.astype(np.uint8)
        self.height, self.width = data.shape

    def is_free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.data[r, c] == 0

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def neighbors(self, r: int, c: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.data[nr, nc] == 0:
                result.append((nr, nc))
        return result


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_grid(
    height: int,
    width: int,
    obstacle_density: float = 0.2,
    seed: Optional[int] = None,
    ensure_connected: bool = True,
    max_retries: int = 50,
) -> Grid:
    rng = np.random.RandomState(seed)

    for _ in range(max_retries):
        data = (rng.random((height, width)) < obstacle_density).astype(np.uint8)
        data[0, 0] = 0
        data[height - 1, width - 1] = 0

        grid = Grid(data)
        if not ensure_connected:
            return grid

        if bfs_shortest_path(grid, Cell(0, 0), Cell(height - 1, width - 1)) is not None:
            return grid

    return Grid(np.zeros((height, width), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Grid I/O
# ---------------------------------------------------------------------------

def save_grid(grid: Grid, filepath: str) -> None:
    np.save(filepath, grid.data)


def load_grid(filepath: str) -> Grid:
    return Grid(np.load(filepath))


# ---------------------------------------------------------------------------
# BFS – full grid
# ---------------------------------------------------------------------------

def bfs_shortest_path(
    grid: Grid,
    source: Cell,
    goal: Cell,
) -> Optional[tuple[list[Cell], float]]:
    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return None
    if source == goal:
        return [source], 0.0

    visited = np.zeros((grid.height, grid.width), dtype=bool)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue: deque[tuple[int, int]] = deque()

    sr, sc = source.row, source.col
    visited[sr, sc] = True
    queue.append((sr, sc))

    while queue:
        r, c = queue.popleft()
        for nr, nc in grid.neighbors(r, c):
            if not visited[nr, nc]:
                visited[nr, nc] = True
                parent[(nr, nc)] = (r, c)
                if nr == goal.row and nc == goal.col:
                    path = _trace_path(parent, source, goal)
                    return path, float(len(path) - 1)
                queue.append((nr, nc))

    return None


# ---------------------------------------------------------------------------
# BFS – restricted to a sub-region (for transfer matrix computation)
# ---------------------------------------------------------------------------

def bfs_within_block(
    grid: Grid,
    source: Cell,
    goal: Cell,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Optional[float]:
    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return None
    if source == goal:
        return 0.0

    h = row_end - row_start
    w = col_end - col_start
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int, float]] = deque()

    lr, lc = source.row - row_start, source.col - col_start
    visited[lr, lc] = True
    queue.append((source.row, source.col, 0.0))

    while queue:
        r, c, d = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if row_start <= nr < row_end and col_start <= nc < col_end and grid.data[nr, nc] == 0:
                lr2, lc2 = nr - row_start, nc - col_start
                if not visited[lr2, lc2]:
                    visited[lr2, lc2] = True
                    if nr == goal.row and nc == goal.col:
                        return d + 1.0
                    queue.append((nr, nc, d + 1.0))
    return None


def bfs_all_distances_within_block(
    grid: Grid,
    source: Cell,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> dict[Cell, float]:
    distances: dict[Cell, float] = {}
    if not grid.is_free(source.row, source.col):
        return distances

    h = row_end - row_start
    w = col_end - col_start
    visited = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int, float]] = deque()

    lr, lc = source.row - row_start, source.col - col_start
    visited[lr, lc] = True
    queue.append((source.row, source.col, 0.0))
    distances[source] = 0.0

    while queue:
        r, c, d = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if row_start <= nr < row_end and col_start <= nc < col_end and grid.data[nr, nc] == 0:
                lr2, lc2 = nr - row_start, nc - col_start
                if not visited[lr2, lc2]:
                    visited[lr2, lc2] = True
                    nd = d + 1.0
                    distances[Cell(nr, nc)] = nd
                    queue.append((nr, nc, nd))
    return distances


def bfs_path_within_block(
    grid: Grid,
    source: Cell,
    goal: Cell,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> Optional[list[Cell]]:
    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return None
    if source == goal:
        return [source]

    h = row_end - row_start
    w = col_end - col_start
    visited = np.zeros((h, w), dtype=bool)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue: deque[tuple[int, int]] = deque()

    lr, lc = source.row - row_start, source.col - col_start
    visited[lr, lc] = True
    queue.append((source.row, source.col))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if row_start <= nr < row_end and col_start <= nc < col_end and grid.data[nr, nc] == 0:
                lr2, lc2 = nr - row_start, nc - col_start
                if not visited[lr2, lc2]:
                    visited[lr2, lc2] = True
                    parent[(nr, nc)] = (r, c)
                    if nr == goal.row and nc == goal.col:
                        return _trace_path(parent, source, goal)
                    queue.append((nr, nc))
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trace_path(
    parent: dict[tuple[int, int], tuple[int, int]],
    source: Cell,
    goal: Cell,
) -> list[Cell]:
    path: list[Cell] = []
    cur = (goal.row, goal.col)
    src = (source.row, source.col)
    while cur != src:
        path.append(Cell(*cur))
        cur = parent[cur]
    path.append(source)
    path.reverse()
    return path
