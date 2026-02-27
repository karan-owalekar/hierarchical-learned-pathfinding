"""Map generators that produce grids with obstacles, plus start and goal positions."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from hlp.grid import Cell, Grid, bfs_shortest_path, generate_grid


# ---------------------------------------------------------------------------
# Public interface — every generator returns (Grid, start, goal)
# ---------------------------------------------------------------------------

def generate_random(
    h: int,
    w: int,
    density: float = 0.2,
    seed: Optional[int] = None,
) -> tuple[Grid, Cell, Cell]:
    grid = generate_grid(h, w, density, seed=seed, ensure_connected=False)
    rng = random.Random(seed)
    start, goal = _place_start_goal(grid.data, rng)
    return grid, start, goal


def generate_dfs_maze(
    h: int,
    w: int,
    seed: Optional[int] = None,
) -> tuple[Grid, Cell, Cell]:
    rng = random.Random(seed)
    data = np.ones((h, w), dtype=np.uint8)

    sr, sc = 1 if h > 1 else 0, 1 if w > 1 else 0
    data[sr, sc] = 0
    stack = [(sr, sc)]
    step = 2

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-step, 0), (step, 0), (0, -step), (0, step)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < h - 1 and 0 < nc < w - 1 and data[nr, nc] == 1:
                neighbors.append((nr, nc, r + dr // 2, c + dc // 2))
        if neighbors:
            nr, nc, wr, wc = rng.choice(neighbors)
            data[wr, wc] = 0
            data[nr, nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    start, goal = _place_start_goal_maze(data, rng)
    return Grid(data), start, goal


def generate_spiral(
    h: int,
    w: int,
    seed: Optional[int] = None,
) -> tuple[Grid, Cell, Cell]:
    rng = random.Random(seed)
    data = np.zeros((h, w), dtype=np.uint8)

    gap_size = max(2, min(h, w) // 16)
    layer = 0
    while True:
        top = layer * 4
        left = layer * 4
        bottom = h - 1 - layer * 4
        right = w - 1 - layer * 4
        if top >= bottom or left >= right:
            break

        wall_row = top + 2
        wall_col = right - 2
        wall_bottom = bottom - 2
        wall_left = left + 2

        if wall_row >= bottom or wall_col <= left or wall_bottom <= top or wall_left >= right:
            break

        for c in range(left, right + 1):
            if c < right + 1:
                data[wall_row, min(c, w - 1)] = 1
        gap_start = rng.randint(right - gap_size, right)
        for c in range(gap_start, min(gap_start + gap_size, w)):
            data[wall_row, c] = 0

        for r in range(wall_row, bottom + 1):
            if wall_col < w:
                data[min(r, h - 1), wall_col] = 1
        gap_start = rng.randint(bottom - gap_size, bottom)
        for r in range(gap_start, min(gap_start + gap_size, h)):
            data[r, wall_col] = 0

        for c in range(right, left - 1, -1):
            if wall_bottom < h:
                data[wall_bottom, max(c, 0)] = 1
        gap_start = rng.randint(left, left + gap_size)
        for c in range(gap_start, min(gap_start + gap_size, w)):
            data[wall_bottom, c] = 0

        for r in range(bottom, wall_row - 1, -1):
            if wall_left >= 0:
                data[max(r, 0), wall_left] = 1
        gap_start = rng.randint(wall_row, wall_row + gap_size)
        for r in range(gap_start, min(gap_start + gap_size, h)):
            data[r, wall_left] = 0

        layer += 1

    start, goal = _place_start_goal(data, rng)
    return Grid(data), start, goal


def generate_recursive_division(
    h: int,
    w: int,
    seed: Optional[int] = None,
) -> tuple[Grid, Cell, Cell]:
    rng = random.Random(seed)
    data = np.zeros((h, w), dtype=np.uint8)

    for r in range(h):
        data[r, 0] = 1
        data[r, w - 1] = 1
    for c in range(w):
        data[0, c] = 1
        data[h - 1, c] = 1

    _divide(data, 1, 1, h - 2, w - 2, rng)

    data[0, 0] = 0
    data[h - 1, w - 1] = 0

    start, goal = _place_start_goal(data, rng)
    return Grid(data), start, goal


def generate_rooms(
    h: int,
    w: int,
    seed: Optional[int] = None,
    num_rooms: Optional[int] = None,
) -> tuple[Grid, Cell, Cell]:
    rng = random.Random(seed)
    data = np.ones((h, w), dtype=np.uint8)

    if num_rooms is None:
        num_rooms = max(4, (h * w) // 400)

    rooms: list[tuple[int, int, int, int]] = []
    for _ in range(num_rooms * 10):
        rh = rng.randint(3, max(3, h // 4))
        rw = rng.randint(3, max(3, w // 4))
        rr = rng.randint(1, max(1, h - rh - 1))
        rc = rng.randint(1, max(1, w - rw - 1))

        overlap = False
        for er, ec, eh, ew in rooms:
            if rr < er + eh + 1 and rr + rh + 1 > er and rc < ec + ew + 1 and rc + rw + 1 > ec:
                overlap = True
                break
        if overlap:
            continue

        for r in range(rr, min(rr + rh, h)):
            for c in range(rc, min(rc + rw, w)):
                data[r, c] = 0
        rooms.append((rr, rc, rh, rw))
        if len(rooms) >= num_rooms:
            break

    for i in range(len(rooms) - 1):
        r1, c1, rh1, rw1 = rooms[i]
        r2, c2, rh2, rw2 = rooms[i + 1]
        cr1, cc1 = r1 + rh1 // 2, c1 + rw1 // 2
        cr2, cc2 = r2 + rh2 // 2, c2 + rw2 // 2
        _carve_corridor(data, cr1, cc1, cr2, cc2, rng)

    start, goal = _place_start_goal(data, rng)
    return Grid(data), start, goal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _place_start_goal(data: np.ndarray, rng: random.Random) -> tuple[Cell, Cell]:
    h, w = data.shape
    free_cells = list(zip(*np.where(data == 0)))
    if len(free_cells) < 2:
        data[0, 0] = 0
        data[h - 1, w - 1] = 0
        return Cell(0, 0), Cell(h - 1, w - 1)

    grid = Grid(data)
    for _ in range(200):
        s = rng.choice(free_cells)
        g = rng.choice(free_cells)
        start = Cell(int(s[0]), int(s[1]))
        goal = Cell(int(g[0]), int(g[1]))
        if start == goal:
            continue
        manhattan = abs(start.row - goal.row) + abs(start.col - goal.col)
        if manhattan < max(4, min(h, w) // 4):
            continue
        if bfs_shortest_path(grid, start, goal) is not None:
            return start, goal

    # Fallback: pick any two connected free cells
    for s in free_cells:
        start = Cell(int(s[0]), int(s[1]))
        for g in free_cells:
            goal = Cell(int(g[0]), int(g[1]))
            if start != goal and bfs_shortest_path(grid, start, goal) is not None:
                return start, goal

    data[0, 0] = 0
    data[h - 1, w - 1] = 0
    return Cell(0, 0), Cell(h - 1, w - 1)


def _place_start_goal_maze(data: np.ndarray, rng: random.Random) -> tuple[Cell, Cell]:
    h, w = data.shape
    free_cells = list(zip(*np.where(data == 0)))
    if len(free_cells) < 2:
        data[1, 1] = 0
        data[h - 2, w - 2] = 0
        return Cell(1, 1), Cell(h - 2, w - 2)

    rng.shuffle(free_cells)
    best_pair = (Cell(int(free_cells[0][0]), int(free_cells[0][1])),
                 Cell(int(free_cells[1][0]), int(free_cells[1][1])))
    best_dist = 0
    candidates = free_cells[: min(30, len(free_cells))]
    grid = Grid(data)
    for i, s in enumerate(candidates):
        for g in candidates[i + 1:]:
            start = Cell(int(s[0]), int(s[1]))
            goal = Cell(int(g[0]), int(g[1]))
            result = bfs_shortest_path(grid, start, goal)
            if result is not None and result[1] > best_dist:
                best_dist = result[1]
                best_pair = (start, goal)
    return best_pair


def _divide(
    data: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    rng: random.Random,
) -> None:
    if height < 3 or width < 3:
        return

    if width >= height:
        wall_col = left + 2 * rng.randint(1, max(1, (width - 1) // 2))
        if wall_col > left + width - 1:
            return
        gap_row = top + rng.randint(0, height - 1)
        for r in range(top, top + height):
            if r != gap_row:
                data[r, wall_col] = 1
        _divide(data, top, left, height, wall_col - left, rng)
        _divide(data, top, wall_col + 1, height, left + width - wall_col - 1, rng)
    else:
        wall_row = top + 2 * rng.randint(1, max(1, (height - 1) // 2))
        if wall_row > top + height - 1:
            return
        gap_col = left + rng.randint(0, width - 1)
        for c in range(left, left + width):
            if c != gap_col:
                data[wall_row, c] = 1
        _divide(data, top, left, wall_row - top, width, rng)
        _divide(data, wall_row + 1, left, top + height - wall_row - 1, width, rng)


def _carve_corridor(
    data: np.ndarray,
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    rng: random.Random,
) -> None:
    h, w = data.shape
    r, c = r1, c1
    if rng.random() < 0.5:
        while c != c2:
            data[max(0, min(r, h - 1)), max(0, min(c, w - 1))] = 0
            c += 1 if c2 > c else -1
        while r != r2:
            data[max(0, min(r, h - 1)), max(0, min(c, w - 1))] = 0
            r += 1 if r2 > r else -1
    else:
        while r != r2:
            data[max(0, min(r, h - 1)), max(0, min(c, w - 1))] = 0
            r += 1 if r2 > r else -1
        while c != c2:
            data[max(0, min(r, h - 1)), max(0, min(c, w - 1))] = 0
            c += 1 if c2 > c else -1
    data[max(0, min(r2, h - 1)), max(0, min(c2, w - 1))] = 0


# ---------------------------------------------------------------------------
# Map type registry
# ---------------------------------------------------------------------------

MAP_TYPE_KEYS = [
    "random_scatter", "dfs_maze", "spiral", "recursive_division", "rooms",
]
MAP_TYPE_NAMES = [
    "Random Scatter", "DFS Maze", "Spiral",
    "Recursive Division", "Rooms & Corridors",
]
MAP_TYPE_GENERATORS = dict(zip(MAP_TYPE_KEYS, [
    generate_random, generate_dfs_maze, generate_spiral,
    generate_recursive_division, generate_rooms,
]))
