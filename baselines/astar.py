"""A* pathfinding with step-by-step exploration tracking for animated visualization."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

from hlp.grid import Cell, Grid


@dataclass
class AStarResult:
    path: Optional[list[Cell]]
    cost: float
    explored_order: list[tuple[int, int]]
    frontier_at_end: list[tuple[int, int]]
    computation_time_ms: float
    nodes_explored: int = 0


@dataclass
class ExplorationStep:
    step: int
    current: tuple[int, int]
    explored: set[tuple[int, int]] = field(default_factory=set)
    frontier: set[tuple[int, int]] = field(default_factory=set)


def _manhattan(r1: int, c1: int, r2: int, c2: int) -> float:
    return abs(r1 - r2) + abs(c1 - c2)


def astar_generator(
    grid: Grid,
    source: Cell,
    goal: Cell,
) -> Generator[ExplorationStep, None, AStarResult]:
    """
    A* generator that yields ExplorationStep at each node expansion,
    then returns the final AStarResult.
    """
    t0 = time.perf_counter()

    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return AStarResult(None, float("inf"), [], [], 0.0, 0)

    if source == goal:
        return AStarResult([source], 0.0, [], [], 0.0, 0)

    sr, sc = source.row, source.col
    gr, gc = goal.row, goal.col

    g_score: dict[tuple[int, int], float] = {(sr, sc): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    open_set: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_set, (_manhattan(sr, sc, gr, gc), 0, sr, sc))
    in_open: set[tuple[int, int]] = {(sr, sc)}
    closed: set[tuple[int, int]] = set()
    explored_order: list[tuple[int, int]] = []
    step = 0

    while open_set:
        _f, _g, r, c = heapq.heappop(open_set)
        if (r, c) in closed:
            continue

        closed.add((r, c))
        in_open.discard((r, c))
        explored_order.append((r, c))
        step += 1

        yield ExplorationStep(
            step=step,
            current=(r, c),
            explored=set(closed),
            frontier=set(in_open),
        )

        if r == gr and c == gc:
            path = _trace(parent, source, goal)
            elapsed = (time.perf_counter() - t0) * 1000
            return AStarResult(path, float(len(path) - 1), explored_order,
                               list(in_open), elapsed, step)

        cur_g = g_score[(r, c)]
        for nr, nc in grid.neighbors(r, c):
            if (nr, nc) in closed:
                continue
            ng = cur_g + 1.0
            if ng < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = ng
                parent[(nr, nc)] = (r, c)
                f = ng + _manhattan(nr, nc, gr, gc)
                heapq.heappush(open_set, (f, int(ng), nr, nc))
                in_open.add((nr, nc))

    elapsed = (time.perf_counter() - t0) * 1000
    return AStarResult(None, float("inf"), explored_order, list(in_open), elapsed, step)


def astar(grid: Grid, source: Cell, goal: Cell) -> AStarResult:
    """Run A* to completion, returning the final result."""
    gen = astar_generator(grid, source, goal)
    result = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        result = e.value
    if result is None:
        result = AStarResult(None, float("inf"), [], [], 0.0, 0)
    return result


def _trace(
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
