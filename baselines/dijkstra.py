"""Dijkstra pathfinding with step-by-step exploration tracking for animated visualization."""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

from hlp.grid import Cell, Grid


@dataclass
class DijkstraResult:
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


def dijkstra_generator(
    grid: Grid,
    source: Cell,
    goal: Cell,
) -> Generator[ExplorationStep, None, DijkstraResult]:
    """
    Dijkstra generator that yields ExplorationStep at each node expansion,
    then returns the final DijkstraResult.
    """
    t0 = time.perf_counter()

    if not grid.is_free(source.row, source.col) or not grid.is_free(goal.row, goal.col):
        return DijkstraResult(None, float("inf"), [], [], 0.0, 0)

    if source == goal:
        return DijkstraResult([source], 0.0, [], [], 0.0, 0)

    sr, sc = source.row, source.col
    gr, gc = goal.row, goal.col

    dist: dict[tuple[int, int], float] = {(sr, sc): 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    heap: list[tuple[float, int, int]] = [(0.0, sr, sc)]
    closed: set[tuple[int, int]] = set()
    in_open: set[tuple[int, int]] = {(sr, sc)}
    explored_order: list[tuple[int, int]] = []
    step = 0

    while heap:
        d, r, c = heapq.heappop(heap)
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
            return DijkstraResult(path, float(len(path) - 1), explored_order,
                                  list(in_open), elapsed, step)

        for nr, nc in grid.neighbors(r, c):
            if (nr, nc) in closed:
                continue
            nd = d + 1.0
            if nd < dist.get((nr, nc), float("inf")):
                dist[(nr, nc)] = nd
                parent[(nr, nc)] = (r, c)
                heapq.heappush(heap, (nd, nr, nc))
                in_open.add((nr, nc))

    elapsed = (time.perf_counter() - t0) * 1000
    return DijkstraResult(None, float("inf"), explored_order, list(in_open), elapsed, step)


def dijkstra(grid: Grid, source: Cell, goal: Cell) -> DijkstraResult:
    """Run Dijkstra to completion, returning the final result."""
    gen = dijkstra_generator(grid, source, goal)
    result = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        result = e.value
    if result is None:
        result = DijkstraResult(None, float("inf"), [], [], 0.0, 0)
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
