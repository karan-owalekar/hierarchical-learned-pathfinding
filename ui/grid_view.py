"""Grid rendering, cell interaction, and A*/Dijkstra animation controller."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pygame

from hlp.grid import Cell, Grid
from ui.theme import (
    BG_DARK,
    CELL_CORRIDOR,
    CELL_EMPTY,
    CELL_EXPLORED,
    CELL_FRONTIER,
    CELL_GOAL,
    CELL_OBSTACLE,
    CELL_PATH,
    CELL_RADIUS,
    CELL_START,
    CELL_GAP,
    MIN_CELL_SIZE,
)


class GridView:
    def __init__(self) -> None:
        self.grid: Optional[Grid] = None
        self.start: Optional[Cell] = None
        self.goal: Optional[Cell] = None
        self.path: Optional[list[Cell]] = None

        # Exploration visualization (A* / Dijkstra)
        self.explored: set[tuple[int, int]] = set()
        self.frontier: set[tuple[int, int]] = set()

        # Corridor visualization (HLP)
        self.corridor_blocks: set[tuple[int, int, int]] = set()
        self.block_size: int = 16

        # Layout
        self.area = pygame.Rect(0, 0, 0, 0)
        self.cell_size = 0
        self.grid_origin = (0, 0)

        # Interaction state
        self.drawing = False
        self.erasing = False
        self.placing_start = False
        self.placing_goal = False

        # Animation
        self.animation: Optional[AnimationController] = None

    # ----- layout -----

    def layout(self, area: pygame.Rect) -> None:
        self.area = area
        self._recalc_cell_size()

    def set_grid(self, grid: Grid, start: Optional[Cell] = None, goal: Optional[Cell] = None) -> None:
        self.grid = grid
        self.start = start
        self.goal = goal
        self.clear_overlays()
        self._recalc_cell_size()

    def clear_overlays(self) -> None:
        self.path = None
        self.explored = set()
        self.frontier = set()
        self.corridor_blocks = set()
        self.animation = None

    def _recalc_cell_size(self) -> None:
        if self.grid is None:
            return
        avail_w = self.area.width - 2
        avail_h = self.area.height - 2
        self.cell_size = max(
            min(avail_w // self.grid.width, avail_h // self.grid.height),
            MIN_CELL_SIZE,
        )
        total_w = self.cell_size * self.grid.width
        total_h = self.cell_size * self.grid.height
        ox = self.area.x + (self.area.width - total_w) // 2
        oy = self.area.y + (self.area.height - total_h) // 2
        self.grid_origin = (ox, oy)

    # ----- event handling -----

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        """Returns a status message string if state changed, else None."""
        if self.grid is None:
            return None

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                self.placing_start = True
                self.placing_goal = False
                return "Click a cell to set START"
            elif event.key == pygame.K_g:
                self.placing_goal = True
                self.placing_start = False
                return "Click a cell to set GOAL"

        if event.type == pygame.MOUSEBUTTONDOWN:
            cell = self._cell_at(event.pos)
            if cell is None:
                return None

            if self.placing_start:
                if self.grid.is_free(cell.row, cell.col):
                    self.start = cell
                    self.placing_start = False
                    return f"Start set to ({cell.row}, {cell.col})"
                return "Cannot place start on obstacle"

            if self.placing_goal:
                if self.grid.is_free(cell.row, cell.col):
                    self.goal = cell
                    self.placing_goal = False
                    return f"Goal set to ({cell.row}, {cell.col})"
                return "Cannot place goal on obstacle"

            if event.button == 1 and cell != self.start and cell != self.goal:
                if self.grid.data[cell.row, cell.col] == 0:
                    self.drawing = True
                    self.grid.data[cell.row, cell.col] = 1
                else:
                    self.erasing = True
                    self.grid.data[cell.row, cell.col] = 0
            elif event.button == 3 and cell != self.start and cell != self.goal:
                self.erasing = True
                self.grid.data[cell.row, cell.col] = 0

        if event.type == pygame.MOUSEMOTION:
            if self.drawing or self.erasing:
                cell = self._cell_at(event.pos)
                if cell and cell != self.start and cell != self.goal:
                    self.grid.data[cell.row, cell.col] = 1 if self.drawing else 0

        if event.type == pygame.MOUSEBUTTONUP:
            self.drawing = False
            self.erasing = False

        return None

    # ----- drawing -----

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, BG_DARK, self.area)
        if self.grid is None:
            return

        if self.animation:
            if self.animation.running:
                self.animation.tick()
                self.explored = self.animation.explored
                self.frontier = self.animation.frontier
            if self.animation.finished and self.animation.final_path:
                self.path = self.animation.final_path
                self.animation.final_path = None

        ox, oy = self.grid_origin
        cs = self.cell_size
        inner = cs - CELL_GAP if cs > 3 else cs
        radius = CELL_RADIUS if cs > 4 else 0

        path_set = {(c.row, c.col) for c in self.path} if self.path else set()
        corridor_cells = self._corridor_cell_set()

        for r in range(self.grid.height):
            for c in range(self.grid.width):
                x = ox + c * cs
                y = oy + r * cs
                rect = pygame.Rect(x, y, inner, inner)

                if self.start and r == self.start.row and c == self.start.col:
                    color = CELL_START
                elif self.goal and r == self.goal.row and c == self.goal.col:
                    color = CELL_GOAL
                elif (r, c) in path_set:
                    color = CELL_PATH
                elif (r, c) in self.frontier:
                    color = CELL_FRONTIER
                elif (r, c) in self.explored:
                    color = CELL_EXPLORED
                elif (r, c) in corridor_cells:
                    color = CELL_CORRIDOR
                elif self.grid.data[r, c] == 1:
                    color = CELL_OBSTACLE
                else:
                    color = CELL_EMPTY

                pygame.draw.rect(surface, color, rect, border_radius=radius)

    # ----- helpers -----

    def _cell_at(self, pos: tuple[int, int]) -> Optional[Cell]:
        if self.grid is None:
            return None
        mx, my = pos
        ox, oy = self.grid_origin
        c = (mx - ox) // self.cell_size
        r = (my - oy) // self.cell_size
        if 0 <= r < self.grid.height and 0 <= c < self.grid.width:
            return Cell(r, c)
        return None

    def _corridor_cell_set(self) -> set[tuple[int, int]]:
        if not self.corridor_blocks or self.grid is None:
            return set()
        cells: set[tuple[int, int]] = set()
        bs = self.block_size
        for _, br, bc in self.corridor_blocks:
            rs = br * bs
            re = min(rs + bs, self.grid.height)
            cs_start = bc * bs
            ce = min(cs_start + bs, self.grid.width)
            for r in range(rs, re):
                for c in range(cs_start, ce):
                    if self.grid.data[r, c] == 0:
                        cells.add((r, c))
        return cells


# ---------------------------------------------------------------------------
# Animation controller for A* / Dijkstra step-by-step exploration
# ---------------------------------------------------------------------------

class AnimationController:
    def __init__(
        self,
        exploration_steps: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]],
        steps_per_frame: int = 10,
        final_path: Optional[list[Cell]] = None,
    ) -> None:
        self.steps = exploration_steps
        self.steps_per_frame = steps_per_frame
        self.final_path = final_path
        self.current = 0
        self.running = True
        self.finished = False
        self.explored: set[tuple[int, int]] = set()
        self.frontier: set[tuple[int, int]] = set()

    def tick(self) -> None:
        if not self.running:
            return
        end = min(self.current + self.steps_per_frame, len(self.steps))
        for i in range(self.current, end):
            self.explored, self.frontier = self.steps[i]
        self.current = end
        if self.current >= len(self.steps):
            self.running = False
            self.finished = True
