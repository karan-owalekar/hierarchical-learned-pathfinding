"""Grid rendering, cell interaction, and animation controllers."""

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
    QUAD_BORDER_REJ,
    QUAD_CANDIDATE,
    QUAD_REJECTED,
    corridor_level_border,
    corridor_level_color,
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

        # Animation (A*/Dijkstra expansion or corridor level-by-level)
        self.animation: Optional[AnimationController] = None
        self.corridor_anim: Optional[CorridorAnimationController] = None

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
        self.corridor_anim = None

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

        if self.corridor_anim:
            if self.corridor_anim.running:
                self.corridor_anim.tick()
            if self.corridor_anim.finished and self.corridor_anim.final_path:
                self.path = self.corridor_anim.final_path
                self.corridor_anim.final_path = None

        ox, oy = self.grid_origin
        cs = self.cell_size
        inner = cs - CELL_GAP if cs > 3 else cs
        radius = CELL_RADIUS if cs > 4 else 0

        path_set = {(c.row, c.col) for c in self.path} if self.path else set()
        corridor_cells = self._corridor_cell_set()

        # Pre-compute corridor animation overlay: maps (r,c) -> color
        corr_cell_colors: dict[tuple[int, int], tuple[int, int, int]] = {}
        if self.corridor_anim and self.grid:
            corr_cell_colors = self.corridor_anim.cell_color_map(self.grid)

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
                elif (r, c) in corr_cell_colors:
                    color = corr_cell_colors[(r, c)]
                elif (r, c) in corridor_cells:
                    color = CELL_CORRIDOR
                elif self.grid.data[r, c] == 1:
                    color = CELL_OBSTACLE
                else:
                    color = CELL_EMPTY

                pygame.draw.rect(surface, color, rect, border_radius=radius)

        if self.corridor_anim:
            self._draw_corridor_anim_borders(surface)

    def _draw_block_rect(
        self,
        surface: pygame.Surface,
        r0: int, r1: int, c0: int, c1: int,
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        if self.grid is None:
            return
        ox, oy = self.grid_origin
        cs = self.cell_size
        r1 = min(r1, self.grid.height)
        c1 = min(c1, self.grid.width)
        px = ox + c0 * cs
        py = oy + r0 * cs
        pw = (c1 - c0) * cs
        ph = (r1 - r0) * cs
        if pw > 0 and ph > 0:
            rect = pygame.Rect(px, py, pw, ph)
            pygame.draw.rect(surface, color, rect, width=width)

    def _draw_corridor_anim_borders(self, surface: pygame.Surface) -> None:
        if not self.corridor_anim or self.grid is None:
            return
        cs = self.cell_size
        bw = max(1, cs // 5)

        level_blocks, rej_blocks, max_levels = self.corridor_anim.border_info()

        for lvl, blocks in level_blocks:
            clr = corridor_level_border(lvl, max_levels)
            for r0, r1, c0, c1 in blocks:
                self._draw_block_rect(surface, r0, r1, c0, c1, clr, bw)
        for r0, r1, c0, c1 in rej_blocks:
            self._draw_block_rect(surface, r0, r1, c0, c1, QUAD_BORDER_REJ, bw)

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


# ---------------------------------------------------------------------------
# Corridor animation controller for Neural / Hybrid level-by-level expansion
# ---------------------------------------------------------------------------

class CorridorAnimationController:
    """Animates the quadtree corridor prediction step by step.

    Each level has three phases:
      Phase 0 ("candidates"): All children of active parents appear in
          the same purple as A* explored cells — the network is evaluating.
      Phase 1 ("decide"):     Rejected quadrants turn red; accepted stay
          purple — the network has decided.
      Phase 2 ("cleanup"):    Red quadrants fade away; accepted blocks
          shift to their level-specific color.

    Accepted blocks from earlier levels remain visible with gradually
    shifting colors so you can see the refinement deepening.
    """

    CANDIDATE_FRAMES = 15
    DECIDE_FRAMES = 20
    CLEANUP_FRAMES = 10

    def __init__(
        self,
        history: list[dict],
        final_path: Optional[list[Cell]] = None,
    ) -> None:
        self.history = history
        self.final_path = final_path
        self.max_levels = max(len(history), 1)
        self.current_level = 0
        self.phase = 0
        self.frame_count = 0
        self.running = bool(history)
        self.finished = False

        # (level, blocks) pairs for previously settled levels
        self._settled: list[tuple[int, list[tuple[int, int, int, int]]]] = []

    def tick(self) -> None:
        if not self.running:
            return
        self.frame_count += 1
        limits = [self.CANDIDATE_FRAMES, self.DECIDE_FRAMES, self.CLEANUP_FRAMES]
        if self.frame_count >= limits[self.phase]:
            self.frame_count = 0
            if self.phase < 2:
                self.phase += 1
            else:
                entry = self.history[self.current_level]
                self._settled.append(
                    (self.current_level, entry["accepted"])
                )
                self.phase = 0
                self.current_level += 1
                if self.current_level >= len(self.history):
                    self.running = False
                    self.finished = True

    def cell_color_map(
        self, grid: Grid,
    ) -> dict[tuple[int, int], tuple[int, int, int]]:
        """Map (r, c) → fill color for all corridor-animation cells."""
        result: dict[tuple[int, int], tuple[int, int, int]] = {}

        for lvl, blocks in self._settled:
            clr = corridor_level_color(lvl, self.max_levels)
            for r0, r1, c0, c1 in blocks:
                for r in range(r0, min(r1, grid.height)):
                    for c in range(c0, min(c1, grid.width)):
                        if grid.data[r, c] == 0:
                            result[(r, c)] = clr

        if self.current_level >= len(self.history):
            return result

        entry = self.history[self.current_level]

        if self.phase == 0:
            for r0, r1, c0, c1 in entry["accepted"]:
                for r in range(r0, min(r1, grid.height)):
                    for c in range(c0, min(c1, grid.width)):
                        if grid.data[r, c] == 0:
                            result[(r, c)] = QUAD_CANDIDATE
        elif self.phase == 1:
            for r0, r1, c0, c1 in entry["accepted"]:
                for r in range(r0, min(r1, grid.height)):
                    for c in range(c0, min(c1, grid.width)):
                        if grid.data[r, c] == 0:
                            result[(r, c)] = QUAD_CANDIDATE
        else:
            clr = corridor_level_color(self.current_level, self.max_levels)
            for r0, r1, c0, c1 in entry["accepted"]:
                for r in range(r0, min(r1, grid.height)):
                    for c in range(c0, min(c1, grid.width)):
                        if grid.data[r, c] == 0:
                            result[(r, c)] = clr

        return result

    def border_info(
        self,
    ) -> tuple[
        list[tuple[int, list[tuple[int, int, int, int]]]],
        list[tuple[int, int, int, int]],
        int,
    ]:
        """Return (level_blocks, rejected_blocks, max_levels) for borders."""
        level_blocks = list(self._settled)
        rej: list[tuple[int, int, int, int]] = []

        if self.current_level < len(self.history):
            entry = self.history[self.current_level]
            if self.phase <= 1:
                level_blocks.append(
                    (self.current_level, entry["accepted"])
                )
            else:
                level_blocks.append(
                    (self.current_level, entry["accepted"])
                )
            if self.phase == 1:
                rej = entry["rejected"]

        return level_blocks, rej, self.max_levels
