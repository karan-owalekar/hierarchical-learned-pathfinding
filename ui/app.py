"""Main Pygame application: event loop, layout, toolbar, dispatching."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pygame
import torch

from hlp.config import Config
from hlp.grid import Cell, Grid
from hlp.neural.model import QuadTreeConvNet, recursive_neural_inference
from hlp.pipeline import (
    PathResult,
    run_hybrid,
    run_matrix_only,
    run_neural_only,
)
from baselines.astar import astar, astar_generator
from baselines.dijkstra import dijkstra, dijkstra_generator
from ui.components import Button, Dropdown, InfoOverlay, StatusBar, ToggleButton
from ui.grid_view import AnimationController, CorridorAnimationController, GridView
from ui.map_generators import (
    MAP_TYPE_GENERATORS,
    MAP_TYPE_KEYS,
    MAP_TYPE_NAMES,
    generate_dfs_maze,
    generate_random,
    generate_recursive_division,
    generate_rooms,
    generate_spiral,
)
from ui.theme import (
    BG_DARK,
    BG_TOOLBAR,
    ELEMENT_SPACING,
    PADDING,
    STATUS_HEIGHT,
    TEXT_PRIMARY,
    TOOLBAR_HEIGHT,
    font,
    init_fonts,
)

GRID_SIZES = [32, 64, 128, 256]
METHODS = ["Matrix Only", "Neural Only", "Hybrid", "A*", "Dijkstra"]
MAP_TYPES = MAP_TYPE_NAMES

NEURAL_METHOD_INDICES = {1, 2}  # Neural Only, Hybrid


class App:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.models: dict[str, QuadTreeConvNet] = {}
        self.model: Optional[QuadTreeConvNet] = None
        self.model_available = False

        pygame.init()
        init_fonts()

        info = pygame.display.Info()
        w = int(info.current_w * 0.95)
        h = int(info.current_h * 0.95)
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption("Hierarchical Learned Pathfinding")
        self.clock = pygame.time.Clock()

        # UI components
        self.grid_size_dd = Dropdown(
            [str(s) for s in GRID_SIZES], selected=1, width=100, label="Size",
            on_change=self._on_grid_size_change,
        )
        self.map_dd = Dropdown(
            MAP_TYPES, selected=0, width=220, label="Map",
            on_change=self._on_generate_map,
        )

        self._load_models()

        disabled = NEURAL_METHOD_INDICES if not self.model_available else set()
        default_method = 2 if self.model_available else 3  # Hybrid or A*
        self.method_dd = Dropdown(
            METHODS, selected=default_method, width=170, label="Method",
            disabled_indices=disabled,
        )
        self.gen_btn = Button("Generate", width=110, on_click=self._on_regenerate)
        self.vis_toggle = ToggleButton(
            text_on="Visualize", text_off="Visualize", active=True, width=120,
        )
        self.find_btn = Button("Find Path", accent=True, width=120, on_click=self._on_find_path)
        self.clear_path_btn = Button("Clear Path", width=120, on_click=self._on_clear_path)
        self.clear_grid_btn = Button("Clear Grid", width=120, on_click=self._on_clear_grid)
        self.info_btn = Button("?", square=True, on_click=self._on_info)

        self._map_seed = 0

        self.status = StatusBar()
        self.overlay = InfoOverlay()
        self.grid_view = GridView()
        self.grid_view.block_size = config.block.block_size

        self._dropdowns = [self.grid_size_dd, self.method_dd, self.map_dd]
        self._buttons = [self.gen_btn, self.vis_toggle, self.find_btn, self.clear_path_btn, self.clear_grid_btn, self.info_btn]

        # Initialize with a default grid
        self._create_empty_grid(GRID_SIZES[1])

        self.running = True

    # ----- main loop -----

    def run(self) -> None:
        while self.running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(60)
        pygame.quit()

    # ----- event handling -----

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

            # Overlay first
            if self.overlay.handle_event(event):
                continue

            # Dropdowns (consume clicks when open)
            any_dd_open = any(dd.consumes_click for dd in self._dropdowns)
            if any_dd_open:
                for dd in self._dropdowns:
                    dd.handle_event(event)
                continue

            for dd in self._dropdowns:
                if dd.handle_event(event):
                    break
            else:
                for btn in self._buttons:
                    if btn.handle_event(event):
                        break
                else:
                    msg = self.grid_view.handle_event(event)
                    if msg:
                        self.status.set_message(msg)

    # ----- update -----

    def _update(self) -> None:
        mouse = pygame.mouse.get_pos()
        for dd in self._dropdowns:
            dd.update(mouse)
        for btn in self._buttons:
            btn.update(mouse)

    # ----- drawing -----

    def _draw(self) -> None:
        w, h = self.screen.get_size()
        self.screen.fill(BG_DARK)

        # Toolbar background
        toolbar_rect = pygame.Rect(0, 0, w, TOOLBAR_HEIGHT)
        pygame.draw.rect(self.screen, BG_TOOLBAR, toolbar_rect)

        # Layout toolbar elements — push down to leave room for labels above
        x = PADDING
        y = TOOLBAR_HEIGHT - 34 - 8
        x = self.grid_size_dd.layout(x, y, 100) + ELEMENT_SPACING
        x = self.method_dd.layout(x, y, 170) + ELEMENT_SPACING
        x = self.map_dd.layout(x, y, 220) + ELEMENT_SPACING
        x = self.gen_btn.layout(x, y) + ELEMENT_SPACING
        x = self.vis_toggle.layout(x, y) + ELEMENT_SPACING + 4
        x = self.find_btn.layout(x, y) + ELEMENT_SPACING
        x = self.clear_path_btn.layout(x, y) + ELEMENT_SPACING
        x = self.clear_grid_btn.layout(x, y) + ELEMENT_SPACING
        self.info_btn.layout(x, y)

        # Title (right side, if space)
        title_text = "Hierarchical Learned Pathfinding"
        title_surf = font("title").render(title_text, True, TEXT_PRIMARY)
        title_x = w - title_surf.get_width() - PADDING
        if title_x > self.info_btn.rect.right + 20:
            self.screen.blit(title_surf, (title_x, (TOOLBAR_HEIGHT - title_surf.get_height()) // 2))

        # Grid view
        grid_area = pygame.Rect(0, TOOLBAR_HEIGHT, w, h - TOOLBAR_HEIGHT - STATUS_HEIGHT)
        self.grid_view.layout(grid_area)
        self.grid_view.draw(self.screen)

        # Status bar
        self.status.layout(0, h - STATUS_HEIGHT, w, STATUS_HEIGHT)
        self.status.draw(self.screen)

        # Draw components on top (order matters for dropdown z-order)
        for btn in self._buttons:
            btn.draw(self.screen)
        for dd in self._dropdowns:
            dd.draw(self.screen)

        self.overlay.draw(self.screen)
        pygame.display.flip()

    # ----- callbacks -----

    def _on_grid_size_change(self, idx: int, val: str) -> None:
        self._generate_current_map()

    def _on_generate_map(self, idx: int, val: str) -> None:
        self._map_seed = 0
        self._update_active_model()
        self._generate_current_map()

    def _on_regenerate(self) -> None:
        self._map_seed += 1
        self._generate_current_map()

    def _generate_current_map(self) -> None:
        size = GRID_SIZES[self.grid_size_dd.selected]
        idx = self.map_dd.selected
        key = MAP_TYPE_KEYS[idx]
        name = MAP_TYPE_NAMES[idx]
        gen = MAP_TYPE_GENERATORS[key]
        grid, start, goal = gen(size, size, seed=self._map_seed)
        self.grid_view.set_grid(grid, start, goal)
        self.status.set_message(f"Generated {name} map ({size}×{size})")

    def _on_find_path(self) -> None:
        if self.grid_view.grid is None or self.grid_view.start is None or self.grid_view.goal is None:
            self.status.set_message("Set both start (S) and goal (G) first")
            return

        self.grid_view.clear_overlays()
        method = self.method_dd.value
        grid = self.grid_view.grid
        source = self.grid_view.start
        goal = self.grid_view.goal

        self.status.set_message(f"Running {method}...")
        pygame.display.flip()

        visualize = self.vis_toggle.active

        if method == "A*":
            self._run_astar(grid, source, goal, visualize)
        elif method == "Dijkstra":
            self._run_dijkstra(grid, source, goal, visualize)
        elif method == "Matrix Only":
            result = run_matrix_only(grid, source, goal, self.config)
            self._apply_hlp_result(result)
        elif method == "Neural Only" and self.model is not None:
            self._run_neural_only(grid, source, goal, visualize)
        elif method == "Hybrid" and self.model is not None:
            self._run_hybrid(grid, source, goal, visualize)
        else:
            self.status.set_message("Model not loaded — select a different method")

    def _on_clear_path(self) -> None:
        self.grid_view.clear_overlays()
        self.status.set_message("Path cleared")

    def _on_clear_grid(self) -> None:
        if self.grid_view.grid:
            size = self.grid_view.grid.height
            self._create_empty_grid(size)
            self.status.set_message("Grid cleared")

    def _on_info(self) -> None:
        self.overlay.toggle()

    # ----- helpers -----

    def _create_empty_grid(self, size: int) -> None:
        grid = Grid(np.zeros((size, size), dtype=np.uint8))
        self.grid_view.set_grid(grid)

    def _load_models(self) -> None:
        nc = self.config.neural
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_dir = Path(nc.checkpoint_path).parent

        def _try_load(path: Path) -> Optional[QuadTreeConvNet]:
            if not path.exists():
                return None
            try:
                m = QuadTreeConvNet(
                    d=nc.d, max_levels=nc.max_levels,
                    grid_resolution=nc.grid_resolution,
                ).to(device)
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                m.eval()
                return m
            except Exception:
                return None

        for key in MAP_TYPE_KEYS:
            m = _try_load(ckpt_dir / f"best_{key}.pt")
            if m is not None:
                self.models[key] = m

        fallback = _try_load(Path(nc.checkpoint_path))
        if fallback is not None:
            self.models["_default"] = fallback

        self._update_active_model()

    def _update_active_model(self) -> None:
        key = MAP_TYPE_KEYS[self.map_dd.selected]
        self.model = self.models.get(key) or self.models.get("_default")
        self.model_available = self.model is not None

    def _run_astar(self, grid: Grid, source: Cell, goal: Cell, visualize: bool) -> None:
        gen = astar_generator(grid, source, goal)
        steps: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
        result = None
        try:
            while True:
                step = next(gen)
                steps.append((set(step.explored), set(step.frontier)))
        except StopIteration as e:
            result = e.value

        if result is None:
            self.status.set_message("A*: no path found")
            return

        if visualize:
            speed = max(1, len(steps) // 200)
            self.grid_view.animation = AnimationController(
                steps, steps_per_frame=speed, final_path=result.path,
            )
        else:
            self.grid_view.path = result.path

        if result.path:
            self.status.set_message(
                f"A*: cost={result.cost:.0f} | len={len(result.path)} | "
                f"explored={result.nodes_explored} | {result.computation_time_ms:.1f}ms"
            )
        else:
            self.status.set_message("A*: no path found")

    def _run_dijkstra(self, grid: Grid, source: Cell, goal: Cell, visualize: bool) -> None:
        gen = dijkstra_generator(grid, source, goal)
        steps: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
        result = None
        try:
            while True:
                step = next(gen)
                steps.append((set(step.explored), set(step.frontier)))
        except StopIteration as e:
            result = e.value

        if result is None:
            self.status.set_message("Dijkstra: no path found")
            return

        if visualize:
            speed = max(1, len(steps) // 200)
            self.grid_view.animation = AnimationController(
                steps, steps_per_frame=speed, final_path=result.path,
            )
        else:
            self.grid_view.path = result.path

        if result.path:
            self.status.set_message(
                f"Dijkstra: cost={result.cost:.0f} | len={len(result.path)} | "
                f"explored={result.nodes_explored} | {result.computation_time_ms:.1f}ms"
            )
        else:
            self.status.set_message("Dijkstra: no path found")

    def _run_neural_only(self, grid: Grid, source: Cell, goal: Cell, visualize: bool) -> None:
        result = run_neural_only(grid, source, goal, self.model, self.config)
        if visualize and self.model is not None:
            threshold = self.config.inference.activation_threshold
            _, history = recursive_neural_inference(
                self.model, grid.data, grid.height, grid.width,
                (source.row, source.col), (goal.row, goal.col),
                stop_at_size=1,
                activation_threshold=threshold,
                record_history=True,
            )
            self.grid_view.corridor_anim = CorridorAnimationController(
                history, final_path=result.path,
            )
        else:
            self.grid_view.path = result.path
        self._set_hlp_status(result)

    def _run_hybrid(self, grid: Grid, source: Cell, goal: Cell, visualize: bool) -> None:
        result = run_hybrid(grid, source, goal, self.model, self.config)
        if visualize and self.model is not None:
            threshold = self.config.inference.activation_threshold
            _, history = recursive_neural_inference(
                self.model, grid.data, grid.height, grid.width,
                (source.row, source.col), (goal.row, goal.col),
                stop_at_size=1,
                activation_threshold=threshold,
                record_history=True,
            )
            self.grid_view.corridor_anim = CorridorAnimationController(
                history, final_path=result.path,
            )
        else:
            self.grid_view.path = result.path
        self._set_hlp_status(result)

    def _apply_hlp_result(self, result: PathResult) -> None:
        self.grid_view.path = result.path
        self._set_hlp_status(result)

    def _set_hlp_status(self, result: PathResult) -> None:
        if result.path:
            self.status.set_message(
                f"{result.mode}: cost={result.cost:.0f} | len={len(result.path)} | "
                f"corridor={result.corridor_size}/{result.total_blocks} | "
                f"{result.computation_time_ms:.1f}ms"
            )
        elif result.mode == "neural_only":
            self.status.set_message(
                "neural_only: no path found — try Hybrid (uses BFS fallback)"
            )
        else:
            self.status.set_message(f"{result.mode}: no path found")
