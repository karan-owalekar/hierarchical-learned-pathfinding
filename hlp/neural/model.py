"""QuadTreeConvNet — obstacle-aware hierarchical corridor predictor.

A shared-weight CNN that processes 8×8 downsampled obstacle density maps
at every level of the quadtree hierarchy. Each level sees the grid at the
appropriate resolution: coarse at the top (global routing), detailed at
the bottom (local precision). Combined with source/goal position encoding
and level embeddings, this gives obstacle-aware, size-agnostic corridor
prediction with no embedding propagation needed.

Key components:
  downsample_block  — produce fixed-size obstacle density map from any block
  QuadTreeConvNet   — shared CNN + position encoder + classifier
  recursive_neural_inference — batched level-by-level corridor prediction
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


GRID_RES = 8


# ---------------------------------------------------------------------------
# Grid downsampling
# ---------------------------------------------------------------------------

def downsample_block(block: np.ndarray, resolution: int = GRID_RES) -> np.ndarray:
    """Produce a (resolution, resolution) obstacle density map from a block.

    For blocks >= resolution with compatible dimensions: fast reshape + mean.
    Otherwise: grid-aligned sampling (handles any size).
    """
    h, w = block.shape
    if (h >= resolution and w >= resolution
            and h % resolution == 0 and w % resolution == 0):
        rh, rw = h // resolution, w // resolution
        return (block[:rh * resolution, :rw * resolution]
                .reshape(resolution, rh, resolution, rw)
                .mean(axis=(1, 3))
                .astype(np.float32))

    result = np.zeros((resolution, resolution), dtype=np.float32)
    for r in range(resolution):
        for c in range(resolution):
            r0 = r * h // resolution
            r1 = max((r + 1) * h // resolution, r0 + 1)
            c0 = c * w // resolution
            c1 = max((c + 1) * w // resolution, c0 + 1)
            result[r, c] = block[r0:min(r1, h), c0:min(c1, w)].mean()
    return result


# ---------------------------------------------------------------------------
# BFS utilities (kept for general use, not needed by QuadTreeConvNet)
# ---------------------------------------------------------------------------

def enumerate_boundary_cells(
    grid_data: np.ndarray, r0: int, r1: int, c0: int, c1: int,
) -> list[tuple[int, int]]:
    """Enumerate free boundary cells of a rectangular region in clockwise order."""
    H, W = grid_data.shape
    cells: list[tuple[int, int]] = []
    if r0 >= r1 or c0 >= c1:
        return cells

    for c in range(c0, c1):
        if r0 < H and c < W and grid_data[r0, c] == 0:
            cells.append((r0, c))
    for r in range(r0 + 1, r1 - 1):
        cc = c1 - 1
        if r < H and cc < W and grid_data[r, cc] == 0:
            cells.append((r, cc))
    if r1 - 1 > r0:
        for c in range(c1 - 1, c0 - 1, -1):
            rr = r1 - 1
            if rr < H and c < W and grid_data[rr, c] == 0:
                cells.append((rr, c))
    if c1 - 1 > c0:
        for r in range(r1 - 2, r0, -1):
            if r < H and c0 < W and grid_data[r, c0] == 0:
                cells.append((r, c0))
    return cells


def bfs_all_distances(
    grid_data: np.ndarray, source_r: int, source_c: int,
) -> np.ndarray:
    """BFS from source to all reachable cells. Returns (H, W) float32 distance array."""
    H, W = grid_data.shape
    dist = np.full((H, W), np.inf, dtype=np.float32)
    if source_r < 0 or source_r >= H or source_c < 0 or source_c >= W:
        return dist
    if grid_data[source_r, source_c] != 0:
        return dist
    dist[source_r, source_c] = 0.0
    q: deque[tuple[int, int]] = deque([(source_r, source_c)])
    while q:
        r, c = q.popleft()
        d = dist[r, c] + 1.0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid_data[nr, nc] == 0 and dist[nr, nc] == np.inf:
                dist[nr, nc] = d
                q.append((nr, nc))
    return dist


def compute_boundary_distances(
    dist_map: np.ndarray,
    boundary_cells: list[tuple[int, int]],
    max_boundary_cells: int,
) -> np.ndarray:
    """Extract BFS distances to boundary cells, normalize, pad/truncate."""
    raw = np.full(max_boundary_cells, 0.0, dtype=np.float32)
    n = min(len(boundary_cells), max_boundary_cells)
    for i in range(n):
        r, c = boundary_cells[i]
        raw[i] = dist_map[r, c]

    finite_mask = raw[:n] < np.inf
    finite = raw[:n][finite_mask]
    if len(finite) > 0 and finite.max() > 0:
        max_d = finite.max()
        for i in range(n):
            raw[i] = 1.0 if raw[i] >= np.inf else raw[i] / max_d
    else:
        raw[:n] = np.where(raw[:n] >= np.inf, 1.0, 0.0)
    return raw


# ---------------------------------------------------------------------------
# QuadTreeConvNet
# ---------------------------------------------------------------------------

class QuadTreeConvNet(nn.Module):
    """Obstacle-aware hierarchical corridor predictor.

    A shared-weight CNN processes an 8×8 downsampled obstacle density map
    of the current quadtree block. Combined with normalized source/goal
    positions and a level embedding, it predicts which of the 4 child
    quadrants the shortest path passes through.

    Each node's prediction is self-contained — no embedding propagation
    from parent to child. The hierarchy itself provides context: parent
    predictions prune irrelevant blocks before children are processed.
    """

    def __init__(
        self,
        d: int = 64,
        max_levels: int = 12,
        grid_resolution: int = GRID_RES,
    ) -> None:
        super().__init__()
        self.d = d
        self.grid_resolution = grid_resolution

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, d, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.pos_encoder = nn.Sequential(
            nn.Linear(4, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
        )

        self.level_emb = nn.Embedding(max_levels, d)

        self.head = nn.Sequential(
            nn.Linear(3 * d, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
        )

    def forward(
        self,
        grid_8x8: torch.Tensor,
        positions: torch.Tensor,
        level_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            grid_8x8:  (B, 1, R, R) obstacle density map
            positions: (B, 4) [src_r, src_c, goal_r, goal_c] normalized to block
            level_idx: int or (B,) tensor

        Returns:
            (B, 4) activation probabilities (sigmoid)
        """
        B = grid_8x8.shape[0]

        grid_feat = self.grid_encoder(grid_8x8).view(B, -1)
        pos_feat = self.pos_encoder(positions)

        if isinstance(level_idx, int):
            level_t = torch.full((B,), level_idx, dtype=torch.long,
                                 device=grid_8x8.device)
        else:
            level_t = level_idx
        level_feat = self.level_emb(level_t)

        combined = torch.cat([grid_feat, pos_feat, level_feat], dim=-1)
        return torch.sigmoid(self.head(combined))


# ---------------------------------------------------------------------------
# Batched recursive inference
# ---------------------------------------------------------------------------

def recursive_neural_inference(
    model: QuadTreeConvNet,
    grid_data: np.ndarray,
    grid_h: int,
    grid_w: int,
    source: tuple[int, int],
    goal: tuple[int, int],
    *,
    stop_at_size: int = 1,
    activation_threshold: float = 0.5,
) -> set[tuple[int, int]]:
    """Batched level-by-level corridor prediction.

    At each level, all active blocks are downsampled to 8×8, processed in
    a single batched forward pass, and children of active quadrants are
    queued for the next level. No BFS or embedding propagation anywhere.
    """
    model.eval()
    device = next(model.parameters()).device
    grid_res = model.grid_resolution

    num_levels = max(int(math.ceil(math.log2(max(grid_h, grid_w, 2)))), 1)
    padded_size = 1 << num_levels
    padded = np.ones((padded_size, padded_size), dtype=np.uint8)
    padded[:grid_h, :grid_w] = grid_data

    effective_stop = max(stop_at_size, 2)

    work_items: list[tuple[int, int, int, int]] = [
        (0, padded_size, 0, padded_size),
    ]
    active_cells: set[tuple[int, int]] = set()

    with torch.no_grad():
        for level in range(num_levels + 1):
            if not work_items:
                break

            recurse: list[tuple[int, int, int, int]] = []
            for r0, r1, c0, c1 in work_items:
                if (r1 - r0) <= effective_stop or (c1 - c0) <= effective_stop:
                    for r in range(r0, min(r1, grid_h)):
                        for c in range(c0, min(c1, grid_w)):
                            if padded[r, c] == 0:
                                active_cells.add((r, c))
                else:
                    recurse.append((r0, r1, c0, c1))

            if not recurse:
                break

            grids: list[np.ndarray] = []
            positions: list[list[float]] = []
            for r0, r1, c0, c1 in recurse:
                grids.append(downsample_block(padded[r0:r1, c0:c1], grid_res))
                h, w = r1 - r0, c1 - c0
                positions.append([
                    (source[0] - r0) / h,
                    (source[1] - c0) / w,
                    (goal[0] - r0) / h,
                    (goal[1] - c0) / w,
                ])

            grid_batch = torch.tensor(
                np.array(grids), dtype=torch.float32,
            ).unsqueeze(1).to(device)
            pos_batch = torch.tensor(
                np.array(positions, dtype=np.float32),
            ).to(device)

            act = model(grid_batch, pos_batch, level)

            next_items: list[tuple[int, int, int, int]] = []
            for wi, (r0, r1, c0, c1) in enumerate(recurse):
                mr, mc = (r0 + r1) // 2, (c0 + c1) // 2
                quads = [(r0, mr, c0, mc), (r0, mr, mc, c1),
                         (mr, r1, c0, mc), (mr, r1, mc, c1)]
                for qi, (qr0, qr1, qc0, qc1) in enumerate(quads):
                    if act[wi, qi].item() > activation_threshold:
                        next_items.append((qr0, qr1, qc0, qc1))

            work_items = next_items

    active_cells.add(source)
    active_cells.add(goal)
    return active_cells
