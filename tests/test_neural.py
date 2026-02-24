"""Tests for the QuadTreeConvNet corridor predictor: downsample, model shapes,
gradient flow, BFS utilities, label extraction, loss function, and inference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hlp.grid import Cell, Grid, bfs_shortest_path
from hlp.neural.losses import CorridorLoss
from hlp.neural.model import (
    GRID_RES,
    QuadTreeConvNet,
    bfs_all_distances,
    compute_boundary_distances,
    downsample_block,
    enumerate_boundary_cells,
    recursive_neural_inference,
)
from hlp.neural.dataset import extract_flat_labels


# -----------------------------------------------------------------------
# downsample_block
# -----------------------------------------------------------------------

class TestDownsampleBlock:
    def test_identity_8x8(self):
        block = np.random.rand(8, 8).astype(np.float32)
        result = downsample_block(block, 8)
        np.testing.assert_allclose(result, block, atol=1e-6)

    def test_downsample_16x16(self):
        block = np.zeros((16, 16), dtype=np.uint8)
        result = downsample_block(block, 8)
        assert result.shape == (8, 8)
        np.testing.assert_allclose(result, 0.0)

    def test_downsample_preserves_density(self):
        block = np.ones((16, 16), dtype=np.uint8)
        block[:8, :] = 0  # top half free, bottom half blocked
        result = downsample_block(block, 8)
        assert result.shape == (8, 8)
        assert result[0, 0] == 0.0  # top rows are free
        assert result[7, 0] == 1.0  # bottom rows are blocked

    def test_upsample_4x4(self):
        block = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = downsample_block(block, 8)
        assert result.shape == (8, 8)
        assert result[0, 0] == 0.0  # top-left is 0
        assert result[0, 7] == 1.0  # top-right is 1

    def test_large_block_256(self):
        block = np.random.randint(0, 2, (256, 256), dtype=np.uint8)
        result = downsample_block(block, 8)
        assert result.shape == (8, 8)
        assert result.dtype == np.float32


# -----------------------------------------------------------------------
# BFS utilities (kept from previous architecture)
# -----------------------------------------------------------------------

class TestBFSUtilities:
    def test_bfs_all_distances_open_grid(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        dist = bfs_all_distances(grid, 0, 0)
        assert dist[0, 0] == 0.0
        assert dist[0, 7] == 7.0
        assert dist[7, 7] == 14.0

    def test_bfs_all_distances_blocked(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        grid[1, :] = 1
        grid[1, 3] = 0
        dist = bfs_all_distances(grid, 0, 0)
        assert dist[0, 0] == 0.0
        assert np.isinf(dist[2, 0]) or dist[2, 0] > 4

    def test_bfs_all_distances_source_blocked(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        grid[0, 0] = 1
        dist = bfs_all_distances(grid, 0, 0)
        assert np.all(np.isinf(dist))

    def test_enumerate_boundary_cells_open(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        cells = enumerate_boundary_cells(grid, 0, 8, 0, 8)
        assert len(cells) > 0
        for r, c in cells:
            assert r == 0 or r == 7 or c == 0 or c == 7

    def test_enumerate_boundary_cells_blocked(self):
        grid = np.ones((4, 4), dtype=np.uint8)
        cells = enumerate_boundary_cells(grid, 0, 4, 0, 4)
        assert len(cells) == 0

    def test_compute_boundary_distances_padding(self):
        dist = np.array([[0, 1, 2, 3],
                         [1, 2, 3, 4],
                         [2, 3, 4, 5],
                         [3, 4, 5, 6]], dtype=np.float32)
        cells = [(0, 0), (0, 1), (0, 2)]
        result = compute_boundary_distances(dist, cells, max_boundary_cells=8)
        assert result.shape == (8,)
        assert result[3] == 0.0
        assert result[0] == 0.0


# -----------------------------------------------------------------------
# QuadTreeConvNet
# -----------------------------------------------------------------------

class TestQuadTreeConvNet:
    @pytest.fixture
    def model(self):
        return QuadTreeConvNet(d=32, max_levels=10, grid_resolution=8)

    def test_output_shapes(self, model):
        B = 4
        grid = torch.randn(B, 1, 8, 8)
        pos = torch.randn(B, 4)
        act = model(grid, pos, 0)
        assert act.shape == (B, 4)

    def test_activation_range(self, model):
        grid = torch.randn(2, 1, 8, 8)
        pos = torch.randn(2, 4)
        with torch.no_grad():
            act = model(grid, pos, 3)
        assert act.min() >= 0.0
        assert act.max() <= 1.0

    def test_gradient_flow(self, model):
        grid = torch.randn(1, 1, 8, 8, requires_grad=True)
        pos = torch.randn(1, 4, requires_grad=True)
        act = model(grid, pos, 0)
        loss = act.sum()
        loss.backward()
        assert grid.grad is not None
        assert pos.grad is not None
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_level_embedding_different(self, model):
        grid = torch.randn(1, 1, 8, 8)
        pos = torch.randn(1, 4)
        with torch.no_grad():
            act0 = model(grid, pos, 0)
            act5 = model(grid, pos, 5)
        assert not torch.allclose(act0, act5)

    def test_batch_level_tensor(self, model):
        B = 3
        grid = torch.randn(B, 1, 8, 8)
        pos = torch.randn(B, 4)
        levels = torch.tensor([0, 2, 4], dtype=torch.long)
        act = model(grid, pos, levels)
        assert act.shape == (B, 4)

    def test_param_count_reasonable(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 5_000, f"Model has {total} params — too small"
        assert total < 500_000, f"Model has {total} params — too large for d=32"


# -----------------------------------------------------------------------
# Flat label extraction
# -----------------------------------------------------------------------

class TestFlatLabels:
    def test_basic_horizontal_path(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        path = [Cell(0, c) for c in range(8)]
        examples = extract_flat_labels(grid, path, Cell(0, 0), Cell(0, 7), 8)
        assert len(examples) > 0
        root = examples[0]
        assert root["grid_8x8"].shape == (8, 8)
        assert root["positions"].shape == (4,)
        assert root["activation"].shape == (4,)

    def test_vertical_path_activations(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        path = [Cell(r, 0) for r in range(8)]
        examples = extract_flat_labels(grid, path, Cell(0, 0), Cell(7, 0), 8)
        root = examples[0]
        assert root["activation"][0] == 1.0
        assert root["activation"][2] == 1.0

    def test_position_values(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        path = [Cell(0, c) for c in range(8)]
        examples = extract_flat_labels(grid, path, Cell(0, 0), Cell(0, 7), 8)
        root = examples[0]
        pos = root["positions"]
        assert abs(pos[0] - 0.0) < 1e-6  # src_r / h = 0/8
        assert abs(pos[1] - 0.0) < 1e-6  # src_c / w = 0/8
        assert abs(pos[2] - 0.0) < 1e-6  # goal_r / h = 0/8
        assert abs(pos[3] - 7 / 8) < 1e-6  # goal_c / w = 7/8

    def test_grid_8x8_is_obstacle_density(self):
        grid = np.zeros((16, 16), dtype=np.uint8)
        grid[8:, :] = 1  # bottom half blocked
        path = [Cell(0, c) for c in range(16)]
        examples = extract_flat_labels(grid, path, Cell(0, 0), Cell(0, 15), 16)
        root = examples[0]
        g = root["grid_8x8"]
        assert g[0, 0] == 0.0  # top rows free
        assert g[7, 0] == 1.0  # bottom rows blocked

    def test_recursive_children(self):
        grid = np.zeros((8, 8), dtype=np.uint8)
        path = [Cell(r, r) for r in range(8)]  # diagonal
        examples = extract_flat_labels(grid, path, Cell(0, 0), Cell(7, 7), 8)
        assert len(examples) > 1  # root + children


# -----------------------------------------------------------------------
# CorridorLoss
# -----------------------------------------------------------------------

class TestCorridorLoss:
    def test_basic(self):
        loss_fn = CorridorLoss()
        act = torch.sigmoid(torch.randn(2, 4, requires_grad=True))
        act_label = (torch.rand(2, 4) > 0.5).float()

        loss, metrics = loss_fn(act, act_label)
        assert loss.item() > 0
        assert loss.requires_grad
        assert "loss" in metrics

    def test_perfect_prediction(self):
        loss_fn = CorridorLoss()
        act_label = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        act = act_label.clone()
        loss, _ = loss_fn(act, act_label)
        assert loss.item() < 1e-5

    def test_pos_weight_effect(self):
        act = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
        act_label = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        loss_low, _ = CorridorLoss(pos_weight=1.0)(act, act_label)
        loss_high, _ = CorridorLoss(pos_weight=10.0)(act, act_label)
        assert loss_high.item() > loss_low.item()


# -----------------------------------------------------------------------
# Recursive inference
# -----------------------------------------------------------------------

class TestRecursiveInference:
    def test_basic_open_grid(self):
        model = QuadTreeConvNet(d=32, max_levels=6, grid_resolution=8)
        grid = np.zeros((8, 8), dtype=np.uint8)
        active = recursive_neural_inference(
            model, grid, 8, 8, (0, 0), (7, 7), stop_at_size=1,
        )
        assert len(active) > 0
        assert isinstance(active, set)
        for cell in active:
            assert len(cell) == 2

    def test_stop_at_block_size(self):
        model = QuadTreeConvNet(d=32, max_levels=8, grid_resolution=8)
        grid = np.zeros((32, 32), dtype=np.uint8)
        active = recursive_neural_inference(
            model, grid, 32, 32, (0, 0), (31, 31), stop_at_size=8,
        )
        assert len(active) > 0

    def test_source_and_goal_always_included(self):
        model = QuadTreeConvNet(d=32, max_levels=6, grid_resolution=8)
        grid = np.zeros((16, 16), dtype=np.uint8)
        active = recursive_neural_inference(
            model, grid, 16, 16, (0, 0), (15, 15),
        )
        assert (0, 0) in active
        assert (15, 15) in active

    def test_large_grid(self):
        model = QuadTreeConvNet(d=32, max_levels=10, grid_resolution=8)
        grid = np.zeros((256, 256), dtype=np.uint8)
        active = recursive_neural_inference(
            model, grid, 256, 256, (0, 0), (255, 255),
        )
        assert (0, 0) in active
        assert (255, 255) in active
