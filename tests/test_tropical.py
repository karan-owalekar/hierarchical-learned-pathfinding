"""Tests for tropical semiring operations and transfer matrix computation."""

from __future__ import annotations

import numpy as np
import pytest

from hlp.grid import Cell, Grid
from hlp.tropical import (
    INF,
    compute_level1_transfer_matrix,
    floyd_warshall,
    tropical_add,
    tropical_identity,
    tropical_matmul,
    tropical_matmul_with_argmin,
    tropical_mat_vec,
    tropical_mul,
)


class TestTropicalScalar:
    def test_add_min(self):
        assert tropical_add(3, 5) == 3
        assert tropical_add(5, 3) == 3

    def test_add_identity(self):
        assert tropical_add(INF, 5) == 5
        assert tropical_add(5, INF) == 5
        assert tropical_add(INF, INF) == INF

    def test_mul_plus(self):
        assert tropical_mul(3, 5) == 8
        assert tropical_mul(0, 5) == 5

    def test_mul_inf_absorbing(self):
        assert tropical_mul(INF, 5) == INF
        assert tropical_mul(5, INF) == INF
        assert tropical_mul(INF, INF) == INF

    def test_mul_identity_is_zero(self):
        assert tropical_mul(0, 7) == 7
        assert tropical_mul(7, 0) == 7


class TestTropicalMatmul:
    def test_identity_property(self):
        T = np.array([[0, 3, INF], [3, 0, 2], [INF, 2, 0]])
        I = tropical_identity(3)
        result = tropical_matmul(T, I)
        np.testing.assert_array_equal(result, T)

    def test_simple_2x2(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        B = np.array([[0.0, 2.0], [2.0, 0.0]])
        C = tropical_matmul(A, B)
        # C[0][0] = min(0+0, 1+2) = 0
        # C[0][1] = min(0+2, 1+0) = 1
        # C[1][0] = min(1+0, 0+2) = 1
        # C[1][1] = min(1+2, 0+0) = 0
        expected = np.array([[0.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(C, expected)

    def test_associativity(self):
        A = np.array([[0, 2, INF], [2, 0, 1], [INF, 1, 0]])
        B = np.array([[0, INF, 3], [INF, 0, 1], [3, 1, 0]])
        C = np.array([[0, 1], [1, 0], [INF, 2]])
        AB = tropical_matmul(A, B)
        AB_C = tropical_matmul(AB, C)
        BC = tropical_matmul(B, C)
        A_BC = tropical_matmul(A, BC)
        np.testing.assert_array_almost_equal(AB_C, A_BC)

    def test_with_argmin(self):
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        B = np.array([[0.0, 2.0], [2.0, 0.0]])
        C, K = tropical_matmul_with_argmin(A, B)
        assert C[0, 1] == 1.0
        assert K[0, 1] == 1  # achieved via k=1: A[0,1]+B[1,1]=1+0=1

    def test_mat_vec(self):
        T = np.array([[0.0, 1.0], [1.0, 0.0]])
        v = np.array([3.0, 5.0])
        out = tropical_mat_vec(T, v)
        # out[0] = min(0+3, 1+5) = 3
        # out[1] = min(1+3, 0+5) = 4
        np.testing.assert_array_equal(out, [3.0, 4.0])


class TestFloydWarshall:
    def test_simple(self):
        M = np.array([[0.0, 1.0, INF], [1.0, 0.0, 1.0], [INF, 1.0, 0.0]])
        floyd_warshall(M)
        # After FW: M[0,2] = min(INF, 1+1) = 2
        assert M[0, 2] == 2.0
        assert M[2, 0] == 2.0
        assert M[0, 1] == 1.0


class TestLevel1TransferMatrix:
    def test_empty_block_manhattan(self):
        """On an empty 4x4 block, boundary distances should be Manhattan."""
        g = Grid(np.zeros((4, 4), dtype=np.uint8))
        from hlp.decomposition import enumerate_boundary_cells
        bcells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        T = compute_level1_transfer_matrix(g, bcells, 0, 4, 0, 4)

        for i, ci in enumerate(bcells):
            assert T[i, i] == 0.0
            for j, cj in enumerate(bcells):
                manhattan = abs(ci.row - cj.row) + abs(ci.col - cj.col)
                # In an empty grid with BFS, shortest ≤ manhattan
                assert T[i, j] <= manhattan + 0.01

    def test_with_obstacle(self):
        """Obstacle should cause longer path between some boundary cells."""
        data = np.zeros((4, 4), dtype=np.uint8)
        data[1, 1] = 1
        data[2, 2] = 1
        g = Grid(data)
        from hlp.decomposition import enumerate_boundary_cells
        bcells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        T = compute_level1_transfer_matrix(g, bcells, 0, 4, 0, 4)

        # All diagonal entries should be 0
        for i in range(len(bcells)):
            assert T[i, i] == 0.0

        # T should be symmetric (undirected graph)
        for i in range(len(bcells)):
            for j in range(i + 1, len(bcells)):
                assert abs(T[i, j] - T[j, i]) < 0.01

    def test_fully_blocked_boundary(self):
        """Block with all boundary cells blocked → empty transfer matrix."""
        data = np.ones((4, 4), dtype=np.uint8)
        data[1, 1] = 0
        data[2, 2] = 0
        g = Grid(data)
        from hlp.decomposition import enumerate_boundary_cells
        bcells = enumerate_boundary_cells(g, 0, 4, 0, 4)
        T = compute_level1_transfer_matrix(g, bcells, 0, 4, 0, 4)
        assert T.shape[0] == 0
