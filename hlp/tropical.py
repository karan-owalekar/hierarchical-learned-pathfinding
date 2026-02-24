"""Tropical (min, +) semiring operations and Level-1 transfer matrix computation."""

from __future__ import annotations

import numba
import numpy as np

from hlp.grid import Cell, Grid, bfs_all_distances_within_block

INF = np.inf


# ---------------------------------------------------------------------------
# Scalar tropical ops (plain Python, for clarity / tests)
# ---------------------------------------------------------------------------

def tropical_add(a: float, b: float) -> float:
    return min(a, b)


def tropical_mul(a: float, b: float) -> float:
    if a == INF or b == INF:
        return INF
    return a + b


# ---------------------------------------------------------------------------
# Tropical matrix multiplication — Numba JIT
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    m, p = A.shape
    _p, n = B.shape
    C = np.full((m, n), np.inf)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                val = A[i, k] + B[k, j]
                if val < C[i, j]:
                    C[i, j] = val
    return C


@numba.njit(cache=True)
def tropical_matmul_with_argmin(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tropical matmul that also records the argmin k for path reconstruction."""
    m, p = A.shape
    _p, n = B.shape
    C = np.full((m, n), np.inf)
    argmin_k = np.full((m, n), -1, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                val = A[i, k] + B[k, j]
                if val < C[i, j]:
                    C[i, j] = val
                    argmin_k[i, j] = k
    return C, argmin_k


# ---------------------------------------------------------------------------
# Tropical matrix-vector product
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def tropical_mat_vec(T: np.ndarray, v: np.ndarray) -> np.ndarray:
    """d_out[i] = min_j (T[i][j] + v[j])"""
    m, n = T.shape
    out = np.full(m, np.inf)
    for i in range(m):
        for j in range(n):
            val = T[i, j] + v[j]
            if val < out[i]:
                out[i] = val
    return out


# ---------------------------------------------------------------------------
# Floyd-Warshall (tropical all-pairs shortest paths) — Numba JIT
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def floyd_warshall(M: np.ndarray) -> np.ndarray:
    """In-place tropical Floyd-Warshall. Returns the same array with APSP distances."""
    n = M.shape[0]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                val = M[i, k] + M[k, j]
                if val < M[i, j]:
                    M[i, j] = val
    return M


# ---------------------------------------------------------------------------
# Level-1 transfer matrix computation
# ---------------------------------------------------------------------------

def compute_level1_transfer_matrix(
    grid: Grid,
    boundary_cells: list[Cell],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> np.ndarray:
    n = len(boundary_cells)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)

    T = np.full((n, n), INF, dtype=np.float64)

    for i, src in enumerate(boundary_cells):
        distances = bfs_all_distances_within_block(
            grid, src, row_start, row_end, col_start, col_end,
        )
        for j, dst in enumerate(boundary_cells):
            if dst in distances:
                T[i, j] = distances[dst]

    return T


def tropical_identity(n: int) -> np.ndarray:
    m = np.full((n, n), INF, dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    return m
