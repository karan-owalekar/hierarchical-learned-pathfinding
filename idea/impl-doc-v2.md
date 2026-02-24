# LANCET: Learned Algebraic Navigation via Corridor Estimation with Transfer Matrices
## Complete Implementation Specification v2.0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Definitions and Mathematical Foundations](#2-definitions-and-mathematical-foundations)
3. [Data Structures](#3-data-structures)
4. [Module 1: Grid Engine](#4-module-1-grid-engine)
5. [Module 2: Block Decomposition](#5-module-2-block-decomposition)
6. [Module 3: Tropical Transfer Matrix Engine](#6-module-3-tropical-transfer-matrix-engine)
7. [Module 4: Hierarchical Composition](#7-module-4-hierarchical-composition)
8. [Module 5: Neural Corridor Predictor (Recursive Learned Predictor)](#8-module-5-neural-corridor-predictor)
9. [Module 6: Path Extraction](#9-module-6-path-extraction)
10. [Module 7: Training Pipeline](#10-module-7-training-pipeline)
11. [Module 8: Inference Pipeline & Modes](#11-module-8-inference-pipeline-and-modes)
12. [Module 9: Visualization](#12-module-9-visualization)
13. [Configuration System](#13-configuration-system)
14. [Directory Structure](#14-directory-structure)
15. [Testing Strategy](#15-testing-strategy)
16. [Performance Notes](#16-performance-notes)
17. [End-to-End Walkthrough](#17-end-to-end-walkthrough)

---

## 1. Project Overview

### What This System Does

Given a 2D grid with obstacles, a source cell, and a goal cell, this system finds the shortest (optimal) path using a hierarchical decomposition of the grid into blocks, where shortest-path information within each block is encoded as a **tropical (min, +) semiring transfer matrix**, and a **recursive neural network** predicts which blocks lie on the corridor (reducing computation) by walking DOWN the hierarchy level-by-level.

### The Two Strategies

The system uses two complementary strategies:

**Strategy A: The Algebraic Teacher.** Transfer matrices over the tropical semiring encode exact shortest-path distances between boundary cells of each block. Composing these matrices via tropical matrix multiplication merges blocks while preserving exact distances. This is always correct but computes over ALL blocks.

**Strategy B: The Neural Student (Recursive Learned Predictor).** A small neural network that shares weights across all levels of the hierarchy. At each level, it receives embeddings encoding where the source and goal are relative to the current block, predicts which of the 4 child quadrants the path passes through, produces boundary embeddings for shared internal edges, and passes those embeddings DOWN to the child level. The child levels repeat the process recursively until Level 1 is reached.

### Three Operating Modes

| Mode | Name | Description |
|------|------|-------------|
| `MODE_MATRIX_ONLY` | Algebraic Exact | Full hierarchical transfer matrix computation over ALL blocks. No neural network. Guarantees optimal path. Serves as correctness baseline. |
| `MODE_NEURAL_ONLY` | Neural Heuristic | Recursive neural predictor predicts corridor at all levels. No transfer matrices computed. Uses predicted activations + boundary embeddings to estimate path. Fast but NOT guaranteed optimal. |
| `MODE_HYBRID` | Neural + Algebraic (Full Novel Method) | Neural predictor identifies corridor top-down → transfer matrices computed ONLY within predicted corridor blocks → exact path extracted via algebra. Optimal when corridor prediction has perfect recall. **This is the primary contribution.** |

### Tech Stack

- **Language:** Python 3.10+
- **Core compute:** NumPy, Numba (for JIT-compiled tropical algebra)
- **Neural network:** PyTorch 2.0+
- **Visualization:** Matplotlib
- **Testing:** pytest
- **Config:** YAML via dataclasses

---

## 2. Definitions and Mathematical Foundations

### 2.1 Grid Definition

A grid `G` is a 2D array of shape `(H, W)` where:
- `G[r][c] = 0` means the cell at row `r`, column `c` is **traversable** (free)
- `G[r][c] = 1` means the cell is **blocked** (obstacle)

**Movement model:** 4-connected (up, down, left, right). Each move costs `1.0`. Diagonal movement is NOT supported in this specification.

**Cell indexing:** `(row, col)` with `row` in `[0, H)` and `col` in `[0, W)`. Row 0 is the top.

### 2.2 Tropical Semiring

The **tropical semiring** (min-plus semiring) is `(R ∪ {∞}, ⊕, ⊗)` where:
- `a ⊕ b = min(a, b)` (tropical addition)
- `a ⊗ b = a + b` (tropical multiplication)
- Additive identity: `∞` (since `min(a, ∞) = a`)
- Multiplicative identity: `0` (since `a + 0 = a`)

### 2.3 Transfer Matrix

For a rectangular region (block) with `n` **boundary cells** (free cells on the region's border), the **transfer matrix** `T` is `n × n` where:

```
T[i][j] = shortest path distance from boundary cell i to boundary cell j,
           using ONLY cells inside this region
         = ∞ if no such path exists
```

**Composition property:** For adjacent blocks A and B:
```
(T_A ⊗ T_B)[i][j] = min over k of (T_A[i][k] + T_B[k][j])
```
where `k` ranges over shared boundary cells.

### 2.4 Boundary Cells

A **boundary cell** of a block at rows `[r_start, r_end)` and columns `[c_start, c_end)` is any free cell on the block's edge:

- **Top:** `(r_start, c)` for `c` in `[c_start, c_end)`
- **Bottom:** `(r_end-1, c)` for `c` in `[c_start, c_end)`
- **Left:** `(r, c_start)` for `r` in `[r_start, r_end)`
- **Right:** `(r, c_end-1)` for `r` in `[r_start, r_end)`

Only **free** cells count. Corner cells belong to two boundaries.

**Ordering convention (CRITICAL — clockwise):**
1. Top: left to right
2. Right: top to bottom (skip top-right corner)
3. Bottom: right to left (skip already-counted corners)
4. Left: bottom to top (skip already-counted corners)

### 2.5 Hierarchical Levels

Recursive partitioning:

- **Level 0:** Individual grid cells
- **Level 1:** Grid divided into `B × B` blocks. Transfer matrices computed by BFS.
- **Level L (L ≥ 2):** Groups of `2 × 2` Level-(L-1) blocks form a Level-L block. Transfer matrix computed by composing children.
- Continue until the entire grid is one top-level block.

Number of levels: `L = 1 + ceil(log2(max(H, W) / B))`

### 2.6 Corridor

The **corridor** at a given level is the set of blocks the shortest path passes through. At the top level, the corridor is the single top-level block. At each level going down, the corridor narrows: of the 4 children of each corridor block, only those the path actually passes through are in the corridor.

The neural predictor estimates this corridor top-down, level by level.

### 2.7 Entry and Exit Embeddings

At every level, the source and goal positions are encoded as **entry** and **exit embeddings** relative to the current block:

- **Entry embedding** for a block: a vector encoding the shortest distances from the source to each boundary cell of that block (or the predicted approximation thereof).
- **Exit embedding** for a block: a vector encoding the shortest distances from each boundary cell to the goal.

At the top level, these are computed exactly (BFS from source/goal to the top-level block's boundary). At lower levels, the neural network produces them from the parent level's embeddings.

---

## 3. Data Structures

### 3.1 `Cell`

```python
@dataclass(frozen=True)
class Cell:
    row: int
    col: int
```

Hashable, comparable, used as dictionary keys.

### 3.2 `Grid`

```python
class Grid:
    data: np.ndarray          # shape (H, W), dtype np.uint8, 0=free 1=blocked
    height: int
    width: int

    def is_free(self, r: int, c: int) -> bool
    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]
    def in_bounds(self, r: int, c: int) -> bool
```

### 3.3 `Block`

```python
@dataclass
class Block:
    block_id: Tuple[int, int, int]         # (level, block_row, block_col)
    level: int
    row_start: int                          # inclusive, grid coords
    row_end: int                            # exclusive
    col_start: int                          # inclusive
    col_end: int                            # exclusive
    boundary_cells: List[Cell]              # ordered clockwise
    boundary_cell_to_index: Dict[Cell, int] # reverse lookup
    transfer_matrix: Optional[np.ndarray]   # (n, n), float64, ∞ for no path
    children: Optional[List[Block]]         # 4 children if level > 1
    is_active: bool                         # in the predicted corridor?

    # Shared boundary references (populated during hierarchy building)
    shared_top: Optional[List[Cell]]        # shared with block above
    shared_bottom: Optional[List[Cell]]     # shared with block below
    shared_left: Optional[List[Cell]]       # shared with block to the left
    shared_right: Optional[List[Cell]]      # shared with block to the right
```

### 3.4 `TransferMatrix`

```python
class TransferMatrix:
    matrix: np.ndarray           # (n, n), float64
    boundary_cells: List[Cell]
    cell_to_idx: Dict[Cell, int]

    @staticmethod
    def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """C[i][j] = min_k (A[i][k] + B[k][j])"""

    @staticmethod
    def tropical_matmul_with_argmin(A, B) -> Tuple[np.ndarray, np.ndarray]:
        """Also returns which k achieved the min for path traceback"""
```

### 3.5 `LevelPrediction` (Neural Output at One Level)

```python
@dataclass
class LevelPrediction:
    level: int
    block_id: Tuple[int, int, int]
    quadrant_activations: np.ndarray    # shape (4,), sigmoid probabilities
    active_quadrants: List[int]         # indices of activated children (0-3)
    boundary_embeddings: Dict[str, np.ndarray]
    # Keys: 'horizontal', 'vertical' — embeddings for shared internal edges
    # Each is shape (d,) where d = embedding dimension
```

### 3.6 `PathResult`

```python
@dataclass
class PathResult:
    path: List[Cell]
    cost: float
    optimal: bool
    mode: str
    corridor_blocks: int          # how many Level-1 blocks in corridor
    total_blocks: int
    computation_time_ms: float
    neural_time_ms: float         # time for neural corridor prediction
    algebra_time_ms: float        # time for transfer matrix computation
```

---

## 4. Module 1: Grid Engine

**File:** `lancet/grid.py`

### 4.1 Grid Generation

```python
def generate_grid(height: int, width: int, obstacle_density: float,
                  seed: Optional[int] = None) -> Grid:
    """
    Generate a random grid.

    1. Create HxW array of zeros
    2. Randomly set cells to 1 with probability = obstacle_density
    3. Keep (0,0) and (H-1, W-1) free
    4. Verify connectivity via BFS; retry if disconnected
    """
```

### 4.2 Grid I/O

```python
def save_grid(grid: Grid, filepath: str) -> None:
    """Save as .npy"""

def load_grid(filepath: str) -> Grid:
    """Load from .npy"""
```

### 4.3 BFS Shortest Path

```python
def bfs_shortest_path(grid: Grid, source: Cell, goal: Cell) -> Optional[Tuple[List[Cell], float]]:
    """
    Standard BFS on uniform-cost grid. Returns (path, cost) or None.
    Used as ground truth oracle and for Level-1 transfer matrix computation.
    """

def bfs_within_block(grid: Grid, source: Cell, goal: Cell,
                     row_start: int, row_end: int,
                     col_start: int, col_end: int) -> Optional[float]:
    """BFS restricted to block bounds. Returns distance only."""

def bfs_all_distances_within_block(grid: Grid, source: Cell,
                                    row_start: int, row_end: int,
                                    col_start: int, col_end: int) -> Dict[Cell, float]:
    """BFS from source within block. Returns distances to ALL reachable cells."""
```

### 4.4 Multi-Source BFS

```python
def multi_source_bfs_within_block(grid: Grid, sources: List[Cell],
                                   row_start: int, row_end: int,
                                   col_start: int, col_end: int) -> np.ndarray:
    """
    Run BFS from each source to all others within block bounds.
    Returns (n, n) distance matrix. O(n * B^2) total.
    """
```

---

## 5. Module 2: Block Decomposition

**File:** `lancet/decomposition.py`

### 5.1 Grid Padding

```python
def pad_grid_to_hierarchy(grid: Grid, block_size: int) -> Grid:
    """
    Pad grid so dimensions are power-of-2 multiples of block_size.

    padded_H = block_size * (2 ** ceil(log2(ceil(H / block_size))))
    padded_W = block_size * (2 ** ceil(log2(ceil(W / block_size))))

    Pad with blocked cells (1s).
    """
```

### 5.2 Level 1 Partitioning

```python
def partition_into_blocks(grid: Grid, block_size: int) -> List[List[Block]]:
    """
    Divide padded grid into block_size × block_size Level-1 blocks.
    Returns 2D list: blocks[block_row][block_col].
    """
```

### 5.3 Boundary Cell Enumeration

```python
def enumerate_boundary_cells(grid: Grid, row_start: int, row_end: int,
                              col_start: int, col_end: int) -> List[Cell]:
    """
    Enumerate free boundary cells in clockwise order.

    1. Top: (row_start, c) for c in [col_start, col_end), left to right
    2. Right: (r, col_end-1) for r in [row_start+1, row_end-1), top to bottom
    3. Bottom: (row_end-1, c) for c in [col_end-1, ..., col_start], right to left
    4. Left: (r, col_start) for r in [row_end-2, ..., row_start+1], bottom to top

    Only free cells. Handle degenerate blocks (1-wide, 1-tall).
    """
```

### 5.4 Hierarchical Block Tree

```python
def build_block_hierarchy(grid: Grid, level1_blocks: List[List[Block]]) -> Dict[Tuple, Block]:
    """
    Build full hierarchy. Returns dict: block_id -> Block for ALL levels.

    For level in [2, ..., max_level]:
        Group 2x2 blocks from level-1 into level-L blocks.
        Compute merged bounds and boundary cells.
        Store children references.
        Identify shared boundaries between children.
    """
```

### 5.5 Shared Boundary Computation

```python
def compute_shared_boundaries(parent: Block, children: List[Block]) -> Dict[str, List[Cell]]:
    """
    Identify shared internal boundaries between the 4 children of a parent block.

    Children layout:
        +-------+-------+
        |  TL   |  TR   |
        | [0]   | [1]   |
        +-------+-------+
        |  BL   |  BR   |
        | [2]   | [3]   |
        +-------+-------+

    Shared boundaries:
        'horizontal': between TL.bottom/BL.top AND TR.bottom/BR.top
            = cells at the row boundary between top and bottom halves
        'vertical': between TL.right/TR.left AND BL.right/BR.left
            = cells at the column boundary between left and right halves

    IMPORTANT: These are ADJACENT cells (one step apart), not the same cells.
    TL's right boundary column is (TL.col_end - 1), TR's left column is TR.col_start.
    TL.col_end == TR.col_start, so these cells are adjacent with crossing cost 1.

    Returns dict mapping edge name to list of (cell_from_block_A, cell_from_block_B) pairs.
    """
```

---

## 6. Module 3: Tropical Transfer Matrix Engine

**File:** `lancet/tropical.py`

### 6.1 Tropical Operations

```python
INF = float('inf')

def tropical_add(a: float, b: float) -> float:
    return min(a, b)

def tropical_multiply(a: float, b: float) -> float:
    if a == INF or b == INF:
        return INF
    return a + b
```

### 6.2 Tropical Matrix Multiplication

```python
@numba.njit(cache=True)
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    C[i][j] = min_k (A[i][k] + B[k][j])
    A: (m, p), B: (p, n) -> C: (m, n). All float64, inf-safe.
    """
    m, p = A.shape
    n = B.shape[1]
    C = np.full((m, n), np.inf)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                v = A[i, k] + B[k, j]
                if v < C[i, j]:
                    C[i, j] = v
    return C
```

### 6.3 Tropical Matrix Multiplication with Argmin Traceback

```python
@numba.njit(cache=True)
def tropical_matmul_with_argmin(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Same as tropical_matmul but also returns argmin_k matrix.
    argmin_k[i][j] = the k that achieved min in C[i][j].
    Essential for path reconstruction.
    """
    m, p = A.shape
    n = B.shape[1]
    C = np.full((m, n), np.inf)
    K = np.full((m, n), -1, dtype=np.int64)
    for i in range(m):
        for j in range(n):
            for k in range(p):
                v = A[i, k] + B[k, j]
                if v < C[i, j]:
                    C[i, j] = v
                    K[i, j] = k
    return C, K
```

### 6.4 Level-1 Transfer Matrix Computation

```python
def compute_level1_transfer_matrix(grid: Grid, block: Block) -> np.ndarray:
    """
    Compute transfer matrix for a Level-1 block via BFS.

    n = len(block.boundary_cells)
    T = np.full((n, n), np.inf)

    For each boundary cell i:
        Run BFS from boundary_cells[i] within block bounds
        For each boundary cell j:
            T[i][j] = BFS distance from i to j within block

    T[i][i] = 0 for all i (sanity check)

    Complexity: O(n * B^2), n ≤ 4B.
    For B=16: O(64 * 256) = ~16K per block.
    """
```

---

## 7. Module 4: Hierarchical Composition

**File:** `lancet/composition.py`

### 7.1 Composition Overview

A Level-L block has 4 children arranged as TL, TR, BL, BR. The parent's transfer matrix relates its OUTER boundary cells. Composition eliminates all INTERNAL boundary cells (shared edges between children).

### 7.2 Combined Matrix Construction

```python
def build_combined_boundary_graph(children: List[Block], grid: Grid,
                                   parent: Block) -> Tuple[np.ndarray, List[Cell], List[int]]:
    """
    Build combined distance matrix over ALL boundary cells of ALL children.

    N = total boundary cells across all 4 children.
    M = N×N matrix:

    For cells i, j in SAME child: M[i][j] = child's transfer matrix entry
    For cells i, j in DIFFERENT children:
        M[i][j] = 1 if cells are grid-adjacent (shared boundary, one step)
        M[i][j] = ∞ otherwise

    Returns:
        M: the combined matrix (N, N)
        all_cells: ordered list of all boundary cells
        external_indices: which indices correspond to parent's boundary cells
    """
```

### 7.3 Floyd-Warshall Elimination

```python
@numba.njit(cache=True)
def floyd_warshall(M: np.ndarray) -> np.ndarray:
    """
    All-pairs shortest paths in tropical semiring.

    for k in range(N):
        for i in range(N):
            for j in range(N):
                M[i][j] = min(M[i][j], M[i][k] + M[k][j])

    O(N^3). For N ≤ 256 (4 blocks × 64 boundary cells), this is fast.
    """
```

### 7.4 Full Composition

```python
def compose_transfer_matrices(children: List[Block], grid: Grid,
                               parent: Block) -> np.ndarray:
    """
    1. Build combined matrix M
    2. Floyd-Warshall on M
    3. Extract submatrix for parent's external boundary cells only
    4. Return parent transfer matrix
    """
```

### 7.5 Recursive Top-Down Computation

```python
def compute_all_transfer_matrices(all_blocks: Dict, grid: Grid,
                                   active_only: bool = False,
                                   active_set: Optional[Set] = None) -> None:
    """
    Bottom-up transfer matrix computation.

    1. Level 1: BFS for each block (optionally only active blocks)
    2. Level 2+: Compose from children (only if block is active or active_only=False)

    In MATRIX_ONLY mode: active_only=False, compute everything.
    In HYBRID mode: active_only=True, active_set = neural corridor prediction.
    """
```

---

## 8. Module 5: Neural Corridor Predictor (Recursive Learned Predictor)

**File:** `lancet/neural/model.py`

**THIS IS THE CORE NOVEL COMPONENT.**

The neural network is NOT an image segmentation model. It is a small MLP that is applied RECURSIVELY at each level of the hierarchy, sharing weights across levels. It walks TOP-DOWN through the hierarchy, predicting which children are active and producing boundary embeddings that are passed to child levels.

### 8.1 Core Concept

At every level of the hierarchy, the network answers one question per block: **"Which of my 4 child quadrants does the shortest path pass through?"**

It answers this using:
- **Entry embedding:** encodes where the source is relative to this block (distances from source to boundary cells)
- **Exit embedding:** encodes where the goal is relative to this block
- **Level embedding:** encodes which level of the hierarchy we're at

It produces:
- **Quadrant activations:** 4 sigmoid values, one per child quadrant
- **Boundary embeddings:** vectors for the shared internal edges, which become the entry/exit context for child-level predictions

### 8.2 Input Representation at Each Level

At a given level, for a given block, the network receives:

```
Input vector = concat(entry_emb, exit_emb, level_emb)
```

Where:

**entry_emb: shape (d,)**
- At the TOP level: computed EXACTLY from BFS.
  - Run BFS from source to all boundary cells of the top-level block.
  - Encode as a vector: the raw distances, normalized, then passed through a linear projection to dimension d.
  - `entry_emb = Linear(normalize(bfs_distances_from_source))` → shape (d,)
- At LOWER levels: received from the parent level's boundary embedding output.
  - The parent's "boundary embedding" for the shared edge between this block and its sibling becomes this block's entry or exit context.

**exit_emb: shape (d,)**
- Same as entry_emb but for the goal.
- Top level: `exit_emb = Linear(normalize(bfs_distances_to_goal))`
- Lower levels: from parent's boundary embedding.

**level_emb: shape (d,)**
- A learned embedding vector for each level.
- `level_emb = level_embedding_table[level]` where level_embedding_table is shape (max_levels, d).

**Total input dimension: 3d**

### 8.3 Network Architecture

```python
class RecursiveCorridorPredictor(nn.Module):
    """
    Shared-weight MLP applied at every level of the hierarchy.

    All levels use the SAME weights — this is critical for generalization
    across grid sizes and hierarchy depths.

    Architecture:
        Input: (3d,) = concat(entry_emb, exit_emb, level_emb)

        Backbone (shared across all 3 heads):
            Linear(3d, 4d) → LayerNorm → ReLU
            Linear(4d, 4d) → LayerNorm → ReLU
            Linear(4d, 2d) → LayerNorm → ReLU

        Head 1 — Subregion Activation:
            Linear(2d, d) → ReLU
            Linear(d, 4) → Sigmoid
            Output: (4,) — probability each of 4 child quadrants is active

        Head 2 — Boundary Embeddings:
            For EACH of the 4 inter-quadrant edges:
                Linear(2d, d) → ReLU
                Linear(d, d)
            Output: 4 vectors of shape (d,)
            These represent:
                - 'top_left_to_top_right': vertical shared edge, top half
                - 'top_left_to_bottom_left': horizontal shared edge, left half
                - 'top_right_to_bottom_right': horizontal shared edge, right half
                - 'bottom_left_to_bottom_right': vertical shared edge, bottom half

            ACTUALLY SIMPLIFIED: The paper describes 2 shared internal edges:
                - 'horizontal': between top pair and bottom pair
                - 'vertical': between left pair and right pair
            So 2 boundary embedding outputs, each shape (d,).

        Head 3 — Boundary Cell Prediction (TRAINING ONLY):
            Linear(2d, d) → ReLU
            Linear(d, max_boundary_cells) → Softmax
            Output: (max_boundary_cells,) — probability distribution over
                    which boundary cell the path crosses at each shared edge
            Purpose: provides direct supervision signal for boundary embeddings
            Discarded at inference time.

    Parameters:
        d: embedding dimension (default 64)
        max_boundary_cells: max boundary cells per edge (= block_size for Level-1 edges)
    """

    def __init__(self, d: int = 64, max_levels: int = 10, max_boundary_cells: int = 64):
        super().__init__()
        self.d = d

        # Level embeddings
        self.level_embedding = nn.Embedding(max_levels, d)

        # Entry/exit embedding projectors (used at top level only)
        # Input dimension varies per block, so use a small adaptive network
        self.entry_projector = nn.Sequential(
            nn.Linear(max_boundary_cells, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d)
        )
        self.exit_projector = nn.Sequential(
            nn.Linear(max_boundary_cells, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d)
        )

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(3 * d, 4 * d),
            nn.LayerNorm(4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, 4 * d),
            nn.LayerNorm(4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, 2 * d),
            nn.LayerNorm(2 * d),
            nn.ReLU()
        )

        # Head 1: Subregion activation (4 quadrants)
        self.activation_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 4),
            nn.Sigmoid()
        )

        # Head 2: Boundary embeddings (2 shared edges)
        self.horizontal_boundary_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )
        self.vertical_boundary_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, d)
        )

        # Head 3: Boundary cell prediction (training only)
        self.boundary_cell_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, max_boundary_cells)
            # Softmax applied in loss function
        )

    def forward(self, entry_emb: torch.Tensor, exit_emb: torch.Tensor,
                level: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for ONE block at ONE level.

        Args:
            entry_emb: (batch, d) or (d,) — entry context for this block
            exit_emb: (batch, d) or (d,) — exit context for this block
            level: int — which hierarchy level

        Returns:
            activations: (batch, 4) — quadrant activation probabilities
            horiz_boundary_emb: (batch, d) — boundary embedding for horizontal edge
            vert_boundary_emb: (batch, d) — boundary embedding for vertical edge
            boundary_cell_logits: (batch, max_boundary_cells) — cell prediction logits
        """
        level_emb = self.level_embedding(torch.tensor(level))  # (d,)

        x = torch.cat([entry_emb, exit_emb, level_emb.expand_as(entry_emb)], dim=-1)  # (batch, 3d)

        features = self.backbone(x)  # (batch, 2d)

        activations = self.activation_head(features)  # (batch, 4)
        horiz_emb = self.horizontal_boundary_head(features)  # (batch, d)
        vert_emb = self.vertical_boundary_head(features)  # (batch, d)
        cell_logits = self.boundary_cell_head(features)  # (batch, max_boundary_cells)

        return activations, horiz_emb, vert_emb, cell_logits
```

### 8.4 Entry/Exit Embedding Computation at Top Level

```python
def compute_top_level_embeddings(grid: Grid, source: Cell, goal: Cell,
                                  top_block: Block, model: RecursiveCorridorPredictor,
                                  max_boundary_cells: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entry and exit embeddings for the top-level block.
    These are the ONLY place where we use exact BFS distances.

    Algorithm:
    1. Run BFS from source to all boundary cells of top_block
       → distances_from_source: list of floats, length = n_boundary
    2. Run BFS from goal to all boundary cells of top_block
       → distances_to_goal: list of floats, length = n_boundary
    3. Pad/truncate to max_boundary_cells:
       If n_boundary < max_boundary_cells: pad with large value (999)
       If n_boundary > max_boundary_cells: error (shouldn't happen with proper config)
    4. Normalize: divide by max non-inf value, replace inf with 1.0
    5. Project through model's entry/exit projectors:
       entry_emb = model.entry_projector(distances_from_source_tensor)  → (d,)
       exit_emb = model.exit_projector(distances_to_goal_tensor)  → (d,)

    Returns:
        entry_emb: (d,) tensor
        exit_emb: (d,) tensor
    """
```

### 8.5 Recursive Top-Down Inference

```python
def predict_corridor_recursive(model: RecursiveCorridorPredictor,
                                grid: Grid, source: Cell, goal: Cell,
                                all_blocks: Dict, top_level: int,
                                config: InferenceConfig) -> Set[Tuple]:
    """
    THE MAIN NEURAL INFERENCE PROCEDURE.

    Walks top-down through the hierarchy, predicting active quadrants
    at each level and propagating boundary embeddings downward.

    Returns: set of active Level-1 block IDs = the predicted corridor.

    Algorithm:

    Step 1: Compute top-level entry/exit embeddings (exact, from BFS)
        entry_emb, exit_emb = compute_top_level_embeddings(...)

    Step 2: Initialize recursion queue
        queue = [(top_block_id, entry_emb, exit_emb)]

    Step 3: Process each level top-down
        active_level1_blocks = set()

        while queue:
            current_block_id, entry_emb, exit_emb = queue.pop(0)
            block = all_blocks[current_block_id]

            if block.level == 1:
                # Reached Level 1 — this block is in the corridor
                active_level1_blocks.add(current_block_id)
                continue

            # Run neural network for this block
            activations, horiz_emb, vert_emb, _ = model(entry_emb, exit_emb, block.level)

            # Determine active children
            children = block.children  # [TL, TR, BL, BR]
            for i, child in enumerate(children):
                if child is None:
                    continue
                if activations[i] > config.corridor_threshold:
                    # This child is predicted to be on the corridor

                    # Compute child's entry/exit embeddings from parent's boundary embeddings
                    child_entry, child_exit = derive_child_embeddings(
                        i, entry_emb, exit_emb, horiz_emb, vert_emb, block, child
                    )

                    queue.append((child.block_id, child_entry, child_exit))

        return active_level1_blocks
    """
```

### 8.6 Deriving Child Embeddings from Parent

This is the key propagation mechanism — how the neural information flows down the hierarchy.

```python
def derive_child_embeddings(child_index: int,
                             parent_entry: torch.Tensor,
                             parent_exit: torch.Tensor,
                             horiz_boundary_emb: torch.Tensor,
                             vert_boundary_emb: torch.Tensor,
                             parent_block: Block,
                             child_block: Block) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entry and exit embeddings for a child block, given the parent's
    embeddings and the predicted boundary embeddings.

    The child's "entry" comes from the source side. Depending on which quadrant
    the child is and where the source is relative to it:

    Children layout:
        [0] TL  [1] TR
        [2] BL  [3] BR

    Horizontal boundary embedding: context about the top↔bottom crossing
    Vertical boundary embedding: context about the left↔right crossing

    The idea:
    - If the source is in the same quadrant as this child, the child's entry_emb
      comes directly from the parent's entry_emb (the source info propagates down)
    - If the source is in a DIFFERENT quadrant, the child's entry_emb comes from
      the boundary embedding of the edge BETWEEN the source's quadrant and this one
    - Same logic for exit_emb with the goal

    Simplified mapping (which boundary embedding to use):

    For child_index = 0 (TL):
        entry = parent_entry (source info from parent)
            UNLESS source is in bottom half → entry influenced by horiz_emb
            UNLESS source is in right half → entry influenced by vert_emb
        exit = depends on where goal is

    PRACTICAL SIMPLIFICATION FOR INITIAL IMPLEMENTATION:
    Instead of complex conditional routing, use a small learned combiner:

        child_entry = Linear(concat(parent_entry, horiz_emb, vert_emb, child_position_encoding))
        child_exit = Linear(concat(parent_exit, horiz_emb, vert_emb, child_position_encoding))

    where child_position_encoding is a one-hot (4,) vector indicating which quadrant.

    This lets the network learn HOW to combine the parent context, rather than
    hard-coding the routing logic.
    """
```

```python
class ChildEmbeddingDeriver(nn.Module):
    """
    Learns to produce child entry/exit embeddings from parent context.

    Input: parent_entry (d) + parent_exit (d) + horiz_emb (d) + vert_emb (d) + child_onehot (4)
    Total input: 4d + 4

    Output: child_entry (d), child_exit (d)
    """

    def __init__(self, d: int = 64):
        super().__init__()
        self.entry_net = nn.Sequential(
            nn.Linear(4 * d + 4, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d)
        )
        self.exit_net = nn.Sequential(
            nn.Linear(4 * d + 4, 2 * d),
            nn.ReLU(),
            nn.Linear(2 * d, d)
        )

    def forward(self, parent_entry, parent_exit, horiz_emb, vert_emb, child_index):
        child_onehot = torch.zeros(4)
        child_onehot[child_index] = 1.0

        combined = torch.cat([parent_entry, parent_exit, horiz_emb, vert_emb, child_onehot], dim=-1)

        child_entry = self.entry_net(combined)
        child_exit = self.exit_net(combined)

        return child_entry, child_exit
```

### 8.7 Full Model Assembly

```python
class LANCETNeuralPredictor(nn.Module):
    """
    Complete neural corridor predictor.
    Combines RecursiveCorridorPredictor + ChildEmbeddingDeriver.
    """

    def __init__(self, d: int = 64, max_levels: int = 10, max_boundary_cells: int = 64):
        super().__init__()
        self.d = d
        self.predictor = RecursiveCorridorPredictor(d, max_levels, max_boundary_cells)
        self.child_deriver = ChildEmbeddingDeriver(d)

    def predict_single_level(self, entry_emb, exit_emb, level):
        """Predict activations and boundary embeddings for one block at one level."""
        return self.predictor(entry_emb, exit_emb, level)

    def derive_child_context(self, parent_entry, parent_exit, horiz_emb, vert_emb, child_idx):
        """Derive child entry/exit embeddings."""
        return self.child_deriver(parent_entry, parent_exit, horiz_emb, vert_emb, child_idx)
```

---

## 9. Module 6: Path Extraction

**File:** `lancet/extraction.py`

### 9.1 Source/Goal Embedding Computation

```python
def compute_entry_exit_embeddings(grid: Grid, source: Cell, goal: Cell,
                                    block: Block) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact entry and exit distance vectors for a block containing source/goal.

    entry_distances[i] = BFS distance from source to boundary_cells[i] (within block)
    exit_distances[i] = BFS distance from boundary_cells[i] to goal (within block)

    If source is not in this block: entry uses the transfer matrix to reach this block's
    boundary from the corridor.
    """
```

### 9.2 Corridor Transfer Matrix Assembly

```python
def assemble_corridor_transfer_matrix(source_block: Block, goal_block: Block,
                                       corridor_blocks: Set,
                                       all_blocks: Dict,
                                       grid: Grid) -> np.ndarray:
    """
    Build the transfer matrix spanning from source block's boundary to
    goal block's boundary, through the corridor.

    SIMPLIFIED APPROACH (implement first):
        1. Collect all boundary cells of all corridor blocks
        2. Build combined distance matrix:
           - Within-block distances from transfer matrices
           - Between adjacent corridor blocks: crossing cost = 1
        3. Floyd-Warshall on combined matrix
        4. Extract submatrix: source block boundary → goal block boundary

    Returns: (n_source_boundary, n_goal_boundary) matrix
    """
```

### 9.3 Distance Query

```python
def query_optimal_distance(source: Cell, goal: Cell,
                            source_block: Block, goal_block: Block,
                            corridor_tm: np.ndarray) -> float:
    """
    distance = min over (i, j) of:
        entry_distances[i] + corridor_tm[i][j] + exit_distances[j]

    This is two tropical matrix-vector products:
        temp[j] = min_i (entry[i] + corridor_tm[i][j])   — tropical mat-vec
        distance = min_j (temp[j] + exit[j])               — tropical dot product
    """
```

### 9.4 Path Reconstruction

```python
def reconstruct_path(source: Cell, goal: Cell,
                      source_block: Block, goal_block: Block,
                      corridor_blocks: List[Block],
                      all_blocks: Dict, grid: Grid) -> List[Cell]:
    """
    Full cell-by-cell path reconstruction.

    1. Find optimal (i*, j*) = argmin of entry[i] + corridor_tm[i][j] + exit[j]
    2. Trace through corridor to find boundary cell sequence:
       [source, ..., boundary_i*, ..., boundary_j*, ..., goal]
       Using argmin traceback from tropical_matmul_with_argmin
    3. For each consecutive pair in the same block:
       BFS within block to get cell-level path
    4. For each consecutive pair in adjacent blocks:
       Single grid step (cost 1)
    5. Concatenate all segments
    """
```

---

## 10. Module 7: Training Pipeline

**File:** `lancet/neural/train.py`, `lancet/neural/dataset.py`

### 10.1 Training Data Structure

Each training example requires labels at EVERY level of the hierarchy.

```python
@dataclass
class TrainingSample:
    grid: Grid
    source: Cell
    goal: Cell
    optimal_path: List[Cell]
    optimal_cost: float

    # Per-level labels (computed from optimal_path)
    level_labels: Dict[int, LevelLabel]  # level -> LevelLabel

@dataclass
class LevelLabel:
    level: int
    block_id: Tuple[int, int, int]

    # Which quadrants the path passes through (ground truth)
    active_quadrants: np.ndarray          # shape (4,), binary {0, 1}

    # Exact entry embedding (BFS distances from source to block boundary)
    exact_entry_distances: np.ndarray     # shape (n_boundary,)

    # Exact exit embedding (BFS distances from goal to block boundary)
    exact_exit_distances: np.ndarray      # shape (n_boundary,)

    # Which boundary cells the path crosses at each shared edge
    exact_horizontal_crossing: Optional[int]  # index of crossing cell, or None
    exact_vertical_crossing: Optional[int]    # index of crossing cell, or None

    # Exact corridor — which blocks at the level below
    exact_corridor_at_child_level: Set[Tuple]
```

### 10.2 Label Generation

```python
def generate_level_labels(grid: Grid, source: Cell, goal: Cell,
                           optimal_path: List[Cell],
                           all_blocks: Dict, max_level: int,
                           block_size: int) -> Dict[int, List[LevelLabel]]:
    """
    Generate training labels for all levels of the hierarchy.

    For each level from max_level down to 1:
        For each block at this level that the path passes through:
            1. Determine which children the path passes through
               → active_quadrants binary vector

            2. Compute exact entry/exit distances:
               Run BFS from source/goal to this block's boundary cells
               → exact_entry_distances, exact_exit_distances

            3. Determine exact boundary crossings:
               For each shared internal edge (horizontal, vertical):
                 Find where the optimal path crosses this edge
                 → index of the boundary cell pair

    Path → block membership:
        For each cell in the path, determine which Level-1 block it's in.
        For higher levels, a block is "on the path" if any of its descendants are.

    IMPORTANT: A single training example generates labels at ALL levels.
    The top level has 1 block with 1 label.
    Level L-1 has up to 4 blocks with labels (the active children).
    Level L-2 has up to 16 blocks with labels.
    ... etc.
    At Level 1, only the actual corridor blocks have labels.
    """
```

### 10.3 Training Loop

```python
def train_epoch(model: LANCETNeuralPredictor, dataloader, optimizer,
                config: TrainConfig, epoch: int) -> Dict:
    """
    Training loop for one epoch.

    CRITICAL DIFFERENCE FROM TYPICAL TRAINING:
    Each training sample is processed LEVEL BY LEVEL, top-down,
    mirroring the inference procedure.

    For each sample (grid, source, goal, level_labels):

        # Phase 1: Compute top-level embeddings (always exact)
        entry_emb, exit_emb = compute_top_level_embeddings(...)

        total_loss = 0

        # Phase 2: Process each level top-down
        queue = [(top_block_id, entry_emb, exit_emb)]

        while queue:
            block_id, entry_emb, exit_emb = queue.pop(0)
            block = all_blocks[block_id]
            label = level_labels[block.level][block_id]

            if block.level == 1:
                continue  # no children to predict

            # Forward pass for this block
            activations, horiz_emb, vert_emb, cell_logits = model.predict_single_level(
                entry_emb, exit_emb, block.level
            )

            # Loss 1: Subregion activation loss (BCE)
            activation_loss = F.binary_cross_entropy(
                activations, label.active_quadrants_tensor
            )

            # Loss 2: Boundary cell prediction loss (cross-entropy)
            cell_loss = 0
            if label.exact_horizontal_crossing is not None:
                cell_loss += F.cross_entropy(
                    cell_logits_horiz, label.exact_horizontal_crossing_tensor
                )
            if label.exact_vertical_crossing is not None:
                cell_loss += F.cross_entropy(
                    cell_logits_vert, label.exact_vertical_crossing_tensor
                )

            # Total loss for this level
            level_loss = activation_loss + config.cell_loss_weight * cell_loss
            total_loss += level_loss

            # Propagate to children — SCHEDULED SAMPLING
            for i, child in enumerate(block.children):
                if child is None:
                    continue

                # Is this child actually active (ground truth)?
                child_is_active = label.active_quadrants[i] > 0.5

                if not child_is_active:
                    continue  # only recurse into true corridor blocks during training

                # SCHEDULED SAMPLING:
                # With probability `use_predicted_prob`, use network's own embeddings
                # With probability `1 - use_predicted_prob`, use exact embeddings
                use_predicted = random.random() < scheduled_sampling_prob(epoch, config)

                if use_predicted:
                    child_entry, child_exit = model.derive_child_context(
                        entry_emb, exit_emb, horiz_emb, vert_emb, i
                    )
                else:
                    # Use exact embeddings from BFS (ground truth)
                    child_entry = model.predictor.entry_projector(
                        label_for_child.exact_entry_distances_tensor
                    )
                    child_exit = model.predictor.exit_projector(
                        label_for_child.exact_exit_distances_tensor
                    )

                queue.append((child.block_id, child_entry, child_exit))

        # Backprop
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    """
```

### 10.4 Scheduled Sampling Schedule

```python
def scheduled_sampling_prob(epoch: int, config: TrainConfig) -> float:
    """
    Probability of using the model's own predicted embeddings instead of
    exact (ground truth) embeddings when propagating to child levels.

    Phase 1 (epochs 0 to warmup): 0.0 (always use exact)
        Network learns with perfect inputs first.

    Phase 2 (epochs warmup to transition): linear increase from 0.0 to max_prob
        Gradually expose the network to its own errors.

    Phase 3 (epochs transition onward): max_prob (e.g., 0.5)
        Network trains with a mix of exact and predicted.

    Default schedule:
        warmup = 10 epochs
        transition = 30 epochs
        max_prob = 0.5

    This is exactly the scheduled sampling from sequence-to-sequence models
    (Bengio et al., 2015), applied to spatial hierarchy instead of temporal sequence.
    """
    if epoch < config.warmup_epochs:
        return 0.0
    elif epoch < config.transition_epochs:
        progress = (epoch - config.warmup_epochs) / (config.transition_epochs - config.warmup_epochs)
        return progress * config.max_sampling_prob
    else:
        return config.max_sampling_prob
```

### 10.5 Curriculum Training

```python
def train_full(model: LANCETNeuralPredictor, config: TrainConfig) -> None:
    """
    Full training with curriculum over grid sizes.

    Stage 1 (epochs 1-15):
        Grid size: 64×64 (small, 4×4 blocks at Level 1, ~3 levels)
        Obstacle density: 0.1-0.2
        LR: 1e-3
        Scheduled sampling: 0.0 (all exact embeddings)
        Purpose: Learn basic quadrant activation

    Stage 2 (epochs 16-30):
        Grid size: 64×64 and 128×128 (~4 levels)
        Obstacle density: 0.1-0.3
        LR: 5e-4
        Scheduled sampling: ramp from 0 to 0.3
        Purpose: Learn deeper hierarchies, begin handling own errors

    Stage 3 (epochs 31-50):
        Grid size: 128×128 and 256×256 (~5 levels)
        Obstacle density: 0.1-0.35
        LR: 1e-4
        Scheduled sampling: ramp from 0.3 to 0.5
        Purpose: Full-scale operation, robust to self-generated embeddings

    Because the network shares weights across levels, training on small grids
    (fewer levels) directly improves performance on large grids (more levels).
    The level embedding distinguishes behavior at different depths.
    """
```

### 10.6 Auxiliary Loss: Boundary Cell Prediction

```python
def boundary_cell_loss(cell_logits: torch.Tensor, target_crossing: int,
                        num_valid_cells: int) -> torch.Tensor:
    """
    Cross-entropy loss for predicting which boundary cell the path crosses
    at a shared internal edge.

    cell_logits: (max_boundary_cells,) — raw logits from Head 3
    target_crossing: int — index of the correct boundary cell
    num_valid_cells: int — how many boundary cells actually exist at this edge
                          (mask out the rest)

    This loss ONLY shapes the boundary embeddings during training.
    It forces the boundary embeddings to encode meaningful spatial information
    about WHERE the path crosses internal edges.

    At inference, Head 3 is never called — only Head 1 (activations)
    and Head 2 (boundary embeddings) are used.
    """
    # Mask logits beyond num_valid_cells
    mask = torch.full_like(cell_logits, float('-inf'))
    mask[:num_valid_cells] = 0
    masked_logits = cell_logits + mask

    return F.cross_entropy(masked_logits.unsqueeze(0),
                           torch.tensor([target_crossing]))
```

### 10.7 Training Metrics

```python
def evaluate_model(model: LANCETNeuralPredictor, val_data, all_blocks, grid, config):
    """
    Evaluation metrics:

    Per-level metrics:
        - Quadrant activation accuracy: % of quadrants correctly classified
        - Quadrant activation recall: % of TRUE active quadrants predicted as active
          (CRITICAL — must be >99%)
        - Quadrant activation precision: % of predicted active quadrants that are true

    End-to-end metrics:
        - Corridor recall: % of true Level-1 corridor blocks in predicted corridor
        - Corridor precision: % of predicted corridor blocks that are true
        - Corridor ratio: predicted corridor size / total blocks
        - Path optimality: for HYBRID mode, path_cost / bfs_optimal_cost
          (should be 1.0 when recall is perfect)

    Breakdown by level:
        Report all metrics separately for each hierarchy level.
        Early levels (top) are easier; deeper levels are harder.
    """
```

---

## 11. Module 8: Inference Pipeline and Modes

**File:** `lancet/pipeline.py`

### 11.1 Mode: MATRIX_ONLY

```python
def run_matrix_only(grid: Grid, source: Cell, goal: Cell,
                    config: Config) -> PathResult:
    """
    Full algebraic method. No neural network.

    1. Pad grid
    2. Partition into Level-1 blocks
    3. Compute ALL Level-1 transfer matrices (BFS)
    4. Build hierarchy, compose ALL higher-level transfer matrices
    5. Compute entry/exit embeddings for source/goal blocks
    6. Assemble corridor TM (corridor = all blocks)
    7. Query optimal distance
    8. Reconstruct path

    Properties:
    - GUARANTEED OPTIMAL
    - No training needed
    - Computes ALL blocks — slowest mode
    """
```

### 11.2 Mode: NEURAL_ONLY

```python
def run_neural_only(grid: Grid, source: Cell, goal: Cell,
                    model: LANCETNeuralPredictor,
                    config: Config) -> PathResult:
    """
    Neural prediction only. No transfer matrices.

    1. Compute top-level entry/exit embeddings (BFS to top block boundary)
    2. Run recursive corridor prediction top-down
    3. At Level 1, the activated blocks form the corridor
    4. Use a simple greedy search WITHIN the corridor to find approximate path:
       - Only explore cells in activated Level-1 blocks
       - Use Manhattan distance as heuristic
       - A* restricted to corridor blocks
    5. Return path (NOT guaranteed optimal — corridor may be incomplete,
       and we haven't computed exact distances)

    Properties:
    - FASTEST mode
    - NOT guaranteed optimal
    - Useful for real-time applications / initial estimates
    """
```

### 11.3 Mode: HYBRID (The Novel Method)

```python
def run_hybrid(grid: Grid, source: Cell, goal: Cell,
               model: LANCETNeuralPredictor,
               config: Config) -> PathResult:
    """
    THE FULL NOVEL METHOD: Neural corridor prediction + algebraic exact path.

    PHASE 1: NEURAL CORRIDOR PREDICTION (top-down)
    ─────────────────────────────────────────────────
    1a. Compute top-level entry/exit embeddings
        - BFS from source to top block's boundary → entry distances
        - BFS from goal to top block's boundary → exit distances
        - Project through model.entry_projector, model.exit_projector → (d,) each
        Time: O(H*W) for two BFS runs, ~10ms for 1024×1024

    1b. Recursive top-down prediction
        - Start at top level with entry_emb, exit_emb
        - At each level, for each active block:
            - model.predict_single_level(entry_emb, exit_emb, level)
            - → activations (4 quadrant probabilities), boundary embeddings
            - For each quadrant with activation > threshold:
                - model.derive_child_context(parent_entry, parent_exit,
                                             horiz_emb, vert_emb, child_idx)
                - → child_entry, child_exit
                - Recurse into that child
        - Collect all activated Level-1 blocks = predicted corridor
        Time: O(corridor_size * model_forward) ≈ O(C * d^2) where C = corridor blocks

    1c. Safety measures
        - Force-include source and goal blocks
        - Dilate corridor by `dilation` blocks (add neighbors)
        Time: O(corridor_size)

    PHASE 2: CONFIDENCE CHECK
    ─────────────────────────────
    2a. Examine corridor statistics
        - corridor_fraction = num_corridor_blocks / total_blocks
        - If corridor_fraction > 0.8: fall back to MATRIX_ONLY
          (corridor too broad, neural net isn't helping)
        - If corridor_fraction < 0.01: fall back to MATRIX_ONLY
          (corridor suspiciously small, probably missing blocks)

    PHASE 3: ALGEBRAIC EXACT PATH (bottom-up within corridor)
    ────────────────────────────────────────────────────────────
    3a. Compute Level-1 transfer matrices for corridor blocks ONLY
        - For each activated Level-1 block: run BFS between boundary pairs
        - Skip all non-corridor blocks (THE KEY SPEEDUP)
        Time: O(corridor_blocks * B^3) instead of O(all_blocks * B^3)

    3b. Assemble corridor transfer matrix
        - Build combined boundary graph for all corridor blocks
        - Floyd-Warshall to compute all-pairs shortest paths
        - Extract source_boundary → goal_boundary submatrix
        Time: O((corridor_boundary_cells)^3)

    PHASE 4: PATH EXTRACTION
    ─────────────────────────
    4a. Entry/exit embeddings (exact, algebraic)
        - BFS from source to source_block boundary → entry distances
        - BFS from goal to goal_block boundary → exit distances

    4b. Optimal distance
        - distance = min_{i,j} (entry[i] + corridor_tm[i][j] + exit[j])

    4c. Path reconstruction
        - Trace argmins through corridor TM
        - BFS within each block for cell-level path segments
        - Concatenate

    PHASE 5: OPTIONAL VERIFICATION
    ─────────────────────────────────
    5a. If config.verify_optimality:
        - Run BFS on full grid
        - Compare costs
        - If different: log warning, return BFS path
        Time: O(H*W), defeats the purpose, use only for debugging

    Returns: PathResult with mode="hybrid", optimal=True (if corridor recall is perfect)
    """
```

### 11.4 Adaptive Mode Selection

```python
def run_adaptive(grid: Grid, source: Cell, goal: Cell,
                 model: LANCETNeuralPredictor,
                 config: Config) -> PathResult:
    """
    Automatically selects the best mode.

    1. If grid is small (< 128×128): use MATRIX_ONLY
    2. If no model loaded: use MATRIX_ONLY
    3. Run HYBRID
    4. If HYBRID's confidence check fails: fallback to MATRIX_ONLY
    5. Return result
    """
```

---

## 12. Module 9: Visualization

**File:** `lancet/viz.py`

### 12.1 Grid + Path

```python
def visualize_grid_path(grid, path, source, goal, title="", save_path=None):
    """
    Grid: white=free, black=blocked
    Path: blue line
    Source: green dot, Goal: red dot
    """
```

### 12.2 Corridor Overlay

```python
def visualize_corridor(grid, corridor_blocks, true_path=None,
                        block_size=16, confidence=None, save_path=None):
    """
    Grid in grayscale + corridor blocks highlighted.
    Green = activated, intensity = confidence.
    True path in blue if provided.
    Block grid lines in gray.
    """
```

### 12.3 Hierarchical Corridor (Multi-Level)

```python
def visualize_hierarchical_corridor(grid, predictions_by_level: Dict[int, Set],
                                     block_size=16, save_path=None):
    """
    Show corridor predictions at each level side-by-side.
    Level L: 1 block, Level L-1: up to 4, ..., Level 1: many.
    Visualize how the corridor narrows top-down.
    """
```

### 12.4 Transfer Matrix Heatmap

```python
def visualize_transfer_matrix(block, save_path=None):
    """Heatmap of transfer matrix. Dark=short, light=long, white=inf."""
```

### 12.5 Training Curves

```python
def plot_training_curves(log_path, save_path=None):
    """
    Plot: activation loss, cell prediction loss, corridor recall,
    corridor precision, per-level accuracy — all vs epoch.
    """
```

---

## 13. Configuration System

**File:** `lancet/config.py`

```python
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class GridConfig:
    height: int = 256
    width: int = 256
    obstacle_density: float = 0.2
    seed: Optional[int] = None

@dataclass
class BlockConfig:
    block_size: int = 16

@dataclass
class NeuralConfig:
    d: int = 64                           # embedding dimension
    max_levels: int = 10                  # max hierarchy depth
    max_boundary_cells: int = 64          # max boundary cells per block edge
    checkpoint_path: str = "checkpoints/best.pt"

@dataclass
class TrainConfig:
    num_train: int = 50000
    num_val: int = 5000
    batch_size: int = 32                  # samples per batch (each processed level-by-level)
    epochs: int = 50
    learning_rate: float = 1e-3
    activation_loss_weight: float = 1.0
    cell_loss_weight: float = 0.5         # weight for boundary cell prediction loss
    pos_weight: float = 5.0               # weight for positive class in activation BCE
    warmup_epochs: int = 10               # scheduled sampling: all exact
    transition_epochs: int = 30           # scheduled sampling: ramp to max
    max_sampling_prob: float = 0.5        # scheduled sampling: max replacement prob
    curriculum_stages: List = field(default_factory=lambda: [
        {"epochs": 15, "grid_sizes": [64], "densities": [0.1, 0.2], "lr": 1e-3},
        {"epochs": 15, "grid_sizes": [64, 128], "densities": [0.1, 0.3], "lr": 5e-4},
        {"epochs": 20, "grid_sizes": [128, 256], "densities": [0.1, 0.35], "lr": 1e-4},
    ])

@dataclass
class InferenceConfig:
    mode: str = "hybrid"                  # "matrix_only", "neural_only", "hybrid"
    corridor_threshold: float = 0.5       # quadrant activation threshold
    corridor_dilation: int = 1            # safety margin in blocks
    max_corridor_fraction: float = 0.8    # above this, fall back to matrix_only
    min_corridor_fraction: float = 0.01   # below this, fall back to matrix_only
    verify_optimality: bool = False       # BFS verification (debug only)

@dataclass
class Config:
    grid: GridConfig = field(default_factory=GridConfig)
    block: BlockConfig = field(default_factory=BlockConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
```

---

## 14. Directory Structure

```
lancet/
│
├── lancet/
│   ├── __init__.py
│   ├── config.py                     # Configuration dataclasses
│   ├── grid.py                       # Grid, BFS, generation
│   ├── decomposition.py              # Block partitioning, hierarchy
│   ├── tropical.py                   # Tropical semiring, Level-1 TMs
│   ├── composition.py                # Hierarchical TM composition
│   ├── extraction.py                 # Path extraction and reconstruction
│   ├── pipeline.py                   # 3 inference modes + adaptive
│   ├── viz.py                        # All visualization
│   │
│   └── neural/
│       ├── __init__.py
│       ├── model.py                  # RecursiveCorridorPredictor
│       │                             # ChildEmbeddingDeriver
│       │                             # LANCETNeuralPredictor
│       ├── dataset.py                # TrainingSample, label generation
│       ├── train.py                  # Training loop, curriculum, scheduled sampling
│       └── losses.py                 # CorridorLoss, boundary cell loss
│
├── scripts/
│   ├── generate_data.py              # Generate training data (grids + BFS paths + labels)
│   ├── train.py                      # Launch training
│   ├── evaluate.py                   # Run evaluation
│   ├── demo.py                       # Interactive demo
│   └── benchmark.py                  # Timing benchmarks
│
├── tests/
│   ├── test_tropical.py
│   ├── test_grid.py
│   ├── test_decomposition.py
│   ├── test_composition.py
│   ├── test_extraction.py
│   ├── test_neural_shapes.py         # Verify tensor shapes through recursion
│   ├── test_neural_gradient_flow.py  # Verify gradients flow through all levels
│   ├── test_pipeline_matrix_only.py  # MATRIX_ONLY matches BFS
│   ├── test_pipeline_hybrid.py       # HYBRID with perfect corridor matches BFS
│   └── test_scheduled_sampling.py    # Verify sampling schedule
│
├── configs/
│   ├── default.yaml
│   ├── small_grid.yaml               # 64×64 for quick testing
│   └── large_grid.yaml               # 1024×1024 for benchmarks
│
├── checkpoints/
├── data/
├── results/
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 15. Testing Strategy

### 15.1 Tropical Algebra Tests

```python
def test_tropical_identity():
    """T ⊗ I = T"""

def test_tropical_matmul_hand_computed():
    """Verify against hand-computed 2×2 and 3×3 examples"""

def test_tropical_associativity():
    """(A ⊗ B) ⊗ C = A ⊗ (B ⊗ C)"""

def test_tropical_argmin_consistency():
    """Argmin matrix correctly identifies the minimizing k"""
```

### 15.2 Transfer Matrix Tests

```python
def test_empty_block_manhattan():
    """No obstacles: TM entries = Manhattan distances"""

def test_blocked_boundary():
    """Block with blocked boundary cells: TM has inf entries"""

def test_full_block_obstacle():
    """Completely blocked block: empty TM"""

def test_single_obstacle():
    """One obstacle: verify path goes around it"""
```

### 15.3 Composition Tests

```python
def test_compose_two_empty_blocks():
    """Two adjacent empty blocks: composed TM should match direct BFS on merged region"""

def test_compose_four_blocks_matches_bfs():
    """4-block composition matches BFS on the full 2×2-block region"""

def test_hierarchical_composition_full_grid():
    """Full hierarchy composition matches BFS for 64×64 grid"""
```

### 15.4 Neural Network Tests

```python
def test_model_output_shapes():
    """Verify output shapes for various input sizes"""
    model = LANCETNeuralPredictor(d=32, max_levels=6, max_boundary_cells=32)
    entry = torch.randn(1, 32)
    exit_ = torch.randn(1, 32)
    acts, h_emb, v_emb, logits = model.predict_single_level(entry, exit_, level=3)
    assert acts.shape == (1, 4)
    assert h_emb.shape == (1, 32)
    assert v_emb.shape == (1, 32)
    assert logits.shape == (1, 32)

def test_child_derivation_shapes():
    """Verify child embedding derivation produces correct shapes"""

def test_gradient_flows_through_levels():
    """Loss at Level 1 produces gradients in all model parameters"""
    # Create a 3-level hierarchy
    # Forward through all levels, compute loss, backward
    # Check that model.predictor.backbone[0].weight.grad is not None

def test_recursive_prediction_depth():
    """Run recursive prediction on 5-level hierarchy, verify correct number of Level-1 outputs"""
```

### 15.5 End-to-End Pipeline Tests

```python
def test_matrix_only_matches_bfs_many_grids():
    """100 random grids: MATRIX_ONLY cost == BFS cost"""

def test_hybrid_with_all_blocks_active():
    """HYBRID mode with corridor = all blocks should match MATRIX_ONLY exactly"""

def test_hybrid_with_true_corridor():
    """HYBRID mode with corridor = ground truth blocks should match BFS"""

def test_hybrid_with_dilated_corridor():
    """HYBRID with ground truth + dilation should still match BFS"""

def test_neural_only_finds_path():
    """NEURAL_ONLY mode produces a valid path (may not be optimal)"""

def test_adaptive_mode_selection():
    """Verify adaptive mode selects correctly for small/large grids"""
```

---

## 16. Performance Notes

### 16.1 Numba JIT for Tropical Operations

```python
@numba.njit(cache=True)
def tropical_matmul(A, B):
    ...  # 100x faster than pure Python
```

First call incurs JIT compilation overhead (~1s). Cache avoids this on subsequent runs.

### 16.2 Parallel Level-1 Computation

```python
from concurrent.futures import ProcessPoolExecutor

def compute_all_level1_parallel(blocks, grid, num_workers=8):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_level1_tm, grid, b): b for b in blocks}
        for future in as_completed(futures):
            block = futures[future]
            block.transfer_matrix = future.result()
```

### 16.3 Batched Neural Inference

At each level, all active blocks can be batched for a single forward pass:

```python
def batched_level_prediction(model, entries, exits, level):
    """
    entries: (B, d) — batch of entry embeddings for all active blocks at this level
    exits: (B, d) — batch of exit embeddings
    Returns batched outputs.
    """
    return model.predict_single_level(entries, exits, level)
```

This is MUCH faster than processing blocks one-by-one. At each level, batch all active blocks together.

### 16.4 Memory Estimates

For 1024×1024 grid, B=16:
- Level-1 blocks: 64×64 = 4096
- TM per block: ~60×60×8 = ~28KB
- All Level-1 TMs: ~115MB (only corridor in HYBRID: ~15-30MB)
- Neural model: ~1-5MB (small MLP, shared weights)
- Total in HYBRID: ~20-35MB

For 4096×4096:
- Level-1 blocks: 256×256 = 65,536
- All Level-1 TMs: ~1.8GB
- Corridor only (20%): ~360MB
- Still manageable on modern hardware

### 16.5 Block Size Tuning

| Block Size | Boundary Cells | TM Size | BFS Cost | Best For |
|---|---|---|---|---|
| 8 | ≤28 | 28×28 | Fast | Grids ≤256×256 |
| 16 | ≤60 | 60×60 | Moderate | Grids 256–2048 |
| 32 | ≤124 | 124×124 | Slow | Grids ≥2048 |

### 16.6 Neural Network Latency

The recursive prediction visits O(C) blocks where C = corridor size. Each visit is one MLP forward pass with input dimension 3d ≈ 192.

With d=64 on GPU: each forward pass ≈ 0.01ms.
For C=100 corridor blocks across all levels: ≈ 1ms total neural time.
The neural phase is negligible compared to the algebraic phase.

---

## 17. End-to-End Walkthrough

**Grid:** 128×128, 20% obstacles, block_size = 16
**Source:** (5, 10), **Goal:** (120, 115)
**Levels:** Level 1 (8×8 grid of 16×16 blocks), Level 2 (4×4), Level 3 (2×2), Level 4 (1×1 = whole grid)

### MATRIX_ONLY walkthrough:

1. Pad: 128 is fine (128/16 = 8, which is 2^3). No padding.
2. Partition: 8×8 = 64 Level-1 blocks
3. Compute 64 Level-1 TMs: 64 × BFS(~60 sources × 256 cells) ≈ 1M operations
4. Level 2: 16 blocks, each composing 4 Level-1 blocks via Floyd-Warshall
5. Level 3: 4 blocks, Level 4: 1 block
6. Total composition: ~30M operations
7. Entry/exit embeddings: BFS from (5,10) and (120,115) to their block boundaries
8. Assemble corridor TM (all blocks), query optimal distance, reconstruct path
9. **Total time: ~50-100ms**

### HYBRID walkthrough:

1. **Neural phase:**
   - Top-level entry/exit: BFS from source/goal to Level-4 block boundary
   - Level 4: model predicts all 4 quadrants active (obvious for distant source/goal)
   - Level 3: model predicts 3 of 4 quadrants active in each Level-3 block
     (narrows from 4 to 3 per parent = ~12 blocks)
   - Level 2: model predicts 2-3 of 4 active per block (~8-12 blocks)
   - Level 1: ~20-25 blocks activated (out of 64)
   - **Neural time: ~2ms**

2. **Algebraic phase:**
   - Compute Level-1 TMs for 25 blocks (instead of 64): 40% of the work
   - Compose corridor
   - **Algebra time: ~25-40ms**

3. **Path extraction:** ~5ms

4. **Total time: ~30-45ms** (vs 50-100ms for MATRIX_ONLY) = 2-3x speedup

On larger grids (1024×1024), the corridor is typically 10-15% of blocks, giving 5-10x speedup.

### NEURAL_ONLY walkthrough:

1. Neural phase: same as HYBRID step 1 → ~2ms
2. A* restricted to corridor blocks → ~5ms
3. **Total: ~7ms** but NOT guaranteed optimal

---

## Appendix A: Key Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Neural architecture | Recursive shared-weight MLP | Generalizes across grid sizes and depths; constant model size |
| Image model (U-Net)? | NO | Doesn't scale to large grids; doesn't leverage hierarchy |
| How info flows down | Learned boundary embeddings | Network learns to compress spatial info; no hand-coded routing |
| Scheduled sampling | Yes, ramp 0→0.5 | Prevents train/inference mismatch; network sees its own errors |
| Boundary cell head | Training only, discarded at inference | Auxiliary loss shapes embeddings; no cost at inference |
| Corridor safety | Dilation + confidence fallback | Rare missed blocks cause suboptimality; cheap to add margin |
| Composition method | Floyd-Warshall (initial) | Simpler than Schur complement; optimize later |
| Transfer matrix storage | Compute on demand (HYBRID) | Avoids storing all TMs; only corridor blocks computed |

## Appendix B: What Each Mode Tests

| Concern | MATRIX_ONLY | NEURAL_ONLY | HYBRID |
|---|---|---|---|
| Tropical algebra correct? | ✅ Primary | ❌ Not used | ✅ Used |
| Composition correct? | ✅ Primary | ❌ Not used | ✅ Used |
| Neural prediction useful? | ❌ Not used | ✅ Primary | ✅ Used |
| Path guaranteed optimal? | ✅ Always | ❌ Never | ✅ If recall=100% |
| Fast on large grids? | ❌ No | ✅ Yes | ✅ Yes |
| Needs training? | ❌ No | ✅ Yes | ✅ Yes |

## Appendix C: Failure Modes and Mitigations

| Failure | Cause | Mitigation |
|---|---|---|
| Suboptimal path in HYBRID | Neural net missed a corridor block (recall < 100%) | Dilation, confidence fallback, verify_optimality flag |
| Very slow HYBRID | Corridor too broad (>80% of blocks) | Fall back to MATRIX_ONLY |
| Wrong composition | Boundary cell ordering mismatch | Rigorous unit tests, clockwise convention |
| Training divergence | Scheduled sampling too aggressive | Warmup period, gradual ramp |
| OOM on large grids | Too many transfer matrices | Compute only corridor blocks; stream Level-1 TMs |
| Neural net never trains | Gradients don't flow through levels | Test gradient flow; ensure loss at all levels |
