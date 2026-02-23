# Hierarchical Tropical Pathfinding (HTP)
## Complete Implementation Specification v1.0

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Definitions and Mathematical Foundations](#2-definitions-and-mathematical-foundations)
3. [Data Structures](#3-data-structures)
4. [Module 1: Grid Engine](#4-module-1-grid-engine)
5. [Module 2: Block Decomposition](#5-module-2-block-decomposition)
6. [Module 3: Tropical Transfer Matrix Engine](#6-module-3-tropical-transfer-matrix-engine)
7. [Module 4: Hierarchical Composition](#7-module-4-hierarchical-composition)
8. [Module 5: Neural Corridor Predictor](#8-module-5-neural-corridor-predictor)
9. [Module 6: Path Extraction](#9-module-6-path-extraction)
10. [Module 7: Training Pipeline](#10-module-7-training-pipeline)
11. [Module 8: Inference Pipeline & Modes](#11-module-8-inference-pipeline-and-modes)
12. [Module 9: Visualization](#12-module-9-visualization)
13. [Configuration System](#13-configuration-system)
14. [Directory Structure](#14-directory-structure)
15. [Testing Strategy](#15-testing-strategy)
16. [Performance Notes](#16-performance-notes)

---

## 1. Project Overview

### What This System Does

Given a 2D grid with obstacles, a source cell, and a goal cell, this system finds the shortest (optimal) path using a hierarchical decomposition of the grid into blocks, where shortest-path information within each block is encoded as a **tropical (min, +) semiring transfer matrix**, and a **neural network** optionally predicts which blocks lie on the corridor (reducing computation).

### Three Operating Modes

| Mode | Name | Description |
|------|------|-------------|
| `MODE_MATRIX_ONLY` | Algebraic Exact | Full hierarchical transfer matrix computation over ALL blocks. No neural network. Guarantees optimal path. Serves as correctness baseline. |
| `MODE_NEURAL_ONLY` | Neural Heuristic | Neural network predicts the corridor and approximate path. No transfer matrices. Fast but NOT guaranteed optimal. Useful for latency-critical applications. |
| `MODE_HYBRID` | Neural + Algebraic (Full Method) | Neural network predicts corridor → transfer matrices computed ONLY within corridor → exact path extracted. Optimal when corridor prediction has perfect recall. This is the primary novel method. |

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

**Movement model:** 4-connected (up, down, left, right). Each move costs `1.0`. Diagonal movement is NOT supported in this specification (extending to 8-connected or weighted grids is discussed in notes).

**Cell indexing:** A cell is identified by `(row, col)` with `row` in `[0, H)` and `col` in `[0, W)`. Row 0 is the top of the grid.

### 2.2 Tropical Semiring

The **tropical semiring** (also called the min-plus semiring) is the algebraic structure `(R ∪ {∞}, ⊕, ⊗)` where:
- `a ⊕ b = min(a, b)` (tropical addition)
- `a ⊗ b = a + b` (tropical multiplication)
- Additive identity: `∞` (since `min(a, ∞) = a`)
- Multiplicative identity: `0` (since `a + 0 = a`)

### 2.3 Transfer Matrix

For a rectangular region (block) of the grid with `n` **boundary cells** (free cells on the border of the region), the **transfer matrix** `T` is an `n × n` matrix where:

```
T[i][j] = shortest path distance from boundary cell i to boundary cell j,
           using ONLY cells inside this region
         = ∞ if no such path exists
```

**Critical property:** Given two adjacent blocks A and B with transfer matrices `T_A` and `T_B`, the combined transfer matrix for the merged region is computed via **tropical matrix multiplication**:

```
(T_A ⊗ T_B)[i][j] = min over k of (T_A[i][k] + T_B[k][j])
```

where `k` ranges over the **shared boundary cells** between A and B.

### 2.4 Boundary Cells

A **boundary cell** of a block is any free (traversable) cell that lies on the edge of the block. Specifically, for a block occupying rows `[r_start, r_end)` and columns `[c_start, c_end)`:

- **Top boundary:** cells at row `r_start`, columns `[c_start, c_end)`
- **Bottom boundary:** cells at row `r_end - 1`, columns `[c_start, c_end)`
- **Left boundary:** cells at column `c_start`, rows `[r_start, r_end)`
- **Right boundary:** cells at column `c_end - 1`, rows `[r_start, r_end)`

Corner cells belong to TWO boundaries (e.g., top-left cell is both top and left).

Only **free** cells on these edges count as boundary cells. Blocked cells on edges are excluded.

### 2.5 Hierarchical Levels

The grid is recursively partitioned:

- **Level 0:** Individual grid cells. No transfer matrices needed at this level.
- **Level 1:** The grid is divided into `B × B` blocks (e.g., `16 × 16`). Each block has a transfer matrix computed by running shortest-path (BFS, since uniform cost) between all pairs of its boundary cells.
- **Level 2:** Groups of `2 × 2` Level-1 blocks form a Level-2 block. Its transfer matrix is computed by composing the four Level-1 transfer matrices.
- **Level L:** Groups of `2 × 2` Level-(L-1) blocks form a Level-L block. Continue until the entire grid is one block.

Number of levels: `L = 1 + ceil(log2(max(H, W) / B))`

### 2.6 Corridor

The **corridor** is the set of blocks (at some hierarchical level) that the shortest path passes through. In `MODE_HYBRID`, the neural network predicts this set, and transfer matrices are computed only for corridor blocks.

---

## 3. Data Structures

### 3.1 `Cell`

```python
@dataclass
class Cell:
    row: int
    col: int
```

Hashable. Comparable. Used as dictionary keys.

### 3.2 `Grid`

```python
class Grid:
    data: np.ndarray          # shape (H, W), dtype np.uint8, 0=free 1=blocked
    height: int
    width: int

    def is_free(self, r: int, c: int) -> bool
    def neighbors(self, r: int, c: int) -> List[Tuple[int, int]]  # 4-connected free neighbors
    def in_bounds(self, r: int, c: int) -> bool
```

### 3.3 `Block`

```python
@dataclass
class Block:
    block_id: Tuple[int, int, int]     # (level, block_row, block_col)
    level: int
    row_start: int                      # inclusive, in grid coordinates
    row_end: int                        # exclusive
    col_start: int                      # inclusive
    col_end: int                        # exclusive
    boundary_cells: List[Cell]          # ordered list of free boundary cells
    boundary_cell_to_index: Dict[Cell, int]  # reverse lookup
    transfer_matrix: np.ndarray         # shape (n, n), dtype float64, ∞ for no path
    children: Optional[List[Block]]     # 4 children if level > 1, None if level 1
    is_active: bool                     # whether this block is in the predicted corridor
```

**Boundary cell ordering convention (CRITICAL):**

Boundary cells MUST be ordered consistently to ensure transfer matrix composition works. The ordering is:

1. **Top boundary:** left to right → cells `(r_start, c)` for `c` in `[c_start, c_end)`, if free
2. **Right boundary:** top to bottom → cells `(r, c_end-1)` for `r` in `[r_start+1, r_end-1)`, if free (skip corners already counted)
3. **Bottom boundary:** right to left → cells `(r_end-1, c)` for `c` in `[c_end-1, ..., c_start]` (reverse), if free (skip corners already counted)
4. **Left boundary:** bottom to top → cells `(r, c_start)` for `r` in `[r_end-2, ..., r_start+1]` (reverse), if free (skip corners already counted)

This clockwise ordering ensures that shared boundaries between adjacent blocks have cells listed in **reverse order** relative to each other, which is essential for correct matrix composition.

**IMPORTANT CORRECTION on shared boundaries:**

When block A is to the LEFT of block B:
- A's right boundary cells are ordered top-to-bottom
- B's left boundary cells are ordered bottom-to-top
- These are the SAME physical cells, listed in REVERSE order
- When composing, you must align them (reverse one list)

When block A is ABOVE block B:
- A's bottom boundary cells are ordered right-to-left
- B's top boundary cells are ordered left-to-right
- Same reversal situation

The composition code MUST handle this reversal. See Module 4.

### 3.4 `TransferMatrix`

```python
class TransferMatrix:
    matrix: np.ndarray           # (n, n), float64
    boundary_cells: List[Cell]   # which cells the rows/columns correspond to
    cell_to_idx: Dict[Cell, int]

    @staticmethod
    def identity(n: int) -> 'TransferMatrix':
        """Multiplicative identity: 0 on diagonal, ∞ elsewhere"""
        m = np.full((n, n), np.inf)
        np.fill_diagonal(m, 0.0)
        return TransferMatrix(m, ...)

    @staticmethod
    def tropical_multiply(A: 'TransferMatrix', B: 'TransferMatrix',
                          shared_cells: List[Cell]) -> 'TransferMatrix':
        """See Module 4 for full algorithm"""
        ...
```

### 3.5 `Corridor`

```python
@dataclass
class Corridor:
    active_blocks: Set[Tuple[int, int, int]]  # set of block_ids in the corridor
    confidence: Dict[Tuple[int, int, int], float]  # per-block confidence
    level: int  # which hierarchical level this corridor is defined at
```

### 3.6 `PathResult`

```python
@dataclass
class PathResult:
    path: List[Cell]               # ordered list of cells from source to goal
    cost: float                    # total path cost
    optimal: bool                  # whether this is guaranteed optimal
    mode: str                      # which mode produced this
    corridor_size: int             # how many blocks were in the corridor
    total_blocks: int              # total blocks at the corridor level
    computation_time_ms: float     # wall-clock time
```

---

## 4. Module 1: Grid Engine

**File:** `htp/grid.py`

### 4.1 Grid Generation

```python
def generate_grid(height: int, width: int, obstacle_density: float,
                  seed: Optional[int] = None) -> Grid:
    """
    Generate a random grid.

    Args:
        height: number of rows
        width: number of columns
        obstacle_density: float in [0, 1), fraction of cells that are blocked
        seed: random seed for reproducibility

    Returns:
        Grid object

    Algorithm:
        1. Create HxW array of zeros (all free)
        2. Randomly set cells to 1 with probability = obstacle_density
        3. ALWAYS keep (0,0) and (H-1, W-1) free (common source/goal)
        4. Optionally verify connectivity between (0,0) and (H-1,W-1) via BFS
           If not connected and retries < max_retries, regenerate
    """
```

### 4.2 Grid I/O

```python
def save_grid(grid: Grid, filepath: str) -> None:
    """Save grid as .npy file"""

def load_grid(filepath: str) -> Grid:
    """Load grid from .npy file"""
```

### 4.3 BFS Shortest Path (Reference Implementation)

```python
def bfs_shortest_path(grid: Grid, source: Cell, goal: Cell) -> Optional[Tuple[List[Cell], float]]:
    """
    Standard BFS shortest path on uniform-cost grid.
    Returns (path, cost) or None if no path exists.

    This is used:
    1. To compute transfer matrices at Level 1 (between boundary cell pairs)
    2. As ground truth for training data generation
    3. As correctness verification oracle

    Algorithm:
        Standard BFS with parent tracking.
        Queue: collections.deque
        Visited: set or 2D boolean array (prefer array for speed)
        Parent: dict mapping cell -> predecessor cell

    IMPORTANT: BFS must be RESTRICTED to within a block's boundaries when
    computing transfer matrices. Accept optional bounds parameters:
    """

def bfs_within_block(grid: Grid, source: Cell, goal: Cell,
                     row_start: int, row_end: int,
                     col_start: int, col_end: int) -> Optional[float]:
    """
    BFS restricted to cells within [row_start, row_end) x [col_start, col_end).
    Returns only the distance (float), not the path.
    Used for transfer matrix computation.
    """
```

### 4.4 Multi-Source BFS (Optimization)

```python
def multi_source_bfs_within_block(grid: Grid, sources: List[Cell],
                                   row_start: int, row_end: int,
                                   col_start: int, col_end: int) -> np.ndarray:
    """
    Run BFS from EACH source to ALL other sources, within block bounds.
    Returns distance matrix of shape (len(sources), len(sources)).

    This is more efficient than calling bfs_within_block for every pair,
    because each single-source BFS finds distances to ALL other boundary cells.

    For a block with n boundary cells, this requires n BFS runs instead of n^2.
    Each BFS is O(B^2) for a BxB block, so total is O(n * B^2).
    Since n ≤ 4B, this is O(B^3).

    Algorithm:
        dist_matrix = np.full((n, n), np.inf)
        for i, src in enumerate(sources):
            distances = bfs_from_source(grid, src, bounds)  # returns dict or array
            for j, dst in enumerate(sources):
                dist_matrix[i][j] = distances.get(dst, np.inf)
        return dist_matrix
    """
```

---

## 5. Module 2: Block Decomposition

**File:** `htp/decomposition.py`

### 5.1 Level 1 Block Partitioning

```python
def partition_into_blocks(grid: Grid, block_size: int) -> List[List[Block]]:
    """
    Divide the grid into blocks of block_size x block_size.

    Args:
        grid: the full grid
        block_size: side length of each Level-1 block (e.g., 16)

    Returns:
        2D list of Block objects: blocks[block_row][block_col]
        where block_row in [0, ceil(H/block_size))
        and block_col in [0, ceil(W/block_size))

    Edge handling:
        If H or W is not divisible by block_size, the last row/column of
        blocks will be SMALLER than block_size. This is fine — boundary
        cell counts will just be smaller.

    IMPORTANT: The grid dimensions H and W should ideally be padded to the
    nearest power-of-2 multiple of block_size BEFORE partitioning, to ensure
    clean hierarchical composition. Padding cells are marked as blocked.

    Padding algorithm:
        padded_H = block_size * (2 ** ceil(log2(ceil(H / block_size))))
        padded_W = block_size * (2 ** ceil(log2(ceil(W / block_size))))
        padded_grid = np.ones((padded_H, padded_W), dtype=np.uint8)  # all blocked
        padded_grid[:H, :W] = grid.data  # copy original

    For each block:
        1. Compute row_start, row_end, col_start, col_end
        2. Enumerate boundary cells (see ordering in Section 3.3)
        3. Create Block object with empty transfer_matrix (filled later)
    """
```

### 5.2 Boundary Cell Enumeration

```python
def enumerate_boundary_cells(grid: Grid, row_start: int, row_end: int,
                              col_start: int, col_end: int) -> List[Cell]:
    """
    Enumerate free boundary cells in clockwise order.

    Traversal order:
    1. Top edge: (row_start, c) for c in [col_start, col_end), left to right
    2. Right edge: (r, col_end-1) for r in [row_start+1, row_end-1), top to bottom
    3. Bottom edge: (row_end-1, c) for c in [col_end-1, ..., col_start], right to left
    4. Left edge: (r, col_start) for r in [row_end-2, ..., row_start+1], bottom to top

    Only include cell if grid.is_free(r, c).

    Handle degenerate cases:
    - 1-cell wide block: top and bottom are the same row. Don't double-count.
    - 1-cell tall block: similar.
    - Fully blocked boundary: return empty list.

    Returns: ordered list of Cell objects
    """
```

### 5.3 Hierarchical Block Tree

```python
def build_block_hierarchy(level1_blocks: List[List[Block]]) -> Dict[Tuple[int,int,int], Block]:
    """
    Build the full hierarchical block tree.

    Args:
        level1_blocks: 2D grid of Level-1 blocks from partition_into_blocks

    Returns:
        Dictionary mapping block_id (level, row, col) to Block object
        for ALL levels.

    Algorithm:
        num_block_rows = len(level1_blocks)
        num_block_cols = len(level1_blocks[0])
        # After padding, both should be powers of 2

        max_level = 1 + ceil(log2(max(num_block_rows, num_block_cols)))

        all_blocks = {}

        # Register Level 1
        for br in range(num_block_rows):
            for bc in range(num_block_cols):
                all_blocks[(1, br, bc)] = level1_blocks[br][bc]

        # Build higher levels
        for level in range(2, max_level + 1):
            prev_rows = ceil(num_block_rows / 2^(level-2))
            prev_cols = ceil(num_block_cols / 2^(level-2))
            curr_rows = ceil(prev_rows / 2)
            curr_cols = ceil(prev_cols / 2)

            for br in range(curr_rows):
                for bc in range(curr_cols):
                    # Children are 4 blocks at level-1:
                    children_ids = [
                        (level-1, 2*br,   2*bc),    # top-left
                        (level-1, 2*br,   2*bc+1),  # top-right
                        (level-1, 2*br+1, 2*bc),    # bottom-left
                        (level-1, 2*br+1, 2*bc+1),  # bottom-right
                    ]
                    children = [all_blocks.get(cid) for cid in children_ids]
                    # Some children may be None if grid isn't perfectly square

                    # Compute merged bounds
                    row_start = min(c.row_start for c in children if c)
                    row_end = max(c.row_end for c in children if c)
                    col_start = min(c.col_start for c in children if c)
                    col_end = max(c.col_end for c in children if c)

                    # Boundary cells of the merged block
                    boundary = enumerate_boundary_cells(grid, row_start, row_end,
                                                         col_start, col_end)

                    block = Block(
                        block_id=(level, br, bc),
                        level=level,
                        row_start=row_start, row_end=row_end,
                        col_start=col_start, col_end=col_end,
                        boundary_cells=boundary,
                        boundary_cell_to_index={c: i for i, c in enumerate(boundary)},
                        transfer_matrix=None,  # computed later
                        children=[c for c in children if c],
                        is_active=True
                    )
                    all_blocks[(level, br, bc)] = block

        return all_blocks
    """
```

---

## 6. Module 3: Tropical Transfer Matrix Engine

**File:** `htp/tropical.py`

This is the mathematical core. Correctness here is NON-NEGOTIABLE.

### 6.1 Tropical Operations

```python
INF = float('inf')

def tropical_add(a: float, b: float) -> float:
    """min(a, b)"""
    return min(a, b)

def tropical_multiply(a: float, b: float) -> float:
    """a + b, with inf + anything = inf"""
    if a == INF or b == INF:
        return INF
    return a + b
```

### 6.2 Tropical Matrix Multiplication

```python
@numba.njit  # JIT compile for performance
def tropical_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Tropical matrix multiplication: C[i][j] = min_k (A[i][k] + B[k][j])

    Args:
        A: shape (m, p), float64
        B: shape (p, n), float64

    Returns:
        C: shape (m, n), float64

    MUST handle inf correctly: inf + x = inf, min(inf, x) = x

    Implementation:
        C = np.full((m, n), np.inf)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    val = A[i, k] + B[k, j]  # works with inf in float64
                    if val < C[i, j]:
                        C[i, j] = val
        return C

    Performance note: For large matrices, consider blocked/tiled matrix
    multiplication. For typical boundary sizes (≤64), the naive O(n^3) is fine.
    Numba JIT compilation makes this fast enough.
    """
```

### 6.3 Level-1 Transfer Matrix Computation

```python
def compute_level1_transfer_matrix(grid: Grid, block: Block) -> np.ndarray:
    """
    Compute the transfer matrix for a Level-1 block by running BFS.

    Args:
        grid: the full grid
        block: a Level-1 Block object with boundary_cells already set

    Returns:
        np.ndarray of shape (n, n) where n = len(block.boundary_cells)

    Algorithm:
        n = len(block.boundary_cells)
        if n == 0:
            return np.empty((0, 0))

        T = np.full((n, n), np.inf)

        for i in range(n):
            src = block.boundary_cells[i]
            # Run BFS from src within the block bounds
            distances = bfs_all_distances_within_block(
                grid, src,
                block.row_start, block.row_end,
                block.col_start, block.col_end
            )
            for j in range(n):
                dst = block.boundary_cells[j]
                if dst in distances:
                    T[i][j] = distances[dst]

        # T[i][i] should be 0 (distance from cell to itself)
        # Verify this as a sanity check

        return T

    Complexity: O(n * B^2) where B = block_size, n ≤ 4*B
    For B=16: n ≤ 64, cost = 64 * 256 = ~16K operations per block
    """
```

```python
def bfs_all_distances_within_block(grid: Grid, source: Cell,
                                    row_start: int, row_end: int,
                                    col_start: int, col_end: int) -> Dict[Cell, float]:
    """
    BFS from source, restricted to block bounds. Returns distances to ALL reachable cells.

    Uses a deque. Only enqueues cells within [row_start, row_end) x [col_start, col_end)
    that are free.

    Returns dict: Cell -> distance (float, but actually int since uniform cost)
    """
```

---

## 7. Module 4: Hierarchical Composition

**File:** `htp/composition.py`

This is the most algorithmically complex module. It composes transfer matrices from child blocks into parent block transfer matrices.

### 7.1 Core Concept

A Level-L block has 4 children (Level-(L-1) blocks) arranged as:

```
+-------+-------+
|  TL   |  TR   |
| (0,0) | (0,1) |
+-------+-------+
|  BL   |  BR   |
| (1,0) | (1,1) |
+-------+-------+
```

The parent's transfer matrix relates its OUTER boundary cells. The composition must "eliminate" all INTERNAL boundary cells (shared edges between children).

### 7.2 Shared Boundary Identification

```python
def identify_shared_boundaries(children: List[Block]) -> Dict[str, List[Cell]]:
    """
    Identify the internal (shared) boundaries between the 4 children.

    Returns dict with keys:
        'horizontal': shared boundary between top pair and bottom pair
                      = bottom boundary of TL/TR = top boundary of BL/BR
        'vertical':   shared boundary between left pair and right pair
                      = right boundary of TL/BL = left boundary of TR/BR
        'center':     the single cell (or few cells) at the intersection

    Shared horizontal boundary cells:
        Row = TL.row_end - 1 (= BL.row_start... wait, no)

    CORRECTION — Shared boundaries are BETWEEN blocks, not ON blocks:

    Actually in a grid, the "shared boundary" between TL and TR means:
    - TL's RIGHT boundary cells (column = TL.col_end - 1)
    - TR's LEFT boundary cells (column = TR.col_start)
    - These are ADJACENT cells (not the same cells) because TL.col_end = TR.col_start

    WAIT — this depends on how blocks are defined.

    If TL covers columns [0, 16) and TR covers columns [16, 32), then:
    - TL's right boundary is column 15
    - TR's left boundary is column 16
    - These are DIFFERENT cells, but adjacent (distance 1 apart)

    THIS IS THE KEY SUBTLETY.

    The shared boundary between adjacent blocks consists of PAIRS of adjacent cells,
    one from each block. The transfer between them has cost 1 (one step).

    So composition isn't just tropical matrix multiplication — we need to account
    for the crossing cost of 1 between adjacent boundary cells.
    """
```

### 7.3 Composition Algorithm (DETAILED)

This is the most important algorithm in the system. Here is the full procedure.

**Setup:** We have 4 child blocks TL, TR, BL, BR with transfer matrices `T_TL, T_TR, T_BL, T_BR`. We want to compute the parent's transfer matrix `T_parent`.

**Step 1: Identify all boundary cell categories**

For each child block, classify its boundary cells into:
- **External cells:** cells on the outer boundary of the parent block
- **Internal cells:** cells on shared edges with sibling blocks

```python
def classify_boundary_cells(parent: Block, child: Block) -> Tuple[List[int], List[int]]:
    """
    For a child block, determine which of its boundary cells are external
    (on the parent's boundary) vs internal (shared with siblings).

    Returns:
        external_indices: indices into child.boundary_cells that are external
        internal_indices: indices into child.boundary_cells that are internal
    """
    external = []
    internal = []
    parent_boundary_set = set((c.row, c.col) for c in parent.boundary_cells)

    for i, cell in enumerate(child.boundary_cells):
        if (cell.row, cell.col) in parent_boundary_set:
            external.append(i)
        else:
            internal.append(i)

    return external, internal
```

**Step 2: Build the combined internal distance matrix**

Create a single large matrix that encodes:
- Distances within each child block (from their transfer matrices)
- Crossing distances between adjacent children (cost = 1 for each grid step)

```python
def build_combined_matrix(children: List[Block], grid: Grid,
                          parent: Block) -> Tuple[np.ndarray, List[Cell], List[int]]:
    """
    Build a combined distance matrix over ALL boundary cells of ALL children.

    Let N = total number of boundary cells across all 4 children.
    Build an N×N matrix M where:

    For cells i, j in the SAME child block:
        M[i][j] = child's transfer matrix entry

    For cells i, j in DIFFERENT child blocks:
        M[i][j] = 1 if cells i and j are adjacent in the grid
                           (i.e., they are on shared boundaries and one step apart)
        M[i][j] = ∞ otherwise (will be resolved through composition)

    Also return:
        all_cells: the ordered list of all boundary cells
        external_mask: which indices are external (parent boundary) cells

    NOTE: This builds the full combined matrix. The composition then
    eliminates internal nodes via tropical matrix closure operations.
    """
```

**Step 3: Eliminate internal cells via tropical Floyd-Warshall**

```python
def compose_transfer_matrices(children: List[Block], grid: Grid,
                               parent: Block) -> np.ndarray:
    """
    Full composition algorithm.

    1. Build combined matrix M over all children's boundary cells
    2. Run Floyd-Warshall (tropical all-pairs shortest path) on M
       This resolves all shortest paths through internal shared boundaries
    3. Extract the submatrix corresponding to ONLY the parent's external
       boundary cells

    Floyd-Warshall in tropical semiring:
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    M[i][j] = min(M[i][j], M[i][k] + M[k][j])

    This is O(N^3) where N = total boundary cells of all children.
    For 4 children each with ≤ 4B boundary cells (B = block_size):
        N ≤ 16B, so cost = O(B^3)
    For B=16: N ≤ 256, cost = ~16M operations (fast with Numba)

    4. Extract: identify which rows/columns of M correspond to parent's
       boundary cells. Return that submatrix.

    CRITICAL: Must maintain correct cell-to-index mapping throughout.
    """
```

**Alternative approach — Schur complement (more efficient but harder):**

Instead of Floyd-Warshall on the full matrix, use the tropical Schur complement to directly eliminate internal variables. This is mathematically equivalent but can be faster for sparse internal boundaries. Implement Floyd-Warshall first, optimize later.

### 7.4 Recursive Composition

```python
def compute_all_transfer_matrices(all_blocks: Dict, grid: Grid, block_size: int,
                                   active_only: bool = False,
                                   active_set: Optional[Set] = None) -> None:
    """
    Compute transfer matrices for all blocks, bottom-up.

    Args:
        all_blocks: the full block hierarchy
        grid: the grid
        block_size: Level-1 block size
        active_only: if True, only compute for blocks in active_set
        active_set: set of block_ids to compute (used in HYBRID mode)

    Algorithm:
        1. Compute Level-1 transfer matrices using BFS (Module 3)
        2. For each level from 2 to max_level:
            For each block at this level:
                If active_only and block not in active_set: skip
                Compose from children's transfer matrices

    This is the main computation and is the bottleneck.
    """
```

---

## 8. Module 5: Neural Corridor Predictor

**File:** `htp/neural/model.py`, `htp/neural/dataset.py`

### 8.1 Network Architecture

The neural network takes a grid + source/goal encoding and predicts a corridor mask over Level-1 blocks.

**Input tensor:** shape `(C, H, W)` where C = 3 channels:
- Channel 0: **Grid map** — the grid itself (0 = free, 1 = blocked), at full resolution
- Channel 1: **Source encoding** — Gaussian blob centered at source cell, σ = 2 * block_size. Or simpler: a single 1 at the source cell, 0 elsewhere.
- Channel 2: **Goal encoding** — same as source but for goal cell.

**NOTE:** If the grid is too large for the network (e.g., 4096×4096), downsample it to a fixed resolution (e.g., 256×256) and map predictions back to the original block grid.

**Architecture: U-Net with residual blocks**

```
Encoder:
    Conv2d(3, 32, 3, padding=1) + BatchNorm + ReLU
    ResBlock(32, 32)
    MaxPool2d(2)                          # H/2 × W/2

    Conv2d(32, 64, 3, padding=1) + BatchNorm + ReLU
    ResBlock(64, 64)
    MaxPool2d(2)                          # H/4 × W/4

    Conv2d(64, 128, 3, padding=1) + BatchNorm + ReLU
    ResBlock(128, 128)
    MaxPool2d(2)                          # H/8 × W/8

    Conv2d(128, 256, 3, padding=1) + BatchNorm + ReLU
    ResBlock(256, 256)

Decoder:
    Upsample(2) + Conv2d(256, 128, 3, padding=1) + skip connection from encoder
    ResBlock(256, 128)

    Upsample(2) + Conv2d(128, 64, 3, padding=1) + skip connection
    ResBlock(128, 64)

    Upsample(2) + Conv2d(64, 32, 3, padding=1) + skip connection
    ResBlock(64, 32)

Output head:
    Conv2d(32, 1, 1) → Sigmoid

Output: shape (1, H, W) — per-pixel probability that this cell is on the corridor
```

**ResBlock definition:**

```python
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
```

### 8.2 Pixel-to-Block Aggregation

The network outputs per-pixel predictions. Convert to per-block predictions:

```python
def pixel_predictions_to_block_corridor(pixel_probs: np.ndarray,
                                         block_size: int,
                                         threshold: float = 0.3) -> Tuple[Set, Dict]:
    """
    Convert pixel-level corridor probability map to block-level decisions.

    Args:
        pixel_probs: (H, W) array of per-pixel corridor probabilities
        block_size: Level-1 block size
        threshold: probability threshold for including a block

    For each Level-1 block:
        block_prob = max of pixel_probs within that block's region
        (Using max rather than mean because we want high recall —
         if ANY pixel in a block is likely on the corridor, include the block)

    Alternative aggregation strategies:
        - max (highest recall, recommended)
        - mean (balanced)
        - fraction > 0.5 (counts confident pixels)

    Returns:
        active_blocks: set of (1, block_row, block_col) tuples
        confidence: dict mapping block_id -> aggregated probability
    """
```

### 8.3 Corridor Dilation

To ensure safety (high recall), dilate the predicted corridor:

```python
def dilate_corridor(active_blocks: Set, all_block_ids: Set,
                    dilation: int = 1) -> Set:
    """
    Add neighboring blocks to the corridor.

    For each active block (1, br, bc), also activate:
        (1, br+dr, bc+dc) for all (dr, dc) with |dr| <= dilation and |dc| <= dilation

    This is a safety margin. dilation=1 adds a 1-block border around the corridor.
    Critical for early training when the network isn't confident yet.
    """
```

### 8.4 Dataset

```python
class PathfindingDataset(torch.utils.data.Dataset):
    """
    Dataset for training the corridor predictor.

    Each sample contains:
        - input_tensor: (3, H, W) grid + source + goal encoding
        - target_mask: (1, H, W) binary mask — 1 for cells on the shortest path
                       (or cells within `margin` distance of the path)
        - block_target: (num_block_rows, num_block_cols) binary — 1 if block
                        contains any path cell

    Generation process (offline, saved to disk):
        For each training example:
        1. Generate or load a grid
        2. Sample random source and goal (both free cells, minimum distance apart)
        3. Run A* or BFS to find shortest path
        4. Create pixel mask: set path cells to 1
        5. Optionally dilate the pixel mask by `margin` pixels
        6. Create block mask: for each block, 1 if any path pixel is inside
        7. Store (grid, source, goal, pixel_mask, block_mask)
    """

    def __init__(self, data_dir: str, grid_size: int, block_size: int,
                 num_samples: int, obstacle_density: float,
                 path_margin: int = 2):
        ...

    def __getitem__(self, idx):
        # Load pre-generated sample or generate on-the-fly
        grid = ...
        source = ...
        goal = ...
        path = ...

        # Build input tensor
        input_tensor = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        input_tensor[0] = grid.data.astype(np.float32)
        input_tensor[1, source.row, source.col] = 1.0
        input_tensor[2, goal.row, goal.col] = 1.0

        # Build target mask
        target_mask = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        for cell in path:
            target_mask[0, cell.row, cell.col] = 1.0

        # Dilate target mask
        if self.path_margin > 0:
            from scipy.ndimage import binary_dilation
            struct = np.ones((2*self.path_margin+1, 2*self.path_margin+1))
            target_mask[0] = binary_dilation(target_mask[0], structure=struct).astype(np.float32)

        return torch.tensor(input_tensor), torch.tensor(target_mask)
```

### 8.5 Loss Function

```python
class CorridorLoss(nn.Module):
    """
    Weighted binary cross-entropy that prioritizes RECALL over precision.

    The corridor typically covers 5-20% of blocks. We need to NOT miss any
    corridor block (false negative = potentially suboptimal path), but can
    tolerate extra blocks (false positive = wasted computation, not wrong answer).

    Loss = -[w_pos * y * log(p) + w_neg * (1-y) * log(1-p)]

    where w_pos >> w_neg. Typical: w_pos = 5.0, w_neg = 1.0

    Alternatively, use focal loss to focus on hard examples:
    FL(p) = -alpha * (1-p)^gamma * log(p)  for positive
    FL(p) = -(1-alpha) * p^gamma * log(1-p)  for negative
    with alpha=0.75, gamma=2.0
    """

    def __init__(self, pos_weight: float = 5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(
            pred, target,
            weight=target * self.pos_weight + (1 - target) * 1.0,
            reduction='mean'
        )
        return bce
```

---

## 9. Module 6: Path Extraction

**File:** `htp/extraction.py`

Once we have the top-level transfer matrix, we need to extract the actual cell-by-cell path.

### 9.1 Top-Level: Source-to-Goal Distance

```python
def query_path_distance(source: Cell, goal: Cell,
                        all_blocks: Dict, grid: Grid,
                        corridor: Optional[Corridor] = None) -> float:
    """
    Find the shortest path distance from source to goal.

    Step 1: Identify which Level-1 blocks contain source and goal.
        source_block_id = (1, source.row // block_size, source.col // block_size)
        goal_block_id = (1, goal.row // block_size, goal.col // block_size)

    Step 2: Compute entry/exit embeddings.
        For the SOURCE block:
            Run BFS from source to all boundary cells of source's block.
            This gives a vector: entry_embed[i] = dist(source, boundary_cell_i)

        For the GOAL block:
            Run BFS from goal to all boundary cells of goal's block.
            This gives a vector: exit_embed[j] = dist(boundary_cell_j, goal)

    Step 3: Use the hierarchical transfer matrices.
        The full-grid transfer matrix T_full relates boundary cells of the
        top-level block (= the entire grid's boundary). But we need distances
        between INTERIOR cells (source and goal).

        APPROACH: Use a "corridor composition."

        Actually, the correct approach is:

        We need to compose:
            entry_embed → T_corridor → exit_embed

        where T_corridor is the transfer matrix from source block's boundary
        to goal block's boundary, through the corridor.

        distance = min over (i, j) of:
            entry_embed[i] + T_corridor[i][j] + exit_embed[j]

        This is a double tropical matrix-vector product.

    Returns: shortest path distance (float)
    """
```

### 9.2 Corridor Transfer Matrix Computation

```python
def compute_corridor_transfer_matrix(source_block: Block, goal_block: Block,
                                      corridor: Corridor,
                                      all_blocks: Dict,
                                      grid: Grid) -> np.ndarray:
    """
    Compute the transfer matrix from source_block's boundary to
    goal_block's boundary, through the corridor blocks.

    This is the KEY inference computation.

    Algorithm:
    1. Identify ALL corridor blocks at Level 1.
    2. Ensure all Level-1 transfer matrices are computed for corridor blocks.
    3. Determine the composition ORDER — which blocks to compose and in what sequence.

    The composition order follows the spatial layout:
    - Sort corridor blocks by their position
    - Compose row-by-row, then compose rows together

    SIMPLIFIED APPROACH (for initial implementation):
    Use Floyd-Warshall on a graph where:
    - Nodes = all boundary cells of all corridor blocks
    - Edge weights = transfer matrix entries (within-block) and
                     crossing costs (between adjacent blocks, cost = 1)

    This avoids the complexity of hierarchical composition order and
    gives the same result. It's slower (O(N^3) where N = total boundary
    cells in corridor) but correct and simple.

    For N corridor blocks each with ~4B boundary cells:
        Total nodes = ~4BN
        Floyd-Warshall cost = O((4BN)^3)
        For B=16, N=20 corridor blocks: ~(1280)^3 ≈ 2 billion ops
        This is ~2 seconds. Acceptable for prototyping, optimize later.

    OPTIMIZED APPROACH (implement after prototype works):
    Compose hierarchically:
    - At Level 1, compose adjacent corridor blocks left-to-right within each row
    - Then compose rows top-to-bottom
    - Use the hierarchical structure to skip non-corridor regions

    Returns: matrix of shape (source_boundary_count, goal_boundary_count)
    """
```

### 9.3 Path Reconstruction (Cell-by-Cell)

After finding the optimal boundary-to-boundary route, reconstruct the actual path:

```python
def reconstruct_full_path(source: Cell, goal: Cell,
                           optimal_boundary_sequence: List[Cell],
                           all_blocks: Dict, grid: Grid) -> List[Cell]:
    """
    Given the sequence of boundary cells the optimal path passes through,
    reconstruct the full cell-by-cell path.

    Args:
        source: source cell
        goal: goal cell
        optimal_boundary_sequence: ordered list of boundary cells that the
            optimal path crosses. Computed from the transfer matrix argmins.
        all_blocks: block hierarchy
        grid: the grid

    Algorithm:
    The optimal_boundary_sequence looks like:
        [source, b1_exit, b2_entry, b2_exit, b3_entry, ..., goal]

    where b1_exit and b2_entry are adjacent cells across a block boundary.

    For each consecutive pair (cell_a, cell_b) that are in the SAME block:
        Run BFS from cell_a to cell_b within that block.
        Append the path segment.

    For each consecutive pair (cell_a, cell_b) that are in ADJACENT blocks:
        They are grid-neighbors (distance 1). Just step from cell_a to cell_b.

    Concatenate all segments.

    Returns: complete cell-by-cell path from source to goal
    """
```

### 9.4 Waypoint Recovery from Transfer Matrices

To find optimal_boundary_sequence, we need to track WHICH intermediate boundary cells achieved the minimum during tropical matrix multiplication:

```python
def tropical_matmul_with_traceback(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tropical matrix multiplication that also records the argmin.

    Returns:
        C: the result matrix, C[i][j] = min_k (A[i][k] + B[k][j])
        argmin_k: matrix of same shape, argmin_k[i][j] = the k that achieved the min

    This enables path reconstruction by tracing back through the hierarchy.
    """
    m, p = A.shape
    _, n = B.shape
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
```

---

## 10. Module 7: Training Pipeline

**File:** `htp/neural/train.py`

### 10.1 Data Generation

```python
def generate_training_data(config: TrainConfig) -> None:
    """
    Pre-generate training examples and save to disk.

    Config parameters:
        num_train: 50000 (start), up to 500000
        num_val: 5000
        grid_sizes: [64, 128, 256]  # train on multiple sizes
        block_size: 16
        obstacle_densities: [0.1, 0.2, 0.3]  # vary difficulty
        min_path_distance: 20  # skip trivially short paths
        path_margin: 2  # dilate path for target mask

    For each example:
        1. Random grid_size from grid_sizes
        2. Random obstacle_density from obstacle_densities
        3. Generate grid
        4. Sample source, goal (both free, distance ≥ min_path_distance)
        5. BFS shortest path
        6. If no path: discard, regenerate
        7. Save: (grid, source, goal, path, path_cost)
    """
```

### 10.2 Curriculum Training

Train in stages, progressing from easy to hard:

```python
def train(config: TrainConfig) -> None:
    """
    Training loop with curriculum.

    Stage 1 (epochs 1-20):
        Grid size: 64×64 only
        Obstacle density: 0.1-0.2
        Learning rate: 1e-3
        Purpose: Learn basic spatial reasoning

    Stage 2 (epochs 21-40):
        Grid size: 64×64 and 128×128
        Obstacle density: 0.1-0.3
        Learning rate: 5e-4
        Purpose: Learn to generalize across scales

    Stage 3 (epochs 41-60):
        Grid size: 128×128 and 256×256
        Obstacle density: 0.1-0.35
        Learning rate: 1e-4
        Purpose: Handle large grids

    For each epoch:
        for batch in dataloader:
            input_tensor, target_mask = batch
            pred = model(input_tensor)
            loss = criterion(pred, target_mask)
            loss.backward()
            optimizer.step()

        # Validation
        val_metrics = evaluate(model, val_loader)
        log(val_metrics)

        # Save checkpoint if best val recall
    """
```

### 10.3 Scheduled Sampling

During later training stages, replace some ground-truth corridor labels with the model's own predictions to improve robustness:

```python
def scheduled_sampling_step(model, grid, source, goal, target_mask,
                             sampling_prob: float):
    """
    With probability `sampling_prob`, use model's own corridor prediction
    instead of ground truth for the corridor mask during loss computation.

    This simulates the noise the model will encounter at inference time
    (using its own imperfect predictions) and makes it more robust.

    Schedule: sampling_prob starts at 0.0 (all ground truth) and linearly
    increases to 0.5 over training.
    """
```

### 10.4 Metrics

```python
def evaluate_corridor_prediction(pred_blocks: Set, true_blocks: Set, all_blocks: Set):
    """
    Compute corridor prediction metrics:

    Recall = |pred ∩ true| / |true|
        CRITICAL metric. Must be >0.99 for the system to work.
        Missing even one true corridor block can cause suboptimal paths.

    Precision = |pred ∩ true| / |pred|
        Nice to have. Higher precision = less wasted computation.
        Target: >0.5

    F1 = 2 * precision * recall / (precision + recall)

    Corridor ratio = |pred| / |all|
        What fraction of blocks we need to process.
        Target: <0.3 (70% savings over processing all blocks)

    Path optimality (MOST IMPORTANT):
        Run the full HYBRID pipeline with this corridor.
        Compare resulting path cost to BFS optimal path cost.
        Optimality = BFS_cost / hybrid_cost (should be 1.0)
    """
```

---

## 11. Module 8: Inference Pipeline and Modes

**File:** `htp/pipeline.py`

### 11.1 Mode: MATRIX_ONLY

```python
def run_matrix_only(grid: Grid, source: Cell, goal: Cell,
                    config: Config) -> PathResult:
    """
    Full algebraic method. No neural network.

    Steps:
    1. Pad grid to power-of-2 multiple of block_size
    2. Partition into Level-1 blocks
    3. Compute ALL Level-1 transfer matrices (BFS)
    4. Build hierarchy and compose ALL higher-level transfer matrices
    5. Query: compute entry/exit embeddings for source/goal
    6. Find optimal boundary sequence via tropical matrix-vector products
    7. Reconstruct cell-by-cell path

    Properties:
    - GUARANTEED OPTIMAL
    - No training needed
    - Slowest mode for large grids (computes ALL blocks)
    - Useful as correctness oracle

    Time complexity:
        Level-1: O(num_blocks * B^3) for BFS
        Composition: O(num_blocks * B^3) for tropical matmul
        Total: O((H*W/B^2) * B^3) = O(H*W*B)
        For 1024×1024 with B=16: ~16M * 16 = ~260M operations
    """
```

### 11.2 Mode: NEURAL_ONLY

```python
def run_neural_only(grid: Grid, source: Cell, goal: Cell,
                    model: nn.Module, config: Config) -> PathResult:
    """
    Neural network predicts corridor and approximate path. No transfer matrices.

    Steps:
    1. Prepare input tensor (grid + source/goal encoding)
    2. Run neural network forward pass → pixel probability map
    3. Threshold pixel map to get binary corridor mask
    4. Extract approximate path by following high-probability corridor pixels
       using greedy A* with neural heuristic:
       - Standard A* but use (1 - neural_prob[cell]) as additional cost
       - OR: only explore cells where neural_prob > threshold
    5. Return path (NOT guaranteed optimal)

    Properties:
    - FAST (single neural network forward pass + lightweight search)
    - NOT guaranteed optimal
    - Quality depends on model accuracy
    - Useful for real-time applications where speed > optimality

    Approximate path extraction (detailed):
        Run A* on the grid, but restrict exploration to cells where
        the neural probability > explore_threshold (default 0.1).
        If A* fails (corridor too narrow), fall back to unrestricted A*.
    """
```

### 11.3 Mode: HYBRID (The Novel Method)

```python
def run_hybrid(grid: Grid, source: Cell, goal: Cell,
               model: nn.Module, config: Config) -> PathResult:
    """
    The full novel method: Neural corridor + Algebraic exact pathfinding.

    Steps:
    1. NEURAL PHASE: Predict corridor
        a. Prepare input tensor
        b. Run model forward pass → pixel probabilities
        c. Aggregate to block-level corridor decisions
        d. Dilate corridor by `dilation` blocks for safety
        e. Force-include source block and goal block

    2. CONFIDENCE CHECK:
        min_confidence = min corridor block confidence
        avg_confidence = mean corridor block confidence
        corridor_fraction = num_corridor_blocks / total_blocks

        If avg_confidence < confidence_threshold (default 0.7):
            FALLBACK: switch to MATRIX_ONLY mode
            (better to be slow and correct than fast and wrong)

    3. ALGEBRAIC PHASE: Compute transfer matrices for corridor only
        a. For each Level-1 block in the corridor: compute transfer matrix (BFS)
        b. Build hierarchical composition ONLY for corridor blocks
           - At each level, a parent is "active" if at least one child is active
           - Compose only active parents
        c. Compute corridor transfer matrix (source boundary → goal boundary)

    4. PATH EXTRACTION:
        a. Compute entry embedding: BFS from source to source block's boundary
        b. Compute exit embedding: BFS from goal to goal block's boundary
        c. Find optimal distance: min_ij (entry[i] + T_corridor[i][j] + exit[j])
        d. Find optimal boundary sequence via argmin traceback
        e. Reconstruct cell-by-cell path within each block

    5. VERIFICATION (optional, for debugging):
        Run BFS on full grid, compare cost.
        If costs differ: log warning, return the BFS path instead.

    Properties:
    - Optimal when corridor has perfect recall (contains true path)
    - Much faster than MATRIX_ONLY (corridor is typically 10-30% of blocks)
    - Speed depends on corridor size and neural network latency
    - The central contribution of this work

    Time complexity:
        Neural forward pass: O(model_size) ≈ fixed cost, ~10-50ms on GPU
        Level-1 TMs: O(corridor_blocks * B^3) instead of O(all_blocks * B^3)
        Composition: proportional to corridor size
        Overall: O(corridor_fraction * H * W * B)
        Typical corridor_fraction: 0.1-0.3, giving 3-10x speedup
    """
```

### 11.4 Confidence-Aware Fallback

```python
def adaptive_inference(grid: Grid, source: Cell, goal: Cell,
                       model: nn.Module, config: Config) -> PathResult:
    """
    Smart routing between modes based on problem characteristics.

    Decision logic:
        1. If grid is small (< 64×64): use MATRIX_ONLY (fast enough, guaranteed optimal)
        2. Run NEURAL corridor prediction
        3. If corridor_fraction > 0.8: use MATRIX_ONLY (corridor too broad to help)
        4. If min_block_confidence < 0.3: use MATRIX_ONLY (model uncertain)
        5. Otherwise: use HYBRID

    This is the recommended entry point for production use.
    """
```

---

## 12. Module 9: Visualization

**File:** `htp/viz.py`

### 12.1 Grid + Path Visualization

```python
def visualize_grid_and_path(grid: Grid, path: List[Cell],
                             source: Cell, goal: Cell,
                             title: str = "",
                             save_path: Optional[str] = None) -> None:
    """
    Plot the grid with the path overlaid.

    - Free cells: white
    - Blocked cells: black
    - Path cells: blue line
    - Source: green circle
    - Goal: red circle

    Use matplotlib.pyplot.imshow for the grid, plt.plot for the path.
    """
```

### 12.2 Corridor Visualization

```python
def visualize_corridor(grid: Grid, corridor_blocks: Set,
                        true_path: Optional[List[Cell]] = None,
                        block_size: int = 16,
                        confidence: Optional[Dict] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Overlay corridor block predictions on the grid.

    - Grid shown in grayscale
    - Corridor blocks highlighted with semi-transparent colored overlay
      - Color intensity proportional to confidence (if provided)
      - Green = high confidence corridor
      - Yellow = low confidence corridor
    - True path shown in blue (if provided)
    - Block grid lines shown as thin gray lines
    - Non-corridor blocks dimmed
    """
```

### 12.3 Transfer Matrix Heatmap

```python
def visualize_transfer_matrix(block: Block, save_path: Optional[str] = None) -> None:
    """
    Show the transfer matrix as a heatmap.

    - Rows and columns labeled with boundary cell coordinates
    - Infinity values shown in white/light color
    - Color scale: dark = short distance, light = long distance
    - Useful for debugging and understanding block structure
    """
```

---

## 13. Configuration System

**File:** `htp/config.py`

```python
from dataclasses import dataclass, field

@dataclass
class GridConfig:
    height: int = 256
    width: int = 256
    obstacle_density: float = 0.2
    seed: Optional[int] = None

@dataclass
class BlockConfig:
    block_size: int = 16          # Level-1 block side length
    # Derived: num_block_rows, num_block_cols, num_levels

@dataclass
class NeuralConfig:
    model_type: str = "unet"      # only option for now
    input_channels: int = 3
    base_channels: int = 32
    num_encoder_levels: int = 3
    input_resolution: int = 256   # resize grid to this before feeding to model
    checkpoint_path: str = "checkpoints/best.pt"

@dataclass
class TrainConfig:
    num_train: int = 50000
    num_val: int = 5000
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-3
    pos_weight: float = 5.0       # positive class weight in loss
    path_margin: int = 2          # dilate ground truth path by this many pixels
    curriculum_stages: List = field(default_factory=lambda: [
        {"epochs": 20, "grid_sizes": [64], "densities": [0.1, 0.2], "lr": 1e-3},
        {"epochs": 20, "grid_sizes": [64, 128], "densities": [0.1, 0.3], "lr": 5e-4},
        {"epochs": 20, "grid_sizes": [128, 256], "densities": [0.1, 0.35], "lr": 1e-4},
    ])

@dataclass
class InferenceConfig:
    mode: str = "hybrid"          # "matrix_only", "neural_only", "hybrid"
    corridor_threshold: float = 0.3     # block activation threshold
    corridor_dilation: int = 1          # safety margin in blocks
    confidence_threshold: float = 0.7   # below this, fall back to matrix_only
    verify_optimality: bool = False     # run BFS to double-check (slow, for debugging)

@dataclass
class Config:
    grid: GridConfig = GridConfig()
    block: BlockConfig = BlockConfig()
    neural: NeuralConfig = NeuralConfig()
    train: TrainConfig = TrainConfig()
    inference: InferenceConfig = InferenceConfig()
```

---

## 14. Directory Structure

```
hierarchical-tropical-pathfinding/
│
├── htp/
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclasses
│   ├── grid.py                   # Grid class, BFS, grid generation
│   ├── decomposition.py          # Block partitioning, hierarchy building
│   ├── tropical.py               # Tropical semiring, transfer matrix computation
│   ├── composition.py            # Hierarchical transfer matrix composition
│   ├── extraction.py             # Path extraction and reconstruction
│   ├── pipeline.py               # Main inference pipelines (3 modes)
│   ├── viz.py                    # Visualization utilities
│   │
│   └── neural/
│       ├── __init__.py
│       ├── model.py              # U-Net corridor predictor
│       ├── dataset.py            # Dataset and data generation
│       ├── train.py              # Training loop with curriculum
│       └── losses.py             # Loss functions
│
├── scripts/
│   ├── generate_data.py          # Generate training data
│   ├── train.py                  # Launch training
│   ├── evaluate.py               # Run evaluation benchmarks
│   ├── demo.py                   # Interactive demo
│   └── benchmark.py              # Timing benchmarks across grid sizes
│
├── tests/
│   ├── test_tropical.py          # Test tropical operations
│   ├── test_grid.py              # Test grid and BFS
│   ├── test_decomposition.py     # Test block partitioning
│   ├── test_composition.py       # Test transfer matrix composition
│   ├── test_extraction.py        # Test path extraction
│   ├── test_pipeline.py          # End-to-end tests
│   └── test_neural.py            # Test model forward pass shapes
│
├── configs/
│   └── default.yaml              # Default configuration
│
├── checkpoints/                  # Saved model weights
├── data/                         # Generated training data
├── results/                      # Benchmark results and figures
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 15. Testing Strategy

### 15.1 Unit Tests: Tropical Algebra

```python
def test_tropical_add():
    assert tropical_add(3, 5) == 3
    assert tropical_add(INF, 5) == 5
    assert tropical_add(INF, INF) == INF

def test_tropical_multiply():
    assert tropical_multiply(3, 5) == 8
    assert tropical_multiply(INF, 5) == INF
    assert tropical_multiply(0, 5) == 5

def test_tropical_matmul_identity():
    """T ⊗ I = T for any transfer matrix T"""
    T = np.array([[0, 3, INF], [3, 0, 2], [INF, 2, 0]])
    I = np.array([[0, INF, INF], [INF, 0, INF], [INF, INF, 0]])
    result = tropical_matmul(T, I)
    assert np.array_equal(result, T)

def test_tropical_matmul_simple():
    """2x2 hand-computed example"""
    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0, 2], [2, 0]])
    # C[0][0] = min(0+0, 1+2) = 0
    # C[0][1] = min(0+2, 1+0) = 1
    # C[1][0] = min(1+0, 0+2) = 1
    # C[1][1] = min(1+2, 0+0) = 0
    expected = np.array([[0, 1], [1, 0]])
    result = tropical_matmul(A, B)
    assert np.array_equal(result, expected)
```

### 15.2 Unit Tests: Transfer Matrix Correctness

```python
def test_level1_transfer_matrix_empty_block():
    """A block with no obstacles: all distances are Manhattan distances"""
    grid = Grid(np.zeros((4, 4), dtype=np.uint8))
    block = Block(...)  # 4x4 block
    T = compute_level1_transfer_matrix(grid, block)
    # T[i][j] should equal Manhattan distance between boundary cells i and j
    for i in range(len(block.boundary_cells)):
        for j in range(len(block.boundary_cells)):
            ci, cj = block.boundary_cells[i], block.boundary_cells[j]
            expected = abs(ci.row - cj.row) + abs(ci.col - cj.col)
            assert T[i][j] == expected

def test_level1_transfer_matrix_with_obstacle():
    """Block with a wall: some paths should be longer than Manhattan"""
    grid_data = np.array([
        [0, 0, 0, 0],
        [0, 1, 1, 0],  # wall in the middle
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.uint8)
    # Verify specific boundary-to-boundary distances
```

### 15.3 Integration Tests: Full Pipeline Correctness

```python
def test_matrix_only_matches_bfs():
    """For various grids, MATRIX_ONLY path cost should equal BFS path cost"""
    for seed in range(100):
        grid = generate_grid(64, 64, 0.2, seed=seed)
        source = Cell(0, 0)
        goal = Cell(63, 63)

        bfs_result = bfs_shortest_path(grid, source, goal)
        if bfs_result is None:
            continue
        bfs_path, bfs_cost = bfs_result

        matrix_result = run_matrix_only(grid, source, goal, config)
        assert abs(matrix_result.cost - bfs_cost) < 1e-6, \
            f"Seed {seed}: matrix={matrix_result.cost}, bfs={bfs_cost}"

def test_hybrid_matches_bfs_with_perfect_corridor():
    """When corridor = all blocks, HYBRID should match BFS exactly"""
    # ... same as above but run in hybrid mode with all blocks active

def test_hybrid_matches_bfs_with_true_corridor():
    """When corridor = actual blocks containing the BFS path, should match"""
    # Compute BFS path, determine which blocks it passes through,
    # set those as the corridor, run hybrid, verify costs match
```

### 15.4 Stress Tests

```python
def test_large_grid():
    """Verify correctness on 512×512 grid"""

def test_adversarial_grid():
    """Grid where shortest path is very non-obvious (spiral maze)"""

def test_no_path():
    """Source and goal in disconnected components"""

def test_source_equals_goal():
    """Path of length 0"""

def test_adjacent_source_goal():
    """Path of length 1"""

def test_all_blocked_boundary():
    """A block whose entire boundary is blocked (isolated interior)"""
```

---

## 16. Performance Notes

### 16.1 Numba JIT Compilation

The tropical matrix multiplication is the inner-loop bottleneck. Use Numba:

```python
@numba.njit(cache=True)
def tropical_matmul(A, B):
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

First call will be slow (JIT compilation). Subsequent calls: ~100x faster than pure Python.

### 16.2 Parallel Level-1 Computation

Level-1 transfer matrices are independent and can be computed in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

def compute_all_level1_parallel(blocks, grid, num_workers=8):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_level1_tm, grid, b): b for b in blocks}
        for future in as_completed(futures):
            block = futures[future]
            block.transfer_matrix = future.result()
```

### 16.3 Memory Management

For a 1024×1024 grid with B=16:
- Number of Level-1 blocks: 64×64 = 4096
- Boundary cells per block: ≤ 60 (4×16 minus corners minus blocked)
- Transfer matrix per block: 60×60 × 8 bytes = ~28 KB
- Total Level-1 storage: 4096 × 28 KB = ~115 MB

For 4096×4096:
- Number of Level-1 blocks: 256×256 = 65536
- Total Level-1 storage: ~1.8 GB
- **Consider: compute on-the-fly and discard, or use memory-mapped files**

### 16.4 Block Size Tuning

| Block Size | Boundary Cells | TM Size | BFS Cost | Composition Cost | Recommended For |
|---|---|---|---|---|---|
| 8 | ≤28 | 28×28 | Fast | Fast | Small grids (≤256×256) |
| 16 | ≤60 | 60×60 | Moderate | Moderate | Medium grids (256–2048) |
| 32 | ≤124 | 124×124 | Slow | Slow | Large grids (≥2048) |

Trade-off: larger blocks = fewer blocks (less composition work) but larger transfer matrices (more per-block work). Block size 16 is a good default.

---

## Appendix A: End-to-End Example Walkthrough

**Grid:** 32×32 with 20% obstacles, block_size = 8
**Source:** (2, 3), **Goal:** (29, 28)

1. **Pad grid:** 32 is already a power-of-2 multiple of 8. No padding needed.

2. **Partition:** 4×4 grid of 8×8 blocks.
   - Block (1,0,0): rows [0,8), cols [0,8)
   - Block (1,0,1): rows [0,8), cols [8,16)
   - ... etc.

3. **Level-1 TMs:** For each of 16 blocks, run BFS between all boundary cell pairs.
   - Each 8×8 block has ≤ 28 boundary cells
   - 16 blocks × 28 BFS runs × 64 cells per BFS = ~28K operations total

4. **Hierarchy:**
   - Level 2: 2×2 blocks, each merging 4 Level-1 blocks
   - Level 3: 1×1 block (the whole grid)

5. **Source/goal blocks:**
   - Source (2,3) is in block (1, 0, 0)
   - Goal (29,28) is in block (1, 3, 3)

6. **Neural corridor (HYBRID mode):**
   - Model predicts blocks (1,0,0), (1,0,1), (1,1,1), (1,1,2), (1,2,2), (1,2,3), (1,3,3)
   - Plus dilation: adds neighbors → maybe 12 out of 16 blocks active

7. **Compute TMs for corridor blocks only:** 12 blocks instead of 16 (25% savings)

8. **Compose corridor:** Build combined distance matrix, Floyd-Warshall, extract.

9. **Entry/exit embeddings:** BFS from (2,3) to block (1,0,0)'s boundary; BFS from (29,28) to block (1,3,3)'s boundary.

10. **Query:** min_{i,j} (entry[i] + T_corridor[i][j] + exit[j]) = 52 (example)

11. **Reconstruct path:** Trace argmins back through blocks, BFS within each block for cell-level path.

12. **Verify:** BFS on full grid gives cost 52. Match confirmed. Output path.
