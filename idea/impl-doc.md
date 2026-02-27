# Hierarchical Learned Pathfinding (HLP)
## Implementation Document

---

## Table of Contents

1. [Overview](#1-overview)
2. [Core Idea](#2-core-idea)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Data Structures](#4-data-structures)
5. [Grid Engine](#5-grid-engine)
6. [Block Decomposition and Hierarchy](#6-block-decomposition-and-hierarchy)
7. [Tropical Transfer Matrix Engine](#7-tropical-transfer-matrix-engine)
8. [Hierarchical Composition](#8-hierarchical-composition)
9. [Neural Corridor Predictor — QuadTreeConvNet](#9-neural-corridor-predictor)
10. [Training Pipeline](#10-training-pipeline)
11. [Mixture of Experts — Specialist Models](#11-mixture-of-experts)
12. [Inference Pipeline and Modes](#12-inference-pipeline-and-modes)
13. [Path Extraction](#13-path-extraction)
14. [Configuration](#14-configuration)
15. [Benchmark Results](#15-benchmark-results)
16. [UI and Visualization](#16-ui-and-visualization)
17. [What We Tried That Didn't Work](#17-what-we-tried-that-didnt-work)

---

## 1. Overview

HLP finds shortest paths on 2D obstacle grids by combining a learned neural
corridor predictor with classical algebraic pathfinding. The system divides the
grid into a quadtree of blocks, uses a CNN to predict which blocks the shortest
path passes through (the "corridor"), and then runs exact algebraic
pathfinding only inside that corridor. This lets HLP skip most of the grid
while still finding optimal paths.

### Tech Stack

- Python 3.10+, NumPy, Numba, PyTorch, Pygame
- Package manager: uv
- Configuration: YAML via dataclasses

---

## 2. Core Idea

Classical pathfinding algorithms like A\*, BFS, and Dijkstra must explore a
large fraction of the grid. On a 512×512 grid that can mean hundreds of
thousands of cells, most of which are nowhere near the actual shortest path.

HLP's insight is: a small CNN can look at a coarsened view of the obstacles and
predict which quadrants of the grid the path passes through. By recursively
subdividing the grid and throwing away the quadrants predicted to be
irrelevant, we narrow down to a corridor that typically covers only 8–15% of
the grid. Search then runs only inside that corridor.

### Why it works

1. **The grid is spatially redundant.** Most cells are far from the shortest
   path and can be safely ignored. The path between two points only touches a
   narrow band of the grid.

2. **Obstacle structure is visible at coarse scales.** A 512×512 block
   downsampled to 8×8 still shows walls, corridors, and density patterns. This
   is enough for the network to route the path at a coarse level.

3. **The quadtree provides built-in multi-resolution.** At the top level the
   network sees the whole grid at 8×8 and makes a rough routing decision. At
   deeper levels it sees fine-grained obstacle detail within each active
   quadrant. Coarse levels handle global routing; fine levels handle local
   precision.

4. **Each prediction is self-contained.** The network receives the 8×8
   obstacle density map of the current block, the source/goal positions
   relative to that block, and a level embedding. It makes a prediction using
   only this local information — no embedding propagation from parent to child,
   so errors cannot accumulate across levels.

5. **Algebra guarantees optimality.** In Hybrid mode the neural prediction is
   only used to select which blocks to compute. The actual pathfinding within
   those blocks uses exact tropical semiring transfer matrices. As long as the
   corridor includes all the blocks the true shortest path passes through, the
   result is provably optimal.

---

## 3. Mathematical Foundations

### 3.1 Grid

A grid `G` is a 2D array of shape `(H, W)`. `G[r][c] = 0` is a free
(traversable) cell; `G[r][c] = 1` is blocked. Movement is 4-connected (up,
down, left, right), each step costs 1.

### 3.2 Tropical Semiring

The tropical (min-plus) semiring `(R ∪ {∞}, ⊕, ⊗)`:
- `a ⊕ b = min(a, b)`
- `a ⊗ b = a + b`
- Additive identity: `∞`; multiplicative identity: `0`

### 3.3 Transfer Matrix

For a rectangular block with `n` free boundary cells, the transfer matrix
`T` is `n × n` where `T[i][j]` is the shortest distance from boundary cell
`i` to boundary cell `j` using only cells inside the block. `∞` means no
path exists within the block.

Composition: for adjacent blocks A and B, the merged transfer matrix is
`(T_A ⊗ T_B)[i][j] = min_k (T_A[i][k] + T_B[k][j])` over shared boundary
cells `k`.

### 3.4 Boundary Cells

Free cells on the edge of a block, enumerated clockwise:
1. Top edge, left to right
2. Right edge, top to bottom (skip top-right corner)
3. Bottom edge, right to left (skip corners)
4. Left edge, bottom to top (skip corners)

This clockwise ordering ensures shared boundaries between adjacent blocks
align correctly for composition.

### 3.5 Hierarchical Levels

- **Level 1:** Grid divided into `B × B` blocks (B=16 by default). Transfer
  matrices computed via BFS between all boundary cell pairs.
- **Level L ≥ 2:** Groups of 2×2 Level-(L-1) blocks form a Level-L block.
  Transfer matrix computed by composing children.
- Continue until one top-level block covers the entire grid.

### 3.6 Corridor

The set of blocks the shortest path passes through. The neural network
predicts this set. Transfer matrices are only computed for corridor blocks,
which is the source of the speedup.

---

## 4. Data Structures

### Cell

```python
@dataclass(frozen=True)
class Cell:
    row: int
    col: int
```

### Grid

```python
class Grid:
    data: np.ndarray   # (H, W), uint8, 0=free 1=blocked
    height: int
    width: int
```

### Block

```python
@dataclass
class Block:
    block_id: tuple[int, int, int]        # (level, block_row, block_col)
    level: int
    row_start: int                         # inclusive
    row_end: int                           # exclusive
    col_start: int
    col_end: int
    boundary_cells: list[Cell]             # clockwise order
    boundary_cell_to_index: dict[Cell, int]
    transfer_matrix: Optional[np.ndarray]  # (n, n), float64
    children: Optional[list[Block]]
    is_active: bool
```

### PathResult

```python
@dataclass
class PathResult:
    path: Optional[list[Cell]]
    cost: float
    optimal: bool
    mode: str
    corridor_size: int
    total_blocks: int
    computation_time_ms: float
```

---

## 5. Grid Engine

**File:** `hlp/grid.py`

- `generate_grid(h, w, density, seed, ensure_connected)` — random obstacle
  grid.
- `bfs_shortest_path(grid, source, goal)` — full-grid BFS oracle. Used for
  ground truth in training, Level-1 transfer matrices, and correctness
  verification.
- `bfs_within_block(...)` — BFS restricted to a rectangular block.
- `bfs_all_distances_within_block(...)` — BFS from a source to all reachable
  cells within a block.

Five map generators live in `ui/map_generators.py`:

| Key | Generator | Description |
|---|---|---|
| `random_scatter` | `generate_random` | Random obstacle placement at given density |
| `dfs_maze` | `generate_dfs_maze` | DFS-carved maze with step size 2 |
| `spiral` | `generate_spiral` | Spiral wall pattern with random gaps |
| `recursive_division` | `generate_recursive_division` | Recursive division maze |
| `rooms` | `generate_rooms` | Random rooms connected by corridors |

---

## 6. Block Decomposition and Hierarchy

**File:** `hlp/decomposition.py`

1. **Padding.** Grid is padded to a power-of-2 multiple of `block_size` with
   blocked cells so the quadtree is clean.
2. **Level-1 partitioning.** Padded grid is divided into `B × B` blocks.
   Boundary cells are enumerated in clockwise order for each block.
3. **Hierarchy building.** Groups of 2×2 Level-1 blocks form Level-2 blocks,
   and so on up to a single top-level block covering the whole grid.

---

## 7. Tropical Transfer Matrix Engine

**File:** `hlp/tropical.py`

Numba-JIT-compiled tropical matrix operations:

```python
@numba.njit(cache=True)
def tropical_matmul(A, B):
    C[i][j] = min_k (A[i][k] + B[k][j])
```

Level-1 transfer matrices are computed by running BFS from each boundary cell
to all other boundary cells within the block. For a 16×16 block with up to 60
boundary cells, this costs O(60 × 256) ≈ 15K operations per block.

---

## 8. Hierarchical Composition

**File:** `hlp/composition.py`

A Level-L block has 4 children (TL, TR, BL, BR). To compose their transfer
matrices into the parent's:

1. Build a combined distance matrix over all children's boundary cells. Within
   the same child: distances come from the child's transfer matrix. Between
   adjacent children: crossing cost = 1 for grid-adjacent boundary cells.
2. Run Floyd-Warshall (tropical all-pairs shortest paths) on the combined
   matrix.
3. Extract the submatrix for the parent's external boundary cells.

The full hierarchy is computed bottom-up. In Hybrid mode, only corridor blocks
are computed — the key speedup.

---

## 9. Neural Corridor Predictor

**Files:** `hlp/neural/model.py`

### QuadTreeConvNet Architecture

A shared-weight CNN that processes 8×8 downsampled obstacle density maps at
every level of the quadtree. Each prediction is self-contained — no embedding
propagation between levels.

**Inputs** (per block):
- 8×8 obstacle density map (1 channel)
- 4 floats: normalized source/goal positions relative to block
- Level index (integer)

**Architecture:**

```
Grid Encoder:
    Conv2d(1, 16, 3, padding=1) → ReLU
    Conv2d(16, 32, 3, padding=1) → ReLU
    Conv2d(32, d, 3, padding=1) → ReLU
    AdaptiveAvgPool2d(1) → d-dim vector

Position Encoder:
    Linear(4, 2d) → ReLU → Linear(2d, d) → d-dim vector

Level Embedding:
    Embedding(max_levels, d) → d-dim vector

Head:
    concat [grid_feat, pos_feat, level_feat] → 3d
    Linear(3d, 2d) → ReLU
    Linear(2d, d) → ReLU
    Linear(d, 4) → Sigmoid → 4 activation probabilities
```

**Parameters:** d=64, max_levels=12, grid_resolution=8. Total: ~75K
parameters.

**Output:** 4 sigmoid values, one per child quadrant. Values above the
activation threshold (0.3) mean the quadrant is predicted to contain the
shortest path.

### Downsampling

```python
def downsample_block(block, resolution=8):
    # For blocks with compatible dimensions: reshape + mean
    # Otherwise: grid-aligned sampling
```

Each block at any level is always reduced to 8×8, so the CNN sees a fixed
input size regardless of the grid's actual dimensions. The hierarchy provides
the multi-scale aspect: the same CNN at level 10 sees the entire 1024×1024
grid compressed to 8×8 (coarse routing), while at level 1 it sees a 2×2 block
at 8×8 (fine detail).

### Batched Recursive Inference

```python
def recursive_neural_inference(model, grid_data, grid_h, grid_w,
                                source, goal, stop_at_size=1,
                                activation_threshold=0.5):
```

1. Pad grid to power-of-2.
2. Start with the whole grid as a single work item.
3. At each level, for all active blocks:
   - Downsample each block to 8×8.
   - Normalize source/goal positions relative to block.
   - Batch all blocks into a single GPU forward pass.
   - For each block, check which of 4 child quadrants exceed the threshold.
   - Queue accepted children for the next level.
4. When blocks reach `stop_at_size`, collect their free cells.
5. Return the set of active cells (the corridor).

`stop_at_size=1` (Neural Only): refine down to individual cells.
`stop_at_size=block_size` (Hybrid): stop at Level-1 blocks and hand off to
the algebraic engine.

---

## 10. Training Pipeline

**File:** `hlp/neural/train.py`

### Two-Phase Training

**Phase 1 — Flat Teacher Forcing** (30 epochs, lr=1e-3):

Training data is generated by running BFS on random grids. For each
(grid, source, goal, BFS path) query, the quadtree is walked top-down and
one training example is extracted per active node:

```
example = {
    grid_8x8:    8×8 obstacle density of this block,
    positions:   [src_r, src_c, goal_r, goal_c] normalized to block,
    level:       hierarchy level (integer),
    activation:  [0/1, 0/1, 0/1, 0/1] which children the path passes through,
}
```

Because each node prediction is self-contained, training is flat standard
supervised learning — no recursive curriculum or scheduled sampling needed.

- **Optimizer:** Adam, lr=1e-3
- **Loss:** Weighted BCE with `pos_weight=5.0` (penalize false negatives 5×
  more than false positives, because missing a corridor block causes
  suboptimal paths while extra blocks only waste computation).
- **Saving criterion:** Best `val_loss`.
- **Early stopping:** Patience 5, minimum 10 epochs before stopping.
- **Data:** 50K flat training examples, 5K validation, drawn from grid sizes
  [32, 64, 128, 256] and densities [0.1, 0.2, 0.3].
- **End-to-end monitoring:** Every 5 epochs, e2e recall and corridor ratio
  are computed on a recursive validation set (actual inference on full grids).

**Phase 2 — Adversarial Mining** (5 rounds, 10K queries each, lr=1e-4):

1. Run the trained model on random grids.
2. For each query, run BFS to get the ground truth path.
3. If any path cell is NOT inside the predicted corridor, extract flat
   training examples from that failure case.
4. Retrain the model on the hard examples.
5. Repeat for 5 rounds or until no more failures are found.

This addresses distribution shift: the flat dataset may not cover all the
edge cases the model encounters during actual inference. Adversarial mining
finds and patches those gaps.

### Loss Function

**File:** `hlp/neural/losses.py`

```python
class CorridorLoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        ...

    def forward(self, activation, activation_label):
        weight = activation_label * self.pos_weight + (1.0 - activation_label)
        loss = F.binary_cross_entropy(activation, activation_label,
                                       weight=weight, reduction="mean")
        return loss, {"loss": loss.item()}
```

### Data Generation

**File:** `hlp/neural/dataset.py`

- `generate_flat_dataset(...)` — generates per-node flat examples from BFS
  paths. Each query produces ~log2(grid_size) examples (one per active
  quadtree node).
- `generate_recursive_dataset(...)` — stores full (grid, source, goal, path)
  queries for end-to-end validation.
- `extract_flat_labels(...)` — walks the quadtree top-down for one query and
  extracts training examples at each active node.

For map-type specialists, the appropriate map generator is used instead of
the default random scatter generator.

---

## 11. Mixture of Experts

Instead of one model for all grid types, HLP trains independent specialist
networks — one per map type. Each specialist is a fresh QuadTreeConvNet
(~75K params) trained from scratch on grids generated by the corresponding
map generator.

### Map Types and Checkpoints

| Map Type | Checkpoint | Generator |
|---|---|---|
| Random Scatter | `best_random_scatter.pt` | Random obstacle placement |
| DFS Maze | `best_dfs_maze.pt` | DFS-carved maze |
| Spiral | `best_spiral.pt` | Spiral walls with gaps |
| Recursive Division | `best_recursive_division.pt` | Recursive wall division |
| Rooms & Corridors | `best_rooms.pt` | Random rooms + corridors |

At inference time the appropriate specialist is selected based on the map type.
The UI dropdown automatically loads the correct model. The benchmark script
iterates through all map types with their corresponding specialists.

### Training All Specialists

```bash
python -m hlp.neural.train --all
```

This loops through all 5 map types sequentially. Each specialist is trained
from randomly initialized weights with the same hyperparameters (50K
examples, 30 epochs, patience 5, 5 adversarial rounds with 10K queries).

---

## 12. Inference Pipeline and Modes

**File:** `hlp/pipeline.py`

### Matrix Only

Full algebraic computation. No neural network.

1. Partition grid into Level-1 blocks.
2. Compute ALL Level-1 transfer matrices via BFS.
3. Compose all higher-level transfer matrices.
4. Extract path from the hierarchical structure.

Always optimal. Slowest mode. Used as correctness baseline.

### Neural Only

Pure neural corridor prediction + BFS within corridor.

1. Run `recursive_neural_inference` with `stop_at_size=1` to get a set of
   active cells.
2. Run BFS restricted to the active cells.
3. If BFS fails (corridor too narrow), return no path.

Fastest mode for large grids. Not guaranteed optimal (depends on corridor
recall). In practice achieves 90–100% optimality depending on map type.

### Hybrid (Primary Method)

Neural corridor prediction + algebraic verification.

1. Run `recursive_neural_inference` with `stop_at_size=block_size` to identify
   active Level-1 blocks.
2. Force-include source and goal blocks.
3. Compute transfer matrices ONLY for active blocks (the corridor).
4. Compose and extract path through corridor blocks.
5. If path extraction fails, fall back to full-grid BFS.

Optimal when the corridor includes all blocks the true path passes through.
Faster than Matrix Only because only a fraction of blocks are computed. Falls
back gracefully on corridor misses.

---

## 13. Path Extraction

**File:** `hlp/extraction.py`

Once transfer matrices are computed for the corridor blocks, the path is
reconstructed:

1. Compute entry distances: BFS from source to the source block's boundary
   cells.
2. Compute exit distances: BFS from goal to the goal block's boundary cells.
3. Assemble the corridor transfer matrix using Floyd-Warshall over all
   corridor block boundaries.
4. Find optimal distance: `min_{i,j} (entry[i] + corridor_tm[i][j] + exit[j])`.
5. Trace argmins for the boundary cell sequence.
6. BFS within each block for cell-level path segments.
7. Concatenate all segments into the final path.

---

## 14. Configuration

**File:** `hlp/config.py`, `configs/default.yaml`

```yaml
grid:
  height: 256
  width: 256
  obstacle_density: 0.2

block:
  block_size: 16

neural:
  d: 64
  max_levels: 12
  grid_resolution: 8
  checkpoint_path: checkpoints/best.pt

train:
  num_train: 50000
  num_val: 5000
  batch_size: 64
  min_path_distance: 10
  teacher_epochs: 30
  lr_teacher: 0.001
  adversarial_rounds: 5
  adversarial_queries: 10000
  lr_adversarial: 0.0001
  pos_weight: 5.0
  early_stop_patience: 5

inference:
  mode: hybrid
  activation_threshold: 0.3
  verify_optimality: false
```

---

## 15. Benchmark Results

Benchmarked on 20 trials per grid size, 5 grid sizes (32–512), across all 5
map types. Each specialist model is used for its corresponding map type.

### DFS Maze — Best Results

DFS Maze shows the strongest speedup because mazes have long winding paths
that classical methods struggle with:

| Size | A\* (ms) | Neural Only (ms) | Hybrid (ms) | Optimality |
|---:|---:|---:|---:|---|
| 32 | 1.6 | 15.8 | 4.7 | 100% |
| 64 | 15.9 | 54.0 | 18.9 | 100% |
| 128 | 402.9 | 244.8 | 107.6 | 100% |
| 256 | 8,384.8 | 838.0 | 314.1 | 100% |
| 512 | 135,736.6 | 3,276.9 | 1,394.8 | 100% |

At 512×512, **Hybrid is 97× faster than A\*** while maintaining 100%
optimality. Neural Only is 41× faster.

### Recursive Division — 100% Optimality Across All Methods

| Size | A\* (ms) | Neural Only (ms) | Hybrid (ms) | Optimality |
|---:|---:|---:|---:|---|
| 32 | 0.24 | 13.5 | 4.4 | 100% |
| 64 | 0.32 | 25.1 | 17.6 | 100% |
| 128 | 0.09 | 5.7 | 5.7 | 100% |
| 256 | 0.03 | 4.0 | 20.3 | 100% |
| 512 | 0.03 | 5.7 | 99.4 | 100% |

### Random Scatter

| Size | A\* (ms) | Neural Only (ms) | Hybrid (ms) | Optimality |
|---:|---:|---:|---:|---|
| 32 | 0.4 | 18.2 | 36.2 | 100% |
| 128 | 4.3 | 28.3 | 139.7 | 100% |
| 512 | 2,984.9 | 324.8 | 1,368.1 | 100% |

At 512×512 Neural Only is 9× faster than A\*. Hybrid is 2× faster.

### Summary

- **Hybrid mode maintains 100% optimality** on DFS Maze, Spiral, Recursive
  Division, and Random Scatter for all tested sizes.
- **Greatest speedup on maze-like grids** where A\* is slow and the corridor
  is narrow.
- **Neural Only is fastest** but may miss paths on large complex grids
  (corridor recall < 100%).
- On small grids or very open maps, A\* overhead is already tiny and Hybrid's
  neural + algebraic overhead makes it slower in absolute terms.

---

## 16. UI and Visualization

**Files:** `ui/app.py`, `ui/grid_view.py`, `ui/components.py`, `ui/theme.py`

A Pygame-based interactive application:

- Grid sizes: 32, 64, 128, 256
- All 5 map types via dropdown
- Methods: Matrix Only, Neural Only, Hybrid, A\*, Dijkstra
- Click to place/remove obstacles, S/G keys for start/goal
- "Generate" button for new random maps
- Animated path drawing
- Corridor overlay for neural methods (shows which blocks are active)

The UI loads all specialist checkpoints on startup and automatically selects
the correct specialist based on the map type dropdown.

---

## 17. What We Tried That Didn't Work

### 17.1 U-Net Image Segmentation (Rejected Early)

The original plan was a U-Net that takes the full grid as an image and
predicts a pixel-level corridor mask.

**Why it failed:**
- Doesn't scale to large grids (4096×4096 as CNN input is prohibitive).
- Ignores the hierarchical structure entirely.
- O(H×W) forward pass negates any speedup.

### 17.2 Recursive MLP with Embedding Propagation (Rejected)

A shared-weight MLP where the parent produces boundary embeddings passed
down to children. Three heads: activation, boundary embeddings, boundary cell
prediction. Required scheduled sampling to handle train/inference mismatch.

**Why it failed:**
- **Error accumulation:** Small errors at top levels compound through the
  hierarchy as embeddings drift.
- **Obstacle-blind:** Received only positional embeddings, no actual obstacle
  information from the grid.
- **Complex training:** Required scheduled sampling, multi-phase curriculum,
  auxiliary losses, and a ChildEmbeddingDeriver module.

The QuadTreeConvNet's self-contained predictions (each node sees actual
obstacles via the 8×8 grid) eliminated all of these problems.

### 17.3 Saving on e2e_recall Instead of val_loss

We tried saving the best checkpoint based on end-to-end recall (actual
corridor coverage on full inference runs) instead of validation loss.

**Why it failed:**
- Early epochs have artificially high e2e_recall because the model predicts
  very large corridors that happen to cover the path.
- The model got saved too early with a high recall but terrible precision
  (50%+ of the grid as corridor), then early stopping kicked in.
- Validation loss proved to be a more reliable proxy for overall model
  quality during Phase 1.

### 17.4 Composite Saving Score (e2e_recall × (1 - corridor))

Tried a composite metric that rewards high recall while penalizing large
corridors.

**Why it failed:**
- Over-penalized the model during early training when corridors are naturally
  wide.
- Created an unstable optimization target that didn't correlate well with
  actual benchmark performance.

### 17.5 Learning Rate Scheduler (ReduceLROnPlateau)

Added `ReduceLROnPlateau` on `val_loss` during Phase 1.

**Why it failed:**
- The original simple Adam with fixed lr=1e-3 and early stopping was already
  finding good minima.
- The scheduler made training less predictable and sometimes reduced the LR
  too aggressively, leading to worse final performance.
- Reverted to the simpler approach that consistently produced the best
  benchmarks.

### 17.6 200K Training Examples (More Data = Worse)

Increased training data from 50K to 200K flat examples.

**Why it failed:**
- Benchmark showed 90% optimality instead of 100%, with very tight corridors
  (4.4–5.9%).
- The model overfit to precision/pruning — it learned to make very tight
  corridors that missed valid path cells.
- With 50K examples the model learned a better recall-precision balance,
  consistently achieving 100% optimality with 8–12% corridors.
- More data paradoxically made the model more conservative, not more robust.

### 17.7 80 Epochs, Patience 7, e2e Validation Every Epoch

Extended training with longer patience and more frequent end-to-end
evaluation.

**Why it failed:**
- Longer training didn't improve over the simpler 30-epoch, patience-5
  configuration.
- Computing e2e metrics every epoch slowed down training significantly
  without improving the saved checkpoint.
- The "champion" configuration (30 epochs, patience 5, save on val_loss)
  consistently matched or beat all longer training runs.

### 17.8 Global Context Features

Tried adding global grid statistics (overall density, grid size encoding)
as additional inputs to the model.

**Why it failed:**
- The 8×8 downsampled grid already implicitly encodes density and obstacle
  structure at the appropriate scale for each level.
- Extra features didn't improve performance and added unnecessary complexity.

### 17.9 Edge Permeability Features

4 floats per block representing the fraction of free cells along each
internal midline (top-bottom, left-right splits). Computed in O(1) from an
integral image.

**Status:** Explored in the idea phase but not included in the final model.
The 8×8 grid proved sufficient for detecting walls and obstacle patterns. Edge
permeabilities might help catch thin single-pixel walls invisible in the
downsample, but this edge case didn't surface in practice.

### Summary: What Actually Works

The configuration that consistently produces the best results:

| Parameter | Value |
|---|---|
| Architecture | QuadTreeConvNet, d=64, 1-channel input |
| Training data | 50K flat examples |
| Grid sizes | [32, 64, 128, 256] |
| Densities | [0.1, 0.2, 0.3] |
| Phase 1 | 30 epochs, Adam lr=1e-3, patience 5 |
| Phase 2 | 5 adversarial rounds, 10K queries, lr=1e-4 |
| Loss | Weighted BCE, pos_weight=5.0 |
| Save criterion | Best val_loss |
| Activation threshold | 0.3 |
| Specialist per map type | Yes (Mixture of Experts) |

Simple, no fancy tricks. The architecture and training are straightforward;
the novelty is in the hierarchical decomposition + neural corridor prediction
combination.
