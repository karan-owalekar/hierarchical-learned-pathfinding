# LANCET Ideas Bank
## All Discussed Ideas, Ranked by Viability

---

## 1. Neural Architecture Ideas

### 1.1 QuadTreeConvNet — Self-Contained Predictions (ADOPTED — Current Design)

**Status: Implemented in main spec. This is the baseline.**

Each quadtree node independently predicts which of its 4 child quadrants the path passes through, using:
- 8×8 downsampled obstacle density map of the current block
- 4 floats: source/goal position relative to block
- 4 floats: edge permeabilities from integral image
- Level embedding

No embedding propagation between levels. Context flows through pruning (which blocks get visited), not through learned vectors. Training is flat — each node is an independent supervised example.

**Why it works:** The network sees actual obstacles. Self-contained means zero error accumulation. Hierarchy naturally provides coarse-to-fine resolution.

**Limitation:** No cross-boundary awareness. No adaptation to grid character beyond what the 8×8 grid implicitly carries.

---

### 1.2 Self-Modulating QuadTreeNet (RECOMMENDED UPGRADE)

**Status: Discussed, strong candidate for v2.**

The grid encoder's output serves dual purpose:
1. Contributes obstacle features to the prediction
2. Generates modulation parameters (FiLM gamma/beta, feature attention weights, temperature) that control how position and permeability information is processed

**Key insight:** The visual character of the 8×8 grid already encodes everything that level number was proxying for. A muddy low-contrast 8×8 (dense grid at coarse level OR sparse grid at fine level) should be processed cautiously regardless of what "level" label it carries.

**Components:**
- **FiLM modulation:** Grid features generate scale (gamma) and shift (beta) applied to position/permeability features. ~8K parameters.
- **Feature attention:** Grid features generate soft weights over [grid, position, permeability] determining what the network focuses on. ~200 parameters.
- **Learned temperature:** Grid features control prediction confidence — sharp when obstacles are clear, hedging when ambiguous. ~70 parameters.
- **Level bias (optional):** Lightweight additive bias from level embedding. Provides subtle scale hint without being the primary conditioning mechanism. ~800 parameters.

**Why better than level conditioning:** A 1024×1024 sparse grid and a 64×64 dense grid can produce identical 8×8 downsamples. They should be processed identically. Level conditioning would process them differently (different level → different behavior). Self-modulation processes them the same because they look the same.

**Total overhead:** ~9K parameters on top of ~75K base model. Negligible.

**Implementation:**
```python
class SelfModulatingQuadTreeNet(nn.Module):
    def __init__(self, d=64, max_levels=12):
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, d, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mod_gamma = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())
        self.mod_beta = nn.Sequential(nn.Linear(d, d), nn.Tanh())
        self.feature_attention = nn.Sequential(nn.Linear(d, 3), nn.Softmax(dim=-1))
        self.temperature_head = nn.Sequential(nn.Linear(d, 1), nn.Softplus())
        self.pos_encoder = nn.Sequential(nn.Linear(4, 2*d), nn.ReLU(), nn.Linear(2*d, d))
        self.perm_encoder = nn.Sequential(nn.Linear(4, d), nn.ReLU(), nn.Linear(d, d))
        self.level_bias = nn.Embedding(max_levels, d)
        self.backbone = nn.Sequential(nn.Linear(d, 2*d), nn.ReLU(), nn.Linear(2*d, d), nn.ReLU())
        self.head = nn.Linear(d, 4)

    def forward(self, grid_8x8, positions, permeabilities, level):
        grid_feat = self.grid_encoder(grid_8x8).squeeze(-1).squeeze(-1)
        gamma = self.mod_gamma(grid_feat) * 2.0
        beta = self.mod_beta(grid_feat)
        attn = self.feature_attention(grid_feat)
        temp = self.temperature_head(grid_feat) + 0.1
        pos_feat = gamma * self.pos_encoder(positions) + beta
        perm_feat = gamma * self.perm_encoder(permeabilities) + beta
        combined = (attn[:, 0:1] * pos_feat +
                    attn[:, 1:2] * perm_feat +
                    attn[:, 2:3] * grid_feat)
        combined = combined + self.level_bias(level)
        features = self.backbone(combined)
        logits = self.head(features)
        return torch.sigmoid(logits / temp)
```

**Novelty assessment:** Individual techniques (FiLM, SE-Net, learned temperature) are known. The combination applied to hierarchical spatial prediction is new but unlikely to be a standalone contribution. Valuable as engineering, not as the paper's core novelty.

---

### 1.3 Overlapping Input Context (RECOMMENDED — Easy Win)

**Status: Discussed, recommended for v2 as one-line improvement.**

**Problem:** When downsampling a block to 8×8, obstacles right outside the block boundary are invisible. A wall 1 pixel outside the block affects routing but the CNN can't see it.

**Solution:** When downsampling block `[r0, r1) × [c0, c1)`, include a margin:

```python
margin = max(1, (r1 - r0) // 8)  # ~1 pixel in the 8×8 space
padded = grid.data[max(0, r0-margin):min(H, r1+margin),
                   max(0, c0-margin):min(W, c1+margin)]
grid_10x10 = downsample(padded, target_size=10)  # or 12x12
```

**CNN change:** `nn.Conv2d(1, 16, 3, padding=1)` input changes from (1, 8, 8) to (1, 10, 10). Negligible cost increase. The center 8×8 represents the block; the border represents context.

**Why this helps:** Paths near block boundaries are the hardest case. Seeing 1-2 pixels of context from neighbors lets the CNN understand cross-boundary flow without any architectural complexity.

**Why NOT overlapping blocks in the decomposition:** Blocks must partition the grid cleanly for tropical algebra to work. Overlapping blocks in the hierarchy breaks composition. Only overlap the CNN's INPUT, not the block structure itself.

**Cost:** Near zero. Downsampling a 10×10 region instead of 8×8 adds ~50% more pixels, but the CNN cost is dominated by kernel operations, not input size. Maybe 0.1ms total difference across all levels.

---

### 1.4 Level-Dependent Activation Thresholds (RECOMMENDED — Free)

**Status: Discussed, zero-cost improvement.**

Not a neural mechanism — just different sigmoid thresholds per level during inference:

```python
thresholds = {
    8: 0.20,  # top: permissive, hedge
    7: 0.25,
    6: 0.30,
    5: 0.35,
    4: 0.40,
    3: 0.45,
    2: 0.50,  # bottom: precise, tight
}
```

**Why:** At the top level, missing a quadrant is catastrophic (entire grid region excluded). At the bottom level, extra quadrants waste minimal computation. So be loose at the top, tight at the bottom.

**Can be learned:** Treat thresholds as hyperparameters, optimize on validation set via grid search. Or learn them as part of the self-modulating architecture's temperature mechanism.

**Zero parameters. Zero compute. Meaningful improvement in recall-at-top-levels.**

---

### 1.5 FiLM Conditioning on Level (SUPERSEDED by 1.2)

**Status: Discussed, superseded by self-modulation.**

Apply per-level FiLM (Feature-wise Linear Modulation) to the shared backbone's features:

```python
gamma = self.film_gamma(level_embedding)  # (d,) scale
beta = self.film_beta(level_embedding)    # (d,) shift
features = gamma * features + beta
```

**Why superseded:** This conditions on level number, which is a proxy for grid character. Self-modulation (1.2) conditions on actual grid character via the grid encoder, which is strictly more informative. Level number can optionally be a minor input but shouldn't drive modulation.

**Still useful as:** A lightweight alternative if self-modulation adds too much complexity for v1. FiLM on level is ~1,500 parameters and simple to implement. Can serve as intermediate step before full self-modulation.

---

### 1.6 Level-Dependent Feature Attention (SUPERSEDED by 1.2)

**Status: Discussed, folded into self-modulation.**

Per-level soft attention weights over input feature groups:

```python
weights = softmax(attention_weights[level])  # [grid_w, pos_w, perm_w]
combined = weights[0]*grid_feat + weights[1]*pos_feat + weights[2]*perm_feat
```

Learns: top levels focus on position (routing direction), bottom levels focus on obstacles/permeabilities (precision).

**48 parameters. Superseded by self-modulation which does this driven by grid character instead of level number.**

---

### 1.7 Level-Dependent Temperature (SUPERSEDED by 1.2)

**Status: Discussed, folded into self-modulation.**

```python
temperature = self.level_temperature[level]  # scalar per level
activations = sigmoid(logits / temperature)
```

Low temperature → sharp confident predictions. High temperature → hedging. 12 parameters.

**Superseded by learned temperature from grid features in self-modulation. But useful as standalone if self-modulation is too complex.**

---

### 1.8 Separate Networks Per Level (REJECTED)

**Status: Discussed, rejected.**

Different specialized CNN for each hierarchy level.

**Why rejected:**
- Breaks generalization across grid sizes (level 8 only exists on large grids, need massive grids to train level-8 specialist)
- Multiplies training complexity by number of levels
- The task is structurally identical at every level (predict quadrant activation from 8×8 obstacle view + positions)
- Level embedding or self-modulation achieves the same specialization with shared weights

---

### 1.9 Orchestrator Network Routing Levels to Specialists (REJECTED)

**Status: Discussed, rejected.**

A meta-network that decides which specialist handles each level.

**Why rejected:**
- Adds a second neural architecture that needs its own training
- The routing problem is trivial (level → specialist) making the orchestrator unnecessary
- More complex than the problem it solves
- Self-modulation achieves adaptive behavior without separate networks

---

### 1.10 Recursive MLP with Embedding Propagation (REJECTED — Original Design)

**Status: Was the original architecture, rejected in favor of QuadTreeConvNet.**

Small MLP shared across levels. Parent produces boundary embeddings passed to children. Three heads: activation, boundary embeddings, boundary cell prediction. Required scheduled sampling to handle train/inference mismatch.

**Why rejected:**
- Error accumulation: small errors at top levels compound through the hierarchy
- Obstacle-blind: received only positional embeddings, no grid information
- Complex training: required scheduled sampling, multi-phase curriculum, auxiliary losses
- ChildEmbeddingDeriver added significant complexity

**The fundamental flaw:** Information flowed through learned embeddings that could drift. The QuadTreeConvNet's self-contained predictions eliminate this entirely.

---

### 1.11 U-Net Image Segmentation (REJECTED — First Attempt)

**Status: Rejected early, fundamentally wrong for this problem.**

Full U-Net taking the entire grid as an image, predicting a pixel-level corridor mask.

**Why rejected:**
- Doesn't scale (4096×4096 grid as CNN input)
- Doesn't leverage the hierarchical structure
- Completely ignores the algebraic side of the system
- O(H×W) forward pass defeats the speed purpose

---

### 1.12 Pure Integral Image Features — 8 Floats Only (PARTIALLY ADOPTED)

**Status: Edge permeabilities adopted as supplement to 8×8 grid. Full 8-float approach rejected.**

Pass only 4 quadrant densities + 4 edge permeabilities per block. No CNN. No downsampled grid.

**Why partially rejected:** 8 floats cannot distinguish obstacle structures with the same density but different layout. A horizontal wall and scattered rocks can have identical quadrant densities. The 8×8 grid captures this structural difference.

**What was kept:** The 4 edge permeabilities are adopted as a supplementary input alongside the 8×8 grid because they catch thin walls invisible in the downsample. Best of both worlds.

---

## 2. Obstacle Awareness Ideas

### 2.1 Integral Image for O(1) Rectangle Queries (ADOPTED)

**Status: Core infrastructure, used for edge permeabilities.**

```python
integral = np.cumsum(np.cumsum(grid.data, axis=0), axis=1)  # ~0.3ms
# O(1) blocked-cell count for any rectangle thereafter
```

Used for:
- Computing edge permeabilities per block (4 floats, O(1) each)
- Could be used for downsampling if reshape approach doesn't work (non-divisible dims)
- Could provide quadrant densities as additional features

**Cost:** ~0.3ms for 1024×1024. One-time computation.

---

### 2.2 8×8 Downsampled Grid (ADOPTED — Primary Obstacle Input)

**Status: Core of QuadTreeConvNet.**

```python
grid_8x8 = block.reshape(8, h//8, 8, w//8).mean(axis=(1, 3))
```

Naturally captures obstacle structure at each hierarchy level's resolution. The hierarchy provides coarse-to-fine detail automatically.

---

### 2.3 Edge Permeabilities (ADOPTED — Supplementary)

4 floats per block: fraction of free cells along each internal midline. O(1) from integral image.

**Critical for:** Detecting thin walls (1-pixel wide spanning walls) invisible in 8×8 downsample. If permeability = 0, path CANNOT cross that edge regardless of what the 8×8 grid shows.

---

### 2.4 Bottom-Up Feature Pyramid with CNN per Block (REJECTED)

**Status: Rejected as too expensive.**

Tiny CNN extracting (d,)-dimensional features per Level-1 block, aggregated bottom-up through hierarchy with a learned aggregator.

**Why rejected:** O(H×W) precompute with neural network overhead. Even with a tiny CNN, running it on every Level-1 block adds significant cost. The 8×8 downsample + integral image achieves similar obstacle awareness for near-zero cost.

---

### 2.5 Handcrafted Features Per Block (SUPERSEDED by 8×8 grid)

**Status: Discussed, superseded.**

~20 hand-engineered floats per block: density, quadrant densities, edge permeability, cross-permeability, flow directionality, connected components per edge.

**Why superseded:** The 8×8 grid + CNN learns to extract whatever features are useful, including all of these and more. Handcrafted features are a subset of what the CNN can learn. The only handcrafted feature we keep is edge permeability (because it catches thin walls the CNN misses).

---

## 3. Boundary and Decomposition Ideas

### 3.1 Overlapping Input Context to CNN (RECOMMENDED — see 1.3)

Include a margin of neighboring cells when downsampling each block for the CNN. Gives cross-boundary awareness without changing the algebraic decomposition.

---

### 3.2 Overlapping Blocks in Hierarchy (REJECTED)

**Status: Rejected, architecturally dangerous.**

Make Level-1 blocks overlap by some cells, so neighboring blocks share interior cells.

**Why rejected:**
- Breaks clean quadtree partition that tropical algebra relies on
- Shared cells appear in multiple transfer matrices — reconciliation is undefined
- Composition (tropical matrix multiplication) assumes non-overlapping regions
- Optimality guarantees become much harder to maintain
- Would require fundamentally different algebraic framework

**The rule:** Overlap the CNN's input (soft, informational). Never overlap the algebraic decomposition (hard, structural).

---

### 3.3 Adaptive Block Sizes (NOTED — Future Research)

**Status: Not implemented. Potential Direction B extension.**

Instead of uniform 16×16 blocks, adapt block size to obstacle density. Dense regions get smaller blocks (more precise transfer matrices). Sparse regions get larger blocks (fewer blocks to compose).

**Challenge:** Non-uniform blocks complicate hierarchical composition. The quadtree assumes uniform 2×2 grouping. Adaptive blocks would need a more general composition framework.

**Potential approach:** Use the quadtree structure but allow "early termination" — if a large block has no obstacles, don't subdivide it further. Its transfer matrix is trivially computable (Manhattan distances). This saves computation without breaking the hierarchy.

---

## 4. Training Ideas

### 4.1 Flat Independent Training (ADOPTED)

**Status: Current design. Each quadtree node is an independent training sample.**

Generate (8×8 grid, positions, permeabilities, level) → (4 quadrant activations) pairs from BFS paths on random grids. Standard supervised learning with BCE loss.

**Why this works:** Self-contained predictions mean each sample is independent. No sequential dependencies. Large batch sizes (256+). Standard PyTorch DataLoader.

---

### 4.2 Recall-Weighted BCE Loss (ADOPTED)

**Status: Current design.**

```python
loss = BCE(pred, target, weight = target * pos_weight + (1-target) * 1.0)
```

pos_weight = 3-5 penalizes false negatives (missed corridor blocks) more than false positives (extra blocks).

**Rationale:** Missing a corridor block → suboptimal path (BAD). Extra corridor blocks → wasted computation (tolerable, algebra still finds optimal within the corridor).

---

### 4.3 Scheduled Sampling (REMOVED)

**Status: Was part of recursive MLP architecture, removed with QuadTreeConvNet.**

No longer needed because self-contained predictions have no train/inference mismatch. Each node's prediction depends only on its own inputs, not on parent predictions.

---

### 4.4 Curriculum Over Grid Sizes (OPTIONAL)

**Status: Optional improvement, not required.**

```python
stages = [
    {"epochs": 15, "grid_sizes": [64, 128], "densities": [0.1, 0.2]},
    {"epochs": 15, "grid_sizes": [128, 256], "densities": [0.1, 0.3]},
    {"epochs": 20, "grid_sizes": [256, 512], "densities": [0.1, 0.35]},
]
```

**Why optional:** Shared weights + level embedding already generalize across scales. But curriculum ensures the network sees diverse hierarchy depths during training.

---

### 4.5 Negative Mining (NOTED — Worth Trying)

**Status: Not implemented but potentially useful.**

During training, the network sees mostly "easy negatives" (clearly blocked quadrants). Hard negatives are quadrants that look plausible but aren't on the optimal path.

**Approach:** For each training grid, after initial training, run inference and find cases where the network falsely activates quadrants. Add these to the training set with extra weight.

**Expected benefit:** Tighter corridors (better precision without sacrificing recall). Most useful in later training stages after the network has learned basic patterns.

---

### 4.6 Multi-Path Training Data (NOTED — Robustness)

**Status: Not implemented. Potential improvement for grids with multiple near-optimal paths.**

For some source/goal pairs, multiple paths have the same (or nearly same) cost. The BFS finds one, but another is equally valid. Training on only one path could cause the network to learn arbitrary preferences.

**Approach:** For each training example, find the top-K shortest paths (or all paths within cost + epsilon). Mark ALL blocks touched by any near-optimal path as positive.

**Expected benefit:** Network learns to activate the full "band" of near-optimal routes, not just one arbitrary choice. Especially important for sparse grids with many equally short paths.

---

## 5. Inference Ideas

### 5.1 Three Operating Modes (ADOPTED)

MATRIX_ONLY (exact, slow), NEURAL_ONLY (fast, approximate), HYBRID (fast, exact when recall is perfect).

---

### 5.2 Adaptive Mode Selection (ADOPTED)

Automatically choose mode based on grid size, corridor fraction, and confidence.

---

### 5.3 Confidence-Based Fallback (ADOPTED)

If corridor is >80% of blocks or <1% of blocks, fall back to MATRIX_ONLY.

---

### 5.4 Level-by-Level Batching (ADOPTED)

At each hierarchy level, batch ALL active blocks into a single GPU forward pass instead of processing one-by-one. Major speedup for levels with many active blocks.

---

### 5.5 Corridor Dilation as Safety Margin (ADOPTED)

Add 1-block border around predicted corridor. Cheap insurance against edge-case misses.

---

### 5.6 Multi-Query Amortization (NOTED — Important for Games/Robotics)

**Status: Not implemented. Natural extension for applications with many queries on the same map.**

For a fixed map with many source/goal queries:
1. Precompute ALL Level-1 transfer matrices once (expensive, ~50-100ms for 1024×1024)
2. For each query: only run neural corridor prediction + corridor assembly + path extraction
3. Per-query cost drops from ~20-50ms to ~5-10ms

**Why this matters:** In games, the map is fixed (or changes rarely) but thousands of agents need paths every frame. The transfer matrices become a reusable asset.

**Extension:** Cache transfer matrices at higher levels too. If the corridor for query #2 overlaps with query #1's corridor, reuse the already-computed compositions.

---

### 5.7 Hierarchical Confidence (NOTED — Smart Fallback)

**Status: Not implemented. Refinement of confidence fallback.**

Instead of binary "use HYBRID or fall back to MATRIX_ONLY," use a per-level confidence:

- If the top level is confident: trust the pruning, proceed normally
- If level 3 is uncertain: at level 3 only, activate all 4 children instead of the predicted subset
- Below that uncertain level: resume normal prediction

This means: instead of falling back to full-grid computation, only expand the corridor locally where the network is uncertain. Much cheaper than full fallback.

---

## 6. Research Directions (Ordered by Impact)

### 6.1 Direction D: Dominant Benchmarks (HIGHEST PRIORITY)

**Status: Not started. Most important for both paper and PhD admissions.**

Benchmark against JPS, HPA*, Contraction Hierarchies, Neural A*, and VIN on MovingAI standard maps. If LANCET is 5-10x faster than CH while maintaining 100% optimality, the engineering speaks for itself.

**What's needed:**
- Implementations of all baselines (some have open-source code)
- MovingAI benchmark maps (standard, freely available)
- Careful timing methodology (wall-clock, excluding disk I/O)
- Statistical analysis (mean, median, p95, p99 per grid size)
- Breakdown: neural time vs algebra time vs total time

**Target claim:** "LANCET matches the optimality of exact methods while achieving inference times competitive with heuristic methods, across grid sizes from 256×256 to 4096×4096."

---

### 6.2 Direction B: Dynamic Grid Efficiency (HIGH IMPACT)

**Status: Not started. Strongest differentiator from existing methods.**

When obstacles move:
- Contraction Hierarchies: O(full rebuild) — expensive
- HPA*: O(affected region rebuild) — moderate
- LANCET: Only recompute transfer matrices for affected Level-1 blocks + re-run neural prediction (which is grid-size independent)

**Claim:** "LANCET's update cost for a single obstacle change is O(B²) for the affected block's transfer matrix + O(corridor × d²) for re-prediction. This is O(1) relative to grid size."

**Benchmark:** Random obstacle additions/removals per frame. Measure amortized per-query cost as the map evolves. Compare against CH rebuild cost.

---

### 6.3 Direction A: Theoretical Guarantees (MEDIUM IMPACT, HIGH PhD VALUE)

**Status: Not started. Most valuable for admissions to theory-leaning groups.**

Possible theorems:
1. "If the corridor predictor has recall ≥ r on the true Level-1 corridor blocks, then the HYBRID path is optimal with probability ≥ f(r)."
2. "The expected corridor fraction for random grids with density ρ and source-goal distance d is Θ(d / H), giving O(d × B) expected algebraic computation instead of O(H × W × B)."
3. "The tropical transfer matrix composition computes the exact shortest path in the corridor in O(C × n³) where C = corridor blocks and n = boundary cells per block, compared to O(H × W) for Dijkstra."

Even partial results (bounds, not exact) are valuable.

---

### 6.4 Direction C: Hierarchical Region Activation as a Learning Problem (MEDIUM IMPACT)

**Status: Not started. Most novel conceptually.**

Abstract the corridor prediction away from pathfinding:

"Given a function f defined on a grid and a quadtree decomposition, predict which quadtree nodes contribute to f(query) using only multi-scale downsampled views. We study sample complexity and generalization bounds for this prediction problem."

This has applications beyond pathfinding: hierarchical simulation, adaptive mesh refinement, multi-resolution rendering. But it's a harder paper to write and may be too abstract for pathfinding venues.

---

### 6.5 Direction E: 3D Voxel Grids (MEDIUM IMPACT)

**Status: Not started. Natural extension.**

Extend from 2D grid to 3D voxel grid (robotics, game worlds with vertical movement). A* becomes catastrophically slow in 3D. The octree decomposition is a natural 3D analog of the quadtree.

**Changes needed:**
- 8×8×8 downsampled voxel input instead of 8×8
- 3D Conv instead of 2D Conv
- 8 children per node instead of 4
- Transfer matrices on block surfaces instead of block edges

**Challenge:** Boundary cell count grows as O(B²) instead of O(B), making transfer matrices O(B² × B²) = O(B⁴). May need sparser representations.

---

### 6.6 Direction F: Variable-Cost Grids and Real-World Maps (NEEDED FOR PAPER)

**Status: Not started. Required for a convincing paper.**

Move beyond uniform-cost binary grids. Support:
- Terrain with variable movement costs (roads=1, grass=2, swamp=5)
- Real-world map data from MovingAI benchmarks
- Continuous cost functions

**Transfer matrix change:** BFS within blocks becomes Dijkstra within blocks. Transfer matrices still encode shortest distances, just with non-uniform costs. The tropical semiring handles this natively — no algorithmic change needed.

**CNN change:** Input becomes continuous density map (cost per cell) instead of binary (blocked/free). The 8×8 downsample naturally averages costs.

---

## 7. Implementation Priority Order

### Phase 1: Core System (Weeks 1-7)
1. Grid engine + BFS ✓
2. Tropical algebra + tests ✓
3. Block decomposition + hierarchy ✓
4. Level-1 transfer matrices ✓
5. Hierarchical composition + tests ✓
6. Path extraction + reconstruction ✓
7. MATRIX_ONLY pipeline + verify against BFS ✓

### Phase 2: Neural System (Weeks 8-11)
8. QuadTreeConvNet (Idea 1.1) — baseline neural architecture
9. Integral image + edge permeabilities (Idea 2.1, 2.3)
10. Training data generation (flat, Idea 4.1)
11. Training loop with recall-weighted loss (Idea 4.2)
12. HYBRID pipeline — wire neural + algebraic
13. NEURAL_ONLY pipeline

### Phase 3: Improvements (Weeks 12-14)
14. Level-dependent thresholds (Idea 1.4) — free improvement
15. Overlapping input context (Idea 1.3) — one-line change
16. Self-modulating architecture (Idea 1.2) — if baseline needs improvement
17. Multi-query amortization (Idea 5.6) — if targeting games/robotics

### Phase 4: Research (Weeks 15-20)
18. Benchmarks against JPS, HPA*, CH (Direction D)
19. MovingAI maps (Direction F)
20. Dynamic grid experiments (Direction B)
21. Theoretical analysis (Direction A)
22. Write paper

---

## 8. Quick Reference: What Was Tried and Why

| Idea | Status | One-Line Reason |
|------|--------|-----------------|
| U-Net on full grid | ❌ Rejected | Doesn't scale, ignores hierarchy |
| Recursive MLP + embeddings | ❌ Rejected | Error accumulation, obstacle-blind |
| Position-only encoding | ❌ Rejected | No obstacle info → corridors too wide |
| Pure integral image (8 floats) | ⚠️ Partial | Can't distinguish structures with same density |
| Bottom-up feature pyramid | ❌ Rejected | O(H×W) precompute too expensive |
| Separate networks per level | ❌ Rejected | Breaks generalization across grid sizes |
| Orchestrator meta-network | ❌ Rejected | Solves non-existent problem |
| Scheduled sampling | ❌ Removed | Not needed with self-contained predictions |
| Overlapping block decomposition | ❌ Rejected | Breaks tropical algebra composition |
| **QuadTreeConvNet** | ✅ Adopted | Self-contained, sees obstacles, simple training |
| **Edge permeabilities** | ✅ Adopted | Catches thin walls, O(1), supplements CNN |
| **Integral image** | ✅ Adopted | O(1) queries, one-line precompute |
| **Overlapping input context** | 🔜 v2 | Cross-boundary awareness, one-line change |
| **Level-dependent thresholds** | 🔜 v2 | Free recall improvement at coarse levels |
| **Self-modulation (FiLM)** | 🔜 v2 | Grid-driven adaptation, replaces level conditioning |
| **Multi-query amortization** | 🔜 v3 | Cache TMs for repeated queries on same map |
| **Negative mining** | 🔜 v3 | Tighter corridors after initial training |
| **Multi-path training** | 🔜 v3 | Robustness for grids with multiple optimal paths |
| **Hierarchical confidence** | 🔜 v3 | Smart partial fallback instead of all-or-nothing |

---

## 9. Open Questions

1. **What's the minimum corridor recall for practical optimality?** Is 95% recall enough (accepting rare 1-2% suboptimality), or do we need 99.9%?

2. **Does the self-modulating architecture actually improve corridor tightness over plain QuadTreeConvNet?** Need ablation study.

3. **How does corridor fraction scale with grid size?** If corridor fraction grows with grid size (bad) vs stays constant (good), it determines whether LANCET's advantage grows or shrinks on larger grids.

4. **What grid types break the method?** Mazes? Spirals? Grids with many equally-optimal paths? Need adversarial evaluation.

5. **Is the 8×8 downsample resolution optimal?** Would 4×4 (cheaper, coarser) or 16×16 (more expensive, finer) be better? Need ablation.

6. **Can we prove a tighter bound than "optimal when recall is 100%"?** Something like "optimal with probability 1 - δ when recall ≥ 1 - ε, where δ decays exponentially in ε"?
