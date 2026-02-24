"""Trainer for the QuadTreeConvNet corridor predictor.

Phase 1 — Flat teacher forcing:
    Train on pre-extracted per-node examples (8×8 grid, positions, level,
    activations). This is the only supervised training phase — each node
    prediction is self-contained so no recursive curriculum is needed.

Phase 2 — Adversarial mining:
    Run inference, verify with BFS, extract flat examples from failures,
    retrain. This finds and fixes cases the flat data distribution missed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from hlp.config import Config
from hlp.grid import Cell, Grid, bfs_shortest_path, generate_grid
from hlp.neural.dataset import (
    FlatDataset,
    RecursiveDataset,
    extract_flat_labels,
    generate_flat_dataset,
    generate_recursive_dataset,
)
from hlp.neural.losses import CorridorLoss
from hlp.neural.model import QuadTreeConvNet, recursive_neural_inference


# ---------------------------------------------------------------------------
# Small in-memory dataset for adversarial examples
# ---------------------------------------------------------------------------

class _InMemoryFlatDataset(Dataset):
    def __init__(self, examples: list[dict[str, np.ndarray]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "grid_8x8": torch.from_numpy(ex["grid_8x8"]),
            "positions": torch.from_numpy(ex["positions"]),
            "level": torch.tensor(int(ex["level"]), dtype=torch.long),
            "activation": torch.from_numpy(ex["activation"]),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, config: Config, device: Optional[str] = None) -> None:
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        nc = config.neural
        self.model = QuadTreeConvNet(
            d=nc.d,
            max_levels=nc.max_levels,
            grid_resolution=nc.grid_resolution,
        ).to(self.device)

        tc = config.train
        self.criterion = CorridorLoss(pos_weight=tc.pos_weight)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"QuadTreeConvNet: {total_params:,} parameters on {self.device}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        data_dir: str = "data",
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        tc = self.config.train
        nc = self.config.neural

        flat_train = Path(data_dir) / "flat_train"
        flat_val = Path(data_dir) / "flat_val"
        recur_val = Path(data_dir) / "recur_val"

        gen_kw = dict(
            grid_sizes=[32, 64, 128, 256],
            densities=[0.1, 0.2, 0.3],
            min_path_distance=tc.min_path_distance,
            grid_resolution=nc.grid_resolution,
        )

        if not flat_train.exists() or not any(flat_train.glob("*.npz")):
            print("Generating flat training data...")
            generate_flat_dataset(str(flat_train), tc.num_train, seed=42, **gen_kw)
        if not flat_val.exists() or not any(flat_val.glob("*.npz")):
            print("Generating flat validation data...")
            generate_flat_dataset(str(flat_val), max(tc.num_val, 2000), seed=99, **gen_kw)
        if not recur_val.exists() or not any(recur_val.glob("*.npz")):
            print("Generating recursive validation queries...")
            generate_recursive_dataset(
                str(recur_val), max(tc.num_val // 5, 500), seed=13,
                grid_sizes=[32, 64, 128, 256], densities=[0.1, 0.2, 0.3],
                min_path_distance=tc.min_path_distance,
            )

        train_ds = FlatDataset(str(flat_train))
        val_ds = FlatDataset(str(flat_val))
        recur_val_ds = RecursiveDataset(str(recur_val))

        patience = tc.early_stop_patience

        # Phase 1 — Flat Teacher Forcing
        print("\n=== Phase 1: Flat Teacher Forcing ===")
        self._phase_flat(
            train_ds, val_ds, recur_val_ds,
            tc.teacher_epochs, tc.lr_teacher,
            checkpoint_dir, patience,
        )

        # Phase 2 — Adversarial Mining
        print("\n=== Phase 2: Adversarial Mining ===")
        self._phase_adversarial(
            tc.adversarial_rounds, tc.adversarial_queries,
            tc.lr_adversarial, checkpoint_dir,
        )

        # Final end-to-end recall
        print("\n=== Final Evaluation ===")
        recall = self._validate_recursive(recur_val_ds)
        print(f"  End-to-end recall: {recall:.4f}")

        final = Path(checkpoint_dir) / "final.pt"
        torch.save(self.model.state_dict(), final)
        print(f"\nTraining complete. Final recall: {recall:.4f}")

    # ------------------------------------------------------------------
    # Phase 1 — flat batched teacher forcing
    # ------------------------------------------------------------------

    def _phase_flat(
        self,
        train_ds: FlatDataset,
        val_ds: FlatDataset,
        recur_val_ds: RecursiveDataset,
        epochs: int,
        lr: float,
        ckpt_dir: str,
        patience: int,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        no_improve = 0
        best_loss = float("inf")
        min_epochs_before_stop = max(epochs // 3, 5)

        for epoch in range(1, epochs + 1):
            train_loss = self._train_flat_epoch(train_ds, optimizer)
            val_metrics = self._validate_flat(val_ds)

            recall_str = ""
            if epoch % 5 == 0 or epoch == epochs:
                recall = self._validate_recursive(recur_val_ds)
                recall_str = f" | e2e_recall={recall:.4f}"

            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.4f} | "
                f"val={val_metrics['loss']:.4f} | "
                f"recall={val_metrics['recall']:.4f}"
                f"{recall_str}"
            )

            val_loss = val_metrics["loss"]
            if val_loss < best_loss:
                best_loss = val_loss
                no_improve = 0
                torch.save(self.model.state_dict(), Path(ckpt_dir) / "best.pt")
                print(f"    -> saved (loss={best_loss:.4f})")
            else:
                no_improve += 1

            if epoch >= min_epochs_before_stop and no_improve >= patience:
                print(f"    Early stopping (no improvement for {patience} epochs)")
                break

    def _train_flat_epoch(
        self, dataset: Dataset, optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        loader = DataLoader(
            dataset, batch_size=self.config.train.batch_size,
            shuffle=True, num_workers=0,
        )
        total_loss = 0.0
        count = 0

        pbar = tqdm(loader, desc="  Training", leave=False, unit="batch")
        for batch in pbar:
            grid_8x8 = batch["grid_8x8"].unsqueeze(1).to(self.device)
            positions = batch["positions"].to(self.device)
            levels = batch["level"].to(self.device)
            act_label = batch["activation"].to(self.device)

            act = self.model(grid_8x8, positions, levels)
            loss, _ = self.criterion(act, act_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = grid_8x8.size(0)
            total_loss += loss.item() * n
            count += n
            pbar.set_postfix(loss=f"{total_loss / count:.4f}")

        pbar.close()
        return total_loss / max(count, 1)

    @torch.no_grad()
    def _validate_flat(self, dataset: FlatDataset) -> dict[str, float]:
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.config.train.batch_size,
            shuffle=False, num_workers=0,
        )
        total_loss = 0.0
        tp = fp = fn = 0
        count = 0

        for batch in tqdm(loader, desc="  Validating", leave=False, unit="batch"):
            grid_8x8 = batch["grid_8x8"].unsqueeze(1).to(self.device)
            positions = batch["positions"].to(self.device)
            levels = batch["level"].to(self.device)
            act_label = batch["activation"].to(self.device)

            act = self.model(grid_8x8, positions, levels)
            loss, _ = self.criterion(act, act_label)

            n = grid_8x8.size(0)
            total_loss += loss.item() * n
            count += n

            pred_bin = (act > 0.5).float()
            tp += (pred_bin * act_label).sum().item()
            fp += (pred_bin * (1.0 - act_label)).sum().item()
            fn += ((1.0 - pred_bin) * act_label).sum().item()

        return {
            "loss": total_loss / max(count, 1),
            "recall": tp / max(tp + fn, 1e-8),
            "precision": tp / max(tp + fp, 1e-8),
        }

    # ------------------------------------------------------------------
    # End-to-end recursive validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_recursive(self, dataset: RecursiveDataset) -> float:
        self.model.eval()
        tp = fn = 0
        n_val = min(len(dataset), 100)

        for idx in tqdm(range(n_val), desc="  E2E Validation", leave=False, unit="ex"):
            sample = dataset[idx]
            grid_data = sample["grid"]
            gs = grid_data.shape[0]
            source = (int(sample["source"][0]), int(sample["source"][1]))
            goal = (int(sample["goal"][0]), int(sample["goal"][1]))
            path_arr = sample["path"]
            path_cells = {(int(r), int(c)) for r, c in path_arr}

            active = recursive_neural_inference(
                self.model, grid_data, gs, gs, source, goal,
            )

            for pc in path_cells:
                if pc in active:
                    tp += 1
                else:
                    fn += 1

        return tp / max(tp + fn, 1e-8)

    # ------------------------------------------------------------------
    # Phase 2 — adversarial mining
    # ------------------------------------------------------------------

    def _phase_adversarial(
        self,
        rounds: int,
        queries_per_round: int,
        lr: float,
        ckpt_dir: str,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for rnd in range(1, rounds + 1):
            flat_examples = self._mine_hard_examples(queries_per_round)
            if not flat_examples:
                print(f"  Round {rnd}: no hard examples — stopping")
                break

            print(f"  Round {rnd}: training on {len(flat_examples)} flat examples")
            ds = _InMemoryFlatDataset(flat_examples)
            self._train_flat_epoch(ds, optimizer)
            torch.save(self.model.state_dict(), Path(ckpt_dir) / "best.pt")

    def _mine_hard_examples(
        self, num_queries: int,
    ) -> list[dict[str, np.ndarray]]:
        """Run inference, verify with BFS, extract flat examples from failures."""
        self.model.eval()
        flat_examples: list[dict[str, np.ndarray]] = []
        rng = np.random.RandomState()
        n_hard = 0
        grid_res = self.config.neural.grid_resolution

        for _ in tqdm(range(num_queries), desc="  Mining", leave=False, unit="q"):
            gs = int(rng.choice([32, 64, 128, 256]))
            dens = float(rng.choice([0.1, 0.2, 0.3]))
            grid = generate_grid(
                gs, gs, dens, seed=int(rng.randint(0, 2**31)),
                ensure_connected=False,
            )
            free = list(zip(*np.where(grid.data == 0)))
            if len(free) < 2:
                continue

            si, gi = rng.choice(len(free), size=2, replace=False)
            source = Cell(int(free[si][0]), int(free[si][1]))
            goal = Cell(int(free[gi][0]), int(free[gi][1]))

            bfs_result = bfs_shortest_path(grid, source, goal)
            if bfs_result is None:
                continue

            bfs_path, _ = bfs_result
            active = recursive_neural_inference(
                self.model, grid.data, gs, gs,
                (source.row, source.col), (goal.row, goal.col),
            )

            path_cells = {(c.row, c.col) for c in bfs_path}
            missed = path_cells - active
            if missed:
                num_levels = max(int(math.ceil(math.log2(max(gs, 2)))), 1)
                padded_size = 1 << num_levels
                padded = np.ones((padded_size, padded_size), dtype=np.uint8)
                padded[:gs, :gs] = grid.data

                node_examples = extract_flat_labels(
                    padded, bfs_path, source, goal, padded_size,
                    grid_resolution=grid_res,
                )
                flat_examples.extend(node_examples)
                n_hard += 1

            if n_hard >= 500:
                break

        return flat_examples
