"""Loss function for the QuadTreeConvNet corridor predictor.

L = weighted_BCE(activation, activation_label)

Positive examples (active quadrants) are weighted higher to favor recall
over precision — missing a path cell is much worse than including an
extra cell in the corridor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CorridorLoss(nn.Module):
    """Weighted binary cross-entropy on quadrant activations."""

    def __init__(self, pos_weight: float = 5.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(
        self,
        activation: torch.Tensor,
        activation_label: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            activation:       (B, 4) predicted sigmoid probabilities
            activation_label: (B, 4) binary ground truth

        Returns:
            loss, metrics_dict
        """
        weight = activation_label * self.pos_weight + (1.0 - activation_label)
        loss = F.binary_cross_entropy(
            activation, activation_label, weight=weight, reduction="mean",
        )

        return loss, {"loss": loss.item()}
