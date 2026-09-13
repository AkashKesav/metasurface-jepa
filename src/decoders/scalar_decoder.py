"""Scalar decode heads (architecture_v5.md §3.6/§4.2).

Three small MLP heads reading the predicted scalar-summary latent → (l, h, r).
Loss: plain L1 (or Huber) regression against ground truth, only on unknown
scalar positions. This head reads z_hat's scalar-summary query, NOT a latent
target — there is no scalar EMA loss target (§3.6 EMA rules).
"""

import torch
from torch import nn


class ScalarDecoder(nn.Module):
    """Decode the hidden-D scalar-summary query prediction into (l, h, r).

    Each scalar gets its own small MLP head. Initialization is deliberately
    NON-ZERO at the output: the final weight is zero-initialized (the head
    starts as a constant predictor) and the final bias is set to the dataset
    means, so at step 0 each head predicts its mean scalar — a non-zero,
    in-distribution geometry for the frozen surrogate. A fully zero-initialized
    head would emit all-zero scalars, collapsing the geometry and leaving the
    surrogate's ReLU6 activations in a dead zone (zero Jacobian), which would
    kill gradient flow through the student encoder (Phase 4 MD §3: "NO
    zero-init on this one"). Audit B15: the previous docstring claimed a
    "default Kaiming-uniform final layer", which the code never did.
    """

    def __init__(self, hidden=192, mlp_hidden=64, n_scalars=3):
        super().__init__()
        self.n_scalars = n_scalars
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, 1),
            )
            for _ in range(n_scalars)
        ])
        # Final layer: initialize bias to the approximate MetaDiT dataset means
        # (l_lattice ≈ 2.75, h_atom ≈ 0.75, r_atom ≈ 4.25, from external/metadit/datapipe.py)
        # so the initial geometry is non-zero and within the surrogate's training
        # distribution. Zero/empty init would collapse geometry to all-zeros,
        # and the surrogate's ReLU6 activations have zero Jacobian at zero input.
        init_biases = [2.75, 0.75, 4.25]
        for i, head in enumerate(self.heads):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, init_biases[i])

    def forward(self, scalar_summary_pred):
        """scalar_summary_pred: (B, hidden) → scalars: (B, n_scalars)."""
        return torch.stack(
            [head(scalar_summary_pred).squeeze(-1) for head in self.heads],
            dim=-1,
        )
