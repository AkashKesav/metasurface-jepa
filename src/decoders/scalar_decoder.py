"""Scalar decode heads (architecture_v5.md §3.6/§4.2).

Three small MLP heads reading the predicted scalar-summary latent → (l, h, r).
Loss: plain L1 (or Huber) regression against ground truth, only on unknown
scalar positions. This head reads z_hat's scalar-summary query, NOT a latent
target — there is no scalar EMA loss target (§3.6 EMA rules).
"""

import math

import torch
from torch import nn


class ScalarDecoder(nn.Module):
    """Decode the hidden-D scalar-summary query prediction into (l, h, r).

    Each scalar gets its own small MLP head. Final layer uses the default
    PyTorch init (Kaiming-uniform) rather than zero-init: the scalar outputs
    feed directly into assemble_metadit_geometry → surrogate, so zero scalars
    at init produce all-zero geometry and the surrogate's ReLU6 activations are
    in a dead zone (zero Jacobian), killing gradient flow through the entire
    student encoder. Non-zero init ensures a non-zero geometry for the surrogate
    to produce a non-trivial Jacobian (Phase 4 MD §3: "NO zero-init on this one").
    """

    def __init__(self, hidden=192, mlp_hidden=64, n_scalars=3,
                 bounds=None):
        super().__init__()
        self.n_scalars = n_scalars
        if bounds is None:
            bounds = ((2.5, 3.0), (0.5, 1.0), (3.5, 5.0))
        if len(bounds) != n_scalars:
            raise ValueError("scalar bounds must contain one (lo, hi) pair per scalar")
        bounds = torch.as_tensor(bounds, dtype=torch.float32)
        if bounds.shape != (n_scalars, 2) or not torch.isfinite(bounds).all():
            raise ValueError("scalar bounds must have finite shape [n_scalars, 2]")
        if not torch.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("each scalar bound must satisfy hi > lo")
        self.register_buffer("bounds", bounds)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, 1),
            )
            for _ in range(n_scalars)
        ])
        # Final layer: initialize bias in LOGIT space so sigmoid maps back to
        # the dataset means. raw = bias -> sigmoid(bias) must equal
        # (mean-lo)/(hi-lo) = 0.5 for all three scalars here, i.e. bias ~= 0.
        # Setting bias to physical means (2.75/0.75/4.25) saturates the sigmoid
        # (e.g. sigmoid(2.75)=0.94 -> l_lattice init 2.97 near hi) and puts
        # init geometry at the bound edge, off the surrogate distribution.
        means = [2.75, 0.75, 4.25]
        bounds_list = bounds.tolist()
        for i, head in enumerate(self.heads):
            last = list(head.children())[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            lo_i, hi_i = float(bounds_list[i][0]), float(bounds_list[i][1])
            p = min(max((means[i] - lo_i) / (hi_i - lo_i), 1e-6), 1.0 - 1e-6)
            assert last.bias is not None
            nn.init.constant_(last.bias, math.log(p / (1.0 - p)))

    def forward(self, scalar_summary_pred):
        """scalar_summary_pred: (B, hidden) → scalars: (B, n_scalars)."""
        raw = torch.stack(
            [head(scalar_summary_pred).squeeze(-1) for head in self.heads], dim=-1)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * torch.sigmoid(raw)
