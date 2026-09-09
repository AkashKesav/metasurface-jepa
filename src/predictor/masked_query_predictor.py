"""Student masked-query predictor — predicts the joint target latent only at
masked spatial locations.

Implements the student side of the Joint Target Redesign
(`docs/JOINT_TARGET_REDESIGN.md` §4, adopted 2026-09-08).

Only masked spatial tokens are predicted. For every masked token a query is
built from a learned mask token + its position embedding + a scalar summary,
then a self-attention + cross-attention-to-visible-context stack produces the
predicted joint latent for that masked position. The predictions are scattered
back to their original spatial positions to form `Z_pred_joint [B, 256, hidden]`;
visible positions keep the context representation unchanged in the first
implementation.

    q_i = mask_token + position_i + scalar_summary           (per masked token)
    self-attn over masked queries
    cross-attn(Q = masked queries, KV = visible geometry + physics goal)
    FFN
    -> predicted masked latent tokens

Mask convention (preserved exactly): **1 = visible, 0 = masked**. The query
index <-> spatial token index mapping is preserved exactly so the scatter-back
is lossless and the JEPA loss (computed only on masked positions) aligns
predicted tokens with the corresponding target tokens.

Stage-A scope (§9 Stage A): joint target only — this predictor is trained
against `Z_joint` with `L_JEPA` alone; the decoder / physics / goal-sensitivity
terms come later (Stages B-D). This module therefore emits only the predicted
joint latent (no geometry / spectrum output); the decoder that consumes it is a
separate Stage-B component.

Shapes (default unified line, hidden=192, 256 = 16x16 spatial tokens):
    occ       : (B, 1, 64, 64)          occupancy (the geometry mask lives here)
    mask      : (B, 256)               1 = visible token, 0 = masked token
    z_visible : (B, N_vis, hidden)     context (geometry) tokens at visible pos
    a_goal    : (B, 16, hidden)         physics goal tokens
    c_scalar  : (B, hidden)             FiLM scalar summary (known scalars)
    out       : (B, 256, hidden)        Z_pred_joint — full spatial grid,
                                        visible positions == z_visible scattered
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MaskedQueryPredictor"]


class MaskedQueryPredictor(nn.Module):
    """Predict the joint target latent at masked spatial tokens.

    Transformer stack: self-attention over masked queries, then cross-
    attention from masked queries to (visible geometry + physics goal) keys/
    values, then an FFN. Predictions are scattered back to their spatial
    positions; visible positions keep the context representation.

    Parameters
    ----------
    hidden : int
        Token embedding dimension (192 on the unified line).
    num_heads : int
        Attention head count for both self- and cross-attention.
    ffn_mult : int
        FFN hidden multiplier (FFN hidden = ffn_mult * hidden).
    num_layers : int
        Number of (self-attn -> cross-attn -> FFN) blocks.
    dropout : float
        Dropout in attention and FFN.
    n_spatial_tokens : int
        Total spatial token count (256 on the unified line) — used only to size
        the learned positional embedding. The mask can mark any subset masked.
    """

    def __init__(self, hidden: int = 192, num_heads: int = 6,
                 ffn_mult: int = 4, num_layers: int = 2, dropout: float = 0.0,
                 n_spatial_tokens: int = 256):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"MaskedQueryPredictor: hidden ({hidden}) must be "
                             f"divisible by num_heads ({num_heads})")
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.num_layers = num_layers
        self.n_spatial_tokens = n_spatial_tokens
        # Learnable mask token (broadcast to every masked query) and positional
        # embedding so each masked query carries its spatial position.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_spatial_tokens, hidden))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Project the scalar summary into the query space (FiLM-style add).
        self.scalar_to_q = nn.Linear(hidden, hidden)
        # KV projection for the cross-attention context (visible geom + goal).
        self.ctx_kv = nn.Linear(hidden, 2 * hidden)
        self.blocks = nn.ModuleList([
            _PredBlock(hidden, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(hidden)

    def forward(self, z_visible: torch.Tensor, a_goal: torch.Tensor,
                mask: torch.Tensor, c_scalar: torch.Tensor
                ) -> torch.Tensor:
        """Predict the joint latent at masked positions, scatter to full grid.

        Parameters
        ----------
        z_visible : (B, N_vis, hidden)
            Context (geometry) tokens AT THE VISIBLE POSITIONS, in visible
            order (length N_vis = mask.sum(1)[b] per batch — must be a dense
            packed tensor of the visible tokens, not the full grid).
        a_goal : (B, N_goal, hidden)
            Physics goal tokens (16 on the unified line).
        mask : (B, N_spatial)
            1 = visible, 0 = masked. Booleans/0-1 ints accepted. N_spatial must
            equal self.n_spatial_tokens.
        c_scalar : (B, hidden)
            FiLM scalar summary (known scalars projected to hidden).

        Returns
        -------
        z_pred_joint : (B, N_spatial, hidden)
            Predicted joint latent on the full spatial grid. At masked positions
            this is the predictor output; at visible positions it is the input
            z_visible scattered back unchanged (first-implementation choice per
            §4: "For visible positions, preserve the context representation
            unchanged").
        """
        b = mask.shape[0]
        n_spatial = mask.shape[1]
        if n_spatial != self.n_spatial_tokens:
            raise ValueError(
                f"mask spatial dim {n_spatial} != n_spatial_tokens "
                f"{self.n_spatial_tokens}")
        if z_visible.dim() != 3 or z_visible.shape[-1] != self.hidden:
            raise ValueError(f"z_visible must be (B, N_vis, {self.hidden}), got "
                             f"{tuple(z_visible.shape)}")
        if a_goal.dim() != 3 or a_goal.shape[-1] != self.hidden:
            raise ValueError(f"a_goal must be (B, N_goal, {self.hidden}), got "
                             f"{tuple(a_goal.shape)}")
        if c_scalar.dim() != 2 or c_scalar.shape[-1] != self.hidden:
            raise ValueError(f"c_scalar must be (B, {self.hidden}), got "
                             f"{tuple(c_scalar.shape)}")

        mask_b = mask.bool() if mask.dtype != torch.bool else mask
        # Per-batch masked-token counts (Stage-A: assume identical mask shape
        # across batch — the BlockMasker produces a single mask grid broadcast
        # across the batch). Validate and broadcast.
        n_masked_per = (~mask_b).sum(dim=1)
        # torch.equal needs same shape; compare every value to the first.
        n0 = int(n_masked_per[0].item())
        if not bool(torch.all(n_masked_per == n0)):
            raise NotImplementedError(
                "MaskedQueryPredictor (Stage A) requires a single mask shape "
                f"broadcast across the batch; got per-batch masked counts "
                f"{n_masked_per.tolist()}. Per-sample masks arrive with Stage F.")
        n_masked = n0

        # Build the masked queries: mask_token + position + scalar summary.
        # Gather the positional embeddings of the masked positions (same set
        # for every batch row since the mask is broadcast).
        masked_idx = (~mask_b[0]).nonzero(as_tuple=False).squeeze(-1)  # (n_masked,)
        pos = self.pos_embed[0, masked_idx]                              # (n_masked, H)
        scalar_bias = self.scalar_to_q(c_scalar).unsqueeze(1)           # (B, 1, H)
        q = self.mask_token.expand(b, n_masked, -1) + pos.unsqueeze(0) \
            + scalar_bias.expand(b, n_masked, -1)                       # (B, n_mask, H)

        # Cross-attention KV: visible geometry tokens + physics goal tokens,
        # packed along the sequence axis (per §4 K,V = visible + goal).
        kv = torch.cat([z_visible, a_goal], dim=1)                     # (B, N_vis+N_goal, H)

        # Predict.
        for blk in self.blocks:
            q = blk(q, kv)
        z_masked = self.norm_out(q)                                      # (B, n_mask, H)

        # Scatter back to the full spatial grid. Visible positions get the
        # context representation (z_visible) unchanged; masked positions get
        # the predicted tokens. z_visible is packed by visible position; the
        # visible positions are the complement of masked_idx.
        visible_idx = mask_b[0].nonzero(as_tuple=False).squeeze(-1)     # (N_vis,)
        out = z_visible.new_zeros(b, n_spatial, self.hidden)
        # Broadcast scatter across batch.
        out[:, masked_idx] = z_masked
        out[:, visible_idx] = z_visible
        return out


class _PredBlock(nn.Module):
    """One predictor block: self-attn over masked queries -> cross-attn(Q=
    masked, KV=context) -> FFN, each with a residual + LayerNorm (pre-norm)."""

    def __init__(self, hidden: int, num_heads: int, ffn_mult: int,
                 dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.self_attn = nn.MultiheadAttention(hidden, num_heads,
                                               dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(hidden)
        ffn_hidden = ffn_mult * hidden
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Self-attention among masked queries (queries attend to each other so
        # a masked token can use the joint prediction of neighbouring masked
        # tokens — matches §4 "self-attention" before cross-attention).
        a, _ = self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q),
                              need_weights=False)
        q = q + a
        # Cross-attention: masked queries attend to visible geometry + goal.
        a, _ = self.cross_attn(self.norm2(q), kv, kv, need_weights=False)
        q = q + a
        q = q + self.ffn(self.norm3(q))
        return q
