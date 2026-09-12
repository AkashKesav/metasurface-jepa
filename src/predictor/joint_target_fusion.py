"""Joint target fusion — Z_joint = J(Z_G, Z_S).

Implements the central scientific change of the Joint Target Redesign
(`docs/JOINT_TARGET_REDESIGN.md` §3, adopted 2026-09-08).

The previous unified target was geometry-only:

    z_target = EMA_geometry_encoder(G_true)          # [B, 256, hidden]

A geometry-only target can only ever *reward* spectrum use by the student, never
*require* it (§14 of `UNIFIED_ORIGINAL_ARCHITECTURE.md`; four goal-conditioning
attempts on that target failed or were rejected). The redesign couples the
geometry target with the electromagnetic (spectrum) representation through
explicit cross-attention so the target latent genuinely depends on the target
spectrum:

    Z_G = EMA_geometry_encoder(G_true)              # [B, 256, hidden]   (geometry)
    Z_S = A_goal = frozen_spectrum_encoder(S_true)  # [B, K, hidden]    (physics)
    delta = cross_attn(query=Z_G, key=Z_S, value=Z_S)
    Z_joint = Z_G + tanh(gate) * delta

The zero-initialized scalar gate (`gate` parameter, init 0.0) gives a stable
start (Z_joint == Z_G at step 0, so the JEPA loss is finite and the EMA/target
path is identical to the old geometry-only target at initialization) while
letting the teacher learn a physics-conditioned structural representation as
the gate opens. This is deliberately the *minimum* coupling mechanism — a
two-way / reciprocal attention or sparse top-k routing belongs to §13
("do not add yet") and must require a measured failure first.

Teacher invariants (enforced by the caller, not this module): EMA geometry
weights, frozen spectrum encoder, stop-gradient, eval mode. This module only
implements the fusion math; it holds the single learnable `gate` parameter and
a multi-head cross-attention. It is meant to be invoked inside a `no_grad`
target-construction context by the teacher, but its `gate` (and attention
parameters) ARE trainable — gradients to them must flow, so the teacher's
`no_grad` context must be opened by the caller *around* the frozen encoders
only, not around this fusion. (Per §3 target-path pseudocode the whole block is
under `no_grad`; in that pseudocode the fusion parameters are treated as
non-teacher and updated separately. The clean split is implemented by the
training loop, not here.)

Shapes (default unified line, hidden=192):
    Z_G: (B, 256, hidden)           geometry tokens (256 = 16x16 patch grid)
    Z_S: (B, K, hidden)             physics/goal tokens (K=16 from SpectrumPath)
    Z_joint: (B, 256, hidden)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["JointTargetFusion"]


class JointTargetFusion(nn.Module):
    """Z_joint = J(Z_G, Z_S) = Z_G + tanh(gate) * CrossAttn(Q=Z_G, KV=Z_S).

    A single zero-initialized scalar gate controls how much the spectrum-
    conditioned delta modifies the geometry target. At init (gate=0) the
    output is bit-identical to the geometry-only target, preserving the
    stable-initialization property of the old JEPA target while opening a
    learnable physics-coupling path.

    Parameters
    ----------
    hidden : int
        Token embedding dimension (192 on the unified line). Must match both
        Z_G and Z_S feature dims — the cross-attention is single-space.
    num_heads : int
        Attention head count. head_dim = hidden // num_heads must be >= 1.
    dropout : float
        Attention dropout. 0 by default (target path is eval/no_grad anyway;
        keep 0 so the target is deterministic per-sample).
    """

    def __init__(self, hidden: int = 192, num_heads: int = 6, dropout: float = 0.0):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(
                f"JointTargetFusion: hidden ({hidden}) must be divisible by "
                f"num_heads ({num_heads})")
        self.hidden = hidden
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        # Single-space cross-attention (geometry queries, physics keys/values).
        self.q_proj = nn.Linear(hidden, hidden, bias=True)
        self.k_proj = nn.Linear(hidden, hidden, bias=True)
        self.v_proj = nn.Linear(hidden, hidden, bias=True)
        self.out_proj = nn.Linear(hidden, hidden, bias=True)
        self.dropout = nn.Dropout(dropout)
        # Zero-initialized scalar gate: at construction Z_joint == Z_G exactly.
        # tanh(0) = 0 -> delta contributes nothing until training opens the gate.
        self.gate = nn.Parameter(torch.zeros(()))
        # Stage-A remediation (STAGE_A_VERDICT §8.1): bound the physics delta
        # so an open gate cannot swamp Z_G. norm_delta normalizes the
        # cross-attention OUTPUT only — the residual Z_G passes through
        # unmodified, so gate=0 still gives bit-identical Z_G.
        self.norm_delta = nn.LayerNorm(hidden)
        # Pre-norms on the attention INPUTS only (geometry is the residual
        # stream and passes through unmodified, so at gate=0 the output is
        # bit-identical to Z_G per §3 "Initialize gate = 0 ... stable
        # initialization"). Normalizing the residual stream itself would break
        # that identity — do not add an output norm here.
        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)

    def forward(self, z_g: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        """Compute the joint target latent.

        Parameters
        ----------
        z_g : (B, N_g, hidden)
            Geometry tokens from the EMA geometry encoder (the teacher path).
        z_s : (B, N_s, hidden)
            Physics/goal tokens from the frozen released spectrum encoder
            (SpectrumPath's A_goal, (B, 16, hidden)).

        Returns
        -------
        z_joint : (B, N_g, hidden)
            Same shape as z_g. At init (gate == 0) this equals z_g.
        """
        if z_g.dim() != 3:
            raise ValueError(f"z_g must be (B, N_g, hidden), got {tuple(z_g.shape)}")
        if z_s.dim() != 3:
            raise ValueError(f"z_s must be (B, N_s, hidden), got {tuple(z_s.shape)}")
        if z_g.shape[-1] != self.hidden:
            raise ValueError(f"z_g feature dim {z_g.shape[-1]} != hidden {self.hidden}")
        if z_s.shape[-1] != self.hidden:
            raise ValueError(f"z_s feature dim {z_s.shape[-1]} != hidden {self.hidden}")
        b, n_g, _ = z_g.shape
        n_s = z_s.shape[1]
        # Single-space cross-attention: Q from geometry, KV from physics.
        q = self.q_proj(self.norm_q(z_g))
        k = self.k_proj(self.norm_kv(z_s))
        v = self.v_proj(self.norm_kv(z_s))
        q = q.reshape(b, n_g, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n_s, self.num_heads, self.head_dim).transpose(1, 2)
        # SDPA: full attention (no causal/local mask — every geometry token
        # attends to every physics token). K=16 is tiny, so cost is negligible.
        delta = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        delta = delta.transpose(1, 2).reshape(b, n_g, self.hidden)
        delta = self.out_proj(delta)
        # Bound delta scale (see __init__): LayerNorm on delta only keeps
        # ||delta|| O(sqrt(hidden)) per token once the gate opens.
        delta = self.norm_delta(delta)
        delta = self.dropout(delta)
        # Gated residual: at init gate=tanh(0)=0 -> z_joint == z_g exactly
        # (bit-identical to the geometry-only target — the §3 stable-init
        # property). No output norm: it would renormalize Z_G at gate=0 and
        # break the identity.
        return z_g + torch.tanh(self.gate) * delta

    def extra_repr(self) -> str:
        return (f"hidden={self.hidden}, num_heads={self.num_heads}, "
                f"head_dim={self.head_dim}, gate_init=0.0")
