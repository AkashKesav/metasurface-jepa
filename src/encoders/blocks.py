"""Shared transformer building blocks (moved out of `encoders/geometry_encoder.py`).

These classes are not geometry-specific: the active 192-D unified architecture uses
them verbatim — `encoders/occupancy_encoder.py` (TransformerBlock + pos-embed),
`encoders/fusion_encoder.py` (TransformerBlock), and `predictor/gclct.py`
(Attention / CrossAttention). They previously lived in `geometry_encoder.py`,
whose `GeometryEncoder` belonged to the retired 384-D path; the shared blocks were
extracted here so the 192-D path does not depend on that file.

Parameter layout matches the released MetaDiT DiTBlock minus the adaLN path
(affine-less LayerNorms, eps=1e-6) so released weights transfer exactly — the
convention the legacy 384-D encoder relied on and the fused/occupancy stacks keep.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """(grid_size*grid_size, embed_dim) 2-D sincos positional embedding (MAE-style)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(grid_size, dtype=np.float64)
    out_h = np.einsum("m,d->md", pos, omega)
    out_w = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate(
        [np.repeat(out_h, grid_size, axis=0), np.tile(out_w, (grid_size, 1))], axis=1)
    return emb


class Attention(nn.Module):
    """timm-style multi-head self-attention with qk-norm (matches released DiTBlock.attn)."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(b, n, self.head_dim * self.num_heads))


class CrossAttention(nn.Module):
    """Standard cross-attention: Q from query tokens, K/V from context tokens."""

    def __init__(self, dim, num_heads, qkv_bias=True, qk_norm=True):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Q comes ONLY from query tokens.
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # K and V come ONLY from context tokens.
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x, kv, need_weights=False):
        """
        x  : [B, Nq, D]   query tokens
        kv : [B, Nk, D]   context tokens

        returns:
            out     : [B, Nq, D]
            weights : [B, H, Nq, Nk] if requested, else None
        """
        B, Nq, D = x.shape
        Nk = kv.shape[1]

        # --------------------------------------------------
        # Separate Q / K / V projections
        # --------------------------------------------------
        q = self.q(x)       # [B, Nq, D]
        k = self.k(kv)      # [B, Nk, D]
        v = self.v(kv)      # [B, Nk, D]

        # --------------------------------------------------
        # Split heads
        # --------------------------------------------------
        q = q.reshape(
            B, Nq, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B,H,Nq,d]

        k = k.reshape(
            B, Nk, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B,H,Nk,d]

        v = v.reshape(
            B, Nk, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)  # [B,H,Nk,d]

        # --------------------------------------------------
        # Q/K normalization
        # --------------------------------------------------
        q = self.q_norm(q)
        k = self.k_norm(k)

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------
        if need_weights:
            scores = (
                q @ k.transpose(-2, -1)
            ) / math.sqrt(self.head_dim)

            weights = torch.softmax(scores, dim=-1)
            out = weights @ v
        else:
            weights = None
            out = F.scaled_dot_product_attention(
                q, k, v
            )

        # --------------------------------------------------
        # Merge heads
        # --------------------------------------------------
        out = out.permute(0, 2, 1, 3).reshape(
            B, Nq, D
        )

        out = self.proj(out)

        return out, weights


class TransformerBlock(nn.Module):
    """Plain pre-norm block; parameter layout matches released DiTBlock (no adaLN)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
