"""Direct spectrum-to-masked-query residual route for unified JEPA."""

import torch
from torch import nn

from encoders.geometry_encoder import CrossAttention


class GoalResidualRoute(nn.Module):
    """Cross-attend base occupancy latents directly to spectrum goal tokens."""

    def __init__(self, hidden=192, goal_dim=384, num_heads=6):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.goal_proj = nn.Linear(goal_dim, hidden)
        self.cross_attn = CrossAttention(hidden, num_heads)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden, hidden),
        )
        nn.init.normal_(self.out[-1].weight, std=0.02)
        nn.init.zeros_(self.out[-1].bias)
        self.goal_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_base, a_goal, masked_tokens):
        q = self.query_norm(z_base)
        kv = self.goal_proj(a_goal)
        delta, _ = self.cross_attn(q, kv)
        delta = self.out(delta)
        return self.goal_scale * delta * masked_tokens.to(delta.dtype).unsqueeze(-1)

