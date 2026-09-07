import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

import torch

from predictor.goal_residual import GoalResidualRoute


def test_goal_residual_route_is_masked_and_trainable():
    torch.manual_seed(7)
    route = GoalResidualRoute(hidden=12, goal_dim=24, num_heads=3)
    z_base = torch.randn(2, 5, 12, requires_grad=True)
    a_goal = torch.randn(2, 4, 24)
    masked = torch.tensor(
        [[True, False, True, False, False],
         [False, True, False, True, False]])

    residual = route(z_base, a_goal, masked)

    assert residual.shape == z_base.shape
    assert torch.allclose(residual[~masked], torch.zeros_like(residual[~masked]))
    residual.square().mean().backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in route.parameters()
    )
