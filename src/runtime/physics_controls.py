"""Centralized physics-conditioning controls (hardening spec §8).

Canonical real/null/shuffled-spectrum controls with derangement for shuffled.
Canonical metric names for consistent reporting across evaluators.
"""

import torch
from runtime.device import resolve_device


def derangement_permutation(batch_size: int, device: torch.device | str,
                            seed: int | None = None,
                            generator: torch.Generator | None = None) -> torch.Tensor:
    """Generate a derangement permutation (perm[i] != i for all i).

    A derangement is a permutation with no fixed points. This ensures that
    in the shuffled-spectrum control, no sample receives its own spectrum.

    Args:
        batch_size: Number of samples (must be >= 2).
        device: Target device for the permutation tensor.
        seed: Optional seed for reproducibility (creates new generator).
        generator: Optional existing generator to use.

    Returns:
        Tensor of shape (batch_size,) with deranged indices.

    Raises:
        ValueError: If batch_size < 2.
    """
    if batch_size < 2:
        raise ValueError("derangement requires batch_size >= 2")

    device = resolve_device(device)
    if generator is None:
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

    # torch.randperm requires the generator's device to match the draw device,
    # and a caller may legitimately hand over a CPU generator together with a
    # CUDA target — the evaluator's seeded shuffled control does exactly that
    # (audit B12). Forwarding the TARGET device raised
    #   RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'
    # and killed the whole acceptance-gate evaluation on the GPU run (audit B22).
    # Draw on the generator's own device and move the permutation to the target:
    # the generator's stream is untouched by the move, so a fixed seed stays
    # reproducible.
    draw_device = generator.device
    positions = torch.arange(batch_size, device=device)

    # Simple rejection sampling for derangement
    # For small batch sizes this is efficient; for large sizes use more sophisticated algorithms
    max_attempts = 100
    for _ in range(max_attempts):
        perm = torch.randperm(batch_size, generator=generator,
                              device=draw_device).to(device)
        if not torch.any(perm == positions):
            return perm
    # Fallback: cyclic shift (guaranteed derangement for n >= 2)
    return torch.roll(positions, shifts=1)


def derange_batch_tensor(X: torch.Tensor, generator: torch.Generator | None = None,
                         seed: int | None = None) -> torch.Tensor:
    """Generalized batch-tensor derangement control (cleanup item 4).

    output[i] = X[perm[i]] with perm[i] != i for all i.

    Applies to any leading-dimension tensor (spectra, scalars, ...): the
    batch axis is deranged, trailing dimensions are carried along.

    Args:
        X: Tensor of shape (B, ...).
        generator: Optional RNG generator.
        seed: Optional seed.

    Returns:
        Deranged tensor X_shuf where X_shuf[i] = X[perm[i]] and perm is a
        derangement.

    Raises:
        ValueError: If B < 2 — no valid derangement exists (explicit
            infeasible result, never a silent identity).
    """
    b = X.shape[0]
    if b < 2:
        raise ValueError(
            f"derangement requires batch size >= 2 (got {b}): no valid "
            "derangement exists for a single sample")
    perm = derangement_permutation(b, X.device, seed=seed, generator=generator)
    return X[perm]


def make_shuffled_spectrum(S: torch.Tensor, generator: torch.Generator | None = None,
                           seed: int | None = None) -> torch.Tensor:
    """Create shuffled spectrum tensor with derangement.

    Thin wrapper around the generic derange_batch_tensor (cleanup item 4).

    B < 2 (no valid derangement): preserved existing behavior — returns the
    input unchanged. Callers that need an EXPLICIT infeasible result (e.g.
    evaluators that must not silently present an identity shuffle as a valid
    control) should guard on B < 2 themselves, exactly as they already do
    (real_null_shuffled / scalar_dependence report "shuffled_infeasible").

    Args:
        S: Spectrum tensor of shape (B, ...).
        generator: Optional RNG generator.
        seed: Optional seed.

    Returns:
        Shuffled spectrum tensor S_shuf where S_shuf[i] = S[perm[i]] and perm is a derangement.
    """
    b = S.shape[0]
    if b < 2:
        return S  # preserved existing B<2 behavior: no derangement possible
    return derange_batch_tensor(S, generator=generator, seed=seed)


# Canonical metric names for physics conditioning (hardening spec §8)
PHYSICS_METRICS = {
    "L_real": "raw_jepa_loss_real",
    "L_null": "raw_jepa_loss_null",
    "L_shuffled": "raw_jepa_loss_shuffled",
    "gap_null": "utility_gap_null",
    "gap_shuffled": "utility_gap_shuffled",
    "sensitivity_null": "predictor_sensitivity_real_vs_null",
    "sensitivity_shuffled": "predictor_sensitivity_real_vs_shuffled",
}


def validate_goal_mode(goal_mode: str) -> None:
    """Validate goal_mode is one of the allowed values.

    Args:
        goal_mode: String to validate.

    Raises:
        ValueError: If goal_mode not in {"real", "null"}.
    """
    if goal_mode not in ("real", "null"):
        raise ValueError(
            f"goal_mode must be 'real' or 'null', got {goal_mode!r}"
        )