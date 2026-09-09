"""Top-level model assembly (§11.1 data-flow spec; Milestone B slice).

Variant 'jepa'  — GoalConditionedJEPA: block-masked context -> Ẑ_y against EMA target
                   latent, L = L_J only (§4.1 Phase 2).

Frozen released components stay outside the trainable state: the released spectrum encoder
keys are filtered from saved checkpoints (re-loaded from disk on every build), and the EM
surrogate / released DiT are constructed by the training script on demand.

The historical direct masked generator (Baseline 2, §10.1) is not part of this module;
it lives as a self-contained reference implementation in `src/reference/` and is
unreachable from the active training path.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
METADIT_SRC = os.path.join(REPO_ROOT, "external", "metadit")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if METADIT_SRC not in sys.path:
    sys.path.insert(0, METADIT_SRC)

import torch
import torch.nn.functional as F
from torch import nn

from data.mask import apply_mask_to_pixels
from encoders.context_encoder import ContextEncoder
from encoders.geometry_encoder import GeometryEncoder
from encoders.occupancy_encoder import OccupancyEncoder
from encoders.scalar_encoder import ScalarEncoder
from fusion.fusion_encoder import FusionEncoder
from decoders.scalar_decoder import ScalarDecoder
from decoders.occupancy_decoder import OccupancyDecoder
from data.factorize import assemble_metadit_geometry
from encoders.spectrum_encoder import ReleasedSpectrumEncoder, SpectrumPath
from encoders.target_encoder import EMAEncoder
from losses.jepa_loss import jepa_loss
from predictor.gclct import GCLCT
from predictor.joint_target_fusion import JointTargetFusion
from predictor.masked_query_predictor import MaskedQueryPredictor
# goal_residual route RETIRED 2026-09-08 (Joint Target Redesign supersedes it).

PIXEL_GRID = 16  # 64 / patch_size 4


class _JEPAForwardMixin:
    """Shared context/spectrum/predictor forward producing latent delta predictions."""

    def _encode(self, G, S, M, goal_mode, need_attn):
        """Full student-side encode: masked context -> z_x; spectrum -> (c_physics,
        a_goal); predictor -> z_hat. Returns them all so the model output contract
        (spec §30) exposes every representation boundary explicitly."""
        G_c = apply_mask_to_pixels(G, M)

        # Full-resolution context representation.
        z_x = self.context_encoder(G_c, M)

        hidden = self.hidden
        assert z_x.ndim == 3, (
            f"Context encoder output must be [B,256,{hidden}], got {tuple(z_x.shape)}"
        )
        assert z_x.shape[1] == 256, (
            f"Context encoder must preserve 256 tokens, got {z_x.shape[1]}"
        )
        assert z_x.shape[2] == hidden, (
            f"Context encoder embedding dim must be {hidden}, got {z_x.shape[2]}"
        )

        c_physics, a_goal = self.spectrum_path(
            S,
            goal_mode=goal_mode,
        )

        mask = (M.view(M.shape[0], -1) == 0)

        assert mask.shape[1] == 256, (
            f"Mask must have 256 token positions, got {tuple(mask.shape)}"
        )
        assert mask.any(dim=1).all(), (
            "Every sample must contain at least one masked token"
        )

        mask_token = self.context_encoder.mask_token
        pos = self.context_encoder.geo.pos_embed

        queries = torch.where(
            mask.unsqueeze(-1),
            mask_token + pos,
            z_x,
        )

        # KEEP ALL 256 geometry tokens.
        kv = torch.cat([z_x, a_goal], dim=1)

        z_hat, weights = self.predictor(
            queries,
            kv,
            c_physics,
            need_weights=need_attn,
        )

        hidden = self.hidden
        assert z_hat.ndim == 3, (
            f"Predictor output must be [B,256,{hidden}], got {tuple(z_hat.shape)}"
        )
        assert z_hat.shape[1] == 256, (
            f"Predictor must output 256 tokens, got {z_hat.shape[1]}"
        )
        assert z_hat.shape[2] == hidden, (
            f"Predictor output dim must be {hidden}, got {z_hat.shape[2]}"
        )

        assert z_x.shape == z_hat.shape, (
            f"Context/prediction mismatch: "
            f"z_x={tuple(z_x.shape)}, z_hat={tuple(z_hat.shape)}"
        )

        return z_hat, z_x, mask, weights, c_physics, a_goal

    def query_predictions(self, G, S, M, goal_mode="real"):
        z_hat, *_ = self._encode(
            G, S, M, goal_mode, need_attn=False
        )
        return z_hat


class GoalConditionedJEPA(_JEPAForwardMixin, nn.Module):
    def __init__(self, hidden=384, num_heads=6, geo_depth=6, predictor_depth=6,
                 goal_tokens=16, num_predictor_heads=6,
                 momentum_start=0.996, momentum_end=0.999):
        super().__init__()
        self.hidden = hidden
        geo = GeometryEncoder(hidden=hidden, num_heads=num_heads, depth=geo_depth)
        self.context_encoder = ContextEncoder(geo, hidden=hidden)
        self.spectrum_path = SpectrumPath(None, hidden=hidden, goal_tokens=goal_tokens)
        self.predictor = GCLCT(depth=predictor_depth, hidden=hidden,
                               num_heads=num_predictor_heads)
        self.ema = EMAEncoder(
            geo,
            momentum_start=momentum_start,
            momentum_end=momentum_end
        )
        for name, param in self.ema.named_parameters():
            assert not param.requires_grad, (
                f"EMA target parameter is trainable: {name}"
            )

        self.geometry_encoder = geo

    def enforce_frozen_reference_modes(self):
        """Keep frozen reference modules in eval() regardless of the student's mode.

        `model.train()` recursively flips children to training mode, but the EMA
        target and the released spectrum encoder are parameter-frozen reference
        components that must behave deterministically at inference. Called from
        `train()` and `forward()` so every mode switch / checkpoint load / resume
        path is covered. The released encoder may be absent in unit-test stubs.
        """
        self.ema.target.eval()
        released = getattr(self.spectrum_path, "released", None)
        if released is not None:
            released.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enforce_frozen_reference_modes()
        return self

    def forward(self, G, S, M, goal_mode="real", need_attn=False, with_target=True,):
        self.enforce_frozen_reference_modes()
        z_hat, z_x, mask, weights, c_physics, a_goal = self._encode(
            G, S, M, goal_mode, need_attn
        )

        b = G.shape[0]
        hidden = self.hidden
        assert c_physics.shape == (b, hidden), (
            f"c_physics must be [B,{hidden}], got {tuple(c_physics.shape)}"
        )
        assert a_goal.shape == (b, 16, hidden), (
            f"a_goal must be [B,16,{hidden}], got {tuple(a_goal.shape)}"
        )
        assert mask.shape == (b, 256), (
            f"mask must be [B,256], got {tuple(mask.shape)}"
        )

        out = dict(
            z_hat=z_hat,
            z_x=z_x,
            mask=mask,
            attn_weights=weights,
            c_physics=c_physics,
            a_goal=a_goal,
        )

        if with_target:
            z_y_raw = self.ema(G)

            assert z_y_raw.ndim == 3, (
                f"EMA target output must be [B,256,{hidden}], got {tuple(z_y_raw.shape)}"
            )
            assert z_y_raw.shape[1] == 256, (
                f"EMA target must output 256 tokens, got {z_y_raw.shape[1]}"
            )
            assert z_y_raw.shape[2] == hidden, (
                f"EMA target dim must be {hidden}, got {z_y_raw.shape[2]}"
            )

            assert z_y_raw.shape == z_hat.shape, (
                f"Target/prediction mismatch: "
                f"z_y_raw={tuple(z_y_raw.shape)}, z_hat={tuple(z_hat.shape)}"
            )

            # Spec §10: expose the raw target AND an explicit feature-wise
            # normalization boundary (F.layer_norm over the 384-D feature axis,
            # per-sample per-token — no learnable weights, never overwrites raw).
            # `z_y` is kept ONLY as a backward-compatible raw alias; active code
            # must consume `z_y_raw` / `z_y_normalized` explicitly (hardening §2).
            out["z_y_raw"] = z_y_raw
            out["z_y_normalized"] = F.layer_norm(z_y_raw, (z_y_raw.shape[-1],))
            out["z_y"] = z_y_raw            # compat alias — do NOT consume

        return out

    def loss(self, G, S, M, goal_mode="real"):
        out = self.forward(G, S, M, goal_mode=goal_mode)
        L, per_sample = jepa_loss(out["z_hat"], out["z_y_raw"], out["mask"], proj=None)
        return L, out


def load_released_metadit_state_dict(weights_path):
    return torch.load(weights_path, map_location="cpu")


def init_geometry_from_metadit(model, metadit_weights, blocks_to_take=6):
    sd = load_released_metadit_state_dict(metadit_weights)
    model.geometry_encoder.init_from_metadit(sd, blocks_to_take=blocks_to_take)
    return sd


def set_spectrum_path(model, spec_weights, device):
    released = ReleasedSpectrumEncoder(spec_weights, device=device)
    model.spectrum_path.released = released
    model.spectrum_path.released.to(device)


def build_model(cfg, spec_weights, device="cpu", init_from_metadit=True,
                metadit_weights=None, blocks_to_take=6):
    variant = cfg.get("variant", "jepa")

    if variant != "jepa":
        raise RuntimeError(
            "Only the JEPA variant is enabled during the architecture refactor."
        )
    kwargs = dict(hidden=cfg.get("hidden", 384),
                  num_heads=cfg.get("num_heads", 6),
                  geo_depth=cfg.get("geo_depth", 6),
                  predictor_depth=cfg.get("predictor_depth", 8),
                  goal_tokens=cfg.get("goal_tokens", 16),
                  num_predictor_heads=cfg.get("num_predictor_heads", 6))
    kwargs.update(
        momentum_start=cfg.get("ema_momentum_start", 0.996),
        momentum_end=cfg.get("ema_momentum_end", 0.999),
    )

    model = GoalConditionedJEPA(**kwargs)

    set_spectrum_path(model, spec_weights, device)
    if init_from_metadit:
        assert metadit_weights is not None
        init_geometry_from_metadit(model, metadit_weights,
                                   blocks_to_take=cfg.get("geo_depth", 6))
    if variant == "jepa":
        model.ema.target.load_state_dict(model.geometry_encoder.state_dict())
    model.to(device)
    return model


SAVED_EXCLUDES = (".released.",)


def saveable_state_dict(model):
    """Drop frozen released components (re-loaded from disk on rebuild)."""
    return {k: v for k, v in model.state_dict().items()
            if not any(x in k for x in SAVED_EXCLUDES)}


def load_into_model(model, sd, device, strict=True):
    """Load a saved state dict into a model, refusing silent mismatches.

    strict=True (Bug #11): a checkpoint whose keys do not exactly match the model
    raises instead of silently leaving parameters at init — previously strict=False
    could load a stale/renamed checkpoint and bias every downstream result without
    any warning. Frozen released components (SAVED_EXCLUDES) are filtered on BOTH
    sides: they are excluded from checkpoints at save time (re-loaded from disk on
    every build via set_spectrum_path) and therefore excluded from the strict
    comparison here too.
    """
    keys = [k for k in sd if not any(x in k for x in SAVED_EXCLUDES)]
    filtered = {k: sd[k] for k in keys}
    if strict:
        model_keys = set(model.state_dict())
        released = {k for k in model_keys if any(x in k for x in SAVED_EXCLUDES)}
        expected = set(filtered)
        # missing: keys the model expects but checkpoint lacks
        missing = sorted(model_keys - expected - released)
        # unexpected: keys in checkpoint that model doesn't expect
        unexpected = sorted(expected - model_keys)
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint/model key mismatch (refusing silent non-strict load; "
                f"missing={missing[:8]}{'...' if len(missing) > 8 else ''} "
                f"unexpected={unexpected[:8]}{'...' if len(unexpected) > 8 else ''})")
    model_keys = model.state_dict()
    model.load_state_dict({k: filtered[k] for k in filtered if k in model_keys},
                          strict=False)
    model.to(device)


# ===========================================================================
# Unified JEPA model — architecture_v5.md §3.1-§3.6, §4.1, §5
#
# New internal representation: occupancy M[64,64] + l_lattice/h_atom/r_atom as
# explicit scalars with known/unknown flags + target spectrum [2,301].
# 192-D throughout (except c_physics/a_goal at 384 from the frozen SpectrumPath,
# projected downstream to 192).
# ===========================================================================

UNIFIED_ARCHITECTURE_ID = "unified_occ_param_spectrum_jepa_v1"


class UnifiedJEPA(nn.Module):
    """Unified occupancy + scalar + spectrum JEPA (architecture_v5.md §3.1-§3.6).

    Active model accepts semantically:
        occupancy      [B,1,64,64]  single-channel binary occupancy
        scalar_values  [B,3]        (l_lattice, h_atom, r_atom)
        scalar_known   [B,3] bool   which scalars are observed
        spectrum       [B,2,301]    target electromagnetic spectrum
        mask           [B,16,16]    1=visible, 0=masked (at token-grid resolution)

    Internal flow (§11.1):
        occupancy + masked scalars + FiLM → OccupancyEncoder → z_x [B,256,192]
        spectrum → SpectrumPath(frozen) → c_physics [B,384], a_goal [B,16,384]

    GCLCT branch (legacy, predictor_type="gclct"):
        z_x + proj(a_goal) + scalar_summary → FusionEncoder → fused [B,273,192]
        256 mask-token queries + 1 scalar-summary query → GCLCT(c_physics)
        → z_hat [B,257,192] → occupancy_pred + scalar_summary_pred → scalar_pred
        EMA target encoder (occupancy_ema + scalar_mlp_ema) → z_y_raw [B,256,192]

    Joint Target Redesign branch (docs/JOINT_TARGET_REDESIGN.md, adopted
    2026-09-08; predictor_type="masked_query" / joint_target=True):
        visible z_x tokens + goal_proj(a_goal) [B,16,192] + scalar summary
        → MaskedQueryPredictor → z_hat [B,256,192] (predicted at masked
        positions only; visible positions keep the context representation, §4)
        teacher: EMA(occupancy) → z_y_geo [B,256,192] (stop-grad) and
        Z_S = goal_proj(a_goal) (stop-grad) → JointTargetFusion
        → z_y_joint = Z_G + tanh(gate)·cross_attn(Z_G, Z_S)   (§3)
        The fusion's gate/attention parameters ARE trainable — the objective
        gradient reaches them through z_y_joint; everything upstream of the
        fusion is frozen/detached (EMA geometry, released spectrum encoder).

    EMA rules (§3.6):
        - occupancy EMA = JEPA target for occupancy tokens only
        - scalar_mlp_ema = target-side FiLM conditioning only
        - NO scalar EMA latent loss target
    """

    architecture_id = UNIFIED_ARCHITECTURE_ID

    def __init__(self, hidden=192, num_heads=6, geo_depth=6, predictor_depth=8,
                 goal_tokens=16, num_predictor_heads=6, scalar_hidden=128,
                 n_film_blocks=6, spec_dim=256, scalar_bounds=None,
                 momentum_start=0.996, momentum_end=0.999,
                 joint_target=False, predictor_type="gclct",
                 mq_predictor_layers=2):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        self.goal_tokens = goal_tokens

        # --- Joint Target Redesign switches (docs/JOINT_TARGET_REDESIGN.md) ---
        # joint_target=True            : teacher target becomes Z_joint (§3)
        # predictor_type="masked_query": student predicts masked tokens only (§4)
        # Defaults keep the pre-redesign GCLCT architecture bit-for-bit.
        self.joint_target = bool(joint_target)
        if predictor_type not in ("gclct", "masked_query"):
            raise ValueError(
                f"predictor_type must be 'gclct' or 'masked_query', "
                f"got {predictor_type!r}")
        self.predictor_type = predictor_type
        suffix = []
        if joint_target:
            suffix.append("joint")
        if predictor_type == "masked_query":
            suffix.append("mq")
        self.architecture_id = UNIFIED_ARCHITECTURE_ID + (
            "_" + "-".join(suffix) if suffix else "")

        # Student encoders
        self.occupancy_encoder = OccupancyEncoder(
            hidden=hidden, num_heads=num_heads, depth=geo_depth
        )
        self.scalar_encoder = ScalarEncoder(
            hidden=hidden, scalar_hidden=scalar_hidden, n_film_blocks=n_film_blocks
        )

        # Spectrum path — stays at 384-D (architecture_v5.md §3.3)
        self.spectrum_path = SpectrumPath(
            None, spec_dim=spec_dim, hidden=384, goal_tokens=goal_tokens,
            num_heads=4,
        )

        if predictor_type == "masked_query" or joint_target:
            # Shared 384→192 goal projection (§2: "Project a_goal to 192-D
            # for the joint predictor"). One projection serves both the
            # student predictor's KV and the teacher fusion's KV, so Z_S is
            # the same representation on both sides of the JEPA objective.
            self.goal_proj = nn.Linear(384, hidden)

        if predictor_type == "masked_query":
            # §4 student predictor: masked queries only. The GCLCT path
            # (fusion_encoder + GCLCT predictor) is NOT constructed — the two
            # are alternative predictors, not complements.
            self.masked_query_predictor = MaskedQueryPredictor(
                hidden=hidden, num_heads=num_predictor_heads,
                num_layers=mq_predictor_layers, n_spatial_tokens=256)
        else:
            # Fusion (192-D, projects a_goal 384→192 internally)
            self.fusion_encoder = FusionEncoder(
                hidden=hidden, num_heads=num_heads, depth=2, goal_dim_in=384
            )

            # Predictor — accepts 384-D c_physics, projects to 192 internally
            self.predictor = GCLCT(
                depth=predictor_depth, hidden=hidden, num_heads=num_predictor_heads,
                c_physics_dim=384,
            )
        # goal_residual route RETIRED 2026-09-08.

        if joint_target:
            # §3 joint target fusion: Z_joint = Z_G + tanh(gate) * cross_attn(
            # Q=Z_G, KV=Z_S), gate zero-init (bit-identical to the geometry-
            # only target at step 0). NOT EMA-copied — its gate/attention
            # parameters are trained directly by the objective gradient that
            # flows through z_y_joint; everything upstream is detached.
            self.joint_target_fusion = JointTargetFusion(
                hidden=hidden, num_heads=num_heads)

        # Scalar decode heads
        self.scalar_decoder = ScalarDecoder(hidden=hidden, bounds=scalar_bounds)

        # Occupancy decoder: latent → occupancy logits only, FiLM-conditioned
        # by effective (l,h,r) at every layer (architecture_v5.md §4.1).
        # No 3-channel geometry head: the broadcast tensor is assembled once,
        # at the physics-surrogate boundary.
        self.geometry_decoder = OccupancyDecoder(
            hidden=hidden, base_dim=hidden // 2, scalar_hidden=scalar_hidden,
        )

        # EMA target for occupancy encoder (z_y_raw JEPA target)
        self.ema = EMAEncoder(
            self.occupancy_encoder,
            momentum_start=momentum_start,
            momentum_end=momentum_end,
        )

        # EMA shadow copy of the scalar MLP — target-side FiLM conditioning only
        self.scalar_mlp_ema = EMAEncoder(
            self.scalar_encoder,
            momentum_start=momentum_start,
            momentum_end=momentum_end,
        )

        # Learned mask / query tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.mask_token, std=0.02)
        self.scalar_query_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.scalar_query_token, std=0.02)

        # Verify EMA params are frozen
        for name, param in self.ema.named_parameters():
            assert not param.requires_grad, f"occupancy EMA trainable: {name}"
        for name, param in self.scalar_mlp_ema.named_parameters():
            assert not param.requires_grad, f"scalar_mlp_ema trainable: {name}"

    # --- EMA helpers -------------------------------------------------------

    def set_total_steps(self, n):
        self.ema.set_total_steps(n)
        self.scalar_mlp_ema.set_total_steps(n)

    @property
    def requires_broadcast_mask(self):
        """MaskedQueryPredictor (Stage A) operates on a single mask shape
        broadcast across the batch (§4; per-sample masks arrive with Stage F).
        Mask samplers must honor this and produce identical masked positions
        for every sample in the batch — BlockMasker.sample draws per-sample
        placements by default, so callers route through batch-1 sampling +
        broadcast when this flag is set."""
        return self.predictor_type == "masked_query"

    def enforce_frozen_reference_modes(self):
        """Keep frozen reference modules in eval() regardless of student mode."""
        self.ema.target.eval()
        self.scalar_mlp_ema.target.eval()
        released = getattr(self.spectrum_path, "released", None)
        if released is not None:
            released.eval()

    def train(self, mode=True):
        super().train(mode)
        self.enforce_frozen_reference_modes()
        return self

    # --- Forward -----------------------------------------------------------

    def _build_scalar_input(self, scalar_values, scalar_known):
        """Build the 6-D scalar MLP input [l_val, l_known, h_val, h_known,
        r_val, r_known] with values zeroed where unknown."""
        known_f = scalar_known.float()
        masked = torch.where(scalar_known, scalar_values,
                             torch.zeros_like(scalar_values))
        return torch.stack([
            masked[:, 0], known_f[:, 0],
            masked[:, 1], known_f[:, 1],
            masked[:, 2], known_f[:, 2],
        ], dim=-1)

    def forward(self, occupancy, scalar_values, scalar_known, spectrum, mask,
                goal_mode="real", with_target=True, need_attn=False):
        """Unified forward.

        Args:
            occupancy:      [B,1,64,64] binary float occupancy.
            scalar_values:  [B,3] (l_lattice, h_atom, r_atom)
            scalar_known:   [B,3] bool — which scalars are observed.
            spectrum:       [B,2,301] target spectrum.
            mask:           [B,16,16]  1=visible, 0=masked.
            goal_mode:      "real" | "null" (only). "shuffled" is NOT a model
                            goal_mode — SpectrumPath treats any non-"null"
                            value as "real" and validate_goal_mode rejects
                            it; shuffled controls are built externally by
                            deranging the spectrum tensor
                            (runtime.physics_controls.make_shuffled_spectrum)
                            and then run with goal_mode="real".
            with_target:    compute EMA target latent z_y_raw.
            need_attn:      return attention weights.

        Returns dict with z_hat, z_x, mask, c_physics, a_goal, scalar_pred,
        scalar_summary_pred, and (if with_target) z_y_raw, z_y_normalized, z_y.
        """
        self.enforce_frozen_reference_modes()
        b = occupancy.shape[0]
        hidden = self.hidden

        # 1. Build scalar MLP input from values + known flags
        scalar_mlp_input = self._build_scalar_input(scalar_values, scalar_known)

        # 2. Scalar encoder (live) → FiLM params + scalar summary token
        film_params, scalar_summary = self.scalar_encoder(scalar_mlp_input)

        # 3. Occupancy encoder (student) with mask replacement + FiLM
        masked_occ = apply_mask_to_pixels(occupancy, mask)
        z_x = self.occupancy_encoder(
            masked_occ, film_params=film_params,
            mask=mask, mask_token=self.mask_token,
        )  # (B, 256, hidden)

        # 4. Spectrum path (frozen)
        c_physics, a_goal = self.spectrum_path(spectrum, goal_mode=goal_mode)
        # c_physics: (B, 384), a_goal: (B, 16, 384)

        # 5-8. Student prediction (predictor branch)
        vis_mask = (mask.view(b, -1) > 0.5)  # True = visible, (B, 256)
        if self.predictor_type == "masked_query":
            # §4: only masked spatial tokens are predicted; visible positions
            # keep the context representation unchanged (first-implementation
            # choice per the spec). The Stage-A contract requires a single
            # mask shape across the batch (see requires_broadcast_mask).
            n_masked = int((~vis_mask[0]).sum().item())
            if n_masked == 0:
                # Nothing to predict — the full grid is visible context.
                occupancy_pred = z_x
            else:
                visible_idx = vis_mask[0].nonzero(as_tuple=False).squeeze(-1)
                z_visible = z_x[:, visible_idx, :]        # (B, N_vis, hidden)
                a_goal_h = self.goal_proj(a_goal)         # (B, 16, hidden)
                c_scalar = scalar_summary.squeeze(1)      # (B, hidden)
                occupancy_pred = self.masked_query_predictor(
                    z_visible, a_goal_h, vis_mask, c_scalar)  # (B, 256, hidden)
            # Stage-A scalar path: this branch has no predictor scalar-query
            # (a context-attending scalar query is a §14-step-5 / Stage-B
            # concern). Decode from the scalar encoder's summary token so the
            # output contract holds; under the Stage-A curriculum (scalars
            # all known) this head is neither trained nor consumed.
            scalar_summary_pred = scalar_summary.squeeze(1)  # (B, hidden)
        else:
            # 5. Fusion: 256 occupancy + 16 goal (projected 384→192) + 1 scalar summary
            fused = self.fusion_encoder(z_x, a_goal, scalar_summary)  # (B, 273, hidden)
            assert fused.shape[1] == 273, (
                f"Fusion must output 273 tokens (256+16+1), got {fused.shape[1]}"
            )

            # 6. Construct predictor queries
            pos = self.occupancy_encoder.pos_embed  # (1, 256, hidden)
            occ_queries = torch.where(
                vis_mask.unsqueeze(-1),
                fused[:, :256, :],          # visible: fused tokens
                self.mask_token + pos,      # masked: mask_token + pos
            )  # (B, 256, hidden)
            scalar_query = self.scalar_query_token.expand(b, -1, -1)  # (B, 1, hidden)
            queries = torch.cat([occ_queries, scalar_query], dim=1)    # (B, 257, hidden)

            # 7. Predictor (c_physics 384→192 via c_phys_proj)
            z_hat_base, _ = self.predictor(queries, fused, c_physics)  # (B, 257, hidden)

            # 8. Split predictions
            occupancy_pred = z_hat_base[:, :256, :]      # (B, 256, hidden)
            scalar_summary_pred = z_hat_base[:, 256, :]   # (B, hidden)

        # 9. Scalar decode
        scalar_pred = self.scalar_decoder(scalar_summary_pred)  # (B, 3)

        # 10. Loss mask: True = masked position (for JEPA loss)
        loss_mask = ~vis_mask  # (B, 256)

        out = dict(
            z_hat=occupancy_pred,
            z_hat_base=occupancy_pred,
            z_x=z_x,
            mask=loss_mask,
            c_physics=c_physics,
            a_goal=a_goal,
            scalar_pred=scalar_pred,
            scalar_summary_pred=scalar_summary_pred,
        )

        # Decode occupancy once here so the active objective can supervise it
        # directly. The physics loop reuses these logits when available,
        # avoiding a second occupancy-decoder forward for the same prediction.
        effective_scalars = torch.where(
            scalar_known, scalar_values, scalar_pred)
        out["occupancy_logits"] = self.geometry_decoder(
            occupancy_pred, effective_scalars)
        out["effective_scalars"] = effective_scalars

        if with_target:
            with torch.no_grad():
                # True scalars (all known) for target-side FiLM
                true_input = torch.stack([
                    scalar_values[:, 0], torch.ones_like(scalar_values[:, 0]),
                    scalar_values[:, 1], torch.ones_like(scalar_values[:, 1]),
                    scalar_values[:, 2], torch.ones_like(scalar_values[:, 2]),
                ], dim=-1)  # (B, 6)
                film_params_ema, _ = self.scalar_mlp_ema(true_input)
                z_y_geo = self.ema(occupancy, film_params=film_params_ema)
                out["z_y_raw"] = z_y_geo
                out["z_y_normalized"] = F.layer_norm(
                    z_y_geo, (z_y_geo.shape[-1],)
                )
                out["z_y"] = z_y_geo  # compat alias

            if self.joint_target:
                # §3 joint target: Z_joint = J(Z_G, Z_S). z_y_geo (EMA
                # geometry) and a_s (released spectrum encoder, detached) are
                # both stop-gradient — the ONLY trainable parameters receiving
                # the objective's target-side gradient are the fusion's own
                # gate/attention weights (plus the shared goal_proj), which is
                # exactly how the teacher learns a physics-conditioned
                # representation. NOTE: the joint target is meaningful for
                # goal_mode="real" (Stage-A contract S_goal = S_true); a null
                # goal zeroes Z_S and the fusion degenerates to a constant,
                # spectrum-free delta.
                a_s = self.goal_proj(a_goal.detach())  # (B, 16, hidden)
                out["z_y_joint"] = self.joint_target_fusion(z_y_geo, a_s)

        return out

    def decode_geometry(self, z_hat, scalar_pred, occ_input=None, mask=None,
                        scalar_known=None, scalar_values=None, use_ste=False,
                        hard_forward=False, occupancy_logits=None):
        """Decode predicted latents to surrogate-ready geometry (Phase 4 MD §1-§3,
        architecture_v5.md §4.1).

        Args:
            z_hat:       [B, 256, hidden] predicted occupancy latents.
            scalar_pred: [B, 3] (l_lattice, h_atom, r_atom) physical values.
            occ_input:   [B, 1, 64, 64] original binary occupancy (for retention).
            mask:        [B, 16, 16] 1=visible, 0=masked — retains visible pixels.
            scalar_known: [B, 3] bool — which scalars are observed. When provided
                         together with scalar_values, known scalars are
                         substituted with their true values (the scalar analog
                         of visible-occupancy retention); unknown scalars use
                         scalar_pred.
            scalar_values: [B, 3] true scalar values (used only where
                         scalar_known is True).
            use_ste:     If True (AND self.training), use hard occupancy for the
                         surrogate input with a soft backward path
                         (straight-through estimator). Training behavior only.
            hard_forward: If True, threshold occupancy to binary for the forward
                         geometry regardless of training mode (used by the
                         soft-vs-hard diagnostic — see physics_loop).

        Returns:
            geometry:    [B, 3, 64, 64] — r_atom/5, h_atom, l_lattice/3.
            soft_occ:    [B, 1, 64, 64] — sigmoid occupancy logits.
        """
        # Effective scalar rule (architecture_v5.md §4.1): decode-time FiLM and
        # assembly use the true value where known, the prediction where unknown —
        # identical in training and inference.
        if scalar_known is not None and scalar_values is not None:
            scalar_for_assembly = torch.where(
                scalar_known, scalar_values, scalar_pred)
        else:
            scalar_for_assembly = scalar_pred

        # Decoder is FiLM-conditioned by the effective (l,h,r). Reuse logits
        # from UnifiedJEPA.forward when the caller has them.
        if occupancy_logits is None:
            occupancy_logits = self.geometry_decoder(z_hat, scalar_for_assembly)
        occ_logits = occupancy_logits
        soft_occ = torch.sigmoid(occ_logits)  # (B, 1, 64, 64)

        if use_ste and self.training:
            hard_occ = (soft_occ > 0.5).float()
            occ_for_assembly = hard_occ + soft_occ - soft_occ.detach()
        elif hard_forward:
            occ_for_assembly = (soft_occ > 0.5).float()
        else:
            occ_for_assembly = soft_occ

        if occ_input is not None and mask is not None:
            # Retain visible pixels from the input (Phase 4 MD §6: L_preserve)
            up = mask.view(z_hat.shape[0], 1, 16, 16).repeat_interleave(4, 2).repeat_interleave(4, 3)
            vis = (up > 0.5).float()
            occ_for_assembly = occ_input * vis + occ_for_assembly * (1 - vis)

        l = scalar_for_assembly[:, 0]
        h = scalar_for_assembly[:, 1]
        r = scalar_for_assembly[:, 2]
        geometry = assemble_metadit_geometry(occ_for_assembly, l, h, r)
        return geometry, occ_for_assembly

    def loss(self, occupancy, scalar_values, scalar_known, spectrum, mask,
             goal_mode="real"):
        """Phase-2 loss: L_JEPA + scalar L1 (on unknown positions only)."""
        out = self.forward(
            occupancy, scalar_values, scalar_known, spectrum, mask,
            goal_mode=goal_mode,
        )
        L_jepa, _ = jepa_loss(
            out["z_hat"], out.get("z_y_joint", out["z_y_raw"]),
            out["mask"], proj=None,
        )
        unknown = ~scalar_known  # (B, 3)
        scalar_err = (out["scalar_pred"] - scalar_values).abs() * unknown.float()
        n_unknown = unknown.sum().clamp(min=1)
        L_scalar = scalar_err.sum() / n_unknown
        L = L_jepa + L_scalar
        out["loss_components"] = {"L_jepa": L_jepa.detach().item(), "L_scalar": L_scalar.detach().item()}
        return L, out


def build_unified_model(cfg, spec_weights, device="cpu",
                        spec_config=None):
    """Build the unified JEPA model (architecture_v5.md §3.1-§3.6).

    Uses released MetaDiT spec encoder weights only where shapes genuinely permit.
    The 192-D student components (occupancy encoder, scalar encoder, fusion,
    predictor) are initialized normally — old 384-D Milestone-B weights are NOT
    loaded into the 192-D architecture.
    """
    kwargs = dict(
        hidden=cfg.get("hidden", 192),
        num_heads=cfg.get("num_heads", 6),
        geo_depth=cfg.get("geo_depth", 6),
        predictor_depth=cfg.get("predictor_depth", 8),
        goal_tokens=cfg.get("goal_tokens", 16),
        num_predictor_heads=cfg.get("num_predictor_heads", 6),
        scalar_hidden=cfg.get("scalar_hidden", 128),
        n_film_blocks=cfg.get("n_film_blocks", 6),
        spec_dim=cfg.get("spec_dim", 256),
        scalar_bounds=tuple(
            tuple(cfg.get("scalar_bounds", {}).get(name, default))
            for name, default in (
                ("l_lattice", (2.5, 3.0)),
                ("h_atom", (0.5, 1.0)),
                ("r_atom", (3.5, 5.0)),
            )
        ),
    )
    kwargs.update(
        momentum_start=cfg.get("ema_momentum_start", 0.996),
        momentum_end=cfg.get("ema_momentum_end", 0.999),
        # Joint Target Redesign (docs/JOINT_TARGET_REDESIGN.md): both default
        # OFF so pre-redesign configs (configs/unified.yaml and everything
        # built from it) keep the exact GCLCT architecture. The Stage-A
        # config (configs/unified_stage_a.yaml) enables them explicitly.
        joint_target=cfg.get("joint_target", False),
        predictor_type=cfg.get("predictor_type", "gclct"),
        mq_predictor_layers=cfg.get("mq_predictor_layers", 2),
    )

    model = UnifiedJEPA(**kwargs)
    set_spectrum_path(model, spec_weights, device)

    # Initialize EMA targets from students (NOT from old 384-D checkpoints)
    model.ema.target.load_state_dict(model.occupancy_encoder.state_dict())
    model.scalar_mlp_ema.target.load_state_dict(model.scalar_encoder.state_dict())

    model.to(device)
    return model
