import os, sys
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import torch
from test_unified_losses import _build_model, _batch
from losses.unified_losses import UnifiedJEPALoss
from losses.vicreg import vicreg_branch_terms

model = _build_model(); model.train()
obj = UnifiedJEPALoss(hidden=192, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
                      lambda_scalar=1.0, lambda_occ=1.0, lambda_phys=0.0,
                      lambda_summary=1.0, gamma=1.0, eps=1e-4)
obj.train()
occ, sv, spec, M = _batch(seed=3)
sk = torch.zeros(2, 3, dtype=torch.bool); sk[:, 0] = True

def d(t): return f"{type(t.grad_fn).__name__ if t.grad_fn else None}"
o = model(occ, sv, sk, spec, M, goal_mode="real")
print("z_hat:", o["z_hat"].requires_grad, d(o["z_hat"]))
print("scalar_summary:", o["scalar_summary"].requires_grad, d(o["scalar_summary"]))
print("z_y_raw:", o["z_y_raw"].requires_grad, d(o["z_y_raw"]))
print("mask all-True:", bool(o["mask"].all()), "n:", int(o["mask"].sum()), "/", o["mask"].numel())

ph = obj.projector(o["z_hat"]); py = obj.projector(o["z_y_raw"])
print("p_hat:", ph.requires_grad, d(ph), "selected:", tuple(ph[o["mask"]].shape))
L_inv, L_var, L_cov = vicreg_branch_terms(ph[o["mask"]], py[o["mask"]], gamma=1.0, eps=1e-4)
for n, L in (("L_inv", L_inv), ("L_var", L_var), ("L_cov", L_cov)):
    print(f"  {n}: {float(L):.6f} requires_grad={L.requires_grad} grad_fn={d(L)}")
L_scalar = obj.scalar_loss(o["scalar_pred"], sv, sk)
print(f"  L_scalar: {float(L_scalar):.6f} requires_grad={L_scalar.requires_grad} grad_fn={d(L_scalar)}")
readout = obj.summary_readout(o["scalar_summary"])
L_summary = ((readout - sv).abs() * sk.float()).sum() / sk.sum().clamp(min=1)
print(f"  L_summary: {float(L_summary):.6f} requires_grad={L_summary.requires_grad} grad_fn={d(L_summary)}")
logits = model.decode_occupancy_logits(o["z_hat"], o["scalar_pred"], scalar_known=sk, scalar_values=sv)
mp = M.view(2,1,16,16).repeat_interleave(4,2).repeat_interleave(4,3) > 0.5
print("mp all-True:", bool(mp.all()), "n:", int(mp.sum()), "/", mp.numel())
L_occ = torch.nn.functional.binary_cross_entropy_with_logits(logits[mp], occ[mp])
print(f"  L_occ: {float(L_occ):.6f} requires_grad={L_occ.requires_grad} grad_fn={d(L_occ)}")
