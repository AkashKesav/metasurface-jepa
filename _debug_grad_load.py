import os, sys
REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "tests"))
import torch
from test_unified_losses import _build_model, _batch
from assembly import load_into_model, saveable_state_dict
from losses.unified_losses import UnifiedJEPALoss
from losses.vicreg import vicreg_branch_terms

model = _build_model(); model.train()
obj = UnifiedJEPALoss(hidden=192, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
                      lambda_scalar=1.0, lambda_occ=1.0, lambda_phys=0.0,
                      lambda_summary=1.0, gamma=1.0, eps=1e-4)
obj.train()
# emulate a round-trip through a checkpoint, as the cloud probe does
ck = {"model": saveable_state_dict(model), "objective_state": obj.state_dict()}
model2 = _build_model()
load_into_model(model2, ck["model"], device="cpu", strict=True)
model2.train()
print("after load_into_model: model.training =", model2.training)
n_req = sum(1 for p in model2.parameters() if p.requires_grad)
print("trainable params after load:", n_req, "of", sum(1 for _ in model2.parameters()))
obj2 = UnifiedJEPALoss(hidden=192, lambda_inv=25.0, lambda_var=25.0, lambda_cov=1.0,
                       lambda_scalar=1.0, lambda_occ=1.0, lambda_phys=0.0,
                       lambda_summary=1.0, gamma=1.0, eps=1e-4)
obj2.train()
info = obj2.load_state_dict(ck["objective_state"], strict=False)
print("objective load: missing:", [k for k in info.missing_keys][:4], "unexpected:", list(info.unexpected_keys)[:4])
print("projector trainable:", sum(1 for p in obj2.projector.parameters() if p.requires_grad))

occ, sv, spec, M = _batch(seed=3)
sk = torch.zeros(2, 3, dtype=torch.bool); sk[:, 0] = True
o = model2(occ, sv, sk, spec, M, goal_mode="real")
ph = obj2.projector(o["z_hat"]); py = obj2.projector(o["z_y_raw"])
print("z_hat grad_fn:", type(o["z_hat"].grad_fn).__name__ if o["z_hat"].grad_fn else None)
print("p_hat grad_fn:", type(ph.grad_fn).__name__ if ph.grad_fn else None)
mb = o["mask"]; print("mask n True:", int(mb.sum()), "/", mb.numel())
L_inv, _, _ = vicreg_branch_terms(ph[mb], py[mb], gamma=1.0, eps=1e-4)
print("L_inv:", float(L_inv), "grad_fn:", type(L_inv.grad_fn).__name__ if L_inv.grad_fn else None)
L_inv.backward(retain_graph=True)
print("backward OK")
