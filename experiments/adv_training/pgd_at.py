# adv_training/pgd_at.py
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

@torch.no_grad()
def _rand_start(x, eps):
    return (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0,1)

def pgd_attack(model, x, y, eps=8/255, step=2/255, iters=10):
    x0 = x.detach()
    x_adv = _rand_start(x0, eps).detach()
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        for _ in range(iters):
            x_adv.requires_grad_(True)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                x_adv += step * x_adv.grad.sign()
                eta = (x_adv - x0).clamp(-eps, eps)
                x_adv[:] = (x0 + eta).clamp(0,1)
            x_adv = x_adv.detach()
    if was_training:
        model.train()
    return x_adv

def train_pgd_at(model, train_loader, val_loader, epochs=10,
                 eps=8/255, step=2/255, iters=7, opt=None, device=None, log=None, sched=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)
    model.to(device)
    best = {"robust": 0.0, "state": None}
    for ep in range(1, epochs+1):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        for x,y in tqdm(train_loader, desc=f"Epoch {ep}/{epochs} train", leave=False):
            x,y = x.to(device), y.to(device)
            x_adv = pgd_attack(model, x, y, eps=eps, step=step, iters=iters)
            logits = model(x_adv)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            train_loss_sum += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.numel()
        if sched:
            sched.step()
        train_loss = train_loss_sum / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # quick robust eval (PGD-10) on val_loader
        model.eval(); correct=0; total=0; val_loss_sum=0.0
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc=f"Epoch {ep}/{epochs} eval", leave=False):
                x,y = x.to(device), y.to(device)
                # evaluate on adversarial examples
                xa = pgd_attack(model, x, y, eps=eps, step=step, iters=10)
                logits = model(xa)
                val_loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
                pred = logits.argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
        rob = correct/total
        val_loss = val_loss_sum / max(total,1)
        print(f"Epoch {ep}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_rob_acc={rob:.4f} val_rob_loss={val_loss:.4f}")
        if rob > best["robust"]:
            best["robust"] = rob
            best["state"] = {k:v.cpu() for k,v in model.state_dict().items()}
    return best
