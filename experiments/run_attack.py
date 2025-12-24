
import argparse
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders, test_tensor
from attacks.fgsm import fgsm
from experiments.constants import (
    ATTACK_CHOICES,
    ATTACK_LINF,
    DEFAULT_EPS255,
    eps_to_str,
    linf_adv_path,
    normalize_attack,
    ensure_attack_allowed,
)


def load_model(ckpt_path: str):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = CifarResNet18().to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model, device


def pgd_attack(model, x, y, eps: float, step: float, iters: int, restarts: int = 1):
    """Linf PGD with optional restarts; returns adversarial tensor."""
    x0 = x.detach()
    best_adv = None
    best_loss = None
    for _ in range(max(1, restarts)):
        xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0, 1).detach()
        for _ in range(iters):
            xa.requires_grad_(True)
            logits = model(xa)
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, xa, only_inputs=True)[0]
            with torch.no_grad():
                xa = xa + step * grad.sign()
                eta = (xa - x0).clamp(-eps, eps)
                xa = (x0 + eta).clamp(0, 1)
        with torch.no_grad():
            logits = model(xa)
            loss = F.cross_entropy(logits, y, reduction="none")
        if best_adv is None:
            best_adv, best_loss = xa.detach(), loss.detach()
        else:
            replace = loss > best_loss
            best_adv[replace] = xa.detach()[replace]
            best_loss[replace] = loss.detach()[replace]
    return best_adv.detach()


def parse_eps_list(args) -> List[Tuple[float, int]]:
    if args.attack not in ATTACK_LINF:
        return []
    if args.eps255:
        return [(e / 255.0, int(e)) for e in args.eps255]
    if args.eps:
        return [(float(e), int(round(float(e) * 255))) for e in args.eps]
    raise SystemExit("Provide --eps or --eps255 for Linf attacks.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", type=str, required=True, help=f"attack in {ATTACK_CHOICES}")
    ap.add_argument("--eps", nargs="+", type=str, default=None, help="epsilon in [0,1]")
    ap.add_argument("--eps255", nargs="+", type=int, default=DEFAULT_EPS255, help="epsilon in 1/255 units (e.g., 4 8 12) for Linf attacks")
    ap.add_argument("--iters", type=int, default=20, help="PGD iterations (pgd20 canonical)")
    ap.add_argument("--step255", type=float, default=2.0, help="PGD step size in 1/255 units")
    ap.add_argument("--restarts", type=int, default=1, help="PGD restarts for generation")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="checkpoint path")
    ap.add_argument("--tag", type=str, required=True, help="suffix for saved files, e.g., std|at")
    args = ap.parse_args()

    args.attack = normalize_attack(args.attack)
    ensure_attack_allowed(args.attack)

    eps_items = parse_eps_list(args)
    model, device = load_model(args.ckpt)
    _, _, test_loader = cifar_loaders(batch_size=args.batch)
    X, y = test_tensor(test_loader)
    X, y = X.to(device), y.to(device)

    Path("data/adv").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if args.attack == "fgsm":
        for eps_f, eps255 in eps_items:
            print(f"[fgsm] eps={eps_f:.6f} ({eps255}/255) tag={args.tag}")
            adv_batches = []
            for i in range(0, X.shape[0], args.batch):
                xb, yb = X[i:i+args.batch], y[i:i+args.batch]
                adv_batches.append(fgsm(model, xb, yb, epsilon=eps_f))
            adv = torch.cat(adv_batches, 0).detach().cpu()
            torch.save({"images": adv, "labels": y.cpu()}, linf_adv_path("fgsm", eps_f, args.tag))
    elif args.attack == "pgd20":
        step = args.step255 / 255.0
        for eps_f, eps255 in eps_items:
            print(f"[pgd20] eps={eps_f:.6f} ({eps255}/255) iters={args.iters} step={step:.6f} restarts={args.restarts} tag={args.tag}")
            adv_batches = []
            for i in range(0, X.shape[0], args.batch):
                xb, yb = X[i:i+args.batch], y[i:i+args.batch]
                adv_batches.append(pgd_attack(model, xb, yb, eps=eps_f, step=step, iters=args.iters, restarts=args.restarts))
            adv = torch.cat(adv_batches, 0).detach().cpu()
            torch.save({"images": adv, "labels": y.cpu()}, linf_adv_path("pgd20", eps_f, args.tag))
    else:
        raise SystemExit(f"Attack {args.attack} not implemented in run_attack.py")

    print(f"Attack gen elapsed: {time.perf_counter()-t0:.1f}s")
