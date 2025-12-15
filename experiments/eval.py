import argparse, json, time
import torch
import torch.nn.functional as F
from pathlib import Path
from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders, test_tensor
from defenses import apply_defense, DEFENSES
from utils.io import append_csv, git_commit_hash

def build_model(name: str):
    return CifarResNet18(num_classes=10)

def pgd_eval(model, X, y, eps, step, iters, defense="none", k=None, sigma=None, bpda=True, device=None, batch=256, mode=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else
                                    "mps" if torch.backends.mps.is_available() else "cpu")
    N = X.shape[0]; correct = 0
    for i in range(0, N, batch):
        xb = X[i:i+batch].to(device)
        yb = y[i:i+batch].to(device)
        x0 = xb.detach()
        x_adv = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0,1).detach().requires_grad_(True)
        for _ in range(iters):
            model.zero_grad(set_to_none=True)
            if defense == "none":
                logits = model(x_adv)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                g = x_adv.grad.detach() if x_adv.grad is not None else torch.zeros_like(x_adv)
            else:
                x_def = apply_defense(x_adv.detach(), {"defense": defense, "k": k, "sigma": sigma, "mode": mode}).requires_grad_(True)
                logits = model(x_def)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                if bpda:
                    g = x_def.grad.detach() if x_def.grad is not None else torch.zeros_like(x_adv)
                else:
                    g = torch.zeros_like(x_adv)
            with torch.no_grad():
                x_adv += step * g.sign()
                eta = (x_adv - x0).clamp(-eps, eps)
                x_adv[:] = (x0 + eta).clamp(0,1)
                if x_adv.grad is not None:
                    x_adv.grad.zero_()
        with torch.no_grad():
            x_in = apply_defense(x_adv, {"defense": defense, "k": k, "sigma": sigma, "mode": mode}) if defense != "none" else x_adv
            pred = model(x_in).argmax(1)
            correct += (pred == yb).sum().item()
    return correct / N

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18")
    ap.add_argument("--model_name", type=str, default=None, help="label for results rows")
    ap.add_argument("--mode", type=str, default="baseline", choices=["baseline","pgd","full"], help="baseline=clean only, pgd=robust only, full=both")
    ap.add_argument("--eps255", nargs="+", type=int, default=[8])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--step255", type=float, default=2.0)
    ap.add_argument("--defense", type=str, default="none", choices=list(DEFENSES.keys()))
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--sobel_mode", type=str, default="per_channel", choices=["per_channel","luma"])
    ap.add_argument("--bpda", action="store_true", default=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--results_csv", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")

    _, _, test_loader = cifar_loaders(batch_size=args.batch)
    X, y = test_tensor(test_loader)
    X, y = X.to(device), y.to(device)

    model = build_model(args.model).to(device).eval()
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)

    model_label = args.model_name or args.model
    print(f"# Eval | model={model_label} | ckpt={args.ckpt} | defense={args.defense} | mode={args.mode}")

    # Clean eval
    with torch.no_grad():
        correct = 0
        for i in range(0, X.shape[0], args.batch):
            xb, yb = X[i:i+args.batch], y[i:i+args.batch]
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
    acc_clean = correct / y.shape[0]
    print(f"clean_acc={acc_clean:.4f}")

    # PGD eval if requested
    if args.mode in ("pgd","full"):
        run_id = f"{int(time.time())}-{git_commit_hash()[:7]}"
        for e in args.eps255:
            eps = e/255.0
            step = args.step255/255.0
            acc = pgd_eval(model, X, y, eps=eps, step=step, iters=args.iters,
                           defense=args.defense, k=args.k, sigma=args.sigma,
                           bpda=args.bpda, device=device, batch=args.batch, mode=args.sobel_mode)
            print(f"eps={e}/255, robust_acc_pgd{args.iters}={acc:.4f}")
            if args.results_csv:
                row = {
                    "model": model_label,
                    "attack": f"pgd{args.iters}",
                    "eps": f"{eps:.6f}",
                    "defense": args.defense,
                    "params": json.dumps({"k": args.k, "sigma": args.sigma, "mode": args.sobel_mode if args.defense=='sobel' else None}),
                    "acc": acc,
                    "acc_adv": acc,
                    "clean_acc": acc_clean,
                    "clean_def_acc": acc_clean,
                    "clean_penalty": 0.0,
                    "time_ms_per_img": "",
                    "commit": git_commit_hash(),
                    "notes": "pgd_eval",
                    "split": "test",
                    "device": str(device),
                    "tag": args.tag,
                    "drop": acc_clean - acc,
                    "recovery": 0.0,
                    "run_id": run_id
                }
                append_csv(Path(args.results_csv), row)
