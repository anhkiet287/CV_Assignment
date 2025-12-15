import argparse, time, json, itertools, torch
from pathlib import Path
from models.cnn_cifar import CifarResNet18
from utils.io import append_csv, git_commit_hash
from utils.metrics import accuracy_from_logits
from utils.data import cifar_loaders, test_tensor
from defenses import apply_defense, DEFENSES

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()

@torch.no_grad()
def eval_acc(model, X, y, batch=256, spec=None):
    N = X.shape[0]; correct = 0
    t0 = time.perf_counter()
    for i in range(0, N, batch):
        xb, yb = X[i:i+batch], y[i:i+batch]
        xb = apply_defense(xb, spec or {"defense": "none"})
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    sync_device(next(model.parameters()).device)
    dt = time.perf_counter() - t0
    return correct / N, (dt / N) * 1000.0  # acc, ms/img

if __name__ == "__main__":
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", type=str, default="fgsm")
    ap.add_argument("--eps", nargs="+", type=str, default=None, help="eps in [0,1]")
    ap.add_argument("--eps255", nargs="+", type=int, default=None, help="eps in 1/255 units")
    ap.add_argument("--defense", type=str, required=True,
                    choices=list(DEFENSES.keys()))
    ap.add_argument("--k", type=int, nargs="*", default=None, help="kernel sizes (accepts multiple)")
    ap.add_argument("--sigma", type=float, nargs="*", default=None, help="sigmas (accepts multiple)")
    ap.add_argument("--results_csv", type=str, default="experiments/results_main.csv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--model_name", type=str, default="ResNet18")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--sobel_mode", type=str, default="per_channel", choices=["per_channel","luma"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pgd_iters", type=int, default=20)
    ap.add_argument("--pgd_step255", type=float, default=2.0)
    ap.add_argument("--smoke", action="store_true", help="limit to small subset for quick logging")
    args = ap.parse_args()

    if args.eps255:
        eps_list = [f"{e/255:.6f}" for e in args.eps255]
    elif args.eps:
        eps_list = args.eps
    else:
        raise SystemExit("Provide --eps or --eps255")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    Path("experiments").mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        Path(args.results_csv).unlink(missing_ok=True)

    # Load model
    model = CifarResNet18().to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    run_id = f"{int(time.time())}-{git_commit_hash()[:7]}"

    # Prepare clean set once (for acc_clean & clean_def_acc)
    _, _, test_loader = cifar_loaders(batch_size=args.batch)
    X_clean, y_clean = test_tensor(test_loader)
    if args.smoke:
        X_clean, y_clean = X_clean[:min(256, X_clean.shape[0])], y_clean[:min(256, y_clean.shape[0])]
    X_clean, y_clean = X_clean.to(device), y_clean.to(device)
    acc_clean, _ = eval_acc(model, X_clean, y_clean, batch=args.batch, spec={"defense": "none"})

    adv_cache = {}
    # Always emit a baseline 'none' row per eps (acc_adv)
    for eps in eps_list:
        if args.attack.lower() == "pgd":
            if eps not in adv_cache:
                eps_float = float(eps)
                step = args.pgd_step255 / 255.0
                iters = args.pgd_iters
                max_keep = 256 if args.smoke else None
                adv_batches = []
                lbl_batches = []
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    x0 = xb.detach()
                    x_adv = (x0 + torch.empty_like(x0).uniform_(-eps_float, eps_float)).clamp(0,1).detach()
                    for _ in range(iters):
                        x_adv.requires_grad_(True)
                        logits = model(x_adv)
                        loss = torch.nn.functional.cross_entropy(logits, yb)
                        model.zero_grad(set_to_none=True)
                        loss.backward()
                        with torch.no_grad():
                            grad = x_adv.grad.detach() if x_adv.grad is not None else torch.zeros_like(x_adv)
                            x_adv += step * grad.sign()
                            eta = (x_adv - x0).clamp(-eps_float, eps_float)
                        x_adv[:] = (x0 + eta).clamp(0,1)
                    x_adv = x_adv.detach()
                    adv_batches.append(x_adv.cpu())
                    lbl_batches.append(yb.cpu())
                    if max_keep and sum(t.shape[0] for t in adv_batches) >= max_keep:
                        break
                adv = torch.cat(adv_batches)
                lbl = torch.cat(lbl_batches)
                if max_keep:
                    adv, lbl = adv[:max_keep], lbl[:max_keep]
                adv_cache[eps] = (adv.to(device), lbl.to(device))
            X_adv, y_adv = adv_cache[eps]
        else:
            suffix = f"_{args.tag}" if args.tag else ""
            adv_file = f"data/adv/{args.attack}_eps{eps}{suffix}.pt"
            pack = torch.load(adv_file, map_location=device)
            X_adv, y_adv = pack["images"].to(device), pack["labels"].to(device)
            if args.smoke:
                keep = min(256, X_adv.shape[0])
                X_adv, y_adv = X_adv[:keep], y_adv[:keep]

        # Baseline adversarial accuracy (no defense)
        acc_adv, ms_none = eval_acc(model, X_adv, y_adv, batch=args.batch, spec={"defense": "none"})
        drop = acc_clean - acc_adv
        row_none = {
            "model": args.model_name,
            "attack": args.attack,
            "eps": eps,
            "defense": "none",
            "params": json.dumps({}),
            "acc": acc_adv,
            "acc_adv": acc_adv,
            "clean_acc": acc_clean,
            "clean_def_acc": acc_clean,            # same as no defense
            "clean_penalty": 0.0,
            "time_ms_per_img": ms_none,
            "commit": git_commit_hash(),
            "notes": "baseline-no-defense",
            "split": "test",
            "device": str(device),
            "tag": args.tag,
            "drop": drop,
            "recovery": 0.0,
            "run_id": run_id
        }
        append_csv(Path(args.results_csv), row_none)
        print(row_none)
        print("-"*40)

        # Enumerate parameter combos for the chosen defense
        ks = args.k if args.k else [None]
        sigmas = args.sigma if args.sigma else [None]
        combos = []
        if args.defense == "gaussian":
            combos = list(itertools.product(ks, sigmas))
        elif args.defense in ("median","opening","closing"):
            combos = [(k, None) for k in ks]
        elif args.defense == "sobel":
            combos = [(None, None)]
        elif args.defense == "none":
            continue  # already wrote baseline row

        for k, s in combos:
            spec = {"defense": args.defense, "k": k, "sigma": s, "mode": args.sobel_mode}
            _ = apply_defense(X_adv[:1], spec)
            acc_def, ms = eval_acc(model, X_adv, y_adv, batch=args.batch, spec=spec)
            # clean_def_acc & penalty on clean images
            clean_def_acc, _ = eval_acc(model, X_clean, y_clean, batch=args.batch, spec=spec)
            row = {
                "model": args.model_name,
                "attack": args.attack,
                "eps": eps,
                "defense": args.defense,
                "params": json.dumps({"k": k, "sigma": s, "mode": args.sobel_mode if args.defense=="sobel" else None}),
                "acc": acc_def,
                "acc_adv": acc_adv,
                "clean_acc": acc_clean,
                "clean_def_acc": clean_def_acc,
                "clean_penalty": clean_def_acc - acc_clean,
                "time_ms_per_img": ms,
                "commit": git_commit_hash(),
                "notes": "",
                "split": "test",
                "device": str(device),
                "tag": args.tag,
                "drop": drop,
                "recovery": (acc_def - acc_adv) / max(drop, 1e-6),
                "run_id": run_id
            }
            append_csv(Path(args.results_csv), row)
            print(row)
            print("-"*40)
    print(f"Eval elapsed: {time.perf_counter()-t_start:.1f}s")
