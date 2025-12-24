import argparse, time, json, torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch.nn.functional as F

from models.cnn_cifar import CifarResNet18
from utils.io import append_csv, git_commit_hash
from utils.data import cifar_loaders, test_tensor
from defenses import apply_defense
from attacks.fgsm import fgsm
from experiments.constants import (
    ATTACK_CHOICES,
    ATTACK_LINF,
    DEFENSE_CHOICES,
    DEFAULT_EPS255,
    CSV_COLUMNS,
    eps_to_str,
    linf_adv_path,
    l2_adv_path,
    normalize_attack,
    ensure_attack_allowed,
    ensure_defense_allowed,
)


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@torch.no_grad()
def eval_acc(model, X, y, batch: int, spec: Dict) -> Tuple[float, float]:
    """Return accuracy and ms/img for given defense spec."""
    N = X.shape[0]
    correct = 0
    t0 = time.perf_counter()
    for i in range(0, N, batch):
        xb, yb = X[i:i + batch], y[i:i + batch]
        xb = apply_defense(xb, spec)
        logits = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    sync_device(next(model.parameters()).device)
    dt = time.perf_counter() - t0
    return correct / N, (dt / max(1, N)) * 1000.0


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


def ordered_row(data: Dict) -> Dict:
    """Align row dict to canonical CSV columns, filling missing keys with empty strings."""
    return {col: data.get(col, "") for col in CSV_COLUMNS}


def eps_pairs(args) -> List[Tuple[float, int]]:
    if args.attack not in ATTACK_LINF:
        return [(None, None)]
    if args.eps255:
        return [(e / 255.0, int(e)) for e in args.eps255]
    if args.eps:
        return [(float(e), int(round(float(e) * 255))) for e in args.eps]
    raise SystemExit("Provide --eps or --eps255 for Linf attacks.")


def load_or_generate_adv(args, model, device, eps_f: Optional[float], eps255: Optional[int], X_clean, y_clean):
    """Load canonical adv set if present; generate FGSM/PGD20 otherwise."""
    if args.attack in ATTACK_LINF:
        adv_path = linf_adv_path(args.attack, eps_f, args.tag)
        meta = {}
        if adv_path.exists():
            pack = torch.load(adv_path, map_location=device)
            X_adv, y_adv = pack["images"].to(device), pack["labels"].to(device)
            meta["notes"] = meta.get("notes", "") + "loaded"
        else:
            print(f"No cached {adv_path}, generating on-the-fly.")
            if args.attack == "fgsm":
                batches = []
                for i in range(0, X_clean.shape[0], args.batch):
                    xb, yb = X_clean[i:i + args.batch], y_clean[i:i + args.batch]
                    batches.append(fgsm(model, xb, yb, epsilon=eps_f))
                X_adv = torch.cat(batches, 0).detach()
                y_adv = y_clean.detach()
            elif args.attack == "pgd20":
                step = args.pgd_step255 / 255.0
                batches = []
                for i in range(0, X_clean.shape[0], args.batch):
                    xb, yb = X_clean[i:i + args.batch], y_clean[i:i + args.batch]
                    batches.append(pgd_attack(model, xb, yb, eps=eps_f, step=step, iters=args.pgd_iters, restarts=args.restarts))
                X_adv = torch.cat(batches, 0).detach()
                y_adv = y_clean.detach()
            else:
                raise SystemExit(f"Attack {args.attack} not supported for generation.")
            if args.save_generated:
                torch.save({"images": X_adv.cpu(), "labels": y_adv.cpu()}, adv_path)
                meta["notes"] = "generated"
        return X_adv.to(device), y_adv.to(device), meta
    # L2 attacks expect pre-generated files
    adv_path = l2_adv_path(args.attack, args.tag)
    if not adv_path.exists():
        raise SystemExit(f"Missing L2 adversarial set: {adv_path}")
    pack = torch.load(adv_path, map_location=device)
    meta = {"median_l2": pack.get("median_l2", ""), "mean_l2": pack.get("mean_l2", ""), "notes": "loaded"}
    return pack["images"].to(device), pack["labels"].to(device), meta


def maybe_subset(X, y, n: Optional[int]):
    if n is None:
        return X, y
    keep = min(n, X.shape[0])
    return X[:keep], y[:keep]


def verify_acc_adv_constant(csv_path: Path, tol: float = 1e-6):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    required_cols = {"model_name", "attack", "eps", "eps255", "tag", "acc_adv"}
    if not required_cols.issubset(set(df.columns)):
        return
    group_cols = [c for c in ["model_name", "attack", "eps", "eps255", "tag", "iters", "step255", "restarts", "subset_n"] if c in df.columns]
    grouped = df.groupby(group_cols)
    for key, g in grouped:
        diff = g["acc_adv"].max() - g["acc_adv"].min()
        if diff > tol:
            print(f"[acc_adv inconsistency] {key} diff={diff:.6f}")
            raise SystemExit(1)


def compose_notes(meta_note: str, user_note: str, fallback: str) -> str:
    parts = [p for p in (meta_note, user_note, fallback) if p]
    # Preserve order but drop duplicates
    deduped = []
    for p in parts:
        if p not in deduped:
            deduped.append(p)
    return "; ".join(deduped)


def check_restart_monotonicity(csv_path: Path, tol: float = 1e-4):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    needed = {"model_name", "attack", "eps", "eps255", "tag", "defense", "restarts", "acc_def"}
    if not needed.issubset(set(df.columns)):
        return
    df = df.copy()
    df["restarts_num"] = pd.to_numeric(df["restarts"], errors="coerce")
    df = df.dropna(subset=["restarts_num"])
    base = df[df["restarts_num"] <= 1.0]
    higher = df[df["restarts_num"] > 1.0]
    for _, row in higher.iterrows():
        key = (
            row["model_name"],
            row["attack"],
            row["eps"],
            row["eps255"],
            row["tag"],
            row["defense"],
        )
        mask = (
            (base["model_name"] == key[0]) &
            (base["attack"] == key[1]) &
            (base["eps"] == key[2]) &
            (base["eps255"] == key[3]) &
            (base["tag"] == key[4]) &
            (base["defense"] == key[5])
        )
        if mask.sum() == 0:
            continue
        max_base = base.loc[mask, "acc_def"].max()
        if row["acc_def"] > max_base + tol:
            print(f"[restarts sanity] restarts={row['restarts']} acc_def {row['acc_def']:.4f} > restarts<=1 {max_base:.4f} for {key}")
            raise SystemExit(1)


if __name__ == "__main__":
    t_start = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", type=str, required=True, help=f"attack in {ATTACK_CHOICES}")
    ap.add_argument("--eps", nargs="+", type=str, default=None, help="eps in [0,1] (Linf)")
    ap.add_argument("--eps255", nargs="+", type=int, default=DEFAULT_EPS255, help="eps in 1/255 units (Linf)")
    ap.add_argument("--defense", type=str, required=True, help=f"defense in {DEFENSE_CHOICES}")
    ap.add_argument("--k", type=int, nargs="*", default=None, help="kernel sizes (accepts multiple)")
    ap.add_argument("--sigma", type=float, nargs="*", default=None, help="sigmas (accepts multiple)")
    ap.add_argument("--results_csv", type=str, default="experiments/results_std.csv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--model_name", type=str, default="ResNet18")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--sobel_mode", type=str, default="per_channel", choices=["per_channel", "luma"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pgd_iters", type=int, default=20)
    ap.add_argument("--pgd_step255", type=float, default=2.0)
    ap.add_argument("--restarts", type=int, default=1, help="PGD restarts when generating attacks")
    ap.add_argument("--subset_n", type=int, default=None, help="optional subset size for quick eval")
    ap.add_argument("--save_generated", action="store_true", help="save generated adversarial sets if missing")
    ap.add_argument("--notes", type=str, default="", help="additional notes to record in CSV")
    args = ap.parse_args()

    args.attack = normalize_attack(args.attack)
    ensure_attack_allowed(args.attack)
    args.defense = args.defense.lower()
    ensure_defense_allowed(args.defense)

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
    X_clean, y_clean = maybe_subset(X_clean, y_clean, args.subset_n)
    X_clean, y_clean = X_clean.to(device), y_clean.to(device)
    acc_clean, _ = eval_acc(model, X_clean, y_clean, batch=args.batch, spec={"defense": "none"})

    eps_list = eps_pairs(args)
    for eps_f, eps255 in eps_list:
        eps_str = eps_to_str(eps_f) if eps_f is not None else ""
        X_adv, y_adv, meta = load_or_generate_adv(args, model, device, eps_f, eps255, X_clean, y_clean)
        X_adv, y_adv = maybe_subset(X_adv, y_adv, args.subset_n)

        # Baseline adversarial accuracy (no defense)
        acc_adv, ms_none = eval_acc(model, X_adv, y_adv, batch=args.batch, spec={"defense": "none"})
        drop = acc_clean - acc_adv
        base_row = ordered_row({
            "model_name": args.model_name,
            "tag": args.tag,
            "split": "test",
            "device": str(device),
            "commit": git_commit_hash(),
            "attack": args.attack,
            "eps": eps_str,
            "eps255": eps255 if eps255 is not None else "",
            "defense": "none",
            "params": json.dumps({}),
            "acc_clean": acc_clean,
            "clean_def_acc": acc_clean,
            "clean_penalty": 0.0,
            "acc_adv": acc_adv,
            "acc_def": acc_adv,
            "drop": drop,
            "recovery": 0.0,
            "time_ms_per_img": ms_none,
            "iters": args.pgd_iters if args.attack == "pgd20" else "",
            "step255": args.pgd_step255 if args.attack == "pgd20" else "",
            "restarts": args.restarts if args.attack == "pgd20" else "",
            "subset_n": args.subset_n or "",
            "median_l2": meta.get("median_l2", "") if args.attack not in ATTACK_LINF else "",
            "mean_l2": meta.get("mean_l2", "") if args.attack not in ATTACK_LINF else "",
            "notes": compose_notes(meta.get("notes", ""), args.notes, "baseline-no-defense"),
            "run_id": run_id,
        })
        append_csv(Path(args.results_csv), base_row)
        print(base_row)
        print("-" * 40)

        if args.defense == "none":
            continue

        # Enumerate parameter combos for the chosen defense
        ks = args.k if args.k else [None]
        sigmas = args.sigma if args.sigma else [None]
        combos: List[Tuple[Optional[int], Optional[float]]] = []
        if args.defense == "gaussian":
            for k in ks:
                for s in sigmas:
                    combos.append((k, s))
        elif args.defense in ("median", "opening", "closing"):
            combos = [(k, None) for k in ks]
        elif args.defense == "sobel":
            combos = [(None, None)]

        for k, s in combos:
            spec = {"defense": args.defense, "k": k, "sigma": s, "mode": args.sobel_mode}
            _ = apply_defense(X_adv[:1], spec)  # shape sanity
            acc_def, ms = eval_acc(model, X_adv, y_adv, batch=args.batch, spec=spec)
            clean_def_acc, _ = eval_acc(model, X_clean, y_clean, batch=args.batch, spec=spec)
            row = ordered_row({
                "model_name": args.model_name,
                "tag": args.tag,
                "split": "test",
                "device": str(device),
                "commit": git_commit_hash(),
                "attack": args.attack,
                "eps": eps_str,
                "eps255": eps255 if eps255 is not None else "",
                "defense": args.defense,
                "params": json.dumps({"k": k, "sigma": s, "mode": args.sobel_mode if args.defense == "sobel" else None}),
                "acc_clean": acc_clean,
                "clean_def_acc": clean_def_acc,
                "clean_penalty": clean_def_acc - acc_clean,
                "acc_adv": acc_adv,
                "acc_def": acc_def,
                "drop": drop,
                "recovery": (acc_def - acc_adv) / max(drop, 1e-6),
                "time_ms_per_img": ms,
                "iters": args.pgd_iters if args.attack == "pgd20" else "",
                "step255": args.pgd_step255 if args.attack == "pgd20" else "",
                "restarts": args.restarts if args.attack == "pgd20" else "",
                "subset_n": args.subset_n or "",
                "median_l2": meta.get("median_l2", "") if args.attack not in ATTACK_LINF else "",
                "mean_l2": meta.get("mean_l2", "") if args.attack not in ATTACK_LINF else "",
                "notes": compose_notes(meta.get("notes", ""), args.notes, ""),
                "run_id": run_id,
            })
            append_csv(Path(args.results_csv), row)
            print(row)
            print("-" * 40)

    csv_path = Path(args.results_csv)
    verify_acc_adv_constant(csv_path)
    check_restart_monotonicity(csv_path)
    print(f"Eval elapsed: {time.perf_counter()-t_start:.1f}s")
