import argparse, json, torch
from pathlib import Path
from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders, test_tensor
from defenses.filters import gaussian, median
from defenses.morphology import opening, closing
from utils.vis import save_grid

def build_model(name: str):
    return CifarResNet18(num_classes=10)

def apply_defense(x, spec):
    name = spec.get("defense", "none")
    k = spec.get("k")
    sigma = spec.get("sigma")
    if name == "gaussian": return gaussian(x, k or 3, float(sigma or 1.0))
    if name == "median":   return median(x, k or 3)
    if name == "opening":  return opening(x, k or 3)
    if name == "closing":  return closing(x, k or 3)
    return x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18"])
    ap.add_argument("--attack_file", type=str, required=True)
    ap.add_argument("--defense", type=str, default='{"defense":"opening","k":3}')
    ap.add_argument("--out", type=str, default="figures/case_study.png")
    args = ap.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    spec = json.loads(args.defense)
    model = build_model(args.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    pack = torch.load(args.attack_file, map_location=device)
    X_adv, y = pack["images"].to(device), pack["labels"].to(device)

    # clean references from held-out test split
    _, _, test_loader = cifar_loaders(batch_size=256)
    X_clean, _ = test_tensor(test_loader); X_clean = X_clean.to(device)

    with torch.no_grad():
        pred_clean = model(X_clean).argmax(1)
        pred_adv = model(X_adv).argmax(1)
        pred_def = model(apply_defense(X_adv, spec)).argmax(1)

    idx = []
    limit = min(5000, y.shape[0])
    for i in range(limit):
        if pred_clean[i] == y[i] and pred_adv[i] != y[i] and pred_def[i] == y[i]:
            idx.append(i)
        if len(idx) >= 32:
            break
    if len(idx) < 16:
        idx = list(range(min(32, y.shape[0])))

    grid = torch.cat([X_clean[idx], X_adv[idx], apply_defense(X_adv[idx], spec)], dim=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_grid(Path(args.out), grid, nrow=len(idx), title=f"Top: clean | Mid: adv | Bot: defended {spec}")
    print(f"Saved {args.out}")
