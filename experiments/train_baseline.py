
import argparse
import time
import json
import torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders
from utils.metrics import accuracy_from_logits
from utils.seed import set_seed
from tqdm import tqdm

# REUSE_FROM_NOTEBOOK: your Trainer/Evaluator if available.
# Minimal quick trainer:
def train_quick(epochs=50, lr=0.1, model_name="resnet18", device=None):
    device = device or torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_loader, val_loader, _ = cifar_loaders(batch_size=128)
    model = CifarResNet18().to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = optim.lr_scheduler.MultiStepLR(opt, milestones=[25, 40], gamma=0.1)
    best_acc, best_state = 0.0, None

    for ep in range(1, epochs+1):
        model.train()
        for x,y in tqdm(train_loader, desc=f"Train[{ep}/{epochs}]"):
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
        # eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1)==y).sum().item()
                total += y.numel()
        acc = correct/total
        print(f"Val acc: {acc:.4f}")
        sched.step()
        if acc > best_acc:
            best_acc, best_state = acc, {k:v.cpu() for k,v in model.state_dict().items()}

    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(best_state, "checkpoints/best.pt")
    print(f"Saved best.pt with acc_clean={best_acc:.4f}")
    return best_acc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.1)
    args = ap.parse_args()

    set_seed(42)
    t0 = time.perf_counter()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    best_acc = train_quick(epochs=args.epochs, lr=args.lr, model_name=args.model, device=device)
    elapsed = time.perf_counter()-t0
    print(f"Train elapsed: {elapsed:.1f}s")
    Path("tables").mkdir(parents=True, exist_ok=True)
    log = {
        "run": "baseline",
        "model": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "best_acc": best_acc,
        "elapsed_sec": elapsed,
        "device": str(device)
    }
    with open("tables/train_logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")
