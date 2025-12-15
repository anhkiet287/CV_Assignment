# experiments/train_adv_pgd.py
import torch, torch.optim as optim
import time, json
from pathlib import Path
from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders
from utils.seed import set_seed
from adv_training.pgd_at import train_pgd_at

if __name__ == "__main__":
    set_seed(42)
    t0 = time.perf_counter()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_loader, val_loader, _ = cifar_loaders(batch_size=128)
    model = CifarResNet18()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    best = train_pgd_at(model, train_loader, val_loader,
                        epochs=50, eps=8/255, step=2/255, iters=7, opt=opt, device=device, sched=sched)
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], "checkpoints/adv_pgd_best.pt")
    print(f"Saved adv_pgd_best.pt with robust_acc_pgd10={best['robust']:.4f}")
    elapsed = time.perf_counter()-t0
    print(f"Train elapsed: {elapsed:.1f}s")
    Path("tables").mkdir(parents=True, exist_ok=True)
    log = {
        "run": "adv_pgd",
        "model": "resnet18",
        "epochs": 50,
        "iters": 7,
        "eps": 8/255,
        "step": 2/255,
        "lr": 0.1,
        "best_robust": best["robust"],
        "elapsed_sec": elapsed,
        "device": str(device)
    }
    with open("tables/train_logs.json", "a") as f:
        f.write(json.dumps(log) + "\n")
