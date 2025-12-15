
import argparse, torch
import time
from pathlib import Path
from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders, test_tensor
from attacks.fgsm import fgsm

if __name__ == "__main__":
    def load_model(ckpt_path):
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", nargs="+", type=str, default=None, help="epsilon in [0,1]")
    ap.add_argument("--eps255", nargs="+", type=int, default=None, help="epsilon in 1/255 units (e.g., 4 8 12)")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt", help="checkpoint path")
    ap.add_argument("--tag", type=str, default="", help="suffix for saved files, e.g., 'at'")
    args = ap.parse_args()

    if args.eps255:
        eps_list = [f"{e/255:.6f}" for e in args.eps255]
    elif args.eps:
        eps_list = args.eps
    else:
        raise SystemExit("Provide --eps or --eps255")

    from utils.data import cifar_loaders, test_tensor
    def make_adv_sets_tagged(eps_list, tag, ckpt):
        from attacks.fgsm import fgsm
        from pathlib import Path
        import torch
        model, device = load_model(ckpt)
        _, _, test_loader = cifar_loaders(batch_size=256)
        X, y = test_tensor(test_loader); X, y = X.to(device), y.to(device)
        Path("data/adv").mkdir(parents=True, exist_ok=True); Path("figures/qual").mkdir(parents=True, exist_ok=True)
        for eps in eps_list:
            print(f"Generating FGSM eps={eps} (tag={tag})")
            X_adv = []
            for i in range(0, X.shape[0], 256):
                xb, yb = X[i:i+256], y[i:i+256]
                X_adv.append(fgsm(model, xb, yb, epsilon=float(eps)))
            X_adv = torch.cat(X_adv, 0).detach().cpu()
            suffix = f"_{tag}" if tag else ""
            torch.save({"images": X_adv, "labels": y.cpu()}, f"data/adv/fgsm_eps{eps}{suffix}.pt")

    t0 = time.perf_counter()
    make_adv_sets_tagged(eps_list, args.tag, args.ckpt)
    print(f"Attack gen elapsed: {time.perf_counter()-t0:.1f}s")
