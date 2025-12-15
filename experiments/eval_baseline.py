
import argparse, torch, time
from models.cnn_cifar import CifarResNet18
from utils.data import cifar_loaders
from utils.metrics import accuracy_from_logits

def build_model(name: str):
    return CifarResNet18(num_classes=10)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18"])
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    _, _, test_loader = cifar_loaders(batch_size=args.batch)
    model = build_model(args.model).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
    print(f"acc_clean={correct/total:.4f}  (model={args.model}, ckpt={args.ckpt})  elapsed={time.perf_counter()-t0:.1f}s")
