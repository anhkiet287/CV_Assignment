import platform, subprocess, json
from pathlib import Path
import torch
import torchvision

info = {
    "python": platform.python_version(),
    "system": platform.platform(),
    "device": "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "git_commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(),
    "torch_cuda": torch.version.cuda,
    "mps_available": torch.backends.mps.is_available(),
}
out = Path("tables/env.md")
out.parent.mkdir(parents=True, exist_ok=True)
lines = ["# Environment\n"]
for k,v in info.items():
    lines.append(f"- {k}: {v}")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out}")
