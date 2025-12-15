
from pathlib import Path
import pandas as pd
import subprocess

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def git_commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def append_csv(path: Path, row: dict):
    ensure_dir(path.parent)
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
