
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from utils.vis import bar_plot

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=str, default="experiments/results_std.csv")
ap.add_argument("--outdir", type=str, default="figures")
args = ap.parse_args()

SRC = Path(args.csv)
FIG_DIR = Path(args.outdir)
FIG_DIR.mkdir(parents=True, exist_ok=True)

if not SRC.exists():
    print(f"No {SRC} yet."); raise SystemExit(0)

df = pd.read_csv(SRC)
acc_col = "acc_def" if "acc_def" in df.columns else "acc"
eps_keys = []
for raw in df["eps"].unique():
    try:
        eps_keys.append((float(raw), raw))
    except Exception:
        continue
for _, eps_raw in sorted(eps_keys):
    sub = df[df["eps"]==eps_raw]
    labels = (sub["defense"] + ":" + sub["params"].fillna("{}")).tolist()
    values = sub[acc_col].tolist()
    bar_plot(FIG_DIR / f"bars_eps{eps_raw}.png", labels, values, title=f"Accuracy by defense @ ε={eps_raw}", ylim=(0,1))

    # recovery vs clean_penalty scatter
    if "recovery" in sub.columns and "clean_penalty" in sub.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(sub["clean_penalty"], sub["recovery"], c="tab:blue")
        for i, lbl in enumerate(labels):
            plt.annotate(lbl, (sub["clean_penalty"].iloc[i], sub["recovery"].iloc[i]), fontsize=6)
        plt.axvline(x=-0.06, color="red", linestyle="--", linewidth=1)
        plt.xlabel("clean_penalty")
        plt.ylabel("recovery")
        plt.title(f"Trade-off @ ε={eps_raw}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"tradeoff_eps{eps_raw}.png", dpi=200)
        plt.close()

print(f"Wrote bar and trade-off figures to {FIG_DIR}/")
