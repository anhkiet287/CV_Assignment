import argparse
import pandas as pd
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--std", type=str, default="experiments/results_main.csv")
ap.add_argument("--at", type=str, default="experiments/results_at.csv")
ap.add_argument("--out", type=str, default="tables/compare_std_vs_at.md")
args = ap.parse_args()

std = pd.read_csv(args.std) if Path(args.std).exists() else None
at = pd.read_csv(args.at) if Path(args.at).exists() else None
if std is None or at is None:
    print("Missing input CSVs"); raise SystemExit(0)

def summarize(df, label):
    best_acc = (df.sort_values(["eps","acc","time_ms_per_img"], ascending=[True,False,True])
                  .groupby("eps").head(1))
    if "recovery" not in df.columns:
        df = df.copy()
        df["drop"] = df["clean_acc"] - df["acc_adv"]
        df["recovery"] = (df["acc"] - df["acc_adv"]) / df["drop"].clip(lower=1e-6)
    trade = (df[df["clean_penalty"] >= -0.06]
             .sort_values(["eps","recovery","time_ms_per_img"], ascending=[True,False,True])
             .groupby("eps").head(1))
    best_acc.insert(0, "model_group", label)
    trade.insert(0, "model_group", label)
    return best_acc, trade

ba_std, tr_std = summarize(std, "std")
ba_at, tr_at = summarize(at, "at")

OUT = Path(args.out); OUT.parent.mkdir(parents=True, exist_ok=True)
lines = [
    "# Best-by-accuracy (Std vs AT)\n",
    pd.concat([ba_std, ba_at]).to_markdown(index=False),
    "\n\n# Best-by-tradeoff (Std vs AT)\n",
    pd.concat([tr_std, tr_at]).to_markdown(index=False),
]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
