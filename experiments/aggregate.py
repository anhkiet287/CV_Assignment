
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=str, default="experiments/results_main.csv")
ap.add_argument("--out", type=str, default="tables/summary.md")
ap.add_argument("--penalty_limit", type=float, default=-0.06)
ap.add_argument("--max_rows", type=int, default=None)
args = ap.parse_args()

SRC = Path(args.csv); OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)
if not SRC.exists():
    print(f"No {SRC}"); raise SystemExit(0)

df = pd.read_csv(SRC)
if "drop" in df.columns:
    df = df[df["drop"] > 1e-6]
dedup_keys = [c for c in ["model","attack","eps","defense","params","tag","run_id"] if c in df.columns]
if dedup_keys:
    df = df.drop_duplicates(subset=dedup_keys, keep="last")
df["drop"] = df["clean_acc"] - df["acc_adv"]
df["recovery"] = np.where(df["drop"] > 1e-6, (df["acc"] - df["acc_adv"]) / df["drop"], 0.0)

best_acc = (df.sort_values(["eps","acc","time_ms_per_img"], ascending=[True,False,True])
              .groupby("eps").head(1))

trade = (df[df["clean_penalty"] >= args.penalty_limit]
         .sort_values(["eps","recovery","time_ms_per_img"], ascending=[True,False,True])
         .groupby("eps").head(1))

def df_to_md(df):
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)

lines = ["# Summary (best-by-accuracy)\n", df_to_md(best_acc),
         "\n\n# Summary (best-by-tradeoff)\n", df_to_md(trade),
         "\n\n# Full Results\n", df_to_md(df if args.max_rows is None else df.head(args.max_rows))]
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}")
