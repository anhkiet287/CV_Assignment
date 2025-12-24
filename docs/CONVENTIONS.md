# Conventions

## Attacks
- Canonical names (CLI + CSV): `fgsm`, `pgd20`, `square`, `deepfool`, `cw` (L2). Do not log `pgd`, `pgd-20`, etc.
- L∞ file naming: `data/adv/{attack}_eps{eps:.6f}_{tag}.pt` (e.g., `data/adv/pgd20_eps0.031373_std.pt`).
- L2 file naming: `data/adv/{attack}_{tag}.pt`.
- Epsilon: pass `--eps255 4 8 12`; store both `eps255` (int) and `eps=eps255/255` in CSV. L2 attacks leave `eps/eps255` empty and record `median_l2`/`mean_l2` if available.

## Defenses
- Names: `none`, `gaussian`, `median`, `opening`, `closing`, `sobel`.
- Inputs: NCHW float32 in [0,1]; outputs clamped to [0,1].

## CSV schema (per row)
Required columns: `model_name, tag, split, device, commit, attack, eps, eps255, defense, params(JSON), acc_clean, clean_def_acc, clean_penalty, acc_adv, acc_def, drop, recovery, time_ms_per_img`.  
Optional columns: `iters, step255, restarts, subset_n, query_budget, queries_per_img, median_l2, mean_l2, notes, run_id`.

Metric definitions:
- `acc_clean`: clean accuracy with `defense=none`.
- `clean_def_acc`: clean accuracy with current defense; `clean_penalty = clean_def_acc - acc_clean`.
- `acc_adv`: adversarial accuracy with `defense=none` (baseline for this attack/eps/tag/model).
- `acc_def`: adversarial accuracy with current defense.
- `drop = acc_clean - acc_adv`; `recovery = (acc_def - acc_adv)/max(1e-6, drop)`.
- `acc_adv` must be identical across defenses for the same `(model_name, attack, eps, eps255, tag)`.

## PGD settings
- Canonical PGD-20: `iters=20`, `step255=2`, random start.
- Restarts: log the `restarts` column; for sanity checks expect `restarts=5` to be ≤ (or ~equal to) `restarts=1`.
- When timing, synchronize CUDA/MPS before computing `time_ms_per_img`.

## Outputs
- STD results → `experiments/results_std.csv`; AT results → `experiments/results_at.csv`.
- Summaries → `tables/summary_{std,at}.md`; figures → `figures_{std,at}/*`.
