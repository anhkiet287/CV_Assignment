# Classical Image Processing as Input Defenses (CIFAR-10, FGSM)

## Goal
Evaluate classical inference-time defenses (Gaussian, Median, Opening/Closing, Sobel) against FGSM adversarial examples on CIFAR-10 classifiers (ResNet-18 baseline and PGD-AT).

## Frozen I/O
- Tensor layout: **NCHW**
- Range: **[0,1]** float32
- Adversarial sets (L∞): `data/adv/{attack}_eps{eps:.6f}_{tag}.pt` (e.g., `data/adv/fgsm_eps0.031373_std.pt`) with `{"images": Tensor[N,C,H,W], "labels": LongTensor[N]}`

## Pipeline
1) Train baseline / PGD-AT (or reuse checkpoints)
2) Generate FGSM/PGD20 sets for ε∈{4/255,8/255,12/255} → `data/adv/*.pt`
3) Run defenses (none, opening k=3, median k=3, sobel) → `experiments/results_{std,at}.csv`
4) Aggregate & figures → `tables/summary_*.md`, `figures_*/*.png`

## Metrics
- `acc_clean`, `acc_adv(ε)`, `acc_def(ε)`
- `drop(ε)=acc_clean−acc_adv(ε)`
- `recovery(ε)=(acc_def(ε)−acc_adv(ε))/max(1e-6,drop(ε))`
- `clean_penalty=acc_def_clean−acc_clean`
- `time_ms_per_img` (over 1k images)

## Deliverables
- CSVs: `experiments/results_std.csv`, `experiments/results_at.csv`
- Tables: `tables/summary_std.md`, `tables/summary_at.md`, `tables/compare_std_vs_at.md`, `tables/env.md`
- Figures: `figures_std/*.png`, `figures_at/*.png`, case studies `figures/case_std.png`, `figures/case_at.png`
- Checkpoints: `checkpoints/best*.pt`, `checkpoints/adv_pgd_best.pt`

## Quickstart (env + reproducible run)
```bash
# 1) Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Capture environment metadata
python experiments/env_capture.py

# 3) Train (optional if you have checkpoints)
python experiments/train_baseline.py --model resnet18 --epochs 50 --lr 0.1
python experiments/train_adv_pgd.py   # PGD-AT ResNet-18, 50 epochs

# 4) Generate adversarial sets (FGSM + PGD20) for std/at
make fgsm_sets_std fgsm_sets_at
make pgd20_sets_std pgd20_sets_at

# 5) Run defenses (none, opening k=3, median k=3, sobel) and aggregate/figures
make std_full   # writes experiments/results_std.csv, tables/summary_std.md, figures_std/*
make at_full    # writes experiments/results_at.csv, tables/summary_at.md, figures_at/*

# 6) PGD-20 robustness sanity (optional; logs into the same CSVs)
make sanity_pgd_r5_std
make sanity_pgd_r5_at

# 7) Compare std vs AT tables
python experiments/compare_std_vs_at.py --std experiments/results_std.csv --at experiments/results_at.csv --out tables/compare_std_vs_at.md
```

## Defenses supported
- Gaussian (k=3/5, sigma=0.8/1.2)
- Median (k=3/5)
- Opening/Closing (k=3/5)
- Sobel gradient magnitude (per-channel, replicated to 3ch)
- None (baseline)

All operate on NCHW float32 in [0,1], CPU/CUDA/MPS.

### Sobel defense details
- Implementation: `defenses/edges.py` → `sobel_mag` computes per-channel (or luma) Sobel gradients, normalizes the magnitude per image to [0,1], and replicates to 3 channels before the model.
- Usage in experiments: `experiments/run_defense.py` applies Sobel once at inference time to both clean and adversarial inputs; the PGD crafted there ignores the defense (non-adaptive) so the robustness gains mostly reflect gradient masking/obfuscation.
- Adaptive check: `experiments/eval.py` supports `--defense sobel --bpda` so PGD backprops through the Sobel step (BPDA by default). Use `experiments/eval.py --defense sobel --mode pgd` or the `sanity_pgd_r5_*` targets to see how robustness changes under an attack aware of the preprocessing.
