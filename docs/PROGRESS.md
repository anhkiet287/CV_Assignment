# Progress

- Checkpoints ready: `checkpoints/best94.pt` (STD, 50 epochs) and `checkpoints/adv_pgd_best.pt` (PGD-AT, 50 epochs). No retraining needed.

- Remaining runs (in order, no retraining):
  1) Generate adversarial sets: `make fgsm_sets_std fgsm_sets_at pgd20_sets_std pgd20_sets_at`
  2) Evaluate defenses on STD: `make defenses_std_fgsm defenses_std_pgd20`
  3) Evaluate defenses on PGD-AT: `make defenses_at_fgsm defenses_at_pgd20`
  4) Optional PGD restart sanity checks (subset 1k, restarts=5): `make sanity_pgd_r5_std sanity_pgd_r5_at`
  5) Aggregate + figures: `make aggregate_std aggregate_at figures_std figures_at`
  6) Compare STD vs AT tables: `python experiments/compare_std_vs_at.py --std experiments/results_std.csv --at experiments/results_at.csv --out tables/compare_std_vs_at.md`

- Evaluation-only checks:
  - Clean accuracy sanity: `make eval_std` and `make eval_at`
  - Adaptive PGD through defenses: `python experiments/eval.py --ckpt checkpoints/best94.pt --model_name ResNet18_STD --defense sobel --mode pgd --tag std --results_csv experiments/results_std.csv`
