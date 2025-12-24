PY=python
STD_CKPT=checkpoints/best94.pt
AT_CKPT=checkpoints/adv_pgd_best.pt
EPS255=4 8 12
RESULT_STD=experiments/results_std.csv
RESULT_AT=experiments/results_at.csv

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	$(PY) experiments/train_baseline.py

test:
	pytest -q

eval_std:
	$(PY) experiments/eval.py --ckpt $(STD_CKPT) --model_name ResNet18_STD --mode baseline --defense none --tag std

eval_at:
	$(PY) experiments/eval.py --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --mode baseline --defense none --tag at

fgsm_sets_std:
	$(PY) experiments/run_attack.py --attack fgsm --eps255 $(EPS255) --ckpt $(STD_CKPT) --tag std

fgsm_sets_at:
	$(PY) experiments/run_attack.py --attack fgsm --eps255 $(EPS255) --ckpt $(AT_CKPT) --tag at

pgd20_sets_std:
	$(PY) experiments/run_attack.py --attack pgd20 --eps255 $(EPS255) --ckpt $(STD_CKPT) --tag std

pgd20_sets_at:
	$(PY) experiments/run_attack.py --attack pgd20 --eps255 $(EPS255) --ckpt $(AT_CKPT) --tag at

defenses_std_fgsm:
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense none    --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --overwrite
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense median  --k 3 --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD)
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense opening --k 3 --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD)
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense closing --k 3 --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD)
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense sobel           --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD)

defenses_at_fgsm:
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense none    --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT) --overwrite
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense median  --k 3 --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT)
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense opening --k 3 --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT)
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense closing --k 3 --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT)
	$(PY) experiments/run_defense.py --attack fgsm --eps255 $(EPS255) --defense sobel           --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT)

defenses_std_pgd20:
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense none    --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --save_generated
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense opening --k 3 --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --save_generated
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense median  --k 3 --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --save_generated
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense sobel           --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --save_generated

defenses_at_pgd20:
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense none    --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT) --save_generated
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense opening --k 3 --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT) --save_generated
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense median  --k 3 --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT) --save_generated
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 $(EPS255) --defense sobel           --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT) --save_generated

aggregate_std:
	$(PY) experiments/aggregate.py --csv $(RESULT_STD) --out tables/summary_std.md

aggregate_at:
	$(PY) experiments/aggregate.py --csv $(RESULT_AT) --out tables/summary_at.md

figures_std:
	$(PY) experiments/make_figures.py --csv $(RESULT_STD) --outdir figures_std

figures_at:
	$(PY) experiments/make_figures.py --csv $(RESULT_AT) --outdir figures_at

sanity_pgd_r5_std:
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 8 --defense none --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --subset_n 1000 --restarts 5 --notes sanity_pgd_restarts --save_generated

sanity_pgd_r5_at:
	$(PY) experiments/run_defense.py --attack pgd20 --eps255 8 --defense none --ckpt $(AT_CKPT) --model_name ResNet18_PGDAT --tag at --results_csv $(RESULT_AT) --subset_n 1000 --restarts 5 --notes sanity_pgd_restarts --save_generated

aggregate: aggregate_std aggregate_at
figures: figures_std figures_at

std_full: fgsm_sets_std pgd20_sets_std defenses_std_fgsm defenses_std_pgd20 aggregate_std figures_std

at_full: fgsm_sets_at pgd20_sets_at defenses_at_fgsm defenses_at_pgd20 aggregate_at figures_at

attack: fgsm_sets_std fgsm_sets_at
defense: defenses_std_fgsm defenses_at_fgsm

reproduce: std_full at_full

smoke:
	$(PY) experiments/run_attack.py --attack fgsm --eps255 4 --ckpt $(STD_CKPT) --tag std
	$(PY) experiments/run_defense.py --attack fgsm --eps255 4 --defense none --ckpt $(STD_CKPT) --model_name ResNet18_STD --tag std --results_csv $(RESULT_STD) --overwrite --subset_n 256
