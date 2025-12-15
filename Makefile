setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python experiments/train_baseline.py

attack:
	python experiments/run_attack.py --eps255 4 8 12 --ckpt checkpoints/best94.pt --tag std
	python experiments/run_attack.py --eps255 4 8 12 --ckpt checkpoints/adv_pgd_best.pt --tag at

defense:
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense none    --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv --overwrite
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense median  --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense opening --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense closing --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense sobel           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack pgd  --eps255 4 --defense none           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack pgd  --eps255 8 --defense none           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack pgd  --eps255 12 --defense none           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv

defense_std_fgsm:
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense none    --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv --overwrite
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense median  --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense opening --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense closing --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense sobel           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv

defense_std_pgd:
	python experiments/run_defense.py --attack pgd  --eps255 4 --defense none           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack pgd  --eps255 8 --defense none           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack pgd  --eps255 12 --defense none           --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv

defense_at_fgsm:
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense none    --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv --overwrite
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense median  --k 3 --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense opening --k 3 --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense closing --k 3 --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense sobel           --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv

defense_at_pgd:
	python experiments/run_defense.py --attack pgd  --eps255 4 --defense none           --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack pgd  --eps255 8 --defense none           --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack pgd  --eps255 12 --defense none           --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv

aggregate:
	python experiments/aggregate.py

figures:
	python experiments/make_figures.py

reproduce: train attack defense aggregate figures

std_full:
	python experiments/run_attack.py --eps255 4 8 12 --ckpt checkpoints/best94.pt --tag std
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense none    --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv --overwrite
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense opening --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense median  --k 3 --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense sobel --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/run_defense.py --attack pgd  --eps255 4 8 12 --defense none  --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv
	python experiments/aggregate.py --csv experiments/results_main.csv --out tables/summary_std.md
	python experiments/make_figures.py --csv experiments/results_main.csv --outdir figures_std

at_full:
	python experiments/run_attack.py --eps255 4 8 12 --ckpt checkpoints/adv_pgd_best.pt --tag at
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense none    --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv --overwrite
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense opening --k 3 --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense median  --k 3 --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense sobel --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/run_defense.py --attack pgd  --eps255 4 8 12 --defense none  --ckpt checkpoints/adv_pgd_best.pt --model_name ResNet18_PGDAT --tag at --results_csv experiments/results_at.csv
	python experiments/aggregate.py --csv experiments/results_at.csv --out tables/summary_at.md
	python experiments/make_figures.py --csv experiments/results_at.csv --outdir figures_at

test:
	pytest -q

smoke:
	python experiments/run_attack.py --eps255 4 8 12 --ckpt checkpoints/best94.pt --tag std
	python experiments/run_defense.py --attack fgsm --eps255 4 8 12 --defense none --ckpt checkpoints/best94.pt --model_name ResNet18_STD --tag std --results_csv experiments/results_main.csv --overwrite --smoke
	python experiments/aggregate.py --csv experiments/results_main.csv --out tables/summary_std.md