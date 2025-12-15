
import subprocess, sys

def run(cmd):
    print(">>", cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        sys.exit(ret)

if __name__ == "__main__":
    run("python experiments/train_baseline.py")
    run("python experiments/run_attack.py --eps 0.015686 0.031373 0.047059")
    run("python experiments/run_defense.py --attack fgsm --eps 0.015686 0.031373 0.047059 --defense none")
    run("python experiments/run_defense.py --attack fgsm --eps 0.015686 0.031373 0.047059 --defense gaussian --k 3 5 --sigma 0.8 1.2")
    run("python experiments/run_defense.py --attack fgsm --eps 0.015686 0.031373 0.047059 --defense median --k 3 5")
    run("python experiments/run_defense.py --attack fgsm --eps 0.015686 0.031373 0.047059 --defense opening --k 3 5")
    run("python experiments/run_defense.py --attack fgsm --eps 0.015686 0.031373 0.047059 --defense closing --k 3 5")
    run("python experiments/aggregate.py")
    run("python experiments/make_figures.py")
