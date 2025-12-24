"""
Backward-compatible shim that redirects to experiments/eval.py (canonical PGD-20 evaluator).
"""
import subprocess
import sys

if __name__ == "__main__":
    cmd = ["python", "experiments/eval.py"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
