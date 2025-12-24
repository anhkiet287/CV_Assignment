"""
Shared conventions for attacks, defenses, filenames, and CSV schemas.

This centralizes the canonical attack strings and helpers so scripts stay aligned.
"""
from pathlib import Path
from typing import Iterable, Tuple

# Canonical attacks
ATTACK_LINF = ("fgsm", "pgd20", "square")
ATTACK_L2 = ("deepfool", "cw")
ATTACK_CHOICES = ATTACK_LINF + ATTACK_L2

# Defenses
DEFENSE_CHOICES = ("none", "gaussian", "median", "opening", "closing", "sobel")

# Default eps schedule (4/8/12 over 255)
DEFAULT_EPS255 = (4, 8, 12)

# Canonical CSV columns (order used when writing new files)
CSV_COLUMNS: Tuple[str, ...] = (
    "model_name",
    "tag",
    "split",
    "device",
    "commit",
    "attack",
    "eps",
    "eps255",
    "defense",
    "params",
    "acc_clean",
    "clean_def_acc",
    "clean_penalty",
    "acc_adv",
    "acc_def",
    "drop",
    "recovery",
    "time_ms_per_img",
    # Optional/attack-specific metadata (may be empty)
    "iters",
    "step255",
    "restarts",
    "subset_n",
    "query_budget",
    "queries_per_img",
    "median_l2",
    "mean_l2",
    "notes",
    "run_id",
)


def eps_to_str(eps: float) -> str:
    return f"{float(eps):.6f}"


def linf_adv_path(attack: str, eps: float, tag: str) -> Path:
    if attack not in ATTACK_LINF:
        raise ValueError(f"{attack} is not an L-infinity attack")
    if not tag:
        raise ValueError("tag is required for canonical file naming")
    return Path("data/adv") / f"{attack}_eps{eps_to_str(eps)}_{tag}.pt"


def l2_adv_path(attack: str, tag: str) -> Path:
    if attack not in ATTACK_L2:
        raise ValueError(f"{attack} is not an L2 attack")
    if not tag:
        raise ValueError("tag is required for canonical file naming")
    return Path("data/adv") / f"{attack}_{tag}.pt"


def normalize_attack(name: str) -> str:
    """Map common aliases to canonical attack strings."""
    if name.lower() in ("pgd", "pgd-20", "pgd_20", "pgd20"):
        return "pgd20"
    return name.lower()


def ensure_attack_allowed(name: str):
    if name not in ATTACK_CHOICES:
        raise SystemExit(f"Attack must be one of {ATTACK_CHOICES}, got {name}")


def ensure_defense_allowed(name: str):
    if name not in DEFENSE_CHOICES:
        raise SystemExit(f"Defense must be one of {DEFENSE_CHOICES}, got {name}")


def to_list(val: Iterable) -> list:
    return list(val) if isinstance(val, Iterable) else [val]
