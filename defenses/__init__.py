"""
Inference-time defenses registry.

Supported names:
- gaussian: params {\"k\": int, \"sigma\": float}
- median: params {\"k\": int}
- opening/closing: params {\"k\": int}
- sobel: params {\"mode\": \"per_channel\"|\"luma\"}

All expect inputs NCHW float32 in [0,1] and clamp outputs to [0,1].
"""
from typing import Dict
import torch
from .filters import gaussian, median
from .morphology import opening, closing
from .edges import sobel_mag

DEFENSES = {k: k for k in ["none", "gaussian", "median", "opening", "closing", "sobel"]}

def apply_defense(x: torch.Tensor, spec: Dict) -> torch.Tensor:
    """Apply a registered defense to tensor x based on spec dict."""
    name = spec.get("defense", "none")
    k = spec.get("k")
    sigma = spec.get("sigma")
    mode = spec.get("mode", "per_channel")
    if name == "none":
        return x
    if name == "gaussian":
        return gaussian(x, k or 3, float(sigma or 1.0))
    if name == "median":
        return median(x, k or 3)
    if name == "opening":
        return opening(x, k or 3)
    if name == "closing":
        return closing(x, k or 3)
    if name == "sobel":
        return sobel_mag(x, mode=mode)
    raise ValueError(f"Unknown defense: {name}")
