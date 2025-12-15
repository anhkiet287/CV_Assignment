import torch
from defenses import apply_defense

def _make_input():
    torch.manual_seed(0)
    return torch.rand(2, 3, 32, 32)

def test_defense_shapes_and_range():
    x = _make_input()
    specs = [
        {"defense": "gaussian", "k": 3, "sigma": 1.0},
        {"defense": "median", "k": 3},
        {"defense": "opening", "k": 3},
        {"defense": "closing", "k": 3},
        {"defense": "sobel", "mode": "per_channel"},
    ]
    for spec in specs:
        out = apply_defense(x, spec)
        assert out.shape == x.shape
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert (out >= 0).all() and (out <= 1).all()
