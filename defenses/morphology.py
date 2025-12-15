import torch
import torch.nn.functional as F

@torch.no_grad()
def _dilate(x, k=3):
    """Dilation via max-pooling; preserves NCHW."""
    pad = k // 2
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)

@torch.no_grad()
def _erode(x, k=3):
    """Erosion via negative dilation."""
    return -_dilate(-x, k=k)

@torch.no_grad()
def opening(x, k=3):
    """Morphological opening: erosion then dilation."""
    out = _dilate(_erode(x, k=k), k=k)
    return out.clamp(0, 1)

@torch.no_grad()
def closing(x, k=3):
    """Morphological closing: dilation then erosion."""
    out = _erode(_dilate(x, k=k), k=k)
    return out.clamp(0, 1)
