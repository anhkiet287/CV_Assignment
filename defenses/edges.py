import torch
import torch.nn.functional as F

@torch.no_grad()
def sobel_mag(x: torch.Tensor, mode: str = "per_channel") -> torch.Tensor:
    """
    Sobel gradient magnitude. Supports per-channel or luma conversion.
    Outputs are normalized per-image to [0,1] and replicated to 3 channels.
    """
    if mode not in ("per_channel", "luma"):
        raise ValueError("mode must be per_channel or luma")
    if mode == "luma":
        # convert to luminance then add channel dim
        weights = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype)
        x = (x * weights.view(1,3,1,1)).sum(dim=1, keepdim=True)
    c = x.shape[1]
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype)
    kx = kx.view(1,1,3,3).repeat(c,1,1,1)
    ky = ky.view(1,1,3,3).repeat(c,1,1,1)
    pad = 1
    gx = F.conv2d(x, kx, padding=pad, groups=c)
    gy = F.conv2d(x, ky, padding=pad, groups=c)
    mag = torch.sqrt(gx*gx + gy*gy)
    # normalize per-image
    maxv = mag.amax(dim=(1,2,3), keepdim=True).clamp(min=1e-6)
    mag = (mag / maxv).clamp(0,1)
    if mode == "luma":
        mag = mag.repeat(1,3,1,1)
    return mag
