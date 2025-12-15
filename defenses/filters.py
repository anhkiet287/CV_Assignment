
import torch, torch.nn.functional as F

def _gaussian_kernel(k, sigma, device, channels):
    ax = torch.arange(k, device=device) - (k-1)/2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    ker = torch.exp(-(xx**2+yy**2)/(2*sigma**2))
    ker = ker / ker.sum()
    ker = ker.view(1,1,k,k).repeat(channels,1,1,1)
    return ker

@torch.no_grad()
def gaussian(x, k=3, sigma=1.0):
    """Gaussian blur per-channel; clamps output to [0,1]."""
    c = x.shape[1]
    ker = _gaussian_kernel(k, sigma, x.device, c)
    pad = k//2
    out = F.conv2d(x, ker, stride=1, padding=pad, groups=c)
    return out.clamp(0,1)

@torch.no_grad()
def median(x, k=3):
    """Median filter per-channel using unfolding; clamps to [0,1]."""
    pad = k//2
    x_pad = F.pad(x, (pad,pad,pad,pad), mode="reflect")
    n,c,h,w = x.shape
    patches = F.unfold(x_pad, kernel_size=k)  # [n, c*k*k, h*w]
    patches = patches.view(n, c, k*k, h*w)
    med = patches.median(dim=2).values
    med = med.view(n, c, h, w)
    return med.clamp(0,1)
