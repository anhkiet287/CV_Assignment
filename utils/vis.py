
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_grid(path: Path, images, nrow=8, title=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs = images.clamp(0,1).cpu().permute(0,2,3,1).numpy()
    h = max(4, int(len(imgs)/nrow)+2)
    fig, axes = plt.subplots(h, nrow, figsize=(nrow, h))
    axes = axes.flatten()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < len(imgs):
            axes[i].imshow(imgs[i])
    if title: fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)

def bar_plot(path: Path, labels, values, title, ylim=(0,1)):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(labels, values)
    ax.set_title(title); ax.set_ylim(*ylim)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    plt.savefig(path, dpi=150); plt.close(fig)
