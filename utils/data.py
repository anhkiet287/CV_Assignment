
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

def cifar_loaders(batch_size=128, num_workers=2, val_size=1000, seed=42):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tfm_test = transforms.ToTensor()  # keep eval in [0,1]
    full_train = datasets.CIFAR10(root="data", train=True, download=True, transform=tfm_train)
    test  = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm_test)

    # Deterministic split: 49k train / 1k val
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(full_train), generator=g)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train = Subset(full_train, train_indices)
    val = Subset(full_train, val_indices)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

@torch.no_grad()
def test_tensor(loader):
    xs, ys = [], []
    for x,y in loader:
        xs.append(x); ys.append(y)
    X = torch.cat(xs, 0)  # [N,C,H,W]
    y = torch.cat(ys, 0)  # [N]
    return X, y
