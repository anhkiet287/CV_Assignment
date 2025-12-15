
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models as tvm

# REUSE_FROM_NOTEBOOK: your CIFAR model if you have one.
# This is a compact default model that trains quickly.
class SmallCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 4x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class CifarResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        m = tvm.resnet18(weights=None)
        # Adapt for 32x32: smaller kernel/stride and remove initial maxpool
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m

    def forward(self, x):
        return self.model(x)
