import torch
import torch.nn as nn
import torch.nn.functional as F
from dlx.registry import register

@register("model", "cnn_small")
class CNNSmall(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # 32x32 input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(128 * 4 * 4, num_classes)  # after 3 pools: 32->16->8->4

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.head(x)