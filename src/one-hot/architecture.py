from torch import nn
from torch import Tensor as T


class Model(nn.Module):
    def __init__(self, num_categories: int):
        super().__init__()
        self.a = nn.ReLU()
        self.c1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.gp = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_categories)

    def forward(self, x: T) -> T:
        x = self.p1(self.a(self.bn1(self.c1(x))))
        x = self.p2(self.a(self.bn2(self.c2(x))))
        x = self.a(self.bn3(self.c3(x)))
        x = self.gp(x).flatten(1)
        x = self.fc(x)
        return x
