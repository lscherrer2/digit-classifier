from torch import Tensor as T
import torch.nn.functional as F
from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
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
        self.fc = nn.Linear(64, embedding_dim)

    def forward(self, x: T) -> T:
        x = self.p1(self.a(self.bn1(self.c1(x))))
        x = self.p2(self.a(self.bn2(self.c2(x))))
        x = self.a(self.bn3(self.c3(x)))
        x = self.gp(x).flatten(1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class Embeddings(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.prototypes = nn.Embedding(num_classes, embedding_dim)
        with torch.no_grad():
            self.prototypes.weight.uniform_(-0.1, 0.1)
            self.prototypes.weight[:] = F.normalize(self.prototypes.weight, p=2, dim=1)

    def forward(self) -> T:
        return F.normalize(self.prototypes.weight, p=2.0, dim=1)

    def get(self, idx: torch.LongTensor) -> T:
        return F.normalize(self.prototypes(idx), p=2.0, dim=1)

    def similarity(self, x: T) -> T:
        protos = self.forward()
        return x @ protos.t()
