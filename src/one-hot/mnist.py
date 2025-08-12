from torch.utils.data import Dataset
from pathlib import Path
import torch
import numpy as np


class MNIST(Dataset):
    def __init__(self, images: Path, labels: Path):
        super().__init__()
        self.images: torch.Tensor = torch.from_numpy(np.load(images))
        self.labels: torch.Tensor = torch.from_numpy(np.load(labels))
        if self.images.size(0) != self.labels.size(0):
            raise RuntimeError(f"Number of images and labels does not match")

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]
