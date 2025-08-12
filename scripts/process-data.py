import struct
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


def read_idx_images(path: Path) -> NDArray[np.uint8]:
    with open(path, "rb") as f:
        (magic,) = struct.unpack(">I", f.read(4))
        if magic != 2051:
            raise ValueError("Not an IDX image file")
        n, rows, cols = struct.unpack(">III", f.read(12))
        data = f.read(n * rows * cols)
    return np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)


def read_idx_labels(path: Path) -> NDArray[np.uint8]:
    with open(path, "rb") as f:
        (magic,) = struct.unpack(">I", f.read(4))
        if magic != 2049:
            raise ValueError("Not an IDX label file")
        (n,) = struct.unpack(">I", f.read(4))
        data = f.read(n)
    return np.frombuffer(data, dtype=np.uint8)


def normalize_images(images: np.ndarray) -> NDArray[np.float64]:
    return images.astype(np.float64) / 255.0


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> NDArray[np.float64]:
    out = np.zeros((labels.size, num_classes), dtype=np.float64)
    out[np.arange(labels.size), labels] = 1.0
    return out


test_images = read_idx_images(Path("data/unrefined/t10k-images.idx3-ubyte"))
train_images = read_idx_images(Path("data/unrefined/train-images.idx3-ubyte"))
test_labels = read_idx_labels(Path("data/unrefined/t10k-labels.idx1-ubyte"))
train_labels = read_idx_labels(Path("data/unrefined/train-labels.idx1-ubyte"))

all_images = np.concatenate([train_images, test_images], axis=0)
all_labels = np.concatenate([train_labels, test_labels], axis=0)

imgs = normalize_images(all_images)
labels = all_labels
labels_one_hot = one_hot_encode(all_labels, num_classes=10)

out_dir = Path("data/refined")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"images_norm: {imgs.shape}")
print(f"labels_one_hot: {labels_one_hot.shape}")
print(f"labels: {labels.shape}")

np.save(out_dir / "images_norm.npy", imgs, allow_pickle=True)
np.save(out_dir / "labels_one_hot.npy", labels_one_hot, allow_pickle=True)
np.save(out_dir / "labels.npy", labels, allow_pickle=True)
