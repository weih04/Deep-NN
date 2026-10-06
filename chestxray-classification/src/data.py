import numpy as np
import torch
from torch.utils.data import TensorDataset


def fit_normalisation(train_images):
    scaled = train_images.astype(np.float32) / 255.0
    return float(scaled.mean()), float(scaled.std())


def make_dataset(images, labels, mean, std):
    scaled = images.astype(np.float32) / 255.0
    normalised = (scaled - mean) / std
    # (N, 64, 64) -> (N, 1, 64, 64): one grey channel.
    inputs = torch.from_numpy(normalised).unsqueeze(1)
    targets = torch.from_numpy(labels.astype(np.float32))
    return TensorDataset(inputs, targets)


def prepare_data(npz_path):
    """Load the pre-split partitions; statistics come from the training set only."""
    arrays = np.load(npz_path)
    mean, std = fit_normalisation(arrays["train_images"])

    return {
        split: make_dataset(
            arrays[f"{split}_images"], arrays[f"{split}_labels"], mean, std
        )
        for split in ("train", "val", "test")
    } | {"mean": mean, "std": std}
