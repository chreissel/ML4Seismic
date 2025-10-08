import os
import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import h5py
import numpy as np
import torch

class GS13SequenceDataset(Dataset):
    """
    Loads preprocessed .npy arrays saved by data_prep.py:
      data/train_{time}.npy, data/val_{time}.npy, data/test_{time}.npy
    Files are shaped (channels, T). We extract channel `target_index` (default=9: GS13X),
    and create causal windows of length `seq_length` to predict the next sample.
    """
    def __init__(self, split: str, time: int, seq_length: int = 120, target_index: int = 9,
                 normalize: bool = False):
        super().__init__()
        assert split in {"train", "val", "test"}
        path = f"data/{split}_{time}.npy"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {path}. Did you run data_prep.py?")
        # (C, T)
        arr = np.load(path)
        # Univariate GS13.x (target channel)
        series = arr[target_index]  # shape (T,)
        
        # Optional normalization (fit only on train normally).
        # For simplicity, apply per-split if normalize=True and no saved stats are present.
        if normalize:
            mu = series.mean()
            sigma = series.std() + 1e-8
            series = (series - mu) / sigma

        self.seq_length = int(seq_length)
        xs, ys = [], []
        # causal windows
        for i in range(0, len(series) - self.seq_length):
            xs.append(series[i:i+self.seq_length])
            ys.append(series[i+self.seq_length])
        X = np.array(xs, dtype=np.float32)         # (N, L)
        y = np.array(ys, dtype=np.float32)         # (N,)
        self.X = torch.from_numpy(X).unsqueeze(-1) # (N, L, 1)
        self.y = torch.from_numpy(y).unsqueeze(-1) # (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class GenericDataModule(L.LightningDataModule):
    def __init__(self,batch_size=32,num_workers=4,pin_memory=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.loader_kwargs = {"batch_size":self.batch_size,
                              "num_workers":self.num_workers,
                              "pin_memory":self.pin_memory}

class LitDataModule(GenericDataModule):
    """
    If time is provided, use GS13SequenceDataset. Otherwise, fall back to sine toy.
    Explicit signature is required for LightningCLI validation.
    """
    def __init__(
        self,
        time: int | None = None,
        seq_length: int = 120,
        target_index: int = 9,
        normalize: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        self.use_gs13 = time is not None
        self.time = time
        self.seq_length = seq_length
        self.target_index = target_index
        self.normalize = normalize

        if self.use_gs13:
            self.train_dataset = GS13SequenceDataset("train", time=self.time, seq_length=self.seq_length,
                                                     target_index=self.target_index, normalize=self.normalize)
            self.val_dataset   = GS13SequenceDataset("val",   time=self.time, seq_length=self.seq_length,
                                                     target_index=self.target_index, normalize=self.normalize)
            self.test_dataset  = GS13SequenceDataset("test",  time=self.time, seq_length=self.seq_length,
                                                     target_index=self.target_index, normalize=self.normalize)
        else:
            print("GS13 data not found. Fallback to sine toy.")
            dataset = TimeSeriesSegmentDataset()
            n_total = len(dataset)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            n_test = n_total - n_train - n_val
            self.train_dataset = Subset(dataset, list(range(0, n_train)))
            self.val_dataset   = Subset(dataset, list(range(n_train, n_train + n_val)))
            self.test_dataset  = Subset(dataset, list(range(n_train + n_val, n_total)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=False, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)