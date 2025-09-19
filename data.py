import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset, Subset
import h5py
import numpy as np
import torch

class TimeSeriesSegmentDataset(Dataset):

    def __init__(self):

        super().__init__()

        self.time = np.arange(0,400,0.1)
        self.data = np.sin(self.time) + 0.2 * np.random.randn(len(self.time)) # sine wave with noise
        self.seq_length = 30

        xs, ys = [], []
        for i in range(len(self.data) - self.seq_length):
            xs.append(self.data[i:i+self.seq_length])
            ys.append(self.data[i+self.seq_length])
        X = np.array(xs)
        y = np.array(ys)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        """ Return the number of stride """
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dataset = TimeSeriesSegmentDataset()

        n_total = len(self.dataset)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        n_test = n_total - n_train - n_val

        train_indices = list(range(0, n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n_total))

        self.train_dataset = Subset(self.dataset, train_indices)
        self.val_dataset = Subset(self.dataset, val_indices)
        self.test_dataset = Subset(self.dataset, test_indices)


    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=False, **self.loader_kwargs)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader
