import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning as L
from gwpy.timeseries import TimeSeries
import scipy.io

class SeismicForecastDataset(Dataset):
    
    def __init__(
        self,
        mat_file: str = '',
        train_channels: list[str] = ['L1:ISI-HAM5_BLND_GS13X_IN1_DQ'],
        target_channels: list[str] = ['L1:ISI-HAM5_BLND_GS13X_IN1_DQ'],
        skip_seconds: int = 100,
        sample_rate: int = 128,
        target_rate: int = 4,
        bandpass: list[float] = [0.1,0.3],
        seq_length: int = 240,
        predict_horizon: int = 4,
        normalize: bool = False,
    ):
        super().__init__()
        self.mat_file = mat_file
        self.train_channels = train_channels
        self.target_channels = target_channels
        self.skip_seconds = skip_seconds
        self.sample_rate = sample_rate
        self.target_rate = target_rate
        self.bandpass = bandpass
        self.seq_length = seq_length
        self.predict_horizon = predict_horizon

        self.channels = list(set(self.train_channels+self.target_channels))
        
        raw_data = self.load_mat_file()
        processed = {}
        for i,channel in enumerate(self.channels,1):
            processed[channel] = self.preprocessing(raw_data[channel])
        self.processed = np.stack([processed[ch] for ch in self.channels], axis=0)

        # Normalization (optional)
        if normalize:
            self.mean = self.processed.mean(axis=1, keepdims=True)
            self.std = self.processed.std(axis=1, keepdims=True) + 1e-8

            self.processed = (self.processed - self.mean) / self.std

        self.X, self.y = self._create_sequences()

    def load_mat_file(self):
        file = scipy.io.loadmat(self.mat_file)
        data = file['data_matrix']

        dic = {}
        for idx,c in enumerate(self.channels):
            dic[c] = data[:,idx]
        return dic

    def preprocessing(self, data: np.ndarray):

        # Skip initial seconds for stability
        start_idx = self.skip_seconds * self.sample_rate
        if start_idx >= len(data):
            raise ValueError(f"skip_seconds ({self.skip_seconds}) is too large for data length ({len(data) / self.sample_rate:.2f}s)")
        
        data_stable = data[start_idx:]
        
        # Convert to GWpy TimeSeries
        ts = TimeSeries(
            data_stable, 
            dt=1.0/self.sample_rate, 
            #t0=self.time + self.skip_seconds
        )
        
        # Resample to target rate
        ts_resampled = ts.resample(self.target_rate)
        
        # Apply bandpass filter
        ts_filtered = ts_resampled.bandpass(self.bandpass[0], self.bandpass[1])

        return np.array(ts_filtered.value)

    def _create_sequences(self):

        data = self.processed
        T = data.shape[1]

        X_list = []
        y_list = []

        input_indices = [i for i, n in enumerate(self.channels) if n in self.train_channels] 
        target_indices = [i for i, n in enumerate(self.channels) if n in self.target_channels]

        for i in range(T - self.seq_length - self.predict_horizon + 1):
            x = data[input_indices, i:i+self.seq_length]
            y = data[target_indices, i+self.seq_length:i+self.seq_length+self.predict_horizon]
            X_list.append(x)
            y_list.append(y)

        X = np.array(X_list, dtype=np.float32)  
        y = np.array(y_list, dtype=np.float32) 

        # Transpose X to (N, seq_length, n_channels) for easier processing
        X = np.transpose(X, (0, 2, 1)) 
        y = np.transpose(y, (0, 2, 1)) 

        # Convert to tensors
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LitDataModule(L.LightningDataModule):
    
    def __init__(
        self,
        mat_file: str = '',
        train_channels: list[str] = ['L1:ISI-HAM5_BLND_GS13X_IN1_DQ'],
        target_channels: list[str] = ['L1:ISI-HAM5_BLND_GS13X_IN1_DQ'],
        skip_seconds: int = 100,
        sample_rate: int = 128,
        target_rate: int = 4,
        bandpass: list[float] = [0.1,0.3],
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seq_length: int = 240,
        predict_horizon: int = 4,
        normalize: bool = False,
        batch_size: int = 32,
        num_workers: int = 1
    ):

        super().__init__()

        self.mat_file = mat_file
        self.train_channels = train_channels
        self.target_channels = target_channels
        self.skip_seconds = skip_seconds
        self.sample_rate = sample_rate
        self.target_rate = target_rate
        self.bandpass = bandpass
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seq_length = seq_length
        self.predict_horizon = predict_horizon
        self.normalize = normalize

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = SeismicForecastDataset(self.mat_file, self.train_channels, self.target_channels, self.skip_seconds, self.sample_rate, self.target_rate, self.bandpass, self.seq_length, self.predict_horizon, self.normalize)
        n_samples = len(self.dataset)
        train_size = int(self.train_frac * n_samples)
        val_size = int(self.val_frac * n_samples)

        self.train_dataset = Subset(self.dataset, list(range(train_size))) 
        self.val_dataset = Subset(self.dataset, list(range(train_size, train_size+val_size)))
        self.test_dataset = Subset(self.dataset, list(range(train_size+val_size, n_samples)))

        self.save_hyperparameters()

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
 
    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        return loader

