import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import numpy as np
import torch

# everything following is strongly influenced/ copied from the original DeepClean setup as documented here: https://git.ligo.org/tri.nguyen/deepclean-prod/-/tree/master?ref_type=heads
class TimeSeriesSegmentDataset(Dataset):

    def __init__(self, data, kernel, stride, fs, pad_mode='median', target_idx=0):

        super().__init__()

        self.kernel = kernel
        self.stride = stride
        self.fs = fs
        self.pad_mode = pad_mode
        self.data = data
        self.target_idx = target_idx

    def __len__(self):
        """ Return the number of stride """
        nsamp = self.data.shape[-1]
        kernel = int(self.kernel * self.fs)
        stride = int(self.stride * self.fs)
        n_stride = int(np.ceil((nsamp - kernel) / stride) + 1)
        return max(0, n_stride)

    def __getitem__(self, idx):
        """ Get sample Tensor for a given index """
        # check if idx is valid:
        if idx < 0:
            idx +=  self.__len__()
        if idx >= self.__len__():
            raise IndexError(
                f'index {idx} is out of bound with size {self.__len__()}.')

        # get sample
        kernel = int(self.kernel * self.fs)
        stride = int(self.stride * self.fs)
        idx_start = idx * stride
        idx_stop = idx_start + kernel
        data = self.data[:, idx_start: idx_stop].copy()

        # apply padding if needed
        nsamp = data.shape[-1]
        if nsamp < kernel:
            pad = kernel - nsamp
            data = np.pad(data, ((0, 0), (0, pad)), mode=self.pad_mode)

        # separate into target strain and witnesses
        target = data[self.target_idx]
        target = target[:, np.newaxis]
        aux = np.delete(data, self.target_idx, axis=0)

        # convert into Tensor
        target = torch.Tensor(target)
        aux = torch.Tensor(aux)

        return aux, target


# Channel ordering of the preprocessed .npy arrays (see data_prep.py). Index 9
# (the GS13 X-direction channel) is the target; the rest are witnesses.
CHANNELS = [
    'L1:ISI-GND_STS_ITMY_X_DQ', 'L1:ISI-GND_STS_ITMY_Y_DQ', 'L1:ISI-GND_STS_ITMY_Z_DQ',
    'L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Y_IN_DQ', 'L1:ISI-HAM5_SCSUM_CPS_Z_IN_DQ',
    'L1:ISI-HAM5_BLND_CPSRX_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRY_IN1_DQ', 'L1:ISI-HAM5_BLND_CPSRZ_IN1_DQ',
    'L1:ISI-HAM5_BLND_GS13X_IN1_DQ',
]


class WitnessSequenceDataset(Dataset):
    """Sliding-window dataset for witness->target regression.

    Each item is ``(x, y)`` where ``x`` are the witness channels over an input
    window of length ``seq_length`` and ``y`` is the target over the most recent
    ``predict_horizon`` samples of that window (causal now-casting). This replaces
    the previous autoregressive setup, where the target's own past was the input.
    """

    def __init__(self, data, witness_idx, target_idx, seq_length, predict_horizon):
        super().__init__()
        self.data = np.asarray(data, dtype=np.float32)
        self.witness_idx = list(witness_idx)
        self.target_idx = int(target_idx)
        self.seq_length = int(seq_length)
        self.predict_horizon = int(predict_horizon)

    def __len__(self):
        return max(0, self.data.shape[1] - self.seq_length + 1)

    def __getitem__(self, idx):
        if idx < 0:
            idx += len(self)
        if idx >= len(self):
            raise IndexError(f'index {idx} out of bound with size {len(self)}.')
        s, h = self.seq_length, self.predict_horizon
        x = self.data[self.witness_idx, idx:idx + s]            # (Cw, seq)
        y = self.data[self.target_idx, idx + s - h:idx + s]     # (horizon,)
        x = torch.from_numpy(np.ascontiguousarray(x.T))         # (seq, Cw)
        y = torch.from_numpy(np.ascontiguousarray(y[:, None]))  # (horizon, 1)
        return x, y


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
    def __init__(self, time, kernel, stride, fs=4, pad_mode='median', target_idx=9, **kwargs):
        super().__init__(**kwargs)
        
        train_data = np.load('data/train_{}.npy'.format(time))
        val_data = np.load('data/val_{}.npy'.format(time))
        test_data = np.load('data/test_{}.npy'.format(time))
        
        self.train_dataset = TimeSeriesSegmentDataset(train_data, kernel, stride, fs, pad_mode, target_idx)
        self.val_dataset = TimeSeriesSegmentDataset(val_data, kernel, stride, fs, pad_mode, target_idx)
        self.test_dataset = TimeSeriesSegmentDataset(test_data, kernel, stride, fs, pad_mode, target_idx)
        self.save_hyperparameters()

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset,shuffle=False, **self.loader_kwargs)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)
        return loader

    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
        return loader


class LSTMDataModule(GenericDataModule):
    """Witness->target regression data for the LSTM (reads the .npy splits).

    ``witness_channels`` selects the input sensors and ``target_channel`` the
    channel to predict. By default the witnesses are the GND and CPS sensors in
    the *targeted* (X) direction; the non-targeted Y/Z directions are dropped.
    Set ``witness_channels`` to add/remove channels -- keep ``input_size`` of the
    model in sync with ``len(witness_channels)``.
    """

    def __init__(self, time,
                 witness_channels=('L1:ISI-GND_STS_ITMY_X_DQ',
                                   'L1:ISI-HAM5_SCSUM_CPS_X_IN_DQ'),
                 target_channel='L1:ISI-HAM5_BLND_GS13X_IN1_DQ',
                 seq_length=240, predict_horizon=4, normalize=True, **kwargs):
        super().__init__(**kwargs)

        train_data = np.load('data/train_{}.npy'.format(time))
        val_data = np.load('data/val_{}.npy'.format(time))
        test_data = np.load('data/test_{}.npy'.format(time))

        witness_idx = [CHANNELS.index(c) for c in witness_channels]
        target_idx = CHANNELS.index(target_channel)

        if normalize:
            # standardise per channel using the training statistics only
            mean = train_data.mean(axis=1, keepdims=True)
            std = train_data.std(axis=1, keepdims=True) + 1e-8
            train_data = (train_data - mean) / std
            val_data = (val_data - mean) / std
            test_data = (test_data - mean) / std

        self.train_dataset = WitnessSequenceDataset(
            train_data, witness_idx, target_idx, seq_length, predict_horizon)
        self.val_dataset = WitnessSequenceDataset(
            val_data, witness_idx, target_idx, seq_length, predict_horizon)
        self.test_dataset = WitnessSequenceDataset(
            test_data, witness_idx, target_idx, seq_length, predict_horizon)
        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, shuffle=False, **self.loader_kwargs)
