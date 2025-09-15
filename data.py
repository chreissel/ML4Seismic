import lightning as L
from torch.utils.data import random_split, DataLoader, Dataset
import h5py
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
