from os import path as osp

import h5py
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class HDF5_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def __len__(self):
        if not hasattr(self, "len"):
            data = h5py.File(self.data_path, "r")
            self.len = data["/x"].shape[0]
            data.close()
            return self.len
        else:
            return self.len

    def __getitem__(self, idx):
        data = h5py.File(osp.join(self.data_path, self.data_name), "r")
        x = data["/x"][idx]
        y = data["/y"][idx]
        data.close()
        return x, y
