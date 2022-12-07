from os import path as osp

import h5py
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

