import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np


def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, dataset_type):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path)

        gt1 = data["gt"][...]
        if dataset_type in ["wv3", "qb"]:
            gt1 = np.array(gt1, dtype=np.float32) / 2047. 
        elif dataset_type in ["gf2"]:
            gt1 = np.array(gt1, dtype=np.float32) / 1023.
        self.gt = torch.from_numpy(gt1)

        lms1 = data["lms"][...]
        if dataset_type in ["wv3", "qb"]:
            lms1 = np.array(lms1, dtype=np.float32) / 2047.
        elif dataset_type in ["gf2"]:
            lms1 = np.array(lms1, dtype=np.float32) / 1023.
        self.lms = torch.from_numpy(lms1)

        ms1 = data["ms"][...]
        if dataset_type in ["wv3", "qb"]:
            ms1 = np.array(ms1, dtype=np.float32) / 2047.
        elif dataset_type in ["gf2"]:
            ms1 = np.array(ms1, dtype=np.float32) / 1023.
        self.ms = torch.from_numpy(ms1)

        if dataset_type in ["wv3", "qb"]:
            ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.
        elif dataset_type in ["gf2"]:
            ms1 = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 1023.
        ms1_tmp = get_edge(ms1)
        self.ms_hp = torch.from_numpy(ms1_tmp).permute(0, 3, 1, 2)

        pan1 = data['pan'][...]
        if dataset_type in ["wv3", "qb"]:
            pan = np.array(pan1, dtype=np.float32) / 2047.
        elif dataset_type in ["gf2"]:
            pan = np.array(pan1, dtype=np.float32) / 1023.
        self.pan = torch.from_numpy(pan)

        if dataset_type in ["wv3", "qb"]:
            pan_hp = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.
        elif dataset_type in ["gf2"]:
            pan_hp = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 1023.
        pan_hp = np.squeeze(pan_hp, axis=3)  # NxHxW
        pan_hp = get_edge(pan_hp)  # NxHxW
        pan_hp = np.expand_dims(pan_hp, axis=3)  # NxHxWx1
        self.pan_hp = torch.from_numpy(pan_hp).permute(0, 3, 1, 2)

    def __getitem__(self, index):
        return self.gt[index, :, :, :].float(), \
               self.lms[index, :, :, :].float(), \
               self.ms_hp[index, :, :, :].float(), \
               self.pan[index, :, :, :].float(), \
               self.ms[index, :, :, :].float(), \
               self.pan_hp[index, :, :, :].float()

    def __len__(self):
        return self.gt.shape[0]

