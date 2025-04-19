import cv2
import h5py
import torch
import numpy as np
import hdf5storage
import scipy.io as sio
import os

def get_edge(data):
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs


def load_mat(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    ms = torch.from_numpy(data['ms'] / 2047.0)  # HxWxC = 256x256x8
    ms = ms.permute(2, 0, 1)
    pan = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256

    return ms, pan


def load_h5py(file_path):
    data = h5py.File(file_path)

    ms = data["ms"][...]  # W H C N
    ms = np.array(ms, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)

    lms = data["lms"][...]  # W H C N
    lms = np.array(lms, dtype=np.float32) / 2047.
    lms = torch.from_numpy(lms)

    pan = data["pan"][...]  # W H C N
    pan = np.array(pan, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)

    return ms, lms, pan


def load_h5py_hp(file_path):
    data = h5py.File(file_path)

    ms1 = data["ms"][...]
    ms = np.array(ms1, dtype=np.float32) / 2047.
    ms = torch.from_numpy(ms)

    ms_hp = np.array(ms1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWxC
    ms_hp = get_edge(ms_hp)
    ms_hp = torch.from_numpy(ms_hp).permute(0, 3, 1, 2)

    pan1 = data["pan"][...]  # W H C N
    pan = np.array(pan1, dtype=np.float32) / 2047.
    pan = torch.from_numpy(pan)

    pan_hp = np.array(pan1.transpose(0, 2, 3, 1), dtype=np.float32) / 2047.  # NxHxWx1
    pan_hp = np.squeeze(pan_hp, axis=3)  # NxHxW
    pan_hp = get_edge(pan_hp)  # NxHxW
    pan_hp = np.expand_dims(pan_hp, axis=3)  # NxHxWx1
    pan_hp = torch.from_numpy(pan_hp).permute(0, 3, 1, 2)

    return ms, ms_hp, pan, pan_hp 

def init_test(config):
    device = torch.device(config["device"])

    if config["model_type"] == "BWNET_LAGConv":
        from model.bwnet_lagconv_sota import BWNet
    elif config["model_type"] == "BWNET_DICNN":
        from model.bwnet_dicnn import BWNet
    elif config["model_type"] == "BWNET_FusionNet":
        from model.bwnet_fusionnet import BWNet
    elif config["model_type"] == "BWNET_LGPNet":
        from model.bwnet_lgpnet import BWNet

    if config["dataset_type"] == "wv3":
        model = BWNet(ms_dim=8).to(device).eval()
    elif config["dataset_type"] in ["gf2", "qb"]:
        model = BWNet(ms_dim=4).to(device).eval()

    # Load model weights
    weight = torch.load(config["ckpt"], map_location=device)
    model.load_state_dict(weight)

    return model, device

def test(config):
    model, device = init_test(config)

    if config["dataset_scale"] == "Reduced":
        file_path = os.path.join(config["data_path"], f'test_{config["dataset_type"]}_multiExm1.h5')
    else:
        file_path = os.path.join(config["data_path"], f'test_{config["dataset_type"]}_OrigScale_multiExm1.h5')

    save_path = os.path.join(config["save_path"], f"{config['dataset_type'].upper()}_{config['dataset_scale']}", config["model_type"], 'results')
    os.makedirs(save_path, exist_ok=True)

    ms, _, pan = load_h5py(file_path)
    B, C, _, _ = ms.shape

    for k in range(B):
        with torch.no_grad():
            MS, PAN = ms[k, :, :, :], pan[k, 0, :, :]
            PAN = PAN.cuda().unsqueeze(0).unsqueeze(0).float()
            MS = MS.cuda().unsqueeze(0).float()

            output = model(MS, PAN)
            output = torch.clamp(output, 0, 1)

            output = torch.squeeze(output).permute(1, 2, 0).cpu().detach().numpy() * 2047.  # HxWxC
            save_name = os.path.join(save_path, "output_mulExm_" + str(k) + ".mat")
            sio.savemat(save_name, {'sr': output})

if __name__ == "__main__":
    config = {
        "ckpt": "",
        "data_path": "",
        "save_path": "",
        "dataset_type": "wv3",  # Type of dataset: 'wv3', 'gf2', or 'qb'
        "dataset_scale": "Reduced",  # Scale of dataset: 'Reduced' or 'Full'
        "model_type": "",  # Model type: 'BWNET_LAGConv' or 'BWNET_DICNN' etc.
        "device": "cuda",
    }
    test(config)
