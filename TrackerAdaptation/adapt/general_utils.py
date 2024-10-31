import os
from contextlib import contextmanager

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T
import cv2


def train_val_split(dataset: Dataset, val_ratio: float):
    dataset_size = len(dataset)
    n_val = int(dataset_size * val_ratio)

    interval = int(1 / val_ratio)
    indices = range(dataset_size)
    indices_val = list(indices[::interval])
    indices_val = indices_val[:n_val] # avoid rounding errors
    indices_train = [v for i,v in enumerate(indices) if i % interval != 0 or v > indices_val[-1]]
    
    assert len(indices_val) == n_val
    assert len(indices_train) + len(indices_val) == dataset_size
    return Subset(dataset, indices_train), Subset(dataset, indices_val)

def draw_crop_rectangles(imgs, boxes):
    imgs = imgs * 255.0 # scale to [0,255]
    perm = imgs.shape[1] <= 4
    if perm: imgs = imgs.permute(0,2,3,1) # BCHW to BHWC

    device = imgs.device
    for i,  (img, bbox) in enumerate(zip(imgs, boxes)):
        x, y, s = (int(v) for v in bbox)
        imgs[i] = torch.from_numpy(cv2.rectangle(img.cpu().contiguous().numpy(), (x, y), (x+s, y+s), (0,255,0), 1)).to(device)

    if perm: imgs = imgs.permute(0,3,1,2) # BHWC to BCHW
    imgs = imgs / 255.0 # scale back to [0,1]
    return imgs

def get_optical_flow_model(device):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    model_raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model_raft = model_raft.eval()
    preprocess_img_raft = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=0.5, std=0.5), # map [0, 1] to [-1, 1]
        #T.Resize(size=(scaled_h, scaled_w)),
    ])
    return model_raft, preprocess_img_raft

def compute_optical_flows(img1, img2, model, preprocess):
    if img1.shape[1] <= 4:
        img1 = img1.permute(0,2,3,1) # BCHW to BHWC
    if img2.shape[1] <= 4:
        img2 = img2.permute(0,2,3,1) # BCHW to BHWC
    img1 = preprocess(img1.permute((0,3,1,2)))
    img2 = preprocess(img2.permute((0,3,1,2)))
    predicted_flows = model(img1, img2)[-1]
    # return predicted_flows.permute((0,2,3,1)) # B2HW to BHW2
    return predicted_flows # B2HW

def gaussian_kernel(ksize: int, sigma: float):
    x = torch.arange(0, ksize) - ksize // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    return gauss / gauss.sum()

def apply_featurewise_conv1d(signal: torch.Tensor, kernel: torch.Tensor, pad_mode="replicate") -> torch.Tensor:
    # signal: (N, n_features)
    # kernel: (kernel_size)
    _, n_features = signal.shape
    kernel_size = kernel.shape[0]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(n_features,1,1) # (n_features,1,kernel_size) (we want as many output channels as features)
    signal = signal.permute(1,0).unsqueeze(0) # (1, n_features, N)
    # Pad input signal
    padding = kernel_size // 2 # to maintain the original signal length
    padded_signal = torch.nn.functional.pad(signal, (padding, padding), mode=pad_mode)
    # Perform convolution
    filtered_signal = torch.nn.functional.conv1d(padded_signal, kernel, groups=n_features) # (1, n_features, N)
    return filtered_signal.squeeze(0).permute(1,0) # (N, n_features)

@contextmanager
def working_dir(dir: str):
    """ Temporarily change the working directory, then restore it. """
    workdir_before = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(workdir_before)
