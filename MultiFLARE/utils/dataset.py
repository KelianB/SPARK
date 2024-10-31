from typing import List, Iterable

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import imageio
import numpy as np
import cv2


def load_img(filename: str) -> Tensor:
    img = imageio.imread(filename)
    if img.dtype == np.float32:
        img = torch.tensor(img, dtype=torch.float32)
    else:
        img = torch.tensor(img / 255, dtype=torch.float32)
    return img

def load_mask(filename: str) -> Tensor:
    alpha = imageio.imread(filename, mode='F') 
    mask = torch.tensor(alpha / 255., dtype=torch.float32).unsqueeze(-1)
    mask[mask < 0.5] = 0.0
    return mask

# this is not an Enum because we want to be able to use the values as indices directly
class SemanticMask:
    BACKGROUND = 0
    SKIN = 1
    ALL_SKIN = 2
    EYEBROWS = 3
    EYES = 4
    HAIR = 5
    CLOTH_NECKLACE = 6
    EARS = 7
    NOSE = 8
    MOUTH_INTERIOR = 9
    UPPER_LIP = 10
    LOWER_LIP = 11
    NECK = 12
    HAT = 13


def load_semantic_mask(filename: str) -> Tensor:
    img = imageio.imread(filename, mode='F')
    H, W = img.shape
    sem = np.zeros((14, H, W))
    sem[SemanticMask.SKIN] = (img == 1) # skin
    sem[SemanticMask.ALL_SKIN] = ((img == 1) + (img == 10) + (img == 8) + (img == 7) + (img == 14) + (img == 6) + (img == 12) + (img == 13)) >= 1 # skin, nose, ears, neck, lips
    sem[SemanticMask.EYES] = ((img == 4) + (img == 5)) >= 1 # left eye, right eye
    sem[SemanticMask.EYEBROWS] = ((img == 2) + (img == 3)) >= 1 # left eyebrow, right eyebrow
    sem[SemanticMask.MOUTH_INTERIOR] = (img == 11) # mouth interior
    sem[SemanticMask.CLOTH_NECKLACE] = ((img == 15) + (img == 16)) >= 1 # cloth, necklace
    sem[SemanticMask.HAIR] = ((img == 17) + (img == 9)) >= 1 # hair
    sem[SemanticMask.BACKGROUND] = np.clip(1. - np.sum(sem[:8], 0), 0, 1) # background
    sem[SemanticMask.EARS] = ((img == 7) + (img == 8)) >= 1
    sem[SemanticMask.NOSE] = (img == 10) >= 1
    sem[SemanticMask.UPPER_LIP] = (img == 12) >= 1
    sem[SemanticMask.LOWER_LIP] = (img == 13) >= 1
    sem[SemanticMask.NECK] = (img == 14) >= 1
    sem[SemanticMask.HAT] = (img == 18) >= 1

    # Original BiseNet mapping:
    # skin: 1, l_brow: 2, r_brow: 3, l_eye: 4, r_eye: 5, eye_g: 6, l_ear: 7, r_ear: 8, ear_r: 9
    # nose: 10, mouth: 11, u_lip: 12, l_lip: 13, neck: 14, neck_l: 15, cloth: 16, hair: 17, hat: 18

    sem = torch.tensor(sem, dtype=torch.bool).permute(1,2,0) # CHW to HWC
    return sem

def get_K_Rt_from_P(P):
    K, R, t, *_ = cv2.decomposeProjectionMatrix(P)
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K/K[2,2]
    pose = np.eye(4, dtype=np.float32)
    pose[:3,:3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]
    return intrinsics, pose

#----------------------------------------------------------------------------
# Generic utilities for torch data handling
#----------------------------------------------------------------------------

def to_device_recursive(v, device):
    """ Move a whole structure of tensors (dict, list, tuple) to the given device. """
    if torch.is_tensor(v):
        return v.to(device)
    elif isinstance(v, dict):
        return {k: to_device_recursive(x, device) for k,x in v.items()}
    elif isinstance(v, list):
        return [to_device_recursive(x, device) for x in v]
    elif isinstance(v, tuple):
        return tuple(to_device_recursive(x, device) for x in v)
    elif callable(getattr(v, "to", None)):
        return v.to(device)
    return v

def find_collate(d: Dataset):
    """ Find a 'dataset.collate' method from nested datasets. """
    if hasattr(d, "collate"):
        return d.collate
    elif hasattr(d, "dataset"):
        return find_collate(d.dataset)
    else:
        raise RuntimeError("No collate fn found")

class DeviceDataLoader(DataLoader):
    """
    Wrapper of PyTorch's `DataLoader` class for automatically sending data to device.

    Args:
        dataset (`torch.utils.data.Dataset`):
            The dataset to use to build this dataloader.
        device (`torch.Device`, defaults to cpu):
            A device to send tensors to.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, dataset: Dataset, device='cpu', **kwargs):
        super().__init__(dataset, **kwargs)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            yield to_device_recursive(batch, self.device)

class DatasetCache(Dataset):
    """
    Wrapper of PyTorch's `Dataset` class for caching samples.

    Args:
        dataset (`torch.utils.data.Dataset`):
            The child dataset to cache.
        kwargs:
            Keyword arguments to pass to the regular `Dataset` constructor.
    """

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.cache = dict()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        if idx not in self.cache:
            self.cache[idx] = self.dataset[idx]
        return self.cache[idx]
       
    def __len__(self):
        return len(self.dataset)

    def release_cache(self, indices: Iterable[int] = None):
        if indices is None:
            self.cache = dict()
        else:
            for i in indices:
                if int(i) in self.cache:
                    del self.cache[int(i)]

    # Fallback: defer all other method calls to underlying dataset    
    def __getattr__(self, *args):
        return self.dataset.__getattribute__(*args)

class ZipDatasets(Dataset):
    """
    Wrapper of PyTorch's `Dataset` class that acts like the `zip` function.

    Args:
        datasets (`torch.utils.data.Dataset`):
            The datasets to zip.
        kwargs:
            Keyword arguments to pass to the regular `Dataset` constructor.
    """
    def __init__(self, *datasets, **kwargs):
        super().__init__(**kwargs)
        self.datasets = datasets
        self.collate_fns = [find_collate(d) for d in datasets]

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
    
    def collate(self, batch: tuple):
        assert type(batch) in [tuple, list]
        return tuple(collate_fn([tup[i] for tup in batch]) for i, collate_fn in enumerate(self.collate_fns))
