import logging
from typing import List

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

# MultiFLARE imports
from utils.dataset import DeviceDataLoader, DatasetCache, find_collate
from utils.color import rgb_to_srgb

from adapt.general_utils import gaussian_kernel, apply_featurewise_conv1d
from adapt.deca_helpers import get_deca_gt_mask
from adapt.constants import DATASET_RESOLUTION

class CropDataset(Dataset):
    def __init__(self, dataset, target_resolution, prune_original_views=False):
        self.dataset = dataset
        self.dataset_collate_fn = find_collate(dataset)
        self.target_resolution = target_resolution
        self.transforms = transforms.Compose([transforms.Resize((target_resolution, target_resolution))])
        self.crops_all = None
        self.prune_original_views = prune_original_views
        if prune_original_views:
            S = DATASET_RESOLUTION
            self.placeholders = {
                "mask": torch.zeros((1, S, S, 1), dtype=torch.bool),
                "semantic_mask": torch.zeros((1, S, S, 20), dtype=torch.bool),
                "flame_pose": torch.zeros((1, 18), dtype=torch.float),
                "flame_expression": torch.zeros((1, 50), dtype=torch.float),
                "landmarks": torch.zeros((1, 70, 2), dtype=torch.float),
                "landmarks_mediapipe": torch.zeros((1, 478, 2), dtype=torch.float),
            }

    def _compute_crops(self, views) -> List[torch.Tensor]:
        crop_scale = 1.35

        img = views["img"]
        B, H, W, _ = img.shape
        boxes = torch.zeros((B, 3), dtype=torch.int, device=img.device)

        for i, lmk in enumerate(views["landmarks"]):
            lmk_x, lmk_y = lmk.unbind(1) # (L,2)
            lmk_x = (lmk_x+1) * (W/2)
            lmk_y = (lmk_y+1) * (H/2)
            # Calculate a box based on the landmarks
            left, right = lmk_x.min(), lmk_x.max()
            top, bottom = lmk_y.min(), lmk_y.max()
            # Make the box a square and scale it up
            s = (right - left + bottom - top) / 2
            center_x = right - (right - left) / 2.0
            center_y =  bottom - (bottom - top) / 2.0
            s *= crop_scale
            x, y = center_x - s/2, center_y - s/2
            # Ensure the box is fully within the image
            boxes[i, 0] = x.round().int().clamp(min=0, max=W-s)
            boxes[i, 1] = y.round().int().clamp(min=0, max=H-s)
            boxes[i, 2] = s.round().int().clamp(min=0, max=min(H, W))

        return boxes

    def preload_and_smooth_crops(self, device, batch_size):
        # We need to load all the frames to apply a low-pass kernel to the crop boxes.
        # Ideally we don't want to have to load the original frames a second time later.
        # To achieve this, we:
        # 1. cache the original dataset
        # 2. compute all crop boxes, smooth and save them
        # 3. while we have the original dataset cached, preload the the cropped one
        # 4. release the original dataset cache

        # 1.
        self.dataset = DatasetCache(self.dataset)

        # 2.
        dataloader = DeviceDataLoader(self.dataset, device=device, batch_size=batch_size, collate_fn=self.dataset_collate_fn, num_workers=0)
        logging.info("Preloading dataset to compute crops")
        crops_all = torch.tensor([], dtype=torch.float, device=device)
        sequence_starts = [0] # dataset index at which each sequence starts
        current_sequence = 0
        for views in tqdm(dataloader):
            boxes = self._compute_crops(views)
            crops_all = torch.cat((crops_all, boxes), dim=0)
            for k, idx in enumerate(views["idx"]):
                if views["seq_idx"][k] != current_sequence:
                    sequence_starts.append(idx.item())
                    current_sequence += 1

        logging.info("Applying low-pass filters to crops")        
        conv_weights = gaussian_kernel(ksize=15, sigma=4).to(device)

        # boxes_all stores the crops ordered by frame since we didn't shuffle
        # we need to apply the filtering individually per sequence
        for sidx in range(len(sequence_starts)):
            first_i = sequence_starts[sidx]
            last_i = sequence_starts[sidx+1] if sidx < len(sequence_starts)-1 else len(crops_all)
            sequence_crops = crops_all[first_i:last_i]
            # Apply a gaussian filter to the box coordinates and dimensions
            sequence_crops = apply_featurewise_conv1d(sequence_crops, conv_weights, pad_mode="replicate")
            sequence_crops[..., -1] = sequence_crops[0, -1] # force constant size
            crops_all[first_i:last_i] = sequence_crops

        self.crops_all = crops_all.round().int().cpu()

        # 3. & 4.
        logging.info("Pre-loading final crops & releasing non-cropped dataset from cache")        
        self_cached = DatasetCache(self)
        dataloader = DeviceDataLoader(self_cached, device=device, batch_size=batch_size, collate_fn=self.collate, num_workers=0)
        for views in tqdm(dataloader):
            # Simply iterating over the dataset will load it in cache.
            # We still need to release the original frames from cache
            self_cached.release_cache(views["idx"])

        # Remove the cache on the original dataset
        self.dataset = self.dataset.dataset

        return self_cached

    def __getitem__(self, i):
        # Get the view and collate it just so we're working with the usual formats (with a batch of 1)
        it = self.dataset[i]
        views = self.dataset_collate_fn([it])

        resolution_initial = views["img"].shape[1]

        if self.crops_all is None:
            box = self._compute_crops(views)[0]
        else:
            box = self.crops_all[views["idx"][0].item()]
        box_x, box_y, box_s = box
        crop = views["img"][0, box_y:box_y+box_s, box_x:box_x+box_s]

        # Transform
        crop = rgb_to_srgb(crop)
        crop = self.transforms(crop.permute(2,0,1)) # HW3 to 3HW

        # Also crop and transform the semantic mask
        semantic_mask = views["semantic_mask"][0, box_y:box_y+box_s, box_x:box_x+box_s, :]
        semantic_mask = self.transforms(semantic_mask.permute(2,0,1)).permute(1,2,0) # HWC to CHW to HWC
        mask = get_deca_gt_mask(semantic_mask)

        # De-normalize landmarks
        lmk = (views["landmarks"][0] + 1) * resolution_initial / 2
        # Apply crop to landmarks        
        lmk[:, 0] -= box_x
        lmk[:, 1] -= box_y
        # Apply resize to landmarks (from s to resolution_inp)
        lmk *= self.target_resolution / box_s
        # Normalize landmarks
        lmk = lmk / (self.target_resolution/2) - 1

        if "landmarks_mediapipe" in views:
            lmk_mp = (views["landmarks_mediapipe"][0] + 1) * resolution_initial / 2
            lmk_mp[:, 0] -= box_x
            lmk_mp[:, 1] -= box_y
            lmk_mp *= self.target_resolution / box_s
            lmk_mp = lmk_mp / (self.target_resolution/2) - 1

        if self.prune_original_views:
            # Remove some elements from the original view (or rather, point them to a common value) to save on cache space.
            # These are items that won't be necessary once the cropped views have been computed           
            it = dict((key, self.placeholders[key] if key in self.placeholders else value) for key, value in it.items())

        return {
            "original": it,
            "crop": {
                "image": crop,
                "semantic_mask": semantic_mask,
                "mask": mask,
                "landmark": lmk[:68], # discard iris landmarks
                **({"landmark_mediapipe": lmk_mp} if "landmarks_mediapipe" in views else {}),
                "bbox": box,
            }
        }
    
    def __len__(self):
        return len(self.dataset)
    
    def collate(self, batch):
        batch_original = [item["original"] for item in batch]
        batch_crop = [item["crop"] for item in batch]
        return {
            **self.dataset_collate_fn(batch_original),
            "crop": {
                "image": torch.stack([item["image"] for item in batch_crop]),
                "semantic_mask": torch.stack([item["semantic_mask"] for item in batch_crop]),
                "mask": torch.stack([item["mask"] for item in batch_crop]),
                "landmark": torch.stack([item["landmark"] for item in batch_crop]),
                **({"landmark_mediapipe": torch.stack([item["landmark_mediapipe"] for item in batch_crop])} if "landmark_mediapipe" in batch_crop[0] else {}),
                "bbox": torch.stack([item["bbox"] for item in batch_crop]),
            }
        }

    # Fallback: defer all other method calls to underlying dataset    
    def __getattr__(self, *args):
        return self.dataset.__getattribute__(*args)
