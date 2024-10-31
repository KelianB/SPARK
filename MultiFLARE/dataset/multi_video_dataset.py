import torch
import numpy as np
import json
from pathlib import Path
import logging
from typing import List

from flare.core import Camera
from utils.dataset import load_img, load_mask, load_semantic_mask, get_K_Rt_from_P, SemanticMask
from utils.color import srgb_to_rgb

class MultiVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_dir: Path, train_dirs: List[str], sample_ratio: int, head_only: bool):
        self.train_dirs = train_dirs
        self.num_seq = len(train_dirs)
        self.base_dir = base_dir
        self.head_only = head_only

        self.landmarks_mp = []
        self.landmarks = []

        per_sequence_intrinsics = []
        self.frame_info = []
        for seq_idx, dir in enumerate(self.train_dirs): 
            json_file = self.base_dir / dir / "flame_params_optimized.json"

            with open(json_file, 'r') as f:
                json_data = json.load(f)
                for item in json_data["frames"]:
                    # keep track of the subfolder
                    item.update({"dir": dir, "seq_idx": seq_idx})
                self.frame_info.extend(json_data["frames"])
                
                per_sequence_intrinsics.append(json_data["intrinsics"])

            # Load MediaPipe landmarks
            landmarks_mp_file: Path = self.base_dir / dir / "landmarks_mp.pt"
            if not landmarks_mp_file.exists():
                raise RuntimeError(f"Cannot find Mediapipe landmarks for sequence '{dir}'")
            landmarks_mp = torch.load(landmarks_mp_file)
            self.landmarks_mp.append(landmarks_mp)

            # Load FAN landmarks
            landmarks_fan_file: Path = self.base_dir / dir / "landmarks_fan.pt"
            if not landmarks_fan_file.exists():
                raise RuntimeError(f"Cannot find FAN landmarks for sequence '{dir}'")
            landmarks = torch.load(landmarks_fan_file)
            # Load Iris landmarks
            iris_file: Path = self.base_dir / dir / "landmarks_iris.pt"
            if iris_file.exists():
                landmarks_iris = torch.load(iris_file)
                landmarks = torch.cat((landmarks, landmarks_iris), dim=1)
            self.landmarks.append(landmarks)

        self.landmarks_mp = torch.cat(self.landmarks_mp, dim=0)
        self.landmarks = torch.cat(self.landmarks, dim=0)

        if sample_ratio > 1:
            self.frame_info = self.frame_info[::sample_ratio]
            self.landmarks_mp = self.landmarks_mp[::sample_ratio]
            self.landmarks = self.landmarks[::sample_ratio]

        self.size = len(self.frame_info)
        logging.info(f"Dataset: {self.size:d} views ({self.num_seq} sequences)")
        test_path = self.base_dir / self.frame_info[0]["dir"] / Path(self.frame_info[0]["file_path"] + ".png")
        self.resolution = load_img(test_path).shape[0:2]

        self.K = torch.eye(3).unsqueeze(0).repeat(self.num_seq, 1, 1)
        for seq_idx, (fx, fy, cx, cy) in enumerate(per_sequence_intrinsics):      
            self.K[seq_idx, 0, 0] = fx * self.resolution[0]
            self.K[seq_idx, 1, 1] = fy * self.resolution[1]
            self.K[seq_idx, 0, 2] = cx * self.resolution[0]
            self.K[seq_idx, 1, 2] = cy * self.resolution[1]

        frames_per_seq = [[] for _ in range(self.num_seq)]
        for idx in range(self.size):
            seq_idx = self.frame_info[idx]["seq_idx"]
            frames_per_seq[seq_idx].append(idx)
        self.frames_per_seq = [torch.tensor(x, dtype=torch.long) for x in frames_per_seq]

        self.shape_params = torch.tensor(json_data["shape_params"]).float().unsqueeze(0)

    def get_flame_pose(self, idx, device):
        json_dict = self.frame_info[idx]
        return torch.tensor(json_dict["pose"], dtype=torch.float32, device=device)

    def get_flame_expression(self, idx, device):
        json_dict = self.frame_info[idx]
        return torch.tensor(json_dict["expression"], dtype=torch.float32, device=device)

    def get_mean_expression(self):
        all_expression = torch.stack([self.get_flame_expression(i, "cpu") for i in range(self.size)])
        return all_expression.mean(dim=0, keepdim=True)

    def __len__(self):
        return self.size

    def __getitem__(self, itr):
        idx = itr % self.size

        json_dict = self.frame_info[idx]
        img_path = self.base_dir / json_dict["dir"] / Path(json_dict["file_path"] + ".png")
        seq_idx = json_dict["seq_idx"]

        # ================ semantics =======================
        semantic_parent = img_path.parent.parent / "semantic"
        semantic_path = semantic_parent / (img_path.stem + ".png")
        semantic = load_semantic_mask(semantic_path)
    
        # ================ img & mask =======================
        img  = load_img(img_path)
        img = srgb_to_rgb(img)
        img_size = img.shape[1]

        mask_parent = img_path.parent.parent / "mask"
        if mask_parent.is_dir():
            mask_path = mask_parent / (img_path.stem + ".png")
            mask = load_mask(mask_path)
        elif img.ndim == 4:
            mask = img[..., 3].unsqueeze(-1)
            mask[mask < 0.5] = 0.0
            img = img[..., :3]
        else:
            raise RuntimeError(f"No mask found for image '{img_path}'")
        
        if self.head_only:
            head_mask = semantic[...,(SemanticMask.ALL_SKIN,SemanticMask.EYES,SemanticMask.EYEBROWS,SemanticMask.MOUTH_INTERIOR,SemanticMask.HAIR)].sum(dim=-1, keepdim=True) >= 1
            mask *= head_mask
        #img = img * mask 
        
        # ================ flame and camera params =======================
        # flame params
        flame_pose = torch.tensor(json_dict["pose"], dtype=torch.float32)
        flame_expression = torch.tensor(json_dict["expression"], dtype=torch.float32)
        
        # camera to world matrix
        world_mat = get_K_Rt_from_P(np.array(json_dict["world_mat"]).astype(np.float32))[1]
        world_mat = torch.tensor(world_mat, dtype=torch.float32)
        # camera matrix to openGL format 
        R = world_mat[:3, :3]
        R[1] *= -1
        R[2] *= -1
        t = world_mat[:3, 3]
        camera = Camera(self.K[seq_idx], R, t)

        landmarks = self.landmarks[idx].float() * 2 / img_size - 1
        landmarks_mp = self.landmarks_mp[idx,:,:2].float() * 2 / img_size - 1

        frame_name = img_path.stem

        # Add batch dimension
        return {
            "img": img[None],
            "mask": mask[None],
            "semantic_mask": semantic[None],
            "flame_pose": flame_pose[None],
            "flame_expression": flame_expression[None],
            "camera": camera,
            "frame_name": frame_name,
            "idx": idx,
            "seq_idx": seq_idx,
            "landmarks": landmarks[None],
            "landmarks_mediapipe": landmarks_mp[None],
        }

    def collate(self, batch):
        return MultiVideoDataset._collate(batch)

    ########## Static methods ##########

    def _collate(batch):
        return {
            "img": torch.cat([item["img"] for item in batch], dim=0),
            "mask": torch.cat([item["mask"] for item in batch], dim=0),
            "semantic_mask": torch.cat([item["semantic_mask"] for item in batch], dim=0),
            "flame_pose": torch.cat([item["flame_pose"] for item in batch], dim=0),
            "flame_expression" : torch.cat([item["flame_expression"] for item in batch], dim=0),
            "camera": [item["camera"] for item in batch],
            "frame_name": [item["frame_name"] for item in batch],
            "idx": torch.LongTensor([item["idx"] for item in batch]),
            "seq_idx": torch.LongTensor([item["seq_idx"] for item in batch]),
            "landmarks" : torch.cat([item["landmarks"] for item in batch], dim=0),
            "landmarks_mediapipe" : torch.cat([item["landmarks_mediapipe"] for item in batch], dim=0),
        }
     
    def override_values_batch(views, per_idx_pose=None, per_idx_expr=None, per_seq_cam_K=None):
        if per_seq_cam_K is not None:
            for i, seq_idx in enumerate(views["seq_idx"]):
                views["camera"][i].K = per_seq_cam_K[seq_idx]
        if per_idx_pose is not None:
            views["flame_pose"] = per_idx_pose[views["idx"]]
        if per_idx_expr is not None:
            views["flame_expression"] = per_idx_expr[views["idx"]]
