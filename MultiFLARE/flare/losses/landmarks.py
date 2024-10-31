from typing import List
import torch

from flame.FLAME import FLAME, MEDIAPIPE_LMK_EMBEDDING_INDICES
from flare.core import Renderer, Camera

# Indices of the interior FAN landmarks
LANDMARKS_FAN_INTERIOR = [*range(27, 36), *range(36, 48), *range(48, 68), *range(68, 70)] # nose + eyes + lips + iris

def landmarks_L1_loss(flame: FLAME, landmarks_detections: torch.Tensor,
                      cams: List[Camera], renderer: Renderer, deformed_vertices: torch.Tensor,
                      pose: torch.Tensor, lmk_idx=None, per_frame=False):
    if lmk_idx is None:
        lmk_idx = torch.arange(0, 70, dtype=torch.long, device=deformed_vertices.device)

    lmk_pos = flame.get_landmark_positions_2d(deformed_vertices, pose)[:,lmk_idx] # (B, 70, 3)
    lmk_screen = renderer.to_screen_space(lmk_pos, cams) / (renderer.resolution[0]/2) # (B, 70, 2)
    if per_frame:
        return (lmk_screen - landmarks_detections[:,lmk_idx]).abs().sum(-1).mean(1)
    else:
        return (lmk_screen - landmarks_detections[:,lmk_idx]).abs().sum(-1).mean()

def iris_L1_loss(flame: FLAME, landmarks_detections: torch.Tensor,
                 cams: List[Camera], renderer: Renderer, deformed_vertices: torch.Tensor,
                 pose: torch.Tensor):
    return landmarks_L1_loss(flame, landmarks_detections, cams, renderer, deformed_vertices, pose, lmk_idx=[68,69])

def landmarks_L1_loss_mediapipe(flame: FLAME, landmarks_detections: torch.Tensor,
                      cams: List[Camera], renderer: Renderer, deformed_vertices: torch.Tensor,
                      per_frame=False):
    lmk_pos = flame.get_landmark_positions(deformed_vertices, which="static_mediapipe") # (B, L, 3)
    lmk_screen = renderer.to_screen_space(lmk_pos, cams) / (renderer.resolution[0]/2) # (B, L, 2)
    if per_frame:
        return (lmk_screen - landmarks_detections[:,MEDIAPIPE_LMK_EMBEDDING_INDICES]).abs().sum(-1).mean(1)
    else:
        return (lmk_screen - landmarks_detections[:,MEDIAPIPE_LMK_EMBEDDING_INDICES]).abs().sum(-1).mean()

