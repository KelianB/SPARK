from typing import Union

from torch import Tensor


class DotDict(dict):
    # Support dot notation
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# This is the type returned by DECA
class EncodedValues(DotDict):
    expcode: Tensor
    posecode: Tensor
    cam: Tensor
    shapecode: Union[Tensor, None]
    texcode: Union[Tensor, None]
    lightcode: Union[Tensor, None]
    detailcode: Union[Tensor, None]
    detailemocode: Union[Tensor, None]
    eyelids: Union[Tensor, None]
    images: Union[Tensor, None] 
    lmk: Union[Tensor, None]
    lmk_mp: Union[Tensor, None]
    masks: Tensor

class RenderTensors(DotDict):
    grid: Tensor # uv coords (B, H, W, 2) - only used in DECA/EMOCA's detail mode
    alpha_images: Tensor # (B, 1, H, W)
    # Output of DECA's render:
    # 'images': images * alpha_images,
    # 'albedo_images': albedo_images,
    # 'pos_mask': pos_mask,
    # 'shading_images': shading_images,
    # 'normals': normals,
    # 'normal_images': normal_images,
    # 'transformed_normals': transformed_normals,

class DecodedValues(EncodedValues):
    verts: Tensor # (B, V, 3)
    trans_verts: Tensor # (B, V, 3)
    predicted_landmarks: Tensor # (B, 68, 2)
    predicted_landmarks_mediapipe: Tensor # (B, 105, 20)
    predicted_images: Tensor # (B, 3, H, W)
    ops: RenderTensors
    uv_detail_normals: Union[Tensor, None] # only used in DECA/EMOCA's detail mode
