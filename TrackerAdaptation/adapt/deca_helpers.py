from contextlib import contextmanager
from pathlib import Path
from glob import glob
from typing import Dict

import torch
from omegaconf import OmegaConf

# MultiFLARE imports
from utils.dataset import SemanticMask
from utils.color import rgb_to_srgb

from adapt.common_types import EncodedValues
from adapt.constants import EMOCA_PATH

def make_outputs_full_sized(encoded_values: EncodedValues, views: Dict):
    """ Convert encoded values for rendering using the code of DECA at the size of the crops of MultiFLARE. """
    RES = views["img"].shape[1]
    HRES = RES / 2

    images = rgb_to_srgb(views["img"]).permute(0,3,1,2).contiguous() # BHWC to BCHW
    lmk = views["landmarks"][:,:68]
    lmk_mp = views["landmarks_mediapipe"]
    bbox = views["crop"]["bbox"]

    # We need to account for the training constraints in DECA which can double the batch size artificially during encoding
    if views["img"].shape[0] != encoded_values["images"].shape[0]:
        images = torch.cat([images, images], dim=0)
        lmk = torch.cat([lmk, lmk], dim=0)
        lmk_mp = torch.cat([lmk_mp, lmk_mp], dim=0)
        bbox = torch.cat([bbox, bbox], dim=0)
        if views["img"].shape[0] != encoded_values["images"].shape[0]:
            raise ValueError("Unexpected images tensor shape in encoded values")

    encoded_values["lmk"] = lmk
    encoded_values["lmk_mp"] = lmk_mp
    encoded_values["images"] = images # this breaks the emonet loss
    encoded_values["masks"] = torch.stack([get_deca_gt_mask(m) for m in views["semantic_mask"]])

    box_x, box_y, box_size = bbox.unbind(-1)
    s, tx, ty = encoded_values["cam"].unbind(-1)
    s = s * box_size / RES
    tx = tx + ((box_x + box_size/2 - HRES) / HRES) / s
    ty = ty - ((box_y + box_size/2 - HRES) / HRES) / s
    encoded_values["cam"] = torch.stack([s, tx, ty], dim=-1)

    # Update the focal length
    for cam, box_s in zip(views["camera"], box_size):
        cam.K[0,0] *= RES / box_s # fx
        cam.K[1,1] *= RES / box_s # fy

def get_deca_gt_mask(semantic_mask: torch.Tensor):
    """Get the mask used in DECA for segMode 'gt'."""
    s = semantic_mask
    mask = (s[..., SemanticMask.BACKGROUND] + s[..., SemanticMask.EARS] + s[..., SemanticMask.HAIR] + \
            s[..., SemanticMask.HAT] + s[..., SemanticMask.NECK] + s[..., SemanticMask.CLOTH_NECKLACE]) == 0
    return mask.float() # (H,W)

@contextmanager
def deca_set_render_faces(deca, faces, uv_coords=None):
    """ Temporarily change the faces tensor used by DECA's renderer, then restore it. """
    if faces.ndim == 2:
        faces = faces.unsqueeze(0)
    assert faces.ndim == 3 # (1, F, 3)
    n_faces = faces.shape[1]

    try:
        copy_faces = deca.deca.render.faces.clone()
        copy_uvfaces = deca.deca.render.uvfaces.clone()
        copy_face_uvcoords = deca.deca.render.face_uvcoords.clone()
        copy_face_colors = deca.deca.render.face_colors.clone() # (1, 9976, 3, 3)
        deca.deca.render.faces = faces
        deca.deca.render.face_colors = deca.deca.render.face_colors[:,(0,)].repeat(1, n_faces, 1, 1)

        if uv_coords is None:
            if n_faces < copy_faces.shape[1]:
                deca.deca.render.face_uvcoords = copy_face_uvcoords[:, :n_faces]
                deca.deca.render.uvfaces = copy_uvfaces[:, :n_faces]
            elif n_faces > copy_faces.shape[1]:
                new_uvcoords = torch.zeros((1, n_faces - copy_faces.shape[1], 3, 3), dtype=torch.float, device=faces.device)
                deca.deca.render.face_uvcoords = torch.cat((copy_face_uvcoords, new_uvcoords), dim=1)
                # new_uvfaces = torch.zeros((1,n_faces - copy_faces.shape[1], 3), dtype=torch.float, device=faces.device)
                # deca.deca.render.uvfaces = torch.cat((copy_uvfaces, new_uvfaces), dim=1)
        else:
            # Convert per-vertex uvs to per-triangle per-vertex uvs
            uv_coords = uv_coords * 2 - 1 # (V, 2)
            uv_coords[..., 1] = -uv_coords[..., 1]
            face_uvcoords = uv_coords[faces] # (1, V, 3, 2)
            face_uvcoords = torch.cat((face_uvcoords, torch.zeros_like(face_uvcoords[...,0:1])), dim=-1)
            deca.deca.render.face_uvcoords = face_uvcoords  # (1, V, 3, 3)
        yield
    finally:
        deca.deca.render.faces = copy_faces
        deca.deca.render.uvfaces = copy_uvfaces
        deca.deca.render.face_uvcoords = copy_face_uvcoords
        deca.deca.render.face_colors = copy_face_colors
 
def create_deca(model="DECA", cfg="cfg.yaml", detail_mode=False, inference_only=False, checkpoint: str=None):
    mode = "detail" if detail_mode else "coarse"
    model = EMOCA_PATH / "assets/EMOCA/models" / model
    if not checkpoint:
        checkpoint = glob(str(model / "**/*.ckpt"), recursive=True)[0]

    import warnings
    warnings.filterwarnings("ignore", ".*Found keys that are in the model state dict but not in the checkpoint*")
    warnings.filterwarnings("ignore", ".*Found keys that are not in the model state dict but in the checkpoint*")
    warnings.filterwarnings("ignore", ".*To copy construct from a tensor, it is recommended to use*")
    warnings.filterwarnings("ignore", ".*Mtl file does not exist*")
 
    cfg = OmegaConf.load(model / cfg)
    from gdl.models.DECA import instantiate_deca
    from gdl_apps.EMOCA.utils.load import replace_asset_dirs
    cfg = replace_asset_dirs(cfg, Path("/bulk/tmp_test/out_emoca_tmp/emoca"))
    cfg = cfg[mode]
    cfg.model.use_texture = True # this is forced to false in the EMOCA code for packaging pre-trained models

    # Avoid loading stuff when not testing
    if inference_only:
        if "lipread_loss" in cfg.model:
            cfg.model.lipread_loss.load = False
        cfg.model.emonet_model_path = ""
    else:
        if "lipread_loss" in cfg.model:
            cfg.model.lipread_loss.load = True

    checkpoint_kwargs = {"model_params": cfg.model, "learning_params": cfg.learning, "inout_params": cfg.inout, "stage_name": ""}
    deca = instantiate_deca(cfg, "train", "", checkpoint, checkpoint_kwargs)

    if inference_only:
        # Close the mouth interior (for the semantic IoU metric)
        extra_faces = [[2930, 2933, 2862], [2783, 2782, 2854], [2941, 2857, 2933], [1835, 1830, 1747], [2941, 2783, 2854], [2731, 2945, 2930], [2861, 2731, 2930], [2731, 2730, 2708], [1572, 1595, 1862], [1860, 1573, 1862], [1666, 1835, 1665], [3514, 2783, 2941], [1594, 1595, 1572], [2945, 2731, 2708], [1595, 1830, 1862], [2857, 2862, 2933], [1830, 1595, 1746], [1739, 1835, 1742], [1830, 1746, 1747], [2861, 2930, 2862], [1742, 1835, 1747], [3497, 3514, 2941], [3497, 1852, 3514], [2945, 2708, 2709], [1665, 1835, 1739], [2943, 2945, 2709], [1852, 1835, 1666], [3514, 1852, 1666], [2857, 2941, 2854], [1862, 1573, 1572]]
        extra_faces = torch.tensor(extra_faces, dtype=torch.long).unsqueeze(0)
        n_extra_faces = extra_faces.shape[1]
        deca.deca.render.faces = torch.cat([deca.deca.render.faces, extra_faces], dim=1)
        deca.deca.render.uvfaces = torch.cat([deca.deca.render.uvfaces, torch.zeros_like(deca.deca.render.uvfaces)[:,:n_extra_faces]], dim=1)
        deca.deca.render.face_uvcoords = torch.cat([deca.deca.render.face_uvcoords, torch.zeros_like(deca.deca.render.face_uvcoords)[:,:n_extra_faces]], dim=1)
        deca.deca.render.face_colors = torch.cat([deca.deca.render.face_colors, torch.zeros_like(deca.deca.render.face_colors)[:,:n_extra_faces]], dim=1)

    return deca
