# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# This version of the FLAME class was initially taken from GaussianAvatars: 
# https://github.com/ShenhanQian/GaussianAvatars/blob/main/flame_model/flame.py
# Which was heavily inspired by:
# https://github.com/HavenFeng/photometric_optimization/blob/master/models/FLAME.py.
# Modified by Kelian Baert

from typing import Dict

from .lbs import lbs, vertices2landmarks, blend_shapes, lbs_pose_only, forward_skinning, invert_lbs, batch_rodrigues

import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from collections import defaultdict
from pytorch3d.io import load_obj
from pytorch3d import ops

FLAME_MESH_PATH = "assets/flame/head_template_mesh.obj"
FLAME_LMK_PATH = "assets/flame/landmark_embedding_with_eyes.npy"

# to be downloaded from https://flame.is.tue.mpg.de/download.php
# FLAME_MODEL_PATH = "assets/flame/flame2023.pkl" # FLAME 2023 (versions w/ jaw rotation)
FLAME_MODEL_PATH = "assets/flame/flame2020.pkl" # FLAME 2020
FLAME_PARTS_PATH = "assets/flame/FLAME_masks.pkl" # FLAME Vertex Masks
# Indices of mediapipe landmars for which we have a correspondence the FLAME mesh
MEDIAPIPE_LMK_EMBEDDING_INDICES = [276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398, 466, 7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246, 168, 6, 197, 195, 5, 4, 129, 98, 97, 2, 326, 327, 358, 0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]
MEDIAPIPE_LMK_EMBEDDING_PATH = "assets/flame/mediapipe_landmark_embedding.npz"

def to_tensor(array, dtype=torch.float32):
    if "torch.tensor" not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


class FLAME(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """

    def __init__(
        self,
        shape_params,
        expr_params,
        canonical_pose,
        canonical_expr,
        baked_identity_params=None,
        flame_model_path=FLAME_MODEL_PATH,
        flame_lmk_embedding_path=FLAME_LMK_PATH,
        flame_template_mesh_path=FLAME_MESH_PATH,
        include_mask=True,
        add_teeth=False,
        close_mouth_interior=False,
    ):
        super().__init__()
        self.jaw_enabled = True # jaw is enabled by default
        self.n_exp = expr_params
        self.n_joints = 5 # global, neck, jaw, eyes

        scale = 1

        self.n_shape_params = shape_params
        self.n_expr_params = expr_params

        with open(flame_model_path, "rb") as f:
            ss = pickle.load(f, encoding="latin1")
            flame_model = Struct(**ss)

        self.dtype = torch.float32
        # The vertices of the template model
        self.register_buffer(
            "v_template", to_tensor(to_np(flame_model.v_template), dtype=self.dtype) * scale
        )

        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype) * scale
        shapedirs = torch.cat(
            [shapedirs[:, :, :shape_params], shapedirs[:, :, 300 : 300 + expr_params]],
            2,
        )
        self.register_buffer("shapedirs", shapedirs)

        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer("posedirs", to_tensor(to_np(posedirs), dtype=self.dtype) * scale)
        #
        self.register_buffer(
            "J_regressor", to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype)
        )
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents)
        self.register_buffer(
            "lbs_weights", to_tensor(to_np(flame_model.weights), dtype=self.dtype)
        )

        # Landmark embeddings for FLAME
        lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer("full_lmk_faces_idx", torch.tensor(lmk_embeddings["full_lmk_faces_idx"], dtype=torch.long)) # (L)
        self.register_buffer("full_lmk_bary_coords", torch.tensor(lmk_embeddings["full_lmk_bary_coords"], dtype=self.dtype)) # (1, L, 3)
        self.register_buffer("static_lmk_faces_idx", torch.tensor(lmk_embeddings["static_lmk_faces_idx"], dtype=torch.long)) # (Ls)
        self.register_buffer("static_lmk_bary_coords", torch.tensor(lmk_embeddings["static_lmk_bary_coords"], dtype=self.dtype).unsqueeze(0)) # (1, Ls, 3)
        self.register_buffer("dynamic_lmk_faces_idx", lmk_embeddings["dynamic_lmk_faces_idx"].long()) # (A, Ld)
        self.register_buffer("dynamic_lmk_bary_coords", lmk_embeddings["dynamic_lmk_bary_coords"].to(self.dtype)) # (A, Ld, 3)
        # Static MediaPipe landmark embeddings for FLAME
        lmk_embeddings_mediapipe = np.load(MEDIAPIPE_LMK_EMBEDDING_PATH, allow_pickle=True, encoding='latin1')
        self.register_buffer("static_lmk_faces_idx_mediapipe", torch.tensor(lmk_embeddings_mediapipe["lmk_face_idx"].astype(np.int64), dtype=torch.long))
        self.register_buffer("static_lmk_bary_coords_mediapipe", torch.tensor(lmk_embeddings_mediapipe["lmk_b_coords"], dtype=self.dtype).unsqueeze(0))

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        # add faces and uvs
        verts, faces, aux = load_obj(flame_template_mesh_path, load_textures=False)

        vertex_uvs = aux.verts_uvs
        face_uvs_idx = faces.textures_idx  # index into verts_uvs

        # create uvcoords per face --> this is what you can use for uv map rendering
        # range from -1 to 1 (-1, -1) = left top; (+1, +1) = right bottom
        # pad 1 to the end
        pad = torch.ones(vertex_uvs.shape[0], 1)
        vertex_uvs = torch.cat([vertex_uvs, pad], dim=-1)
        vertex_uvs = vertex_uvs * 2 - 1
        vertex_uvs[..., 1] = -vertex_uvs[..., 1]

        face_uv_coords = face_vertices(vertex_uvs[None], face_uvs_idx[None])[0]
        self.register_buffer("face_uvcoords", face_uv_coords, persistent=False)
        self.register_buffer("faces", faces.verts_idx, persistent=False)

        self.register_buffer("verts_uvs", aux.verts_uvs, persistent=False)
        self.register_buffer("textures_idx", faces.textures_idx, persistent=False)

        # Convert faces of landmarks embeddings to vertex indices
        self.register_buffer("full_lmk_verts_idx", torch.index_select(self.faces, 0, self.full_lmk_faces_idx.view(-1)).view(-1, 3)) # (L, 3)
        self.register_buffer("static_lmk_verts_idx", torch.index_select(self.faces, 0, self.static_lmk_faces_idx.view(-1)).view(-1, 3)) # (Ls, 3)
        self.register_buffer("dynamic_lmk_verts_idx", torch.index_select(self.faces, 0, self.dynamic_lmk_faces_idx.view(-1)).view(*self.dynamic_lmk_faces_idx.shape, 3)) # (A, Ld, 3)
        self.register_buffer("static_lmk_verts_idx_mediapipe", torch.index_select(self.faces, 0, self.static_lmk_faces_idx_mediapipe.view(-1)).view(-1, 3)) # (Ls, 3)

        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer("neck_kin_chain", torch.stack(neck_kin_chain))

        if include_mask:
            self.mask = FlameMask(
                faces=self.faces, 
                faces_t=self.textures_idx,
                num_verts=self.v_template.shape[0], 
                num_faces=self.faces.shape[0], 
            )

        self.has_teeth = add_teeth
        if add_teeth:
            self.add_teeth()
        
        if close_mouth_interior:
            self.close_mouth_interior()

        self.register_buffer("shapedirs_identity", self.shapedirs[:, :, :shape_params], persistent=False)
        self.register_buffer("shapedirs_expression", self.shapedirs[:, :, shape_params:], persistent=False)

        # Per-corner UVs (3 UV coordinates per face)
        self.register_buffer("uvs", self.verts_uvs[self.textures_idx], persistent=False) # (F, 3, 2)

        self.register_buffer("canonical_pose", canonical_pose, persistent=False)
        self.register_buffer("canonical_expr", torch.zeros((self.n_expr_params), dtype=torch.float) if canonical_expr is None else canonical_expr, persistent=False)

        # Compute canonical_verts (template_v in canonical pose)
        pose = self.canonical_pose.unsqueeze(0)
        rotation, neck, jaw, eyes = pose[:, :3], pose[:, 3:6], pose[:, 6:9], pose[:, 9:]
        translation = torch.zeros((1, 3), dtype=torch.float)
        expression = self.canonical_expr.unsqueeze(0)
        canonical_verts = self(baked_identity_params, expression, rotation, neck, jaw, eyes, translation, return_landmarks=False)
        canonical_pose_feature, canonical_transformations = lbs_pose_only(pose, canonical_verts, self.J_regressor, self.parents)

        self.register_buffer("canonical_verts", canonical_verts, persistent=False)
        self.register_buffer("canonical_pose_feature", canonical_pose_feature, persistent=False)
        self.register_buffer("canonical_transformations", canonical_transformations, persistent=False)

    def add_teeth(self):
        # teeth placement parameters
        dist_to_lips = 2 # default: 1.5
        lower_teeth_back_dist = 0.6 # default: 0.4
        thickness_scale = 0.5 # default: 1.0
        y_offset_upper = -0.7
        y_offset_lower = 0.6

        # get reference vertices from lips
        vid_lip_outside_ring_upper = self.mask.get_vid_by_region(['lip_outside_ring_upper'], keep_order=True)

        vid_lip_outside_ring_lower = self.mask.get_vid_by_region(['lip_outside_ring_lower'], keep_order=True)

        v_lip_upper = self.v_template[vid_lip_outside_ring_upper]
        v_lip_lower = self.v_template[vid_lip_outside_ring_lower]

        # construct vertices for teeth
        mean_dist = (v_lip_upper - v_lip_lower).norm(dim=-1, keepdim=True).mean()
        v_teeth_middle = (v_lip_upper + v_lip_lower) / 2
        v_teeth_middle[:, 1] = v_teeth_middle[:, [1]].mean(dim=0, keepdim=True)
        # v_teeth_middle[:, 2] -= mean_dist * 2.5  # how far the teeth are from the lips
        # v_teeth_middle[:, 2] -= mean_dist * 2  # how far the teeth are from the lips
        v_teeth_middle[:, 2] -= mean_dist * dist_to_lips  # how far the teeth are from the lips

        # upper, front
        v_teeth_upper_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]]) * y_offset_upper
        v_teeth_upper_root = v_teeth_upper_edge + torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # lower, front
        v_teeth_lower_edge = v_teeth_middle.clone() + torch.tensor([[0, mean_dist, 0]]) * y_offset_lower
        # v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * 0.2  # slightly move the lower teeth to the back
        v_teeth_lower_edge -= torch.tensor([[0, 0, mean_dist]]) * lower_teeth_back_dist  # slightly move the lower teeth to the back
        v_teeth_lower_root = v_teeth_lower_edge - torch.tensor([[0, mean_dist, 0]]) * 2  # scale the height of teeth

        # thickness = mean_dist * 0.5
        thickness = mean_dist * thickness_scale
        # upper, back
        v_teeth_upper_root_back = v_teeth_upper_root.clone()
        v_teeth_upper_edge_back = v_teeth_upper_edge.clone()
        v_teeth_upper_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_upper_edge_back[:, 2] -= thickness  # how thick the teeth are

        # lower, back
        v_teeth_lower_root_back = v_teeth_lower_root.clone()
        v_teeth_lower_edge_back = v_teeth_lower_edge.clone()
        v_teeth_lower_root_back[:, 2] -= thickness  # how thick the teeth are
        v_teeth_lower_edge_back[:, 2] -= thickness  # how thick the teeth are

        # concatenate to v_template
        num_verts_orig = self.v_template.shape[0]
        v_teeth = torch.cat([
            v_teeth_upper_root,  # num_verts_orig + 0-14 
            v_teeth_lower_root,  # num_verts_orig + 15-29
            v_teeth_upper_edge,  # num_verts_orig + 30-44
            v_teeth_lower_edge,  # num_verts_orig + 45-59
            v_teeth_upper_root_back,  # num_verts_orig + 60-74
            v_teeth_upper_edge_back,  # num_verts_orig + 75-89
            v_teeth_lower_root_back,  # num_verts_orig + 90-104
            v_teeth_lower_edge_back,  # num_verts_orig + 105-119
        ], dim=0)
        num_verts_teeth = v_teeth.shape[0]
        self.v_template = torch.cat([self.v_template, v_teeth], dim=0)

        vid_teeth_upper_root = torch.arange(0, 15) + num_verts_orig
        vid_teeth_lower_root = torch.arange(15, 30) + num_verts_orig
        vid_teeth_upper_edge = torch.arange(30, 45) + num_verts_orig
        vid_teeth_lower_edge = torch.arange(45, 60) + num_verts_orig
        vid_teeth_upper_root_back = torch.arange(60, 75) + num_verts_orig
        vid_teeth_upper_edge_back = torch.arange(75, 90) + num_verts_orig
        vid_teeth_lower_root_back = torch.arange(90, 105) + num_verts_orig
        vid_teeth_lower_edge_back = torch.arange(105, 120) + num_verts_orig
        
        vid_teeth_upper = torch.cat([vid_teeth_upper_root, vid_teeth_upper_edge, vid_teeth_upper_root_back, vid_teeth_upper_edge_back], dim=0)
        vid_teeth_lower = torch.cat([vid_teeth_lower_root, vid_teeth_lower_edge, vid_teeth_lower_root_back, vid_teeth_lower_edge_back], dim=0)
        vid_teeth = torch.cat([vid_teeth_upper, vid_teeth_lower], dim=0)

        # update vertex masks
        self.mask.v.register_buffer("teeth_upper", vid_teeth_upper)
        self.mask.v.register_buffer("teeth_lower", vid_teeth_lower)
        self.mask.v.register_buffer("teeth", vid_teeth)
        self.mask.v.left_half = torch.cat([
            self.mask.v.left_half, 
            torch.tensor([
                5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030, 5038, 5039, 5040, 5041, 5042, 5043, 5044, 5045, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5068, 5069, 5070, 5071, 5072, 5073, 5074, 5075, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5105, 5113, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 
            ])], dim=0)

        self.mask.v.right_half = torch.cat([
            self.mask.v.right_half, 
            torch.tensor([
                5030, 5031, 5032, 5033, 5034, 5035, 5036, 5037, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5060, 5061, 5062, 5063, 5064, 5065, 5066, 5067, 5075, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5090, 5091, 5092, 5093, 5094, 5095, 5097, 5105, 5106, 5107, 5108, 5109, 5110, 5111, 5112, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5135, 5136, 5137, 5138, 5139, 5140, 5141, 5142, 
            ])], dim=0)

        # construct uv vertices for teeth
        u = torch.linspace(0.62, 0.38, 15)
        v = torch.linspace(1-0.0083, 1-0.0425, 7)
        # v = v[[0, 2, 1, 1]]
        # v = v[[0, 3, 1, 4, 3, 2, 6, 5]]
        v = v[[3, 2, 0, 1, 3, 4, 6, 5]]  # TODO: with this order, teeth_lower is not rendered correctly in the uv space
        uv = torch.stack(torch.meshgrid(u, v, indexing='ij'), dim=-1).permute(1, 0, 2).reshape(num_verts_teeth, 2)  # (#num_teeth, 2)
        num_verts_uv_orig = self.verts_uvs.shape[0]
        num_verts_uv_teeth = uv.shape[0]
        self.verts_uvs = torch.cat([self.verts_uvs, uv], dim=0)

        # shapedirs copy from lips
        self.shapedirs = torch.cat([self.shapedirs, torch.zeros_like(self.shapedirs[:num_verts_teeth])], dim=0)
        shape_dirs_mean = (self.shapedirs[vid_lip_outside_ring_upper, :, :self.n_shape_params] + self.shapedirs[vid_lip_outside_ring_lower, :, :self.n_shape_params]) / 2
        self.shapedirs[vid_teeth_upper_root, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_root_back, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_upper_edge_back, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_root_back, :, :self.n_shape_params] = shape_dirs_mean
        self.shapedirs[vid_teeth_lower_edge_back, :, :self.n_shape_params] = shape_dirs_mean

        # posedirs set to zero
        posedirs = self.posedirs.reshape(len(self.parents)-1, 9, num_verts_orig, 3)  # (J*9, V*3) -> (J, 9, V, 3)
        posedirs = torch.cat([posedirs, torch.zeros_like(posedirs[:, :, :num_verts_teeth])], dim=2)  # (J, 9, V+num_verts_teeth, 3)
        self.posedirs = posedirs.reshape((len(self.parents)-1)*9, (num_verts_orig+num_verts_teeth)*3)  # (J*9, (V+num_verts_teeth)*3)

        # J_regressor set to zero
        self.J_regressor = torch.cat([self.J_regressor, torch.zeros_like(self.J_regressor[:, :num_verts_teeth])], dim=1)  # (5, J) -> (5, J+num_verts_teeth)

        # lbs_weights manually set
        self.lbs_weights = torch.cat([self.lbs_weights, torch.zeros_like(self.lbs_weights[:num_verts_teeth])], dim=0)  # (V, 5) -> (V+num_verts_teeth, 5)
        self.lbs_weights[vid_teeth_upper, 1] += 1  # move with neck
        self.lbs_weights[vid_teeth_lower, 2] += 1  # move with jaw

        # add faces for teeth
        f_teeth_upper = torch.tensor([
            [0, 31, 30],  #0
            [0, 1, 31],  #1
            [1, 32, 31],  #2
            [1, 2, 32],  #3
            [2, 33, 32],  #4
            [2, 3, 33],  #5
            [3, 34, 33],  #6
            [3, 4, 34],  #7
            [4, 35, 34],  #8
            [4, 5, 35],  #9
            [5, 36, 35],  #10
            [5, 6, 36],  #11
            [6, 37, 36],  #12
            [6, 7, 37],  #13
            [7, 8, 37],  #14
            [8, 38, 37],  #15
            [8, 9, 38],  #16
            [9, 39, 38],  #17
            [9, 10, 39],  #18
            [10, 40, 39],  #19
            [10, 11, 40],  #20
            [11, 41, 40],  #21
            [11, 12, 41],  #22
            [12, 42, 41],  #23
            [12, 13, 42],  #24
            [13, 43, 42],  #25
            [13, 14, 43],  #26
            [14, 44, 43],  #27
            [60, 75, 76],  # 56
            [60, 76, 61],  # 57
            [61, 76, 77],  # 58
            [61, 77, 62],  # 59
            [62, 77, 78],  # 60
            [62, 78, 63],  # 61
            [63, 78, 79],  # 62
            [63, 79, 64],  # 63
            [64, 79, 80],  # 64
            [64, 80, 65],  # 65
            [65, 80, 81],  # 66
            [65, 81, 66],  # 67
            [66, 81, 82],  # 68
            [66, 82, 67],  # 69
            [67, 82, 68],  # 70
            [68, 82, 83],  # 71
            [68, 83, 69],  # 72
            [69, 83, 84],  # 73
            [69, 84, 70],  # 74
            [70, 84, 85],  # 75
            [70, 85, 71],  # 76
            [71, 85, 86],  # 77
            [71, 86, 72],  # 78
            [72, 86, 87],  # 79
            [72, 87, 73],  # 80
            [73, 87, 88],  # 81
            [73, 88, 74],  # 82
            [74, 88, 89],  # 83
            [75, 30, 76],  # 84
            [76, 30, 31],  # 85
            [76, 31, 77],  # 86
            [77, 31, 32],  # 87
            [77, 32, 78],  # 88
            [78, 32, 33],  # 89
            [78, 33, 79],  # 90
            [79, 33, 34],  # 91
            [79, 34, 80],  # 92
            [80, 34, 35],  # 93
            [80, 35, 81],  # 94
            [81, 35, 36],  # 95
            [81, 36, 82],  # 96
            [82, 36, 37],  # 97
            [82, 37, 38],  # 98
            [82, 38, 83],  # 99
            [83, 38, 39],  # 100
            [83, 39, 84],  # 101
            [84, 39, 40],  # 102
            [84, 40, 85],  # 103
            [85, 40, 41],  # 104
            [85, 41, 86],  # 105
            [86, 41, 42],  # 106
            [86, 42, 87],  # 107
            [87, 42, 43],  # 108
            [87, 43, 88],  # 109
            [88, 43, 44],  # 110
            [88, 44, 89],  # 111
        ])
        f_teeth_lower = torch.tensor([
            [45, 46, 15],  # 28           
            [46, 16, 15],  # 29
            [46, 47, 16],  # 30
            [47, 17, 16],  # 31
            [47, 48, 17],  # 32
            [48, 18, 17],  # 33
            [48, 49, 18],  # 34
            [49, 19, 18],  # 35
            [49, 50, 19],  # 36
            [50, 20, 19],  # 37
            [50, 51, 20],  # 38
            [51, 21, 20],  # 39
            [51, 52, 21],  # 40
            [52, 22, 21],  # 41
            [52, 23, 22],  # 42
            [52, 53, 23],  # 43
            [53, 24, 23],  # 44
            [53, 54, 24],  # 45
            [54, 25, 24],  # 46
            [54, 55, 25],  # 47
            [55, 26, 25],  # 48
            [55, 56, 26],  # 49
            [56, 27, 26],  # 50
            [56, 57, 27],  # 51
            [57, 28, 27],  # 52
            [57, 58, 28],  # 53
            [58, 29, 28],  # 54
            [58, 59, 29],  # 55
            [90, 106, 105],  # 112
            [90, 91, 106],  # 113
            [91, 107, 106],  # 114
            [91, 92, 107],  # 115
            [92, 108, 107],  # 116
            [92, 93, 108],  # 117
            [93, 109, 108],  # 118
            [93, 94, 109],  # 119
            [94, 110, 109],  # 120
            [94, 95, 110],  # 121
            [95, 111, 110],  # 122
            [95, 96, 111],  # 123
            [96, 112, 111],  # 124
            [96, 97, 112],  # 125
            [97, 98, 112],  # 126
            [98, 113, 112],  # 127
            [98, 99, 113],  # 128
            [99, 114, 113],  # 129
            [99, 100, 114],  # 130
            [100, 115, 114],  # 131
            [100, 101, 115],  # 132
            [101, 116, 114],  # 133
            [101, 102, 116],  # 134
            [102, 117, 116],  # 135
            [102, 103, 117],  # 136
            [103, 118, 117],  # 137
            [103, 104, 118],  # 138
            [104, 119, 118],  # 139
            [105, 106, 45],  # 140
            [106, 46, 45],  # 141
            [106, 107, 46],  # 142
            [107, 47, 46],  # 143
            [107, 108, 47],  # 144
            [108, 48, 47],  # 145
            [108, 109, 48],  # 146
            [109, 49, 48],  # 147
            [109, 110, 49],  # 148
            [110, 50, 49],  # 149
            [110, 111, 50],  # 150
            [111, 51, 50],  # 151
            [111, 112, 51],  # 152
            [112, 52, 51],  # 153
            [112, 53, 52],  # 154
            [112, 113, 53],  # 155
            [113, 54, 53],  # 156
            [113, 114, 54],  # 157
            [114, 55, 54],  # 158
            [114, 115, 55],  # 159
            [115, 56, 55],  # 160
            [115, 116, 56],  # 161
            [116, 57, 56],  # 162
            [116, 117, 57],  # 163
            [117, 58, 57],  # 164
            [117, 118, 58],  # 165
            [118, 59, 58],  # 166
            [118, 119, 59],  # 167
        ])
        self.faces = torch.cat([self.faces, f_teeth_upper+num_verts_orig, f_teeth_lower+num_verts_orig], dim=0)
        self.textures_idx = torch.cat([self.textures_idx, f_teeth_upper+num_verts_uv_orig, f_teeth_lower+num_verts_uv_orig], dim=0)

        self.mask.update(self.faces, self.textures_idx)

    def forward(
        self,
        shape,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        translation,
        zero_centered_at_root_node=False,  # otherwise, zero centered at the face
        return_landmarks=True,
        return_verts_cano=False,
        static_offset=None,
        dynamic_offset=None,
    ):
        """
        Input:
            shape_params: N X number of shape parameters
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (6)
        return:d
            vertices: N X V X 3
            landmarks: N X number of landmarks X 3
        """
        batch_size = shape.shape[0]

        betas = torch.cat([shape, expr], dim=1)
        full_pose = torch.cat([rotation, neck, jaw * (1 if self.jaw_enabled else 0), eyes], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add shape contribution
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)

        # Add personal offsets
        if static_offset is not None:
            v_shaped += static_offset

        vertices, J, mat_rot = lbs(
            full_pose,
            v_shaped,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype,
        )

        if zero_centered_at_root_node:
            vertices = vertices - J[:, [0]]
            J = J - J[:, [0]]

        vertices = vertices + translation[:, None, :]
        J = J + translation[:, None, :]

        ret_vals = [vertices]

        if return_verts_cano:
            ret_vals.append(v_shaped)

        # compute landmarks if desired
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices2landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1),
            )
            ret_vals.append(landmarks)

        if len(ret_vals) > 1:
            return ret_vals
        else:
            return ret_vals[0]
   
    ############################################################################################################################

    def close_mouth_interior(self):
        extra_faces = [[2930, 2933, 2862], [2783, 2782, 2854], [2941, 2857, 2933], [1835, 1830, 1747], [2941, 2783, 2854], [2731, 2945, 2930], [2861, 2731, 2930], [2731, 2730, 2708], [1572, 1595, 1862], [1860, 1573, 1862], [1666, 1835, 1665], [3514, 2783, 2941], [1594, 1595, 1572], [2945, 2731, 2708], [1595, 1830, 1862], [2857, 2862, 2933], [1830, 1595, 1746], [1739, 1835, 1742], [1830, 1746, 1747], [2861, 2930, 2862], [1742, 1835, 1747], [3497, 3514, 2941], [3497, 1852, 3514], [2945, 2708, 2709], [1665, 1835, 1739], [2943, 2945, 2709], [1852, 1835, 1666], [3514, 1852, 1666], [2857, 2941, 2854], [1862, 1573, 1572]]
        extra_faces = torch.tensor(extra_faces, dtype=torch.long)
        self.faces = torch.cat([self.faces, extra_faces], dim=0)
        logging.warning("UV coordinates for mouth interior are not computed.")
        self.textures_idx = torch.cat((self.textures_idx, torch.zeros_like(extra_faces)), dim=0)

    def get_landmark_positions(self, vertices, which="full"):
        """Calculates landmarks positions by barycentric interpolation

        Parameters
        ----------
        vertices: torch.tensor BxVx3, dtype = torch.float32
            The tensor of input vertices
        Returns
        -------
        landmarks: torch.tensor BxLx3, dtype = torch.float32
            The coordinates of the landmarks for each mesh in the batch
        """
        if which == "full":
            lmk_verts = self.full_lmk_verts_idx # (L, 3)
            lmk_bary_coords = self.full_lmk_bary_coords # (1, L, 3)
        elif which == "static":
            lmk_verts = self.static_lmk_verts_idx # (L, 3)
            lmk_bary_coords = self.static_lmk_bary_coords # (1, L, 3)
        elif which == "dynamic":
            lmk_verts = self.dynamic_lmk_verts_idx.view(-1, 3) # (A*L, 3)
            lmk_bary_coords = self.dynamic_lmk_bary_coords.view(1, -1, 3) # (1, A*L, 3)
        elif which == "static_mediapipe":
            lmk_verts = self.static_lmk_verts_idx_mediapipe # (L, 3)
            lmk_bary_coords = self.static_lmk_bary_coords_mediapipe # (1, L, 3)
        else:
            raise ValueError(f"no landmarks '{which}'")

        lmk_verts_pos = vertices[:, lmk_verts] # (B, L, 3, 3)
        landmarks = torch.einsum("blfi,blf->bli", [lmk_verts_pos, lmk_bary_coords]) # (B, L, 3)
        return landmarks

    def _find_dynamic_lmk_idx_and_bcoords(self, pose):
        """
            Selects the face contour depending on the relative position of the head
            Input:
                pose: N X full pose
            return:
                The contour face indexes and the corresponding barycentric weights
        """
        def rot_mat_to_euler(rot_mats):
            # Calculates rotation matrix to euler angles
            # Careful for extreme cases of eular angles like [0.0, pi, 0.0]
            sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                            rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
            return torch.atan2(-rot_mats[:, 2, 0], sy)

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1, self.neck_kin_chain)
        rot_mats = batch_rodrigues(aa_pose.view(-1, 3), dtype=self.dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device, dtype=self.dtype).unsqueeze(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(self.neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle)

        dynamic_lmk_verts_idx = torch.index_select(self.dynamic_lmk_verts_idx, 0, y_rot_angle)
        dynamic_lmk_bary_coords = torch.index_select(self.dynamic_lmk_bary_coords, 0, y_rot_angle)
        return dynamic_lmk_verts_idx, dynamic_lmk_bary_coords
    
    def get_landmark_positions_2d(self, vertices, full_pose):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                full_pose: torch.tensor N X 12, dtype = torch.float32
                    The tensor with global pose, neck pose, jaw pose and eye pose (respectively) in axis angle format

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        batch_size = vertices.shape[0]
        static_lmk_verts = self.static_lmk_verts_idx.unsqueeze(0).expand(batch_size, -1, 3) # (B, Ls, 3)
        static_lmk_bary_coords = self.static_lmk_bary_coords.expand(batch_size, -1, 3) # (B, Ls, 3)

        dyn_lmk_verts, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(full_pose) # (B, Ld, 3)

        lmk_verts = torch.cat([dyn_lmk_verts, static_lmk_verts], 1) # (B, L, 3)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, static_lmk_bary_coords], 1)

        lmk_verts_pos = torch.stack([vertices[i, v] for i,v in enumerate(lmk_verts)]) # (B, L, 3, 3)
        landmarks = torch.einsum("blfi,blf->bli", [lmk_verts_pos, lmk_bary_coords]) # (B, L, 3)
        return landmarks

    def forward_skinning(
        self,
        canonical_verts,
        expr,
        rotation,
        neck,
        jaw,
        eyes,
        shapedirs,
        posedirs,
        lbs_weights,
    ):
        """Similar to `forward`, but we revert the canonical pose first.
        Also, we don't account for the identity part of shapedirs because it is assumed to be baked in canonical_verts.
        """
        batch_size = expr.shape[0]
        n_verts = shapedirs.shape[0]

        full_pose = torch.cat([rotation, neck, jaw * (1 if self.jaw_enabled else 0), eyes], dim=1)

        # Compose pose features and transformations
        #template_canonical = self.v_template.unsqueeze(0).expand(batch_size, -1, -1).to(full_pose.device)
        template_canonical = self.canonical_verts.expand(batch_size, -1, -1).to(full_pose.device)
        pose_feature, transformations = lbs_pose_only(full_pose, template_canonical, self.J_regressor, self.parents, dtype=self.dtype)

        # First we must cancel the canonical_pose transformation
        # When using remeshing, this only makes sense if posedirs and lbs_weights have been properly updated
        canonical_verts = invert_lbs(canonical_verts, 
                                self.canonical_transformations.unsqueeze(0).repeat(batch_size, n_verts, 1, 1, 1), 
                                self.canonical_pose_feature.repeat(batch_size, 1), # (1, 36) => (B, 36)
                                posedirs, lbs_weights)
        # Also cancel out the canonical expression
        canonical_verts = canonical_verts - blend_shapes(self.canonical_expr.unsqueeze(0).repeat(batch_size, 1), shapedirs)

        vertices = forward_skinning(expr, full_pose, canonical_verts, shapedirs, posedirs, lbs_weights,
                                    pose_feature, transformations)
        return vertices
    
    def blendshapes_nearest(self, canonical_vertices, c_pts_masked=None):
        '''
        Computes the nearest neighbours for blendshapes and skinning weights
        '''
        # find nearest indx
        knn_v = self.canonical_verts.clone()
        flame_distances, idx, _ = ops.knn_points(canonical_vertices.unsqueeze(0), knn_v, K=1, return_nn=True)
        idx = idx.reshape(-1)

        # reshape flame posedirs to fit the shape of deformer net
        nearest_posedirs = self.posedirs.view((self.n_joints-1)*9, -1, 3).permute(1, 0, 2)[idx, :, :]  # (J*9, V*3) -> (V, J*9, 3)
        nearest_shapedirs = self.shapedirs_expression[idx, :, :]
        nearest_lbs_weights = self.lbs_weights[idx, :]   

        if c_pts_masked is not None:
            # mouth interior does not deform with expression/pose
            _, idx_mouth_nearest, _ = ops.knn_points(c_pts_masked[0].unsqueeze(0), canonical_vertices.unsqueeze(0), K=1, return_nn=True)
            idx_mouth = torch.unique(idx_mouth_nearest)
            nearest_shapedirs[idx_mouth, ...] = 0.0
            nearest_posedirs[idx_mouth, ...] = 0.0

            if c_pts_masked[1] is not None:
                # cloth does not deform with expression
                _, idx_cloth_nearest, _ = ops.knn_points(c_pts_masked[1].unsqueeze(0), canonical_vertices.unsqueeze(0), K=1, return_nn=True)
                idx_cloth = torch.unique(idx_cloth_nearest)
                nearest_shapedirs[idx_cloth, ...] = 0.0

        return nearest_shapedirs, nearest_posedirs, nearest_lbs_weights, flame_distances

class BufferContainer(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        main_str = super().__repr__() + '\n'
        for name, buf in self.named_buffers():
            main_str += f'    {name:20}\t{buf.shape}\t{buf.dtype}\n'
        return main_str
    
    def __iter__(self):
        for name, buf in self.named_buffers():
            yield name, buf
    
    def keys(self):
        return [name for name, buf in self.named_buffers()]
    
    def items(self):
        return [(name, buf) for name, buf in self.named_buffers()]
    
class FlameMask(nn.Module):
    def __init__(
            self, 
            flame_parts_path=FLAME_PARTS_PATH, 
            faces=None, 
            faces_t=None, 
            num_verts=5023,
            num_faces=9976,
            face_clusters=[],
        ):
        super().__init__()
        self.faces = faces
        self.faces_t = faces_t
        self.face_clusters = face_clusters
        self.num_verts = num_verts
        if faces is not None:
            self.num_faces = faces.shape[0]
        else:
            self.num_faces = num_faces

        self.process_vertex_mask(flame_parts_path)
        self.create_float_masks()

        # We don't need these
        # if self.faces is not None:
        #     self.construct_vid_table()
        #     self.process_face_mask(self.faces)
        #     self.process_face_clusters(self.face_clusters)
        #     if self.faces_t is not None:
        #         self.process_vt_mask(self.faces, self.faces_t)
    
    def update(self, faces=None, faces_t=None, face_clusters=None):
        """Update the faces properties when vertex masks are changed"""
        if faces is not None:
            self.faces = faces
            self.num_faces = faces.shape[0]
        if faces_t is not None:
            self.faces_t = faces_t
        if face_clusters is not None:
            self.face_clusters = face_clusters
        
        self.create_float_masks()

        # self.construct_vid_table()
        # self.process_face_mask(self.faces)
        # self.process_face_clusters(self.face_clusters)
        # if self.faces_t is not None:
        #     self.process_vt_mask(self.faces, self.faces_t)

    def process_vertex_mask(self, flame_parts_path):
        """Load the vertex masks from the FLAME model and add custom masks"""

        part_masks = np.load(flame_parts_path, allow_pickle=True, encoding="latin1")
        """ Available part masks from the FLAME model: 
                face, neck, scalp, boundary, right_eyeball, left_eyeball, 
                right_ear, left_ear, forehead, eye_region, nose, lips,
                right_eye_region, left_eye_region.
        """

        self.v = BufferContainer()
        for k, v_mask in part_masks.items():
            self.v.register_buffer(k, torch.tensor(v_mask, dtype=torch.long))
        
        self.create_custom_mask()

    def create_custom_mask(self):
        """Add some cutom masks based on the original FLAME masks"""

        self.v.register_buffer("neck_left_point", torch.tensor([3193]))
        self.v.register_buffer("neck_right_point", torch.tensor([3296]))
        self.v.register_buffer("front_middle_bottom_point_boundary", torch.tensor([3285]))
        self.v.register_buffer("back_middle_bottom_point_boundary", torch.tensor([3248]))

        self.v.register_buffer(
            "neck_top", 
            torch.tensor([
                10, 11, 111, 112, 784, 795, 1325, 1901, 2115, 2162, 2251, 2254, 2483, 2979, 3142, 3174, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3562, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 
            ])
        )

        self.v.register_buffer(
            "lip_inside_ring_upper", 
            torch.tensor([
                1595, 1746, 1747, 1742, 1739, 1665, 1666, 3514, 2783, 2782, 2854, 2857, 2862, 2861, 2731
            ])
        )

        self.v.register_buffer(
            "lip_inside_ring_lower", 
            torch.tensor([
                1572, 1573, 1860, 1862, 1830, 1835, 1852, 3497, 2941, 2933, 2930, 2945, 2943, 2709, 2708
            ])
        )

        self.v.register_buffer(
            "lip_outside_ring_upper", 
            torch.tensor([
                1713, 1715, 1716, 1735, 1696, 1694, 1657, 3543, 2774, 2811, 2813, 2850, 2833, 2832, 2830
            ])
        )

        self.v.register_buffer(
            "lip_outside_ring_lower", 
            torch.tensor([
                1576, 1577, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880, 2713, 2712
            ])
        )

        self.v.register_buffer(
            "lip_inside_upper", 
            torch.tensor([
                1588, 1589, 1590, 1591, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1724, 1725, 1739, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 2724, 2725, 2726, 2727, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2841, 2842, 2854, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 3514, 3547, 3549, 
            ])
        )

        self.v.register_buffer(
            "lip_inside_lower", 
            torch.tensor([
                1572, 1573, 1592, 1593, 1764, 1765, 1779, 1780, 1781, 1830, 1831, 1832, 1835, 1846, 1847, 1851, 1852, 1854, 1860, 1861, 1862, 2708, 2709, 2728, 2729, 2872, 2873, 2886, 2887, 2888, 2930, 2931, 2932, 2933, 2935, 2936, 2940, 2941, 2942, 2943, 2944, 2945, 3497, 3500, 3512, 
            ])
        )

        self.v.register_buffer(
            "lip_inside", 
            torch.tensor([
                1572, 1573, 1580, 1581, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1718, 1719, 1722, 1724, 1725, 1728, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1764, 1765, 1777, 1778, 1779, 1780, 1781, 1782, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1851, 1852, 1854, 1860, 1861, 1862, 2708, 2709, 2716, 2717, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2835, 2836, 2839, 2841, 2842, 2843, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2872, 2873, 2884, 2885, 2886, 2887, 2888, 2889, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2940, 2941, 2942, 2943, 2944, 2945, 3497, 3500, 3512, 3513, 3514, 3533, 3547, 3549, 
            ])
        )

        self.v.register_buffer(
            "neck_upper", 
            torch.tensor([
                10, 11, 12, 13, 14, 15, 111, 112, 219, 220, 221, 222, 372, 373, 374, 375, 462, 463, 496, 497, 552, 553, 558, 559, 563, 564, 649, 650, 736, 737, 784, 795, 1210, 1211, 1212, 1213, 1325, 1326, 1359, 1360, 1386, 1726, 1727, 1759, 1790, 1886, 1898, 1901, 1931, 1932, 1933, 1934, 1940, 1941, 1948, 1949, 2036, 2115, 2149, 2150, 2151, 2162, 2218, 2219, 2251, 2254, 2483, 2484, 2531, 2870, 2893, 2964, 2976, 2979, 3012, 3013, 3142, 3174, 3184, 3185, 3186, 3187, 3188, 3189, 3193, 3194, 3196, 3199, 3200, 3202, 3203, 3206, 3209, 3281, 3282, 3286, 3291, 3292, 3296, 3297, 3299, 3302, 3303, 3305, 3306, 3309, 3312, 3376, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3494, 3496, 3544, 3562, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 3695, 3697, 3698, 3701, 3703, 3707, 3709, 3713, 
            ])
        )

        self.v.register_buffer(
            "neck_lower", 
            torch.tensor([
                3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3222, 3223, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3250, 3251, 3253, 3254, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3275, 3276, 3277, 3278, 3281, 3282, 3283, 3286, 3288, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3323, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3352, 3353, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3376, 3378, 
            ])
        )

        # the bottomline of "neck"
        self.v.register_buffer(
            "neck_base", 
            torch.tensor([
                3231, 3232, 3237, 3238, 3240, 3242, 3243, 3251, 3263, 3290, 3332, 3333, 3338, 3339, 3341, 3343, 3344, 3350, 3362,  # 4-th ring from bottom (drop 7 front verts)
            ])
        )

        # As a subset of "boundary", "bottomline" only contains vertices on the edge
        self.v.register_buffer(
            "bottomline", 
            torch.tensor([
                3218, 3219, 3226, 3272, 3273, 3229, 3228, 3261, 3260, 3248, 3359, 3360, 3329, 3330, 3372, 3371, 3327, 3322, 3321, 3355, 3354, 3356, 3357, 3379, 3285, 3289, 3258, 3257, 3255, 3256
            ])
        )

        self.v.register_buffer(
            "left_iris", 
            torch.tensor([
                3931, 3932, 3933, 3935, 3936, 3937, 3939, 3940, 3941, 3943, 3944, 3945, 3947, 3948, 3949, 3951, 3952, 3953, 3955, 3956, 3957, 3959, 3960, 3961, 3963, 3964, 3965, 3967, 3968, 3969, 3971, 3972, 3973, 3975, 3976, 3977, 3979, 3980, 3981, 3983, 3984, 3985, 3987, 3988, 3989, 3991, 3992, 3993, 3995, 3996, 3997, 3999, 4000, 4001, 4003, 4004, 4005, 4007, 4008, 4009, 4011, 4012, 4013, 4015, 4016, 4017, 4019, 4020, 4021, 4023, 4024, 4025, 4027, 4028, 4029, 4031, 4032, 4033, 4035, 4036, 4037, 4039, 4040, 4041, 4043, 4044, 4045, 4047, 4048, 4049, 4051, 4052, 4053, 4054, 4056, 4057, 4058, 
            ])
        )

        self.v.register_buffer(
            "right_iris", 
            torch.tensor([
                4477, 4478, 4479, 4481, 4482, 4483, 4485, 4486, 4487, 4489, 4490, 4491, 4493, 4494, 4495, 4497, 4498, 4499, 4501, 4502, 4503, 4505, 4506, 4507, 4509, 4510, 4511, 4513, 4514, 4515, 4517, 4518, 4519, 4521, 4522, 4523, 4525, 4526, 4527, 4529, 4530, 4531, 4533, 4534, 4535, 4537, 4538, 4539, 4541, 4542, 4543, 4545, 4546, 4547, 4549, 4550, 4551, 4553, 4554, 4555, 4557, 4558, 4559, 4561, 4562, 4563, 4565, 4566, 4567, 4569, 4570, 4571, 4573, 4574, 4575, 4577, 4578, 4579, 4581, 4582, 4583, 4585, 4586, 4587, 4589, 4590, 4591, 4593, 4594, 4595, 4597, 4598, 4599, 4600, 4602, 4603, 4604, 
            ])
        )

        self.v.register_buffer(
            "left_eyelid",  # 30 vertices
            torch.tensor([
                807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 841, 842, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 903, 904, 905, 922, 923, 924, 926, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 991, 992, 993, 994, 995, 999, 1000, 1003, 1006, 1008, 1011, 1023, 1033, 1034, 1045, 1046, 1059, 1060, 1061, 1062, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1125, 1126, 1132, 1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1170, 1175, 1182, 1183, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1227, 1230, 1232, 1233, 1243, 1244, 1283, 1289, 1292, 1293, 1294, 1320, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 3827, 3832, 3833, 3835, 3853, 3855, 3856, 3861, 
            ])
        )

        self.v.register_buffer(
            "right_eyelid",  # 30 vertices
            torch.tensor([
                2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2282, 2283, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2303, 2304, 2305, 2312, 2313, 2314, 2315, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2364, 2365, 2367, 2369, 2381, 2382, 2383, 2386, 2387, 2388, 2389, 2390, 2391, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2411, 2412, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2436, 2437, 2440, 2441, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2457, 2460, 2461, 2462, 2465, 2466, 2467, 2470, 2471, 2472, 2473, 2478, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 3619, 3631, 3632, 3638, 3687, 3689, 3690, 3700, 
            ])
        )

        self.v.register_buffer(
            "left_eyelid_extended",
            torch.tensor([
                807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 841, 842, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 903, 904, 905, 922, 923, 924, 926, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 991, 992, 993, 994, 995, 999, 1000, 1003, 1006, 1008, 1011, 1012, 1013, 1015, 1019, 1020, 1021, 1022, 1023, 1033, 1034, 1043, 1044, 1045, 1046, 1059, 1060, 1061, 1062, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1125, 1126, 1132, 1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1170, 1175, 1182, 1183, 1184, 1190, 1193, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1227, 1230, 1232, 1233, 1243, 1244, 1283, 1289, 1292, 1293, 1294, 1320, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 1362, 1366, 1367, 1450, 1451, 1452, 1453, 1458, 1461, 3827, 3832, 3833, 3835, 3852, 3853, 3855, 3856, 3857, 3858, 3860, 3861
            ])
        )

        self.v.register_buffer(
            "right_eyelid_extended",
            torch.tensor([
                2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2282, 2283, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2303, 2304, 2305, 2312, 2313, 2314, 2315, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2364, 2365, 2367, 2369, 2370, 2371, 2373, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2411, 2412, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2436, 2437, 2440, 2441, 2442, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2457, 2460, 2461, 2462, 2465, 2466, 2467, 2470, 2471, 2472, 2473, 2478, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2515, 2516, 2587, 2588, 2589, 2590, 2595, 2598, 3619, 3631, 3632, 3638, 3686, 3687, 3689, 3690, 3692, 3694, 3699, 3700
            ])
        )

        self.v.register_buffer(
            "lips_tight",  # 30 vertices
            torch.tensor([
                1572, 1573, 1578, 1580, 1581, 1582, 1583, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1750, 1751, 1758, 1764, 1765, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1802, 1803, 1804, 1826, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1854, 1860, 1861, 1862, 1865, 2708, 2709, 2714, 2716, 2717, 2718, 2719, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2865, 2866, 2869, 2872, 2873, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2905, 2906, 2907, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2948, 3497, 3500, 3503, 3504, 3506, 3509, 3512, 3513, 3514, 3531, 3533, 3546, 3547, 3549, 
            ])
        )

        self.v.register_buffer(
            "left_half",
            torch.tensor([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 530, 531, 532, 533, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 588, 589, 590, 591, 592, 593, 594, 603, 604, 605, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 638, 639, 644, 645, 646, 647, 648, 649, 650, 667, 668, 669, 670, 671, 672, 673, 674, 679, 680, 681, 682, 683, 688, 691, 692, 693, 694, 695, 696, 697, 702, 703, 704, 705, 706, 707, 708, 709, 712, 713, 714, 715, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 745, 746, 747, 748, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 783, 784, 785, 786, 795, 796, 797, 798, 799, 802, 803, 804, 805, 806, 807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 837, 838, 840, 841, 842, 846, 847, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 896, 897, 898, 899, 902, 903, 904, 905, 906, 907, 908, 909, 918, 919, 922, 923, 924, 926, 927, 928, 929, 939, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 977, 978, 979, 980, 985, 986, 991, 992, 993, 994, 995, 999, 1000, 1001, 1002, 1003, 1006, 1007, 1008, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1033, 1034, 1043, 1044, 1045, 1046, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1068, 1075, 1085, 1086, 1087, 1088, 1092, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1116, 1117, 1125, 1126, 1127, 1128, 1129, 1132, 1134, 1135, 1142, 1143, 1144, 1146, 1147, 1150, 1151, 1152, 1153, 1154, 1155, 1161, 1162, 1163, 1164, 1168, 1169, 1170, 1175, 1176, 1181, 1182, 1183, 1184, 1189, 1190, 1193, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1241, 1242, 1243, 1244, 1283, 1284, 1287, 1289, 1292, 1293, 1294, 1298, 1299, 1308, 1309, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1617, 1618, 1623, 1624, 1625, 1626, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723, 1724, 1725, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1756, 1757, 1758, 1759, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1787, 1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1823, 1824, 1825, 1826, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1854, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1914, 1915, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1938, 1939, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2004, 2009, 2010, 2011, 2012, 2021, 2022, 2023, 2024, 2025, 2026, 2029, 2030, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2125, 2126, 2127, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2148, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3399, 3400, 3401, 3404, 3414, 3442, 3457, 3459, 3461, 3463, 3487, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3576, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3587, 3588, 3593, 3594, 3595, 3596, 3598, 3599, 3600, 3601, 3604, 3605, 3611, 3614, 3623, 3624, 3625, 3626, 3628, 3629, 3630, 3634, 3635, 3636, 3637, 3643, 3644, 3646, 3649, 3650, 3652, 3653, 3654, 3655, 3656, 3658, 3659, 3660, 3662, 3663, 3664, 3665, 3666, 3667, 3668, 3670, 3671, 3672, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 3691, 3693, 3695, 3697, 3698, 3701, 3703, 3704, 3707, 3709, 3713, 3714, 3715, 3716, 3717, 3722, 3724, 3725, 3726, 3727, 3728, 3730, 3734, 3737, 3738, 3739, 3740, 3742, 3745, 3752, 3753, 3754, 3756, 3757, 3760, 3761, 3762, 3769, 3771, 3772, 3785, 3786, 3790, 3801, 3807, 3808, 3809, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3829, 3830, 3831, 3832, 3833, 3834, 3835, 3836, 3837, 3838, 3839, 3840, 3841, 3842, 3843, 3844, 3845, 3846, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3863, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 4366, 4367, 4368, 4369, 4370, 4371, 4372, 4373, 4374, 4375, 4376, 4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4427, 4428, 4429, 4430, 4431, 4432, 4433, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4453, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476, 
            ])
        )

        self.v.register_buffer(
            "right_half",
            torch.tensor([
                19, 20, 21, 22, 23, 24, 25, 26, 109, 110, 111, 112, 219, 220, 221, 222, 335, 336, 337, 338, 522, 523, 524, 525, 526, 527, 528, 529, 534, 535, 536, 537, 554, 555, 556, 557, 584, 585, 586, 587, 595, 596, 597, 598, 599, 600, 601, 602, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 634, 635, 636, 637, 640, 641, 642, 643, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 675, 676, 677, 678, 684, 685, 686, 687, 689, 690, 698, 699, 700, 701, 710, 711, 716, 717, 718, 719, 720, 721, 722, 741, 742, 743, 744, 749, 750, 751, 752, 776, 777, 778, 779, 780, 781, 782, 787, 788, 789, 790, 791, 792, 793, 794, 800, 801, 810, 811, 812, 813, 817, 818, 819, 820, 830, 831, 832, 833, 834, 835, 836, 839, 843, 844, 845, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 900, 901, 910, 911, 912, 913, 914, 915, 916, 917, 920, 921, 925, 930, 931, 932, 933, 934, 935, 936, 937, 938, 940, 941, 956, 957, 973, 974, 975, 976, 981, 982, 983, 984, 987, 988, 989, 990, 996, 997, 998, 1004, 1005, 1009, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1066, 1067, 1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1089, 1090, 1091, 1094, 1095, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1130, 1131, 1133, 1136, 1137, 1138, 1139, 1140, 1141, 1145, 1148, 1149, 1156, 1157, 1158, 1159, 1160, 1165, 1166, 1167, 1171, 1172, 1173, 1174, 1177, 1178, 1179, 1180, 1185, 1186, 1187, 1188, 1191, 1192, 1196, 1197, 1198, 1199, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1219, 1220, 1221, 1222, 1223, 1231, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1285, 1286, 1288, 1290, 1291, 1295, 1296, 1297, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1327, 1328, 1330, 1332, 1333, 1334, 1335, 1359, 1360, 1379, 1380, 1381, 1382, 1392, 1393, 1394, 1395, 1406, 1407, 1408, 1409, 1488, 1613, 1614, 1615, 1616, 1619, 1620, 1621, 1622, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1726, 1727, 1752, 1753, 1754, 1755, 1760, 1761, 1762, 1772, 1783, 1784, 1785, 1786, 1822, 1828, 1829, 1833, 1834, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1853, 1855, 1856, 1857, 1858, 1859, 1870, 1882, 1883, 1884, 1885, 1912, 1913, 1916, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1940, 1941, 1960, 1961, 1962, 1963, 1982, 1983, 1984, 1985, 2000, 2001, 2002, 2003, 2005, 2006, 2007, 2008, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2027, 2028, 2031, 2032, 2036, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2123, 2124, 2128, 2129, 2130, 2131, 2132, 2133, 2144, 2145, 2146, 2147, 2149, 2150, 2151, 2165, 2166, 2167, 2168, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2484, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2758, 2759, 2760, 2761, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2770, 2771, 2772, 2773, 2774, 2775, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2786, 2787, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2838, 2839, 2840, 2841, 2842, 2843, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2898, 2899, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2907, 2908, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2927, 2928, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086, 3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3222, 3223, 3248, 3249, 3275, 3276, 3277, 3278, 3281, 3282, 3283, 3284, 3285, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3496, 3497, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3506, 3507, 3508, 3509, 3510, 3511, 3512, 3513, 3514, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3530, 3531, 3532, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3540, 3541, 3542, 3543, 3544, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3552, 3553, 3554, 3555, 3556, 3557, 3558, 3559, 3560, 3561, 3562, 3563, 3564, 3565, 3566, 3567, 3568, 3569, 3570, 3571, 3572, 3573, 3574, 3575, 3585, 3586, 3589, 3590, 3591, 3592, 3597, 3602, 3603, 3606, 3607, 3608, 3609, 3610, 3612, 3613, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3627, 3631, 3632, 3633, 3638, 3639, 3640, 3641, 3642, 3645, 3647, 3648, 3651, 3657, 3661, 3668, 3669, 3674, 3675, 3682, 3683, 3684, 3686, 3687, 3688, 3689, 3690, 3692, 3694, 3696, 3699, 3700, 3702, 3704, 3705, 3706, 3708, 3710, 3711, 3712, 3718, 3719, 3720, 3721, 3723, 3729, 3731, 3732, 3733, 3735, 3736, 3741, 3743, 3744, 3746, 3747, 3748, 3749, 3750, 3751, 3755, 3758, 3759, 3763, 3764, 3765, 3766, 3767, 3768, 3770, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3930, 4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4536, 4537, 4538, 4539, 4540, 4541, 4542, 4543, 4544, 4545, 4546, 4547, 4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4570, 4571, 4572, 4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4583, 4584, 4585, 4586, 4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4723, 4724, 4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4738, 4739, 4740, 4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4759, 4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4800, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810, 4811, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4879, 4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887, 4888, 4889, 4890, 4891, 4892, 4893, 4894, 4895, 4896, 4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4909, 4910, 4911, 4912, 4913, 4914, 4915, 4916, 4917, 4918, 4919, 4920, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930, 4931, 4932, 4933, 4934, 4935, 4936, 4937, 4938, 4939, 4940, 4941, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4957, 4958, 4959, 4960, 4961, 4962, 4963, 4964, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022
            ])
        )

        # remove the intersection with neck from scalp and get the region for hair
        face_and_neck = torch.cat([self.v.face, self.v.neck]).unique()
        # get the intersection between scalp and face_and_neck
        uniques, counts = torch.cat([self.v.scalp, face_and_neck]).unique(return_counts=True)
        intersection = uniques[counts == 2]
        uniques, counts = torch.cat([self.v.scalp, intersection]).unique(return_counts=True)
        hair = uniques[counts == 1]
        self.v.register_buffer("hair", hair)

        # unions
        self.v.register_buffer("ears", torch.cat([self.v.right_ear, self.v.left_ear]))
        self.v.register_buffer("eyeballs", torch.cat([self.v.right_eyeball, self.v.left_eyeball]))
        self.v.register_buffer("irises", torch.cat([self.v.right_iris, self.v.left_iris]))
        self.v.register_buffer("left_eye", torch.cat([self.v.left_eye_region, self.v.left_eyeball]))
        self.v.register_buffer("right_eye", torch.cat([self.v.right_eye_region, self.v.right_eyeball]))
        self.v.register_buffer("eyelids", torch.cat([self.v.left_eyelid, self.v.right_eyelid]))
        self.v.register_buffer("lip_inside_ring", torch.cat([self.v.lip_inside_ring_upper, self.v.lip_inside_ring_lower, torch.tensor([1594, 2730])]))

        # remove the intersection with irises from eyeballs and get the region for scleras
        uniques, counts = torch.cat([self.v.eyeballs, self.v.irises]).unique(return_counts=True)
        intersection = uniques[counts == 2]
        uniques, counts = torch.cat([self.v.eyeballs, intersection]).unique(return_counts=True)
        sclerae = uniques[counts == 1]
        self.v.register_buffer("sclerae", sclerae)

        # skin
        skin_except = ["eyeballs", "hair", "lips_tight", "boundary"]
        if self.num_verts == 5083:
            skin_except.append("teeth")
        skin = self.get_vid_except_region(skin_except)
        self.v.register_buffer("skin", skin)

        # Correspondences for BiseNet semantic masks
        self.v.register_buffer(
            "sem_ear_left", # 588 vertices
            torch.tensor([19, 20, 21, 22, 23, 24, 25, 26, 522, 523, 524, 525, 526, 527, 528, 529, 534, 535, 536, 537, 554, 555, 556, 557, 584, 585, 586, 587, 595, 596, 597, 598, 599, 600, 601, 602, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 634, 635, 636, 637, 640, 641, 642, 643, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 675, 676, 677, 678, 684, 685, 686, 687, 689, 690, 698, 699, 700, 701, 710, 711, 716, 717, 718, 719, 720, 721, 722, 741, 742, 743, 744, 749, 750, 751, 752, 776, 777, 778, 779, 780, 781, 782, 787, 788, 789, 790, 791, 792, 793, 794, 800, 801, 810, 811, 812, 813, 817, 818, 819, 820, 830, 831, 832, 833, 834, 835, 836, 839, 843, 844, 845, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 890, 891, 892, 893, 894, 895, 900, 901, 910, 911, 912, 913, 914, 915, 916, 917, 920, 921, 925, 930, 931, 932, 933, 934, 935, 936, 937, 938, 940, 941, 973, 974, 975, 976, 981, 982, 983, 984, 987, 988, 989, 990, 996, 997, 998, 1004, 1005, 1009, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1066, 1067, 1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1089, 1090, 1091, 1094, 1095, 1097, 1098, 1099, 1100, 1102, 1103, 1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1130, 1131, 1133, 1136, 1137, 1138, 1139, 1140, 1141, 1145, 1156, 1157, 1158, 1159, 1160, 1165, 1166, 1167, 1171, 1172, 1173, 1174, 1177, 1178, 1179, 1180, 1185, 1186, 1187, 1188, 1191, 1192, 1196, 1197, 1198, 1199, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1214, 1215, 1219, 1220, 1221, 1222, 1223, 1231, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1282, 1285, 1286, 1288, 1290, 1291, 1295, 1296, 1297, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1327, 1328, 1330, 1332, 1333, 1334, 1335, 1488, 1760, 1761, 1762, 1772, 1783, 1784, 1785, 1786, 1822, 1828, 1829, 1833, 1834, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1853, 1855, 1856, 1859, 1870, 1882, 1883, 1884, 1885, 1912, 1913, 1916, 1929, 1930, 1935, 1936, 1937, 2027, 2028, 2208, 2209, 2250, 2255, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2646, 2647, 2648, 2649, 2650, 2651, 2652, 2653, 2654, 2655, 2656, 2657, 2658, 2659, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2667, 2668, 2669, 2670, 2671, 2672, 2673, 2674, 2675, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2685, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2977, 2978, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3031, 3032])
        )
        self.v.register_buffer(
            "sem_ear_right", # 588 vertices
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 223, 224, 225, 226, 227, 228, 229, 230, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 476, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 530, 531, 532, 533, 538, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 560, 561, 562, 593, 594, 726, 727, 783, 796, 1479, 1480, 1481, 1482, 1483, 1484, 1485, 1486, 1487, 1489, 1490, 1491, 1492, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1899, 1900, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1914, 1915, 1917, 1918, 1919, 1920, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1938, 1939, 1942, 1943, 1944, 1945, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1964, 1971, 1972])
        )
        self.v.register_buffer(
            "sem_eye_left", # 546 vertices
            torch.tensor([4477, 4478, 4479, 4480, 4481, 4482, 4483, 4484, 4485, 4486, 4487, 4488, 4489, 4490, 4491, 4492, 4493, 4494, 4495, 4496, 4497, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4508, 4509, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4528, 4529, 4530, 4531, 4532, 4533, 4534, 4535, 4536, 4537, 4538, 4539, 4540, 4541, 4542, 4543, 4544, 4545, 4546, 4547, 4548, 4549, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 4565, 4566, 4567, 4568, 4569, 4570, 4571, 4572, 4573, 4574, 4575, 4576, 4577, 4578, 4579, 4580, 4581, 4582, 4583, 4584, 4585, 4586, 4587, 4588, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4596, 4597, 4598, 4599, 4600, 4601, 4602, 4603, 4604, 4605, 4606, 4607, 4608, 4609, 4610, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4628, 4629, 4630, 4631, 4632, 4633, 4634, 4635, 4636, 4637, 4638, 4639, 4640, 4641, 4642, 4643, 4644, 4645, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4654, 4655, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679, 4680, 4681, 4682, 4683, 4684, 4685, 4686, 4687, 4688, 4689, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4698, 4699, 4700, 4701, 4702, 4703, 4704, 4705, 4706, 4707, 4708, 4709, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4719, 4720, 4721, 4722, 4723, 4724, 4725, 4726, 4727, 4728, 4729, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4738, 4739, 4740, 4741, 4742, 4743, 4744, 4745, 4746, 4747, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4759, 4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4800, 4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810, 4811, 4812, 4813, 4814, 4815, 4816, 4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832, 4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848, 4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4879, 4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887, 4888, 4889, 4890, 4891, 4892, 4893, 4894, 4895, 4896, 4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4909, 4910, 4911, 4912, 4913, 4914, 4915, 4916, 4917, 4918, 4919, 4920, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928, 4929, 4930, 4931, 4932, 4933, 4934, 4935, 4936, 4937, 4938, 4939, 4940, 4941, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4957, 4958, 4959, 4960, 4961, 4962, 4963, 4964, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022])
        )
        self.v.register_buffer(
            "sem_eye_right", # 546 vertices
            torch.tensor([3931, 3932, 3933, 3934, 3935, 3936, 3937, 3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4162, 4163, 4164, 4165, 4166, 4167, 4168, 4169, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188, 4189, 4190, 4191, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4200, 4201, 4202, 4203, 4204, 4205, 4206, 4207, 4208, 4209, 4210, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4228, 4229, 4230, 4231, 4232, 4233, 4234, 4235, 4236, 4237, 4238, 4239, 4240, 4241, 4242, 4243, 4244, 4245, 4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259, 4260, 4261, 4262, 4263, 4264, 4265, 4266, 4267, 4268, 4269, 4270, 4271, 4272, 4273, 4274, 4275, 4276, 4277, 4278, 4279, 4280, 4281, 4282, 4283, 4284, 4285, 4286, 4287, 4288, 4289, 4290, 4291, 4292, 4293, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4302, 4303, 4304, 4305, 4306, 4307, 4308, 4309, 4310, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4326, 4327, 4328, 4329, 4330, 4331, 4332, 4333, 4334, 4335, 4336, 4337, 4338, 4339, 4340, 4341, 4342, 4343, 4344, 4345, 4346, 4347, 4348, 4349, 4350, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4359, 4360, 4361, 4362, 4363, 4364, 4365, 4366, 4367, 4368, 4369, 4370, 4371, 4372, 4373, 4374, 4375, 4376, 4377, 4378, 4379, 4380, 4381, 4382, 4383, 4384, 4385, 4386, 4387, 4388, 4389, 4390, 4391, 4392, 4393, 4394, 4395, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 4404, 4405, 4406, 4407, 4408, 4409, 4410, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4427, 4428, 4429, 4430, 4431, 4432, 4433, 4434, 4435, 4436, 4437, 4438, 4439, 4440, 4441, 4442, 4443, 4444, 4445, 4446, 4447, 4448, 4449, 4450, 4451, 4452, 4453, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4461, 4462, 4463, 4464, 4465, 4466, 4467, 4468, 4469, 4470, 4471, 4472, 4473, 4474, 4475, 4476])                                                                                                                                                                                              
        )
        self.v.register_buffer(
            "sem_eyebrow_left", # 18 vertices
            torch.tensor([335, 336, 337, 338, 2177, 2178, 2566, 2567, 2575, 2578, 3080, 3153, 3154, 3157, 3158, 3159, 3705, 3712])
        )
        self.v.register_buffer(
            "sem_eyebrow_right", # 18 vertices
            torch.tensor([16, 17, 18, 27, 672, 673, 1429, 1430, 1438, 1441, 2045, 2134, 2135, 2138, 2139, 2140, 3863, 3868])                                                                                                                      
        )
        self.v.register_buffer(
            "sem_lip_lower", # 62 vertices
            torch.tensor([1576, 1577, 1578, 1579, 1582, 1583, 1729, 1730, 1758, 1773, 1774, 1775, 1776, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1802, 1803, 1804, 1826, 1848, 1849, 1850, 1865, 2712, 2713, 2714, 2715, 2718, 2719, 2844, 2845, 2869, 2880, 2881, 2882, 2883, 2890, 2891, 2892, 2894, 2895, 2896, 2897, 2898, 2905, 2906, 2907, 2928, 2937, 2938, 2939, 2948, 3503, 3504, 3506, 3509])   
        )
        self.v.register_buffer(
            "sem_lip_upper", # 64 vertices
            torch.tensor([1657, 1658, 1669, 1670, 1693, 1694, 1695, 1696, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1720, 1721, 1723, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1749, 1750, 1751, 2774, 2775, 2786, 2787, 2810, 2811, 2812, 2813, 2827, 2828, 2829, 2830, 2831, 2832, 2833, 2834, 2837, 2838, 2840, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2864, 2865, 2866, 3531, 3541, 3543, 3546])                                                                                                                                                                                                          
        )
        self.v.register_buffer(
            "sem_mouth_interior", # 128 vertices
            torch.tensor([1572, 1573, 1580, 1581, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1718, 1719, 1722, 1724, 1725, 1728, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1764, 1765, 1777, 1778, 1779, 1780, 1781, 1782, 1827, 1830, 1831, 1832, 1835, 1836, 1846, 1847, 1851, 1852, 1854, 1860, 1861, 1862, 2708, 2709, 2716, 2717, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2784, 2785, 2835, 2836, 2839, 2841, 2842, 2843, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2872, 2873, 2884, 2885, 2886, 2887, 2888, 2889, 2929, 2930, 2931, 2932, 2933, 2934, 2935, 2936, 2940, 2941, 2942, 2943, 2944, 2945, 3497, 3500, 3512, 3513, 3514, 3533, 3547, 3549])
        )
        self.v.register_buffer(
            "sem_nose", # 351 vertices
            torch.tensor([464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 697, 702, 703, 704, 738, 739, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 1379, 1380, 1381, 1382, 1392, 1393, 1394, 1395, 1406, 1407, 1408, 1409, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1643, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1677, 1678, 1679, 1680, 1681, 1682, 1683, 1684, 1685, 1756, 1757, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1823, 1824, 1895, 2052, 2053, 2054, 2055, 2056, 2057, 2059, 2064, 2065, 2066, 2067, 2068, 2116, 2117, 2163, 2164, 2169, 2170, 2171, 2172, 2173, 2174, 2192, 2193, 2194, 2195, 2220, 2221, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2753, 2754, 2755, 2756, 2757, 2758, 2760, 2762, 2763, 2764, 2765, 2766, 2767, 2768, 2769, 2794, 2795, 2796, 2797, 2798, 2799, 2800, 2801, 2802, 2867, 2868, 2909, 2910, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2973, 3087, 3088, 3089, 3090, 3091, 3092, 3094, 3099, 3100, 3101, 3102, 3103, 3143, 3144, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3380, 3501, 3508, 3516, 3518, 3521, 3526, 3527, 3534, 3542, 3548, 3551, 3552, 3553, 3560, 3561, 3563, 3564, 3571, 3573, 3575, 3576, 3585, 3586, 3590, 3591, 3597, 3602, 3606, 3607, 3608, 3609, 3610, 3612, 3613, 3617, 3618, 3621, 3622, 3627, 3633, 3639, 3640, 3641, 3642, 3647, 3648, 3651, 3657, 3661, 3668, 3669, 3807, 3808, 3810, 3811, 3813, 3814, 3816, 3817, 3818, 3819, 3820, 3821, 3822, 3825, 3826, 3829, 3830, 3831, 3834, 3836, 3837, 3838, 3839, 3841, 3842, 3843, 3844, 3845, 3846])
        )
        self.v.register_buffer(
            "sem_neck", # 378 vertices
            torch.tensor([8, 9, 10, 11, 12, 13, 14, 15, 109, 110, 111, 112, 193, 194, 219, 220, 221, 222, 232, 333, 372, 373, 374, 375, 462, 463, 496, 497, 539, 552, 553, 558, 559, 563, 564, 649, 650, 736, 737, 784, 785, 786, 795, 886, 887, 957, 1148, 1210, 1211, 1212, 1213, 1325, 1326, 1359, 1360, 1386, 1493, 1726, 1727, 1759, 1790, 1857, 1886, 1896, 1897, 1898, 1901, 1902, 1903, 1931, 1932, 1933, 1934, 1940, 1941, 1948, 1949, 1998, 1999, 2036, 2115, 2149, 2150, 2151, 2162, 2175, 2218, 2219, 2251, 2252, 2253, 2254, 2483, 2484, 2531, 2629, 2870, 2893, 2964, 2974, 2975, 2976, 2979, 2980, 2981, 3012, 3013, 3054, 3055, 3142, 3174, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236, 3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3245, 3246, 3247, 3248, 3249, 3250, 3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3263, 3264, 3265, 3266, 3267, 3268, 3269, 3270, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333, 3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3342, 3343, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3352, 3353, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3413, 3414, 3415, 3416, 3417, 3419, 3420, 3421, 3422, 3423, 3424, 3441, 3442, 3443, 3444, 3445, 3446, 3447, 3448, 3449, 3450, 3451, 3452, 3453, 3454, 3455, 3456, 3457, 3458, 3459, 3460, 3461, 3462, 3463, 3488, 3494, 3496, 3510, 3544, 3562, 3569, 3630, 3634, 3635, 3636, 3643, 3644, 3646, 3649, 3650, 3652, 3673, 3676, 3677, 3678, 3679, 3680, 3681, 3685, 3691, 3693, 3695, 3697, 3698, 3701, 3703, 3707, 3709, 3713, 3760])
        )
        self.v.register_buffer(
            "sem_skin", # 1736 vertices
            torch.tensor([195, 196, 231, 334, 411, 412, 445, 446, 447, 498, 499, 500, 501, 540, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 588, 589, 590, 591, 592, 603, 604, 605, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 638, 639, 644, 645, 646, 647, 648, 667, 668, 669, 670, 671, 674, 679, 680, 681, 682, 683, 688, 691, 692, 693, 694, 695, 696, 705, 706, 707, 708, 709, 712, 713, 714, 715, 723, 724, 725, 728, 729, 730, 731, 732, 733, 734, 735, 740, 745, 746, 747, 748, 753, 797, 798, 799, 802, 803, 804, 805, 806, 807, 808, 809, 814, 815, 816, 821, 822, 823, 824, 825, 826, 827, 828, 829, 837, 838, 840, 841, 842, 846, 847, 848, 864, 865, 877, 878, 879, 880, 881, 882, 883, 884, 885, 888, 889, 896, 897, 898, 899, 902, 903, 904, 905, 906, 907, 908, 909, 918, 919, 922, 923, 924, 926, 927, 928, 929, 939, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 977, 978, 979, 980, 985, 986, 991, 992, 993, 994, 995, 999, 1000, 1001, 1002, 1003, 1006, 1007, 1008, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1033, 1034, 1043, 1044, 1045, 1046, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1068, 1075, 1085, 1086, 1087, 1088, 1092, 1093, 1096, 1101, 1108, 1113, 1114, 1115, 1116, 1117, 1125, 1126, 1127, 1128, 1129, 1132, 1134, 1135, 1142, 1143, 1144, 1146, 1147, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1161, 1162, 1163, 1164, 1168, 1169, 1170, 1175, 1176, 1181, 1182, 1183, 1184, 1189, 1190, 1193, 1194, 1195, 1200, 1201, 1202, 1216, 1217, 1218, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1241, 1242, 1243, 1244, 1265, 1266, 1283, 1284, 1287, 1289, 1292, 1293, 1294, 1298, 1299, 1308, 1309, 1310, 1311, 1312, 1320, 1321, 1322, 1323, 1324, 1329, 1331, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1383, 1384, 1385, 1387, 1388, 1389, 1390, 1391, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1410, 1411, 1412, 1413, 1414, 1415, 1416, 1417, 1418, 1419, 1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1439, 1440, 1442, 1443, 1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1476, 1477, 1478, 1494, 1570, 1571, 1574, 1575, 1584, 1585, 1586, 1587, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1623, 1624, 1642, 1644, 1653, 1654, 1655, 1656, 1671, 1672, 1673, 1674, 1675, 1676, 1686, 1687, 1688, 1689, 1690, 1691, 1692, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1752, 1753, 1754, 1755, 1763, 1766, 1767, 1768, 1769, 1770, 1771, 1796, 1797, 1798, 1799, 1800, 1801, 1805, 1825, 1858, 1863, 1864, 1866, 1867, 1868, 1869, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1921, 1946, 1947, 1950, 1960, 1961, 1962, 1963, 1965, 1966, 1967, 1968, 1969, 1970, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2037, 2038, 2039, 2040, 2041, 2042, 2043, 2044, 2046, 2047, 2048, 2049, 2050, 2051, 2058, 2060, 2061, 2062, 2063, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086, 2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2113, 2114, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2136, 2137, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2165, 2166, 2167, 2168, 2176, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2222, 2223, 2224, 2225, 2226, 2227, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2284, 2285, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2297, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2306, 2307, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394, 2395, 2396, 2397, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2485, 2486, 2487, 2488, 2489, 2490, 2491, 2492, 2493, 2494, 2495, 2496, 2497, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2506, 2507, 2508, 2509, 2510, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2562, 2563, 2564, 2565, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2576, 2577, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2630, 2706, 2707, 2710, 2711, 2720, 2721, 2722, 2723, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2751, 2752, 2759, 2761, 2770, 2771, 2772, 2773, 2788, 2789, 2790, 2791, 2792, 2793, 2803, 2804, 2805, 2806, 2807, 2808, 2809, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2871, 2874, 2875, 2876, 2877, 2878, 2879, 2899, 2900, 2901, 2902, 2903, 2904, 2908, 2927, 2946, 2947, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2996, 3010, 3011, 3014, 3025, 3026, 3027, 3028, 3029, 3030, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3078, 3079, 3081, 3082, 3083, 3084, 3085, 3086, 3093, 3095, 3096, 3097, 3098, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3145, 3146, 3147, 3148, 3149, 3150, 3151, 3152, 3155, 3156, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3418, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3464, 3465, 3466, 3467, 3468, 3469, 3470, 3471, 3472, 3473, 3474, 3475, 3476, 3477, 3478, 3479, 3480, 3481, 3482, 3483, 3484, 3485, 3486, 3487, 3489, 3490, 3491, 3492, 3493, 3495, 3498, 3499, 3502, 3505, 3507, 3511, 3515, 3517, 3519, 3520, 3522, 3523, 3524, 3525, 3528, 3529, 3530, 3532, 3535, 3536, 3537, 3538, 3539, 3540, 3545, 3550, 3554, 3555, 3556, 3557, 3558, 3559, 3565, 3566, 3567, 3568, 3570, 3572, 3574, 3577, 3578, 3579, 3580, 3581, 3582, 3583, 3584, 3587, 3588, 3589, 3592, 3593, 3594, 3595, 3596, 3598, 3599, 3600, 3601, 3603, 3604, 3605, 3611, 3614, 3615, 3616, 3619, 3620, 3623, 3624, 3625, 3626, 3628, 3629, 3631, 3632, 3637, 3638, 3645, 3653, 3654, 3655, 3656, 3658, 3659, 3660, 3662, 3663, 3664, 3665, 3666, 3667, 3670, 3671, 3672, 3674, 3675, 3682, 3683, 3684, 3686, 3687, 3688, 3689, 3690, 3692, 3694, 3696, 3699, 3700, 3702, 3704, 3706, 3708, 3710, 3711, 3714, 3715, 3716, 3717, 3718, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3729, 3730, 3731, 3732, 3733, 3734, 3735, 3736, 3737, 3738, 3739, 3740, 3741, 3742, 3743, 3744, 3745, 3746, 3747, 3748, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3756, 3757, 3758, 3759, 3761, 3762, 3763, 3764, 3765, 3766, 3767, 3768, 3769, 3770, 3771, 3772, 3773, 3774, 3775, 3776, 3777, 3778, 3779, 3780, 3781, 3782, 3783, 3784, 3785, 3786, 3787, 3788, 3789, 3790, 3791, 3792, 3793, 3794, 3795, 3796, 3797, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3809, 3812, 3815, 3823, 3824, 3827, 3828, 3832, 3833, 3835, 3840, 3847, 3848, 3849, 3850, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3858, 3859, 3860, 3861, 3862, 3864, 3865, 3866, 3867, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3906, 3907, 3908, 3909, 3910, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3929, 3930])
        )


    def create_float_masks(self):
        """ Initialize float masks from the vertex index masks. """

        self.f = BufferContainer()
        for k, v_mask in self.v.items():
            f_mask = torch.zeros((self.num_verts), dtype=torch.float, device=v_mask.device)
            f_mask[v_mask] = 1.0
            self.f.register_buffer(k, f_mask)
    
    def construct_vid_table(self):
        self.vid_to_region = defaultdict(list)  # vertex id -> region name
        for region_name, v_mask in self.v:
            for v_id in v_mask:
                self.vid_to_region[v_id.item()].append(region_name)
    
    def process_face_mask(self, faces):
        
        face_masks = defaultdict(list)  # region name -> face id
        for f_id, f in enumerate(faces):
            counters = defaultdict(int)
            for v_id in f:
                for region_name in self.vid_to_region[v_id.item()]:
                    counters[region_name] += 1
            
            for region_name, count in counters.items():
                if count >= 3:  # create straight boundaries, with seams
                # if count > 1:  # create zigzag boundaries, no seams
                    face_masks[region_name].append(f_id)

        self.f = BufferContainer()
        for region_name, f_mask in face_masks.items():
            self.f.register_buffer(region_name, torch.tensor(f_mask, dtype=torch.long))
    
    def process_face_clusters(self, face_clusters):
        """ Construct a lookup table from face id to cluster id.
            
            cluster #0: background
            cluster #1: foreground
            cluster #2: faces in face_clusters[0]
            cluster #3: faces in face_clusters[1]
            ...
        """
        fid2cid = torch.ones(self.num_faces+1, dtype=torch.long)  # faces are always treated as foreground
        for cid, cluster in enumerate(face_clusters):
            try:
                fids = self.get_fid_by_region([cluster])
            except Exception as e:
                continue
            fid2cid[fids] = cid + 2  # reserve cluster #0 for the background and #1 for faces that do not belong to any cluster
        self.register_buffer("fid2cid", fid2cid)
    
    def process_vt_mask(self, faces, faces_t):
        vt_masks = defaultdict(list)  # region name -> vt id
        for f_id, (face, face_t) in enumerate(zip(faces, faces_t)):
            for v_id, vt_id in zip(face, face_t):
                for region_name in self.vid_to_region[v_id.item()]:
                    vt_masks[region_name].append(vt_id.item())

        self.vt = BufferContainer()
        for region_name, vt_mask in vt_masks.items():
            self.vt.register_buffer(region_name, torch.tensor(vt_mask, dtype=torch.long))
    
    def get_vid_by_region(self, regions, keep_order=False):
        """Get vertex indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            vid = torch.cat([self.v.get_buffer(k) for k in regions])
            if keep_order:
                return vid
            else:
                return vid.unique()
        else:
            return torch.tensor([], dtype=torch.long)
    
    def get_vid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.v.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the vertex indicies that are not included by regions
        vert_idx = torch.arange(0, self.num_verts, device=indices.device)
        combined = torch.cat((indices, vert_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]

    def get_fid_by_region(self, regions):
        """Get face indicies by regions"""
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            return torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            return torch.tensor([], dtype=torch.long)
    
    def get_fid_except_region(self, regions):
        if isinstance(regions, str):
            regions = [regions]
        if len(regions) > 0:
            indices = torch.cat([self.f.get_buffer(k) for k in regions]).unique()
        else:
            indices = torch.tensor([], dtype=torch.long)

        # get the face indicies that are not included by regions
        face_idx = torch.arange(0, self.num_faces, device=indices.device)
        combined = torch.cat((indices, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]
    
    def get_fid_except_fids(self, fids):
        # get the face indicies that are not included
        face_idx = torch.arange(0, self.num_faces, device=fids.device)
        combined = torch.cat((fids, face_idx))
        uniques, counts = combined.unique(return_counts=True)
        return uniques[counts == 1]
