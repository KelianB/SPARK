# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

# Modified/Adapted by: Kelian Baert

import os

from flare.modules.fc import FC
from flare.modules.embedder import get_embedder
from flare.modules.embedding_roughness_np import generate_ide_fn
from flare.core.mesh import dot, reflect, safe_normalize
import numpy as np
import torch
import tinycudann as tcnn
import nvdiffrast.torch as dr
import logging


class NeuralShader(torch.nn.Module):

    def __init__(self,
                 aabb=None,
                 activation='relu',
                 material_embedding='positional',
                 material_mlp_dims=None,
                 material_last_activation=None,
                 light_mlp_activation='none',
                 light_mlp_dims=None,
                 multi_sequence_lighting='single',
                 num_seq=1,
                 progressive_hash=False,
                 progressive_hash_iters=None,
                 hash_include_input=False,
                 hash_max_resolution=4096,
                 hash_levels=16,
                 hash_log2_size=19,
                 device='cpu'
                ):

        super().__init__()
        self.device = device
        self.aabb = aabb
        self.material_input_dims = 3
        self.progressive_hash = progressive_hash
        self.hash_include_input = hash_include_input

        # Store the config
        self._config = {
            "aabb": aabb,
            "activation": activation,
            "material_embedding": material_embedding,
            "material_mlp_dims": material_mlp_dims,
            "material_last_activation": material_last_activation,
            "light_mlp_activation": light_mlp_activation,
            "light_mlp_dims": light_mlp_dims,
            "multi_sequence_lighting": multi_sequence_lighting,
            "num_seq": num_seq,
            "progressive_hash": progressive_hash,
            "progressive_hash_iters": progressive_hash_iters,
            "hash_include_input": hash_include_input,
            "hash_max_resolution": hash_max_resolution,
            "hash_levels": hash_levels,
            "hash_log2_size": hash_log2_size,
        }

        self._FG_LUT = torch.as_tensor(np.fromfile("assets/bsdf_256_256.bin", dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device=device)

        # Positional Encoding
        if material_embedding == "positional":
            logging.info("Using fourier positional encoding for intrinsic materials")
            self.embed_fn, channels = get_embedder(multires=4, n_input_dims=self.material_input_dims)
            self.inp_size = channels
        elif material_embedding == "hashgrid":
            logging.info("Using hashgrid (tinycudann) for intrinsic materials")
            # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
            desired_resolution = hash_max_resolution
            num_levels = hash_levels
            log2_hashmap_size = hash_log2_size
            base_grid_resolution = 16

            per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))
            enc_cfg =  {
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": 2,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_grid_resolution,
                "per_level_scale" : per_level_scale
            }

            if progressive_hash:
                def apply_progressive_hash(points_enc, iter):
                    # see https://github.com/NVlabs/neuralangelo/blob/94390b64683c067c620d9e075224ccfe582647d0/projects/neuralangelo/utils/modules.py#L92
                    active_levels = sum(1 for it in progressive_hash_iters if iter >= it)
                    mask = torch.zeros_like(points_enc)
                    mask[..., :(active_levels * enc_cfg["n_features_per_level"])] = 1
                    return points_enc * mask
                self.apply_progressive_hash = apply_progressive_hash

            self.embed_fn = tcnn.Encoding(self.material_input_dims, enc_cfg).to(device)
            self.inp_size = self.embed_fn.n_output_dims
            if hash_include_input:
                self.inp_size += self.material_input_dims

        # Note: the tonemapping that is used to compute the final image clamps the color values.
        # Outside of the clamp range, the gradients are erased. As a result, if the lighting MLP begins to predict values outside of this
        # range (e.g., negative values) in early iterations, it will get completely stuck (and the final image will look black).
        # To avoid this, we can run our own activation here to soft clamp the color values beforehand.
        if light_mlp_activation == "none":
            self.light_mlp_activation = torch.nn.Identity()
        elif light_mlp_activation == 'softplus':
            self.light_mlp_activation = torch.nn.Softplus(beta=100.0)
        else:
            raise ValueError(f"Unknown activation fn: '{light_mlp_activation}'")
        
        # ==============================================================================================
        # create MLP
        # ==============================================================================================
        self.material_mlp = FC([self.inp_size, *material_mlp_dims, 5], activation, material_last_activation).to(device) #sigmoid
        
        self.light_mlp = FC([38, *light_mlp_dims, 3], activation=activation, last_activation=self.light_mlp_activation, bias=True).to(device) 
        self.dir_enc_func = generate_ide_fn(deg_view=4, device=self.device)

        self.per_sequence_lighting = False

        if num_seq == 1 or multi_sequence_lighting == "single":
            # No change (default behavior, one single lighting MLP)
            pass
        elif multi_sequence_lighting == "mlp_per_sequence":
            # Use an entirely different light MLP for each sequence
            self.per_sequence_lighting = True
            self.light_mlp = torch.nn.ModuleList([FC([38, *light_mlp_dims, 3], activation=activation, last_activation=self.light_mlp_activation, bias=True).to(device)
                    for _ in range(num_seq)])
        elif multi_sequence_lighting == "mlp_per_sequence_share":
            # Use different light MLPs but share some layers
            self.per_sequence_lighting = True
            light_common = FC([38, *light_mlp_dims], activation=activation, last_activation=torch.nn.ReLU(inplace=True), bias=True).to(device)
            self.light_mlp = torch.nn.ModuleList([
                    torch.nn.Sequential(light_common, FC([light_mlp_dims[-1], light_mlp_dims[-1], 3], activation=torch.nn.ReLU(inplace=True), last_activation=self.light_mlp_activation, bias=True).to(device))
                    for _ in range(num_seq)])
        else:
            raise ValueError(f"Unsupported multi_sequence_lighting '{multi_sequence_lighting}'")

        if self.per_sequence_lighting:
            if multi_sequence_lighting in ["mlp_per_sequence", "mlp_per_sequence_share"]:
                def shading_fn_per_seq(seq_idx, iter):
                    def fn(t: torch.Tensor):
                        # t.shape: (bz*h*w, 38)
                        batched_t = t.view(len(seq_idx), -1, 38) # (bz, h*w, 38)
                        return torch.cat([self.light_mlp[seqi](batched_t[i]) for i,seqi in enumerate(seq_idx)], dim=0) # (bz*h*w, 3)
                    return fn
            self.shading_fn_per_seq = shading_fn_per_seq

    """ Compute material properties (albedo, roughness, specular intensity) """
    def compute_material(self, position, iteration):
        assert position.ndim == 4
        B, H, W, _ = position.shape
        pe_input = self.apply_pe(position=position, iteration=iteration)
        all_tex = self.material_mlp(pe_input.view(-1, self.inp_size)) 
        albedo = all_tex[..., :3].view(B, H, W, -1) 
        roughness = all_tex[..., 3:4].view(B, H, W, 1)
        spec_int = all_tex[..., 4:5].view(B, H, W, 1)
        return albedo, roughness, spec_int

    """ Compute the shaded color (adapted from nvdiffrec and FLARE). """
    def forward(self, gbuffers, views, mesh, iteration, texture=None):
        cam_pos = torch.cat([v.center.unsqueeze(0) for v in views['camera']], dim=0)
        cam_pos = cam_pos[:, None, None, :]
        deformed_pos = gbuffers["position"]
        normals = gbuffers["shading_normals"] if "shading_normals" in gbuffers else self.get_shading_normals(deformed_pos, cam_pos, gbuffers, mesh)
        shading_fn = self.shading_fn_per_seq(views["seq_idx"], iteration) if self.per_sequence_lighting else self.light_mlp

        position = gbuffers["canonical_position"]

        B, H, W, _ = position.shape
        device = position.device

        # Skin mask for fresnel coefficient
        # skin_mask = (torch.sum(views["semantic_mask"][..., (SemanticMask.ALL_SKIN, SemanticMask.EYES, SemanticMask.EYEBROWS)], axis=-1)).unsqueeze(-1)
        # skin_mask = skin_mask * views["mask"] 
        # skin_mask = (skin_mask > 0.0).int().bool()
        skin_mask = None
        if skin_mask is not None:
            fresnel_constant = torch.ones((B, H, W, 1), dtype=torch.float, device=device) * 0.047
            fresnel_constant[skin_mask] = 0.028
        else:
            fresnel_constant = 0.04

        # Compute material properties (albedo, roughness, specular intensity)
        albedo, roughness, spec_int = self.compute_material(position, iteration)

        # Optionally override the albedo with a texture
        if texture is not None:
            # texture: (B, th, tw, 3)
            uvs = gbuffers["uvs"] * 2 - 1 # (B, H, W, 2)
            uvs[..., 1] = -uvs[..., 1]
            albedo = torch.nn.functional.grid_sample(texture.permute(0,3,1,2), uvs, mode="bilinear", padding_mode="border").permute(0,2,3,1) # (B, H, W, 3|4)
            # Handle alpha
            if albedo.shape[-1] == 4:
                albedo, alpha = albedo[..., :3], albedo[..., (3,)]
                gbuffers["mask"] *= alpha  

        #################### Diffuse shading ####################
        # Calculate Integrated Differential Encoding for the diffuse shading
        ide_diffuse = self.dir_enc_func(normals.view(-1, 3), torch.ones_like(roughness).view(-1, 1)) # (B,H,W,38)
        diffuse = shading_fn(ide_diffuse).view(B, H, W, 3)

        #################### Specular shading ####################
        # Reflect the camera direction on the normal vector 
        wo = safe_normalize(cam_pos - deformed_pos)
        wr = safe_normalize(reflect(wo, normals))
        # Calculate Integrated Differential Encoding for the specular shading
        ide_specular = self.dir_enc_func(wr.view(-1, 3), roughness.view(-1, 1))
        specular = shading_fn(ide_specular).view(B, H, W, 3)
        # Compute the FG term using the LUT
        fg_uv = torch.cat((dot(wo, normals).clamp(min=1e-4), roughness), dim=-1)
        # Sample the (256, 256, 2) FG-LUT using wo.n_d and roughness
        fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode="linear", boundary_mode="clamp")
        reflectance = fresnel_constant * fg_lookup[...,0:1] + fg_lookup[...,1:2]

        #################### Final shading ####################
        color = diffuse*albedo + spec_int*specular*reflectance # (B, H, W, 3)

        # Mask using the alpha values from the rasterizer
        pred_color_masked = torch.cat([color, torch.ones_like(color[..., 0:1]).to(self.device)], dim=-1) * gbuffers["mask"]
        albedo_masked = albedo * gbuffers["mask"]
        roughness_masked = roughness * gbuffers["mask"]
        spec_int_masked = spec_int * gbuffers["mask"]
        # Antialias the final color
        pred_color_masked = dr.antialias(pred_color_masked.contiguous(), gbuffers["rast"], gbuffers["deformed_verts_clip_space"], mesh.indices.int())
        roughness_masked = dr.antialias(roughness_masked.contiguous(), gbuffers["rast"], gbuffers["deformed_verts_clip_space"], mesh.indices.int())
        spec_int_masked = dr.antialias(spec_int_masked.contiguous(), gbuffers["rast"], gbuffers["deformed_verts_clip_space"], mesh.indices.int())

        cbuffers = {
            "diffuse": diffuse,
            "albedo": albedo_masked,
            "roughness": roughness_masked,
            "spec_int": spec_int_masked,
        }
        return pred_color_masked[..., :3], cbuffers, pred_color_masked[..., -1:]

    def get_shading_normals(self, position, view_pos, gbuffers, mesh):
        """ Flip backward-facing normals """

        view_vec = safe_normalize(view_pos - position)
        vert_normals = safe_normalize(gbuffers["vertex_normals"])
        face_normals = safe_normalize(gbuffers["face_normals"])

        # Swap direction of backfacing normals
        frontfacing_px = dot(face_normals, view_vec) > 0
        vert_normals = torch.where(frontfacing_px, vert_normals, -vert_normals)
        face_normals = torch.where(frontfacing_px, face_normals, -face_normals)

        # This is taken from nvdiffrec
        NORMAL_THRESHOLD = 0.1
        t = torch.clamp(dot(view_vec, vert_normals) / NORMAL_THRESHOLD, min=0, max=1)
        normal = torch.lerp(face_normals, vert_normals, t)

        gbuffers["normal"] =  dr.antialias(normal.contiguous(), gbuffers["rast"], gbuffers["deformed_verts_clip_space"], mesh.indices.int())
        return gbuffers["normal"]
    
    def apply_pe(self, position, iteration=None):
        position = position.view(-1, self.material_input_dims)
        # Normalize input
        position = (position - self.aabb[0]) / (self.aabb[1] - self.aabb[0])
        position = torch.clamp(position, min=0, max=1)
        pe_input = self.embed_fn(position.contiguous()).float()
        if self.progressive_hash:
            pe_input = self.apply_progressive_hash(pe_input, iteration)
        if self.hash_include_input:
            pe_input = torch.cat([position, pe_input], dim=-1)
        return pe_input

    @classmethod
    def revive(cls, path, device='cpu'):
        assert os.path.exists(path)
        data = torch.load(path, map_location=device)
        shader = cls(**data['config'], device=device)
        shader.load_state_dict(data['state_dict'], strict=False)
        return shader

    def load(self, path):
        assert os.path.exists(path)
        data = torch.load(path, map_location=self.device)
        self.load_state_dict(data['state_dict'], strict=False)

    def save(self, path):
        data = {
            'version': 2,
            'config': self._config,
            'state_dict': self.state_dict()
        }
        torch.save(data, path)

    def clone(self, device):
        new_shader = NeuralShader(**self._config, device=device)
        new_shader.load_state_dict(self.state_dict(), strict=False)
        return new_shader
