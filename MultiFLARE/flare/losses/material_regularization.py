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

import torch

from utils.dataset import SemanticMask

def albedo_regularization(_adaptive, shader, mesh, device, iteration):
    position = mesh.vertices
    pe_input = shader.apply_pe(position=position, iteration=iteration)
    val = shader.material_mlp(pe_input)[..., :4]

    # add jitter for loss function
    jitter_pos = position + torch.normal(mean=0, std=0.01, size=position.shape, device=device)
    jitter_pe_input = shader.apply_pe(position=jitter_pos, iteration=iteration)

    val_jitter = shader.material_mlp(jitter_pe_input)[..., :4]
    loss_fn = torch.nn.MSELoss(reduction='none')
    loss = loss_fn(val_jitter, val)
    loss = torch.mean(_adaptive.lossfun(loss.view(-1, 4)))
    return loss * min(1.0, iteration / 500)


def white_light(cbuffers):
    diffuse = cbuffers["diffuse"]
    white = diffuse.mean(dim=-1, keepdim=True) # (R+G+B)/3
    masked_pts = (diffuse - white)
    loss = torch.mean(torch.abs(masked_pts))
    return loss

def roughness_regularization(roughness, semantic, mask, r_mean):
    skin_mask = (torch.sum(semantic[..., (SemanticMask.ALL_SKIN, SemanticMask.EYES, SemanticMask.EYEBROWS)], axis=-1)).unsqueeze(-1)
    skin_mask = skin_mask * mask 

    loss = 0.0
    mask = (skin_mask > 0.0).int().bool()
    roughness_skin = roughness[mask]
    # Ablation tabulated in Section 5.
    mean = r_mean # 0.5 default
    std = 0.100
    z_score = (roughness_skin-mean) / std

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss

def spec_intensity_regularization(rho, semantic, mask):
    skin_mask = (torch.sum(semantic[..., (SemanticMask.ALL_SKIN, SemanticMask.EYES, SemanticMask.EYEBROWS)], axis=-1)).unsqueeze(-1)
    skin_mask = skin_mask * mask 

    loss = 0.0
    mask = (skin_mask > 0.0).int().bool()
    rho_skin = rho[mask]
    # pre-computed
    mean = 0.3753
    std = 0.1655
    z_score = (rho_skin-mean) / std

    loss = torch.mean(torch.max(torch.zeros_like(z_score), (torch.abs(z_score) - 2)))
    return loss