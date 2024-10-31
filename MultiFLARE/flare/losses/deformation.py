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

"""
This is the original FLAME regularization using the nearest neighbor.
It is not used anymore. 
"""

def flame_regularization(flame, lbs_weights, shapedirs, posedirs, canonical_vertices, 
                         iteration, gbuffer, views_subset, loss_function = torch.nn.MSELoss(reduction='none')):

    # since the canonical mesh is not initialized, we wait for a small warm-up iteration before applying specific mask based loss
    if iteration > 300: 
        mouth_interior_canonical_points = gbuffer["canonical_position"] * views_subset["semantic_mask"][..., SemanticMask.MOUTH_INTERIOR].unsqueeze(-1)
        nonZeroRows = torch.abs(mouth_interior_canonical_points.view(-1, 3)).sum(dim=1) > 0
        mouth_interior_c_pts = mouth_interior_canonical_points.view(-1, 3)[nonZeroRows]

        cloth_canonical_points = gbuffer["canonical_position"] * views_subset["semantic_mask"][..., SemanticMask.CLOTH_NECKLACE].unsqueeze(-1)
        nonZeroRows_cloth = torch.abs(cloth_canonical_points.view(-1, 3)).sum(dim=1) > 0
        cloth_c_pts = cloth_canonical_points.view(-1, 3)[nonZeroRows_cloth]
        c_pts_masked = [mouth_interior_c_pts, cloth_c_pts]
    else:
        c_pts_masked = None

    loss = 0.0
    num_points = lbs_weights.shape[0]
    flame_shapedirs, flame_posedirs, flame_lbs_weights, flame_distances = flame.blendshapes_nearest(canonical_vertices, c_pts_masked)

    # Hyperparameter taken from IMavatar
    thresh = 0.001
    flame_mask = (flame_distances.squeeze(0).squeeze(-1) < thresh)
    if flame_mask.sum() == 0:
        return 0.0, {"gt_shapedirs": flame_shapedirs, "gt_posedirs": flame_posedirs, "gt_lbs": flame_lbs_weights} 

    loss_lbs = loss_function(flame_lbs_weights.reshape(num_points, -1)[flame_mask, :], lbs_weights.reshape(num_points, -1)[flame_mask, :]).mean()
    loss += loss_lbs * 0.5
    loss_shapedir = loss_function(flame_shapedirs.reshape(num_points, -1)[flame_mask, :] * 10., shapedirs.reshape(num_points, -1)[flame_mask, :] * 10.).mean() 
    loss += loss_shapedir * 10.
    loss_posedir = loss_function(flame_posedirs.reshape(num_points, -1)[flame_mask, :] * 10., posedirs.reshape(num_points, -1)[flame_mask, :] * 10.).mean() 
    loss += loss_posedir * 10.
    
    return loss, {"gt_shapedirs": flame_shapedirs, "gt_posedirs": flame_posedirs, "gt_lbs": flame_lbs_weights} 