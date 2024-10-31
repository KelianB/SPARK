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
from utils.color import rgb_to_srgb

#----------------------------------------------------------------------------
# HDR image losses
#----------------------------------------------------------------------------

def image_loss_fn(img, target, reduction="mean", weight_map=None):
    img    = rgb_to_srgb(torch.log(torch.clamp(img, min=0, max=65535) + 1))
    target = rgb_to_srgb(torch.log(torch.clamp(target, min=0, max=65535) + 1))

    if weight_map is not None:
        img, target = img * weight_map, target * weight_map

    out = torch.nn.functional.l1_loss(img, target, reduction=reduction)
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of image_loss contains inf or NaN"
    return out, img, target

def mask_loss(masks, gbuffers, loss_function = torch.nn.MSELoss()):
    """ Compute the mask term as the mean difference between the original masks and the rendered masks.
    
    Args:
        views (List[View]): Views with masks
        gbuffers (List[Dict[str, torch.Tensor]]): G-buffers for each view with the 'mask' channel
        loss_function (Callable): Function for comparing the masks or generally a set of pixels
    """

    loss = 0.0
    for gt_mask, gbuffer_mask in zip(masks, gbuffers):
        loss += loss_function(gt_mask, gbuffer_mask)
    return loss / len(masks)

def shading_loss_batch(pred_color_masked, img, mask, weight_map=None):
    """ Compute the image loss term as the mean difference between the original images and the rendered images from a shader. """
    color_loss, tonemap_pred, tonemap_target = image_loss_fn(pred_color_masked[..., :3] * mask, img * mask, weight_map=weight_map)
    return color_loss, pred_color_masked[..., :3], [tonemap_pred, tonemap_target]
 
def shading_loss_batch_framewise(pred_color_masked, img, mask):
    """ This is the same as shading_loss_batch, but we keep a value per frame. """
    color_loss, _, _ = image_loss_fn(pred_color_masked[..., :3] * mask, img * mask, reduction="none")
    return color_loss.mean((1,2,3))
