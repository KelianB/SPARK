## Code: https://github.com/zhengyuf/IMavatar
# Modified/Adapted by: Shrisha Bharadwaj, Kelian Baert

import math
from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F

def img_IoU(img1: Tensor, img2: Tensor, label_colors: List[torch.Tensor], mask=None) -> Tensor:    
    # Make mask (B,H,W)
    if mask is not None and mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    B = img1.shape[0]
    device = img1.device
    ious = torch.zeros((B, len(label_colors)), dtype=torch.float, device=device)
    eps = 1e-5
    for class_index, color in enumerate(label_colors):
        color = color.to(device).unsqueeze(0).unsqueeze(0) # (1,1,3)
        mask1 = (img1 - color).abs().sum(-1) < eps # (B,H,W)
        mask2 = (img2 - color).abs().sum(-1) < eps # (B,H,W)
        if mask is not None:
            mask1 = mask1 * mask
            mask2 = mask2 * mask
        intersection = torch.count_nonzero((mask1.int() + mask2.int()) == 2, dim=(1,2)) # (B)
        union = torch.count_nonzero(mask1.int() + mask2.int(), dim=(1,2)) # (B)
        # This formulation will output 1 when both masks are empty (i.e. the prediction is correct)
        iou = (intersection + eps) / (union + eps) # (B)
        # print("intersection", intersection, "\nunion", union, "\niou", iou)
        ious[:, class_index] = iou
    return ious

def img_mse(pred, gt, mask=None, error_type='mse', return_all=False, use_mask=False):
    """
    MSE and variants
    Input:
        pred        :  bsize x 3 x h x w
        gt          :  bsize x 3 x h x w
        error_type  :  'mse' | 'rmse' | 'mae' | 'L21'
    MSE/RMSE/MAE between predicted and ground-truth images.
    Returns one value per-batch element
    pred, gt: bsize x 3 x h x w
    """
    assert pred.dim() == 4
    # Ensure shape is BCHW
    if pred.shape[1] > 4:
        pred = pred.permute(0,3,1,2)
        gt = gt.permute(0,3,1,2)
    if mask is not None and mask.shape[1] > 4:
        mask = mask.permute(0,3,1,2)

    bsize = pred.size(0)

    if error_type == 'mae':
        all_errors = (pred-gt).abs()
    elif error_type == 'L21':
        all_errors = torch.norm(pred-gt, dim=1)
    elif error_type == "L1":
        all_errors = torch.norm(pred - gt, dim=1, p=1)
    else:
        all_errors = (pred-gt).square()

    if mask is not None and use_mask:
        assert mask.size(1) == 1

        nc = pred.size(1)
        # nnz = torch.sum(mask.reshape(bsize, -1), 1) * nc
        nnz = torch.sum(torch.ones_like(mask.reshape(bsize, -1)), 1) * nc
        all_errors = mask.expand(-1, nc, -1, -1) * all_errors
        errors = all_errors.reshape(bsize, -1).sum(1) / nnz
    else:
        errors = all_errors.reshape(bsize, -1).mean(1)

    if error_type == 'rmse':
        errors = errors.sqrt()

    if return_all:
        return errors, all_errors
    else:
        return errors

def img_psnr(pred, gt, mask=None, rmse=None):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    if torch.max(pred) > 128:   max_val = 255.
    else:                       max_val = 1.

    if rmse is None:
        rmse = img_mse(pred, gt, mask, error_type='rmse', use_mask=mask is not None)

    EPS = 1e-8
    return 20 * torch.log10(max_val / (rmse+EPS))

def _gaussian(w_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
    return gauss/gauss.sum()

def _create_window(w_size, channel=1):
    _1D_window = _gaussian(w_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
    return window

def img_ssim(pred, gt, w_size=11, full=False):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    
    # Ensure shape is BCHW
    if pred.shape[1] > 4:
        pred = pred.permute(0,3,1,2)
        gt = gt.permute(0,3,1,2)

    if torch.max(pred) > 128:   max_val = 255
    else:                       max_val = 1

    if torch.min(pred) < -0.5:  min_val = -1
    else:                       min_val = 0

    L = max_val - min_val

    padd = 0
    (_, channel, height, width) = pred.size()
    window = _create_window(w_size, channel=channel).to(pred.device)

    mu1 = F.conv2d(pred, window, padding=padd, groups=channel)
    mu2 = F.conv2d(gt, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if False:
    # if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret
