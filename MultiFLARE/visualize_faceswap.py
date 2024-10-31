import os
import sys
import logging
from pathlib import Path
from typing import List

from arguments import config_parser
from dataset import MultiVideoDataset 
from utils.color import rgb_to_srgb
from utils.visualization import save_img
from utils.dataset import DeviceDataLoader
from Avatar import Avatar

import argparse
import torch
from tqdm import tqdm

def blend(img1, img2, mask):
    """ Blend img1 onto img2 with given alpha mask (bool or float)"""
    return img1 * mask.float() + img2 * (1 - mask.float())

def gaussian_kernel(ksize: int, sigma: float):
    x = torch.arange(0, ksize) - ksize // 2
    gauss = torch.exp(-x.pow(2) / (2 * sigma * sigma))
    return gauss / gauss.sum()

def apply_featurewise_conv1d(signal: torch.Tensor, kernel: torch.Tensor, pad_mode="replicate") -> torch.Tensor:
    # signal: (N, n_features)
    # kernel: (kernel_size)
    _, n_features = signal.shape
    kernel_size = kernel.shape[0]
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(n_features,1,1) # (n_features,1,kernel_size) (we want as many output channels as features)
    signal = signal.permute(1,0).unsqueeze(0) # (1, n_features, N)
    # Pad input signal
    padding = kernel_size // 2 # to maintain the original signal length
    padded_signal = torch.nn.functional.pad(signal, (padding, padding), mode=pad_mode)
    # Perform convolution
    filtered_signal = torch.nn.functional.conv1d(padded_signal, kernel, groups=n_features) # (1, n_features, N)
    return filtered_signal.squeeze(0).permute(1,0) # (N, n_features)

@torch.no_grad()
def main(args, avatar_a: Avatar, avatar_b: Avatar):
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = avatar_a.device

    frame_interval = 1
    hide_neck = True

    fps = 24 // frame_interval

    dataset_a, shader_a = avatar_a.dataset_train, avatar_a.shader
    dataset_b, pose_b, expr_b, cams_K_b, shader_b = avatar_b.dataset_train, avatar_b.pose_train, avatar_b.expr_train, avatar_b.cams_K_train, avatar_b.shader

    view_indices = dataset_b.frames_per_seq[args.sequence][::frame_interval]

    conv_weights = gaussian_kernel(ksize=9, sigma=2).to(device)
    # pose_b[view_indices] = apply_featurewise_conv1d(pose_b[view_indices], conv_weights, pad_mode="replicate")
    # pose_b = apply_featurewise_conv1d(pose_b, conv_weights, pad_mode="replicate")
    pose_b[view_indices] = apply_featurewise_conv1d(pose_b[view_indices], conv_weights, pad_mode="replicate")
    expr_b[view_indices] = apply_featurewise_conv1d(expr_b[view_indices], conv_weights, pad_mode="replicate")

    dataset_b_subset = torch.utils.data.Subset(dataset_b, view_indices)
    dataloader_b = DeviceDataLoader(dataset_b_subset, device=device, batch_size=8, collate_fn=dataset_a.collate, shuffle=False, drop_last=False, num_workers=4)

    # Replace material MLP of B with material MLP of A - doesn't work
    # shader = shader_b
    # shader.material_mlp = shader_a.material_mlp
    # Replace light MLP of A with light MLP of B
    shader = shader_a
    shader.light_mlp = shader_b.light_mlp

    for views_b in tqdm(dataloader_b):
        # Use pose, expression and camera from sequence B
        MultiVideoDataset.override_values_batch(views_b, pose_b, expr_b, cams_K_b)
        # Render A with lighting from B
        rgb_imgs, gbuffers, _, _, _, _, gbuffer_mask, *_ = avatar_a.run(views_b, avatar_a.args.resume, shader=shader)            
        
        masks = rasterize_semantic_masks(avatar_a, gbuffers["rast"], ["sem_neck", "sem_eye_left", "sem_eye_right", "sem_mouth_interior"])

        display_mask = gbuffer_mask
        if hide_neck:
            display_mask *= (masks["sem_neck"] == 0).float()
        if args.keep_eyes:
            display_mask *= (1 - masks["sem_eye_left"]) * (1 - masks["sem_eye_right"])
        if args.keep_mouth:
            display_mask *= 1 - masks["sem_mouth_interior"]

        # Soften the mask a little bit
        # from torchvision.transforms import GaussianBlur
        # display_mask = GaussianBlur(9)(display_mask.permute(0,3,1,2)).permute(0,2,3,1)

        rgb_imgs = blend(rgb_imgs, views_b["img"], display_mask)

        # Normal map in camera space
        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
        normal_imgs = (0.5*(gbuffers["normal"] @ views_b["camera"][0].R.T @ R.T + 1)) * display_mask

        for (vidx, original_img, render, normal_img) in zip(views_b["idx"], views_b["img"], rgb_imgs, normal_imgs):
            img = torch.cat((
                    rgb_to_srgb(original_img),
                    rgb_to_srgb(render),
                    normal_img
                ), 1) # concatenate along width dimension
            save_img(out_dir / f"{vidx:04d}.png", img)

    logging.info("Creating video")
    os.system(f"/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i '{out_dir / '*.png'}' -c:v libx264 -pix_fmt yuv420p {out_dir / 'video.mp4'}")
    logging.info("Done")

def get_prefixed_args(args: List[str], prefix: str):
    """ Extract all command line arguments with the given prefix and remove that prefix. """
    extracted_args = []
    for i in range(len(args)):
        s = args[i]
        if s.startswith("-" + prefix):
            extracted_args += ["-" + s[len(prefix)+1:]]
        elif s.startswith("--" + prefix):
            extracted_args += ["--" + s[len(prefix)+2:], args[i+1]]
            i += 1
    return extracted_args

def rasterize_semantic_masks(avatar, rast, semantic_keys):
    flame = avatar.flame
    device = rast.device
    B, H, W, _ = rast.shape
    if not semantic_keys:
        return torch.zeros((B, 1, H, W), dtype=torch.float, device=device)

    bu, bv, _, triangle_ids = rast.flatten(1, 2).unbind(-1) # (B, H*W)
    # if triangle_id is zero, then no triangle was rendered here, otherwise we need to offset the ID by one
    triangle_ids = triangle_ids.long() - 1 # (B, W*H)

    pixel_masks = dict()
    for key in semantic_keys:
        vert_mask = getattr(flame.mask.f, key).to(device)
        triangle_mask = torch.stack([vert_mask[v] for v in avatar.canonical_mesh.indices[triangle_ids]]) # (B, H*W, 3)
        pixel_mask = bu * triangle_mask[...,0] + bv * triangle_mask[...,1] + (1-bu-bv) * triangle_mask[...,2] # (B, H*W)
        pixel_mask = pixel_mask.unflatten(1, (H,W)).unsqueeze(-1) # (B, H, W, 1)
        pixel_masks[key] = pixel_mask

    return pixel_masks

if __name__ == '__main__':
    parser_av = config_parser()

    print("Loading Avatar A...")
    args_a = parser_av.parse_args(get_prefixed_args(sys.argv[1:], "a_"))
    avatar_a = Avatar(args_a)
    print("Loading Avatar B...")
    args_b = parser_av.parse_args(get_prefixed_args(sys.argv[1:], "b_"))
    avatar_b = Avatar(args_b)

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, help="Output path for the faceswap result")
    parser.add_argument("--sequence", type=int, default=0, help="Sequence index of B to use")
    parser.add_argument("--keep_eyes", action="store_true", help="Keep eyes of target sequence")
    parser.add_argument("--keep_mouth", action="store_true", help="Keep mouth of target sequence")
    args, _ = parser.parse_known_args()

    main(args, avatar_a, avatar_b)
