from tqdm import tqdm
from pathlib import Path
import logging
import os
from typing import List

from argparse import Namespace
import torch
from torch import Tensor
import torch.nn.functional as F

from main import spark_setup, spark_config_parser, parse_args
from adapt.wrapper import FaceTrackerWrapper
from adapt.async_image_saver import AsyncImageSaver

# MultiFLARE imports
from utils.dataset import DeviceDataLoader, find_collate

RENDER_MODE = "full"
MODE = "normals" # geo_normals | geo | normals 
VIS_HALF = True
REMOVE_NECK = True
HALF_OPACITY = 1.0

@torch.no_grad()
def main(args: Namespace, vis_wrappers: List[FaceTrackerWrapper], vis_args: List[Namespace], dataset):
    device = vis_wrappers[0].device

    out_dir: Path = args.out_dir / args.out 
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "info.txt", "w+") as file:
        info = "\n".join([
            "Video evaluation",
            "Test dirs: " + ", ".join(args.test_dirs),
            f"MultiFLARE: {args.multiflare} (resume: {args.multiflare_resume})",
            f"SPARK: {args.exp_name} (transfer iterations: {args.tracker_resume})",
            "Baselines:",
            *[f"\t{a.exp_name} (transfer iterations: {a.tracker_resume:04d})" for a in vis_args],
            f"Smoothed crops: {args.smooth_crops}",
        ])
        file.write(info)

    dataloader = DeviceDataLoader(dataset, device=device, batch_size=args.batch_size, collate_fn=find_collate(dataset), num_workers=0)

    fps = args.framerate / args.visu_interval

    with AsyncImageSaver() as img_saver:
        for views in tqdm(dataloader):
            inputs = None
            img_columns = []

            for wrapper in vis_wrappers:
                out = wrapper(views, training=False, visdict=True)
                values, visdict = out["values"], out["visdict"]
                if inputs is None:
                    inputs = visdict["inputs"]
                    white = torch.ones_like(inputs)

                masks = wrapper.rasterize_semantic_masks(values, ["sem_neck", "neck_lower", "sem_mouth_interior", "boundary", "right_half", "hair"])
                geometry_render = visdict["geometry_coarse"]
                normal_render = (visdict["normal_images"] + 1) / 2
                rast_mask = values["ops"]["alpha_images"]
                half_mask = (masks["right_half"] > 0.2).float()

                # For DECA, blend the detail rendering with the coarse rendering to avoid the UV seam showing up at the top of the head
                if getattr(wrapper.encoder, "deca_detail", False):
                    geometry_render = blend(geometry_render, visdict["geometry_detail"], masks["hair"])
                    normal_render = blend(normal_render, (visdict["normal_images_detail"] + 1) / 2, masks["hair"])
                
                if MODE == "geo_normals":
                    # Blend the geometry with the normal image on half of the face
                    render = blend(geometry_render, normal_render, half_mask)
                elif MODE == "normals":
                    render = normal_render
                elif MODE == "geo":
                    render = geometry_render

                render_mask = (masks["neck_lower"] + masks["boundary"] + masks["sem_mouth_interior"]) < 0.5
                render = blend(render, white, fix_mask(rast_mask * render_mask)) # blend over white background

                if VIS_HALF:
                    vis_half_mask = render_mask * (1 - half_mask) * ((masks["sem_neck"] < 0.5) if REMOVE_NECK else 1)
                    vis_half = blend(geometry_render, inputs, fix_mask(rast_mask * vis_half_mask) * HALF_OPACITY)
                    render = torch.cat((vis_half, render), dim=2)
                
                img_columns.append(render)

            if VIS_HALF:
                input_column = torch.cat((inputs, white), dim=2)
            else:
                input_column = inputs

            img_columns = [input_column] + img_columns

            img_rows = torch.cat(img_columns, dim=-1).permute(0,2,3,1) # (B, C, H, n_cols*W) -> (B, H, n_cols*W, C)
            # Put the processed images and filenames into the queue
            for vidx, img_row in zip(views["idx"], img_rows):
                img_saver.queue(img_row.cpu(), out_images / f"{vidx:05d}.png")

    logging.info("Creating video")
    os.system(f"/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i '{out_images / '*.png'}' -c:v libx264 -pix_fmt yuv420p {out_dir / 'video.mp4'}")
    logging.info("Done")


def blend(img1, img2, mask):
    """ Blend img1 onto img2 with given alpha mask (bool or float)"""
    return img1 * mask.float() + img2 * (1 - mask.float())

def fix_mask(mask: Tensor) -> Tensor:
    """
    Perform morphological closing operation (erosion and dilation) to fix aliasing problems in the mask.      
    Args:
        mask (Tensor): Input mask of shape (B, 1, H, W).
    Returns:
        Tensor: The processed mask.
    """
    kernel_size = 3
    # Erosion: Use min pooling (by inverting the mask) to shrink the mask
    mask = -F.max_pool2d(-mask, kernel_size, stride=1, padding=kernel_size // 2)
    # Dilation: Use max pooling to expand the mask
    mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
    return mask

if __name__ == "__main__":
    parser = spark_config_parser()
    parser.add_argument("--out", type=str, default="", help="Output directory, relative to the experiment dir or absolute")
    parser.add_argument("--visu_interval", type=int, default=1, help="Interval at which to sample frames for visualization")
    parser.add_argument("--n_frames", type=int, default=-1, help="Number of frames to process (-1 for whole video)")
    parser.add_argument("--framerate", type=int, default=30, help="Framerate for generating the edited video")
    parser.add_argument("--smooth_crops", action="store_true", help="Smooth the crops to reduce jittering")
    args = parse_args(parser)

    if not args.out:
        args.out = f"video_comparison_{args.tracker_resume}"

    wrapper_main, dataset_train, dataset_val, dataset_test = \
        spark_setup(args, render_mode=RENDER_MODE, test_smooth_crops=args.smooth_crops, test_sample_ratio=args.visu_interval, test_max_frames=args.n_frames)

    args_baselines = []

    # DECA
    if True:
        args_baseline_deca = Namespace(**args.__dict__)
        args_baseline_deca.encoder = "DECA"
        args_baseline_deca.decoder = "DECA"
        args_baseline_deca.deca_model = "DECA"
        args_baseline_deca.deca_cfg = "cfg.yaml"
        args_baseline_deca.tracker_resume = 0
        args_baseline_deca.exp_name = "DECA"
        args_baseline_deca.deca_detail = True
        args_baselines.append(args_baseline_deca)

    # EMOCA
    if True:
        args_baseline_emoca = Namespace(**args.__dict__)
        args_baseline_emoca.encoder = "DECA"
        args_baseline_emoca.decoder = "DECA"
        args_baseline_emoca.deca_model = "EMOCA_v2_lr_mse_20"
        args_baseline_emoca.deca_cfg = "cfg_baseline.yaml"
        args_baseline_emoca.tracker_resume = 0
        args_baseline_emoca.exp_name = "EMOCA"
        args_baseline_emoca.deca_detail = True
        args_baselines.append(args_baseline_emoca)

    # SMIRK
    if True:
        args_baseline_smirk = Namespace(**args.__dict__)
        args_baseline_smirk.encoder = "SMIRK"
        args_baseline_smirk.decoder = "SMIRK"
        args_baseline_smirk.tracker_resume = 0
        args_baseline_smirk.exp_name = "SMIRK"
        args_baselines.append(args_baseline_smirk)

    # EMOCA fine-tuned
    if False:
        args_baseline_emoca_tuned = Namespace(**args.__dict__)
        args_baseline_emoca_tuned.encoder = "DECA"
        args_baseline_emoca_tuned.decoder = "DECA"
        args_baseline_emoca_tuned.deca_model = "EMOCA_v2_lr_mse_20"
        args_baseline_emoca_tuned.deca_cfg = "cfg_baseline.yaml"
        args_baseline_emoca_tuned.tracker_resume = 2000
        args_baseline_emoca_tuned.exp_name = "EMOCA_baseline"
        args_baseline_emoca_tuned.deca_detail = True
        args_baselines.append(args_baseline_emoca_tuned)

    wrapper_baselines = [FaceTrackerWrapper(args_baseline, RENDER_MODE, training=False) for args_baseline in args_baselines]

    wrapper_baselines += [wrapper_main]
    args_baselines += [args]

    main(args, wrapper_baselines, args_baselines, dataset_test)
