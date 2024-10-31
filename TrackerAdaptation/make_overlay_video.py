import os
from tqdm import tqdm
from pathlib import Path
import logging

from main import spark_setup, spark_config_parser, parse_args
from adapt.wrapper import FaceTrackerWrapper
from adapt.async_image_saver import AsyncImageSaver
from adapt.face_decoder import MultiFLAREDecoder
from make_comparison_video import fix_mask, blend

# MultiFLARE imports
from utils.dataset import DeviceDataLoader, find_collate, load_img

from argparse import Namespace
import torch

@torch.no_grad()
def main(wrapper: FaceTrackerWrapper, args: Namespace, dataset):
    assert isinstance(wrapper.decoder, MultiFLAREDecoder), "Only MultiFLARE decoder supports textures."

    device = wrapper.device

    out_dir: Path = args.out_dir / args.out 
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images = out_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    dataloader = DeviceDataLoader(dataset, device=device, batch_size=args.batch_size, collate_fn=find_collate(dataset), num_workers=0)

    texture = load_img(args.texture).to(device) # (H, W, 4)
    # texture = srgb_to_rgb(texture)
    wrapper.decoder.texture = texture

    fps = args.framerate / args.visu_interval

    with AsyncImageSaver() as img_saver:
        for views in tqdm(dataloader):
            run_dict = wrapper(views, training=False, visdict=True)
            visdict = run_dict["visdict"]
            values = run_dict["values"]
            rast_mask = values["ops"]["alpha_images"] # rasterization mask (also accounts for the alpha of the texture)
            rast_mask = fix_mask(rast_mask)

            sem_masks = wrapper.rasterize_semantic_masks(values, ["sem_mouth_interior"])
            display_mask = sem_masks["sem_mouth_interior"] < 0.5
            display_mask = rast_mask * display_mask
            overlayed_image = blend(visdict["output_images_coarse"], visdict["inputs"], fix_mask(display_mask) * args.opacity)

            img_columns = [
                visdict["inputs"],
                overlayed_image,
                visdict["geometry_coarse"],
            ]
            img_columns = [img.permute(0,2,3,1) for img in img_columns] # BCHW to BHWC
            img_rows = torch.cat(img_columns, dim=2) # (B, H, n_cols*W, C)

            for vidx, img_row in zip(views["idx"], img_rows):
                img_saver.queue(img_row.cpu(), out_images / f"{vidx:05d}.png")

    logging.info("Creating video")
    os.system(f"/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i '{out_images / '*.png'}' -c:v libx264 -pix_fmt yuv420p {out_dir / 'video.mp4'}")
    logging.info("Done")


if __name__ == "__main__":
    parser = spark_config_parser()
    parser.add_argument("--out", type=str, default="", help="Output directory, relative to the experiment dir or absolute")
    parser.add_argument("--visu_interval", type=int, default=1, help="Interval at which to sample frames for visualization")
    parser.add_argument("--n_frames", type=int, default=-1, help="Number of frames to process (-1 for whole video)")
    parser.add_argument("--framerate", type=int, default=30, help="Framerate for generating the edited video")
    parser.add_argument("--smooth_crops", action="store_true", help="Smooth the crops to reduce jittering")
    parser.add_argument("--texture", type=str, required=True, help="Path to a texture for rendering")
    parser.add_argument("--opacity", type=float, default=1.0, help="Opacity of the texture overlay")
    args = parse_args(parser)

    if not args.out:
        args.out = f"video_overlay_{args.tracker_resume}"

    wrapper, _, _, dataset_test = \
        spark_setup(args, render_mode="full", test_smooth_crops=args.smooth_crops, test_sample_ratio=args.visu_interval, test_max_frames=args.n_frames)

    main(wrapper, args, dataset_test)
