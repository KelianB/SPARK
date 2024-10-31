import sys
import os
import logging

from configargparse import ArgumentParser, Namespace
import torch
from torch.utils.data import Subset

# Add dependencies to PYTHONPATH
from adapt.constants import EMOCA_PATH, MULTIFLARE_PATH, SMIRK_PATH
sys.path.insert(1, str(MULTIFLARE_PATH))
sys.path.insert(1, str(EMOCA_PATH))
sys.path.insert(1, str(SMIRK_PATH))

# MultiFLARE imports
from dataset import MultiVideoDataset
from utils.dataset import DatasetCache
from arguments import config_parser as multiflare_config_parser

from adapt.crop_dataset import CropDataset
from adapt.wrapper import FaceTrackerWrapper
from adapt.general_utils import train_val_split
from adapt.constants import CROP_RESOLUTION

def spark_setup(args: Namespace, render_mode="crop", training=False,
                test_smooth_crops=False, test_sample_ratio=1, test_max_frames=None):
    """
    Args:
    - args: parsed command line arguments
    - render_mode: crop | full
    - training: bool
    - test_smooth_crops: pre-load the dataset and apply a low-pass filter to the crop coordinates
    - test_sample_ratio: subsampling ratio for the test dataset
    - test_max_frames: how many frames of the test set to keep
    """

    wrapper = FaceTrackerWrapper(args, render_mode, training)
    device = wrapper.device

    # Setup the dataset
    mf_args = args.multiflare_args
    dataset_train = MultiVideoDataset(mf_args.input_dir, mf_args.train_dir, sample_ratio=mf_args.sample_idx_ratio, head_only=mf_args.head_only)
    dataset_test = MultiVideoDataset(mf_args.input_dir, args.test_dirs, sample_ratio=test_sample_ratio, head_only=mf_args.head_only)

    if test_max_frames is not None and test_max_frames != -1:
        dataset_test = Subset(dataset_test, range(test_max_frames))

    # Add crops and compatibility for DECA/EMOCA
    dataset = CropDataset(dataset_train, CROP_RESOLUTION, prune_original_views=(render_mode=="crop"))
    dataset_test = CropDataset(dataset_test, CROP_RESOLUTION, prune_original_views=(render_mode=="crop"))
    # Cache the whole dataset - this is costly in RAM but saves us time
    dataset = DatasetCache(dataset)
    if test_smooth_crops:
        dataset_test = dataset_test.preload_and_smooth_crops(device, args.batch_size)
    else:
        dataset_test = DatasetCache(dataset_test)

    # Split train/validation sets
    subset_train, subset_val = train_val_split(dataset, args.val_ratio) if args.val_ratio > 0 else (dataset, [])
    logging.info(f"Size of train set: {len(subset_train)} / validation set: {len(subset_val)} / test set: {len(dataset_test)}")

    return wrapper, subset_train, subset_val, dataset_test

def spark_config_parser():
    parser = ArgumentParser()
    # General
    parser.add_argument("--config", is_config_file=True, help="Config file path")
    parser.add_argument("--device", type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="Which GPU to use; -1 is CPU")
    parser.add_argument("--exp_name", type=str, required=True, help="Name of the experiment, used in the output path")
    parser.add_argument("--encoder", type=str, required=True, choices="DECA|SMIRK", help="The encoder to use (for EMOCA, use DECA with a config file from EMOCA as deca_cfg)")
    parser.add_argument("--decoder", type=str, required=True, choices="DECA|SMIRK|MultiFLARE", help="The decoder to use for geometry and rendering")
    # MultiFLARE
    parser.add_argument("--multiflare", type=str, required=True, help="FLARE config file path")
    parser.add_argument("--multiflare_resume", type=int, default=0, help="FLARE resume iteration")
    # Tracker adaptation
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--tracker_resume", type=int, default=0, help="Tracker adaptation resume iteration")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of images to keep out of the training set for validation")
    parser.add_argument("--test_dirs", type=str, nargs="+", default=[], help="Test sequence directories")
    # DECA-specific parameters
    parser.add_argument("--deca_model", type=str, default="EMOCA_v2_lr_mse_20", help="Pre-trained DECA/EMOCA model")
    parser.add_argument("--deca_cfg", type=str, default="cfg_baseline.yaml", help="Configuration file to use for EMOCA")
    parser.add_argument("--deca_detail", action="store_true", help="Use detail mode of DECA")

    return parser

def parse_args(parser: ArgumentParser):
    args, _ = parser.parse_known_args()

    #################### Validate args #####################
    if not args.exp_name:
        raise ValueError("Please specify exp_name")
    if not args.test_dirs:
        raise ValueError("Please specify test_dirs")
    ########################################################

    # Parse MultiFLARE arguments
    mf_parser = multiflare_config_parser()
    mf_config = MULTIFLARE_PATH / "configs" /args.multiflare
    mf_args, _ = mf_parser.parse_known_args(f"--config {mf_config} --resume {args.multiflare_resume}")
    mf_args.run_name = mf_args.run_name if mf_args.run_name is not None else os.path.splitext(os.path.basename(mf_args.config))[0]
    args.multiflare_args = mf_args

    return args
