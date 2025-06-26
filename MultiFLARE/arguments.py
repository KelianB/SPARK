from pathlib import Path

from configargparse import ArgumentParser
from argparse import BooleanOptionalAction
import torch

def config_parser():
    parser = ArgumentParser()
    # meta
    parser.add_argument("--config", is_config_file=True, help="Config file path")
    parser.add_argument("--run_name", type=str, default=None, help="Name of this run")
    
    # paths
    parser.add_argument("--input_dir", type=Path, help="Root of the input data")
    parser.add_argument("--train_dir", type=Path, nargs="+", help="Path to the training sequences (relative to input_dir)")
    parser.add_argument("--output_dir", type=Path, default="out", help="Path to the output directory")

    # misc
    parser.add_argument("--resume", type=int, default=0, help="Resume training at a given iteration")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of views used per iteration")
    parser.add_argument("--sample_idx_ratio", type=int, default=1, help="To sample less images (mainly for debugging purposes)")    
    parser.add_argument("--device", type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="Which GPU to use; -1 is CPU")
    parser.add_argument("--visualization_views", type=int, nargs="+", default=[], help="Views to use for visualization")
    parser.add_argument("--train_views_whitelist", type=int, nargs="*", default=[], help="Restrict training to these views (for debugging)")

    # iters
    parser.add_argument("--iterations", type=int, default=3000, help="Total number of iterations")
    parser.add_argument("--remesh_iterations", type=int, nargs="*", default=[500, 1000], help="Iterations at which to perform mesh upsampling")
    parser.add_argument("--remesh_method", type=str, default="subdivision", choices=["subdivision", "botsch"], help="Which remeshing method to use (Botsch and Kobbelt's or subdivision)")
    parser.add_argument("--save_frequency", type=int, default=1000, help="Frequency of mesh and network weights saving (in iterations)")
    parser.add_argument("--visualization_frequency", type=int, default=100, help="Frequency of visualizations (in iterations)")
 
    # lr
    parser.add_argument("--lr_vertices", type=float, default=5e-5, help="Step size/learning rate for the vertex positions")
    parser.add_argument("--lr_shader", type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    parser.add_argument("--lr_deformer", type=float, default=2e-4, help="Step size/learning rate for the deformation parameters")
    parser.add_argument("--vertices_lr_decay", type=float, default=0.998, help="Per-iteration decay of the vertices learning rate")

    # loss weights
    parser.add_argument("--weight_mask", type=float, default=0, help="Weight of the mask term")
    parser.add_argument("--weight_normal", type=float, default=0, help="Weight of the normal term")
    parser.add_argument("--weight_laplacian", type=float, default=0, help="Weight of the laplacian term")
    parser.add_argument("--weight_shading", type=float, default=0, help="Weight of the shading term")
    parser.add_argument("--weight_perceptual_loss", type=float, default=0, help="Weight of the perceptual loss")
    parser.add_argument("--weight_albedo_regularization", type=float, default=0, help="Weight of the albedo regularization")
    parser.add_argument("--weight_flame_regularization", type=float, default=0, help="Weight of the FLAME regularization")
    parser.add_argument("--weight_flame_remeshed_regularization", type=float, default=0, help="Weight of the FLAME regularization using the remeshed basis")
    parser.add_argument("--weight_white_lgt_regularization", type=float, default=0, help="Weight of the white light regularization")
    parser.add_argument("--weight_roughness_regularization", type=float, default=0, help="Weight of the roughness regularization")
    parser.add_argument("--weight_fresnel_coeff", type=float, default=0, help="Weight of the specular intensity regularization")
    parser.add_argument("--weight_interior_landmarks_L1", type=float, default=0, help="Weight of the L1 loss on interior face landmarks")
    parser.add_argument("--weight_landmarks_L1", type=float, default=0, help="Weight of the L1 loss on face landmarks")
    parser.add_argument("--weight_landmarks_L1_mediapipe", type=float, default=0, help="Weight of the L1 loss on MediaPipe face landmarks")
    parser.add_argument("--weight_iris_L1", type=float, default=0, help="Weight of the iris L1 loss, using landmarks")
    parser.add_argument("--weight_displacement_regularization_L1", type=float, default=0, help="Weight of the regularization on neutral mesh displacements (L1)")
    parser.add_argument("--weight_mouth_interior_mask", type=float, default=0, help="Weight of the mouth interior semantic mask loss")
    parser.add_argument("--r_mean", type=float, default=0.500, help="Mean roughness for the material regularization")
    parser.add_argument("--decay_flame", type=int, nargs="+", default=[100], help="Iterations at which to decay the FLAME regularization on the deformer")
    parser.add_argument("--mask_loss_type", type=str, default="matte", choices=(["matte", "semantic"]), help="Type of mask to use for the mask loss")
    parser.add_argument("--mask_loss_no_hair", action=BooleanOptionalAction, default=False, help="Mask out the hair region from the mask loss")
    parser.add_argument("--mask_out_mouth_interior", action=BooleanOptionalAction, default=False, help="Mask out the mouth interior from the photometric loss")

    # neural shader
    parser.add_argument("--light_mlp_dims", type=int, nargs="*", default=[64, 64, 64], help="Layer dimensions of the shading MLP")
    parser.add_argument("--light_mlp_activation", type=str, default="softplus", choices=(["none", "softplus"]), help="Activation function to apply at the very end of the light MLP")
    parser.add_argument("--multi_sequence_lighting", type=str, default="mlp_per_sequence", choices=(["single", "mlp_per_sequence", "mlp_per_sequence_share"]), help="Lighting behavior when training on multiple sequences")
    parser.add_argument("--material_embedding", type=str, default="hashgrid", choices=(["positional", "hashgrid"]), help="Input embedding for the material MLP")
    parser.add_argument("--material_mlp_dims", type=int, nargs="*", default=[64, 64, 64], help="Layer dimensions of the material MLP")
    parser.add_argument("--progressive_hash", action=BooleanOptionalAction, default=True, help="Use progressive hash encoding for the material MLP")
    parser.add_argument("--progressive_hash_iters", type=int, nargs="+", default=[0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000], help="Iterations at which hash levels are activated")
    parser.add_argument("--hash_include_input", action=BooleanOptionalAction, default=False, help="Include the input in the hash encoding features")
    parser.add_argument("--hash_max_resolution", type=int, default=4096, help="Max resolution for the hash encoding")
    parser.add_argument("--hash_levels", type=int, default=16, help="Number of resolution levels for the hash encoding")
    parser.add_argument("--hash_log2_size", type=int, default=19, help="Log2 size of the hashmap for hash encoding")

    # geometry
    parser.add_argument("--add_teeth", action=BooleanOptionalAction, default=False, help="Add teeth to the template head")
    parser.add_argument("--close_mouth_interior", action=BooleanOptionalAction, default=True, help="Add faces to close the mouth interior")
    parser.add_argument("--head_only", action=BooleanOptionalAction, default=True, help="Apply a head mask to the ground truth images and remove the torso/shoulders vertices")  
    parser.add_argument("--deformer_dims", type=int, nargs="+", default=[128, 128, 128, 128], help="Layer dimensions of the deformer")
    parser.add_argument("--deformer_pretrain", type=int, default=0, help="Number of iterations to pre-train the deformer with supervision of the original FLAME basis")
    parser.add_argument("--deformer_warmup", type=int, default=0, help="Number of iterations before beginning to train the deformer")
    parser.add_argument("--deformer_embed_freqs", type=int, default=0, help="Number of frequencies in the deformer input embedding (0 means no embedding)")
    parser.add_argument("--deformer_load", type=str, default=None, help="Resume the deformer from a given checkpoint (to save time on pretraining)")
    parser.add_argument("--deformer_overrides", action=BooleanOptionalAction, default=False, help="Force the expected values for LBS weights, posedirs and shapedirs manually for the teeth and eyes")
    parser.add_argument("--deformer_expression_only", action=BooleanOptionalAction, default=False, help="Only use the expression basis from the deformer; simply use FLAME for the LBS weights and posedirs")
    parser.add_argument("--deformer_use_blendshapes", action=BooleanOptionalAction, default=False, help="Whether to use blendshapes for the deformer expression basis")

    parser.add_argument("--blendshapes", type=str, default=None, help="")  
    parser.add_argument("--blendshapes_names", type=str, default=None, help="")  
    parser.add_argument("--blendshapes_neutral", type=str, default=None, help="")  
    parser.add_argument("--disable_jaw_joint", action=BooleanOptionalAction, default=False, help="Disable the jaw joint from FLAME (for use with blendshapes)")

    # wandb
    parser.add_argument("--wandb", action=BooleanOptionalAction, default=False, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_workspace", type=str, default="", help="Weights & Biases workspace")

    # fine-tuning pose & expression
    parser.add_argument("--finetune_tracking", action=BooleanOptionalAction, default=False, help="Toggle optimizing the pose and expression parameters")
    parser.add_argument("--finetune_pose_lr", type=float, default=1e-4, help="Learning rate for fine-tuning the pose parameters")
    parser.add_argument("--finetune_expr_lr", type=float, default=1e-3, help="Learning rate for fine-tuning the expression parameters")
    parser.add_argument("--finetune_tracking_warmup", type=int, default=0, help="Number of iterations before fine-tuning the tracking parameters")
    parser.add_argument("--initial_expr", type=str, default="tracking", help="Initialization of expression coefficients (tracking | zeros | flame2bs_invert | flame2bs_optim)")

    # fine-tuning camera parameters
    parser.add_argument("--finetune_cam_intrinsics", action=BooleanOptionalAction, default=False, help="Toggle optimizing the per-sequence camera intrinsics")
    parser.add_argument("--finetune_cam_intrinsics_lr", type=float, default=1e-2, help="Learning rate for fine-tuning the per-sequence camera intrinsics")

    return parser