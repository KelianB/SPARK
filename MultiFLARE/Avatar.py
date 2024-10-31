import os
import random
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor

from flame import FLAME
from dataset import MultiVideoDataset
from utils.geometry import disconnect_vertices, remove_faces
from flare.core import Mesh, Renderer
from flare.modules import (
    NeuralShader, ForwardDeformer, Displacements,
    pretrain_deformer,
    load_blendshapes, FLAME_to_blendshape_coefficients,
)

# Fix all seeds
torch.manual_seed(1138)
torch.cuda.manual_seed(1138)
np.random.seed(1138)
random.seed(1138)

class Avatar():
    def __init__(self, args):
        self.args = args
        args.run_name = args.run_name if args.run_name is not None else os.path.splitext(os.path.basename(args.config))[0]

        # Dirs / IO
        setup_logging(args.output_dir / args.run_name / "log.txt")
        self._init_dirs() 

        # Select device
        device = torch.device('cpu')
        if torch.cuda.is_available() and args.device >= 0:
            device = torch.device(f'cuda:{args.device}')
            logging.info(f"Using device {device} ({torch.cuda.get_device_name(device)})")
        else:
            logging.info(f"Using device {device}")
        self.device = device

        # Create the dataset
        logging.info("Creating dataset...")
        dataset_train = MultiVideoDataset(args.input_dir, args.train_dir, sample_ratio=args.sample_idx_ratio, head_only=args.head_only)
        self.dataset_train = dataset_train
        self.num_seq = dataset_train.num_seq

        self._init_geometry()
        self._init_appearance()

        flame = self.flame
        deformer_net = self.deformer_net

        #################### Misc loading ####################
        if args.resume > 0:
            tracking_dict = torch.load(self.networks_save_path / f"tracking_{args.resume:06d}.pt")
            pose_train = tracking_dict["pose_train"].to(device)
            expr_train = tracking_dict["expr_train"].to(device)
            cams_K_train =  tracking_dict["cams_K_train"].to(device)
            misc_dict = torch.load(self.networks_save_path / f"misc_{args.resume:06d}.pt")
            for key in misc_dict:
                if key.startswith("mask.v."):
                    flame.mask.v.register_buffer(key[7:], misc_dict[key].to(device))                
                elif key.startswith("mask.f."):
                    flame.mask.f.register_buffer(key[7:], misc_dict[key].to(device))
            flame.full_lmk_verts_idx = misc_dict["full_lmk_verts_idx"].to(device)
            flame.full_lmk_bary_coords = misc_dict["full_lmk_bary_coords"].to(device)
            flame.static_lmk_verts_idx = misc_dict["static_lmk_verts_idx"].to(device)
            flame.static_lmk_bary_coords = misc_dict["static_lmk_bary_coords"].to(device)
            flame.dynamic_lmk_verts_idx = misc_dict["dynamic_lmk_verts_idx"].to(device)
            flame.dynamic_lmk_bary_coords = misc_dict["dynamic_lmk_bary_coords"].to(device)
            flame.static_lmk_verts_idx_mediapipe = misc_dict["static_lmk_verts_idx_mediapipe"].to(device)
            flame.static_lmk_bary_coords_mediapipe = misc_dict["static_lmk_bary_coords_mediapipe"].to(device)
            flame.shapedirs_expression_updated = misc_dict["shapedirs_expression_updated"].to(device)
            flame.lbs_weights_updated = misc_dict["lbs_weights_updated"].to(device)
            flame.posedirs_updated = misc_dict["posedirs_updated"].to(device)
            if "uvs" in misc_dict:
                flame.uvs = misc_dict["uvs"].to(device)
            photo_loss_cache = misc_dict["photo_loss_cache"].to(device)
        else:
            pose_train = torch.stack([dataset_train.get_flame_pose(i, device) for i in range(len(dataset_train))])
            if args.initial_expr == "tracking":
                expr_train = torch.stack([dataset_train.get_flame_expression(i, device) for i in range(len(dataset_train))])       
            elif args.initial_expr == "zeros":
                expr_train = torch.zeros((len(dataset_train), self.n_exp), dtype=torch.float, device=device)
            elif args.initial_expr.startswith("flame2bs"):
                if args.deformer_pretrain == 0:
                    raise ValueError(f"initial_expr '{args.initial_expr}' requires deformer_pretrain > 0")
                if not args.deformer_use_blendshapes:
                    raise ValueError(f"initial_expr '{args.initial_expr}' requires deformer_use_blendshapes")
                expr_train = torch.stack([dataset_train.get_flame_expression(i, device) for i in range(len(dataset_train))])       
                expr_train = FLAME_to_blendshape_coefficients(expr_train, pose_train, args.initial_expr,
                                                                    flame, self.canonical_mesh, deformer_net)
            else:
                raise ValueError(f"Unknown mode for initial_expr: '{args.initial_expr}'")

            cams_K_train = dataset_train.K.clone().to(device) # (num_seq, 3, 3)
            photo_loss_cache = torch.zeros((len(dataset_train)), dtype=torch.float, device=device)
        self.pose_train, self.expr_train, self.cams_K_train, self.photo_loss_cache = pose_train, expr_train, cams_K_train, photo_loss_cache

    def _init_dirs(self):
        args = self.args
        exp_dir = args.output_dir / args.run_name
        self.images_save_path = exp_dir / "images"
        self.images_save_path.mkdir(parents=True, exist_ok=True)
        self.meshes_save_path = exp_dir / "meshes"
        self.meshes_save_path.mkdir(parents=True, exist_ok=True)
        self.networks_save_path = exp_dir / "network_weights"
        self.networks_save_path.mkdir(parents=True, exist_ok=True)
        self.experiment_dir = exp_dir
    
        if args.wandb:
            import wandb
            wandb_path = exp_dir / "wandb_logs"
            wandb_path.mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DIR"] = str(wandb_path)
            wandb.init(project=args.wandb_workspace, name=args.run_name)

    def _init_geometry(self):
        args, device, dataset_train =\
              self.args, self.device, self.dataset_train

        #################### Geometry model ####################
        flame = init_FLAME(device, args, dataset_train)
        self.flame = flame

        if args.blendshapes:
            blendshapes, blendshapes_neutral, blendshapes_names = load_blendshapes(args.blendshapes, args.blendshapes_neutral, args.blendshapes_names, device)
        else:
            blendshapes, blendshapes_neutral, blendshapes_names = None, None, None
            if args.deformer_use_blendshapes:
                raise ValueError("Cannot enable deformer_use_blendshapes without specifying blendshapes.")
        self.blendshapes, self.blendshapes_neutral, self.blendshapes_names = blendshapes, blendshapes_neutral, blendshapes_names
        self.n_exp = blendshapes.shape[-1] if args.deformer_use_blendshapes else flame.n_exp

        #################### Load canonical mesh ####################
        if args.resume > 0:
            mesh_path = self.meshes_save_path / f"mesh_{args.resume:06d}.obj"
            canonical_mesh = Mesh.read(mesh_path, device=device)
        else:
            canonical_mesh = Mesh(flame.canonical_verts.squeeze(0), flame.faces,
                                        uv_coords=flame.verts_uvs, uv_idx=flame.textures_idx, device=device)
        canonical_mesh.compute_connectivity()
        self.canonical_mesh = canonical_mesh

        canonical_verts = flame.canonical_verts.squeeze(0)
        self.canonical_aabb_minmax = [torch.min(canonical_verts, dim=0).values, torch.max(canonical_verts, dim=0).values]

        #################### Vertex displacements ####################
        displacements = Displacements(vertices_shape=canonical_mesh.vertices.shape) 
        displacements.to(device=device)
        self.displacements = displacements

        #################### Deformer ####################
        logging.info("Initializing deformer network")
        deformer_net = ForwardDeformer(flame, dims=args.deformer_dims, multires=args.deformer_embed_freqs, num_exp=self.n_exp, aabb=self.canonical_aabb_minmax,
                                        weight_norm=True, deformer_input=args.deformer_input, overrides=args.deformer_overrides, expr_only=args.deformer_expression_only)
        deformer_net.to(device)
        self.deformer_net = deformer_net

        if args.resume > 0:
            logging.info("Loading deformer network weights")
            deformer_net.load(self.networks_save_path / f"deformer_{args.resume:06d}.pt", device)
        elif args.deformer_pretrain > 0:
            if args.deformer_load:
                logging.info(f"Skipping deformer pre-training, loading from '{args.deformer_load}' instead")
                deformer_net.load(args.deformer_load, device)
            else:
                blendshapes_kwargs = {"blendshapes": blendshapes, "blendshapes_neutral": blendshapes_neutral}
                pretrain_deformer(args, deformer_net, flame, canonical_mesh.vertices, canonical_mesh, device,
                                  **(blendshapes_kwargs if args.deformer_use_blendshapes else {}))

    def _init_appearance(self):
        args, device = self.args, self.device

        #################### Setup the renderer ####################
        renderer = Renderer(device=device)
        channels_gbuffer = ["mask", "position", "normal", "canonical_position"]
        if args.material_input == "flame_pos":
            channels_gbuffer.append("flame_position")
        elif args.material_input == "uvs":
            channels_gbuffer.append("uvs")
        logging.info(f"Rasterizing: {channels_gbuffer}")

        #################### Shading ####################
        logging.info("Initializing neural shader")
        shader = NeuralShader(
            aabb=self.canonical_aabb_minmax,
            material_input=args.material_input, material_embedding=args.material_embedding, material_mlp_dims=args.material_mlp_dims, material_last_activation=torch.nn.Sigmoid(), 
            light_mlp_activation=args.light_mlp_activation, light_mlp_dims=args.light_mlp_dims, multi_sequence_lighting=args.multi_sequence_lighting, num_seq=self.num_seq,
            progressive_hash=args.progressive_hash, progressive_hash_iters=args.progressive_hash_iters,
            hash_include_input=args.hash_include_input, hash_max_resolution=args.hash_max_resolution, hash_levels=args.hash_levels, hash_log2_size=args.hash_log2_size,
            device=device)
        if args.resume > 0:
            logging.info("Loading neural shader weights")
            shader.load(self.networks_save_path / f"shader_{args.resume:06d}.pt")

        self.shader = shader
        self.renderer = renderer
        self.channels_gbuffer = channels_gbuffer

    def save_all(self, suffix: str):
        shader, deformer_net, expr_train, pose_train, cams_K_train, flame, photo_loss_cache = \
            self.shader, self.deformer_net, self.expr_train, self.pose_train, self.cams_K_train, self.flame, self.photo_loss_cache

        # Bake displacements into the canonical mesh
        mesh, _ = Avatar.compute_displaced_mesh(self.canonical_mesh, self.displacements(), flame)

        with torch.no_grad():
            mesh.write(self.meshes_save_path / f"mesh_{suffix}.obj")                                
            shader.save(self.networks_save_path / f'shader_{suffix}.pt')
            deformer_net.save(self.networks_save_path / f'deformer_{suffix}.pt')
            torch.save({
                "expr_train": expr_train.cpu(),
                "pose_train": pose_train.cpu(),
                "cams_K_train": cams_K_train.cpu(),
            }, self.networks_save_path / f"tracking_{suffix}.pt")
            torch.save({
                "photo_loss_cache": photo_loss_cache.cpu(),
                **dict([f"mask.v.{key}", mask.cpu()] for key,mask in flame.mask.v.items()),
                **dict([f"mask.f.{key}", mask.cpu()] for key,mask in flame.mask.f.items()),
                "full_lmk_verts_idx": flame.full_lmk_verts_idx.cpu(),
                "full_lmk_bary_coords": flame.full_lmk_bary_coords.cpu(),
                "static_lmk_verts_idx": flame.static_lmk_verts_idx.cpu(),
                "static_lmk_bary_coords": flame.static_lmk_bary_coords.cpu(),
                "dynamic_lmk_verts_idx": flame.dynamic_lmk_verts_idx.cpu(),
                "dynamic_lmk_bary_coords": flame.dynamic_lmk_bary_coords.cpu(),
                "static_lmk_verts_idx_mediapipe": flame.static_lmk_verts_idx_mediapipe.cpu(),
                "static_lmk_bary_coords_mediapipe": flame.static_lmk_bary_coords_mediapipe.cpu(),
                "shapedirs_expression_updated": flame.shapedirs_expression_updated.cpu(),
                "lbs_weights_updated": flame.lbs_weights_updated.cpu(),
                "posedirs_updated": flame.posedirs_updated.cpu(),
                **({} if flame.uvs is None else {"uvs": flame.uvs.cpu()}),
            }, self.networks_save_path / f"misc_{suffix}.pt")

    def run(self,
            views,
            iteration: int,
            mesh: Mesh = None,
            deformer_net: ForwardDeformer = None,
            shader: NeuralShader = None,
            shapedirs: Tensor = None,
            posedirs: Tensor = None,
            lbs_weights: Tensor = None,
            deformed_vertices: Tensor = None,
            extra_vert_attrs: Dict[str, Tensor] = None,
            texture: Tensor = None,
        ):

        args, flame, renderer, channels_gbuffer = self.args, self.flame, self.renderer, self.channels_gbuffer

        # Canonical mesh
        if mesh is None:
            canonical_offsets = self.displacements()
            mesh, *_ = Avatar.compute_displaced_mesh(self.canonical_mesh, canonical_offsets, flame)

        # Deform
        if deformed_vertices is None:
            if deformer_net is None:
                deformer_net = self.deformer_net
            deformed_vertices, lbs_weights, shapedirs, posedirs = \
                Avatar.compute_deformed_verts(mesh, flame, pose=views["flame_pose"], expression=views["flame_expression"],
                                              deformer_net=deformer_net, shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights)
        d_normals = mesh.fetch_all_normals(deformed_vertices, mesh)

        if extra_vert_attrs is None:
            extra_vert_attrs = dict()
        if args.material_input == "flame_pos":
            extra_vert_attrs["flame_position"] = flame.canonical_verts
        elif args.material_input == "uvs":
            extra_vert_attrs["uv_coords"] = flame.uvs
        if texture is not None:
            extra_vert_attrs["uv_coords"] = flame.uvs

        # Rasterize
        gbuffers = renderer.render_batch(views["camera"], deformed_vertices.contiguous(), d_normals,
                            channels=channels_gbuffer, with_antialiasing=True, 
                            canonical_v=mesh.vertices, canonical_idx=mesh.indices,
                            extra_vert_attrs=extra_vert_attrs)
        
        # Shade the final color
        if shader is None:
            shader = self.shader
        rgb_pred, cbuffers, gbuffer_mask = shader(gbuffers, views, mesh, iteration, material_input=args.material_input, texture=texture)

        return rgb_pred, gbuffers, cbuffers, mesh, canonical_offsets, deformed_vertices, gbuffer_mask, lbs_weights, shapedirs, posedirs


    ########## Static methods ##########

    def compute_displaced_mesh(mesh: Mesh, canonical_offsets: Tensor, flame: FLAME) -> tuple[Mesh, Tensor]:
        no_disp_mask = flame.mask.f.left_eyeball + flame.mask.f.right_eyeball + flame.mask.f.left_eyelid_extended + flame.mask.f.right_eyelid_extended
        if flame.has_teeth:
            no_disp_mask += flame.mask.f.teeth
        disp_mask = 1.0 - no_disp_mask.clamp(0, 1).unsqueeze(1)

        canonical_offset_vertices = mesh.vertices + canonical_offsets * disp_mask
        mesh = mesh.with_vertices(canonical_offset_vertices)
        return mesh, canonical_offset_vertices
    
    def compute_deformed_verts(mesh: Mesh, flame: FLAME, pose: Tensor, expression: Tensor, deformer_net: Tensor,
                               shapedirs: Tensor = None, posedirs: Tensor = None, lbs_weights: Tensor = None) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        if shapedirs is None or posedirs is None or lbs_weights is None:
            shapedirs_, posedirs_, lbs_weights_ = deformer_net.query_weights(mesh.vertices.detach() if deformer_net.input == "canonical_pos" else flame.uvs)
            if shapedirs is None: shapedirs = shapedirs_
            if posedirs is None: posedirs = posedirs_
            if lbs_weights is None: lbs_weights = lbs_weights_
        return Avatar._compute_deformed_verts(mesh, flame, pose, expression, shapedirs, posedirs, lbs_weights), lbs_weights, shapedirs, posedirs

    def compute_deformed_verts_FLAME(mesh: Mesh, flame: FLAME, pose: Tensor, expression: Tensor) -> tuple[Tensor,Tensor,Tensor,Tensor]:
        shapedirs = flame.shapedirs_expression
        posedirs = flame.posedirs
        lbs_weights = flame.lbs_weights
        return Avatar._compute_deformed_verts(mesh, flame, pose, expression, shapedirs, posedirs, lbs_weights), lbs_weights, shapedirs, posedirs

    def _compute_deformed_verts(mesh: Mesh, flame: FLAME, pose: Tensor, expression: Tensor,
                                shapedirs: Tensor, posedirs: Tensor, lbs_weights: Tensor) -> Tensor:
        batch_size = len(pose)
        batched_verts = mesh.vertices.unsqueeze(0).repeat(batch_size, 1, 1)
        rotation, neck, jaw, eyes = pose[:,:3], pose[:,3:6], pose[:,6:9], pose[:,9:15]
        translation = pose[:,15:18]
        return flame.forward_skinning(batched_verts, expression, rotation, neck, jaw, eyes,
                                            shapedirs, posedirs, lbs_weights) + translation.unsqueeze(1)

def init_FLAME(device, args, dataset_train) -> FLAME:
    ### ============== load FLAME mesh ==============================
    flame_shape = dataset_train.shape_params
    #canonical_exp = (dataset_train.get_mean_expression()).to(device)
    canonical_pose = torch.zeros((15), dtype=torch.float)
    canonical_pose[6] = 0.1 # open jaw to avoid upsampling artifacts and to facilitate learning for the deformer
    canonical_expr = torch.zeros((50), dtype=torch.float)
    flame = FLAME(shape_params=100, expr_params=50, baked_identity_params=flame_shape,
                        add_teeth=args.add_teeth, close_mouth_interior=args.close_mouth_interior,
                        canonical_pose=canonical_pose, canonical_expr=canonical_expr).to(device)
    flame.jaw_enabled = not args.disable_jaw_joint

    # Remove some faces
    if args.head_only:
        flame.faces, kept_faces_mask = disconnect_vertices(flame.faces, flame.mask.v.boundary.to(device), get_face_mask=True)
        removed_faces = (kept_faces_mask == 0).nonzero().squeeze(-1)
        flame.verts_uvs, flame.textures_idx = remove_faces(flame.verts_uvs, flame.textures_idx, removed_faces)

    flame.posedirs_updated = flame.posedirs.clone()
    flame.lbs_weights_updated = flame.lbs_weights.clone()
    flame.shapedirs_expression_updated = flame.shapedirs_expression.clone()

    return flame

logging_setup = False
def setup_logging(log_file: Path = None):
    # Ensure we don't setup logging twice
    global logging_setup
    if logging_setup: return
    logging_setup = True

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatting
    formatter = logging.Formatter("[%(asctime)s](%(levelname)s) %(message)s", datefmt="%H:%M:%S")

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
