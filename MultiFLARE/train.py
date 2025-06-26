import time
import math
import numpy as np
import torch
from tqdm import tqdm
import wandb
from arguments import config_parser
import logging

from dataset import MultiVideoDataset
from utils.visualization import save_img, convert_uint, visualize_grid, tensor_vis_landmarks
from utils.dataset import DeviceDataLoader, SemanticMask, to_device_recursive
from utils.geometry import remesh_FLAME, subdivide_FLAME
from flare.core import Mesh
from flare.losses import *
from Avatar import Avatar

def main(avatar: Avatar):
    args = avatar.args
    device, dataset_train, flame, canonical_mesh, displacements, deformer_net, shader, renderer, expr_train, pose_train, cams_K_train, photo_loss_cache = \
        avatar.device, avatar.dataset_train, avatar.flame, avatar.canonical_mesh, avatar.displacements, avatar.deformer_net, avatar.shader, avatar.renderer, avatar.expr_train, avatar.pose_train, avatar.cams_K_train, avatar.photo_loss_cache

    with open(avatar.experiment_dir / "args.txt", "w") as text_file:
        print(str(args), file=text_file)

    if args.train_views_whitelist:
        dataset_train_subset = torch.utils.data.Subset(dataset_train, args.train_views_whitelist)
        dataloader_train = DeviceDataLoader(dataset_train_subset, device=device, batch_size=args.batch_size, collate_fn=dataset_train.collate, shuffle=True, drop_last=False, num_workers=4)
    else:
        dataloader_train = DeviceDataLoader(dataset_train, device=device, batch_size=args.batch_size, collate_fn=dataset_train.collate, shuffle=True, drop_last=True, num_workers=4)

    if args.visualization_views:
        view_indices = args.visualization_views
    else:
        view_indices = [seq_frames[len(seq_frames)//2] for seq_frames in dataset_train.frames_per_seq]
    debug_views = dataset_train.collate([dataset_train.__getitem__(idx) for idx in view_indices])

    debug_views = to_device_recursive(debug_views, device)

    #################### Loss functions and weights ####################
    loss_weights = {
        "mask": args.weight_mask,
        "normal": args.weight_normal,
        "laplacian": args.weight_laplacian,
        "shading": args.weight_shading,
        "perceptual_loss": args.weight_perceptual_loss,
        "albedo_regularization": args.weight_albedo_regularization,
        "roughness_regularization": args.weight_roughness_regularization,
        "white_light_regularization": args.weight_white_lgt_regularization,
        "fresnel_coeff": args.weight_fresnel_coeff,
        "flame_regularization": args.weight_flame_regularization,
        "interior_landmarks_L1": args.weight_interior_landmarks_L1,
        "landmarks_L1": args.weight_landmarks_L1,
        "landmarks_L1_mediapipe": args.weight_landmarks_L1_mediapipe,
        "iris_L1": args.weight_iris_L1,
        "flame_remeshed_regularization": args.weight_flame_remeshed_regularization,
        "mouth_interior_mask": args.weight_mouth_interior_mask,
        "displacement_regularization_L1": args.weight_displacement_regularization_L1,
    }
    loss_weights_initial = {**loss_weights}

    losses = {k: torch.tensor(0.0, device=device) for k in loss_weights}

    if loss_weights["perceptual_loss"] > 0.0:
        VGGloss = VGGPerceptualLoss().to(device)

    # Disable the mask loss in specific regions
    mask_loss_disable_labels = [SemanticMask.CLOTH_NECKLACE, SemanticMask.NECK, SemanticMask.HAT]
    if args.mask_loss_no_hair:
        mask_loss_disable_labels.append(SemanticMask.HAIR)
        
    #################### Create optimizers ####################
    lr_vertices = args.lr_vertices
    def create_displacements_optimizer(args, lr):
        return torch.optim.Adam(list(displacements.parameters()), lr=lr)
    optimizer_vertices = create_displacements_optimizer(args, lr_vertices)
    optimizer_deformer = torch.optim.Adam(list(deformer_net.parameters()), lr=args.lr_deformer)
    
    # Create the optimizer for the neural shader
    params = list(shader.parameters()) 
    if args.weight_albedo_regularization > 0:
        from robust_loss_pytorch.adaptive import AdaptiveLossFunction
        _adaptive = AdaptiveLossFunction(num_dims=4, float_dtype=np.float32, device=device)
        params += list(_adaptive.parameters()) ## need to train it
    optimizer_shader = torch.optim.Adam(params, lr=args.lr_shader)

    if args.finetune_tracking:
        pose_train, expr_train = torch.nn.Parameter(pose_train), torch.nn.Parameter(expr_train)
        optimizer_tracking = torch.optim.Adam([{"params": pose_train, "lr": args.finetune_pose_lr},
                                               {"params": expr_train, "lr": args.finetune_expr_lr}])
    if args.finetune_cam_intrinsics:
        cams_K_train = torch.nn.Parameter(cams_K_train)
        optimizer_cam_intrinsics = torch.optim.Adam([{"params": cams_K_train, "lr": args.finetune_cam_intrinsics_lr}])
    
    shader.train()
    deformer_net.train()
    displacements.train()

    def configure(iteration):
        # Adjust weights and step size after first upsample
        remeshed = len(args.remesh_iterations) > 0 and iteration >= args.remesh_iterations[0]
        for k in ["laplacian", "normal"]:
            loss_weights[k] = loss_weights_initial[k] * (4 if remeshed else 1)
        nonlocal lr_vertices
        lr_vertices = args.lr_vertices * (0.75 if remeshed else 1)
        # Decay LR for displacements
        lr_vertices *= (args.vertices_lr_decay ** iteration)
        optimizer_vertices.param_groups[0]["lr"] = lr_vertices
        # Decay FLAME regularization
        flame_decays = sum(1 for it in args.decay_flame if iteration >= it)
        loss_weights["flame_regularization"] = loss_weights_initial["flame_regularization"] * (0.5 ** flame_decays)

    # ==============================================================================================
    # Training
    # ==============================================================================================
    epochs = math.ceil(args.iterations / len(dataloader_train))
    iteration = args.resume
    last_iteration = args.resume + args.iterations

    print("=="*50)
    logging.info(f"Training from iteration {iteration+1} to {last_iteration} (1 epoch = {len(dataloader_train)} iters)")
    print("=="*50)

    progress_bar = tqdm(range(epochs))
    start = time.time()
    for epoch in progress_bar:
        if epoch > 0 or args.resume >= len(dataloader_train): # if we trained for at least one epoch before
            logging.info(f"Beginning epoch {epoch}. Photometric loss: mean={photo_loss_cache.mean():.6f} median={photo_loss_cache.median():.6f}")

        for views in dataloader_train:
            if iteration >= last_iteration:
                break
            iteration += 1
            progress_bar.set_description(desc=f"Epoch {epoch+1}, Iter {iteration}")
            is_visualize_iter = args.visualization_frequency > 0 and (iteration == 1 or iteration % args.visualization_frequency == 0)
            is_save_iter = args.save_frequency > 0 and (iteration == 1 or iteration % args.save_frequency == 0)

            # Update learning rates, loss weights, etc as needed.
            configure(iteration)

            MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)

            # ==============================================================================================
            # Remesh
            # ==============================================================================================
            if iteration in args.remesh_iterations:
                print("=="*50)
                mesh, *_ = Avatar.compute_displaced_mesh(canonical_mesh, avatar.displacements(), flame)

                logging.info(f"Remeshing at iteration {iteration} (method: {args.remesh_method})")
                if args.remesh_method == "botsch":
                    # Reduce the average edge length
                    e0, e1 = mesh.edges.unbind(1)
                    average_edge_length = torch.linalg.norm(mesh.vertices[e0] - mesh.vertices[e1], dim=-1).mean()
                    h = float(average_edge_length/1.5)

                    # These groups of vertices will not be remeshed
                    separate_verts_masks = {"eye_left": flame.mask.v.left_eyeball, "eye_right": flame.mask.v.right_eyeball}
                    if flame.has_teeth:
                        separate_verts_masks["teeth"] = flame.mask.v.teeth

                    _, v_remeshed, f_remeshed = remesh_FLAME(flame, mesh.vertices, mesh.indices, h, separate_verts_masks)
                else:
                    v_remeshed, f_remeshed = subdivide_FLAME(flame, mesh.vertices, mesh.indices, levels=1)
                    
                F = f_remeshed.shape[0]
                uv_coords = flame.uvs.view(F*3, 2)
                uv_idx = torch.arange(0, F*3, device=device).view(F, 3)
                canonical_mesh = Mesh(v_remeshed, f_remeshed, uv_coords=uv_coords, uv_idx=uv_idx, device=device)
                canonical_mesh.compute_connectivity()
                avatar.canonical_mesh = canonical_mesh

                logging.info(f"Vertices: {v_remeshed.shape} / Faces: {f_remeshed.shape}")
                del v_remeshed, f_remeshed

                displacements.register_parameter("vertex_offsets", torch.nn.Parameter(torch.zeros(canonical_mesh.vertices.shape), requires_grad=True))
                displacements.canonical_vertices = canonical_mesh.vertices
                displacements.vertices_shape = canonical_mesh.vertices.shape
                displacements.to(device=device)
                optimizer_vertices = create_displacements_optimizer(args, lr_vertices)
                print("=="*50)

            extra_rast_attrs = {}
            if args.mask_out_mouth_interior or loss_weights["mouth_interior_mask"] > 0:
                extra_rast_attrs["mouth_interior_mask"] = getattr(flame.mask.f, "sem_mouth_interior").to(device).unsqueeze(-1)

            # Avatar.run call
            rgb_pred, gbuffers, cbuffers, mesh, canonical_offsets, deformed_vertices, gbuffer_mask, lbs_weights, shapedirs, posedirs = \
                 avatar.run(views, iteration, extra_rast_attrs=extra_rast_attrs)

            # ==============================================================================================
            # Loss function 
            # ==============================================================================================
            
            photo_mask = views["mask"]
            if args.mask_out_mouth_interior:
                #photo_mask *= ~views_subset["semantic_mask"][..., SemanticMask.MOUTH_INTERIOR].unsqueeze(-1)
                photo_mask *= 1 - gbuffers["mouth_interior_mask"].detach()

            # Main photometric loss
            losses["shading"], _, tonemapped_colors = shading_loss_batch(rgb_pred, views["img"], photo_mask)
            with torch.no_grad():
                photo_loss_cache[views["idx"]] = shading_loss_batch_framewise(rgb_pred, views["img"], photo_mask)

            # VGG loss
            if loss_weights["perceptual_loss"] > 0:
                losses["perceptual_loss"] = VGGloss(tonemapped_colors[0], tonemapped_colors[1], iteration)

            # Basic geometry regularizations
            if loss_weights["normal"] > 0:
                losses["normal"] = normal_consistency_loss(mesh)
            if loss_weights["laplacian"] > 0:
                losses["laplacian"] = laplacian_loss(mesh)
            if loss_weights["displacement_regularization_L1"] > 0:
                losses["displacement_regularization_L1"] = torch.linalg.vector_norm(canonical_offsets, dim=-1, ord=1).mean()

            # Mask loss 
            if args.mask_loss_type == "matte":
                mask = views["mask"]
            elif args.mask_loss_type == "semantic":
                mask = views["semantic_mask"][...,(SemanticMask.BACKGROUND, SemanticMask.CLOTH_NECKLACE)].int().sum(-1) == 0
                mask = mask.float().unsqueeze(-1)
            if len(mask_loss_disable_labels) > 0:
                # Disable the mask loss in specific regions
                mask_loss_region = sum(views["semantic_mask"][..., r].int() for r in mask_loss_disable_labels).unsqueeze(-1) == 0
                losses["mask"] = mask_loss(mask * mask_loss_region, gbuffer_mask * mask_loss_region)
            else:
                losses["mask"] = mask_loss(mask, gbuffer_mask)

            # Appearance regularizations
            if loss_weights["albedo_regularization"] > 0:
                losses["albedo_regularization"] = albedo_regularization(_adaptive, shader, mesh, device, iteration)
            if loss_weights["white_light_regularization"] > 0:
                losses["white_light_regularization"] = white_light(cbuffers)
            if loss_weights["roughness_regularization"] > 0:
                losses["roughness_regularization"] = roughness_regularization(cbuffers["roughness"], views["semantic_mask"], views["mask"], r_mean=args.r_mean)
            if loss_weights["fresnel_coeff"] > 0:
                losses["fresnel_coeff"] = spec_intensity_regularization(cbuffers["spec_int"], views["semantic_mask"], views["mask"])
            
            # Deformer / FLAME basis regularization
            if iteration > args.deformer_warmup:
                if loss_weights["flame_regularization"] > 0:
                    losses["flame_regularization"], _ = flame_regularization(flame, lbs_weights, shapedirs, posedirs, mesh.vertices, 
                                                                             iteration, views_subset=views, gbuffer=gbuffers)
                if loss_weights["flame_remeshed_regularization"] > 0:
                    l = torch.linalg.vector_norm(shapedirs - flame.shapedirs_expression_updated, dim=1, ord=1).mean(1) # mean over expressions - shape: (V)
                    losses["flame_remeshed_regularization"] = l.mean()
                    # Regularize more outside of the face area (where there are no expressions)
                    # This prevents weird behavior in the neck/hair on datasets where it varies a lot between sequences.
                    higher_reg_mask = 1.0 - flame.mask.f.face
                    losses["flame_remeshed_regularization"] += (l * higher_reg_mask).mean() * 1e5

            # Landmarks losses (experimental)
            pose = views["flame_pose"]
            if loss_weights["interior_landmarks_L1"] > 0:
                losses["interior_landmarks_L1"] = landmarks_L1_loss(flame, views["landmarks"], views["camera"], renderer, deformed_vertices,
                                                                    pose, lmk_idx=LANDMARKS_FAN_INTERIOR)
            if loss_weights["landmarks_L1"] > 0:
                lmk_idx_no_iris = torch.arange(0, 68, dtype=torch.long, device=deformed_vertices.device)
                losses["landmarks_L1"] = landmarks_L1_loss(flame, views["landmarks"], views["camera"], renderer, deformed_vertices, pose, lmk_idx=lmk_idx_no_iris)
            if loss_weights["iris_L1"] > 0:
                losses["iris_L1"] = iris_L1_loss(flame, views["landmarks"], views["camera"], renderer, deformed_vertices, pose)
            if loss_weights["landmarks_L1_mediapipe"] > 0:
                losses["landmarks_L1_mediapipe"] = landmarks_L1_loss_mediapipe(flame, views["landmarks_mediapipe"], views["camera"], renderer, deformed_vertices)

            # Mouth interior mask vs. segmentation (experimental, doesn't really work)
            if loss_weights["mouth_interior_mask"] > 0:
                semantic_mask = views["semantic_mask"][..., (SemanticMask.MOUTH_INTERIOR,)].float() # (B, H, W, 1)  
                render_mask = gbuffers["mouth_interior_mask"]
                losses["mouth_interior_mask"] = (semantic_mask - render_mask).abs().mean() # mean over batch, height and width

            loss = torch.tensor(0., device=device) 
            for k, v in losses.items():
                loss += v * loss_weights[k]

            # ==============================================================================================
            # Optimizer step
            # ==============================================================================================
            optimizer_shader.zero_grad()
            optimizer_vertices.zero_grad()
            optimizer_deformer.zero_grad()
            if args.finetune_tracking:
                optimizer_tracking.zero_grad()
            if args.finetune_cam_intrinsics:
                optimizer_cam_intrinsics.zero_grad()

            loss.backward()
            torch.cuda.synchronize()

            optimizer_shader.step()
            optimizer_vertices.step()
            if iteration > args.deformer_warmup:
                optimizer_deformer.step()
            if args.finetune_tracking and iteration > args.finetune_tracking_warmup:
                optimizer_tracking.step()
            if args.finetune_cam_intrinsics:
                optimizer_cam_intrinsics.step()
            progress_bar.set_postfix({"loss": loss.detach().cpu().item()})

            del gbuffers, cbuffers

            # ==============================================================================================
            # Logging
            # ==============================================================================================
            if args.wandb:
                wandb_data = {"loss_total": loss}
                for key in losses:
                    wandb_data[f"loss_{key}"] = losses[key]

            # ==============================================================================================
            # Visualizations
            # ==============================================================================================
            if is_visualize_iter:
                with torch.no_grad():
                    MultiVideoDataset.override_values_batch(debug_views, pose_train, expr_train, cams_K_train)
                    extra_rast_attrs = {}

                    visualize_uvs = False
                    if visualize_uvs: extra_rast_attrs["uvs"] = flame.uvs

                    debug_rgb_pred, debug_gbuffers, debug_cbuffers, _, _, deformed_vertices, *_ = avatar.run(debug_views, iteration, extra_rast_attrs=extra_rast_attrs)

                    # Landmarks visualization
                    if loss_weights_initial["landmarks_L1"] > 0:
                        lmk_screen = renderer.to_screen_space(flame.get_landmark_positions_2d(deformed_vertices, debug_views["flame_pose"]), debug_views["camera"]) / (renderer.resolution[0] / 2) # (B, 70, 2)
                        debug_gbuffers["landmarks"] = tensor_vis_landmarks(debug_views["img"], lmk_screen, color="r", gt_landmarks=debug_views["landmarks"])
                    if loss_weights_initial["landmarks_L1_mediapipe"] > 0:
                        lmk_screen_mp = renderer.to_screen_space(flame.get_landmark_positions(deformed_vertices, which="static_mediapipe"), debug_views["camera"]) / (renderer.resolution[0] / 2) # (B, L, 2)
                        debug_gbuffers["landmarks_mediapipe"] = tensor_vis_landmarks(debug_views["img"], lmk_screen_mp, color="r", gt_landmarks=debug_views["landmarks_mediapipe"])
                    ## ============== visualize ==============================
                    visualize_grid(debug_rgb_pred, debug_cbuffers, debug_gbuffers, debug_views, avatar.images_save_path / f"grid_{iteration:04d}.png")
                    # visualize_grid_clean(debug_rgb_pred, debug_cbuffers, debug_gbuffer, debug_views, avatar.images_save_path / f"grid_{iteration:04d}_clean.png", flame=flame, faces=canonical_mesh.indices)

                    if visualize_uvs:
                        # Visualize UVs
                        uv_coords = debug_gbuffers["uvs"][0]
                        uv_coords = torch.cat((uv_coords, torch.zeros_like(uv_coords[...,0:1])), dim=-1) # make it rgB
                        outdir = avatar.images_save_path / "uv_coords"
                        outdir.mkdir(parents=True, exist_ok=True)
                        save_img(outdir / f"{iteration:05d}.png", uv_coords)

                    del debug_gbuffers, debug_cbuffers

                    # Send visualization to Weights & Biases
                    if args.wandb:
                        debug_rec_img = convert_uint(debug_rgb_pred[0], to_srgb=True)
                        wandb_data["reconstruction"] = wandb.Image(debug_rec_img, caption=f"reconstruction")

            # Log to Weights & Biases
            if args.wandb:
                wandb.log(wandb_data)

            # Save checkpoint
            if is_save_iter:
                avatar.save_all(f"{iteration:06d}", iteration)

    end = time.time()
    logging.info(f"Done! Time taken: {time.strftime('%H:%M:%S', time.gmtime(end - start))}")

    # Save final checkpoint
    avatar.save_all("latest", iteration)


if __name__ == "__main__":
    parser = config_parser()
    args = parser.parse_args()
    main(Avatar(args))
