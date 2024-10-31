import logging
import os
import json
import torch
from pytorch3d.io import load_obj
import numpy as np
from tqdm import tqdm

from flare.core import Mesh

"""
This is unused code for experimenting with blendshape rigs other than FLAME.
"""

def load_blendshapes(blendshapes_file, blendshapes_neutral_file, blendshapes_names_file, device):
    if not os.path.exists(blendshapes_file):
        raise ValueError(f"Cannot load blendshapes at {blendshapes_file}")

    # Load blendshapes
    blendshapes = np.load(blendshapes_file) # shape (91, 48295, 3)
    blendshapes = torch.from_numpy(blendshapes).to(device).float()

    # Apply transform to blendshapes
    scale = 0.388643
    rotMatrix = torch.tensor([[ 9.9996e-01,  9.3457e-03,  2.4662e-04], [-9.3361e-03,  9.9684e-01,  7.8824e-02], [ 4.9083e-04, -7.8823e-02,  9.9689e-01]], dtype=torch.float, device=device)
    blendshapes = (scale * blendshapes) @ rotMatrix
    blendshapes = blendshapes.permute((1, 2, 0)) # (48295, 3, 91)

    blendshapes_neutral_verts, blendshapes_neutral_faces, _ = load_obj(blendshapes_neutral_file, load_textures=False)
    blendshapes_neutral_verts = blendshapes_neutral_verts.to(device)
    blendshapes_neutral_faces = blendshapes_neutral_faces.verts_idx.to(device)
    blendshapes_neutral = Mesh(blendshapes_neutral_verts, blendshapes_neutral_faces, device=device)

    logging.info(f"Blendshapes shapedirs: {blendshapes.shape}, neutral verts: {blendshapes_neutral_verts.shape}")

    with open(blendshapes_names_file, "r") as bs_names:
        blendshapes_names = json.load(bs_names)
    
    return blendshapes, blendshapes_neutral, blendshapes_names

def FLAME_to_blendshape_coefficients(expr, pose, mode, flame, canonical_mesh, deformer_net):
    from Avatar import Avatar
    logging.info("Converting FLAME expression coefficients to blendshapes coefficients")

    if mode == "flame2bs_invert":
        # Convert expression coefficients from FLAME basis to blendshapes
        # This is a very numerically unstable method!
        # Blendshapes: Bw + f0 / PCA: Uc + e0
        # f0 = e0 (because we aligned the blendshapes neutral to the PCA mean)
        # Bw = Uc <=> w = (B.t B)^-1 B.t Uc
        n_verts = canonical_mesh.vertices.shape[0]
        U = flame.shapedirs_expression.reshape(3*n_verts, -1) # (3*5023, 50)
        B, _, _ = deformer_net.query_weights(canonical_mesh.vertices)
        B = B.reshape(3*n_verts, -1) # (3*5023, 91)
        M = torch.linalg.inv(B.T @ B) @ B.T
        pca_coeffs_to_bs = lambda c: (M @ (U@c.unsqueeze(-1))).squeeze(-1)     
        new_expr = torch.stack([pca_coeffs_to_bs(e) for e in expr])
        return new_expr
    elif mode =="flame2bs_optim":
        epochs = 100
        batch_size = 32
        lr = 1e-3
        logging.info(f"Optimizing blendshapes coefficients for {epochs} epochs")

        with torch.no_grad():
            shapedirs, posedirs, lbs_weights = deformer_net.query_weights(canonical_mesh.vertices)
        
        n_frames = expr.size(0)
        new_expr = 0.15 * torch.ones((n_frames, deformer_net.num_exp),  dtype=torch.float, device=expr.device)

        # rough estimation of interpupillary distance
        rough_ipd = torch.linalg.vector_norm(canonical_mesh.vertices[flame.mask.v.left_eyeball].mean(0) - canonical_mesh.vertices[flame.mask.v.right_eyeball].mean(0))
        geometry_scale_mm = 62.0 / rough_ipd # scales geometry to millimeters, assuming an interpupillary distance of roughly 62mm

        new_expr = torch.nn.Parameter(new_expr)
        optimizer = torch.optim.Adam([new_expr], lr=lr)
        dataloader = torch.utils.data.DataLoader(range(n_frames), batch_size=batch_size, shuffle=True)
        
        # test batch
        test_indices = torch.randperm(n_frames)[:batch_size]

        def compute_distances(indices):
            pose_batch = pose[indices]
            expr_batch = expr[indices]
            new_expr_batch = new_expr[indices]
            # Calculate geometry with the FLAME coefficients and model
            with torch.no_grad():
                jaw_previous_state = flame.jaw_enabled
                flame.jaw_enabled = True
                deformed_vertices_flame, *_ = Avatar.compute_deformed_verts_FLAME(canonical_mesh, flame, pose_batch, expr_batch)
                flame.jaw_enabled = jaw_previous_state
            # Calculate geometry with the optimized coefficients and the blendshapes
            deformed_vertices, *_ = Avatar.compute_deformed_verts(canonical_mesh, flame, pose_batch, new_expr_batch, None,
                                                                 shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights)
            dists_L2 = torch.linalg.vector_norm(deformed_vertices_flame - deformed_vertices, dim=-1) * geometry_scale_mm
            return dists_L2

        threshold, alpha = 0.1, 50

        progress_bar = tqdm(range(epochs))
        for epoch in progress_bar:
            progress_bar.set_description(desc=f"Epoch {epoch+1}")
            with torch.no_grad():
                progress_bar.set_postfix({"avg_dist_L2": compute_distances(test_indices).mean().cpu().item()})
                if epoch == 0 or (epoch+1) % 5 == 0:
                    print()
                    expr_test = new_expr[test_indices]
                    print(f"average number of activated blendshapes: {(expr_test > threshold).sum() // len(test_indices)}")
                    print(f"average activation of activated blendshapes: {expr_test[expr_test > threshold].mean():.03f}")
                    print(f"average activation of deactivated blendshapes: {expr_test[expr_test <= threshold].mean():.03f}")

            for indices in dataloader:
                new_expr_batch = new_expr[indices]
                dists_L2 = compute_distances(indices)
                loss_L2 = dists_L2.mean() # mean over the vertices and batch
                loss_sparsity = new_expr_batch.clamp(min=0).mean()
                loss_softclamp_lower = new_expr_batch.clamp(max=0).pow(2).mean()
                #loss_softclamp_upper = (new_expr_batch.clamp(min=1) - 1).pow(2).mean()
                #loss_count = (new_expr_batch > 0.01).sum()
                loss_count = (1 / (1 + torch.exp(-alpha * (new_expr_batch - threshold)))).mean()

                loss = loss_L2 + loss_softclamp_lower + 0.01 * loss_sparsity + 0.01 * loss_count

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return new_expr
    else:
        raise ValueError(f"Unknown mode for FLAME_to_blendshape_coefficients: '{mode}'")
