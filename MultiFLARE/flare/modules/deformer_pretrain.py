import torch
from pytorch3d import ops
from tqdm import tqdm
import logging

from utils.geometry import barycentric_projection


def pretrain_deformer(args, deformer_net, flame, canonical_verts, mesh, device,
                      blendshapes=None, blendshapes_neutral=None, ignore_pose=False):
    gt_shapedirs = flame.shapedirs_expression
    gt_posedirs = flame.posedirs
    gt_lbs_weights = flame.lbs_weights

    average_edge_length = torch.linalg.norm(canonical_verts[mesh.edges[:,0]] - canonical_verts[mesh.edges[:,1]], dim=-1).mean()

    # Subdivide for training the deformer
    # if args.deformer_pretrain_subdivide:
    #     SUBDIVIDE_LEVELS = 2
    #     logging.info(f"Subdividing the canonical mesh {SUBDIVIDE_LEVELS} time{'s' if SUBDIVIDE_LEVELS > 1 else ''} for pre-training the deformer")
    #     P, V = gt_posedirs.shape[0], canonical_verts.shape[0]
    #     canonical_verts, _, subdivided_attrs = subdivide_mesh(canonical_verts, mesh.indices, vert_attrs={
    #         "shapedirs": gt_shapedirs,
    #         "lbs_weights": gt_lbs_weights,
    #         "posedirs": gt_posedirs.reshape(P, V, 3).permute(1, 0, 2), # (P, 3*V) to (V, P, 3)
    #     }, levels=SUBDIVIDE_LEVELS)
    #     V = canonical_verts.shape[0]
    #     gt_shapedirs = subdivided_attrs["shapedirs"]
    #     gt_lbs_weights = subdivided_attrs["lbs_weights"]
    #     gt_posedirs = subdivided_attrs["posedirs"].permute(1, 0, 2).reshape(P, 3*V) # (V, P, 3) to (P, 3*V) 

    use_blendshapes = blendshapes is not None
    if use_blendshapes:
        dists, idx, _ = ops.knn_points(canonical_verts.unsqueeze(0), blendshapes_neutral.vertices.unsqueeze(0), K=1) # (1, 5023, 1)
        dists = dists.squeeze(0).squeeze(-1) # (5023)
        idx = idx.squeeze(0).squeeze(-1) # (5023)
        close_mask = dists < average_edge_length
        print(f"Mask size for blendshapes supervision is {torch.count_nonzero(close_mask)}/{canonical_verts.size(0)}")

        NEAREST, BARYCENTRIC = 0, 1
        projection_mode = NEAREST
        if projection_mode == NEAREST:
            # Pick blendshape values at the closest vertex in the rig's neutral mesh
            close_idx = idx[close_mask]
            blendshapes_gt = blendshapes[close_idx]
        elif projection_mode == BARYCENTRIC:
            # Pick blendshape values by projecting points on the rig's neutral mesh using barycentric coordinates
            closest_face_indices, barycentric_coordinates = barycentric_projection(blendshapes_neutral.vertices, blendshapes_neutral.indices, canonical_verts[close_mask])
            closest_verts_idx = blendshapes_neutral.indices[closest_face_indices]
            blendshapes_gt = torch.einsum("vfib,vf->vib", [blendshapes[closest_verts_idx], barycentric_coordinates]) # (n, 3, 91)
        else:
            raise ValueError()

    max_iters = args.deformer_pretrain
    lr = 2e-4

    deformer_net = deformer_net.to(device)
    deformer_net.train()

    logging.info(f"Pre-training the deformer for {max_iters} iterations")
    optimizer = torch.optim.Adam(list(deformer_net.parameters()), lr=lr)

    progress_bar = tqdm(range(max_iters))
    for iteration in range(max_iters+1):
        progress_bar.set_description(desc=f"Iter {iteration}")
        shapedirs, posedirs, lbs_weights = deformer_net.query_weights(canonical_verts if deformer_net.input == "canonical_pos" else flame.uvs)

        # Just use a L1 loss - L2 did not work as well for this
        if use_blendshapes:
            # Supervise with blendshapes where we are close to the rig's neutral shape
            loss_shapedirs_close = (shapedirs[close_mask] - blendshapes_gt).abs().sum(-1).sum(-1).mean() # sum over n_exp and xyz then mean over verts
            # Enforce zeros outside of the rig
            loss_shapedirs_far = shapedirs[~close_mask].abs().sum(-1).sum(-1).mean() # sum over n_exp and xyz then mean over verts
            loss_shapedirs = loss_shapedirs_close + loss_shapedirs_far
        else:
            loss_shapedirs = (shapedirs - gt_shapedirs).abs().sum(-1).sum(-1).mean() # sum over n_exp and xyz then mean over verts 
        
        loss = 0.0
        
        loss += loss_shapedirs
        if not ignore_pose:
            loss_posedirs = (posedirs - gt_posedirs).abs().view(36, -1, 3).sum(-1).sum(0).mean() # sum over xyz and pose features then mean over verts
            loss_lbsw = (lbs_weights - gt_lbs_weights).abs().sum(-1).mean() # sum over joints then mean over verts
            loss += loss_posedirs + loss_lbsw

        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to avoid an exploding gradients issue
        #torch.nn.utils.clip_grad_norm_(deformer_net.parameters(), True)
        
        optimizer.step()

        postfix = {"loss": loss.detach().cpu().item(), "loss_shapedirs": loss_shapedirs.detach().cpu().item()}
        if not ignore_pose:
            postfix["loss_posedirs"] = loss_posedirs.detach().cpu().item()
            postfix["loss_lbsw"] = loss_lbsw.detach().cpu().item()
        if use_blendshapes:
            postfix["loss_shapedirs_close"] = loss_shapedirs_close.detach().cpu().item()
            postfix["loss_shapedirs_far"] = loss_shapedirs_far.detach().cpu().item()

        progress_bar.set_postfix(postfix)
