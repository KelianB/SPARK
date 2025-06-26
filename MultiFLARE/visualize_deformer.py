import logging
import itertools

import torch
import matplotlib.pyplot as plt

from arguments import config_parser
from dataset import MultiVideoDataset
from utils.dataset import to_device_recursive
from utils.visualization import save_img, arrange_grid
from flare.core import Mesh
from Avatar import Avatar

def main(avatar: Avatar):   
    args = avatar.args
    device, dataset_train, flame, canonical_mesh, deformer_net, renderer, cams_K_train = \
        avatar.device, avatar.dataset_train, avatar.flame, avatar.canonical_mesh, avatar.deformer_net, avatar.renderer, avatar.cams_K_train

    template_verts = flame.canonical_verts.squeeze(0)
    flame_template_mesh = Mesh(template_verts, flame.faces, device=device)

    n_verts_initial = template_verts.shape[0] # before any upsampling
    n_verts = canonical_mesh.vertices.shape[0]
    n_exp_deformer = deformer_net.num_exp
    n_exp = flame.n_exp
    n_joints = flame.n_joints
    visualization_scale_expr = 100
    visualization_scale_posedirs = 100
    visualization_scale_lbs = 1
    view = dataset_train.collate([dataset_train.__getitem__(40)])
    view = to_device_recursive(view, device)

    visualization_dir = avatar.images_save_path / f"visu_deformer_{args.resume}"
    visualization_dir.mkdir(parents=True, exist_ok=True) 

    def render_heatmap(filename, img):
        save_img(visualization_dir / filename, img)

    def cov(a, b):
        a, b = a.flatten(), b.flatten()
        return a.dot(b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))

    # ==============================================================================================
    # COMPUTE HEATMAPS
    # ==============================================================================================
    print("=="*50)

    MultiVideoDataset.override_values_batch(view, per_seq_cam_K=cams_K_train)

    zero_pose = torch.zeros_like(view["flame_pose"])
    zero_pose[:,17] = -0.7 # z translation
    zero_expr = torch.zeros_like(view["flame_expression"])
    zero_expr_deformer = torch.zeros((1, n_exp_deformer), dtype=torch.float, device=device)

    with torch.no_grad():
        logging.info("Computing deformations")
        # Deformations (with deformer and with default FLAME basis)
        deformed_vertices, deformer_lbs_weights, deformer_shapedirs, deformer_posedirs = Avatar.compute_deformed_verts(canonical_mesh, flame, zero_pose, zero_expr_deformer, deformer_net)
        flame_deformed_vertices, flame_lbs_weights, flame_shapedirs, flame_posedirs = Avatar.compute_deformed_verts_FLAME(flame_template_mesh, flame, zero_pose, zero_expr)

        # Create blendshapes covariance matrix
        logging.info("Computing covariance of blendshapes bases")
        flame_cov_matrix = torch.zeros((n_exp, n_exp), device=device, dtype=torch.float)
        for i,j in itertools.product(range(n_exp), range(n_exp)):
            flame_cov_matrix[i,j] = cov(flame_shapedirs[...,i], flame_shapedirs[...,j])
        deformer_cov_matrix = torch.zeros((n_exp_deformer, n_exp_deformer), device=device, dtype=torch.float)
        for i,j in itertools.product(range(n_exp_deformer), range(n_exp_deformer)):
            deformer_cov_matrix[i,j] = cov(deformer_shapedirs[...,i], deformer_shapedirs[...,j])
        # Visualize covariance
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))       
        fig.suptitle("Blendshapes covariance plot")
        fig.text(0.5, 0.05, "The i, j entry is the covariance between the i-th and j-th blendshape basis vectors", wrap=True, horizontalalignment="center")
        ax0.set_title("Initial FLAME basis")
        ax0.matshow(flame_cov_matrix.cpu())
        ax1.set_title("Learned deformer basis")
        ax1.matshow(deformer_cov_matrix.cpu())
        fig.savefig(visualization_dir / "blendshapes_covariance.png")

        extra_rast_attrs = dict()
        extra_rast_attrs_flame = dict()

        # (P,3*V) => (P,V,3)
        deformer_posedirs = deformer_posedirs.view(-1, n_verts, 3)
        flame_posedirs = flame_posedirs.view(-1, n_verts_initial, 3)

        if not args.color_xyz:
            # Convert all deformations to amplitudes
            flame_shapedirs = torch.linalg.vector_norm(flame_shapedirs, ord=2, dim=1, keepdim=True)
            deformer_shapedirs = torch.linalg.vector_norm(deformer_shapedirs, ord=2, dim=1, keepdim=True)
            flame_posedirs = torch.linalg.vector_norm(flame_posedirs, ord=2, dim=-1, keepdim=True)
            deformer_posedirs = torch.linalg.vector_norm(deformer_posedirs, ord=2, dim=-1, keepdim=True)

        # (P, V, 1|3) to (V, 4, 9, 1|3)
        deformer_posedirs = deformer_posedirs.permute(1, 0, 2).view(n_verts, n_joints-1, 9, -1) # (P,V,3) => (V,P,3) => (V,J,9,3)
        flame_posedirs = flame_posedirs.permute(1, 0, 2).view(n_verts_initial, n_joints-1, 9, -1) # (P,V,3) => (V,P,3) => (V,J,9,3)

        for exp_i in range(n_exp):
            extra_rast_attrs_flame[f"flame_expr_{exp_i}"] = flame_shapedirs[..., exp_i].contiguous()
        for exp_i in range(n_exp_deformer):
            extra_rast_attrs[f"deformer_expr_{exp_i}"] = deformer_shapedirs[..., exp_i].contiguous()

        for j in range(n_joints):
            extra_rast_attrs[f"deformer_lbs_{j}"] = deformer_lbs_weights[...,(j,)].contiguous()
            extra_rast_attrs_flame[f"flame_lbs_{j}"] = flame_lbs_weights[...,(j,)].contiguous()
            if j < n_joints-1:
                extra_rast_attrs[f"deformer_posedirs_{j}"] = deformer_posedirs[:,j].mean(dim=1).contiguous() # mean over all 9 coeffs corresponding to a joint's rot matrix
                extra_rast_attrs_flame[f"flame_posedirs_{j}"] = flame_posedirs[:,j].mean(dim=1).contiguous()

        logging.info("Rasterizing")
        # Rasterization (deformer)
        gbuffers = renderer.render_batch(view["camera"], deformed_vertices.contiguous(), deformed_normals=None, channels=[], with_antialiasing=False,
                                         canonical_v=None, canonical_idx=canonical_mesh.indices, extra_rast_attrs=extra_rast_attrs)
        # Rasterization (FLAME)
        gbuffers_flame = renderer.render_batch(view["camera"], flame_deformed_vertices.contiguous(), deformed_normals=None, channels=[], with_antialiasing=False,
                                               canonical_v=None, canonical_idx=flame.faces, extra_rast_attrs=extra_rast_attrs_flame)

        logging.info("Rendering heatmaps for expression basis")
        deformer_expr_imgs = [gbuffers[f"deformer_expr_{e}"].squeeze(0) for e in range(n_exp_deformer)]
        flame_expr_imgs = [gbuffers_flame[f"flame_expr_{e}"].squeeze(0) for e in range(n_exp)]
        render_heatmap("expr_deformer.png", arrange_grid(deformer_expr_imgs, 10) * visualization_scale_expr)
        render_heatmap("expr_FLAME.png",    arrange_grid(flame_expr_imgs,    10) * visualization_scale_expr)

        logging.info("Rendering heatmaps for pose corrective basis")
        deformer_posedirs_imgs = [gbuffers[f"deformer_posedirs_{j}"].squeeze(0) for j in range(n_joints-1)]
        flame_posedirs_imgs = [gbuffers_flame[f"flame_posedirs_{j}"].squeeze(0) for j in range(n_joints-1)]
        render_heatmap("posedirs_deformer.png", arrange_grid(deformer_posedirs_imgs, n_joints-1) * visualization_scale_posedirs)
        render_heatmap("posedirs_FLAME.png",    arrange_grid(flame_posedirs_imgs,    n_joints-1) * visualization_scale_posedirs)

        logging.info("Rendering heatmaps for LBS weights")
        deformer_lbs_imgs = [gbuffers[f"deformer_lbs_{j}"].squeeze(0) for j in range(n_joints)]
        flame_lbs_imgs = [gbuffers_flame[f"flame_lbs_{j}"].squeeze(0) for j in range(n_joints)]
        render_heatmap("lbs_deformer.png", arrange_grid(deformer_lbs_imgs, n_joints) * visualization_scale_lbs)
        render_heatmap("lbs_FLAME.png",    arrange_grid(flame_lbs_imgs,    n_joints) * visualization_scale_lbs)

    print("=="*50)

if __name__ == '__main__':
    parser = config_parser()
    parser.add_argument("--color_xyz", action="store_true", help="Use RGB to represent xyz displacements instead of showing amplitudes.")
    args = parser.parse_args()

    #################### Validate args ####################
    if args.resume is None:
        raise ValueError("Arg --resume is required")

    main(Avatar(args))
