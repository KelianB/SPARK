import logging

import torch
import nvdiffrast.torch as dr

from arguments import config_parser
from utils.dataset import to_device_recursive
from utils.visualization import rgb_to_srgb
from dataset import MultiVideoDataset
from Avatar import Avatar


@torch.no_grad()
def export_mesh(
    avatar: Avatar,
    tex_resolution: int,
    tex_type: str, # albedo | roughness | spec_int | baked
    out_dir: str,
    out_name: str,
    iteration: int,
    custom_pose=None,
    custom_expr=None,
    custom_seq_idx=0, # which sequence to use for the lighting with tex_type=baked
):
    device = avatar.device

    mesh, _ = Avatar.compute_displaced_mesh(avatar.canonical_mesh, avatar.displacements(), avatar.flame)
    verts = mesh.vertices # (V, 3)
    faces = mesh.indices # (F, 3)
    uvs = mesh._uv_coords # (F*3, 2)
    uvs_idx = mesh._uv_idx.to(device).int() # (F, 3)

    F = faces.shape[0]
    assert uvs.shape == (F*3, 2)
    assert uvs_idx.shape == (F, 3)

    uvs = uvs * 2 - 1
    uvs[..., 1] = -uvs[..., 1]
    # pad uvs
    uvs = torch.cat((uvs, torch.zeros_like(uvs[..., 0:1]), torch.ones_like(uvs[..., 0:1])), -1).view(1, -1, 4)

    rast, _ = dr.rasterize(avatar.renderer.glctx, uvs, uvs_idx, resolution=(tex_resolution,tex_resolution))
    mask = rast[..., -1].long() == 0

    rast_pos, _ = dr.interpolate(verts[faces].view(1, F*3, 3), rast, uvs_idx)

    pose = torch.zeros((1, 18), device=device, dtype=torch.float) if custom_pose is None else custom_pose
    expr = torch.zeros((1, avatar.deformer_net.num_exp), device=device, dtype=torch.float) if custom_expr is None else custom_expr

    if tex_type in ["albedo", "roughness", "spec_int"]:
        albedo, roughness, spec_int = avatar.shader.compute_material(rast_pos, iteration)   
        if tex_type == "albedo": tex = rgb_to_srgb(albedo)
        elif tex_type == "roughness": tex = roughness.repeat(1, 1, 1, 3)
        elif tex_type == "spec_int": tex = spec_int.repeat(1, 1, 1, 3)
    elif tex_type == "baked":
        # Compute shading in UV space
        views = to_device_recursive(MultiVideoDataset._collate([avatar.dataset_train[0]]), device)
        views["seq_idx"].fill_(custom_seq_idx)
        deformed_pos = Avatar.compute_deformed_verts(mesh, avatar.flame, pose, expr, avatar.deformer_net)[0]
        def_normals = mesh.fetch_all_normals(deformed_pos, mesh)
        gbuffers = {
            "position": dr.interpolate(deformed_pos[:,faces].view(1, F*3, 3), rast, uvs_idx)[0],
            "canonical_position": rast_pos,
            "mask": ~mask.unsqueeze(-1),
            "rast": rast,
            "deformed_verts_clip_space": uvs,
            "shading_normals": dr.interpolate(def_normals["vertex_normals"][:,faces].view(1, F*3, 3), rast, uvs_idx)[0],
        }
        render = avatar.shader.forward(gbuffers, views, mesh, iteration)[0]
        tex = rgb_to_srgb(render)
    else:
        raise ValueError(f"--tex_type '{tex_type}' is invalid")

    tex[mask] = 0
    tex = tex.squeeze(0)

    if custom_pose is not None or custom_expr is not None:
        mesh.vertices = Avatar.compute_deformed_verts(mesh, avatar.flame, pose, expr, avatar.deformer_net)[0].squeeze(0)
    mesh.write(avatar.experiment_dir / out_dir / out_name, texture=tex)


if __name__ == "__main__":
    parser = config_parser()
    parser.add_argument("--out_dir", type=str, default="export_mesh", help="Output directory, relative to the experiment dir or absolute")
    parser.add_argument("--out_name", type=str, default="mesh.obj", help="Output file name")
    parser.add_argument("--tex_resolution", type=int, default=512, help="Texture resolution")
    parser.add_argument("--tex_type", type=str, default="albedo", choices=["albedo", "roughness", "spec_int", "baked"], help="Which texture to output")
    parser.add_argument("--baked_lighting_seq", type=int, default=0, help="Which sequence to use for the lighting when tex_type='baked'")
    args = parser.parse_args()
    avatar = Avatar(args)

    custom_pose, custom_expr = None, None
    # enable this for custom poses and expressions
    if False:
        custom_pose = torch.zeros((1, 18), device=avatar.device, dtype=torch.float)
        custom_expr = torch.zeros((1, 50), device=avatar.device, dtype=torch.float)
        # rot.x, rot.y, rot.z, neck.x, neck.y, neck.z, jaw.x, jaw.y, jaw.z, eyeR.x, eyeR.y, eyeR.z, eyeL.x, eyeL.y, eyeL.z, trans.x, trans.y, trans.z
        custom_pose[:, 4] = 0.3
        custom_pose[:, 6] = 0.25

    logging.info("Exporting mesh...")
    export_mesh(avatar, args.tex_resolution, args.tex_type, args.out_dir, args.out_name, iteration=args.resume,
                custom_pose=custom_pose, custom_expr=custom_expr, custom_seq_idx=args.baked_lighting_seq)
