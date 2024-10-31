import os
import torch
from tqdm import tqdm

from arguments import config_parser
from dataset import MultiVideoDataset
from utils.dataset import SemanticMask, DeviceDataLoader, to_device_recursive
from utils.visualization import visualize_grid, visualize_grid_clean, tensor_vis_landmarks, save_img, arrange_grid
from flare.core import Mesh
from Avatar import Avatar

TEST_EXPORT_MESH = False
TEST_FORWARD_TIMING = False
TEST_LANDMARKS = False
TEST_VISUALIZE_FEW = False
TEST_VISUALIZE_ALL = False
TEST_MASKS = False
TEST_SPECIFIC = False
TEST_LIGHTING = False
TEST_SEQUENCE_LIGHTINGS = False
TEST_NEUTRAL_ROTATING = False
TEST_NEUTRAL_STATIC = True
TEST_EXPRESSIONS = False

@torch.no_grad()
def main(avatar: Avatar):
    args = avatar.args
    device, dataset_train, canonical_mesh, cams_K_train, pose_train, expr_train, images_save_path, flame = \
        avatar.device, avatar.dataset_train, avatar.canonical_mesh, avatar.cams_K_train, avatar.pose_train, avatar.expr_train, avatar.images_save_path, avatar.flame

    iteration = args.resume
    out_dir = images_save_path.parent / "visualize"
    out_dir.mkdir(parents=True, exist_ok=True)

    if TEST_EXPORT_MESH:
        pose = torch.zeros((15), device=device, dtype=torch.float)
        expr = torch.zeros((avatar.deformer_net.num_exp), device=device, dtype=torch.float)
        # expr[26] = 0.7
        # expr[73] = 0.7
        # expr[41] = 0.5
        # expr[88] = 0.5
        # expr[90] = 0.5
        # pose[6] = 0.2
        pose = pose_train[724]
        expr = expr_train[724]
        canonical_mesh.vertices = Avatar.compute_deformed_verts(canonical_mesh, avatar.flame, pose.unsqueeze(0), expr.unsqueeze(0), avatar.deformer_net)[0].squeeze(0)
        canonical_mesh.write(out_dir / "test.obj")
        exit(0)

    if TEST_FORWARD_TIMING:
        with torch.no_grad():
            from time import time
            dataloader = DeviceDataLoader(dataset_train, device=device, batch_size=args.batch_size,
                                                        collate_fn=dataset_train.collate, shuffle=True, drop_last=True, num_workers=4)
            total_time = 0
            for views in tqdm(dataloader):
                time_before = time()
                MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
                avatar.run(views, args.resume)
                total_time += time() - time_before
            print(f"average forward pass time per frame: {total_time*1000 / len(dataset_train):.02f} ms")

    if TEST_MASKS:
        masks_dir = out_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        dataloader = DeviceDataLoader(dataset_train, device=device, batch_size=8, collate_fn=dataset_train.collate, num_workers=4)
        mask_idx = SemanticMask.ALL_SKIN
        for i,views in tqdm(enumerate(dataloader), total=len(dataloader)):
            masks = views["semantic_mask"][..., (mask_idx, mask_idx, mask_idx)]
            image = torch.cat([torch.cat((img, mask), dim=0) for img, mask in zip(views["img"], masks)], dim=1)
            save_img(masks_dir / f"{i:03d}.png", image)

    if TEST_LANDMARKS:
        #view_indices = [724]
        #view_indices = [1323, 202, 1286, 1121, 1280, 1282, 1281, 1513, 62, 1287, 314, 242, 1327, 491, 1299, 631, 1328, 398, 1325, 1315]
        #view_indices = list(range(0, len(dataset_train), 10)) # every 10th frame
        view_indices = list(range(len(dataset_train))) # all frames
        draw_indices = False

        dataset_subset = torch.utils.data.Subset(dataset_train, view_indices)
        dataloader = DeviceDataLoader(dataset_subset, device=device, batch_size=8, collate_fn=dataset_train.collate, shuffle=False, drop_last=False, num_workers=4)

        for views in dataloader:
            #views = MultiVideoDataset._collate([dataset_train[idx] for idx in view_indices]) 
            #views = to_device_recursive(views, device)
            MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
            rgb_pred, _, _, _, _, deformed_vertices, *_ = avatar.run(views, iteration)            

            if False:
                pose = pose_train[views["idx"]]
                # FAN landmarks, detections+proj | proj
                landmarks_pos = flame.get_landmark_positions_2d(deformed_verts, pose)
                lmk_screen = avatar.renderer.to_screen_space(landmarks_pos, views["camera"]) / 256 # (B, 70, 2)
                
                vis_landmarks_gt = tensor_vis_landmarks(views["img"], lmk_screen, color="r", gt_landmarks=views["landmarks"], gt_labels=draw_indices)
                vis_landmarks_render = tensor_vis_landmarks(rgb_pred, lmk_screen, color="r")
                vis_img = [torch.cat((gt, r), dim=1) for gt, r in zip(vis_landmarks_gt, vis_landmarks_render)]
            elif False:
                # MediaPipe landmarks, detections+proj | proj
                landmarks_pos_mp = flame.get_landmark_positions(deformed_verts, which="static_mediapipe")
                lmk_screen_mp = avatar.renderer.to_screen_space(landmarks_pos_mp, views["camera"]) / 256 # (B, L, 2)

                vis_landmarks_gt = tensor_vis_landmarks(views["img"], lmk_screen_mp, color="r", gt_landmarks=views["landmarks_mediapipe"], gt_labels=draw_indices)
                vis_landmarks_render = tensor_vis_landmarks(rgb_pred, lmk_screen_mp, color="r")
                vis_img = [torch.cat((gt, r), dim=1) for gt, r in zip(vis_landmarks_gt, vis_landmarks_render)]
            elif True:
                # MediaPipe landmarks, detections only
                vis_landmarks_gt = tensor_vis_landmarks(views["img"], views["landmarks_mediapipe"], color="g")
                vis_img = vis_landmarks_gt

            idx_str = "_".join([str(i.item()) for i in views["idx"]])
            save_img(out_dir / f"test_lmk_{idx_str}.png", arrange_grid(vis_img, 1), to_srgb=True)

    if TEST_VISUALIZE_FEW:
        view_indices = [724]
        views = MultiVideoDataset._collate([dataset_train[idx] for idx in view_indices]) 
        views = to_device_recursive(views, device)
        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
        rgb_pred, gbuffers, cbuffers, *_ = avatar.run(views, iteration)
        visualize_grid(rgb_pred, cbuffers, gbuffers, views, out_dir / f"grid_test{iteration:04d}.png")

    if TEST_VISUALIZE_ALL:
        # view_indices = [dataset_train.frames_per_seq[2][35], dataset_train.frames_per_seq[2][35+4*8]]
        view_indices = list(range(0, len(dataset_train), 4))
        dataset_subset = torch.utils.data.Subset(dataset_train, view_indices)
        batch_size = 4
        dataloader = DeviceDataLoader(dataset_subset, device=device, batch_size=batch_size, collate_fn=dataset_train.collate, shuffle=False, drop_last=False, num_workers=4)

        for i, views in tqdm(enumerate(dataloader), total=len(dataloader)):
            MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
            rgb_pred, gbuffers, cbuffers, *_ = avatar.run(views, iteration)
            visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / f"grid_{i:04d}.png", flame=flame, faces=canonical_mesh.indices)           

    if TEST_SPECIFIC:
        # view_indices = list(range(215*4 - 4, 215*4 + 4))
        view_indices = [862]
        views = MultiVideoDataset._collate([dataset_train[idx] for idx in view_indices]) 
        views = to_device_recursive(views, device)
        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
        views["flame_pose"] *= 0
        views["flame_pose"][:,-1] = -0.67 # z translation
        views["flame_expression"] *= 0

        shapedirs, posedirs, lbs_weights = avatar.deformer_net.query_weights(canonical_mesh.vertices)

        rgb_pred, gbuffers, cbuffers, _, _, deformed_vertices, *_ = avatar.run(views, iteration, posedirs=posedirs, lbs_weights=lbs_weights, shapedirs=shapedirs)
        # rast = debug_gbuffer["rast"]
        visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / "grid_test.png", flame=flame, faces=canonical_mesh.indices)           
        if False:
            mesh = canonical_mesh.with_vertices(deformed_vertices[0])
            mesh.write(out_dir / f"mesh.obj")                                

    if TEST_LIGHTING:
        # view_indices = [dataset_train.frames_per_seq[i][0] for i in range(avatar.num_seq)]
        view_indices = [(27*4+2)*4, (54*4+1)*4]
        views = MultiVideoDataset._collate([dataset_train[idx] for idx in view_indices]) 
        views = to_device_recursive(views, device)
        # views["seq_idx"] = torch.arange(avatar.num_seq, dtype=torch.long, device=views["seq_idx"].device)
        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
        shapedirs, posedirs, lbs_weights = avatar.deformer_net.query_weights(canonical_mesh.vertices)

        # pose = torch.zeros_like(views["flame_pose"])
        # pose[..., -1] = -2.5
        # expr = torch.zeros_like(views["flame_expression"])
        # s = 1.0
        # expr[..., 0] = s*2
        # expr[..., 1] = s
        # expr[..., 2] = -s*5
        # expr[..., 3] = s
        # expr[..., 4] = -s
        pose = views["flame_pose"]
        expr = views["flame_expression"]

        deformed_vertices, *_ = Avatar.compute_deformed_verts(canonical_mesh, flame, pose, expr, avatar.deformer_net,
                                                         shapedirs=shapedirs, posedirs=posedirs, lbs_weights=lbs_weights) 
        rgb_pred, gbuffers, cbuffers, *_ = avatar.run(views, iteration, deformed_vertices=deformed_vertices)
        visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / "grid_test.png", flame=flame, faces=canonical_mesh.indices)           

        for i, v in enumerate(deformed_vertices):
            mesh = canonical_mesh.with_vertices(v)
            mesh.write(out_dir / f"test_posed_mesh_{views['idx'][i]}.obj") 

    if TEST_SEQUENCE_LIGHTINGS:
        view_indices = [seq_frames[0] for seq_frames in dataset_train.frames_per_seq]
        views = dataset_train.collate([dataset_train.__getitem__(idx) for idx in view_indices])
        views = to_device_recursive(views, device)

        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)

        head_scale = (canonical_mesh.vertices[:,0].max() - canonical_mesh.vertices[:,0].min()).item()

        # Replace the face with a sphere
        from pytorch3d.utils import ico_sphere
        sphere_mesh_p3d = ico_sphere(5)
        sphere_verts = sphere_mesh_p3d.verts_list()[0] * head_scale # (V, 3)
        sphere_faces = sphere_mesh_p3d.faces_list()[0] # (F, 3)
        
        sphere_mesh = Mesh(sphere_verts, sphere_faces, device)
        sphere_mesh.compute_connectivity()
        deformed_vertices = sphere_verts.unsqueeze(0).repeat(len(views["idx"]), 1, 1).to(device)
        deformed_vertices[:,:,-1] -= 1.1 # z translation

        rgb_pred, gbuffers, cbuffers, *_ = avatar.run(views, iteration, mesh=sphere_mesh, deformed_vertices=deformed_vertices)
        visualize_grid(rgb_pred, cbuffers, gbuffers, views, out_dir / "grid_lighting.png")

    if TEST_NEUTRAL_ROTATING:
        views = MultiVideoDataset._collate([dataset_train[0]]) 
        views = to_device_recursive(views, device)
        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
        views["flame_pose"] *= 0
        views["flame_pose"][:,-1] = -0.67 # z translation
        views["flame_expression"] *= 0
        views["flame_pose"][:,6] = 0.1 # open jaw

        min_angle = -40 * torch.pi/180
        max_angle = 40 * torch.pi/180

        duration = 8
        fps = 24
        n_frames = duration * fps
        for f in range(n_frames):
            p = f / n_frames
            p = 2 * ((1-p) if p > 0.5 else p) # do a round trip
            views["flame_pose"][:,1] = min_angle + (max_angle - min_angle) * p
            rgb_pred, gbuffers, cbuffers, *_ = avatar.run(views, iteration)
            visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / f"grid_neutral_{f:04d}.png", flame, canonical_mesh.indices)

        os.system(f"/usr/bin/ffmpeg -framerate {fps} -i {out_dir / 'grid_neutral_%004d.png'} -c:v libx264 -pix_fmt yuv420p -vf \"crop=512:512:2560:0\" -y {out_dir / 'neutral_vis.mp4'}")

    if TEST_NEUTRAL_STATIC:
        views = MultiVideoDataset._collate([dataset_train[0]]) 
        views = to_device_recursive(views, device)
        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
        views["flame_pose"] *= 0
        # views["flame_pose"][:,-2] = 0.05 # y translation
        # views["flame_pose"][:,-1] = -0.67 * 4
        views["flame_pose"][:,-2] = 0.01 # y translation
        views["flame_pose"][:,-1] = -0.7 # z translation
        views["flame_expression"] *= 0
        # views["flame_pose"][:,6] = 0.1 # open jaw

        rgb_pred, gbuffers, cbuffers, *_ = avatar.run(views, iteration)
        visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / "grid_neutral.png", flame, canonical_mesh.indices)

    if TEST_EXPRESSIONS:
        """
        view_indices = dataset_train.frames_per_seq[0]
        dataset_subset = torch.utils.data.Subset(dataset_train, view_indices)
        dataloader = DeviceDataLoader(dataset_subset, device=device, batch_size=1, collate_fn=dataset_train.collate, shuffle=False, drop_last=False, num_workers=4)

        fps = 24
        for i, views in tqdm(enumerate(dataloader), total=len(dataloader)):
            MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
            jaw_pose = views["flame_pose"][:, 6:9].clone()
            views["flame_pose"] *= 0
            views["flame_pose"][:,-1] = -0.67 # z translation
            views["flame_pose"][:, 6:9] = jaw_pose
            rgb_pred, gbuffers, cbuffers, _, _, deformed_verts, *_ = avatar.run(views, iteration)            
            visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / f"grid_expr_{i:04d}.png", flame=flame, faces=canonical_mesh.indices)           
        """

        all_expr = torch.stack([dataset_train.get_flame_expression(i, device) for i in range(len(dataset_train))])
        all_poses = torch.stack([dataset_train.get_flame_pose(i, device) for i in range(len(dataset_train))])

        views = MultiVideoDataset._collate([dataset_train[0]]) 
        views = to_device_recursive(views, device)
        MultiVideoDataset.override_values_batch(views, pose_train, expr_train, cams_K_train)
        views["flame_pose"] *= 0
        views["flame_pose"][:,-1] = -0.67 # z translation
        views["flame_expression"] *= 0

        from math import floor, ceil
        duration = 16
        expressions_per_second = 0.8
        exaggerate = 1.5
        n_expressions = ceil(expressions_per_second * duration)
        # interpolation_ids = range(0, len(dataset_train), n_expressions)
        # interpolation_ids = sorted(range(len(dataset_train)), key=lambda i: all_expr[i].pow(2).sum(), reverse=True)[:n_expressions]
        # Cut into blocks and take the most expressive frame for each
        # interpolation_ids = []
        # for i in range(n_expressions):
        #     block_size = len(dataset_train) // n_expressions
        #     ids = range(i * block_size, (i+1) * block_size)
        #     interpolation_ids.append(sorted(ids, key=lambda i: all_expr[i].pow(2).sum(), reverse=True)[0])
        cluster_features = torch.cat((all_poses[:, 6:9], all_expr[:,:10]), dim=-1) # cluster based on jaw pose and first expression components
        cluster_features = (cluster_features - cluster_features.mean(0)) / (1e-8 + cluster_features.std(0))
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_expressions, random_state=0, n_init="auto")
        kmeans.fit(cluster_features.cpu().numpy())
        interpolation_ids = []
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(cluster_features.device)
        for x in cluster_centers:
            # Select the closest frame to the cluster's center
            closest_idx = min(range(len(dataset_train)), key=lambda i: torch.linalg.vector_norm(cluster_features[i] - x, dim=-1))
            interpolation_ids.append(closest_idx)

        fps = 24
        n_frames = duration * fps
        for f in range(n_frames):
            interp_i = (len(interpolation_ids)-1) * f / n_frames
            i_prev, i_next = interpolation_ids[floor(interp_i)], interpolation_ids[ceil(interp_i)]
            p = interp_i % 1
            # lerp jaw pose and expression
            views["flame_pose"][:, 6:9] = (all_poses[i_prev, 6:9] + p * (all_poses[i_next, 6:9] - all_poses[i_prev, 6:9])) * exaggerate
            views["flame_expression"][:] = (all_expr[i_prev] + p * (all_expr[i_next] - all_expr[i_prev])) * exaggerate
            rgb_pred, gbuffers, cbuffers, _, _, deformed_verts, *_ = avatar.run(views, iteration)            
            visualize_grid_clean(rgb_pred, cbuffers, gbuffers, views, out_dir / f"grid_expr_{f:04d}.png", flame=flame, faces=canonical_mesh.indices)           

        # os.system(f"/usr/bin/ffmpeg -framerate {fps} -i {out_dir / 'grid_expr_%004d.png'} -c:v libx264 -pix_fmt yuv420p -vf \"crop=512:512:2560:0\" -y {out_dir / 'expr.mp4'}")
        os.system(f"/usr/bin/ffmpeg -framerate {fps} -i {out_dir / 'grid_expr_%004d.png'} -c:v libx264 -pix_fmt yuv420p -y {out_dir / 'expr.mp4'}")


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    args.deformer_pretrain = 0

    main(Avatar(args))
