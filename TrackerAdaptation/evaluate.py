from tqdm import tqdm
import logging
from typing import Dict

import torch
from torch.utils.data import Subset
from torchvision.utils import _normalized_flow_to_image

from main import spark_setup, spark_config_parser, parse_args
from adapt.wrapper import FaceTrackerWrapper
from adapt.metrics import img_psnr, img_ssim, img_mse, img_IoU
from adapt.general_utils import get_optical_flow_model, compute_optical_flows

# MultiFLARE imports
from utils.dataset import DeviceDataLoader, ZipDatasets, SemanticMask, find_collate
from utils.color import rgb_to_srgb, srgb_to_rgb
from utils.visualization import save_img
from flame.FLAME import MEDIAPIPE_LMK_EMBEDDING_INDICES

# Visualization stuff
SHOW_OCCLUSIONS = True
VISUALIZATION_ERROR_SCALE = 5.0

WARP_METRICS = ["MSE", "PSNR", "SSIM"]
OPTICAL_FLOW = False
FLOW_METRICS = ["L1", "MSE"]

# This controls the supersampling scale used for calculating occlusions.
# it's optional, but the occlusion masks will be less noisy.
OCCLUSION_SUPERSAMPLE = 4 

@torch.no_grad()
def main(wrapper: FaceTrackerWrapper, args, dataset):
    device = wrapper.device
    faces = wrapper.decoder.faces

    out_dir = args.out_dir / (f"evaluation_warp_{args.tracker_resume:04d}_{args.eval_dataset}" + f"_int{args.frame_interval}" + ("_flow" if OPTICAL_FLOW else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if OPTICAL_FLOW:
        model_optical_flow, preprocess_img_optical_flow = get_optical_flow_model(device)

    make_img_row = lambda x: torch.cat(tuple(x), dim=1)
    make_img_col = lambda x: torch.cat(tuple(x), dim=0)

    # Create a zip dataset with (previous, current) pairs
    dataset_frame = Subset(dataset, torch.arange(args.frame_interval, len(dataset))) 
    dataset_prev = Subset(dataset, torch.arange(0, len(dataset)-args.frame_interval))
    dataset_pairs = ZipDatasets(dataset_prev, dataset_frame)
    # Shuffle and take first n pairs
    torch.manual_seed(0)
    dataset_pairs = Subset(dataset_pairs, torch.randperm(len(dataset_pairs)))
    if args.num_frames != -1:
        dataset_pairs = Subset(dataset_pairs, torch.arange(0, args.num_frames))

    dataloader_train = DeviceDataLoader(dataset_pairs, device=device, batch_size=args.batch_size, collate_fn=find_collate(dataset_pairs), shuffle=False, drop_last=False, num_workers=0)

    ALL_METRICS = [f"warp.{v}" for v in WARP_METRICS] + [f"flow.{v}" for v in FLOW_METRICS] + ["semantic_projection_iou"]
    per_frame_errors = dict([(key,{}) for key in ALL_METRICS])
    per_frame_lmk_gt = {}
    per_frame_lmk_proj = {}
    per_frame_lmk_mp_gt = {}
    per_frame_lmk_mp_proj = {}

    for idx, (views_prev, views) in enumerate(tqdm(dataloader_train)):

        # Filter out pairs of views from different sequences
        B = views["img"].shape[0] # batch size
        valid_pairs = views_prev["seq_idx"] == views["seq_idx"]
        n_valid_pairs = torch.count_nonzero(valid_pairs)
        if n_valid_pairs != B:
            batch_select(views, valid_pairs)
            batch_select(views_prev, valid_pairs)
        elif n_valid_pairs == 0:
            continue

        img_curr = srgb_to_rgb(views["crop"]["image"].permute(0,2,3,1))
        img_prev = srgb_to_rgb(views_prev["crop"]["image"].permute(0,2,3,1))
        semantic_mask_curr = views["crop"]["semantic_mask"]
        semantic_mask_prev = views_prev["crop"]["semantic_mask"]
        B, H, W, _ = img_curr.shape

        # Mask out parts of the image we want to ignore for warping
        IGNORE_LABELS = [SemanticMask.BACKGROUND, SemanticMask.NECK, SemanticMask.CLOTH_NECKLACE, SemanticMask.HAIR, SemanticMask.HAT]
        for label in IGNORE_LABELS:
            img_curr[semantic_mask_curr[..., label]] = 0
            img_prev[semantic_mask_prev[..., label]] = 0

        vispath      = (out_dir / f"{idx}_curr.png") if idx == 0 else None
        vispath_prev = (out_dir / f"{idx}_prev.png") if idx == 0 else None
        values = wrapper(views, training=False, vispath=vispath)["values"]
        values_prev = wrapper(views_prev, training=False, vispath=vispath_prev)["values"] 

        # Gather landmarks predictions and projections
        for i,vidx in enumerate(views["idx"]):
            # Landmarks are in NDC space, in the range [-1,1]
            per_frame_lmk_gt[vidx.item()] = values["lmk"][i]
            per_frame_lmk_proj[vidx.item()] = values["predicted_landmarks"][i]
            per_frame_lmk_mp_gt[vidx.item()] = values["lmk_mp"][i, MEDIAPIPE_LMK_EMBEDDING_INDICES]
            per_frame_lmk_mp_proj[vidx.item()] = values["predicted_landmarks_mediapipe"][i]
    
        rast = wrapper.rasterize(values)
        rast_ss = wrapper.rasterize(values, scale=OCCLUSION_SUPERSAMPLE)
        rast_prev = wrapper.rasterize(values_prev)
        rast_prev_ss = wrapper.rasterize(values_prev, scale=OCCLUSION_SUPERSAMPLE)
        rast_mask = rast[..., -1].long() > 0
        rast_mask_prev = rast_prev[..., -1].long() > 0

        labels = list(set(label for group in SEMANTIC_LABEL_MAPPING for label in group["labels_geo"] if label != "background"))
        flame_masks_rast = wrapper.rasterize_semantic_masks(values, labels, rast=rast)
        flame_masks_rast_prev = wrapper.rasterize_semantic_masks(values_prev, ["sem_neck"], rast=rast_prev)
        # Mask out areas we don't want to warp from the rasterization area
        rast *= sum([flame_masks_rast[key] for key in ["sem_neck"]]).permute(0, 2, 3, 1) < 0.5
        rast_prev *= sum([flame_masks_rast_prev[key] for key in ["sem_neck"]]).permute(0, 2, 3, 1) < 0.5
        
        ####################### WARPING ERROR #######################
        img_warped, occlusion_mask = perform_warp(img_prev, values_prev["trans_verts"], faces, rast, rast_ss, rast_prev_ss)

        comparison_mask_warp = ~occlusion_mask.unsqueeze(-1)

        if SHOW_OCCLUSIONS:
            # Color the occlusion mask for visualization
            img_warped_visu = img_warped.clone()
            img_warped_visu[occlusion_mask] = 1

        # Compute a MSE just for visualization purposes
        img_warp_error = ((img_warped - img_curr) * comparison_mask_warp).pow(2).mean(-1, keepdim=True)

        for metric_name in WARP_METRICS:
            if metric_name == "MSE":
                met = img_mse(img_warped * comparison_mask_warp, img_curr * comparison_mask_warp)#, mask=comparison_mask, use_mask=True)
            elif metric_name == "PSNR":
                met = img_psnr(img_warped * comparison_mask_warp, img_curr * comparison_mask_warp)#, mask=comparison_mask)
            elif metric_name == "SSIM":
                met = img_ssim(img_warped * comparison_mask_warp, img_curr * comparison_mask_warp)
            else:
                raise NotImplementedError()

            for i,vidx in enumerate(views["idx"]):
                per_frame_errors["warp."+metric_name][vidx.item()] = met[i]
        #############################################################
        
        ###################### OPTICAL FLOW #########################
        if OPTICAL_FLOW:
            predicted_flows = compute_optical_flows(img_prev, img_curr, model_optical_flow, preprocess_img_optical_flow) # (B, xy, H, W)  
            predicted_flows *= 2 / H # convert to ndc space
            vertex_screen_flows = (values["trans_verts"] - values_prev["trans_verts"])[...,:2] # (B, V, xy)

            geometric_flows = interpolate(vertex_screen_flows, rast_prev_ss, faces)
            _, H_ss, W_ss, _ = rast_prev_ss.shape
            geometric_flows = geometric_flows.unflatten(1, (H_ss,W_ss)) # (B, H, W, xy)
            geometric_flows = geometric_flows.permute(0,3,1,2) # (B, xy, H, W)
            geometric_flows = torch.nn.functional.avg_pool2d(geometric_flows, H_ss // H)
        
            comparison_mask_flow = (
                # ignore pixels that were masked out in prev frame
                (img_prev.sum(-1) != 0) * \
                # ignore pixels that weren't rasterized in prev frame
                rast_mask_prev
            ).unsqueeze(-1).permute(0,3,1,2)
            predicted_flows *= comparison_mask_flow
            geometric_flows *= comparison_mask_flow

            # Scale manually to avoid per-image normalization
            flow_scale = 2
            img_flow_predicted = _normalized_flow_to_image((predicted_flows * flow_scale).clamp(min=-1, max=1)).permute(0,2,3,1) / 255
            img_flow_geometric = _normalized_flow_to_image((geometric_flows * flow_scale).clamp(min=-1, max=1)).permute(0,2,3,1) / 255

            img_flow_error_L1 = (predicted_flows - geometric_flows).permute(0,2,3,1).sum(-1, keepdim=True).abs()
            img_flow_error_MSE = (predicted_flows - geometric_flows).permute(0,2,3,1).pow(2).mean(-1, keepdim=True)

            for metric_name in FLOW_METRICS:
                if metric_name == "L1":
                    met = (predicted_flows - geometric_flows).sum(1).abs().mean((1,2)) # mean over image dimensions
                elif metric_name == "MSE":
                    met = img_mse(predicted_flows, geometric_flows)#, mask=flow_mask)
                
                for i,vidx in enumerate(views["idx"]):
                    per_frame_errors["flow."+metric_name][vidx.item()] = met[i]
        #############################################################

        #################### SEMANTIC PROJECTION ####################
        # Create semantic masks:
        # - from the rasterized geometry, using vertex masks (semantic_mask_geo)
        # - by combining predicted segmentation masks (semantic_mask_gt)
        semantic_mask_geo = torch.zeros((B, H, W, len(SEMANTIC_LABEL_MAPPING)), dtype=torch.bool, device=device) # (B, H, W, m)
        semantic_mask_gt = torch.zeros_like(semantic_mask_geo) # (B, H, W, m)
        for i, group in enumerate(SEMANTIC_LABEL_MAPPING):
            for label in group["labels_geo"]:
                pixel_mask = (~rast_mask) if label == "background" else ((flame_masks_rast[label] >= 0.5).squeeze(1) * rast_mask) # (B, H, W)
                semantic_mask_geo[..., i] = torch.logical_or(semantic_mask_geo[..., i], pixel_mask)
            for label in group["labels_gt"]:
                semantic_mask_gt[..., i] = torch.logical_or(semantic_mask_gt[..., i], semantic_mask_curr[..., label])
        
        semantic_mask_geo_color = color_label_mask(semantic_mask_geo)
        semantic_mask_gt_color = color_label_mask(semantic_mask_gt)

        semantic_comparison_mask = semantic_mask_curr[..., [SemanticMask.HAIR, SemanticMask.HAT]].sum(-1, keepdim=True) == 0
        colors = [group["color"] for group in SEMANTIC_LABEL_MAPPING]
        ious = img_IoU(semantic_mask_geo_color, semantic_mask_gt_color, colors, mask=semantic_comparison_mask)
        for i,vidx in enumerate(views["idx"]):
            per_frame_errors["semantic_projection_iou"][vidx.item()] = ious[i]
        # Compute an image for visualization
        img_sem_error = ((semantic_mask_gt_color - semantic_mask_geo_color) * semantic_comparison_mask).abs().mean(-1, keepdim=True)
        #############################################################

        ####################### VISUALIZATION #######################
        if idx < 5:
            img_columns = [
                rgb_to_srgb(img_prev),
                rgb_to_srgb(img_curr),
                rgb_to_srgb(img_warped_visu),
                rgb_to_srgb(img_warped * comparison_mask_warp),
                img_warp_error * torch.ones_like(img_curr) * VISUALIZATION_ERROR_SCALE,
            ]
            save_img(out_dir / f"warp_{idx:04d}.png", make_img_row(make_img_col(col) for col in img_columns))
            
            if OPTICAL_FLOW:
                img_columns = [
                    rgb_to_srgb(img_prev),
                    rgb_to_srgb(img_curr),
                    img_flow_predicted,
                    img_flow_geometric,
                    img_flow_error_L1.repeat(1,1,1,3) * 10,
                    img_flow_error_MSE.repeat(1,1,1,3) * 10,
                ]
                save_img(out_dir / f"flow_{idx:04d}.png", make_img_row(make_img_col(col) for col in img_columns))

            img_columns = [
                rgb_to_srgb(img_curr),
                semantic_mask_gt_color * semantic_comparison_mask,
                semantic_mask_geo_color * semantic_comparison_mask,
                img_sem_error * torch.ones_like(img_curr),
            ]
            save_img(out_dir / f"sem_{idx:04d}.png", make_img_row(make_img_col(col) for col in img_columns))

            del img_columns
        #############################################################
        
        del rast, rast_prev, views_prev, views, img_warped, img_warp_error

    #############################################################
    log_text = ""
    log_text += f"Evaluation over {len(list(per_frame_errors.values())[0])} frames\n"
    log_text += f"Evaluation set: {args.eval_dataset}\n"
    log_text += f"Tracker adaptation iter: {args.tracker_resume}\n"
    log_text += f"Encoder: {args.encoder}\n"
    log_text += f"Decoder: {args.decoder}\n"
    ####################### Warping error #######################
    log_text += f"Warping error:\n"
    log_text += f"\tFrame interval: {args.frame_interval}\n"
    for metric_name, errors in per_frame_errors.items():
        if metric_name.startswith("warp."):
            metric_name = metric_name.split("warp.")[1]
            err = torch.tensor(list(errors.values()), dtype=torch.float)
            log_text += f"\t{metric_name}: {err.mean():.04f} ± {err.std():.04f} ({err.median():.04f})\n"
    ####################### Optical flow error #######################
    if OPTICAL_FLOW:
        log_text += f"Optical flow error:\n"
        log_text += f"\tFrame interval: {args.frame_interval}\n"
        for metric_name, errors in per_frame_errors.items():
            if metric_name.startswith("flow."):
                metric_name = metric_name.split("flow.")[1]
                err = torch.tensor(list(errors.values()), dtype=torch.float)
                log_text += f"\t{metric_name}: {err.mean():.04f} ± {err.std():.04f} ({err.median():.04f})\n"
    ###################### Landmarks error ######################
    all_lmk_gt = torch.stack([per_frame_lmk_gt[key] for key in per_frame_lmk_gt])
    all_lmk_proj = torch.stack([per_frame_lmk_proj[key] for key in per_frame_lmk_gt])
    lmk_reprojection_error_L1 = torch.linalg.vector_norm(all_lmk_gt - all_lmk_proj, dim=-1, ord=1).mean(1) # mean over landmarks
    all_lmk_mp_gt = torch.stack([per_frame_lmk_mp_gt[key] for key in per_frame_lmk_mp_gt])
    all_lmk_mp_proj = torch.stack([per_frame_lmk_mp_proj[key] for key in per_frame_lmk_mp_gt])
    lmk_mp_reprojection_error_L1 = torch.linalg.vector_norm(all_lmk_mp_gt - all_lmk_mp_proj, dim=-1, ord=1).mean(1) # mean over landmarks
    log_text += "Landmarks reprojection L1:\n"
    log_text += f"\t{lmk_reprojection_error_L1.mean():.04f} ± {lmk_reprojection_error_L1.std():.04f} ({lmk_reprojection_error_L1.median():.04f})\n"
    log_text += "MediaPipe landmarks reprojection L1:\n"
    log_text += f"\t{lmk_mp_reprojection_error_L1.mean():.04f} ± {lmk_mp_reprojection_error_L1.std():.04f} ({lmk_mp_reprojection_error_L1.median():.04f})\n"
    ##################### Semantic proj IoU #####################
    log_text += "Semantic projection IoU:\n"
    per_frame_per_class_iou = torch.stack(list(per_frame_errors["semantic_projection_iou"].values())) 
    # print("per_frame_pre_class_iou")
    # for i, v in enumerate(per_frame_per_class_iou):
    #     print(f"{i:04d}: " + ", ".join([f"{x:.02f}" for x in v]))
    per_frame_iou = per_frame_per_class_iou.mean(1)
    log_text += f"\t{per_frame_iou.mean():.04f} ± {per_frame_iou.std():.04f} ({per_frame_iou.median():.04f})\n"
    per_class_iou = per_frame_per_class_iou.mean(0)
    log_text += "\tper class: " + ", ".join([f"{v:.04f}" for v in per_class_iou]) +"\n"
    #############################################################

    logging.info(log_text)
    with open(out_dir / "results.txt", "w+") as file:
        file.write(log_text)

# Mapping between segmentation masks and FLAME masks
SEMANTIC_LABEL_MAPPING = [
    {"color": [.2,.2,.2], "labels_gt": [SemanticMask.BACKGROUND, SemanticMask.CLOTH_NECKLACE, SemanticMask.NECK], "labels_geo": ["background", "sem_neck"]},
    # {"color": [],       "labels_gt": [SemanticMask.SKIN], "labels_geo": ["sem_skin"]},
    # {"color": [],       "labels_gt": [SemanticMask.EYEBROWS], "labels_geo": ["sem_eyebrow_left", "sem_eyebrow_right"]},
    {"color": [1.,1.,1.], "labels_gt": [SemanticMask.SKIN, SemanticMask.EYEBROWS], "labels_geo": ["sem_skin", "sem_eyebrow_left", "sem_eyebrow_right"]},
    {"color": [0.,0.,1.], "labels_gt": [SemanticMask.LOWER_LIP], "labels_geo": ["sem_lip_lower"]},
    {"color": [0.,1.,0.], "labels_gt": [SemanticMask.UPPER_LIP], "labels_geo": ["sem_lip_upper"]},
    {"color": [1.,1.,0.], "labels_gt": [SemanticMask.NOSE], "labels_geo": ["sem_nose"]},
    {"color": [0.,1.,.5], "labels_gt": [SemanticMask.EARS], "labels_geo": ["sem_ear_left", "sem_ear_right"]},
    {"color": [.5,0.,.5], "labels_gt": [SemanticMask.EYES], "labels_geo": ["sem_eye_left", "sem_eye_right"]},
    {"color": [1.,0.,0.], "labels_gt": [SemanticMask.MOUTH_INTERIOR], "labels_geo": ["sem_mouth_interior"]},
]
for v in SEMANTIC_LABEL_MAPPING:
    v["color"] = torch.tensor(v["color"], dtype=torch.float)

def color_label_mask(mask):
    device = mask.device
    img = torch.zeros((*mask.shape[:-1], 3), dtype=torch.float, device=device)
    for i, v in enumerate(SEMANTIC_LABEL_MAPPING):
        img[mask[..., i]] = v["color"].to(device)
    return img

def interpolate(vert_attribute, rast, faces):
    # vert_attribute: (B, V, f) or (B, V)
    # rast: (B, H, W, 5)
    # faces: (F, 3)
    ndim = vert_attribute.ndim
    if ndim == 2:
        vert_attribute = vert_attribute.unsqueeze(-1)
    bi, bj, bk, _, triangle_ids = rast.flatten(1, 2).unbind(-1) # (B, H*W)
    triangle_ids = triangle_ids.long() - 1
    triangle_vert_attr = torch.stack([vert_attribute[i, v] for i,v in enumerate(faces[triangle_ids])]) # (B, H*W, 3, f) feature per triangle point per pixel
    a, b, c = triangle_vert_attr.unbind(2) # (B, H*W, f)
    pixel_attr = bi.unsqueeze(-1)*a + bj.unsqueeze(-1)*b + bk.unsqueeze(-1)*c # (B, H*W, f)
    if ndim == 2:
        pixel_attr = pixel_attr.squeeze(-1)
    return pixel_attr

def batch_select(batch: Dict[str, torch.Tensor], mask: torch.BoolTensor):
    B = mask.shape[0]
    for key,v in batch.items():
        if torch.is_tensor(v) and v.shape[0] == B:
            batch[key] = batch[key][mask]
        elif isinstance(v, list): # cameras
            batch[key] = [x for i,x in enumerate(batch[key]) if mask[i]]
        elif isinstance(v, dict):
            batch_select(batch[key], mask)

def perform_warp(img_prev, verts_prev_ndc, faces, rast, rast_ss, rast_prev_ss):
    B,H,W,_ = img_prev.shape
    _, H_ss, W_ss, _ = rast_ss.shape

    verts_prev_screen_xy = verts_prev_ndc[...,:-1] * H / 2
    verts_prev_z = verts_prev_ndc[...,-1]

    # Compute screen-space positions at the previous frame of all visible points in the current frame
    # Interpolate screen-space vertex positions of the previous frame using barycentric coordinates of the current frame
    prev_pos_2d = interpolate(verts_prev_screen_xy, rast, faces)
    x,y = (prev_pos_2d + H/2).round().long().clamp(min=0, max=H-1).unbind(-1) # (B, H*W)
    # Warp
    img_warped = torch.stack([img_prev[i,y[i],x[i]] for i in range(B)]) # (B, H*W, rgb)

    ###################### Occlusion Mask ######################
    # Compute an occlusion mask for the previous frame (if a point was occluded in the previous frame, we don't want to backtrack to it).
    # Compute the same positions as above, this time in the higher-scale render
    verts_prev_screen_xy_ss = verts_prev_screen_xy * OCCLUSION_SUPERSAMPLE
    prev_pos_2d_ss = interpolate(verts_prev_screen_xy_ss, rast_ss, faces)
    x_ss,y_ss = (prev_pos_2d_ss + H_ss/2).round().long().clamp(min=0, max=H_ss-1).unbind(-1) # (B, H*W)
    # Compare the depth buffer and the projected depths: any vertex whose projection is further away than the depth buffer value  at its screen position is occluded
    prev_depth = interpolate(verts_prev_z, rast_ss, faces)
    threshold = 0.05
    depth_buffer_prev_ss = rast_prev_ss[..., 3] # (B, H_ss, W_ss)
    occlusion_mask = torch.stack([prev_depth[i] > depth_buffer_prev_ss[i,y_ss[i],x_ss[i]] + threshold for i in range(B)])
    occlusion_mask = occlusion_mask.unflatten(1, (H_ss,W_ss)).unsqueeze(1)
    occlusion_mask = torch.nn.functional.avg_pool2d(occlusion_mask.float(), OCCLUSION_SUPERSAMPLE) > 0.1
    occlusion_mask = occlusion_mask.squeeze(1).flatten(1, 2)
    #############################################################

    rast_mask = rast[..., -1].flatten(1, 2).long() > 0 # (B, W*H)
    img_warped *= rast_mask.unsqueeze(-1) # the warped values make no sense outside of the rasterized region
    occlusion_mask *= rast_mask # ensure we don't mistakenly count non-rasterized areas as occlusions

    occlusion_mask = occlusion_mask.unflatten(1, (H,W))
    img_warped = img_warped.unflatten(1, (H,W))
    return img_warped, occlusion_mask


if __name__ == "__main__":
    parser = spark_config_parser()
    # Add evaluation-specific args
    parser.add_argument("--frame_interval", type=int, default=1, help="Number of frames between the previous and current frame for warping.")
    parser.add_argument("--num_frames", type=int, default=-1, help="Number of image to test (-1 for entire test set).")
    parser.add_argument("--eval_dataset", type=str, default="test", help="test|train")

    args = parse_args(parser)

    # ALWAYS EVALUATE IN RENDER_MODE="crop"! otherwise we can have inconsistent scaling for some of the errors
    wrapper, dataset_train, dataset_val, dataset_test = spark_setup(args, render_mode="crop")

    if args.eval_dataset == "test":
        evaluation_dataset = dataset_test
    elif args.eval_dataset == "train":
        evaluation_dataset = dataset_train
    else:
        raise NotImplementedError()

    main(wrapper, args, evaluation_dataset)
