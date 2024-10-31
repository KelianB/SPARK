import math
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import imageio 
import cv2

from utils.color import rgb_to_srgb

def arrange_grid(imgs: Iterable[Tensor], cols: int) -> Tensor:
    """
    Format an iterable of tensors into a grid image with the specified number of columns.
    Pads with additional cells to complete the grid as needed.
    """
    rows = math.ceil(len(imgs) / cols)
    if len(imgs) < cols * rows:
        imgs += [torch.zeros_like(imgs[0]) for _ in range(cols*rows-len(imgs))]
    assert len(imgs) == cols * rows
    rows = [torch.cat(tuple(imgs[y*cols:(y+1)*cols]), dim=1) for y in range(rows)]
    return torch.cat(rows, dim=0)

def save_img(out_path: str, img: Tensor, to_srgb=False):
    # Quick method for saving an image to disk. img shape; (H,W,C)
    if img.shape[-1] == 1:
        img = img.expand(*img.shape[:-1], 3)
    img = convert_uint(img, to_srgb=to_srgb)
    imageio.imsave(out_path, img)

def convert_uint(x: Tensor, to_srgb=False) -> np.ndarray:
    if to_srgb:
        x = rgb_to_srgb(x)
    x = (x.clamp(0, 1) * 255).round().to(torch.uint8)
    x = x.detach().cpu().numpy()
    return x

def shade_directional_light(normals: Tensor, light_dirs=None) -> Tensor:
    """
    Args:
        - normals: (V, 3)
        - light_dirs: (nl, 3)
    returns:
        - shading: (V, 3)
    """
    if light_dirs is None:
        light_dirs = torch.tensor([
            [-1, 1, 1],
            [ 1, 1, 1],
            [-1,-1, 1],
            [ 1,-1, 1],
            [ 0, 0, 1],
        ], dtype=torch.float, device=normals.device)

    light_dirs = light_dirs.float().to(normals.device) # (nl, 3)
    light_color = torch.ones_like(light_dirs) # white (nl, 3)
    light_intensity = 1.2

    light_dirs = F.normalize(light_dirs.unsqueeze(1).repeat(1, normals.shape[0], 1), dim=-1) # (nl, V, 3)
    normals_dot_lights = (normals.unsqueeze(0) * light_dirs).sum(dim=-1, keepdim=True).clamp(0, 1) # (nl, V, 1)
    shading = normals_dot_lights * light_intensity * light_color.unsqueeze(1)
    return shading.mean(0) # mean over lights

@torch.no_grad()
def visualize_grid(img_render, cbuffers, gbuffers, views, filename: str, background_color="black", custom_mask=None):
    device = img_render.device
    to_srgb = lambda x: rgb_to_srgb(x)
    make_row = lambda x: torch.cat(tuple(x), dim=1)
    make_col = lambda x: torch.cat(tuple(x), dim=0)
    mask = gbuffers["mask"] if custom_mask is None else custom_mask

    if background_color == "black":
        blend_fn = lambda x: x * mask
    elif background_color == "white":
        blend_fn = lambda x: x * mask + torch.ones_like(x) * (1-mask)
    else:
        blend_fn = lambda x: x

    final_img = make_col(to_srgb(views["img"]))
    add_col = lambda x: make_row((final_img, make_col(x)))

    # Landmarks
    if "landmarks" in gbuffers:
        final_img = add_col(to_srgb(gbuffers["landmarks"]))
    if "landmarks_mediapipe" in gbuffers:
        final_img = add_col(to_srgb(gbuffers["landmarks_mediapipe"]))

    # Render
    img_render = blend_fn(img_render)
    final_img = add_col(to_srgb(img_render))

    # cbuffers
    for key in cbuffers:
        cbuffers[key] = blend_fn(cbuffers[key])
    if "albedo" in cbuffers:
        final_img = add_col(to_srgb(cbuffers["albedo"]))
    if "diffuse" in cbuffers:
        final_img = add_col(to_srgb(cbuffers["diffuse"]))
    if False:
        final_img = add_col(cbuffers["roughness"])
        final_img = add_col(cbuffers["spec_int"])

    # Normal image (transform the normals to camera space)
    R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
    img_normals = (0.5 * (gbuffers["normal"] @ views["camera"][0].R.T @ R.T + 1))
    img_normals = blend_fn(img_normals)
    final_img = add_col(img_normals)

    # Geometry image
    img_geom = shade_directional_light(gbuffers["normal"].view(-1, 3)).view(gbuffers["normal"].shape)
    img_geom = blend_fn(img_geom)
    final_img = add_col(img_geom)

    save_img(filename, final_img)

@torch.no_grad()
def visualize_grid_clean(img_render, cbuffers, gbuffers, views, filename: str, flame=None, faces=None):
    """ Similar to visualize_grid, but the background is white and some parts are masked out. """
    device = img_render.device

    # Retrieve rasterized semantic masks that we want to hide
    # hidden_masks = ["sem_mouth_interior", "sem_neck"]
    hidden_masks = ["sem_mouth_interior"]
    vertex_mask = sum(getattr(flame.mask.f, key) for key in hidden_masks).to(device)

    u, v, _, triangle_ids = gbuffers["rast"].flatten(1, 2).unbind(-1) # (B, H*W)
    # where triangle_ids is zero, no triangle was rendered (otherwise we need to offset the ID by one)
    triangle_mask = torch.stack([vertex_mask[v] for v in faces[triangle_ids.long() - 1]]) # (B, H*W, 3)
    pixel_mask = u * triangle_mask[...,0] + v * triangle_mask[...,1] + (1-u-v) * triangle_mask[...,2] # (B, H*W)
    pixel_mask = pixel_mask.unflatten(1, img_render.shape[1:3]).unsqueeze(-1)  # (B, H, W, 1)
    custom_mask = gbuffers["mask"] * (pixel_mask < 0.5)

    visualize_grid(img_render, cbuffers, gbuffers, views, filename, background_color="white", custom_mask=custom_mask)

colors = {"r": (255, 0, 0), "g": (0, 255, 0), "b": (0, 0, 255), "y": (0, 255, 255)}

def plot_points(image, p2d, color='r', labels=False):
    """ Plot given 2D points onto an image. """
    c = colors[color]
    image = image.copy()
    plot_lines = p2d.shape[1] in [68, 70] # detect if we're plotting FAN landmarks

    if labels:
        text_kwargs = {"fontFace": cv2.FONT_HERSHEY_SIMPLEX, "fontScale": 0.4, "thickness": 1, "lineType": cv2.LINE_AA, "color": c}
    if plot_lines:
        end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

    for i,p in enumerate(p2d):
        x, y = int(p[0]), int(p[1])
        image = cv2.circle(image, (x, y), 1, c, 2)
        if labels:
            cv2.putText(image, str(i), (x-5, y-5), **text_kwargs)
        if plot_lines:
            if i in end_list or i >= 68:
                continue
            x2,y2 = p2d[i+1].astype(np.int32)
            image = cv2.line(image, (x, y), (x2, y2), (255, 255, 255), 1)
    return image

# taken from DECA
def tensor_vis_landmarks(images, landmarks, gt_landmarks=None, color = 'g', gt_labels=False, draw_box=None):
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()
    if gt_landmarks is not None:
        gt_landmarks_np = gt_landmarks.detach().cpu().numpy()
    for i, (image, predicted_landmark) in enumerate(zip(images, predicted_landmarks)):
        image = image[..., [2,1,0]] * 255 # RGB to BGR
        predicted_landmark[...,0] = predicted_landmark[...,0]*image.shape[1]/2 + image.shape[1]/2
        predicted_landmark[...,1] = predicted_landmark[...,1]*image.shape[0]/2 + image.shape[0]/2

        image_landmarks = plot_points(image, predicted_landmark, color)
        if gt_landmarks is not None:
            image_landmarks = plot_points(image_landmarks, gt_landmarks_np[i]*image.shape[0]/2 + image.shape[0]/2, "g", labels=gt_labels)
        if draw_box is not None:
            x, y, s = (int(v) for v in draw_box[i])
            image_landmarks = cv2.rectangle(image_landmarks, (x, y), (x+s, y+s), colors["g"], 1)
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)[..., [2,1,0]] / 255 # BGR to RGB
    vis_landmarks = torch.from_numpy(vis_landmarks)
    return vis_landmarks
