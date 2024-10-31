from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
import numpy as np

# MultiFLARE imports
from Avatar import Avatar
from utils.color import rgb_to_srgb
from utils.geometry import reproject_vertex_attributes
from utils.visualization import save_img
from flame.FLAME import FlameMask

from adapt.deca_helpers import deca_set_render_faces
from adapt.common_types import EncodedValues, DecodedValues, RenderTensors
from adapt.constants import DATASET_RESOLUTION, CROP_RESOLUTION, SMIRK_PATH, MULTIFLARE_PATH
from adapt.general_utils import working_dir

class FaceDecoder(torch.nn.Module):
    def __init__(self, device: torch.device, render_mode: str):
        super().__init__()
        self.device = device
        self.render_mode = render_mode
        self.resolution = DATASET_RESOLUTION if render_mode == "full" else CROP_RESOLUTION

    def forward(self, values: EncodedValues, views: Dict, training: bool) -> DecodedValues:
        raise NotImplementedError()

    def loss(self, decoded_values: DecodedValues, views_crop: Dict, training: bool):
        """ Compute losses and metrics. """
        raise NotImplementedError()

    def rasterize(self, decoded_values: DecodedValues, resolution=None):
        raise NotImplementedError()
    
    def get_vertex_mask(self, key: str) -> Tensor:
        raise NotImplementedError()

    def visualize(self, decoded_values: DecodedValues, save_path: str = None):
        raise NotImplementedError()

class DECADecoder(FaceDecoder):
    def __init__(self, device: torch.device, render_mode: str, deca):
        super().__init__(device, render_mode)
        self.deca = deca
        self.faces = deca.deca.render.faces.squeeze(0) #deca.deca.flame.faces_tensor
        self.n_verts = deca.deca.flame.v_template.shape[0]
        
        with working_dir(MULTIFLARE_PATH):
            self.flame_masks = FlameMask(faces=self.faces, num_verts=self.n_verts).to(device)

        deca.deca.render.image_size = self.resolution
        deca.deca.render.rasterizer.raster_settings.__dict__["image_size"] = self.resolution


    def forward(self, values: EncodedValues, views: Dict, training: bool, override_verts=None) -> DecodedValues:
        decoded_values = self.deca.decode(values, training=training, override_verts=override_verts)
        return decoded_values

    def loss(self, decoded_values: DecodedValues, views_crop: Dict, training: bool):
        """ Compute losses and metrics. """
        return self.deca.compute_loss(decoded_values, views_crop, training=training)

    def rasterize(self, decoded_values: DecodedValues, resolution=None):
        if resolution is None:
            resolution = self.resolution
        # Rasterize geometry
        _, bary_coords, triangle_ids, zbuf = self.deca.deca.render.render_shape(decoded_values["verts"], decoded_values["trans_verts"], return_buffers=True, override_image_size=resolution)
        # since we already rendered the geometry, which has the side-effect of adding 10 to the depth, we need to subtrack 10 back
        # zbuf[zbuf != -1] -= 2
        # We only need the triangle with the lowest z-order
        bary_coords, triangle_ids = bary_coords[:,:,:,0], triangle_ids[...,(0,)]
        # triangle ids are indexed in a flattened faces tensor, so we need to correct for that
        for b in range(triangle_ids.shape[0]):
            triangle_ids[b][triangle_ids[b] != -1] -= self.faces.shape[0] * b
        triangle_ids = triangle_ids + 1
        rast = torch.cat((bary_coords, zbuf, triangle_ids), dim=-1)
        return rast
    
    def get_vertex_mask(self, key):
        return getattr(self.flame_masks.f, key)

    def visualize(self, decoded_values: DecodedValues, save_path: str = None):
        """ Create visualizations. """
        v = decoded_values
        # if self.render_mode == "full":
        #     values["images"] = draw_crop_rectangles(v["images"], views["crop"]["bbox"])
        visdict, _ = self.deca._visualization_checkpoint(v["verts"], v["trans_verts"], v["ops"],
                                                         v["uv_detail_normals"] if "uv_detail_normals" in v else None, v, None, None, None)
        if save_path is not None:
            save_visualizations(visdict, save_path)
        return visdict


class MultiFLAREDecoder(DECADecoder):
    def __init__(self, device: torch.device, render_mode: str, deca, multiflare: Avatar):
        super().__init__(device, render_mode, deca)
        self.multiflare = multiflare

        canonical_verts = multiflare.canonical_mesh.vertices
        self.n_verts = canonical_verts.shape[0]
        self.faces = multiflare.canonical_mesh.indices
        self.flame_masks = multiflare.flame.mask

        multiflare.renderer.resolution = (self.resolution, self.resolution)
        with torch.no_grad():
            self.shapedirs, self.posedirs, self.lbs_weights = multiflare.deformer_net.query_weights(canonical_verts)

        self.texture = None

        # Compute the l_eyelid and r_eyelid of SMIRK for the topology of FLARE
        flame = multiflare.flame
        smirk_assets_path = SMIRK_PATH / "assets"
        l_eyelid = torch.from_numpy(np.load(smirk_assets_path / "l_eyelid.npy")).to(torch.float).to(device)
        r_eyelid = torch.from_numpy(np.load(smirk_assets_path / "r_eyelid.npy")).to(torch.float).to(device)
        if self.n_verts != flame.v_template.shape[0]:
            print(f"Reprojecting eyelid verts to MultiFLARE topology (total from {flame.v_template.shape[0]} to {self.n_verts} verts)")
            attrs = {"l_eyelid": l_eyelid, "r_eyelid": r_eyelid}
            attrs_proj = reproject_vertex_attributes(flame.v_template, flame.faces, canonical_verts, attrs)
            l_eyelid = attrs_proj["l_eyelid"]
            r_eyelid = attrs_proj["r_eyelid"]
        self.l_eyelid = l_eyelid.unsqueeze(0)
        self.r_eyelid = r_eyelid.unsqueeze(0)

    def loss(self, decoded_values: DecodedValues, views_crop: Dict, training: bool):
        """ Compute losses and metrics. """
        with deca_set_render_faces(self.deca, self.faces, self.multiflare.flame.uvs):
            return super().loss(decoded_values, views_crop, training)

    def rasterize(self, decoded_values: DecodedValues, resolution=None):
        with deca_set_render_faces(self.deca, self.faces, self.multiflare.flame.uvs):
            return super().rasterize(decoded_values, resolution)

    def visualize(self, decoded_values: DecodedValues, save_path: str = None):
        """ Create visualizations. """
        with deca_set_render_faces(self.deca, self.faces, self.multiflare.flame.uvs):
            # # Grid visualization
            # if visualize_grid:
            #     if self.render_mode == "full":
            #         views["img"] = draw_crop_rectangles(views["img"], views["crop"]["bbox"])
            #     visualize_grid(rgb_pred, cbuffers, gbuffers, views, save_path)
            return super().visualize(decoded_values, save_path)

    def forward(self, enc: EncodedValues, views: Dict, training: bool) -> DecodedValues:
        B = views["img"].shape[0]
        avatar = self.multiflare
        flame, canonical_mesh, renderer = avatar.flame, avatar.canonical_mesh, avatar.renderer

        if training:
            for cam, seq_idx in zip(views["camera"], views["seq_idx"]):
                cam.K = avatar.cams_K_train[seq_idx].clone()

        if self.render_mode == "crop":
            # Update views dict so we can render with FLARE in the context of the crops
            views["img"] = views["crop"]["image"].permute(0,2,3,1) # BCHW to BHWC
            views["landmarks"] = views["crop"]["landmark"]
            views["landmarks_mediapipe"] = views["crop"]["landmark_mediapipe"]
            # Update the renderer and the camera principal points
            self.multiflare.renderer.resolution = (self.resolution, self.resolution)
            for cam in views["camera"]:
                cam.K[0,2] = self.resolution / 2 # cx
                cam.K[1,2] = self.resolution / 2 # cy

        pose = enc["posecode"]
        expression = enc["expcode"]

        # Convert the predictions into values that make sense for MultiFLARE
        # Using the orthographic projection parameters (s, tx, ty), we approximate a position for our mesh under a perspective camera model
        s, tx, ty = enc["cam"].unbind(1)      
        pose = torch.cat((
            pose[:, :3], # global rotation
            torch.zeros_like(pose[:, :3]), # neck rotation
            pose[:, 3:], # jaw
            torch.zeros_like(pose[:, :6]), # eye pose
            torch.stack((tx, ty, tx*0), dim=-1) # translation
        ), dim=-1)
        for i, cam in enumerate(views["camera"]):
            F = (cam.K[0,0] * 2 / self.resolution).detach() # assume Fx = Fy
            pose[i, 17] = -F/s[i]
            cam.t[:] = 0

        # Compute deformed vertices using cached shapedirs, posedirs and lbs_weights
        deformed_vertices, *_ = Avatar.compute_deformed_verts(canonical_mesh, flame, pose, expression, None, shapedirs=self.shapedirs, posedirs=self.posedirs, lbs_weights=self.lbs_weights)
        # Account for SMIRK's eyelid deformation
        if enc.eyelids is not None:
            deformed_vertices = deformed_vertices + self.r_eyelid.expand(B, -1, -1) * enc.eyelids[:, 1:2, None]
            deformed_vertices = deformed_vertices + self.l_eyelid.expand(B, -1, -1) * enc.eyelids[:, 0:1, None]

        texture_batch = None if self.texture is None else self.texture.unsqueeze(0).repeat(B, 1, 1, 1) # (B, H, W, 3|4)
        # Render using FLARE, with pre-computed vertex positions
        rgb_pred, _, _, _, _, _, gbuffer_mask, *_ = avatar.run(views, avatar.args.resume, deformed_vertices=deformed_vertices, texture=texture_batch)

        # Override the render and landmark positions of DECA with the FLARE rendering before computing losses
        values: DecodedValues = enc
        values.verts = deformed_vertices # probably incorrect, but unused
        values.trans_verts = renderer.to_ndc(deformed_vertices, views["camera"])[...,:3]
        values.predicted_landmarks = renderer.to_ndc(flame.get_landmark_positions_2d(deformed_vertices, pose), views["camera"])[:,:68,:2]
        values.predicted_landmarks_mediapipe = renderer.to_ndc(flame.get_landmark_positions(deformed_vertices, which="static_mediapipe"), views["camera"])[:,:,:2]
        values.predicted_images = rgb_to_srgb(rgb_pred).permute(0,3,1,2)
        values.ops = RenderTensors()
        values.ops.alpha_images = gbuffer_mask.permute(0,3,1,2) # (B, 1, H, W)
        # values.ops.grid = gbuffers["rast"]
        # Note that we don't need to fill in RenderTensors completely, as the other values are only used by DECA's decode(),
        # which this method is replacing 
        
        if self.deca.deca.config.useSeg == "gt":
            values.masks = enc.masks.unsqueeze(1)
        else:
            # values.masks = gbuffer_mask.permute(0,3,1,2)
            # values.masks = enc.masks.unsqueeze(1) * gbuffer_mask.permute(0,3,1,2)
            # values.predicted_images = values.predicted_images * values.masks
            raise NotImplementedError() # see decode() of DECA code for other behaviors             

        # Transform the NDC z to a range that doesn't generate artifacts with the DECA renderer
        z = values["trans_verts"][..., 2]
        zd = z.detach()
        values["trans_verts"][..., 2] = 10 * (z - zd.min()) / (zd.max() - zd.min() + 1e-8)

        return values


class SMIRKDecoder(DECADecoder):
    def __init__(self, device: torch.device, render_mode: str, deca, smirk_generator):
        super().__init__(device, render_mode, deca)
        self.smirk_generator = smirk_generator

        # SMIRK imports
        from src.renderer.renderer import Renderer as SmirkRenderer
        from src.FLAME.FLAME import FLAME as SmirkFLAME
        import src.utils.masking as masking_utils

        smirk_assets_path = str(SMIRK_PATH / "assets")
        self.smirk_flame = SmirkFLAME(assets_path=smirk_assets_path).to(device)
        self.smirk_renderer = SmirkRenderer(render_full_head=True, assets_path=smirk_assets_path).to(device)
        self.smirk_renderer.image_size = self.resolution

        # Load triangle probabilities for sampling points on the image (this is necessary for rendering with SMIRK)
        with working_dir(SMIRK_PATH):
            self.face_probabilities = masking_utils.load_probabilities_per_FLAME_triangle()  

    def forward(self, values: EncodedValues, views: Dict, training: bool) -> DecodedValues:
        # Convert back from the common DECA representation to SMIRK's
        # This back and forth is not ideal, but it allows us to unify all computations on the encoded values
        smirk_outputs = {
            "expression_params": values.expcode,
            "pose_params": values.posecode[:, :-3],
            "jaw_params": values.posecode[:, -3:],
            "cam": values.cam,
            "shape_params": values.shapecode,
            "eyelid_params": values.eyelids,
        }

        # use SMIRK's FLAME since it has the controls for eyelids
        smirk_flame_output = self.smirk_flame.forward(smirk_outputs)
        override_verts = (smirk_flame_output["vertices"], smirk_flame_output["landmarks_fan"], smirk_flame_output["landmarks_fan_3d"], smirk_flame_output["landmarks_mp"])
        decoded = super().forward(values, views, training=training, override_verts=override_verts)

        renderer_output = self.smirk_renderer.forward(smirk_flame_output['vertices'], values['cam'], landmarks_fan=smirk_flame_output['landmarks_fan'], landmarks_mp=smirk_flame_output['landmarks_mp'])
        rendered_img = renderer_output['rendered_img']
       
        mask_dilation_radius = 10
        # mask_ratio_mul = 5
        # mask_ratio = 0.01
        # tmask_ratio = mask_ratio * mask_ratio_mul # upper bound on the number of points to sample       
        tmask_ratio = 0.01 #* self.config.train.mask_ratio_mul # upper bound on the number of points to sample
        # select pixel points from face vertices
        import src.utils.masking as masking_utils
        npoints, _ = masking_utils.mesh_based_mask_uniform_faces(renderer_output['transformed_vertices'], 
                                                                flame_faces=self.smirk_flame.faces_tensor,
                                                                face_probabilities=self.face_probabilities,
                                                                mask_ratio=tmask_ratio)
        img = values.images
        masks = values.masks # is this the right mask?

        # mask out face and add random points inside the face
        rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
        
        # construct a mask with the selected points using the initial pixel values
        extra_points = masking_utils.transfer_pixels(img, npoints, npoints)
        
        # completed masked img - mask out the face and add the extra points
        masked_img = masking_utils.masking(img, masks, extra_points, mask_dilation_radius, rendered_mask=rendered_mask)

        smirk_generator_input = torch.cat([rendered_img, masked_img], dim=1)
        decoded["predicted_images"] = decoded["predicted_detailed_image"] = self.smirk_generator(smirk_generator_input)

        return decoded


def save_visualizations(visdict, path: str):
    make_row = lambda x: torch.cat(tuple(x), dim=2)
    make_col = lambda x: torch.cat(tuple(x), dim=1)

    columns = [
        "inputs",
        "output_images_coarse",
        "geometry_coarse",
        "output_images_detail",
        "geometry_detail",
        "landmarks",
        "landmarks_mp",
    ]

    image = make_col(visdict[columns[0]])
    for key in columns[1:]:
        if key in visdict:
            x = visdict[key].to(image.device)
            image = make_row((image, make_col(x)))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_img(path, image.permute(1,2,0)) # CHW to HWC 
