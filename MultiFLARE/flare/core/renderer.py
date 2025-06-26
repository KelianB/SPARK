# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj, Kelian Baert

from typing import Dict

import nvdiffrast.torch as dr
import torch

class Renderer:
    """ Rasterization-based triangle mesh renderer that produces G-buffers for a set of views.

    Args:
        device (torch.device): Device used for rendering (must be a GPU)
        near (float): Near plane distance
        far (float): Far plane distance
    """

    gl_transform = torch.tensor([[1., 0,  0,  0],
                                 [0,  1., 0,  0],
                                 [0,  0, -1., 0],
                                 [0,  0,  0,  1.]], dtype=torch.float)

    def __init__(self, device):
        self.glctx = dr.RasterizeGLContext()
        self.device = device
        self.resolution = (512, 512)
        # This is relatively safe, though beware of extreme z positions
        self.near = 0.1
        self.far = 100

    @staticmethod
    def transform_pos(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        return Renderer.transform_pos_batch(t_mtx.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)

    @staticmethod
    def transform_pos_batch(mtx, pos):
        t_mtx = torch.from_numpy(mtx) if not torch.torch.is_tensor(mtx) else mtx
        t_mtx = t_mtx.to(pos.device)
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones_like(pos[..., 0:1])], axis=2)
        return torch.bmm(posw, t_mtx)

    @staticmethod
    def projection(K, n, f, width, height, device):
        """
        Returns a gl projection matrix
        The memory order of image data in OpenGL, and consequently in nvdiffrast, is bottom-up.
        Note that cy has been inverted 1 - cy!
        Gradients from K are propagated properly.
        [2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0]
        [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0]
        [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)]
        [         0,             0,                           -1,                  0]
        """
        a = torch.tensor([[2/width,         0,     -2/width],
                          [      0,  2/height,    -2/height],
                          [      0,         0,            0]], dtype=torch.float, device=device)
        c = torch.tensor([[      0,         0,            1,                0],
                          [      0,         0,            1,                0],
                          [      0,         0, -(f+n)/(f-n),   -(2*f*n)/(f-n)],
                          [      0,         0,           -1,                0]], dtype=torch.float, device=device)
        aK_4x4 = torch.zeros_like(c)
        aK_4x4[:3,:3] = a*K
        return aK_4x4 + c

    @staticmethod
    def to_gl_camera(camera, resolution, n, f):
        projection_matrix, Rt = Renderer.get_projection_and_transform(camera, resolution, n, f)
        return projection_matrix @ Rt, Rt
    
    @staticmethod
    def to_gl_camera_batch(camera, resolution, n=1000, f=5000):
        p_l = []
        rt_gl = []
        for cam in camera:
            P, Rt_gl = Renderer.to_gl_camera(cam, resolution, n=n, f=f)
            # we transpose here instead of transposing while multiplying
            p_l.append(P.t())
            rt_gl.append(Rt_gl)
        
        return torch.stack(p_l), torch.stack(rt_gl)

    @staticmethod
    def get_projection_and_transform(camera, resolution, n, f):
        device = camera.device
        projection_matrix = Renderer.projection(camera.K, n=n, f=f,
                                                width=resolution[1], height=resolution[0],
                                                device=device)
        Rt = torch.eye(4, device=device)
        Rt[:3, :3] = camera.R
        Rt[:3, 3] = camera.t
        Rt = Renderer.gl_transform.to(device) @ Rt
        return projection_matrix, Rt

    def rasterize(self, vertices, indices, cams) -> Dict[str, torch.Tensor]:
        resolution = self.resolution
        vertices_clip_space = self.to_clip_space(vertices, cams)
        rast, rast_out_db = dr.rasterize(self.glctx, vertices_clip_space, indices.int(), resolution=resolution)
        return rast, rast_out_db, vertices_clip_space

    def render_batch(self, cams, deformed_vertices, deformed_normals, channels, with_antialiasing, canonical_v, canonical_idx, extra_rast_attrs=None) -> dict[str, torch.Tensor]:
        """ Render G-buffers from a set of cameras.

        Args:
            cams (List[Camera])
            ... 
        """
        B = deformed_vertices.shape[0]
        F = canonical_idx.shape[0]

        idx = canonical_idx.int()
        rast, rast_out_db, deformed_vertices_clip_space = self.rasterize(deformed_vertices, idx, cams)

        view_dir = torch.cat([v.center.unsqueeze(0) for v in cams], dim=0)
        view_dir = view_dir[:, None, None, :]
        gbuffer = {}

        # deformed points in G-buffer
        if "position" in channels or "depth" in channels:
            position, _ = dr.interpolate(deformed_vertices, rast, idx)
            gbuffer["position"] = dr.antialias(position, rast, deformed_vertices_clip_space, idx) if with_antialiasing else position

        # canonical points in G-buffer
        if "canonical_position" in channels:
            canonical_verts_batch = canonical_v.unsqueeze(0).repeat(B, 1, 1)
            canonical_position, _ = dr.interpolate(canonical_verts_batch, rast, idx, rast_db=rast_out_db)
            gbuffer["canonical_position"] = dr.antialias(canonical_position, rast, deformed_vertices_clip_space, idx) if with_antialiasing else canonical_position

        # normals in G-buffer
        if "normal" in channels:
            # cache this as it will rarely change
            if not hasattr(self, "_face_idx") or self._face_idx.shape[0] != F:
                self._face_idx = torch.arange(F, dtype=torch.int32, device=self.device).unsqueeze(1).repeat(1, 3) # (F, 3)

            vertex_normals, _ = dr.interpolate(deformed_normals["vertex_normals"], rast, idx)
            face_normals, _ = dr.interpolate(deformed_normals["face_normals"], rast, self._face_idx)
            tangent_normals, _ = dr.interpolate(deformed_normals["tangent_normals"], rast, idx)
            gbuffer["vertex_normals"] = vertex_normals
            gbuffer["face_normals"] = face_normals
            gbuffer["tangent_normals"] = tangent_normals

        if extra_rast_attrs is not None:
            for key, attr in extra_rast_attrs.items():
                A = attr.shape[-1]
                if attr.shape == (B, F, 3, A) or attr.shape == (F, 3, A):         
                    # Flatten to (B?, F*3, A)
                    attr_flat = attr.flatten(-3, -2)
                    # Triangle indices from 0 to F*3 (cached)
                    if not hasattr(self, "_face_idx_2") or self._face_idx_2.shape[0] != F:
                        self._face_idx_2 = torch.arange(F * 3, dtype=torch.int32, device=self.device).reshape(F, 3)
                    buff, _ = dr.interpolate(attr_flat, rast, self._face_idx_2)
                else:
                    buff, _ = dr.interpolate(attr, rast, idx)
                gbuffer[key] = dr.antialias(buff, rast, deformed_vertices_clip_space, idx)

        # mask of mesh in G-buffer
        if "mask" in channels:
            gbuffer["mask"] = (rast[..., -1:] > 0.).float() 

            
        # We store the deformed vertices in clip space, the transformed camera matrix and the barycentric coordinates
        # to antialias texture and mask after computing the color 
        gbuffer["rast"] = rast
        gbuffer["deformed_verts_clip_space"] = deformed_vertices_clip_space

        return gbuffer

    def to_clip_space(self, verts, cams):
        resolution = self.resolution
        P_batch, _ = Renderer.to_gl_camera_batch(cams, resolution, n=self.near, f=self.far) # P_batch: (B,4,4)
        verts_clip = Renderer.transform_pos_batch(P_batch, verts) # (B,V,4)
        return verts_clip
    
    def to_ndc(self, verts, cams):
        verts_clip = self.to_clip_space(verts, cams)
        # Perspective division
        verts_ndc = verts_clip / verts_clip[...,3].unsqueeze(-1) # (B,V,4)
        return verts_ndc

    def to_screen_space(self, verts, cams):
        batch_size = verts.size(0)
        device = verts.device
        w,h = self.resolution
        verts_ndc = self.to_ndc(verts, cams)
        viewport_transform = torch.tensor([[w/2,   0, 0, (w-1)/2],
                                           [  0, h/2, 0, (h-1)/2],
                                           [  0,   0, 1,       0],
                                           [  0,   0, 0,       1]], device=device, dtype=torch.float)
        viewport_transform = viewport_transform.unsqueeze(0).expand((batch_size,-1,-1)) # (B,4,4)
        # NDC to screen space
        verts_screen = torch.bmm(verts_ndc, viewport_transform) # (B,V,4)
        return verts_screen[...,:2] # discard z and w
