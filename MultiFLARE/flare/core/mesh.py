# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj, Kelian Baert

import torch
from torch import Tensor
import xatlas
from pytorch3d.io import load_obj, save_obj
from pathlib import Path

from utils.geometry import find_edges, find_connected_faces, compute_laplacian_uniform

######################################################################################
# Utils
######################################################################################
def dot(x: Tensor, y: Tensor) -> Tensor:
    return (x*y).sum(dim=-1, keepdim=True)

def reflect(x: Tensor, n: Tensor) -> Tensor:
    return 2*dot(x, n)*n - x

def length(x: Tensor, eps: float=1e-20) -> Tensor:
    return torch.sqrt(dot(x,x).clamp(min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: Tensor, eps: float = 1e-20) -> Tensor:
    return torch.nn.functional.normalize(x, eps=eps, dim=-1)

######################################################################################
# Mesh Class
######################################################################################
class Mesh:
    """ Triangle mesh defined by an indexed vertex buffer.

    Args:
        vertices (tensor): Vertex buffer (Vx3)
        indices (tensor): Index buffer (Fx3)
        device (torch.device): Device where the mesh buffers are stored
    """

    def __init__(self, vertices, indices, device='cpu', uv_coords=None, uv_idx=None):
        self.device = device

        self.vertices = vertices.to(device, dtype=torch.float32) if torch.is_tensor(vertices) else torch.tensor(vertices, dtype=torch.float32, device=device)
        self.indices = indices.to(device, dtype=torch.int64) if torch.is_tensor(indices) else torch.tensor(indices, dtype=torch.int64, device=device) if indices is not None else None

        if self.indices is not None:
            self.compute_normals()

        self._edges = None
        self._connected_faces = None
        self._laplacian = None
        self._uv_coords = uv_coords.to(device) if uv_coords is not None else None
        self._uv_idx = uv_idx.to(device) if uv_idx is not None else None

    def to(self, device):
        mesh = Mesh(self.vertices.to(device), self.indices.to(device), device=device)
        mesh._edges = self._edges.to(device) if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.to(device) if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.to(device) if self._laplacian is not None else None
        mesh._uv_coords = self._uv_coords.to(device) if self._uv_coords is not None else None
        mesh._uv_idx = self._uv_idx.to(device) if self._uv_idx is not None else None
        return mesh

    def detach(self):
        mesh = Mesh(self.vertices.detach(), self.indices.detach(), device=self.device)
        mesh.face_normals = self.face_normals.detach()
        mesh.vertex_normals = self.vertex_normals.detach()
        mesh._edges = self._edges.detach() if self._edges is not None else None
        mesh._connected_faces = self._connected_faces.detach() if self._connected_faces is not None else None
        mesh._laplacian = self._laplacian.detach() if self._laplacian is not None else None
        mesh._uv_coords = self._uv_coords.detach() if self._uv_coords is not None else None
        mesh._uv_idx = self._uv_idx.detach() if self._uv_idx is not None else None
        return mesh

    def with_vertices(self, vertices):
        """ Create a mesh with the same connectivity but with different vertex positions. 
        After each optimization step, the vertices are updated.

        Args:
            vertices (tensor): New vertex positions (Vx3)
        """
        assert len(vertices) == len(self.vertices)
        mesh_new = Mesh(vertices, self.indices, self.device)
        mesh_new._edges = self._edges
        mesh_new._connected_faces = self._connected_faces
        mesh_new._laplacian = self._laplacian
        mesh_new._uv_coords = self._uv_coords
        mesh_new._uv_idx = self._uv_idx
        return mesh_new 


    def get_vertices_face_normals(self, vertices):
        """ Calculates vertex and face normals and returns them.
        Args:
            vertices (tensor): New vertex positions (Vx3)
        """
        a = vertices[self.indices][:, 0, :]
        b = vertices[self.indices][:, 1, :]
        c = vertices[self.indices][:, 2, :]
        face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], face_normals)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1) 
    
        return vertex_normals.contiguous(), face_normals.contiguous()

    def fetch_all_normals(self, deformed_vertices, mesh):
        """ All normals are returned: Vertex, face, tangent space normals along with indices of faces. 
        Args:
            deformed vertices (tensor): New vertex positions (Vx3)
            mesh (Mesh class with these new updated vertices)
        """
        d_normals = {"vertex_normals":[], "face_normals":[], "tangent_normals":[]}
        for d_vert in deformed_vertices:
            vertex_normals, face_normals = mesh.get_vertices_face_normals(d_vert)
            tangents = mesh.compute_tangents(d_vert, vertex_normals)
            d_normals["vertex_normals"].append(vertex_normals.unsqueeze(0))
            d_normals["face_normals"].append(face_normals.unsqueeze(0))
            d_normals["tangent_normals"].append(tangents.unsqueeze(0))

        d_normals["vertex_normals"] = torch.cat(d_normals["vertex_normals"], axis=0)
        d_normals["face_normals"] = torch.cat(d_normals["face_normals"], axis=0)
        d_normals["tangent_normals"] = torch.cat(d_normals["tangent_normals"], axis=0)
        
        return d_normals
    
    ######################################################################################
    # Basic Mesh Operations
    ######################################################################################

    @property
    def edges(self):
        if self._edges is None:
            self._edges = find_edges(self.indices)
        return self._edges

    @property
    def connected_faces(self):
        if self._connected_faces is None:
            self._connected_faces = find_connected_faces(self.indices)
        return self._connected_faces

    @property
    def laplacian(self):
        if self._laplacian is None:
            self._laplacian = compute_laplacian_uniform(self)
        return self._laplacian

    def compute_normals(self):
        # Compute the face normals
        a = self.vertices[self.indices][:, 0, :]
        b = self.vertices[self.indices][:, 1, :]
        c = self.vertices[self.indices][:, 2, :]
        self.face_normals = torch.nn.functional.normalize(torch.cross(b - a, c - a), p=2, dim=-1) 

        # Compute the vertex normals
        vertex_normals = torch.zeros_like(self.vertices)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 0], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 1], self.face_normals)
        vertex_normals = vertex_normals.index_add(0, self.indices[:, 2], self.face_normals)
        self.vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=-1)    

    @torch.no_grad()
    def xatlas_uvmap(self):
        import numpy as np
        # Create uvs with xatlas
        v_pos = self.vertices.detach().cpu().numpy()
        t_pos_idx = self.indices.detach().cpu().numpy()
        vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

        # Convert to tensors
        indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
        vmapping_int64 = vmapping.astype(np.uint64, casting='same_kind').view(np.int64)
        vmapping = torch.tensor(vmapping_int64, dtype=torch.int64, device=self.device)

        uvs = torch.tensor(uvs, dtype=torch.float32, device=self.device)
        faces = torch.tensor(indices_int64, dtype=torch.int64, device=self.device)

        self._uv_coords = uvs
        self._uv_idx = faces

    def compute_connectivity(self):
        if self._uv_coords is None or self._uv_idx is None:
            self.xatlas_uvmap()
        self._edges = self.edges
        self._connected_faces = self.connected_faces
        self._laplacian = self.laplacian

    ######################################################################################
    # Compute tangent space from texture map coordinates
    # Follows http://www.mikktspace.com/ conventions
    # Taken from:https://github.com/NVlabs/nvdiffrec
    ######################################################################################
    
    def compute_tangents(self, vertices, vertex_normals):        
        vn_idx = [None] * 3
        pos = [None] * 3
        tex = [None] * 3
        for i in range(0,3):
            # NOTE: VERIFY BY GIVING INDICES TO VERTICES ONCE
            pos[i] = vertices[self.indices[:, i]]
            tex[i] = self._uv_coords[self._uv_idx[:, i]]
            vn_idx[i] = self.indices[:, i]

        tangents = torch.zeros_like(vertex_normals)
        tansum   = torch.zeros_like(vertex_normals)

        # Compute tangent space for each triangle
        uve1 = tex[1] - tex[0]
        uve2 = tex[2] - tex[0]
        pe1  = pos[1] - pos[0]
        pe2  = pos[2] - pos[0]
        
        nom   = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
        denom = (uve1[..., 0:1] * uve2[..., 1:2] - uve1[..., 1:2] * uve2[..., 0:1])
        
        # Avoid division by zero for degenerated texture coordinates
        tang = nom / torch.where(denom > 0.0, torch.clamp(denom, min=1e-6), torch.clamp(denom, max=-1e-6))

        # Update all 3 vertices
        for i in range(0,3):
            idx = vn_idx[i][:, None].repeat(1,3)
            tangents.scatter_add_(0, idx, tang)                # tangents[n_i] = tangents[n_i] + tang
            tansum.scatter_add_(0, idx, torch.ones_like(tang)) # tansum[n_i] = tansum[n_i] + 1
        tangents = tangents / tansum

        # Normalize and make sure tangent is perpendicular to normal
        tangents = safe_normalize(tangents)
        tangents = safe_normalize(tangents - dot(tangents, vertex_normals) * vertex_normals)

        if torch.is_anomaly_enabled():
            assert torch.all(torch.isfinite(tangents))

        return tangents
    
    def write(self, path: str, texture: torch.Tensor = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vertices = self.vertices.detach().cpu()
        indices = self.indices.detach().cpu() if self.indices is not None else None

        if self._uv_coords is not None and self._uv_idx is not None:
            verts_uvs = self._uv_coords.detach().cpu()
            faces_uvs = self._uv_idx.detach().cpu()
            # we need to define a texture map or PyTorch3D won't save UVs
            texture_map = torch.zeros((16, 16, 3), dtype=torch.float) if texture is None else texture
            save_obj(path, vertices, indices, verts_uvs=verts_uvs, faces_uvs=faces_uvs, texture_map=texture_map)

            if texture is None:
                # clean-up the texture and mtl files which we don't want
                path.with_suffix(".png").unlink() # delete texture file
                path.with_suffix(".mtl").unlink() # delete material file
        else:
            save_obj(path, vertices, indices)

    @classmethod
    def read(cls, path: str, device='cpu'):
        verts, faces, aux = load_obj(path, load_textures=False)   
        return cls(verts, faces.verts_idx, device, uv_coords=aux.verts_uvs, uv_idx=faces.textures_idx)