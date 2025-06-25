# Code: https://github.com/fraunhoferhhi/neural-deferred-shading/tree/main
# Modified/Adapted by: Shrisha Bharadwaj, Kelian Baert

import numpy as np
import torch
from typing import Tuple, Dict
from math import prod
from gpytoolbox import remesh_botsch

from pytorch3d.structures import Meshes
from pytorch3d.ops import SubdivideMeshes
from flame import FLAME

def find_edges(indices, remove_duplicates=True):
    # Extract the three edges (in terms of vertex indices) for each face 
    # edges_0 = [f0_e0, ..., fN_e0]
    # edges_1 = [f0_e1, ..., fN_e1]
    # edges_2 = [f0_e2, ..., fN_e2]
    edges_0 = torch.index_select(indices, 1, torch.tensor([0,1], device=indices.device))
    edges_1 = torch.index_select(indices, 1, torch.tensor([1,2], device=indices.device))
    edges_2 = torch.index_select(indices, 1, torch.tensor([2,0], device=indices.device))

    # Merge the into one tensor so that the three edges of one face appear sequentially
    # edges = [f0_e0, f0_e1, f0_e2, ..., fN_e0, fN_e1, fN_e2]
    edges = torch.cat([edges_0, edges_1, edges_2], dim=1).view(indices.shape[0] * 3, -1)

    if remove_duplicates:
        edges, _ = torch.sort(edges, dim=1)
        edges = torch.unique(edges, dim=0)

    return edges

def find_connected_faces(indices):
    edges = find_edges(indices, remove_duplicates=False)

    # Make sure that two edges that share the same vertices have the vertex ids appear in the same order
    edges, _ = torch.sort(edges, dim=1)

    # Now find edges that share the same vertices and make sure there are only manifold edges
    _, inverse_indices, counts = torch.unique(edges, dim=0, sorted=False, return_inverse=True, return_counts=True)
    assert counts.max() == 2

    # We now create a tensor that contains corresponding faces.
    # If the faces with ids fi and fj share the same edge, the tensor contains them as
    # [..., [fi, fj], ...]
    face_ids = torch.arange(indices.shape[0])               
    face_ids = torch.repeat_interleave(face_ids, 3, dim=0) # Tensor with the face id for each edge

    face_correspondences = torch.zeros((counts.shape[0], 2), dtype=torch.int64).to(device=indices.device)
    face_correspondences_indices = torch.zeros(counts.shape[0], dtype=torch.int64)

    # ei = edge index
    for ei, ei_unique in enumerate(list(inverse_indices.cpu().numpy())):
        face_correspondences[ei_unique, face_correspondences_indices[ei_unique]] = face_ids[ei] 
        face_correspondences_indices[ei_unique] += 1

    return face_correspondences[counts == 2].to(device=indices.device)

def compute_laplacian_uniform(mesh):
    """
    Computes the laplacian in packed form.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph
    Returns:
        Sparse FloatTensor of shape (V, V) where V = sum(V_n)
    """

    # This code is adapted from from PyTorch3D 
    # (https://github.com/facebookresearch/pytorch3d/blob/88f5d790886b26efb9f370fb9e1ea2fa17079d19/pytorch3d/structures/meshes.py#L1128)

    verts_packed = mesh.vertices # (sum(V_n), 3)
    edges_packed = mesh.edges    # (sum(E_n), 2)
    V = mesh.vertices.shape[0]

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=mesh.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=mesh.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def compute_barycentric_coordinates(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor,
                                    project=False, clamp_to_triangle=True):
    """
    Compute the barycentric coordinates of a point relative to a triangle.
    If `project` is false, the point is assumed to be on the same plane as (v0,v1,v2).

    Args:
        p: Coordinates of a point.
        v0, v1, v2: Coordinates of the triangle vertices.
        clamp_to_triangle: if true, coordinates will be clamped to be within the triangle.
        project: if true, the points will first be projected to the triangle's plane.
        
    Returns
        bary: (w0, w1, w2) barycentric coordinates (in range [0, 1] if clamp_to_triangle=True).
    """

    if project:
        normals = torch.nn.functional.normalize(torch.cross(v1 - v0, v2 - v0), dim=1)
        # Project (p - v0) onto the face normals
        dot_prods = ((p - v0) * normals).sum(-1, keepdim=True) # (n, 1)
        p = p - dot_prods * normals # (n, 3)

    e0 = v1 - v0
    e1 = v2 - v0
    e2 = p - v0

    dot = lambda x,y: (x*y).sum(-1)

    dot00 = dot(e0, e0)
    dot01 = dot(e0, e1)
    dot02 = dot(e0, e2)
    dot11 = dot(e1, e1)
    dot12 = dot(e1, e2)

    denom = dot00 * dot11 - dot01 * dot01
    v = (dot11 * dot02 - dot01 * dot12) / denom
    w = (dot00 * dot12 - dot01 * dot02) / denom
    u = 1.0 - v - w

    # Clamp within the triangle
    if clamp_to_triangle:
        uneg = u < 0
        if uneg.any():
            t = dot(p-v1, v2-v1) / dot(v2-v1, v2-v1)
            t = t.clamp(min=0, max=1)
            u = torch.where(uneg, 0, u)
            v = torch.where(uneg, 1-t, v)
            w = torch.where(uneg, t, w)
        vneg = v < 0
        if vneg.any():
            t = dot(p-v2, v0-v2) / dot11
            t = t.clamp(min=0, max=1)
            u = torch.where(vneg, t, u)
            v = torch.where(vneg, 0, v)
            w = torch.where(vneg, 1-t, w)
        wneg = w < 0
        if wneg.any():
            t = dot02 / dot00
            t = t.clamp(min=0, max=1)
            u = torch.where(wneg, 1-t, u)
            v = torch.where(wneg, t, v)
            w = torch.where(wneg, 0, w)

        assert torch.count_nonzero((u < 0) + (u > 1) + (v < 0) + (v > 1) + (w < 0) + (w > 1)) == 0

    return torch.stack((u, v, w), dim=-1)

def point_to_triangle_distance(p: torch.Tensor, v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor):
    """
    Compute the squared distance from a point p to a triangle defined by (v0, v1, v2).
    All tensors are shape (..., 3)
    Returns:
        sqr_distance: shape (...)
    """

    uvw = compute_barycentric_coordinates(p, v0, v1, v2, project=False, clamp_to_triangle=False)
    # Clamp and normalize
    uvw = uvw.clamp(min=0, max=1)
    uvw = uvw / uvw.sum(-1, keepdim=True)
    u, v, w = uvw.unsqueeze(-1).unbind(-2)

    proj = u*v0 + v*v1 + w*v2
    dist2 = ((proj - p) ** 2).sum(-1)
    return dist2

def find_closest_triangles(vertices, faces, points, k=8):
    """
    For each point, find the closest triangle from a mesh using:
    1. Top-k nearest triangle midpoints (approx)
    2. Exact point-to-triangle distances over top-k

    Args:
        vertices: (V, 3)
        faces: (F, 3)
        points: (N, 3)
        k: number of candidate triangles to consider per point

    Returns:
        closest_face_indices: (N,) index of the closest triangle per point
    """
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert points.ndim == 2 and points.shape[1] == 3
    assert k > 0
    N = points.shape[0]
    F = faces.shape[0]

    tri_vertices = vertices[faces] # (F, 3, 3)
    face_midpoints = tri_vertices.mean(dim=1) # (F, 3)

    # Step 1: Approximate nearest k triangles using midpoint distance
    dists = torch.cdist(points, face_midpoints)  # (N, F)

    # torch.topk crashes with very large inputs (see https://github.com/pytorch/pytorch/issues/82569)
    if N*F >= 60000000 or k == 1:
        return dists.argmin(dim=1)

    topk_dists, topk_indices = torch.topk(dists, k, dim=1, largest=False)  # (N, k)

    # Step 2: Gather triangle vertices for top-k faces
    v0_topk, v1_topk, v2_topk = vertices[faces[topk_indices]].unbind(2) # (N, k, 3)

    # Repeat points to (N, k, 3)
    p_expanded = points.unsqueeze(1).expand(-1, k, -1)

    # Step 3: Compute exact distances from each point to its k candidate triangles
    dist2 = point_to_triangle_distance(p_expanded, v0_topk, v1_topk, v2_topk)  # (N, k)

    # Step 4: Find minimum index in top-k, map back to global face index
    min_k_indices = dist2.argmin(dim=1)  # (N,)
    closest_face_indices = topk_indices[torch.arange(points.shape[0]), min_k_indices]

    return closest_face_indices

def barycentric_projection(vertices, faces, points):
    # vertices: (V, 3)
    # faces: (F, 3)
    # points: (n, 3)

    closest_face_indices = find_closest_triangles(vertices, faces, points) # (n) 

    # Calculate barycentric coordinates of the projections
    v0, v1, v2 = vertices[faces[closest_face_indices]].double().unbind(1)
    assert v0.shape == v1.shape == v2.shape == points.shape
    barycentric_coordinates = compute_barycentric_coordinates(points, v0, v1, v2, project=True, clamp_to_triangle=True)
    assert barycentric_coordinates.shape == (points.shape[0], 3)

    if barycentric_coordinates.isnan().any():
        # One reason this can happen is using single-point precision inputs
        raise ValueError("Found NaN in barycentric coordinates!")

    barycentric_coordinates = barycentric_coordinates.float()

    return closest_face_indices, barycentric_coordinates

def reproject_vertex_attributes(verts: torch.Tensor, faces: torch.Tensor, pts: torch.Tensor, vert_attrs: Dict[str, torch.Tensor]):
    # Project all points onto the mesh
    face_idx, bary_coords = barycentric_projection(verts, faces, pts) # face_idx: (V), bary_coords: (V, 3)
    verts_idx = faces[face_idx] # (V, 3)

    new_attrs = {}
    for key, attr in vert_attrs.items():
        new_attr = attr.view(verts.shape[0], -1)
        new_attr = torch.einsum("vfi,vf->vi", [new_attr[verts_idx], bary_coords]) # (V, n)        
        new_attrs[key] = new_attr.view(pts.shape[0], *attr.shape[1:])
    return new_attrs

def vert2faces(faces: torch.Tensor):
    """ Compute a dictionary mapping vertex indices to face indices.

    Args:
        faces: Mesh topology of shape (n_faces, 3).

    Returns:
        A Dict that maps each vertex index to a list of face indices.
    """

    num_verts = faces.max() + 1
    faces_per_vert = dict([(vert, []) for vert in range(num_verts)])
    for fidx,f in enumerate(faces):
        faces_per_vert[f[0].item()].append(fidx)
        faces_per_vert[f[1].item()].append(fidx)
        faces_per_vert[f[2].item()].append(fidx)
    return faces_per_vert

def disconnect_vertices(faces: torch.Tensor, disconnect_verts: torch.Tensor, get_face_mask=False):
    """ Disconnect vertices from a mesh.

    Args:
        faces: Mesh topology of shape (n_faces, 3).
        disconnect_verts: Tensor of vertex indices to disconnect.

    Returns:
        Updated faces tensor of shape (n_faces_new, 3), n_faces_new <= n_faces
    """
    # First record a list of face indices for each vertex
    faces_per_vert = vert2faces(faces)
    # Populate a list of faces to remove
    keep_faces = torch.ones((faces.size(0)), dtype=torch.bool)
    for vert in disconnect_verts:
        vert = vert.item()
        if vert in faces_per_vert:
            keep_faces[faces_per_vert[vert]] = False   
    # Return updated list of faces
    return (faces[keep_faces], keep_faces) if get_face_mask else faces[keep_faces]

def remove_vertices(verts: torch.Tensor, faces: torch.Tensor, remove_verts: torch.Tensor, vert_attrs: Dict[str, torch.Tensor]):
    """ Remove vertices from a mesh.

    Args:
        verts: Vertex positions of shape (n_verts, 3).
        faces: Mesh topology of shape (n_faces, 3).
        remove_verts: Tensor of vertex indices to remove.
        vert_attrs: Per-vertex attribute tensors from which to also remove the vertices.

    Returns:
        Updated faces tensor of shape (n_faces_new, 3), n_faces_new <= n_faces
        Updated verts tensor of shape (n_verts_new, 3), n_verts_new = n_verts - len(remove_verts)
    """
    n_verts = verts.shape[0]
    # First, disconnect the vertices from the faces
    faces = disconnect_vertices(faces, remove_verts)

    # Then, compute a mask of vertices to keep
    keep_verts = torch.ones((n_verts), dtype=torch.bool, device=verts.device)
    keep_verts[remove_verts] = 0
    # Reindex the vertices in the faces
    reindex_mapping = torch.zeros((n_verts), dtype=torch.long, device=verts.device)
    offset = 0
    for i in range(n_verts):
        if not keep_verts[i]:
            offset -= 1
        reindex_mapping[i] = i + offset
    faces = faces.cpu().apply_(lambda vi: reindex_mapping[vi]).to(verts.device) # tensor.apply is slow!

    return verts[keep_verts], faces, {key: attr[keep_verts] for key,attr in vert_attrs.items()}

def remove_faces(verts: torch.Tensor, faces: torch.Tensor, remove_faces_idx: torch.Tensor):
    """ Remove faces from a mesh.

    Args:
        verts: Vertex positions of shape (n_verts, 3).
        faces: Mesh topology of shape (n_faces, 3).
        remove_faces_idx: Tensor of face indices to remove.

    Returns:
        Updated faces tensor of shape (n_faces_new, 3), n_faces_new = n_faces - len(remove_faces_idx)
        Updated verts tensor of shape (n_verts_new, 3), n_verts_new <= n_verts
    """
    n_verts, n_faces = verts.shape[0], faces.shape[0]

    face_mask = torch.ones((n_faces), dtype=torch.bool, device=faces.device)
    face_mask[remove_faces_idx] = 0

    # Remove faces
    faces = faces[face_mask]

    # Then, compute a mask of vertices to keep (those that were not orphaned)
    keep_verts = torch.zeros((n_verts), dtype=torch.bool, device=verts.device)
    keep_verts[faces.flatten()] = 1
        
    # Reindex the vertices in the faces
    reindex_mapping = torch.zeros((n_verts), dtype=torch.long, device=verts.device)
    offset = 0
    for i in range(n_verts):
        if not keep_verts[i]:
            offset -= 1
        reindex_mapping[i] = i + offset
    faces = faces.cpu().apply_(lambda vi: reindex_mapping[vi]).to(verts.device) # tensor.apply is slow!

    return verts[keep_verts], faces

def extract_submesh(verts: torch.Tensor, faces: torch.Tensor, extract_verts: torch.Tensor, vert_attrs: Dict[str, torch.Tensor]):
    """ Extract the part of a mesh that comprises the given vertices.

    Args:
        verts: Vertex positions of shape (n_verts, 3).
        faces: Mesh topology of shape (n_faces, 3).
        extract_verts: Tensor with vertex indices to extract.
        vert_attrs: Per-vertex attribute tensors.

    Returns:
        Faces tensor of the extracted mesh, of shape (n_faces_extract, 3), n_faces_extract <= n_faces
        Verts tensor of the extracted mesh, of shape (len(extract_verts), 3)
    """
    # First record a list of face indices for each vertex
    faces_per_vert = vert2faces(faces)    
    # Populate a list of faces to extract
    extract_faces = torch.zeros((faces.size(0)), dtype=torch.bool)
    for vert in extract_verts:
        vert = vert.item()
        if vert in faces_per_vert:
            extract_faces[faces_per_vert[vert]] = True
    # Reindex the vertices from the indexing of the global mesh to the local extracted mesh
    reindex_mapping = dict([vi.item(),i] for i,vi in enumerate(extract_verts)) # invert the extract_verts list
    extracted_faces = faces[extract_faces].clone()
    extracted_faces = extracted_faces.cpu().apply_(lambda vi: reindex_mapping[vi]).to(verts.device) # tensor.apply is slow!
    return verts[extract_verts], extracted_faces, {key: attr[extract_verts] for key,attr in vert_attrs.items()}


def merge_meshes(*meshes: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]):
    assert len(meshes) > 0
    attr_keys = meshes[0][2].keys()

    # We assume these meshes have nothing in common
    verts = torch.cat([mesh[0] for mesh in meshes], dim=0)
    faces = torch.cat([mesh[1] for mesh in meshes], dim=0)
    attrs = {key: torch.cat([mesh[2][key] for mesh in meshes], dim=0) for key in attr_keys}

    updated_verts_idx = [torch.arange(verts.size(0), dtype=torch.long, device=verts.device) for (verts,_,_) in meshes]

    face_index = 0
    vert_offset = 0
    for mi, (mesh_verts, mesh_faces, *mesh_attrs) in enumerate(meshes):
        num_mesh_faces = mesh_faces.size(0)
        faces[face_index:face_index+num_mesh_faces] += vert_offset
        updated_verts_idx[mi] += vert_offset
    
        face_index += num_mesh_faces
        vert_offset += mesh_verts.size(0)

    return (verts, faces, attrs), updated_verts_idx

@torch.no_grad()
def remesh_FLAME(flame: FLAME, vertices: torch.Tensor, faces: torch.Tensor, h: float, separate_verts_masks: Dict):
    # separate_verts_masks: specifies lists of vertices that should not be remeshed. These will be removed from the mesh before remeshing, and re-added afterwards. 
    old_V = len(vertices)
    device = vertices.device

    # Set vertex attributes to be re-projected on the remeshed surface
    vert_attrs = {}
    vert_attrs["shapedirs"] = flame.shapedirs_expression_updated # (old_v, 3, n_exp)
    vert_attrs["lbs_weights"] = flame.lbs_weights_updated # (old_v, J)
    vert_attrs["posedirs"] = flame.posedirs_updated.permute(1, 0).view(old_V, 3, -1) # (P, 3*V) => (3*V, P) => (V, 3, P)
    for key, mask in flame.mask.f.items():
        vert_attrs[f"mask.{key}"] = mask.detach().to(device)
    vert_attrs["uvs"] = flame.uvs

    # Get landmark positions (for updating the indices)
    lmk_pos = flame.get_landmark_positions(vertices.unsqueeze(0)).squeeze(0) # (L, 3)
    lmk_pos_static = flame.get_landmark_positions(vertices.unsqueeze(0), "static").squeeze(0) # (L, 3)
    lmk_pos_dynamic = flame.get_landmark_positions(vertices.unsqueeze(0), "dynamic").squeeze(0) # (L, 3)
    lmk_pos_mediapipe = flame.get_landmark_positions(vertices.unsqueeze(0), "static_mediapipe").squeeze(0) # (L, 3)

    vertices = vertices.detach()

    # Cut the vertices provided in separate_verts_masks into separate meshes
    extracted_meshes = dict()
    for key, vert_mask in separate_verts_masks.items():
        extracted_meshes[key] = extract_submesh(vertices, faces, vert_mask, vert_attrs)
    all_extracted_verts = torch.cat(tuple(separate_verts_masks.values()))

    main_verts, main_faces, main_attrs = remove_vertices(vertices, faces, all_extracted_verts, vert_attrs)

    # Remesh the main mesh
    main_verts_updated, main_faces_updated = \
        remesh_botsch(main_verts.cpu().numpy().astype(np.float64), main_faces.cpu().numpy().astype(np.int32), h=h, project=True)
    main_verts_updated = torch.tensor(np.ascontiguousarray(main_verts_updated), dtype=torch.float32, device=device)
    main_faces_updated = torch.tensor(np.ascontiguousarray(main_faces_updated), dtype=torch.int64, device=device)

    # Upsample the per-vertex attributes of the main mesh
    if old_V == flame.canonical_verts.shape[1]:
        # Using the original canonical verts seems to preserve the attributes better.
        main_template_verts, main_template_faces, _ = remove_vertices(flame.canonical_verts.squeeze(0), flame.faces, all_extracted_verts, {})
        main_attrs_updated = reproject_vertex_attributes(main_template_verts, main_template_faces, main_verts_updated, main_attrs)
    else:
        # The above is not possible on subsequent remeshes as we lost some of the original attributes.
        main_attrs_updated = reproject_vertex_attributes(main_verts, main_faces, main_verts_updated, main_attrs)

    main_mesh_updated = (main_verts_updated, main_faces_updated, main_attrs_updated)

    # Merge the main mesh with the extracted meshes
    (merged_verts, merged_faces, merged_attrs), new_verts_idx = merge_meshes(main_mesh_updated, *extracted_meshes.values())
    # Gather the new indices of the vertices in separate_verts_masks
    new_verts_idx_dict = dict([key, vi] for (key, vi) in zip(separate_verts_masks.keys(), new_verts_idx[1:]))
    
    # Update FLAME basis
    flame.shapedirs_expression_updated = merged_attrs["shapedirs"]
    flame.lbs_weights_updated = merged_attrs["lbs_weights"]
    flame.posedirs_updated = merged_attrs["posedirs"].view(len(merged_verts)*3, -1).permute(1, 0) # (V, 3, P) => (P, 3*V)
    assert flame.shapedirs_expression_updated.shape[1:] == flame.shapedirs_expression.shape[1:]
    assert flame.lbs_weights_updated.shape[1:] == flame.lbs_weights.shape[1:]
    assert flame.posedirs_updated.shape[:-1] == flame.posedirs.shape[:-1]
    for key in merged_attrs:
        if key.startswith("mask."):
            f_mask = merged_attrs[key]
            flame.mask.f.register_buffer(key[5:], f_mask)
            v_mask = (f_mask > 0.5).nonzero().squeeze(1)
            flame.mask.v.register_buffer(key[5:], v_mask)
    flame.uvs = merged_attrs["uvs"]

    # Reproject the landmarks positions onto the remeshed surface
    lmk_face_idx, lmk_bary_coords = barycentric_projection(merged_verts, merged_faces, lmk_pos)
    flame.full_lmk_verts_idx = merged_faces[lmk_face_idx] # (L, 3)
    flame.full_lmk_bary_coords = lmk_bary_coords.unsqueeze(0) # (1, L, 3)
    lmk_static_face_idx, lmk_static_bary_coords = barycentric_projection(merged_verts, merged_faces, lmk_pos_static)
    flame.static_lmk_verts_idx = merged_faces[lmk_static_face_idx] # (L, 3)
    flame.static_lmk_bary_coords = lmk_static_bary_coords.unsqueeze(0) # (1, L, 3)
    lmk_dyn_face_idx, lmk_dyn_bary_coords = barycentric_projection(merged_verts, merged_faces, lmk_pos_dynamic)
    flame.dynamic_lmk_verts_idx = merged_faces[lmk_dyn_face_idx].view_as(flame.dynamic_lmk_verts_idx) # (A, L, 3)
    flame.dynamic_lmk_bary_coords = lmk_dyn_bary_coords.unsqueeze(0).view_as(flame.dynamic_lmk_bary_coords) # (1, A, L, 3)
    lmk_static_face_idx_mediapipe, lmk_static_bary_coords_mediapipe = barycentric_projection(merged_verts, merged_faces, lmk_pos_mediapipe)
    flame.static_lmk_verts_idx_mediapipe = merged_faces[lmk_static_face_idx_mediapipe] # (L, 3)
    flame.static_lmk_bary_coords_mediapipe = lmk_static_bary_coords_mediapipe.unsqueeze(0) # (1, L, 3)

    return new_verts_idx_dict, merged_verts, merged_faces


def attrs_to_single_tensor(attrs: Dict):
    if len(attrs) == 0:
        return None
    any_attr = next(iter(attrs.values()))
    nv = any_attr.size(0) # number of vertices
    assert all(v.size(0) == nv for v in attrs.values())
    # Flatten each attribute
    attrs = {k: v.contiguous().view(nv, -1) for k, v in attrs.items()}
    # Create and fill new tensor
    attrs_tensor = torch.zeros((nv, sum(v.size(1) for v in attrs.values())), dtype=torch.float, device=any_attr.device)
    offset = 0
    for v in attrs.values():
        attrs_tensor[:,offset:offset+v.size(1)] = v
        offset += v.size(1)
    return attrs_tensor

def single_tensor_to_attrs(attrs_tensor: torch.Tensor, original_attrs: Dict):
    attrs = {}
    offset = 0
    for k,v in original_attrs.items():
        flattened_size = prod(v.shape[1:])
        attrs[k] = attrs_tensor[:, offset:offset+flattened_size].view(-1, *v.shape[1:]).to(v.dtype)
        offset += flattened_size
    return attrs

def subdivide_mesh(verts: torch.Tensor, faces: torch.Tensor, vert_attrs: Dict[str, torch.Tensor] = None, levels=1):
    attrs_provided = vert_attrs is not None
    if vert_attrs is None:
        vert_attrs = dict()

    def aux(verts: torch.Tensor, faces: torch.Tensor, vert_attrs: Dict[str, torch.Tensor]):
        # Create a p3d.Meshes object
        mesh = Meshes(verts=[verts], faces=[faces])
        # Serialize the vertex attributes to (V, n) tensors
        feats = attrs_to_single_tensor(vert_attrs)
        # Subdivide the mesh using Pytorch3D
        subdivider = SubdivideMeshes()
        subdivided_mesh, subdivided_feats = subdivider(mesh, feats)
        subdivided_verts = subdivided_mesh.verts_list()[0]
        subdivided_faces = subdivided_mesh.faces_list()[0]
        # Retrieve the subdivided vertex attributes            
        subdivided_attrs = single_tensor_to_attrs(subdivided_feats, vert_attrs)
        return subdivided_verts, subdivided_faces, subdivided_attrs
   
    # Perform subdivisions
    for _ in range(levels):
        verts, faces, vert_attrs = aux(verts, faces, vert_attrs)
    return (verts, faces, vert_attrs) if attrs_provided else (verts, faces)
