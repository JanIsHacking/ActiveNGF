import os
from typing import Tuple

import numpy as np
import trimesh
import rerun as rr

from jans_scripts.visualize import setup_rerun

mesh_num_sample_points = 10000


def barycentric_weights(tris: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    tris: (N,3,3) triangle vertex positions
    pts: (N,3) points inside each respective triangle
    returns: (N,3) barycentric weights for each point (u,v,w)
    """
    v0 = tris[:, 0, :]
    v1 = tris[:, 1, :]
    v2 = tris[:, 2, :]
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pv0 = pts - v0
    d00 = np.einsum("ij,ij->i", v0v1, v0v1)
    d01 = np.einsum("ij,ij->i", v0v1, v0v2)
    d11 = np.einsum("ij,ij->i", v0v2, v0v2)
    d20 = np.einsum("ij,ij->i", pv0, v0v1)
    d21 = np.einsum("ij,ij->i", pv0, v0v2)
    denom = d00 * d11 - d01 * d01
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    v = (d11 * d20 - d01 * d21) / denom_safe
    w = (d00 * d21 - d01 * d20) / denom_safe
    u = 1.0 - v - w
    weights = np.stack([u, v, w], axis=1)
    weights = np.clip(weights, 0.0, 1.0)
    return weights


def extract_vertex_rgb(mesh: trimesh.Trimesh) -> np.ndarray:
    vcols = None
    if hasattr(mesh, "visual") and getattr(mesh.visual, "vertex_colors", None) is not None:
        vcols = mesh.visual.vertex_colors
    elif hasattr(mesh, "visual") and getattr(mesh.visual, "face_colors", None) is not None:
        # fallback if colors are stored per-face: expand to per-vertex by indexing face->vertex
        face_cols = mesh.visual.face_colors
        if face_cols is not None and mesh.faces is not None:
            vcols = np.zeros((len(mesh.vertices), 4), dtype=face_cols.dtype)
            fidx = mesh.faces
            vcols[fidx[:, 0]] = face_cols
            vcols[fidx[:, 1]] = face_cols
            vcols[fidx[:, 2]] = face_cols

    if vcols is None:
        return None

    if vcols.shape[1] >= 3:
        rgb = vcols[:, :3].astype(np.float64)
    else:
        return None

    if rgb.max() <= 1.01:
        rgb = rgb * 255.0

    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
    return rgb


def get_mesh_colors(mesh: trimesh.Trimesh, points: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    vertex_rgb = extract_vertex_rgb(mesh)
    if vertex_rgb is None:
        raise ValueError("Vertex colors are not found")

    faces_idx = mesh.faces[face_indices]
    tri_verts = mesh.vertices[faces_idx]
    tri_colors = vertex_rgb[faces_idx]

    weights = barycentric_weights(tri_verts, points)
    colors = (weights[:, :, None] * tri_colors).sum(axis=1)

    colors = np.rint(colors).astype(np.uint8)

    return colors

def main() -> None:
    setup_rerun("mapping_viz")

    scene_path = "output/GraspNet/test/gt/scene_0100_nbv"
    meshes_dir = os.path.join(scene_path, "mesh")
    for mesh_file in os.listdir(meshes_dir):
        mesh_path = os.path.join(meshes_dir, mesh_file)
        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            scene_meshes = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(scene_meshes)

        points, face_indices = trimesh.sample.sample_surface(mesh, mesh_num_sample_points)

        colors = get_mesh_colors(mesh, points, face_indices)

        rr.log(f"world/mesh/{mesh_file}", rr.Points3D(points, colors=colors.tolist()))

if __name__ == "__main__":
    main()
