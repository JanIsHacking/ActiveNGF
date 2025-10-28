import trimesh
import rerun as rr

from jans_scripts.mapping_viz import extract_vertex_rgb
from jans_scripts.visualize import setup_rerun

def main():
    setup_rerun("check_culled_meshes")
    mesh_file = "output/GraspNet/baseline/scene_0100_nbv/mesh/00000_mesh_culled.ply"
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        scene_meshes = []
        for scene_mesh in mesh.geometry.values():
            scene_meshes.append(scene_mesh)
        mesh = trimesh.util.concatenate(scene_meshes)
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, 10000)

    vertex_rgb = extract_vertex_rgb(mesh)
    faces_idx = mesh.faces[face_indices]
    tri_verts = mesh.vertices[faces_idx]
    tri_colors = vertex_rgb[faces_idx]

    print(tri_colors.shape)
    print(tri_colors)

    rr.log("world/points", rr.Points3D(point_cloud, colors=tri_colors.sum(axis=2).tolist()))

if __name__ == "__main__":
    main()