import trimesh
import torch

from jans_scripts.visualize import visualize_pc_with_normals, setup_rerun

mesh_num_sample_points = 10000

def main():
    setup_rerun("surface_normals_viz")
    mesh_path = "/home/jan/thesis/ActiveNGF/output/GraspNet/scene_0100_nbv/mesh/00000_mesh_culled.ply"
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        scene_meshes = []
        for scene_mesh in mesh.geometry.values():
            scene_meshes.append(scene_mesh)
        mesh = trimesh.util.concatenate(scene_meshes)
    points, face_indices = trimesh.sample.sample_surface(mesh, mesh_num_sample_points)
    face_normals = mesh.face_normals[face_indices]
    visualize_pc_with_normals(torch.from_numpy(points), torch.from_numpy(face_normals), suffix="base", normal_length=0.001)


if __name__ == "__main__":
    main()