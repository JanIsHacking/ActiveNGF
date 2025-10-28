import trimesh
import os
import rerun as rr

from jans_scripts.visualize import setup_rerun
from jans_scripts.mapping_viz import get_mesh_colors


def compare_meshes_culled(mesh_file1, mesh_file2, name):
    mesh1 = trimesh.load(mesh_file1)
    if isinstance(mesh1, trimesh.Scene):
        scene_meshes = []
        for scene_mesh in mesh1.geometry.values():
            scene_meshes.append(scene_mesh)
        mesh1 = trimesh.util.concatenate(scene_meshes)
    mesh2 = trimesh.load(mesh_file2)
    if isinstance(mesh2, trimesh.Scene):
        scene_meshes = []
        for scene_mesh in mesh2.geometry.values():
            scene_meshes.append(scene_mesh)
        mesh2 = trimesh.util.concatenate(scene_meshes)

    points1, face_indices1 = trimesh.sample.sample_surface(mesh1, 10000)
    points2, face_indices2 = trimesh.sample.sample_surface(mesh2, 10000)

    colors1 = get_mesh_colors(mesh1, points1, face_indices1)
    colors2 = get_mesh_colors(mesh2, points2, face_indices2)
    
    rr.log(f"world/{name}/points1", rr.Points3D(points1, colors=colors1.tolist()))
    rr.log(f"world/{name}/points2", rr.Points3D(points2, colors=colors2.tolist()))
    


if __name__ == "__main__":
    setup_rerun("compare_meshes_culled")

    mesh_dir = "output/GraspNet/test/gt/scene_0116_nbv/mesh"

    mesh_file2 = "output/GraspNet/test/gt/scene_0116_nbv/mesh/00001_mesh.ply"

    culled_mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith("_mesh_culled.ply")]
    plain_mesh_files = [f for f in os.listdir(mesh_dir) if f.endswith("_mesh.ply")]

    for culled_mesh_file, plain_mesh_file in zip(culled_mesh_files, plain_mesh_files):
        compare_meshes_culled(os.path.join(mesh_dir, culled_mesh_file), os.path.join(mesh_dir, plain_mesh_file), culled_mesh_file)
