import trimesh
import rerun as rr
import numpy as np
import open3d as o3d

from jans_scripts.visualize import setup_rerun, visualize_mesh
from jans_scripts.graspnet_storage import get_graspnet_data_path
from jans_scripts.graspnet_storage import load_cam0_wrt_table


def main():
    setup_rerun("compare_rayst3r_activengf_mesh")
    camera = "realsense"
    scene_id = "scene_0100"
    view = "0000"
    graspnet_path = get_graspnet_data_path()
    rayst3r_mesh_path = f"{graspnet_path}/output/GraspNet/rayst3r_zero_shot/{camera}_{scene_id}_{view}/mesh/mesh.ply"
    activengf_mesh_path = f"{graspnet_path}/output/GraspNet/baseline/{scene_id}_nbv/mesh/00000_mesh_culled.ply"

    rayst3r_mesh = o3d.io.read_triangle_mesh(rayst3r_mesh_path)
    visualize_mesh(rayst3r_mesh, "rayst3r_mesh")


    activengf_mesh = o3d.io.read_triangle_mesh(activengf_mesh_path)
    visualize_mesh(activengf_mesh, "activengf_mesh")

if __name__ == "__main__":
    main()