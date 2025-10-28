import open3d as o3d
import torch
import numpy as np
import os

from jans_scripts.graspnet_storage import get_graspnet_data_path, get_graspnet_test_scene_ids
from jans_scripts.geometry import compute_pointmap
from jans_scripts.graspnet_storage import load_mask, load_depth, load_rgb, load_intrinsics


def mesh_from_pcd(pcd_path, scene_id, camera, view, visualize=False):
    depth = load_depth(scene_id, camera, view)
    rgb = load_rgb(scene_id, camera, view)
    intrinsics = load_intrinsics(scene_id, camera)[0]
    mask = load_mask(scene_id, camera, view)
    cam2world = torch.eye(4)

    point_map = compute_pointmap(depth, intrinsics, cam2world)[mask]
    point_map_colors = rgb[mask] / 255.0

    if visualize:
        rr.log("world/input/pointmap", rr.Points3D(point_map.cpu().numpy(), colors=point_map_colors.cpu().numpy()))

    pcd = o3d.io.read_point_cloud(pcd_path)

    if visualize:
        rr.log("input/pointcloud_raw", rr.Points3D(np.asarray(pcd.points)))

    print("Downsampling point cloud")
    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    if visualize:
        rr.log("input/pointcloud_downsampled", rr.Points3D(np.asarray(pcd.points)))

    print("Estimating normals")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    print("Orienting normals")
    pcd.orient_normals_consistent_tangent_plane(k=20)

    radii = [0.002, 0.004, 0.008]
    print(f"Running ball pivoting with radii {radii}")

    mesh_bpa = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(radii)
    )
    mesh_bpa = mesh_bpa.remove_unreferenced_vertices()
    mesh_bpa = mesh_bpa.remove_degenerate_triangles()
    mesh_bpa = mesh_bpa.remove_duplicated_triangles()
    mesh_bpa = mesh_bpa.remove_duplicated_vertices()
    mesh_bpa = mesh_bpa.remove_non_manifold_edges()

    point_map_pcd = o3d.geometry.PointCloud()
    point_map_pcd.points = o3d.utility.Vector3dVector(point_map.cpu().numpy())
    
    point_map_tree = o3d.geometry.KDTreeFlann(point_map_pcd)

    colors = []
    for vertex in np.asarray(mesh_bpa.vertices):
        _, idx, _ = point_map_tree.search_knn_vector_3d(vertex, 1)
        color = point_map_colors[idx[0]]
        colors.append(color.cpu().numpy())
    
    mesh_bpa.vertex_colors = o3d.utility.Vector3dVector(np.array(colors).reshape(-1, 3))

    if visualize:
        rr.log(
        f"ball_pivoting/{'_'.join(map(str, radii))}",
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh_bpa.vertices),
            triangle_indices=np.asarray(mesh_bpa.triangles),
            vertex_normals=np.asarray(mesh_bpa.vertex_normals),
            vertex_colors=np.asarray(mesh_bpa.vertex_colors),
        ),
    )
    return mesh_bpa


def main():
    visualize = False
    if visualize:   
        from jans_scripts.visualize import setup_rerun
        import rerun as rr
        setup_rerun("pcd_to_mesh_debug")

    graspnet_data_path = get_graspnet_data_path()
    camera = "realsense"
    view = "0000"
    
    for scene_id in get_graspnet_test_scene_ids():
        print(f"Processing scene {scene_id}")
        scene_dir = f"{graspnet_data_path}/output/GraspNet/rayst3r_zero_shot/{camera}_{scene_id}_{view}"
        pcd_path = os.path.join(scene_dir, "predictions/inference_points.ply")
        mesh_path = os.path.join(scene_dir, 'mesh', "mesh.ply")
        os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
        mesh = mesh_from_pcd(pcd_path, scene_id, camera, view, visualize)
        o3d.io.write_triangle_mesh(mesh_path, mesh) 


if __name__ == "__main__":
    main()
