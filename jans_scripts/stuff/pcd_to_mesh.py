import open3d as o3d
import torch
import numpy as np
import os
import cv2
from matplotlib import cm

from jans_scripts.graspnet_storage import get_graspnet_data_path, get_graspnet_test_scene_ids, load_cam0_wrt_table
from jans_scripts.geometry import compute_pointmap
from jans_scripts.graspnet_storage import load_mask, load_depth, load_rgb, load_intrinsics
from GraspNet.inference import GraspnessPredictor
from src.config import load_config

ADD_TABLE = True
COLOR_WITH_GRASPNESS = True


def tensor_to_colormap(x: torch.Tensor, color_map: str = 'viridis') -> torch.Tensor:
    # Normalize x to [0, 1]
    x_min, x_max = x.min(), x.max()
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    
    # Get viridis colormap from matplotlib
    colormap = cm.get_cmap(color_map)
    
    # Map normalized values to colors (RGBA)
    colors = colormap(x_norm.cpu().numpy())  # shape (n, 4)
    
    # Convert to torch tensor and drop alpha channel
    colors = torch.from_numpy(colors[:, :3]).to(x.device, dtype=x.dtype)
    
    return colors


def color_mesh_from_point_map(mesh, point_map, point_map_colors, table=None, table_threshold: float = 0.002):
    """
    Colors a mesh based on nearest points from point_map, except for vertices near table points
    (within table_threshold), which get the color corresponding to index 0.

    Args:
        mesh: open3d.geometry.TriangleMesh
        point_map: torch.Tensor of shape (N, 3)
        point_map_colors: torch.Tensor of shape (N, 3)
        table: optional torch.Tensor of shape (M, 3)
        table_threshold: float, threshold distance in meters for detecting table proximity

    Returns:
        Colored mesh (open3d.geometry.TriangleMesh)
    """
    # Convert point map to Open3D point cloud
    point_map_pcd = o3d.geometry.PointCloud()
    point_map_pcd.points = o3d.utility.Vector3dVector(point_map.cpu().numpy())
    point_map_tree = o3d.geometry.KDTreeFlann(point_map_pcd)

    # Build KD-tree for the table if provided
    table_tree = None
    if table is not None:
        table_pcd = o3d.geometry.PointCloud()
        table_pcd.points = o3d.utility.Vector3dVector(table.cpu().numpy())
        table_tree = o3d.geometry.KDTreeFlann(table_pcd)

    colors = []
    point_map_np = point_map.cpu().numpy()
    point_map_colors_np = point_map_colors.cpu().numpy()

    for vertex in np.asarray(mesh.vertices):
        # If table is provided, check if the vertex is close to it
        if table_tree is not None:
            _, idx_table, dist_table = table_tree.search_knn_vector_3d(vertex, 1)
            if np.sqrt(dist_table[0]) < table_threshold:
                colors.append(point_map_colors_np[0])
                continue  # Skip normal coloring

        # Otherwise, get color from point map
        _, idx, _ = point_map_tree.search_knn_vector_3d(vertex, 1)
        color = point_map_colors_np[idx[0]]
        colors.append(color)

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors).reshape(-1, 3))
    return mesh


def mesh_from_pcd(pcd_path, scene_id, camera, view, visualize=False):
    if visualize:
        import rerun as rr
    depth = load_depth(scene_id, camera, view)
    rgb = load_rgb(scene_id, camera, view)
    intrinsics = load_intrinsics(scene_id, camera)[0]
    mask = load_mask(scene_id, camera, view)
    cam0_wrt_table = load_cam0_wrt_table(scene_id, camera)

    point_map = compute_pointmap(depth, intrinsics, cam0_wrt_table)
    point_map_points = point_map[mask]
    point_map_colors = rgb[mask] / 255.0

    if visualize:
        rr.log("world/input/pointmap", rr.Points3D(point_map_points.cpu().numpy(), colors=point_map_colors.cpu().numpy()))

    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.transform(cam0_wrt_table.numpy())
    table = None
    if ADD_TABLE:
        inverted_mask = ~mask
        table = point_map[inverted_mask]
        table_pcd = o3d.geometry.PointCloud()
        table_pcd.points = o3d.utility.Vector3dVector(table.cpu().numpy())
        pcd = pcd + table_pcd
        print(f"Added {table.shape[0]} table points")

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

    if COLOR_WITH_GRASPNESS:
        cfg = load_config('configs/ESLAM.yaml')
        cfg['grasp_checkpoint'] = 'ckpts/graspness.tar'
        cfg['cam']['H'] = 720
        cfg['cam']['W'] = 1280
        grasper = GraspnessPredictor(cfg, eslam=None)
        # target_shape = (1200, 680)
        # depth_resized = cv2.resize(depth.cpu().numpy(), target_shape)
        # depth_resized = torch.from_numpy(depth_resized).unsqueeze(0).to(depth.device)
        graspness, objectness = grasper.inference(depth)
        graspness = graspness.squeeze(0)
        point_map_graspness = graspness[mask]
        point_map_graspness = point_map_graspness
        point_map_graspness = tensor_to_colormap(point_map_graspness, color_map='plasma')

        mesh_bpa = color_mesh_from_point_map(mesh_bpa, point_map_points, point_map_graspness, table=table)
    else:
        mesh_bpa = color_mesh_from_point_map(mesh_bpa, point_map_points, point_map_colors, table=table)
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
