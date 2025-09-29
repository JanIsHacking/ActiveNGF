import numpy as np
import trimesh
import rerun as rr
import torch
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple
from scipy.spatial.transform import Rotation as R

from jans_scripts.visualize import setup_rerun, visualize_camera, visualize_gt_points

graspnet_data_path = "/data/user/jan/graspnet"


def parse_graspnet_scene(xml_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a GraspNet XML scene file containing object poses.

    Args:
        xml_path: Path to the XML file.

    Returns:
        A list of dictionaries, one per object, with fields:
        - obj_id (int)
        - obj_name (str)
        - obj_path (str)
        - pos_in_world (list[float], length 3)
        - ori_in_world (list[float], length 4)
    """
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("obj"):
        obj_id = int(obj.find("obj_id").text)
        obj_name = obj.find("obj_name").text.strip()
        obj_path = obj.find("obj_path").text.strip()

        pos_in_world = [float(x) for x in obj.find("pos_in_world").text.split()]
        ori_in_world = [float(x) for x in obj.find("ori_in_world").text.split()]
        ori_in_world = [ori_in_world[1], ori_in_world[2], ori_in_world[3], ori_in_world[0]]

        objects.append(
            {
                "obj_id": obj_id,
                "obj_name": obj_name,
                "obj_path": obj_path,
                "pos_in_world": pos_in_world,
                "ori_in_world": ori_in_world,
            }
        )

    return objects


def transform_pointcloud(
    pointcloud: np.ndarray,
    pos_in_world: Tuple[float, float, float],
    ori_in_world: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Transform a point cloud from object coordinates into world coordinates.

    Args:
        pointcloud: (N, 3) numpy array of 3D points in object coordinates.
        pos_in_world: (3,) translation vector [x, y, z].
        ori_in_world: (4,) quaternion [x, y, z, w].

    Returns:
        (N, 3) numpy array of transformed points in world coordinates.
    """
    pointcloud = np.asarray(pointcloud)
    if pointcloud.shape[1] != 3:
        raise ValueError("pointcloud must be of shape (N, 3)")

    # Create rotation matrix from quaternion
    rot = R.from_quat(ori_in_world)
    R_mat = rot.as_matrix()  # (3, 3)

    # Apply transformation
    transformed = (R_mat @ pointcloud.T).T + np.array(pos_in_world)

    return transformed


def main():
    setup_rerun('ActiveNGF')

    scene_id = "scene_0100"
    camera = 'realsense'
    object_ids_path = f"{graspnet_data_path}/scenes/{scene_id}/object_id_list.txt"
    with open(object_ids_path, "r") as f:
        object_ids = [int(x) for x in f.read().split("\n") if x != ""]
    
    # Load objects
    object_mesh_map = {}
    for object_id in object_ids:
        object_path = f"{graspnet_data_path}/models/{object_id:03d}/textured.obj"
        mesh = trimesh.load(object_path)
        if isinstance(mesh, trimesh.Scene):
            scene_meshes = []
            for scene_mesh in mesh.geometry.values():
                scene_meshes.append(scene_mesh)
            mesh = trimesh.util.concatenate(scene_meshes)
        object_mesh_map[object_id] = mesh
        #rr.log(f"world/{scene_id}/{object_id:03d}/pointmap", rr.Points3D(points, colors=[0, 0, 255]))

    # Load annotations
    view_0 = "0000"
    annotations_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/annotations/{view_0}.xml"
    annotations = parse_graspnet_scene(annotations_path)
    
    gt_points = []
    for annotation in annotations:
        object_id = annotation["obj_id"]
        mesh = object_mesh_map[object_id]
        points = trimesh.sample.sample_surface(mesh, 5000)[0]
        points = transform_pointcloud(points, annotation["pos_in_world"], annotation["ori_in_world"])
        # rr.log(f"world/{scene_id}/{object_id:03d}/pointmap", rr.Points3D(points, colors=[0, 0, 255]))
        gt_points.append(points)
    gt_points = torch.from_numpy(np.concatenate(gt_points, axis=0)).float()

    rs_wrt_kn_path = f"{graspnet_data_path}/scenes/{scene_id}/rs_wrt_kn.npy"
    rs_wrt_kn = np.load(rs_wrt_kn_path)

    # Display the view
    rgb_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/rgb/{view_0}.png"
    rgb = torch.from_numpy(np.array(Image.open(rgb_path))).float()
    depth_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/depth/{view_0}.png"
    depth = torch.from_numpy(np.array(Image.open(depth_path))).float() / 1000.0
    intrinsics_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camK.npy"
    intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()

    cam2world_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camera_poses.npy"
    cam0_wrt_table_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/cam0_wrt_table.npy"
    cam0_wrt_table = torch.from_numpy(np.load(cam0_wrt_table_path)).float()
    cam2world = torch.from_numpy(np.load(cam2world_path)).float()[int(view_0)]
    cam2world = cam2world @ cam0_wrt_table
    visualize_camera(
        rgb=rgb,
        dino_features=None,
        mask=torch.ones_like(rgb),
        depth=depth,
        intrinsics=intrinsics,
        cam2world=cam2world,
        camera_name=view_0
    )

    # The gt points are currently in the camera coordinates for view/camera 0000
    # Use the cam0 wrt table matrix to transform them to the world coordinates
    gt_points_h = torch.cat([gt_points, torch.ones((gt_points.shape[0], 1))], dim=1)
    gt_points_t = gt_points_h @ cam0_wrt_table.T
    gt_points_t = gt_points_t[:, :3]
    visualize_gt_points(gt_points_t)
    

if __name__ == "__main__":
    main()