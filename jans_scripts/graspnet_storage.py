import os
import torch
import numpy as np
import trimesh
from PIL import Image

from jans_scripts.utils import parse_graspnet_scene
from jans_scripts.geometry import transform_pointcloud

graspnet_data_path_container = "/data/graspnet"
graspnet_data_path_tesu = "/data/user/jan/graspnet"

def get_graspnet_test_scene_ids():
    graspnet_data_path = get_graspnet_data_path()
    scene_ids_path = f"{graspnet_data_path}/scenes"
    scene_ids = sorted([x for x in os.listdir(scene_ids_path) if os.path.isdir(os.path.join(scene_ids_path, x))])
    assert len(scene_ids) == 90, "Expected 90 test scenes"
    return scene_ids

def get_graspnet_view_ids():
    return [f'{view:04d}' for view in range(256)]

def get_graspnet_data_path():
    if os.path.exists(graspnet_data_path_tesu):
        graspnet_data_path = graspnet_data_path_tesu
    elif os.path.exists(graspnet_data_path_container):
        graspnet_data_path = graspnet_data_path_container
    else:
        raise ValueError(f"Graspnet data path {graspnet_data_path_tesu} or {graspnet_data_path_container} does not exist")
    return graspnet_data_path   

def load_cam0_wrt_table(scene_id: str, camera: str):
    graspnet_data_path = get_graspnet_data_path()
    cam0_wrt_table_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/cam0_wrt_table.npy"
    return torch.from_numpy(np.load(cam0_wrt_table_path)).float()

def load_rgb(scene_id: str, camera: str, view: str = "0000"):
    graspnet_data_path = get_graspnet_data_path()
    rgb_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/rgb/{view}.png"
    rgb = torch.from_numpy(np.array(Image.open(rgb_path))).float()
    return rgb

def load_depth(scene_id: str, camera: str, view: str = "0000"):
    graspnet_data_path = get_graspnet_data_path()
    depth_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/depth/{view}.png"
    depth = torch.from_numpy(np.array(Image.open(depth_path))).float() / 1000.0
    return depth

def load_mask(scene_id: str, camera: str, view: str = "0000"):
    graspnet_data_path = get_graspnet_data_path()
    mask_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/label/{view}.png"
    mask = torch.from_numpy(np.array(Image.open(mask_path)))
    mask[mask > 0] = 1
    mask = mask.bool()
    return mask

def load_cam2world(scene_id: str, camera: str, view: str = "0000"):
    graspnet_data_path = get_graspnet_data_path()
    cam2world_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camera_poses.npy"
    cam2world = torch.from_numpy(np.load(cam2world_path)).float()[int(view)]
    return cam2world

def load_intrinsics(scene_id: str, camera: str):
    graspnet_data_path = get_graspnet_data_path()
    intrinsics_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camK.npy"
    intrinsics = torch.from_numpy(np.load(intrinsics_path)).float().reshape(1, 3, 3)
    return intrinsics

def get_scene_gt_points(scene_id: str, camera: str, view: str = "0000"):
    graspnet_data_path = get_graspnet_data_path()
    object_ids_path = f"{graspnet_data_path}/scenes/{scene_id}/object_id_list.txt"
    with open(object_ids_path, "r") as f:
        object_ids = [int(x) for x in f.read().split("\n") if x != ""]

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
    
    # Load annotations
    annotations_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/annotations/{view}.xml"
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
    return gt_points


def get_scene_objects_bounding_boxes(scene_id: str, camera: str, view: str = "0000"):
    graspnet_data_path = get_graspnet_data_path()
    object_ids_path = f"{graspnet_data_path}/scenes/{scene_id}/object_id_list.txt"
    with open(object_ids_path, "r") as f:
        object_ids = [int(x) for x in f.read().split("\n") if x != ""]

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
    
    # Load annotations
    annotations_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/annotations/{view}.xml"
    annotations = parse_graspnet_scene(annotations_path)

    bounding_boxes = dict()
    object_ids = []
    cam2world_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camera_poses.npy"
    cam0_wrt_table_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/cam0_wrt_table.npy"
    cam0_wrt_table = torch.from_numpy(np.load(cam0_wrt_table_path)).float()
    cam2world = torch.from_numpy(np.load(cam2world_path)).float()[int(view)]
    cam2world = cam2world @ cam0_wrt_table
    cam2world = cam2world.cpu().numpy()
    for annotation in annotations:
        object_id = annotation["obj_id"]
        mesh = object_mesh_map[object_id]
        bounding_box = np.array(mesh.bounds)
        bounding_box_t = transform_pointcloud(bounding_box, annotation["pos_in_world"], annotation["ori_in_world"])
        bounding_box_t_h = np.concatenate([bounding_box_t, np.ones((bounding_box_t.shape[0], 1))], axis=1)
        bounding_box_t_h_t = (cam2world @ bounding_box_t_h.T).T
        bounding_box_t_h_t = bounding_box_t_h_t[:, :3]
        bounding_boxes[object_id] = bounding_box_t_h_t
        object_ids.append(object_id)
    return bounding_boxes, object_ids