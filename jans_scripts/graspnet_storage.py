import torch
import numpy as np
import trimesh

from jans_scripts.utils import parse_graspnet_scene
from jans_scripts.geometry import transform_pointcloud

# graspnet_data_path = "/data/graspnet"
graspnet_data_path = "/data/user/jan/graspnet"

def get_scene_gt_points(scene_id: str, camera: str, view: str = "0000"):
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