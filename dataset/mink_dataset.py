import os
import sys
import numpy as np
import scipy.io as scio
from PIL import Image
import trimesh

import torch
# from torch._six import container_abcs
import collections.abc as container_abcs
from torch.utils.data import Dataset
import MinkowskiEngine as ME
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_GRASP_DIR = os.path.join(ROOT_DIR, 'Generalizing-Grasp')
sys.path.append(os.path.join(GEN_GRASP_DIR, 'utils'))
from data_utils import remove_invisible_grasp_points
from graspnetAPI.utils.utils import xmlReader, parse_posevector


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


class GraspNetSingleSceneDataset(Dataset):
    def __init__(self, root, mesh_file, scene_id, valid_obj_idxs, grasp_labels, camera='kinect', split='train', num_points=20000,
                 remove_outlier=False, remove_invisible=True,voxel_size = 0.005):
        assert (num_points <= 50000)
        self.root = root
        self.mesh_file = mesh_file
        self.scene_id = scene_id
        self.split = split
        self.num_points = num_points
        self.remove_outlier = remove_outlier
        self.remove_invisible = remove_invisible
        self.valid_obj_idxs = valid_obj_idxs
        self.grasp_labels = grasp_labels
        self.camera = camera
        self.collision_labels = {}
        self.voxel_size = voxel_size

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.get_data(index)

    def get_data(self, index, return_raw_cloud=False):
        assert index == 0
        mesh = trimesh.load(self.mesh_file)
        if isinstance(mesh, trimesh.Scene):
            scene_meshes = []
            for scene_mesh in mesh.geometry.values():
                scene_meshes.append(scene_mesh)
            mesh = trimesh.util.concatenate(scene_meshes)
        point_cloud, face_indices = trimesh.sample.sample_surface(mesh, self.num_points)
        normals = mesh.face_normals[face_indices]

        colors = mesh.visual.vertex_colors[:, :3]
        faces = mesh.faces[face_indices]
        bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_indices], point_cloud)
        sampled_colors = (colors[faces] * bary[..., None]).sum(axis=1)

        # seg = np.array(np.load(self.inspath[index]))
        # seg = np.array(np.load(self.sampath[index]))
        # seg = np.array(np.load(self.labelpath[index]))
        if return_raw_cloud:
            return point_cloud, None

        # if self.camera == "kinect":
        #     mask_x = ((point_cloud[:, 0] > -0.5) & (point_cloud[:, 0] <0.5))
        #     mask_y = ((point_cloud[:, 1] > -0.5) & (point_cloud[:, 1] < 0.5))
        #     mask_z = ((point_cloud[:, 2] > -0.02) & (point_cloud[:, 2] < 0.2))
        #     workspace_mask = (mask_x & mask_y & mask_z)
        #     point_cloud = point_cloud[workspace_mask]
        #     normal = normal[workspace_mask]
        #     color = color[workspace_mask]
            # seg = seg[workspace_mask]

        # sample points
        if len(point_cloud) >= self.num_points:
            idxs = np.random.choice(len(point_cloud), self.num_points, replace=False)
        else:
            idxs1 = np.arange(len(point_cloud))
            idxs2 = np.random.choice(len(point_cloud), self.num_points - len(point_cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)

        cloud_sampled = point_cloud[idxs]
        # seg_sampled = seg[idxs]
        normal_sampled = normals[idxs]
        color_sampled = sampled_colors[idxs]
        ret_dict = {}
        ret_dict['point_clouds'] = cloud_sampled.astype(np.float32)
        ret_dict['coors'] = cloud_sampled.astype(np.float32) / self.voxel_size
        ret_dict['feats'] = normal_sampled.astype(np.float32)
        ret_dict['pcd_color'] = np.concatenate([cloud_sampled.astype(np.float32), color_sampled.astype(np.float32)[:, :3]],
                                               axis=1)

        # ret_dict['instance_mask'] = seg_sampled
        return ret_dict

'''
from https://github.com/rhett-chen/graspness_implementation/blob/main/dataset/graspnet_dataset.py
'''
def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch = ME.utils.sparse_collate([d["coors"] for d in list_data],
                                                                [d["feats"] for d in list_data])
    coordinates_batch, features_batch, _, quantize2original = ME.utils.sparse_quantize(
        coordinates_batch.float(), features_batch.float(), return_index=True, return_inverse=True)
    res = {
        "coors": coordinates_batch,
        "feats": features_batch,
        "quantize2original": quantize2original
    }
    def collate_fn_(batch):
        if type(batch[0]).__module__ == 'numpy':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        elif isinstance(batch[0], container_abcs.Sequence):
            return [[torch.from_numpy(sample) for sample in b] for b in batch]
        elif isinstance(batch[0], container_abcs.Mapping):
            for key in batch[0]:
                if key == 'coors' or key == 'feats':
                    continue
                res[key] = collate_fn_([d[key] for d in batch])
            return res
    res = collate_fn_(list_data)
    return res


def collate_fn(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], container_abcs.Sequence):
        return [[torch.from_numpy(sample) for sample in b] for b in batch]

    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

