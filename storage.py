import json
import os
import torch
import numpy as np
from PIL import Image

def compute_pointmap(depth: torch.Tensor, intrinsics: torch.Tensor, cam2world: torch.Tensor = None) -> torch.Tensor:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    
    i, j = torch.meshgrid(torch.arange(w), torch.arange(h), indexing='xy')
    i = i.to(depth.device)
    j = j.to(depth.device)

    x_cam = (i - cx) * depth / fx
    y_cam = (j - cy) * depth / fy

    points_cam = torch.stack([x_cam, y_cam, depth], axis=-1)

    if cam2world is not None:
        cam2world = cam2world.to(depth.device)
        points_cam = torch.matmul(cam2world[:3, :3], points_cam.reshape(-1, 3).T).T + cam2world[:3, 3]
    points_cam = points_cam.reshape(h, w, 3)

    return points_cam


DEPTH_MAX = 10.0
TORCH_UINT16_MAX = 65535

CAM2WORLD_FILE_NAME = "cam2world.pt"
INTRINSICS_FILE_NAME = "intrinsics.pt"
MASK_FILE_NAME = "mask.png"
RGB_FILE_NAME = "rgb.png"
DINO_FEATURES_FILE_NAME = "dino_features.pt"
DEPTH_FILE_NAME = "depth.png"


def load_mask(camera_dir: str, device: str = 'cuda'):
    mask_path = os.path.join(camera_dir, MASK_FILE_NAME)
    mask = torch.from_numpy(np.array(Image.open(mask_path))).bool().to(device) if os.path.exists(mask_path) else torch.Tensor([]).to(device)
    return mask

def load_intrinsics(camera_dir: str, device: str = 'cuda'):
    intrinsics_path = os.path.join(camera_dir, INTRINSICS_FILE_NAME)
    intrinsics = torch.load(intrinsics_path).to(device) if os.path.exists(intrinsics_path) else torch.Tensor([]).to(device)
    return intrinsics


def load_cam2world(camera_dir: str, device: str = 'cuda'):
    cam2world_path = os.path.join(camera_dir, CAM2WORLD_FILE_NAME)
    cam2world = torch.load(cam2world_path).to(device) if os.path.exists(cam2world_path) else torch.Tensor([]).to(device)
    return cam2world


def load_depth(camera_dir: str, device: str = 'cuda'):
    depth_path = os.path.join(camera_dir, DEPTH_FILE_NAME)
    depth = torch.from_numpy(np.array(Image.open(depth_path))).to(device) if os.path.exists(depth_path) else torch.Tensor([]).to(device)
    depth = _unscale_depth(depth)
    return depth


def load_joint_masked_pointmaps(camera_dirs: list[str], device: str = 'cuda'):
    """
    Note that this acutally loads pointcloud as masking the pointmaps destroys its shape.
    """
    if len(camera_dirs) == 0:
        return torch.Tensor([]).to(device)
    # Join the pointmaps in the coordinate system of the first camera
    cam_0_dir = camera_dirs[0]
    cam_0_depth = load_depth(cam_0_dir, device)
    cam_0_intrinsics = torch.load(os.path.join(cam_0_dir, INTRINSICS_FILE_NAME))
    cam_0_pointmap = compute_pointmap(cam_0_depth, cam_0_intrinsics)
    cam_0_mask = load_mask(cam_0_dir, device)
    cam_0_cam2world = torch.load(os.path.join(cam_0_dir, CAM2WORLD_FILE_NAME)).to(device)
    cam_0_world2cam = torch.inverse(cam_0_cam2world)

    pointmaps = [cam_0_pointmap[cam_0_mask]]
    if len(camera_dirs) > 1:
        for camera_dir in camera_dirs[1:]:
            depth = load_depth(camera_dir, device)
            intrinsics = torch.load(os.path.join(camera_dir, INTRINSICS_FILE_NAME)).to(device)
            cam2world = torch.load(os.path.join(camera_dir, CAM2WORLD_FILE_NAME)).to(device)
            pointmap = compute_pointmap(depth, intrinsics, cam2world)  # this is not in world coordinates
            mask = load_mask(camera_dir, device)
            pointmap = pointmap[mask]
            pointmap_hom = torch.cat([pointmap, torch.ones(pointmap.shape[0], 1, device=device)], dim=1)
            pointmap_world = (cam_0_world2cam @ pointmap_hom.T).T
            pointmap = pointmap_world[:, :3]
            pointmaps.append(pointmap)
    return torch.cat(pointmaps, dim=0)


def load_pointmap_from_depth(camera_dir: str, device: str = 'cuda'):
    depth = load_depth(camera_dir, device)
    intrinsics = torch.load(os.path.join(camera_dir, INTRINSICS_FILE_NAME))
    cam2world = torch.load(os.path.join(camera_dir, CAM2WORLD_FILE_NAME))
    return compute_pointmap(depth, intrinsics, cam2world)
    

def save_tensor_as_png(tensor: torch.Tensor, path: str, dtype: torch.dtype | None = None):
    if dtype is None:
        dtype = tensor.dtype
    Image.fromarray(tensor.to(dtype).cpu().numpy()).save(path)


def _scale_depth(depth: torch.Tensor) -> torch.Tensor:
    return (depth / DEPTH_MAX * TORCH_UINT16_MAX).clamp(0, TORCH_UINT16_MAX).to(torch.uint16)


def _unscale_depth(depth: torch.Tensor) -> torch.Tensor:
    return depth * DEPTH_MAX / TORCH_UINT16_MAX


def load_preprocessed_data(camera_dir: str, device: str = 'cuda'):
    scene_dir = '/'.join(camera_dir.split('/')[:-2])

    rgb_path = os.path.join(camera_dir, RGB_FILE_NAME)
    rgb = torch.from_numpy(np.array(Image.open(rgb_path))).to(device) if os.path.exists(rgb_path) else torch.Tensor([]).to(device)

    dino_features_path = os.path.join(camera_dir, DINO_FEATURES_FILE_NAME)
    dino_features = torch.load(dino_features_path).to(device) if os.path.exists(dino_features_path) else torch.Tensor([]).to(device)

    mask = load_mask(camera_dir, device)
    depth = load_depth(camera_dir, device)

    gt_points_file_names = [x for x in os.listdir(scene_dir) if x.startswith("gt_points_in_camera_")]
    if len(gt_points_file_names) > 0:
        gt_points_file_name = gt_points_file_names.pop()
        gt_points_path = os.path.join(scene_dir, gt_points_file_name)
        gt_points = torch.load(gt_points_path).to(device) if os.path.exists(gt_points_path) else torch.Tensor([]).to(device)
    else:
        gt_points = torch.Tensor([]).to(device)
    
    cam2world_path = os.path.join(camera_dir, CAM2WORLD_FILE_NAME)
    cam2world = torch.load(cam2world_path).to(device) if os.path.exists(cam2world_path) else torch.Tensor([]).to(device)

    intrinsics_path = os.path.join(camera_dir, INTRINSICS_FILE_NAME)
    intrinsics = torch.load(intrinsics_path).to(device) if os.path.exists(intrinsics_path) else torch.Tensor([]).to(device)

    return rgb, dino_features, mask, depth, gt_points, intrinsics, cam2world


def save_frame(
        camera_dir: str,
        rgb: torch.Tensor, 
        mask: torch.Tensor, 
        intrinsics: torch.Tensor, 
        cam2world: torch.Tensor, 
        depth: torch.Tensor, 
        pointmap: torch.Tensor = None,
        dtype: torch.dtype = torch.float32
    ):
    os.makedirs(camera_dir, exist_ok=True)
    if rgb is not None:
        save_tensor_as_png(rgb, os.path.join(camera_dir, RGB_FILE_NAME))
    if mask is not None:
        save_tensor_as_png(mask, os.path.join(camera_dir, MASK_FILE_NAME), dtype=torch.bool)
    if intrinsics is not None:
        torch.save(intrinsics.cpu().to(dtype), os.path.join(camera_dir, INTRINSICS_FILE_NAME))
    if cam2world is not None:
         torch.save(cam2world.cpu().to(dtype), os.path.join(camera_dir, CAM2WORLD_FILE_NAME))
    if depth is not None:
        depth_scaled = _scale_depth(depth)
        save_tensor_as_png(depth_scaled, os.path.join(camera_dir, DEPTH_FILE_NAME), dtype=torch.uint16)


def save_dino_features(dino_features: torch.Tensor, camera_dir: str):
    torch.save(dino_features, os.path.join(camera_dir, DINO_FEATURES_FILE_NAME))


def save_gt_points(frame_dir: str, gt_points: torch.Tensor, camera_name: str = "000000", output_file_name: str | None = None, dtype = torch.float32):
    os.makedirs(frame_dir, exist_ok=True)
    if gt_points is not None:
        if output_file_name is None:
            output_file_name = f"gt_points_in_camera_{camera_name}.pt"
        torch.save(gt_points.to(dtype), os.path.join(frame_dir, output_file_name))