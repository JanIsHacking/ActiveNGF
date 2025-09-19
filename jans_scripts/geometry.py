import torch
import trimesh
import os
import numpy as np


def invert_se3_batch(T: torch.Tensor) -> torch.Tensor:
    """
    Invert a batch of SE(3) transformation matrices.

    Args:
        T (torch.Tensor): Tensor of shape (N, 4, 4)

    Returns:
        torch.Tensor: Inverted transformations of shape (N, 4, 4)
    """
    R = T[:, :3, :3]  # (N, 3, 3)
    t = T[:, :3, 3:]  # (N, 3, 1)

    R_inv = R.transpose(-1, -2)         # R^T
    t_inv = -R_inv @ t                  # -R^T * t

    T_inv = torch.eye(4, device=T.device).repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv.squeeze(-1)

    return T_inv

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
