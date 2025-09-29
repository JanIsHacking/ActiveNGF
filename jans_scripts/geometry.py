import torch
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation as R


def furthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Perform Furthest Point Sampling (FPS) on a set of points.

    Args:
        points (np.ndarray): Array of shape (N, D), where N is the number of points
                             and D is the dimensionality (e.g., 3 for 3D points).
        num_samples (int): Number of points to sample.

    Returns:
        np.ndarray: Subset of points of shape (num_samples, D).
    """
    N, D = points.shape
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.ones(N) * np.inf

    # Pick a random seed point
    seed_idx = np.random.randint(0, N)
    sampled_indices[0] = seed_idx

    for i in range(1, num_samples):
        # Update distances to the set of chosen points
        last_sampled = points[sampled_indices[i - 1]]
        dist = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist)

        # Pick the farthest point from current set
        sampled_indices[i] = np.argmax(distances)

    return points[sampled_indices]


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