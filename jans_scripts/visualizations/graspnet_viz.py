import torch

from jans_scripts.graspnet_storage import get_scene_gt_points, get_cam0_wrt_table
from jans_scripts.visualize import visualize_gt_points

def visualize_graspnet_scene(scene_id: str, camera: str, view: str = "0000"):
    gt_points = get_scene_gt_points(scene_id, camera, view)
    cam0_wrt_table = get_cam0_wrt_table(scene_id, camera)

    gt_points_h = torch.cat([gt_points, torch.ones((gt_points.shape[0], 1))], dim=1)
    gt_points_t = gt_points_h @ cam0_wrt_table.T
    gt_points_t = gt_points_t[:, :3]
    visualize_gt_points(gt_points_t)