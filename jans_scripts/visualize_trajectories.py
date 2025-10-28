import json
import os
import torch
import numpy as np
from PIL import Image

from jans_scripts.visualize import setup_rerun, visualize_camera

def visualize_trajectories():
    scene_id = "scene_0105"
    setup_rerun(f"visualize_trajectories_{scene_id}")

    scene_dir = f"/data/user/jan/graspnet/scenes/{scene_id}/realsense"
    results_path = f"output/GraspNet/baseline/{scene_id}_nbv/results.json"
    trajectory = json.load(open(results_path, "r"))['chosen_indices']

    cam0_wrt_table_path = os.path.join(scene_dir, "cam0_wrt_table.npy")
    cam0_wrt_table = torch.from_numpy(np.load(cam0_wrt_table_path)).float()

    cam2world_path = os.path.join(scene_dir, "camera_poses.npy")
    cam2worlds = torch.from_numpy(np.load(cam2world_path)).float()

    camK_path = os.path.join(scene_dir, "camK.npy")
    camK = torch.from_numpy(np.load(camK_path)).float()

    for idx, frame_idx in enumerate(trajectory):
        rgb_path = os.path.join(scene_dir, "rgb", f"{frame_idx:04d}.png")
        rgb = torch.from_numpy(np.array(Image.open(rgb_path))).float()
        mask = torch.ones_like(rgb)
        depth_path = os.path.join(scene_dir, "depth", f"{frame_idx:04d}.png")
        depth = torch.from_numpy(np.array(Image.open(depth_path))).float() / 1000.0

        cam2world = cam2worlds[frame_idx]
        cam2world = cam0_wrt_table @ cam2world
        visualize_camera(rgb, None, mask, depth, camK, cam2world, num_points=10000, camera_name=idx)
    

if __name__ == "__main__":
    visualize_trajectories()