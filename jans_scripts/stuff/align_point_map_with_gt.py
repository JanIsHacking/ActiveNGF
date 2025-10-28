import numpy as np
import torch
from PIL import Image
from typing import Tuple
from scipy.spatial.transform import Rotation as R

from jans_scripts.visualize import setup_rerun, visualize_camera, visualize_gt_points
from jans_scripts.graspnet_storage import get_scene_gt_points


graspnet_data_path = "/data/user/jan/graspnet"

def main():
    setup_rerun('ActiveNGF')

    scene_id = "scene_0100"
    camera = 'realsense'
    views = [
        "0200",
        "0206",
        "0212",
    ]
    gt_points = get_scene_gt_points(scene_id, camera, views[0])

    for view in views:
        rs_wrt_kn_path = f"{graspnet_data_path}/scenes/{scene_id}/rs_wrt_kn.npy"
        rs_wrt_kn = np.load(rs_wrt_kn_path)

        # Display the view
        rgb_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/rgb/{view}.png"
        rgb = torch.from_numpy(np.array(Image.open(rgb_path))).float()
        depth_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/depth/{view}.png"
        depth = torch.from_numpy(np.array(Image.open(depth_path))).float() / 1000.0
        intrinsics_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camK.npy"
        intrinsics = torch.from_numpy(np.load(intrinsics_path)).float()

        cam2world_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/camera_poses.npy"
        cam0_wrt_table_path = f"{graspnet_data_path}/scenes/{scene_id}/{camera}/cam0_wrt_table.npy"
        cam0_wrt_table = torch.from_numpy(np.load(cam0_wrt_table_path)).float()
        cam2world = torch.from_numpy(np.load(cam2world_path)).float()[int(view)]
        cam2world = cam0_wrt_table @ cam2world
        #cam2world = cam0_wrt_table
        visualize_camera(
            rgb=rgb,
            dino_features=None,
            mask=torch.ones_like(rgb),
            depth=depth,
            intrinsics=intrinsics,
            cam2world=cam2world,
            camera_name=view,
            num_points=10000
        )

    # The gt points are currently in the camera coordinates for view/camera 0000
    # Use the cam0 wrt table matrix to transform them to the world coordinates
    gt_points_h = torch.cat([gt_points, torch.ones((gt_points.shape[0], 1))], dim=1)
    gt_points_t = gt_points_h @ cam0_wrt_table.T
    gt_points_t = gt_points_t[:, :3]
    visualize_gt_points(gt_points_t)
    

if __name__ == "__main__":
    main()