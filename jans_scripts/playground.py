import os
import glob
import numpy as np
import torch
from PIL import Image

from jans_scripts.visualize import setup_rerun, visualize_camera


def main():
    setup_rerun('ActiveNGF')
    scenes_path = '/home/jan/thesis/ActiveNGF/data/graspnet/scenes'
    scene = 'scene_0100'
    scene_path = os.path.join(scenes_path, scene)

    depths_path = os.path.join(scene_path, 'realsense', 'depth')
    rgbs_path = os.path.join(scene_path, 'realsense', 'rgb')
    cam0_wrt_table_path = os.path.join(scene_path, 'realsense', 'cam0_wrt_table.npy')
    camera_poses_path = os.path.join(scene_path, 'realsense', 'camera_poses.npy')
    camK_path = os.path.join(scene_path, 'realsense', 'camK.npy')

    depths = sorted(glob.glob(os.path.join(depths_path, '*.png')))
    rgbs = sorted(glob.glob(os.path.join(rgbs_path, '*.png')))
    cam0_wrt_table = torch.from_numpy(np.load(cam0_wrt_table_path))
    print(cam0_wrt_table)
    camera_poses = torch.from_numpy(np.load(camera_poses_path)).float()
    camK = torch.from_numpy(np.load(camK_path))

    print(f"visualizing {len(depths)} cameras")

    for depth_path, rgb_path, camera_pose, idx in zip(depths, rgbs, camera_poses, range(len(depths))):
        if idx >= 3:
            break
        depth = torch.from_numpy(np.array(Image.open(depth_path))) / 1000.0
        rgb = torch.from_numpy(np.array(Image.open(rgb_path)))
        mask = torch.ones(rgb.shape[0], rgb.shape[1])

        trans = np.dot(cam0_wrt_table.cpu().numpy(), camera_pose.cpu().numpy())
        transform_x = np.asarray([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        trans = np.dot(trans, transform_x)
        c2w = torch.from_numpy(trans).float()

        visualize_camera(rgb, None, mask, depth, camK, camera_pose, num_points=10000, camera_name=idx)
    


if __name__ == "__main__":
    main()