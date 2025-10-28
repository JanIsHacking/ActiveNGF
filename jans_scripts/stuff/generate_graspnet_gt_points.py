import os
import torch

from jans_scripts.graspnet_storage import get_scene_gt_points, get_graspnet_data_path

def main():
    graspnet_data_path = get_graspnet_data_path()
    scenes_dir = os.path.join(graspnet_data_path, "scenes")

    camera = "realsense"
    view = "0000"

    for scene_id in sorted(os.listdir(scenes_dir)):
        gt_points_path = os.path.join(scenes_dir, scene_id, camera, "gt_points_in_camera_0.pt")
        if os.path.exists(gt_points_path):
            print(f"GT points already exist for scene {scene_id}")
            continue
        print(f"Generating GT points for scene {scene_id}")
        gt_points = get_scene_gt_points(scene_id, camera, view)
        torch.save(gt_points, gt_points_path)

if __name__ == "__main__":
    main()
