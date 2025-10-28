import os

from jans_scripts.graspnet_storage import get_graspnet_test_scene_ids, load_rgb, load_depth, load_mask, load_cam2world, load_intrinsics
from jans_scripts.rayst3r_storage import get_rayst3r_data_path, save_frame

camera = "realsense"

def main():
    for scene_id in get_graspnet_test_scene_ids():
        view = "0000"
        camera_dir = os.path.join(get_rayst3r_data_path(), f"{camera}_{scene_id}_{view}")
        # if os.path.exists(camera_dir):
        #     print(f"Frame {scene_id}_{view} already exists")
        #     continue
        os.makedirs(camera_dir, exist_ok=True)
        print(f"Saving frame {scene_id}_{view} to {camera_dir}")

        rgb = load_rgb(scene_id, camera, view)
        depth = load_depth(scene_id, camera, view)
        mask = load_mask(scene_id, camera, view)
        cam2world = load_cam2world(scene_id, camera, view)
        intrinsics = load_intrinsics(scene_id, camera)[0]

        save_frame(camera_dir, rgb, mask, intrinsics, cam2world, depth)

if __name__ == "__main__":
    main()