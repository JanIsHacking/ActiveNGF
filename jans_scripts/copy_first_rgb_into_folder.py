import os
import shutil

from jans_scripts.graspnet_storage import get_graspnet_data_path


def copy_nth_rgb_into_folder(graspnet_data_path: str, output_dir: str, camera: str, n: int):
    os.makedirs(output_dir, exist_ok=True)
    for scene_id in range(100, 190):
        scene_id_str = f"scene_{scene_id:04d}"
        n_str = f"{n:04d}"
        rgb_path = f"{graspnet_data_path}/scenes/{scene_id_str}/{camera}/rgb/{n_str}.png"
        shutil.copy(rgb_path, f"{output_dir}/{camera}_{scene_id_str}_{n_str}.png")


if __name__ == "__main__":
    graspnet_data_path = get_graspnet_data_path()
    output_path = os.path.join(graspnet_data_path, "first_rgbs")
    copy_nth_rgb_into_folder(graspnet_data_path, output_path, "realsense", 0)