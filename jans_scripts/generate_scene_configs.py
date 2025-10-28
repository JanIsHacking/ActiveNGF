from typing import List
import yaml
import os

def generate_scene_yaml(scene_numbers: List[int], config_path: str) -> List[str]:
    base_yaml = {
        "scene": "scene_0100",
        "inherit_from": "configs/GraspNet/graspnet.yaml",
        "config_path": "configs/GraspNet/scene_0100.yaml",
        "data": {
            "input_folder": "/data/graspnet/scenes/scene_0100/realsense",
            "output": "/data/graspnet/output/GraspNet/scene_0100_nbv/"
        },
        "mapping": {
            "keyframe_every": 1,
            "mesh_freq": 1,
            "vis_freq": 100,
            "vis_inside_freq": 100,
            "iters_first": 150,
            "iters": 50,
            "ckpt_freq": 1,
            "no_log_on_first_frame": False,
            "random_nbv": False
        },
        "rendering": {
            "n_stratified": 32,
            "n_importance": 8
        },
        "model": {
            "grasp_output": "online"
        }
    }

    depth_source_yamls = dict()
    for depth_source in ["baseline", "rayst3r", "gt"]:
        for random_nbv in [True, False]:
            yamls = []
            depth_source_str = depth_source + '_random_nbv' if random_nbv else depth_source
            for num in scene_numbers:
                scene_id = f"scene_{num:04d}"
                yaml_copy = base_yaml.copy()
                yaml_copy["scene"] = scene_id
                yaml_copy["config_path"] = f"configs/GraspNet/{depth_source_str}/{scene_id}.yaml"
                yaml_copy["data"]["input_folder"] = f"/data/graspnet/scenes/{scene_id}/realsense"
                yaml_copy["data"]["output"] = f"/data/graspnet/output/GraspNet/{depth_source_str}/{scene_id}_nbv/"
                yaml_copy["mapping"]["random_nbv"] = random_nbv
                yamls.append(yaml.dump(yaml_copy, sort_keys=False))
            depth_source_yamls[depth_source_str] = yamls
    
    return depth_source_yamls

# Example usage:
config_path = "/workspace/configs/GraspNet"
scene_ids = list(range(100, 190))
depth_source_yamls = generate_scene_yaml(scene_ids, config_path)
for depth_source, yamls in depth_source_yamls.items():
    for y, scene_id in zip(yamls, scene_ids):
        yaml_path = os.path.join(config_path, f"{depth_source}/scene_{scene_id:04d}.yaml")
        yaml_dir = os.path.dirname(yaml_path)
        os.makedirs(yaml_dir, exist_ok=True)
        with open(yaml_path, "w") as f:
            f.write(y)
