from typing import List
import yaml
import os

def generate_scene_yaml(scene_numbers: List[int]) -> List[str]:
    base_yaml = {
        "scene": "scene_0100",
        "inherit_from": "configs/GraspNet/graspnet.yaml",
        "config_path": "configs/GraspNet/scene_0100.yaml",
        "data": {
            "input_folder": "/data/graspnet/scenes/scene_0100/realsense",
            "output": "output/GraspNet/scene_0100_nbv/"
        },
        "mapping": {
            "keyframe_every": 1,
            "mesh_freq": 1,
            "vis_freq": 100,
            "vis_inside_freq": 100,
            "iters_first": 150,
            "iters": 50,
            "ckpt_freq": 1,
            "no_log_on_first_frame": False
        },
        "rendering": {
            "n_stratified": 32,
            "n_importance": 8
        },
        "model": {
            "grasp_output": "online"
        }
    }

    yamls = []
    for num in scene_numbers:
        scene_id = f"scene_{num:04d}"
        yaml_copy = base_yaml.copy()
        yaml_copy["scene"] = scene_id
        yaml_copy["config_path"] = f"configs/GraspNet/{scene_id}.yaml"
        yaml_copy["data"]["input_folder"] = f"/data/graspnet/scenes/{scene_id}/realsense"
        yaml_copy["data"]["output"] = f"output/GraspNet/{scene_id}_nbv/"
        yamls.append(yaml.dump(yaml_copy, sort_keys=False))
    
    return yamls

# Example usage:
config_path = "/workspace/configs/GraspNet"
scene_ids = list(range(100, 190))
scene_yamls = generate_scene_yaml(scene_ids)
for y, scene_id in zip(scene_yamls, scene_ids):
    with open(os.path.join(config_path, f"scene_{scene_id:04d}.yaml"), "w") as f:
        f.write(y)
