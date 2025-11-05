import os
import numpy as np
from PIL import Image

from jans_scripts.graspnet_storage import get_graspnet_data_path


methods = [
    "baseline",
    "gt",
    "baseline_random_nbv",
]

def bundle_quali_method_results(scene_id: str) -> None:
    print(f"Bundling method results for scene {scene_id}")
    graspnet_path = get_graspnet_data_path()
    output_dir = os.path.join(graspnet_path, "output", "GraspNet")
    images = []
    for method in methods:
        render_dir = os.path.join(output_dir, method, scene_id, "render")
        if not os.path.exists(render_dir):
            continue
        render_files = os.listdir(render_dir)
        if not render_files:
            continue
        render_file = os.path.join(render_dir, render_files[0])
        
        render_image = np.array(Image.open(render_file))
        # Remove the first and last 100 pixels
        render_image = render_image[100:-100, 100:-100]
        images.append(render_image)
    
    combined_image = np.vstack(images)
    Image.fromarray(combined_image).save(os.path.join(graspnet_path, "output", "GraspNet", f"combined_render_{scene_id}.png"))


if __name__ == "__main__":
    interesting_scenes = [
        125,
        128,
        138,
        142,
        155,
        159,
        160,
        173,
        183
    ]
    for scene_id in interesting_scenes:
        bundle_quali_method_results(f"scene_{scene_id:04d}_nbv")