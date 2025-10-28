import json
import os
import glob
import matplotlib.pyplot as plt

base_path = "output/GraspNet/"
mapping_depth_sources = ["baseline", "rayst3r", "gt"]

# Scene splits according to GraspNet test categories
splits = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}


def process_scene(base_path: str, mapping_depth_source: str, scene_idx: int) -> dict[int, float]:
    """Load mean accuracy per frame for a single scene and depth source."""
    scene_folder = os.path.join(base_path, mapping_depth_source, f"scene_{scene_idx:04d}_nbv", "eval_out")
    frame_acc: dict[int, float] = {}

    if not os.path.exists(scene_folder):
        print(f"Skipping missing folder: {scene_folder}")
        return frame_acc

    for result_file in glob.glob(os.path.join(scene_folder, "results_*.json")):
        frame_index = int(result_file.split("_")[-1].split(".")[0])
        with open(result_file, "r") as f:
            data = json.load(f)
            frame_acc[frame_index] = data.get("mean_accuracy", 0.0)

    return frame_acc


# Main per-scene visualization
for split_name, scene_range in splits.items():
    print(f"\nProcessing split: {split_name}")
    for scene_idx in scene_range:
        scene_data = {}

        # Process each depth source
        for mapping_depth_source in mapping_depth_sources:
            for random_nbv in [True, False]:
                mapping_depth_source_str = mapping_depth_source + "_random_nbv" if random_nbv else mapping_depth_source
                frame_acc = process_scene(base_path, mapping_depth_source_str, scene_idx)
                if frame_acc:
                    scene_data[mapping_depth_source_str] = frame_acc

        if not scene_data:
            continue

        # Plot all depth sources for this scene
        plt.figure(figsize=(10, 6))
        min_value = float("inf")
        max_value = float("-inf")

        for source_name, frame_acc in scene_data.items():
            print(source_name)
            print(list(frame_acc.keys()))
            print(list(frame_acc.values()))
            # Sort the dict by the keys
            frame_acc = dict(sorted(frame_acc.items()))
            print(source_name)
            print(list(frame_acc.keys()))
            print(list(frame_acc.values()))
            plt.plot(
                list(frame_acc.keys()),
                list(frame_acc.values()),
                marker="o",
                label=source_name
            )
            if frame_acc:
                min_value = min(min_value, min(frame_acc.values()))
                max_value = max(max_value, max(frame_acc.values()))

        plt.xlabel("Plan Step")
        plt.ylabel("Mean Accuracy (AP)")
        plt.title(f"Scene {scene_idx:04d} — {split_name.capitalize()} Split")
        plt.ylim(min_value - 0.02, max_value + 0.02)
        plt.grid(True)
        plt.legend()

        out_dir = "data/per_scene_viz"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"scene_{scene_idx:04d}_{split_name}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved per-scene plot → {out_path}")
