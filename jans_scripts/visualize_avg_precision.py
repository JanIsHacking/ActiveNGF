import json
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict

base_path = "/data/graspnet/output/GraspNet/"
mapping_depth_sources = ["baseline", "rayst3r", "gt"]

# Scene splits according to GraspNet test categories
splits = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}


def process_depth_source(base_path: str, mapping_depth_source: str, scene_range: range, random_nbv: bool = False) -> dict:
    """Aggregate mean accuracy per frame index for a given depth source and scene range."""
    frame_acc_sum: dict[int, float] = defaultdict(float)
    frame_count: dict[int, int] = defaultdict(int)
    
    for scene_idx in scene_range:
        scene_folder = os.path.join(base_path, mapping_depth_source_str, f"scene_{scene_idx:04d}_nbv", "eval_out")
        if not os.path.exists(scene_folder):
            continue

        for result_file in glob.glob(os.path.join(scene_folder, "results_*.json")):
            frame_index = int(result_file.split("_")[-1].split(".")[0])
            with open(result_file, "r") as f:
                data = json.load(f)
                mean_acc = data.get("mean_accuracy", 0.0)
                frame_acc_sum[frame_index] += mean_acc
                frame_count[frame_index] += 1

    avg_mean_acc: dict[int, float] = {}
    for frame_index in sorted(frame_acc_sum.keys()):
        if frame_count[frame_index] > 0:
            avg_mean_acc[frame_index] = frame_acc_sum[frame_index] / frame_count[frame_index]
            print(f"[{mapping_depth_source}] Frame {frame_index}: {frame_count[frame_index]} scenes aggregated")

    return avg_mean_acc


# Main aggregation and plotting
for split_name, scene_range in splits.items():
    split_data = {}

    # Process each depth source
    for mapping_depth_source in mapping_depth_sources:
        for random_nbv in [True, False]:
            mapping_depth_source_str = mapping_depth_source + "_random_nbv" if random_nbv else mapping_depth_source
            avg_mean_acc = process_depth_source(base_path, mapping_depth_source_str, scene_range, random_nbv)
            split_data[mapping_depth_source_str] = avg_mean_acc

    # Plot all sources on one figure
    plt.figure(figsize=(10, 6))
    min_value = float('inf')
    max_value = float('-inf')
    for mapping_depth_source_str, avg_mean_acc in split_data.items():
        plt.plot(
            list(avg_mean_acc.keys()),
            list(avg_mean_acc.values()),
            marker="o",
            label=mapping_depth_source_str
        )
        if len(avg_mean_acc.values()) > 0:
            min_value = min(min_value, min(avg_mean_acc.values()))
            max_value = max(max_value, max(avg_mean_acc.values()))

    plt.xlabel("Plan Step")
    plt.ylabel("Average Mean Accuracy (AP)")
    plt.title(f"Average Mean Accuracy vs Frame Index ({split_name.capitalize()} Scenes)")
    plt.ylim(min_value-0.02, max_value+0.02)
    plt.grid(True)
    plt.legend()

    os.makedirs("/workspace/jans_scripts/data", exist_ok=True)
    out_path = f"/workspace/jans_scripts/data/avg_mean_acc_{split_name}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot for {split_name} scenes → {out_path}")
