import json
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

from jans_scripts.graspnet_storage import get_graspnet_data_path


methods = [
    "baseline",
    "baseline_random_nbv",
    "gt",
    "rayst3r_zero_shot",
]

map_method_names = {
    "baseline": "Baseline",
    "baseline_random_nbv": "Random NBV",
    "gt": "Ground Truth Oracle",
    "rayst3r_zero_shot": "RaySt3R",
}

def create_frame_increase_comparison_chart(method_results_path: str, methods, mode: str) -> None:
    modes = ['relative', 'absolute']
    if mode not in modes:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: {modes}")
    
    scene_range = range(100, 190)
    scene_data = {scene_id: {} for scene_id in scene_range}
    infinite_increases = set()
    
    methods_without_rayst3r = [m for m in methods if m != "rayst3r_zero_shot"]
    
    for method in methods_without_rayst3r:
        method_path = os.path.join(method_results_path, method)
        if not os.path.exists(method_path):
            continue

        for scene in sorted(os.listdir(method_path)):
            if not os.path.isdir(os.path.join(method_path, scene)):
                continue
            scene_id = int(scene.split("_")[1])
            if scene_id not in scene_range:
                continue
            
            scene_path = os.path.join(method_path, scene)
            results_files = sorted(glob.glob(os.path.join(scene_path, "eval_out", "results_*.json")))
            if not results_files or len(results_files) < 2:
                continue
            
            first_result_file = results_files[0]
            last_result_file = results_files[-1]
            
            with open(first_result_file, "r") as f:
                first_data = json.load(f)
                first_accuracy = first_data["mean_accuracy"]
            
            with open(last_result_file, "r") as f:
                last_data = json.load(f)
                last_accuracy = last_data["mean_accuracy"]
            
            if mode == 'relative':
                if first_accuracy == 0 and last_accuracy != 0:
                    print(f"First accuracy is 0 for scene {scene_id} and method {method}. First accuracy: {first_accuracy}, Last accuracy: {last_accuracy}")
                    infinite_increases.add((scene_id, method))
                increase = last_accuracy / first_accuracy - 1 if first_accuracy > 0 else 0
            elif mode == 'absolute':
                increase = last_accuracy - first_accuracy
            scene_data[scene_id][method] = increase
    
    scene_ids = sorted([s for s in scene_range if scene_data[s]])
    n_scenes = len(scene_ids)
    n_methods = len(methods_without_rayst3r)
    
    if n_scenes == 0:
        print("No data found for scenes 100-189")
        return
    
    x = np.arange(n_scenes)
    width = 0.8 / n_methods
    
    fig, ax = plt.subplots(figsize=(max(12, n_scenes * 0.3), 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    color_map = dict(zip(methods_without_rayst3r, colors))
    
    for i, method in enumerate(methods_without_rayst3r):
        increases = [scene_data[scene_id].get(method, 0) for scene_id in scene_ids]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, increases, width, label=map_method_names[method], color=color_map[method])
    
    infinite_marker_handle = None
    if mode == 'relative' and infinite_increases:
        marker_added = False
        for j, scene_id in enumerate(scene_ids):
            for i, method in enumerate(methods_without_rayst3r):
                if (scene_id, method) in infinite_increases:
                    offset = (i - n_methods / 2 + 0.5) * width
                    marker_y = 0
                    ax.scatter(j + offset, marker_y, marker='X', s=100, c='black', 
                             zorder=10, linewidths=1.5, edgecolors='white')
                    if not marker_added:
                        infinite_marker_handle = ax.scatter([], [], marker='X', s=100, c='black', 
                                 label='Infinite increase', 
                                 linewidths=1.5, edgecolors='white')
                        marker_added = True
    
    ax.set_xlabel("Scene ID", fontsize=12)
    ax.set_ylabel(f"{mode.capitalize()} Increase in Average Precision", fontsize=12)
    ax.set_title(f"{mode.capitalize()} Increase in Average Precision from First to Last Mapping Step Across All Test Scenes", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scene_ids, rotation=45, ha='right')
    
    if mode == 'relative' and infinite_marker_handle is not None:
        handles, labels = ax.get_legend_handles_labels()
        handles = [h for h in handles if h != infinite_marker_handle] + [infinite_marker_handle]
        labels = [l for l in labels if l != 'Infinite increase'] + ['Infinite increase']
        ax.legend(handles, labels)
    else:
        ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    if mode == 'relative':
        ax.set_ylim(-1.2, 8)
    
    plt.tight_layout()
    
    output_path = os.path.join(method_results_path, f"{mode}_increase_frame_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def create_last_frame_comparison_chart(method_results_path: str, methods) -> None:
    scene_range = range(100, 190)
    scene_data = {scene_id: {} for scene_id in scene_range}
    
    for method in methods:
        method_path = os.path.join(method_results_path, method)
        if not os.path.exists(method_path):
            continue

        for scene in sorted(os.listdir(method_path)):
            if not os.path.isdir(os.path.join(method_path, scene)):
                continue
            if method == "rayst3r_zero_shot":
                scene_id = scene.split("_")[2]
                scene_id = int(scene_id)
            else:
                scene_id = int(scene.split("_")[1])
                if scene_id not in scene_range:
                    continue
            
            scene_path = os.path.join(method_path, scene)
            
            if method == "rayst3r_zero_shot":
                result_file = os.path.join(scene_path, "eval_out", "results.json")
                if not os.path.exists(result_file):
                    continue
                with open(result_file, "r") as f:
                    data = json.load(f)
                    scene_data[scene_id][method] = data["mean_accuracy"]
            else:
                results_files = sorted(glob.glob(os.path.join(scene_path, "eval_out", "results_*.json")))
                if not results_files:
                    continue
                result_file = results_files[-1]
            
                with open(result_file, "r") as f:
                    data = json.load(f)
                    scene_data[scene_id][method] = data["mean_accuracy"]
    
    print(f"scene_data: {scene_data[100].keys()}")
    scene_ids = sorted([s for s in scene_range if scene_data[s]])
    n_scenes = len(scene_ids)
    n_methods = len(methods)
    
    if n_scenes == 0:
        print("No data found for scenes 100-189")
        return
    
    x = np.arange(n_scenes)
    width = 0.8 / n_methods
    
    fig, ax = plt.subplots(figsize=(max(12, n_scenes * 0.3), 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)-1))
    methods_without_rayst3r = [m for m in methods if m != "rayst3r_zero_shot"]
    color_map = dict(zip(methods_without_rayst3r, colors))
    color_map["rayst3r_zero_shot"] = "red"
    
    for i, method in enumerate(methods):
        accuracies = [scene_data[scene_id].get(method, 0) for scene_id in scene_ids]
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, accuracies, width, label=map_method_names[method], color=color_map[method])
    
    ax.set_xlabel("Scene ID", fontsize=12)
    ax.set_ylabel("Average Precision", fontsize=12)
    ax.set_title("Average Precision Comparison Across All Test Scenes (Last Mapping Step)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scene_ids, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = os.path.join(method_results_path, "last_frame_accuracy_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    plt.close()


def main():
    graspnet_path = get_graspnet_data_path()
    method_results_path = os.path.join(graspnet_path, "output", "GraspNet")
    results = dict()
    for method in os.listdir(method_results_path):
        if method not in methods or method == "rayst3r_zero_shot":
            continue
        lowest_increase = (None, np.inf)
        highest_increase = (None, -np.inf)
        lowest_first_frame = (None, np.inf)
        highest_first_frame = (None, -np.inf)
        lowest_last_frame = (None, np.inf)
        highest_last_frame = (None, -np.inf)
        for scene in os.listdir(os.path.join(method_results_path, method)):
            scene_path = os.path.join(method_results_path, method, scene)
            results_files = sorted(glob.glob(os.path.join(scene_path, "eval_out", "results_*.json")))
            accuracies = []
            for i, result_file in enumerate(results_files):
                with open(result_file, "r") as f:
                    data = json.load(f)
                    mean_accuracy = data["mean_accuracy"]
                    accuracies.append(mean_accuracy)
                    if i == 0:
                        if mean_accuracy > highest_first_frame[1]:
                            highest_first_frame = (scene, mean_accuracy)
                        if mean_accuracy < lowest_first_frame[1]:
                            lowest_first_frame = (scene, mean_accuracy)
                    if i == len(results_files) - 1:
                        if mean_accuracy > highest_last_frame[1]:
                            highest_last_frame = (scene, mean_accuracy)
                        if mean_accuracy < lowest_last_frame[1]:
                            lowest_last_frame = (scene, mean_accuracy)
            increase = accuracies[-1] - accuracies[0]
            if increase > highest_increase[1]:
                highest_increase = (scene, increase)
            if increase < lowest_increase[1]:
                lowest_increase = (scene, increase)
        
        results[method] = {
            "highest_increase": highest_increase,
            "lowest_increase": lowest_increase,
            "highest_first_frame": highest_first_frame,
            "lowest_first_frame": lowest_first_frame,
            "highest_last_frame": highest_last_frame,
            "lowest_last_frame": lowest_last_frame,
        }
    
    # Summarize the results
    for method, method_results in results.items():
        print(f"{method}:")
        print(f"  Highest increase: {method_results['highest_increase'][1]} - {method_results['highest_increase'][0]}")
        print(f"  Lowest increase: {method_results['lowest_increase'][1]} - {method_results['lowest_increase'][0]}")
        print(f"  Highest first frame: {method_results['highest_first_frame'][1]} - {method_results['highest_first_frame'][0]}")
        print(f"  Lowest first frame: {method_results['lowest_first_frame'][1]} - {method_results['lowest_first_frame'][0]}")
        print(f"  Highest last frame: {method_results['highest_last_frame'][1]} - {method_results['highest_last_frame'][0]}")
        print(f"  Lowest last frame: {method_results['lowest_last_frame'][1]} - {method_results['lowest_last_frame'][0]}")
    
    create_last_frame_comparison_chart(method_results_path, methods)
    create_frame_increase_comparison_chart(method_results_path, methods, mode='relative')
    create_frame_increase_comparison_chart(method_results_path, methods, mode='absolute')


if __name__ == "__main__":
    main()