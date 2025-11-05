import json
import os
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

base_path = "/data/graspnet/output/GraspNet/"
mapping_depth_sources = ["baseline", "gt"]
camera = 'realsense'

# Scene splits according to GraspNet test categories
splits = {
    "seen": range(100, 130),
    "similar": range(130, 160),
    "novel": range(160, 190),
}


def process_depth_source(base_path: str, mapping_depth_source_str: str, scene_range: range, random_nbv: bool = False):
    """Aggregate mean and std accuracy per frame index for a given depth source and scene range."""
    frame_acc_values: dict[int, list[float]] = defaultdict(list)

    for scene_idx in scene_range:
        scene_folder = os.path.join(base_path, mapping_depth_source_str, f"scene_{scene_idx:04d}_nbv", "eval_out_corrected_ap")
        if not os.path.exists(scene_folder):
            continue
        results_files = glob.glob(os.path.join(scene_folder, "results_*.json"))

        for result_file in results_files:
            frame_index = int(result_file.split("_")[-1].split(".")[0])
            with open(result_file, "r") as f:
                data = json.load(f)
                mean_acc = data["mean_accuracy"]
                frame_acc_values[frame_index].append(mean_acc)

    avg_mean_acc: dict[int, float] = {}
    std_mean_acc: dict[int, float] = {}
    for frame_index in sorted(frame_acc_values.keys()):
        values = frame_acc_values[frame_index]
        if len(values) > 0:
            avg_mean_acc[frame_index] = np.mean(values)
            std_mean_acc[frame_index] = np.std(values)
            print(f"[{mapping_depth_source_str}] Frame {frame_index}: {len(values)} scenes aggregated")

    return avg_mean_acc, std_mean_acc


# Main aggregation and plotting
map_method_names = {
    "baseline": "Baseline",
    "baseline_random_nbv": "Random NBV",
    "gt": "Ground Truth Oracle",
    "eval_out_with_table_and_graspness": "RaySt3R",
    "eval_out_with_table_and_graspness_corrected_ap": "RaySt3R",
}

all_splits_data = {}

for split_name, scene_range in splits.items():
    split_data = {}

    # Process each depth source
    for mapping_depth_source in mapping_depth_sources:
        for random_nbv in [True, False]:
            if random_nbv and mapping_depth_source in ["gt"]:
                continue
            mapping_depth_source_str = mapping_depth_source + "_random_nbv" if random_nbv else mapping_depth_source
            avg_mean_acc, std_mean_acc = process_depth_source(base_path, mapping_depth_source_str, scene_range, random_nbv)
            split_data[mapping_depth_source_str] = (avg_mean_acc, std_mean_acc)

    # Plot all sources on one figure
    plt.figure(figsize=(10, 6))
    min_value = float('inf')
    max_value = float('-inf')

    # Sort the values in the split data in a custom order
    new_order = [
        "baseline",
        "gt",
        "baseline_random_nbv",
    ]
    split_data = {k: split_data[k] for k in new_order}

    # Plot per-frame curves with shaded ±std
    for mapping_depth_source_str, (avg_mean_acc, std_mean_acc) in split_data.items():
        x = list(avg_mean_acc.keys())
        y = list(avg_mean_acc.values())
        y_std = [std_mean_acc[i] for i in x]

        line = plt.plot(
            x, y, marker="o", label=map_method_names[mapping_depth_source_str]
        )[0]
        color = line.get_color()

        plt.fill_between(
            x,
            np.array(y) - np.array(y_std),
            np.array(y) + np.array(y_std),
            color=color,
            alpha=0.2
        )

        if len(y) > 0:
            min_value = min(min_value, min(np.array(y) - np.array(y_std)))
            max_value = max(max_value, max(np.array(y) + np.array(y_std)))

    # --- RaySt3R evaluation ---
    rayst3r_dir = os.path.join(base_path, "rayst3r_zero_shot")
    eval_dirs = [
        "eval_out_with_table_and_graspness_corrected_ap",
    ]
    colors = {
        "eval_out": "blue",
        "eval_out_with_table_and_graspness": "green",
        "eval_out_with_table_and_graspness_corrected_ap": "red",
    }
    rayst3r_acc = {eval_dir: [] for eval_dir in eval_dirs}
    for scene_id in scene_range:
        scene_dir = os.path.join(rayst3r_dir, f"{camera}_scene_{scene_id:04d}_0000")
        for eval_dir in eval_dirs:
            eval_out_dir = os.path.join(scene_dir, eval_dir)
            results_file = os.path.join(eval_out_dir, "results.json")
            try:
                with open(results_file, "r") as f:
                    data = json.load(f)
                    rayst3r_acc[eval_dir].append(data["mean_accuracy"])
            except:
                print(f"No results file found for {eval_dir} in {scene_dir}")
                continue
    
    rayst3r_results = {}
    for eval_dir, acc in rayst3r_acc.items():
        if not acc:
            continue
        avg_acc = np.mean(acc)
        std_acc = np.std(acc)
        print(f"[{eval_dir}] Average Precision: {avg_acc} ± {std_acc:.4f}")
        rayst3r_results[eval_dir] = (avg_acc, std_acc)
        min_value = min(min_value, avg_acc - std_acc)
        max_value = max(max_value, avg_acc + std_acc)

        color = colors[eval_dir]
        plt.axhline(
            y=avg_acc,
            color=color,
            linestyle="--",
            label=f"{map_method_names[eval_dir]}"
        )
        # Add shaded ±std region for RaySt3R
        plt.fill_between(
            [min(x) if len(x) else 0, max(x) if len(x) else 1],
            [avg_acc - std_acc, avg_acc - std_acc],
            [avg_acc + std_acc, avg_acc + std_acc],
            color=color,
            alpha=0.15
        )

    plt.xlabel("Mapping Step")
    plt.ylabel("Mean Average Precision")
    plt.title(f"Mean Average Precision vs Mapping Step ({split_name.capitalize()} Scenes)")
    plt.ylim(min_value - 0.02, max_value + 0.02)
    plt.grid(True)
    plt.legend(loc="upper left")

    os.makedirs("/workspace/jans_scripts/data", exist_ok=True)
    out_path = f"/workspace/jans_scripts/data/avg_mean_acc_{split_name}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot for {split_name} scenes → {out_path}")

    print(f"\n{'='*60}")
    print(f"Statistics for {split_name.upper()} scenes")
    print(f"{'='*60}")

    baseline_data = split_data.get("baseline")
    split_results = {}
    
    if baseline_data:
        baseline_avg, baseline_std = baseline_data
        baseline_step_0 = baseline_avg.get(0)
        baseline_step_5 = baseline_avg.get(5)
        baseline_step_10 = baseline_avg.get(10)
        baseline_std_0 = baseline_std.get(0, 0)
        baseline_std_5 = baseline_std.get(5, 0)
        baseline_std_10 = baseline_std.get(10, 0)

        split_results["baseline"] = {
            0: (baseline_step_0, baseline_std_0) if baseline_step_0 is not None else None,
            5: (baseline_step_5, baseline_std_5) if baseline_step_5 is not None else None,
            10: (baseline_step_10, baseline_std_10) if baseline_step_10 is not None else None,
        }

        if baseline_step_0 is not None:
            cv_0 = (baseline_std_0 / baseline_step_0 * 100) if baseline_step_0 > 0 else 0
            print(f"\nBaseline - Step 0: CV = {cv_0:.2f}% (std: {baseline_std_0:.4f}, mean: {baseline_step_0:.4f})")
        if baseline_step_5 is not None:
            cv_5 = (baseline_std_5 / baseline_step_5 * 100) if baseline_step_5 > 0 else 0
            print(f"Baseline - Step 5: CV = {cv_5:.2f}% (std: {baseline_std_5:.4f}, mean: {baseline_step_5:.4f})")
        if baseline_step_10 is not None:
            cv_10 = (baseline_std_10 / baseline_step_10 * 100) if baseline_step_10 > 0 else 0
            print(f"Baseline - Step 10: CV = {cv_10:.2f}% (std: {baseline_std_10:.4f}, mean: {baseline_step_10:.4f})")

        if baseline_step_10 is not None:
            print(f"\n--- Differences at Step 10 (final step) vs Baseline ---")
            print(f"Baseline (Step 10): {baseline_step_10:.4f} ± {baseline_std_10:.4f}")

            gt_data = split_data.get("gt")
            if gt_data:
                gt_avg, gt_std = gt_data
                gt_step_0 = gt_avg.get(0)
                gt_step_5 = gt_avg.get(5)
                gt_step_10 = gt_avg.get(10)
                gt_std_0 = gt_std.get(0, 0)
                gt_std_5 = gt_std.get(5, 0)
                gt_std_10 = gt_std.get(10, 0)
                
                split_results["gt"] = {
                    0: (gt_step_0, gt_std_0) if gt_step_0 is not None else None,
                    5: (gt_step_5, gt_std_5) if gt_step_5 is not None else None,
                    10: (gt_step_10, gt_std_10) if gt_step_10 is not None else None,
                }
                
                if gt_step_10 is not None:
                    diff = gt_step_10 - baseline_step_10
                    diff_pct = (diff / baseline_step_10 * 100) if baseline_step_10 > 0 else 0
                    print(f"Ground Truth Oracle (Step 10): {gt_step_10:.4f} ± {gt_std_10:.4f}")
                    print(f"  → Difference: {diff:+.4f} ({diff_pct:+.2f}%)")

            random_nbv_data = split_data.get("baseline_random_nbv")
            if random_nbv_data:
                random_avg, random_std = random_nbv_data
                random_step_0 = random_avg.get(0)
                random_step_5 = random_avg.get(5)
                random_step_10 = random_avg.get(10)
                random_std_0 = random_std.get(0, 0)
                random_std_5 = random_std.get(5, 0)
                random_std_10 = random_std.get(10, 0)
                
                split_results["baseline_random_nbv"] = {
                    0: (random_step_0, random_std_0) if random_step_0 is not None else None,
                    5: (random_step_5, random_std_5) if random_step_5 is not None else None,
                    10: (random_step_10, random_std_10) if random_step_10 is not None else None,
                }
                
                if random_step_10 is not None:
                    diff = random_step_10 - baseline_step_10
                    diff_pct = (diff / baseline_step_10 * 100) if baseline_step_10 > 0 else 0
                    print(f"Random NBV (Step 10): {random_step_10:.4f} ± {random_std_10:.4f}")
                    print(f"  → Difference: {diff:+.4f} ({diff_pct:+.2f}%)")

            for eval_dir, (avg_acc, std_acc) in rayst3r_results.items():
                split_results["rayst3r"] = {
                    0: None,
                    5: None,
                    10: (avg_acc, std_acc),
                }
                diff = avg_acc - baseline_step_10
                diff_pct = (diff / baseline_step_10 * 100) if baseline_step_10 > 0 else 0
                print(f"RaySt3R: {avg_acc:.4f} ± {std_acc:.4f}")
                print(f"  → Difference: {diff:+.4f} ({diff_pct:+.2f}%)")

            print(f"\n--- Coefficient of Variation (CV) at Steps 0, 5 and 10 ---")
            for method_name, (avg_mean_acc, std_mean_acc) in split_data.items():
                step_0_mean = avg_mean_acc.get(0)
                step_0_std = std_mean_acc.get(0, 0)
                step_5_mean = avg_mean_acc.get(5)
                step_5_std = std_mean_acc.get(5, 0)
                step_10_mean = avg_mean_acc.get(10)
                step_10_std = std_mean_acc.get(10, 0)
                
                if step_0_mean is not None:
                    cv_0 = (step_0_std / step_0_mean * 100) if step_0_mean > 0 else 0
                    print(f"{map_method_names.get(method_name, method_name)} - Step 0: CV = {cv_0:.2f}%")
                
                if step_5_mean is not None:
                    cv_5 = (step_5_std / step_5_mean * 100) if step_5_mean > 0 else 0
                    print(f"{map_method_names.get(method_name, method_name)} - Step 5: CV = {cv_5:.2f}%")
                
                if step_10_mean is not None:
                    cv_10 = (step_10_std / step_10_mean * 100) if step_10_mean > 0 else 0
                    print(f"{map_method_names.get(method_name, method_name)} - Step 10: CV = {cv_10:.2f}%")

            for eval_dir, (avg_acc, std_acc) in rayst3r_results.items():
                cv = (std_acc / avg_acc * 100) if avg_acc > 0 else 0
                print(f"{map_method_names.get(eval_dir, eval_dir)}: CV = {cv:.2f}%")

    all_splits_data[split_name] = split_results
    print(f"{'='*60}\n")


def format_latex_cell(mean: float, std: float, baseline_mean: float = None, is_baseline: bool = False) -> str:
    if mean is None:
        return "---"
    mean_str = f"{mean:.3f}"
    std_str = f"{std:.3f}"
    result = f"{mean_str}\\tsb{{{std_str}}}"
    
    if is_baseline:
        result += " (0.0\\%)"
    elif baseline_mean is not None and baseline_mean > 0:
        diff_pct = ((mean - baseline_mean) / baseline_mean * 100)
        result += f" ({diff_pct:+.1f}\\%)"
    else:
        result += " (---)"
    
    return result


print("\n" + "="*80)
print("Generating LaTeX Table")
print("="*80)
print("Note: This table requires \\usepackage{multirow} in your LaTeX document preamble.")
print()

method_order = ["baseline", "gt", "baseline_random_nbv", "rayst3r"]
method_display_names = {
    "baseline": "Baseline",
    "gt": "Ground Truth Oracle",
    "baseline_random_nbv": "Random NBV",
    "rayst3r": "RaySt3R",
}

latex_table = """\\begin{table}[t]
    \\caption{Mean Average Precision at mapping steps 0, 5, and 10 for the different methods across scene categories. Values are shown with standard deviations as subscripts. Relative differences to the baseline method are shown in parentheses.}
    \\label{tab:map_comparison}
    \\centering
    \\tiny
    \\newcommand{\\tsb}[1]{\\textsubscript{\\textcolor{gray}{$\\pm$#1}}}
    \\begin{tabular}{llccc}
    \\toprule
    Scene & Method & Step 0 & Step 5 & Step 10 \\\\
    \\midrule
"""

for split_idx, split_name in enumerate(["seen", "similar", "novel"]):
    split_results = all_splits_data.get(split_name, {})
    
    if not split_results:
        continue
    
    baseline_data = split_results.get("baseline", {})
    baseline_step_0_data = baseline_data.get(0)
    baseline_step_5_data = baseline_data.get(5)
    baseline_step_10_data = baseline_data.get(10)
    
    baseline_mean_0 = baseline_step_0_data[0] if baseline_step_0_data else None
    baseline_mean_5 = baseline_step_5_data[0] if baseline_step_5_data else None
    baseline_mean_10 = baseline_step_10_data[0] if baseline_step_10_data else None
    
    available_methods = [method_key for method_key in method_order if method_key in split_results]
    num_methods = len(available_methods)
    
    for method_idx, method_key in enumerate(available_methods):
        method_name = method_display_names[method_key]
        
        method_data = split_results.get(method_key, {})
        step_0_data = method_data.get(0)
        step_5_data = method_data.get(5)
        step_10_data = method_data.get(10)
        
        row_parts = []
        
        if method_idx == 0:
            row_parts.append(f"\\multirow{{{num_methods}}}{{*}}{{{split_name.capitalize()}}}")
        else:
            row_parts.append("")
        
        row_parts.append(method_name)
        
        for step, step_data, baseline_mean in [
            (0, step_0_data, baseline_mean_0),
            (5, step_5_data, baseline_mean_5),
            (10, step_10_data, baseline_mean_10),
        ]:
            if step_data and step_data[0] is not None:
                mean, std = step_data
                cell_value = format_latex_cell(mean, std, baseline_mean, is_baseline=(method_key == "baseline"))
            else:
                cell_value = "---"
            
            row_parts.append(cell_value)
        
        latex_table += "    " + " & ".join(row_parts) + " \\\\\n"
    
    if split_idx < len(["seen", "similar", "novel"]) - 1:
        latex_table += "    \\midrule\n"

latex_table += """    \\bottomrule
    \\end{tabular}
\\end{table}
"""

print(latex_table)
print("\n" + "-"*80 + "\n")
