import os
import random
import matplotlib.pyplot as plt
import torch


def visualize_random_depths(data_dir: str, output_dir: str, num_samples: int = 16) -> None:
    """
    Randomly selects `num_samples` depth map pairs (gt_depth_{idx}.pt and render_depth_{idx}.pt)
    from `data_dir`, arranges them into 4x4 grids (4 per row, 4 rows total), and saves 4 PNG plots
    (each showing 4 comparisons side-by-side).

    Args:
        data_dir (str): Directory containing .pt depth map files.
        output_dir (str): Directory to save PNG visualizations.
        num_samples (int): Number of random pairs to visualize (default: 16).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all gt_depth indices available
    gt_files = [f for f in os.listdir(data_dir) if f.startswith("gt_depth_") and f.endswith(".pt")]
    indices = [f.split("_")[-1].split(".")[0] for f in gt_files]

    # Randomly sample indices
    sampled_indices = random.sample(indices, num_samples)

    for plot_idx in range(4):
        fig, axes = plt.subplots(4, 2, figsize=(10, 20))
        fig.suptitle(f"Depth Map Comparison - Set {plot_idx + 1}", fontsize=16)

        # Each plot has 4 samples
        subset = sampled_indices[plot_idx * 4:(plot_idx + 1) * 4]
        for i, idx in enumerate(subset):
            gt_path = os.path.join(data_dir, f"gt_depth_{idx}.pt")
            render_path = os.path.join(data_dir, f"render_depth_{idx}.pt")

            gt = torch.load(gt_path).squeeze().cpu().numpy()
            render = torch.load(render_path).squeeze().cpu().numpy()

            print(f"gt_path: {gt_path}, render_path: {render_path}, idx: {idx}")
            print(gt.shape, render.shape)
            print(gt.min(), gt.max(), render.min(), render.max())
            print(gt.sum(), render.sum())
            print(gt.mean(), render.mean())

            all_min = min(gt.min(), render.min())
            all_max = max(gt.max(), render.max())

            # Plot GT
            ax_gt = axes[i, 0]
            im_gt = ax_gt.imshow(gt, cmap='viridis', vmin=all_min, vmax=all_max)
            ax_gt.set_title(f"GT Depth {idx}")
            ax_gt.axis("off")
            fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

            # Plot Rendered
            ax_render = axes[i, 1]
            im_render = ax_render.imshow(render, cmap='viridis', vmin=all_min, vmax=all_max)
            ax_render.set_title(f"Render Depth {idx}")
            ax_render.axis("off")
            fig.colorbar(im_render, ax=ax_render, fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(output_dir, f"depth_comparison_{plot_idx + 1}.png")
        plt.savefig(save_path)
        plt.close(fig)


def depth_test_viz():
    test_dir = "/workspace/output/GraspNet/test/gt/scene_0100_nbv"
    output_dir = "/workspace/output/GraspNet/test/gt/scene_0100_nbv/depth_test_viz"
    visualize_random_depths(test_dir, output_dir, num_samples=16)


if __name__ == "__main__":
    depth_test_viz()