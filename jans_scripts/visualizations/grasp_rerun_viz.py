import os
import numpy as np
import rerun as rr
from jans_scripts.visualize import setup_rerun
from jans_scripts.graspnet_viz import visualize_graspnet_scene
import matplotlib.pyplot as plt

gripper_width = 0.08
scene_id = "scene_0100"
camera = "realsense"


def visualize_grasp_array(grasp_array: np.ndarray, mapping_idx: str = f"{0:05d}") -> None:
    """
    Visualize a grasp array in Rerun.
    Each grasp is colored based on its score using the 'Iris' colormap.
    """
    # Compute score range
    scores = grasp_array[:, 0]
    min_score, max_score = np.min(scores), np.max(scores)
    norm = plt.Normalize(vmin=min_score, vmax=max_score)
    cmap = plt.colormaps["viridis"]

    for grasp_idx, grasp in enumerate(grasp_array):
        score, width, height, depth = grasp[:4]
        grasp_rotation = grasp[4:13].reshape(3, 3)
        grasp_center = grasp[13:16]
        object_id = grasp[16].item()

        # Map score to color (convert from 0–1 floats to 0–255 ints)
        color = np.array(cmap(norm(score))[:3]) * 255
        color = color.astype(int).tolist()

        right_vector = [0, gripper_width / 2, 0]
        left_vector = [0, -gripper_width / 2, 0]
        forward_vector = [-gripper_width / 2, 0, 0]

        rr.log(
            f"world/grasp/{mapping_idx}/{grasp_idx}",
            rr.Transform3D(translation=grasp_center, mat3x3=grasp_rotation)
        )
        rr.log(
            f"world/grasp/{mapping_idx}/{grasp_idx}/inner_vectors",
            rr.Arrows3D(
                vectors=[right_vector, left_vector, forward_vector],
                colors=[color, color, color],
            )
        )

        # Right outer finger
        outer_vector_right = [gripper_width / 2, 0, 0]
        rr.log(f"world/grasp/{mapping_idx}/{grasp_idx}/outer_vector_right", rr.Transform3D(translation=right_vector))
        rr.log(
            f"world/grasp/{mapping_idx}/{grasp_idx}/outer_vector_right",
            rr.Arrows3D(vectors=[outer_vector_right], colors=[color]),
        )

        # Left outer finger
        outer_vector_left = [gripper_width / 2, 0, 0]
        rr.log(f"world/grasp/{mapping_idx}/{grasp_idx}/outer_vector_left", rr.Transform3D(translation=left_vector))
        rr.log(
            f"world/grasp/{mapping_idx}/{grasp_idx}/outer_vector_left",
            rr.Arrows3D(vectors=[outer_vector_left], colors=[color]),
        )


def grasp_rerun_viz() -> None:
    setup_rerun("grasp_rerun_viz")

    grasps_path = f"output/GraspNet/test/gt_random_nbv/{scene_id}_nbv/grasps"
    for mapping_idx in os.listdir(grasps_path):
        grasp_file = os.path.join(grasps_path, mapping_idx, scene_id, camera, "result.npy")
        grasp_array = np.load(grasp_file)
        visualize_grasp_array(grasp_array, mapping_idx)
    
    visualize_graspnet_scene(scene_id, camera)


if __name__ == "__main__":
    grasp_rerun_viz()
