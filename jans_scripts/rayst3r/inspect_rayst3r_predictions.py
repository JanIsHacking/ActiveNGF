from jans_scripts.visualize import setup_rerun
import open3d as o3d
import rerun as rr
import numpy as np
import os

def main():
    setup_rerun("inspect_rayst3r_predictions")
    predictions_dir = "/data/user/jan/graspnet/output/GraspNet/rayst3r_zero_shot/realsense_scene_0100_0000/inference_points.ply"
    if not os.path.exists(predictions_dir):
        print(f"Predictions file not found: {predictions_dir}")
        return
    predictions = np.asarray(o3d.io.read_point_cloud(predictions_dir).points)

    rr.log("world/predictions", rr.Points3D(predictions, colors=[255, 0, 0]))

if __name__ == "__main__":
    main()
