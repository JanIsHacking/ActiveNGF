import numpy as np
import rerun as rr

from jans_scripts.visualize import setup_rerun


def main():
    setup_rerun("rerun_grasps")

    grasps = "/home/jan/thesis/ActiveNGF/output/GraspNet/scene_0100_nbv/grasps/00000/scene_0100/realsense/result.npy"
    grasp_array = np.load(grasps)

    for idx, grasp in enumerate(grasp_array):
        grasp_center = grasp[1:4]
        grasp_rotation = grasp[4:7]

        rr.log(f"world/grasp_{idx}",rr.Transform3D(translation=grasp_center, mat3x3=grasp_rotation))

        rr.log(
            f"world/grasp_{idx}",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            )
        )

    


if __name__ == "__main__":
    main()