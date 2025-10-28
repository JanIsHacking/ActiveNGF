import os

from graspnetAPI import GraspGroup
import torch

def main():
    for i in range(10):
        path = f"/workspace/output/GraspNet/test/gt_random_nbv/scene_0100_nbv/grasps/0000{i}/scene_0100/realsense/result.npy"
        if not os.path.exists(path):
            continue
        print(path)
        grasp_group = GraspGroup().from_npy(path).grasp_group_array
        print(grasp_group.shape)
    
    graspable_xyz = torch.load("/workspace/jans_scripts/data/graspable_xyz.pt")
    print(graspable_xyz.shape)

if __name__ == "__main__":
    main()