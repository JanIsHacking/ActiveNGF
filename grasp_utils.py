import os
import sys
import torch
import numpy as np
import open3d as o3d

# Add Generalizing-Grasp to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Generalizing-Grasp'))

from graspnetAPI import GraspGroup
from models.graspnet_sparseconv import GraspNet_MSCQ, pred_decode
from mink_dataset import GraspNetDataset_fusion, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

from torch.utils.data import DataLoader


def load_model(checkpoint_path: str, num_view: int = 300, device: str = "cuda:0"):
    net = GraspNet_MSCQ(
        input_feature_dim=0,
        num_view=num_view,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.08,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()
    return net


def generate_grasps_for_scene(
    dataset_root: str,
    scene_id: str,
    camera: str,
    checkpoint_path: str,
    candidate_points: np.ndarray,
    voxel_size: float = 0.01,
    collision_thresh: float = 0.01,
    device: str = "cuda:0"
):
    """
    Generate grasps from candidate points given a GraspNet scene.

    Args:
        dataset_root: Path to GraspNet dataset root
        scene_id: Scene identifier (e.g. 'scene_0000')
        camera: 'realsense' or 'kinect'
        checkpoint_path: Path to trained checkpoint
        candidate_points: Nx3 array of candidate grasp positions
        voxel_size: voxel size for collision detection
        collision_thresh: threshold for collision filtering
    Returns:
        GraspGroup containing filtered grasps
    """
    # Load dataset
    dataset = GraspNetDataset_fusion(
        dataset_root, 
        valid_obj_idxs=None, 
        grasp_labels=None,
        split='test', 
        camera=camera,
        num_points=20000, 
        remove_outlier=True, 
        augment=False, 
        load_label=False,
        use_fine=False
    )
    scene_idx = dataset.scene_list().index(scene_id)
    data, _ = dataset.get_data(scene_idx, return_raw_cloud=True)

    # Wrap into DataLoader format
    loader = DataLoader([dataset[scene_idx]], 
                        batch_size=1, 
                        shuffle=False,
                        collate_fn=minkowski_collate_fn)

    # Load model
    net = load_model(checkpoint_path, device=device)

    # Forward pass
    for batch_data in loader:
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds, _ = pred_decode(end_points)

    # Convert predictions to GraspGroup
    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)

    # (Optional) Filter grasps by distance to candidate points
    if candidate_points is not None:
        translations = gg.translations
        mask = np.min(np.linalg.norm(translations[:, None, :] - candidate_points[None, :, :], axis=-1), axis=1) < 0.03
        gg = gg[mask]

    # Collision detection
    cloud, _ = dataset.get_data(scene_idx, return_raw_cloud=True)
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


if __name__ == "__main__":
    dataset_root = "/path/to/graspnet"
    checkpoint_path = "/workspace/ckpts2/checkpoint.tar"
    scene_id = "scene_0000"
    camera = "realsense"

    # Example candidate points (3 points in space)
    candidate_points = np.array([
        [0.1, 0.2, 0.3],
        [0.05, -0.1, 0.2],
        [-0.05, 0.0, 0.15]
    ])

    gg = generate_grasps_for_scene(dataset_root, scene_id, camera, checkpoint_path, candidate_points)
    print("Generated", len(gg), "grasps near candidate points.")
    gg.save_npy("grasps_scene0000.npy")
