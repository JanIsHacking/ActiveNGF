import os
import sys
import torch
import numpy as np
import open3d as o3d

# Add Generalizing-Grasp to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'Generalizing-Grasp'))
sys.path.append(os.path.join(BASE_DIR, 'Generalizing-Grasp/pointnet2'))
sys.path.append(os.path.join(BASE_DIR, 'Generalizing-Grasp/utils'))

from graspnetAPI import GraspGroup
# from models.graspnet_sparseconv import GraspNet_MSCQ, pred_decode
from GraspNet.model import GraspNet_MSCQ, pred_decode_reg
# from mink_dataset import minkowski_collate_fn, GraspNetDataset_fusion
from dataset.mink_dataset import GraspNetSingleSceneDataset, minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector

from torch.utils.data import DataLoader
from jans_scripts.graspnet_storage import get_scene_objects_bounding_boxes

def print_end_points_overview(end_points: dict) -> None:
    """
    Print an overview of the end_points dictionary used in GraspNet_MSCQ.
    
    Args:
        end_points: Dictionary containing input data and intermediate results for GraspNet
    """
    print("=" * 80)
    print("GRASPNET END_POINTS DICTIONARY OVERVIEW")
    print("=" * 80)
    
    # Input keys (required for model input)
    input_keys = ['point_clouds', 'coors', 'feats', 'quantize2original']
    
    # Intermediate processing keys
    intermediate_keys = [
        'input_xyz', 'seed_features', 'fp2_xyz', 'fp2_inds', 'fp2_features', 
        'fp2_graspness', 'graspness_score', 'objectness_mask', 'view_score',
        'grasp_top_view_inds', 'grasp_top_view_score', 'grasp_top_view_xyz', 
        'grasp_top_view_rot'
    ]
    
    # Output prediction keys
    output_keys = [
        'grasp_score_pred', 'grasp_angle_cls_pred', 'grasp_width_pred', 
        'grasp_tolerance_pred', 'grasp_angle_pred', 'grasp_angle_value_pred'
    ]
    
    # Training keys (optional, only present during training)
    training_keys = ['graspness_label', 'batch_grasp_view_label']
    
    print(f"\n📊 TOTAL KEYS: {len(end_points)}")
    print(f"📋 KEYS FOUND: {list(end_points.keys())}")
    
    print("\n" + "=" * 50)
    print("INPUT DATA (Required for model input)")
    print("=" * 50)
    for key in input_keys:
        if key in end_points:
            data = end_points[key]
            if hasattr(data, 'shape'):
                print(f"✅ {key:<20}: {type(data).__name__} {data.shape}")
            else:
                print(f"✅ {key:<20}: {type(data).__name__} (len={len(data) if hasattr(data, '__len__') else 'N/A'})")
        else:
            print(f"❌ {key:<20}: MISSING")
    
    print("\n" + "=" * 50)
    print("INTERMEDIATE PROCESSING DATA")
    print("=" * 50)
    for key in intermediate_keys:
        if key in end_points:
            data = end_points[key]
            if hasattr(data, 'shape'):
                print(f"✅ {key:<25}: {type(data).__name__} {data.shape}")
            else:
                print(f"✅ {key:<25}: {type(data).__name__} (len={len(data) if hasattr(data, '__len__') else 'N/A'})")
        else:
            print(f"⚪ {key:<25}: Not present (normal for input)")
    
    print("\n" + "=" * 50)
    print("OUTPUT PREDICTIONS")
    print("=" * 50)
    for key in output_keys:
        if key in end_points:
            data = end_points[key]
            if hasattr(data, 'shape'):
                print(f"✅ {key:<25}: {type(data).__name__} {data.shape}")
            else:
                print(f"✅ {key:<25}: {type(data).__name__} (len={len(data) if hasattr(data, '__len__') else 'N/A'})")
        else:
            print(f"⚪ {key:<25}: Not present (normal for input)")
    
    print("\n" + "=" * 50)
    print("TRAINING DATA (Optional)")
    print("=" * 50)
    for key in training_keys:
        if key in end_points:
            data = end_points[key]
            if hasattr(data, 'shape'):
                print(f"✅ {key:<25}: {type(data).__name__} {data.shape}")
            else:
                print(f"✅ {key:<25}: {type(data).__name__} (len={len(data) if hasattr(data, '__len__') else 'N/A'})")
        else:
            print(f"⚪ {key:<25}: Not present (normal for inference)")
    
    # Identify additional / unexpected keys
    all_known_keys = set(input_keys + intermediate_keys + output_keys + training_keys)
    additional_keys = set(end_points.keys()) - all_known_keys
    
    if additional_keys:
        print("\n" + "=" * 50)
        print("UNEXPECTED / UNKNOWN KEYS")
        print("=" * 50)
        for key in sorted(additional_keys):
            data = end_points[key]
            # Describe based on type and content
            if hasattr(data, 'shape'):
                print(f"🔍 {key:<25}: {type(data).__name__} {data.shape}")
            elif isinstance(data, (list, tuple, set)):
                print(f"🔍 {key:<25}: {type(data).__name__} (len={len(data)})")
            elif isinstance(data, dict):
                print(f"🔍 {key:<25}: dict (keys={list(data.keys())[:5]}{'...' if len(data) > 5 else ''})")
            elif isinstance(data, (int, float, str, bool)):
                print(f"🔍 {key:<25}: {type(data).__name__} value={data}")
            else:
                print(f"🔍 {key:<25}: {type(data).__name__}")
    
    print("\n" + "=" * 50)
    print("EXPECTED SHAPES (for reference)")
    print("=" * 50)
    print("📝 Input shapes (batch_size = B, num_points = N):")
    print("   • point_clouds: (B, N, 3) - Point cloud coordinates")
    print("   • coors: (N_total, 4) - Sparse tensor coordinates [batch_idx, x, y, z]")
    print("   • feats: (N_total, 3) - Point features (usually RGB or XYZ)")
    print("   • quantize2original: (N_total,) - Mapping from quantized to original points")
    
    print("\n📝 Intermediate shapes:")
    print("   • seed_features: (B, 256, N) - Point features from backbone")
    print("   • fp2_xyz: (B, num_seed, 3) - Selected seed point coordinates")
    print("   • fp2_features: (B, 256, num_seed) - Features of selected points")
    print("   • grasp_top_view_xyz: (B, num_seed, 3) - Approach directions")
    print("   • grasp_top_view_rot: (B, num_seed, 3, 3) - Rotation matrices")
    
    print("\n📝 Output shapes:")
    print("   • grasp_score_pred: (B, num_seed, num_depth) - Grasp quality scores")
    print("   • grasp_width_pred: (B, num_seed, num_depth) - Gripper width predictions")
    print("   • grasp_tolerance_pred: (B, num_seed, num_depth) - Grasp tolerance")
    print("   • grasp_angle_value_pred: (B, num_seed, num_depth) - In-plane rotation angles")
    
    print("\n" + "=" * 80)


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
    mesh_file: str,
    scene_id: str,
    camera: str,
    checkpoint_path: str,
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
        voxel_size: voxel size for collision detection
        collision_thresh: threshold for collision filtering
    Returns:
        GraspGroup containing filtered grasps
    """
    # Load dataset
    dataset = GraspNetSingleSceneDataset(
        dataset_root,
        mesh_file,
        scene_id,
        valid_obj_idxs=None, 
        grasp_labels=None,
        split='test', 
        camera=camera,
        num_points=20000, 
        remove_outlier=True, 
    )

    # Wrap into DataLoader format
    loader = DataLoader([dataset[0]],
                        batch_size=1, 
                        shuffle=False,
                        collate_fn=minkowski_collate_fn)

    # Load model
    net = load_model(checkpoint_path, device=device)

    # Get the object bounding boxes
    bounding_boxes, object_ids = get_scene_objects_bounding_boxes(scene_id, camera)

    # Forward pass
    for batch_data in loader:
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        batch_data['object_bounding_boxes'] = bounding_boxes
        batch_data['object_ids'] = object_ids
        
        with torch.no_grad():
            # print_end_points_overview(batch_data)
            # torch.save(batch_data, "jans_scripts/data/batch_data.pt")
            end_points = net(batch_data)
            grasp_preds = pred_decode_reg(end_points)

    # Convert predictions to GraspGroup
    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds)

    # Collision detection
    cloud, _ = dataset.get_data(0, return_raw_cloud=True)
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


if __name__ == "__main__":
    dataset_root = "/path/to/graspnet"
    checkpoint_path = "/workspace/ckpts2/checkpoint.tar"
    scene_id = "scene_0000"
    camera = "realsense"

    gg = generate_grasps_for_scene(dataset_root, scene_id, camera, checkpoint_path)
    print("Generated", len(gg), "grasps near candidate points.")
    gg.save_npy("grasps_scene0000.npy")
