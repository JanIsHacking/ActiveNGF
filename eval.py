import os
import glob
import numpy as np
import trimesh
import torch
import json

from graspnetAPI import GraspNet, GraspGroup

#from jans_scripts.visualize import visualize_pc_with_normals
from src import config
from src.ESLAM import ESLAM
from eval_utils import GraspNetEvalComplete
from grasp_utils import generate_grasps_for_scene
from jans_scripts.geometry import furthest_point_sampling

torch.set_num_threads(8)       # for intra-op parallelism
torch.set_num_interop_threads(8)  # for inter-op parallelism

mesh_num_sample_points = 1024
fps_num_sample_points = 2048
graspness_threshold = 0.5
camera = 'realsense'
dump_folder = ''


def mesh_to_pointcloud(mesh_file: str, num_points: int = 20000) -> np.ndarray:
    """
    Sample a point cloud from a mesh.
    Returns Nx6 array: [x, y, z, nx, ny, nz].
    """
    mesh = trimesh.load(mesh_file, process=True)
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_idx]
    return np.concatenate([pts, normals], axis=1)


def synthesize_grasps(points: np.ndarray, mapper) -> GraspGroup:
    """
    Synthesize grasps on the candidate points.
    This mock implementation generates random values for testing purposes.
    """
    num_grasps = len(points)

    grasps = []
    for i in range(num_grasps):
        score = np.random.uniform(0, 1)  # grasp confidence score
        width = np.random.uniform(0.02, 0.1)  # gripper width (meters)
        height = np.random.uniform(0.01, 0.05)  # gripper height
        depth = np.random.uniform(0.01, 0.05)  # gripper depth

        # Generate a random rotation matrix using QR decomposition
        rand_mat = np.random.randn(3, 3)
        q, _ = np.linalg.qr(rand_mat)
        rotation = q.flatten()  # 9 values

        # Random translation around the candidate point
        translation = points[i] + np.random.normal(scale=0.01, size=3)

        object_id = np.random.randint(0, 100)  # mock object ID

        grasp = np.concatenate(
            [[score, width, height, depth], rotation, translation, [object_id]]
        )
        grasps.append(grasp)

    gg = GraspGroup(np.array(grasps))
    return gg


def evaluate(mesh_file: str, scene_id: int, mapper, gne: GraspNetEvalComplete, frame_idx: int, scene_dir: str, graspnet_checkpoint: str, dataset_root: str, force: bool):
    """
    Evaluate a single mesh reconstruction against GraspNet annotations.
    """
    eval_out_dir = os.path.join(scene_dir, "eval_out")
    mapping_index = mesh_file.split("/")[-1].split("_")[0]
    results_file_path = os.path.join(eval_out_dir, f"results_{mapping_index}.json")
    if os.path.exists(results_file_path) and not force:
        print(f"Results file already exists: {results_file_path}")
        return
    os.makedirs(eval_out_dir, exist_ok=True)

    scene_id_str = f"scene_{scene_id:04d}"

    # Get the graspness values of the points
    # print("Getting graspness values of the points")
    # renderer = mapper.renderer
    # planes = mapper.get_planes()
    # points_torch = torch.from_numpy(points).float().to(mapper.device)
    # graspness = renderer.render_pointcloud(planes, mapper.decoders, points_torch, mapper.device, mapper.truncation)[2]

    # Threshold the graspness values and perform furthest point sampling
    # print("Thresholding the graspness values and performing furthest point sampling")
    # graspness_mask = (graspness > graspness_threshold).cpu().numpy()
    # points = points[graspness_mask]
    # points = furthest_point_sampling(points, fps_num_sample_points)

    # Synthesize grasps on the candidate points
    dump_dir = os.path.join(scene_dir, "grasps", f"{frame_idx:05d}")
    dump_file = os.path.join(dump_dir, scene_id_str, camera, 'result.npy')
    os.makedirs(os.path.dirname(dump_file), exist_ok=True)
    print("Generating grasps for scene: ", scene_id_str)
    grasp_group = generate_grasps_for_scene(
        dataset_root=dataset_root,
        mesh_file=mesh_file,
        scene_id=scene_id_str,
        camera=camera,
        checkpoint_path=graspnet_checkpoint,
    )
    grasp_group.save_npy(dump_file)

    # Run GraspNet evaluation
    scene_accuracy = gne.eval_scene_objects(scene_id, dump_dir, dataset_root, camera, grasps_per_object=5)

    results = {
        "scene_accuracy": scene_accuracy[0].tolist(),
        "mean_accuracy": np.mean(scene_accuracy).tolist(),
    }
    with open(results_file_path, "w") as f:
        json.dump(results, f, indent=4)


def main(scene_dir: str, dataset_root: str, scene_id: int, mapper, graspnet_checkpoint: str, force: bool):
    # Initialize GraspNet evaluation
    gne = GraspNetEvalComplete(
        root=dataset_root,
        camera=camera,
        split='test',
    )

    # Iterate over mesh files
    results = {}
    mesh_dir = os.path.join(scene_dir, "mesh")
    for mesh_file in sorted(glob.glob(os.path.join(mesh_dir, "*.ply"))):
        print("Evaluating mesh file: ", mesh_file)
        # Load the mapping steps mapper
        frame_idx = int(os.path.basename(mesh_file).split("_")[0])
        mapper.load_checkpoint(os.path.join(mapper.output, 'ckpts', f'{frame_idx:05d}.tar'))

        evaluate(mesh_file, scene_id, mapper, gne, frame_idx, scene_dir, graspnet_checkpoint, dataset_root, force)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Eval arguments
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory of scene")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path of GraspNet dataset")
    parser.add_argument("--scene_id", type=int, required=True, help="Scene id to evaluate")
    parser.add_argument("--force", action="store_true", help="Force evaluation even if results file already exists")

    # ESLAM arguments
    parser.add_argument("--config", type=str, required=True, help="Path to ESLAM config file")
    parser.add_argument('--input_folder', type=str, help="input folder, this have higher priority, can overwrite the one in config file")
    parser.add_argument("--output", type=str, help="Path to ESLAM output folder")

    # GraspNet arguments
    parser.add_argument("--graspnet_checkpoint", type=str, required=True, help="Path to GraspNet checkpoint")

    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/ESLAM.yaml')
    eslam = ESLAM(cfg, args, None)
    mapper = eslam.mapper

    main(args.scene_dir, args.dataset_root, args.scene_id, mapper, args.graspnet_checkpoint, args.force)
