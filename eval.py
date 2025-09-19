import os
import glob
import numpy as np
import trimesh
import torch

from graspnetAPI import GraspNet

from src import config
from src.ESLAM import ESLAM
from utils import GraspNetEvalComplete

mesh_num_sample_points = 1024
fps_num_sample_points = 128
graspness_threshold = 0.5
camera = 'realsense'
dump_folder = ''


def furthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Perform Furthest Point Sampling (FPS) on a set of points.

    Args:
        points (np.ndarray): Array of shape (N, D), where N is the number of points
                             and D is the dimensionality (e.g., 3 for 3D points).
        num_samples (int): Number of points to sample.

    Returns:
        np.ndarray: Subset of points of shape (num_samples, D).
    """
    N, D = points.shape
    sampled_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.ones(N) * np.inf

    # Pick a random seed point
    seed_idx = np.random.randint(0, N)
    sampled_indices[0] = seed_idx

    for i in range(1, num_samples):
        # Update distances to the set of chosen points
        last_sampled = points[sampled_indices[i - 1]]
        dist = np.linalg.norm(points - last_sampled, axis=1)
        distances = np.minimum(distances, dist)

        # Pick the farthest point from current set
        sampled_indices[i] = np.argmax(distances)

    return points[sampled_indices]


def mesh_to_pointcloud(mesh_file: str, num_points: int = 20000) -> np.ndarray:
    """
    Sample a point cloud from a mesh.
    Returns Nx6 array: [x, y, z, nx, ny, nz].
    """
    mesh = trimesh.load(mesh_file, process=True)
    pts, face_idx = trimesh.sample.sample_surface(mesh, num_points)
    normals = mesh.face_normals[face_idx]
    return np.concatenate([pts, normals], axis=1)


def synthesize_grasps(points: np.ndarray, mapper):
    """
    Synthesize grasps on the candidate points.
    """
    # TODO: Implement
    return np.eye(4)[None, ...]


def evaluate(mesh_file: str, scene_id: int, mapper, gne: GraspNetEvalComplete, frame_idx: int, scene_dir: str):
    """
    Evaluate a single mesh reconstruction against GraspNet annotations.
    """
    # Read the mesh and sample points
    mesh = trimesh.load(mesh_file, process=True)
    if isinstance(mesh, trimesh.Scene):
        for scene_mesh in mesh.geometry.values():
            scene_meshes.append(scene_mesh)
        mesh = trimesh.util.concatenate(scene_meshes)
    points = trimesh.sample.sample_surface(mesh, mesh_num_sample_points)[0]

    # Get the graspness values of the points
    renderer = mapper.renderer
    planes = mapper.get_planes()
    points_torch = torch.from_numpy(points).float().to(mapper.device)
    graspness = renderer.render_pointcloud(planes, mapper.decoders, points_torch, mapper.device, mapper.truncation)[2]

    # Threshold the graspness values and perform furthest point sampling
    graspness_mask = (graspness > graspness_threshold).cpu().numpy()
    points = points[graspness_mask]
    points = furthest_point_sampling(points, fps_num_sample_points)

    # Synthesize grasps on the candidate points
    dump_dir = os.path.join(scene_dir, "grasps", f"{frame_idx:05d}")
    dump_file = os.path.join(dump_dir, scene_id, camera, 'result.npy')
    os.makedirs(os.path.dirname(dump_file), exist_ok=True)
    cand_grasps = synthesize_grasps(points, mapper)
    np.save(dump_file, cand_grasps)

    # Run GraspNet evaluation
    gne.eval_scene(scene_id, dump_dir)

    return 0


def main(scene_dir: str, dataset_root: str, scene_id: int, mapper):
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
        # Load the mapping steps mapper
        frame_idx = int(os.path.basename(mesh_file).split("_")[0])
        mapper.load_checkpoint(os.path.join(mapper.output, 'ckpts', f'{frame_idx:05d}.tar'))

        evaluate(mesh_file, scene_id, mapper, gne, frame_idx, scene_dir)

    # Save results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Eval arguments
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory of scene")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root path of GraspNet dataset")
    parser.add_argument("--scene_id", type=int, required=True, help="Scene id to evaluate")

    # ESLAM arguments
    parser.add_argument("--config", type=str, required=True, help="Path to ESLAM config file")
    parser.add_argument('--input_folder', type=str, help="input folder, this have higher priority, can overwrite the one in config file")
    parser.add_argument("--output", type=str, help="Path to ESLAM output folder")
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/ESLAM.yaml')
    eslam = ESLAM(cfg, args, None)
    mapper = eslam.mapper

    main(args.scene_dir, args.dataset_root, args.scene_id, mapper)
