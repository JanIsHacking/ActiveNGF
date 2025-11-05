import torch
import rerun as rr
import numpy as np
from sklearn.decomposition import PCA
from jans_scripts.geometry import compute_pointmap
import os
import open3d as o3d

pointmap_colors = [
    [0, 0, 255],
    [0, 255, 0],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 255],
    [128, 128, 128],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 128, 0],
    [0, 128, 128],
    [128, 0, 128],
    [128, 128, 128],
]

def visualize_mesh(mesh: o3d.geometry.TriangleMesh, suffix: str = "mesh"):
    rr.log(
        f"world/mesh/{suffix}",
        rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices),
            triangle_indices=np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals),
            vertex_colors=np.asarray(mesh.vertex_colors),
        ),
    )

def visualize_pc_with_normals(
    points: torch.Tensor,
    normals: torch.Tensor,
    colors: torch.Tensor = None,
    num_points: int = 10000,
    suffix: str = "normals",
    normal_length: float = None
):
    """
    Visualizes a point cloud along with per-point normal vectors in Rerun.

    Args:
        points: (N, 3) torch.Tensor of 3D point positions
        normals: (N, 3) torch.Tensor of corresponding normal vectors
        colors: (N, 3) torch.Tensor of RGB colors in range [0, 255] (optional)
        num_points: number of points to visualize (randomly sampled if larger)
        suffix: identifier suffix for the Rerun log path
        normal_length: desired length for normal vectors (optional, keeps original length if None)
    """
    # Move tensors to CPU and numpy
    points_np = points.cpu().numpy()
    normals_np = normals.cpu().numpy()
    
    # Handle colors
    if colors is not None:
        colors_np = colors.cpu().numpy()
    else:
        colors_np = np.tile(np.array([[0, 255, 255]]), (points_np.shape[0], 1))  # cyan by default

    # Random subsample to avoid overloading the viewer
    num_points = min(num_points, points_np.shape[0])
    indices = np.random.choice(points_np.shape[0], size=num_points, replace=False)

    sampled_points = points_np[indices]
    sampled_normals = normals_np[indices]
    sampled_colors = colors_np[indices]

    # Normalize normal vectors to desired length if specified
    if normal_length is not None:
        # Compute current lengths
        current_lengths = np.linalg.norm(sampled_normals, axis=1, keepdims=True)
        # Avoid division by zero
        current_lengths = np.where(current_lengths == 0, 1, current_lengths)
        # Scale to desired length
        sampled_normals = sampled_normals * (normal_length / current_lengths)

    # Log point cloud
    rr.log(f"world/pc_with_normals/{suffix}/points", rr.Points3D(sampled_points, colors=sampled_colors))

    # Log normals as vectors (arrows) originating from each point
    rr.log(
        f"world/pc_with_normals/{suffix}/normals",
        rr.Arrows3D(
            origins=sampled_points,
            vectors=sampled_normals,
            colors=[[255, 255, 0]]  # yellow arrows by default
        )
    )

def visualize_pc(pc: torch.Tensor, colors: torch.Tensor = None, num_points: int = 10000, suffix: str = "base"):
    pc = pc.cpu().numpy()
    if colors is not None:
        colors = colors.cpu().numpy()
    else:
        colors = [255, 0, 0]
    pc = pc[np.random.choice(pc.shape[0], size=num_points, replace=False)]
    rr.log(f"world/pc/{suffix}", rr.Points3D(pc, colors=colors))

def visualize_gt_points(gt_points: torch.Tensor, num_points: int = 10000, suffix: str = "base"):
    gt_points = gt_points.cpu().numpy()
    rr.log(f"world/gt_points/{suffix}", rr.Points3D(gt_points[np.random.choice(gt_points.shape[0], size=num_points, replace=False)], colors=[255, 0, 0]))

def setup_rerun(name: str, show_origin: bool = True):
    rr.init(name)
    rr.connect_tcp(os.environ.get('RERUN_REMOTE_IP', '0.0.0.0')+":"+os.environ.get('RERUN_REMOTE_PORT', '9800'))
    rr.log("world",rr.Transform3D(translation=[0,0,0], mat3x3=np.eye(3)))

    # show origin axis system
    if show_origin:
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            )
        )


def visualize_camera(
        rgb: torch.Tensor, 
        dino_features: torch.Tensor,
        mask: torch.Tensor, 
        depth: torch.Tensor,
        intrinsics: torch.Tensor, 
        cam2world: torch.Tensor,
        num_points: int = 1000000,
        mask_pointmap: bool = True,
        camera_name: str = None
    ):
    if intrinsics is not None and cam2world is not None and intrinsics.shape[0] > 0 and cam2world.shape[0] > 0:
        rgb = rgb.cpu().numpy()
        intrinsics = intrinsics.cpu().numpy()
        cam2world_cpu = cam2world.cpu().numpy()

        rr.log(
            f"world/scene_visualization/{camera_name}/camera",
            rr.Transform3D(translation=cam2world_cpu[:3, 3], mat3x3=cam2world_cpu[:3, :3])
        )
        rr.log(
            f"world/scene_visualization/{camera_name}/camera/image",
            rr.Pinhole(
                resolution=[rgb.shape[1], rgb.shape[0]],
                focal_length=[intrinsics[0,0], intrinsics[1,1]],
                principal_point=[intrinsics[0,2], intrinsics[1,2]],
            ),
        )

        if rgb is not None and rgb.shape[0] > 0:
            rr.log(
                f"world/scene_visualization/{camera_name}/camera/image", rr.Image(rgb)
            )
            if mask is not None and mask.shape[0] > 0:
                mask_cpu = mask.float().cpu().numpy()
                rr.log(f"world/scene_visualization/{camera_name}/camera/mask", rr.Image(mask_cpu))
        else:
            rr.log(
                f"world/scene_visualization/{camera_name}/camera/image", rr.Image((255*mask.int()).unsqueeze(len(mask.shape)).repeat(1,1,3))
            )

        if dino_features is not None and dino_features.shape[0] > 0:
            dino_features = dino_features.cpu().numpy()
            pca = PCA(n_components=3)
            pca.fit(dino_features)
            projected_tokens = pca.transform(dino_features)

            t = torch.tensor(projected_tokens)
            t_min = t.min(dim=0, keepdim=True).values
            t_max = t.max(dim=0, keepdim=True).values
            normalized_t = (t - t_min) / (t_max - t_min)

            array = (normalized_t * 255).byte().numpy()
            array = array.reshape(rgb.shape[0]//14,rgb.shape[1]//14,3)
            rr.log(f"world/scene_visualization/{camera_name}/camera/dino_features", rr.Image(array))
    
        pointmap = None
        if depth is not None and depth.shape[0] > 0:
            pointmap = compute_pointmap(depth, intrinsics, cam2world)
        if pointmap is not None and pointmap.shape[0] > 0:
            if mask is not None and mask_pointmap:
                pointmap = pointmap.cpu().numpy()[mask.cpu().numpy().astype(bool)].reshape(-1, 3)
            else:
                pointmap = pointmap.cpu().numpy().reshape(-1, 3)
            
            # Transform the pointmap back to world frame using the camera's extrinsics
            #pointmap_hom = np.concatenate([pointmap, np.ones((pointmap.shape[0], 1))], axis=1)
            #pointmap_transformed = (cam2world @ pointmap_hom.T).T
            #pointmap = pointmap_transformed[:, :3]

            num_points_pointmap = min(num_points, pointmap.shape[0])
            pointmap_indices = np.random.choice(pointmap.shape[0], size=num_points_pointmap, replace=False)
            # Color pointmaps differently for each camera
            pointmap_color = pointmap_colors[int(camera_name) % len(pointmap_colors)]
            rr.log(f"world/scene_visualization/{camera_name}/pointmap", rr.Points3D(pointmap[pointmap_indices], colors=pointmap_color))