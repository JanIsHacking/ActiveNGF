import torch
import rerun as rr
import numpy as np
from sklearn.decomposition import PCA
from jans_scripts.geometry import compute_pointmap
import os


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
                resolution=[mask.shape[1], mask.shape[0]],
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