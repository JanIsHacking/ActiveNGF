import torch
import os

from PIL import Image

from jans_scripts.graspnet_storage import get_graspnet_data_path

rayst3r_data_dir = "rayst3r_data"

DEPTH_MAX = 10.0
TORCH_UINT16_MAX = 65535

CAM2WORLD_FILE_NAME = "cam2world.pt"
INTRINSICS_FILE_NAME = "intrinsics.pt"
MASK_FILE_NAME = "mask.png"
RGB_FILE_NAME = "rgb.png"
DINO_FEATURES_FILE_NAME = "dino_features.pt"
DEPTH_FILE_NAME = "depth.png"

def _scale_depth(depth: torch.Tensor) -> torch.Tensor:
    return (depth / DEPTH_MAX * TORCH_UINT16_MAX).clamp(0, TORCH_UINT16_MAX).to(torch.uint16)

def save_tensor_as_png(tensor: torch.Tensor, path: str, dtype: torch.dtype | None = None):
    if dtype is None:
        dtype = tensor.dtype
    Image.fromarray(tensor.to(dtype).cpu().numpy()).save(path)

def save_frame(
        camera_dir: str,
        rgb: torch.Tensor, 
        mask: torch.Tensor, 
        intrinsics: torch.Tensor, 
        cam2world: torch.Tensor, 
        depth: torch.Tensor, 
        pointmap: torch.Tensor = None,
        dtype: torch.dtype = torch.float32
    ):
    os.makedirs(camera_dir, exist_ok=True)
    if rgb is not None:
        save_tensor_as_png(rgb, os.path.join(camera_dir, RGB_FILE_NAME), dtype=torch.uint8)
    if mask is not None:
        save_tensor_as_png(mask, os.path.join(camera_dir, MASK_FILE_NAME), dtype=torch.bool)
    if intrinsics is not None:
        torch.save(intrinsics.cpu().to(dtype), os.path.join(camera_dir, INTRINSICS_FILE_NAME))
    if cam2world is not None:
         torch.save(cam2world.cpu().to(dtype), os.path.join(camera_dir, CAM2WORLD_FILE_NAME))
    if depth is not None:
        depth_scaled = _scale_depth(depth)
        save_tensor_as_png(depth_scaled, os.path.join(camera_dir, DEPTH_FILE_NAME), dtype=torch.uint16)

def get_rayst3r_data_path():
    return os.path.join(get_graspnet_data_path(), rayst3r_data_dir)