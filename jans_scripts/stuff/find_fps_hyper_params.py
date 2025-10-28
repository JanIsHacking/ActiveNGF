from jans_scripts.visualize import visualize_gt_points, setup_rerun

from jans_scripts.geometry import furthest_point_sampling
from jans_scripts.graspnet_storage import get_scene_gt_points

graspnet_data_path = "/data/user/jan/graspnet"

def main():
    setup_rerun('Find FPS Hyper Params')

    scene_id = "scene_0130"
    camera = 'realsense'
    view = "0000"
    gt_points = get_scene_gt_points(scene_id, camera, view)

    for num_samples in [128, 256, 512, 1024, 2048, 4096]:
        sampled_points = furthest_point_sampling(gt_points, num_samples)
        visualize_gt_points(sampled_points, num_points=num_samples, suffix=f"fps_{num_samples}")
    
    # 2048 seems to work the best for now

if __name__ == "__main__":
    main()