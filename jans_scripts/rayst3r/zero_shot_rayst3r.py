from rayst3r.eval_wrapper.eval import EvalWrapper
from huggingface_hub import hf_hub_download
from rayst3r.eval_wrapper.eval_utils import npy2ply
from rayst3r.eval_wrapper.eval import eval_scene

import os
import open3d as o3d
import argparse

camera = "realsense"

def main():
    # data_dir = "/data/graspnet/rayst3r_data"
    # output_dir = "/data/graspnet/output/GraspNet/rayst3r_zero_shot"
    # scene_id = "scene_0100"
    # view = "0000"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data/graspnet/rayst3r_data")
    parser.add_argument("--output_dir", type=str, default="/data/graspnet/output/GraspNet/rayst3r_zero_shot")
    parser.add_argument("--scene_id", type=str, default="0100")
    parser.add_argument("--view", type=str, default="0000")
    parser.add_argument("--force", action="store_true", default=False)

    args = parser.parse_args()

    scene_id_str = f"scene_{args.scene_id}"

    data_dir = os.path.join(args.data_dir, f"{camera}_{scene_id_str}_{args.view}")
    output_dir = os.path.join(args.output_dir, f"{camera}_{scene_id_str}_{args.view}", "predictions")
    os.makedirs(output_dir, exist_ok=True)

    all_points_save = os.path.join(output_dir, "inference_points.ply")
    if os.path.exists(all_points_save):
        if args.force:
            print(f"Found existing predictions for scene {args.scene_id} and view {args.view}, but force flag is set. Overwriting predictions.")
        else:
            print(f"Predictions already exist for scene {args.scene_id} and view {args.view}")
            return
    else:
        print(f"Running RaySt3R for scene {args.scene_id} and view {args.view}")

    visualize = False
    rr_addr = "0.0.0.0:9876"
    run_octmae = False
    set_conf = 5
    no_input_mask = False
    no_pred_mask = False
    no_filter_input_view = False
    false_positive = None
    false_negative = None
    n_pred_views = 5
    filter_all_masks = False
    tsdf = False

    rayst3r_checkpoint = hf_hub_download("bartduis/rayst3r", "rayst3r.pth")
    model = EvalWrapper(rayst3r_checkpoint,distributed=False)

    all_points = eval_scene(model, data_dir, visualize=visualize, rr_addr=rr_addr, run_octmae=run_octmae, set_conf=set_conf,
                            no_input_mask=no_input_mask,no_pred_mask=no_pred_mask,no_filter_input_view=no_filter_input_view,false_positive=false_positive,
                            false_negative=false_negative,n_pred_views=n_pred_views,
                            do_filter_all_masks=filter_all_masks,tsdf=tsdf).cpu().numpy()
                            
    o3d_pc = npy2ply(all_points,colors=None,normals=None)
    o3d.io.write_point_cloud(all_points_save, o3d_pc)

if __name__ == "__main__":
    main()