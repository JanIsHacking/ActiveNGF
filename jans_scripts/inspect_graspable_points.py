import torch
import rerun as rr
import pickle

from jans_scripts.visualize import setup_rerun, pointmap_colors

def inspect_graspable_points():
    points_path = "/home/jan/thesis/ActiveNGF/jans_scripts/data/batch_graspable_xyz_0.pt"
    points = torch.load(points_path)

    bounding_boxes_path = "/home/jan/thesis/ActiveNGF/jans_scripts/data/object_bounding_boxes_0.pkl"
    bounding_boxes = pickle.load(open(bounding_boxes_path, "rb"))

    point_per_object_path = "/home/jan/thesis/ActiveNGF/jans_scripts/data/graspable_xyz.pt"
    point_per_object = torch.load(point_per_object_path)[0]

    setup_rerun("inspect_graspable_points")
    rr.log("world/points", rr.Points3D(points.cpu().numpy()))

    rr.log("world/point_per_object", rr.Points3D(point_per_object.cpu().numpy(), colors=[255, 0, 0]))

    for idx, (object_id, bounding_box) in enumerate(bounding_boxes.items()):
        rr.log(f"world/object_{object_id}", rr.Transform3D(translation=bounding_box[0]))
        rr.log(f"world/object_{object_id}", rr.Arrows3D(vectors=bounding_box[1]-bounding_box[0], colors=[pointmap_colors[idx % len(pointmap_colors)]]))

if __name__ == "__main__":
    inspect_graspable_points()
