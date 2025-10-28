import numpy as np
from graspnetAPI import GraspGroup
import open3d as o3d
import pickle

def mesh_to_dict(mesh: o3d.geometry.TriangleMesh):
    return {
        "vertices": np.asarray(mesh.vertices),
        "triangles": np.asarray(mesh.triangles),
        "vertex_colors": np.asarray(mesh.vertex_colors)
    }

def dict_to_mesh(d):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(d["vertices"])
    mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(d["vertex_colors"])
    return mesh

def visualize_grasps(grasp_group: GraspGroup):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    geom = grasp_group.to_open3d_geometry_list()
    
    # Convert meshes to dictionaries
    geom_dicts = [mesh_to_dict(g) for g in geom]
    
    # Save picklable version
    with open("geom.pkl", "wb") as f:
        pickle.dump(geom_dicts, f)

if __name__ == "__main__":
    gg_path = "/workspace/output/GraspNet/scene_0100_nbv/grasps/00000/scene_0100/realsense/result.npy"
    grasp_group = GraspGroup().from_npy(gg_path)
    visualize_grasps(grasp_group)
