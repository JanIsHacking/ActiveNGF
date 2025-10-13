import open3d as o3d
import pickle

def dict_to_mesh(d):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(d["vertices"])
    mesh.triangles = o3d.utility.Vector3iVector(d["triangles"])
    mesh.vertex_colors = o3d.utility.Vector3dVector(d["vertex_colors"])
    return mesh

# Load pickled meshes
with open("geom.pkl", "rb") as f:
    geom_dicts = pickle.load(f)

geom = [dict_to_mesh(d) for d in geom_dicts]#[::int(len(geom_dicts)/10)]

# Optional coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
o3d.visualization.draw_geometries([*geom, frame])
