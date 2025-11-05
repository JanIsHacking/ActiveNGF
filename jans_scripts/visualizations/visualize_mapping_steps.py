import os
import glob
import io
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from jans_scripts.graspnet_storage import get_graspnet_data_path


def remove_vertices_above_z(mesh: o3d.geometry.TriangleMesh, z_threshold: float = 0.5) -> o3d.geometry.TriangleMesh:
    """
    Removes all vertices (and corresponding faces) from an Open3D TriangleMesh
    whose z-coordinate is greater than z_threshold. Colors are preserved.

    Args:
        mesh: Input Open3D TriangleMesh.
        z_threshold: Threshold for z-coordinate filtering (default = 0.5).

    Returns:
        A new Open3D TriangleMesh with filtered vertices and faces.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    # Keep only vertices with z <= threshold
    mask = vertices[:, 2] <= z_threshold
    valid_indices = np.where(mask)[0]

    # Create a mapping from old to new vertex indices
    index_map = -np.ones(len(vertices), dtype=int)
    index_map[valid_indices] = np.arange(len(valid_indices))

    # Filter out triangles that reference removed vertices
    valid_triangles_mask = np.all(mask[triangles], axis=1)
    filtered_triangles = triangles[valid_triangles_mask]
    remapped_triangles = index_map[filtered_triangles]

    # Create new mesh
    filtered_vertices = vertices[mask]
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(filtered_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)

    # Preserve colors if available
    if colors is not None:
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[mask])

    return new_mesh


def render_mesh(mesh: o3d.geometry.TriangleMesh, mapping_step: int, scale_factor: float = 1.5, target_resolution=(640, 480), first_center=None) -> np.ndarray:
    if not mesh.has_vertices():
        img = np.zeros((target_resolution[1], target_resolution[0], 3), dtype=np.uint8)
        return img
    
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_extent = 1
    
    if max_extent == 0:
        img = np.zeros((target_resolution[1], target_resolution[0], 3), dtype=np.uint8)
        return img
    
    mesh_copy = o3d.geometry.TriangleMesh(mesh)

    # Remove artifacts by removing all points of the mesh that have a z value greater than 0.5
    mesh_copy = remove_vertices_above_z(mesh_copy, z_threshold=0.3)

    mesh_copy.translate(-first_center)
    scale = scale_factor / max_extent
    mesh_copy.scale(scale, center=(0, 0, 0))
    
    vertices = np.asarray(mesh_copy.vertices)
    faces = np.asarray(mesh_copy.triangles)
    
    fig = plt.figure(figsize=(target_resolution[0]/100, target_resolution[1]/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    
    if mesh_copy.has_vertex_colors():
        vertex_colors = np.asarray(mesh_copy.vertex_colors)[:, :3]
        face_colors = vertex_colors[faces].mean(axis=1)
    else:
        mesh_copy.compute_vertex_normals()
        normals = np.asarray(mesh_copy.vertex_normals)
        face_normals = normals[faces].mean(axis=1)
        face_colors = (face_normals + 1.0) / 2.0
    
    triangles = vertices[faces]
    collection = Poly3DCollection(triangles, facecolors=face_colors, edgecolors='none', linewidths=0)
    ax.add_collection3d(collection)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    
    ax.view_init(elev=30, azim=45)
    ax.set_axis_off()
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.set_box_aspect([1, 1, 1])
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches=None, pad_inches=0, facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    
    if img.size != target_resolution:
        img = img.resize(target_resolution, Image.Resampling.LANCZOS)
    
    plt.close(fig)
    
    return np.array(img)


def visualize_mapping_steps(method: str, scene_id: str, visualize_steps, scale_factor: float, target_resolution):
    graspnet_path = get_graspnet_data_path()
    mesh_dir = f"{graspnet_path}/output/GraspNet/{method}/{scene_id}/mesh"
    output_dir = f"{graspnet_path}/output/GraspNet/{method}/{scene_id}/render"
    os.makedirs(output_dir, exist_ok=True)

    render_outputs = []
    first_center = None
    for mesh_path in sorted(glob.glob(os.path.join(mesh_dir, "*.ply")), reverse=True):
        filename = os.path.basename(mesh_path)
        mapping_step = int(filename.split("_")[0])
        if mapping_step not in visualize_steps:
            continue
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if first_center is None:
            first_center = mesh.get_center()
        render_output = render_mesh(mesh, mapping_step, scale_factor=scale_factor, target_resolution=target_resolution, first_center=first_center)
        render_outputs.append((mapping_step, render_output))
    
    render_outputs.sort(key=lambda x: x[0])
    
    if not render_outputs:
        print(f"No meshes found for the specified mapping steps {visualize_steps} in scene {scene_id}")
        return
    
    images = [img for _, img in render_outputs]
    mappings_steps = [step for step, _ in render_outputs]
    print(f"Mappings steps: {mappings_steps}")
    combined_image = np.hstack(images)

    # save each image as a separate file
    # for i, img in enumerate(images):
    #     output_path = os.path.join(output_dir, f"mapping_step_{i}.png")
    #     Image.fromarray(img).save(output_path)
    #     print(f"Saved image {i} to {output_path}")
    
    output_path = os.path.join(output_dir, f"mapping_steps_{scene_id}.png")
    Image.fromarray(combined_image).save(output_path)


if __name__ == "__main__":
    graspnet_path = get_graspnet_data_path()
    method = "baseline_random_nbv"
    visualize_steps = [0, 2, 5, 10]
    scale_factor = 5
    target_resolution = (2*640, 2*640)
    force = True

    for scene in sorted(os.listdir(os.path.join(graspnet_path, "output", "GraspNet", method))):
        # if scene != "scene_0124_nbv":
        #     continue
        if not os.path.isdir(os.path.join(graspnet_path, "output", "GraspNet", method, scene)):
            print(f"Scene {scene} not found")
            continue
        output_path = os.path.join(graspnet_path, "output", "GraspNet", method, scene, "render", f"mapping_steps_{scene}.png")
        if os.path.exists(output_path) and not force:
            print(f"Visualizing scene {scene} - Skipped")
            continue
        print(f"Visualizing scene {scene}")
        visualize_mapping_steps(method, scene, visualize_steps, scale_factor, target_resolution)