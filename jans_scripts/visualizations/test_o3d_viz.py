import open3d as o3d

mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
mat = o3d.visualization.rendering.MaterialRecord()
render.scene.add_geometry("frame", mesh, mat)
img = render.render_to_image()
o3d.io.write_image("test_render.png", img)
print("Saved offscreen render to test_render.png")