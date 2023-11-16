import open3d as o3d

cube   = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
cube_mesh = o3d.t.geometry.TriangleMesh.from_legacy(cube)
cube_scene = o3d.t.geometry.RaycastingScene()
_ = cube_scene.add_triangles(cube_mesh) 

s = cube_scene.compute_signed_distance(o3d.core.Tensor([[-1, -1, 0],[-1, -1, -1],[0.5,0.5,0.5]], dtype=o3d.core.Dtype.Float32))
print(s)