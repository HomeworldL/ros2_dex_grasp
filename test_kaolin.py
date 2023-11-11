import torch
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.metrics.trianglemesh import point_to_mesh_distance

point = torch.tensor([[[0.5, 0.5, 0.5],

                       [3., 4., 5.]]], device='cuda')

vertices = torch.tensor([[[0., 0., 0.],

                          [0., 1., 0.],

                          [0., 0., 1.]]], device='cuda')

faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device='cuda')

face_vertices = index_vertices_by_faces(vertices, faces)

distance, index, dist_type = point_to_mesh_distance(point, face_vertices)

print(distance)


print(index)


print(dist_type)
