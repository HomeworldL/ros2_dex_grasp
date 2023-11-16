import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.object_model import ObjectModel
import numpy as np
import torch
import trimesh
import plotly.graph_objects as go

object_code_list = [
        'sem-Bottle-437678d4bc6be981c8724d5673a063a6',
        'core-mug-8570d9a8d24cb0acbebd3c0c0c70fb03',
        'sem-Camera-7bff4fd4dc53de7496dece3f86cb5dd5',
        ]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('running on', device)

object_model = ObjectModel(
    data_root_path='data/meshdata',
    batch_size_each=8,
    num_samples=2000, 
    device=device
)
object_model.initialize(object_code_list)

object_mesh = object_model.get_plotly_data(9)

# 创建一个 subplot
# fig = go.Figure(object_mesh)
# # fig.update_layout(scene_aspectmode='data')
# fig.show()

mesh_path = "data/meshdata"
data_path = "data/dataset"
grasp_code_list = []
for code in os.listdir(data_path):
    grasp_code_list.append(code[:-4])
object_mesh_origin = trimesh.load(os.path.join(
    mesh_path, grasp_code_list[1], "coacd/decomposed.obj"))
print(grasp_code_list)

object_mesh = object_mesh_origin.copy()
object_mesh.show()

