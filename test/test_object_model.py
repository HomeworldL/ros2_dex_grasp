from utils.object_model import ObjectModel
import numpy as np
import torch
import trimesh
import plotly.graph_objects as go

object_code_list = [
        'sem-Bottle-437678d4bc6be981c8724d5673a063a6',
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

object_mesh = object_model.get_plotly_data(1)

# 创建一个 subplot
fig = go.Figure(object_mesh)
# fig.update_layout(scene_aspectmode='data')
fig.show()
