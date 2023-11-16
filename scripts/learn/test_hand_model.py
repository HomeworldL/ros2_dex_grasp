#!/usr/bin/env python
# coding: utf-8

import os
import sys
# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
from utils.hand_model import HandModel
import numpy as np
import transforms3d
import torch
import trimesh
import plotly.graph_objects as go

xml_file = 'description/xml/freehand_right.xml'
mesh_file = 'description/meshes'

device = torch.device('cuda')
hand_model = HandModel(
    xml_file,
    mesh_file,
    contact_points_path='description/contact_points.json',
    penetration_points_path='description/penetration_points.json',
    device=device
    )

rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
hand_pose = torch.tensor([0.0]*3 + rot + [0.0,1.5,1.2,1.2,
                                          0.0,0.3,0.0,0.0,
                                          0.0,0.0,0.0,0.0,
                                          -0.3,0.0,0.0,0.0,
                                          1.2,0.0,0.9,0.9], 
                         dtype=torch.float, device=device).unsqueeze(0)
contact_point_indices = torch.tensor([0,1,2,3,4], dtype=torch.int64, device=device).unsqueeze(0)
hand_model.set_parameters(hand_pose,contact_point_indices=contact_point_indices)

hand_mesh = hand_model.get_plotly_data(0,with_contact_points=True)

# 创建一个 subplot
fig = go.Figure(hand_mesh)
# fig.update_layout(scene=dict(aspectmode='cube'))
fig.show()

# hand_mesh = hand_model.get_trimesh_data(0)
# (hand_mesh).show()