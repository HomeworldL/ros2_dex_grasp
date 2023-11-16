#!/usr/bin/env python
# coding: utf-8

import os
import sys

# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import random
from utils.hand_model import HandModel
from utils.hand_model_lite import HandModelMJCFLite
import numpy as np
import transforms3d
import torch
import trimesh

xml_file = 'description/xml/freehand_right.xml'
mesh_file = 'description/meshes'

hand_model = HandModelMJCFLite(
    xml_file,
    mesh_file)

rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
# hand_pose = torch.tensor([0.0]*3 + rot + [0.3]*hand_model.n_dofs, dtype=torch.float, device='cpu').unsqueeze(0)
hand_pose = torch.tensor([0.0]*3 + rot + [0.0,1.5,1.2,1.2,
                                          0.0,0.3,0.0,0.0,
                                          0.0,0.0,0.0,0.0,
                                          -0.3,0.0,0.0,0.0,
                                          1.2,0.0,0.9,0.9], 
                         dtype=torch.float, device='cpu').unsqueeze(0)

hand_model.set_parameters(hand_pose)
hand_mesh = hand_model.get_trimesh_data(0)

(hand_mesh).show()

