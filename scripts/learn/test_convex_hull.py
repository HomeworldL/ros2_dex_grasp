#!/usr/bin/env python
# coding: utf-8

import os
import sys
# os.chdir(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.initializations import initialize_convex_hull
import numpy as np
import transforms3d
import torch
import trimesh
import plotly.graph_objects as go
import math
import argparse

# prepare arguments

parser = argparse.ArgumentParser()
# experiment settings
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--gpu', default="0", type=str)
parser.add_argument('--object_code_list', default=
    [
        'sem-Bottle-437678d4bc6be981c8724d5673a063a6',
        # 'sem-Car-27e267f0570f121869a949ac99a843c4',
        # 'sem-Car-669043a8ce40d9d78781f76a6db4ab62',
        # 'sem-Car-58379002fbdaf20e61a47cff24512a0',
        # 'sem-Car-aeeb2fb31215f3249acee38782dd9680',
    ], type=list)
parser.add_argument('--name', default='exp_2', type=str)
parser.add_argument('--n_contact', default=5, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--n_iter', default=2000, type=int)
# hyper parameters (** Magic, don't touch! **)
parser.add_argument('--switch_possibility', default=0.5, type=float)
parser.add_argument('--mu', default=0.98, type=float)
parser.add_argument('--step_size', default=0.005, type=float)
parser.add_argument('--stepsize_period', default=50, type=int)
parser.add_argument('--starting_temperature', default=18, type=float)
parser.add_argument('--annealing_period', default=30, type=int)
parser.add_argument('--temperature_decay', default=0.95, type=float)
parser.add_argument('--w_dis', default=100.0, type=float)
parser.add_argument('--w_pen', default=100.0, type=float)
parser.add_argument('--w_spen', default=10.0, type=float)
parser.add_argument('--w_joints', default=1.0, type=float)
# initialization settings
parser.add_argument('--jitter_strength', default=0.1, type=float)
parser.add_argument('--distance_lower', default=0.2, type=float)
parser.add_argument('--distance_upper', default=0.3, type=float)
parser.add_argument('--theta_lower', default=-math.pi / 6, type=float)
parser.add_argument('--theta_upper', default=math.pi / 6, type=float)
# energy thresholds
parser.add_argument('--thres_fc', default=0.3, type=float)
parser.add_argument('--thres_dis', default=0.005, type=float)
parser.add_argument('--thres_pen', default=0.001, type=float)

args = parser.parse_args()

np.seterr(all='raise')
np.random.seed(args.seed)
torch.manual_seed(args.seed)

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

object_model = ObjectModel(
    data_root_path='data/meshdata',
    batch_size_each=args.batch_size,
    num_samples=2000, 
    device=device
)
object_model.initialize(args.object_code_list)

initialize_convex_hull(hand_model, object_model, args)

object_mesh = object_model.get_plotly_data(0)
hand_mesh = hand_model.get_plotly_data(2, with_contact_points=True)

fig = go.Figure(hand_mesh + object_mesh)
fig.update_layout(scene_aspectmode='data')
fig.show()

