import torch
import pytorch_kinematics as pk

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

chain = pk.build_serial_chain_from_urdf(open("description/urdf/freehand.urdf").read(), "f14")
chain = chain.to(dtype=dtype, device=d)

N = 1000
th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)

# order of magnitudes faster when doing FK in parallel
# elapsed 0.008678913116455078s for N=1000 when parallel
# (N,4,4) transform matrix; only the one for the end effector is returned since end_only=True by default
tg_batch = chain.forward_kinematics(th_batch)

# elapsed 8.44686508178711s for N=1000 when serial
for i in range(N):
    tg = chain.forward_kinematics(th_batch[i])