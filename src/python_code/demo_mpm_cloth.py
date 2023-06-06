import argparse
import contextlib
import os
import sys
from typing import Union

import numpy as np
import torch
from tqdm import tqdm

import diffcloth_py as diffcloth
from src.python_code.pySim.pySim import pySim

sceneConfig = {
    "fabric:k_stiff_stretching": "5500",
    "fabric:k_stiff_bending": "120",
    "fabric:name": "/home/ubuntu/diffclothai/src/assets/meshes/remeshed/Wind/wind12x12.obj",
    "fabric:keepOriginalScalePoint": "true",
    # "fabric:custominitPos": "true",
    # "fabric:initPosFile": "/home/ubuntu/diffclothai/output/wind12x12_perturbed.txt",
    "timeStep": "2e-3",
    "stepNum": "200",
    "forwardConvergenceThresh": "1e-8",
    "backwardConvergenceThresh": "5e-4",
    "attachmentPoints": "CUSTOM_ARRAY",
    "customAttachmentVertexIdx": "0,11",
    # "orientation": "CUSTOM_ORIENTATION",
    # "upVector": "0,0,1",
}

def get_state(sim: diffcloth.Simulation, to_tensor: bool = False) -> tuple:
    state_info_init = sim.getStateInfo()
    x, v = state_info_init.x, state_info_init.v
    clip_pos = np.array(sim.getStateInfo().x_fixedpoints)
    if to_tensor:
        x_t = torch.tensor(x).clone()
        v_t = torch.tensor(v).clone()
        a_t = torch.tensor(clip_pos).clone()
        return x_t, v_t, a_t
    else:
        return x, v, clip_pos

def get_center_pos(
    sim: diffcloth.Simulation, corner_idx: list = [315, 314, 284, 285]
) -> torch.Tensor:
    v_pos, _, _ = get_state(sim, to_tensor=True)
    v_pos = v_pos.reshape(-1, 3)
    center_pos = v_pos[torch.LongTensor(corner_idx)].mean(0)
    return center_pos

def forward_sim_no_control(
    x_i: torch.Tensor,
    v_i: torch.Tensor,
    a_t: torch.Tensor,
    pysim: pySim,
    steps: int,
) -> list:
    """Pure physics simulation."""
    records = []
    for step in tqdm(range(steps)):
        records.append((x_i, v_i))
        da_t = torch.zeros_like(a_t)
        a_t += da_t
        x_i, v_i = pysim(x_i, v_i, a_t)
    records.append((x_i, v_i))
    return records

def forward_sim_targeted_control(
    x_i: torch.Tensor,
    v_i: torch.Tensor,
    a_t: torch.Tensor,
    tgt_pos: torch.Tensor,
    pysim: pySim,
    steps: int,
    action_repeat: int = 4,
) -> list:
    start_pos = a_t.clone().numpy()
    tgt_pos = tgt_pos.numpy()

    records = []
    for step in range(steps):
        ratio = (step + 1) / steps
        point = start_pos + (tgt_pos - start_pos) * ratio
        a_t = torch.tensor(point).clone()
        records.append((x_i, v_i))
        for _ in range(action_repeat):
            x_i, v_i = pysim(x_i, v_i, a_t)

    records.append((x_i, v_i))
    return records

def export_mesh(sim: diffcloth.Simulation, step: int = None):
    if step is None:
        step = (sim.sceneConfig.stepNum - 1,)
    sim.exportCurrentMeshPos(step, "wind12x12_perturbed")

def wrap(args):
    helper = diffcloth.makeOptimizeHelper(args.task_name)
    sim = diffcloth.makeCustomizedSim(exampleName=args.task_name, runBackward=False, config=sceneConfig)
    sim.forwardConvergenceThreshold = 1e-8
    pysim = pySim(sim, helper, True)

    # Reset the system
    sim.resetSystem()
    x0_t, v0_t, a0_t = get_state(sim, to_tensor=True)

    x_tmp = x0_t.reshape(-1, 3)
    print("mean", x_tmp.mean(0))
    print("max", x_tmp.max(0)[0])
    print("min", x_tmp.min(0)[0])
    return

    control_idx = sim.sceneConfig.customAttachmentVertexIdx[0][1]
    control_x0_t = [x0_t.reshape(-1, 3)[idx] for idx in control_idx]
    control_tgt = sum(control_x0_t)
    control_tgt[1] += 2
    control_tgt = torch.cat([control_tgt] * len(control_idx))
    _ = forward_sim_targeted_control(x0_t, v0_t, a0_t, control_tgt, pysim, 200)
    # _ = forward_sim_no_control(x0_t, v0_t, a0_t, pysim, 200)

    # Stablise simulation
    x_t, v_t, a_t = get_state(sim, to_tensor=True)
    _ = forward_sim_no_control(x_t, v_t, a_t, pysim, 30)

    # Rendering the simulationg
    if args.render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=True)

    # Export final configuration into wavefront file
    if args.save:
        export_mesh(sim, step=sim.getStateInfo().stepIdx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perturb flat cloth")
    parser.add_argument(
        "-mode", type=int, default=1,
        help="-1: no change, 0: random perturb, 1: bezier purturb",
    )
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--task-name", type=str, default="mpm_cloth")
    parser.add_argument("--n-openmp-thread", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="cloth_project/")
    parser.add_argument("--seed", type=int, default=8824325)
    args = parser.parse_args()
    
    diffcloth.enableOpenMP(n_threads=args.n_openmp_thread)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wrap(args)