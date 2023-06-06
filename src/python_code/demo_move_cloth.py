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
    "fabric:pmass": "1",
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

class ClothSimulator:
    def __init__(self):
        self.config = sceneConfig
        diffcloth.enableOpenMP(n_threads=args.n_openmp_thread)
        # helper = diffcloth.makeOptimizeHelper(args.task_name)
        sim = diffcloth.makeCustomizedSim(exampleName=args.task_name, runBackward=False, config=sceneConfig)
        sim.forwardConvergenceThreshold = 1e-8
        self.sim = sim
        self.x_init, self.v_init, self.a_init = self.get_state()
        self.records = [self.sim.getStateInfo(), ]

    def reset(self):
        self.sim.resetSystem()
        self.records = [self.sim.getStateInfo(), ]

    def get_state(self):
        state_info = self.sim.getStateInfo()
        x, v, a = state_info.x, state_info.v, state_info.x_fixedpoints
        return x, v, a

    def step(self, x, v, a, f_ext=None):
        if f_ext is None:
            f_ext = np.zeros_like(v)

        idx = self.records[-1].stepIdx + 1
        self.sim.stepCouple(idx, x, v, a, f_ext)

        newRecord = self.sim.getStateInfo()
        self.records.append(newRecord)

        x_next = newRecord.x
        v_next = newRecord.v

        return x_next, v_next

    def render(self):
        diffcloth.render(self.sim, renderPosPairs=True, autoExit=True)

def main(args):
    sim = ClothSimulator()

    x, v, a = sim.x_init, sim.v_init, sim.a_init
    for i in range(200):
        x, v = sim.step(x, v, a)

    if args.render:
        sim.render()

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
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)