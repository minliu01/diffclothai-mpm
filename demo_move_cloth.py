import argparse
import contextlib
import os
import sys
from typing import Union
import time

import numpy as np
import torch
from tqdm import tqdm

import diffcloth_py as diffcloth
from src.python_code.pySim.pySim import pySim

sceneConfig = {
    # "fabric:k_stiff_stretching": "5500",
    # "fabric:k_stiff_bending": "120",
    "fabric:k_stiff_stretching": "0.8",
    "fabric:k_stiff_bending": "0.03",
    "fabric:name": "/home/ubuntu/MPM_CLOTH/envs/assets/towel/towel.obj",
    # "fabric:name": "/home/ubuntu/diffclothai/src/assets/meshes/remeshed/Wind/wind12x12.obj",
    "fabric:keepOriginalScalePoint": "true",
    "fabric:density": "1",
    # "fabric:custominitPos": "true",
    # "fabric:initPosFile": "/home/ubuntu/diffclothai/output/wind12x12_perturbed.txt",
    "timeStep": "2e-3",
    "stepNum": "200",
    "forwardConvergenceThresh": "1e-8",
    "backwardConvergenceThresh": "5e-4",
    # "attachmentPoints": "CUSTOM_ARRAY",
    # "customAttachmentVertexIdx": "0,11",
    "orientation": "CUSTOM_ORIENTATION",
    "upVector": "0,2,-1",
}

class ClothSimulator:
    def __init__(self):
        self.config = sceneConfig
        diffcloth.enableOpenMP(n_threads=args.n_openmp_thread)
        self.helper = diffcloth.makeOptimizeHelper(args.task_name)
        sim = diffcloth.makeCustomizedSim(exampleName=args.task_name, runBackward=True, config=sceneConfig)
        sim.forwardConvergenceThreshold = 1e-8
        self.sim = sim
        self.x_init, self.v_init, self.a_init = self.get_state()
        self.records = [self.sim.getStateInfo(), ]

        self.dL_dx = None
        self.dL_dv = None

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

    def step_grad(self, idx):
        record = self.records[idx + 1]
        backRecord = self.sim.stepBackwardNN(
            self.helper.taskInfo,
            self.dL_dx,
            self.dL_dv,
            record,
            record.stepIdx == 1, # TODO: check if this should be 0 or 1
            np.zeros_like(self.dL_dx),
            np.zeros_like(self.dL_dv))

        self.dL_dx = backRecord.dL_dx
        self.dL_dv = backRecord.dL_dv
        dL_dfext = backRecord.dL_dfext

        dL_da_norm = np.linalg.norm(backRecord.dL_dxfixed)
        if dL_da_norm > 1e-7:
            maxNorm = 4.0
            normalized = backRecord.dL_dxfixed * (max(min(backRecord.dL_dxfixed.shape[0] * maxNorm, dL_da_norm), 0.05) / dL_da_norm )
            dL_da = normalized
        else:
            dL_da = backRecord.dL_dxfixed

        return dL_da, dL_dfext

    def render(self):
        diffcloth.render(self.sim, renderPosPairs=True, autoExit=True)

def main(args):
    sim = ClothSimulator()

    x, v, a = sim.x_init, sim.v_init, sim.a_init
    # x_tgt = np.copy(x).reshape(-1, 3)
    # x_tgt[:, 2] += 2.
    # x_tgt = x_tgt.reshape(-1)
    x_tgt = np.load("demo_move_cloth_target_small.npy")

    f = np.zeros((*x.shape, ))
    f_grad = np.zeros((200, *x.shape))

    for epoch in range(args.epochs):
        tik = time.time()
        sim.reset()
        x, v, a = sim.x_init, sim.v_init, sim.a_init

        for step in range(200):
            x, v = sim.step(x, v, a, f)

        loss = ((x - x_tgt) ** 2).sum()
        sim.dL_dv = np.zeros_like(v)
        sim.dL_dx = np.zeros_like(x)
        sim.dL_dx = 2 * (x - x_tgt)

        for i in range(200 - 1, -1, -1):
            dL_da, dL_df = sim.step_grad(i)
            f_grad[i] = dL_df

        lr = args.lr / (epoch + 1)
        f -= lr * f_grad.mean(0)
        tok = time.time()
        print("Epoch {} | Loss: {:.2f} | Time: {:.2f}".format(epoch, loss, tok-tik))

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
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args)