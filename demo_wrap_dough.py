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

class Bezier:
    def parameterized_two_points(
        self, t: Union[int, float], P1: np.ndarray, P2: np.ndarray
    ) -> np.ndarray:
        """Returns a point between P1 and P2, parametised by t."""
        Q1 = (1 - t) * P1 + t * P2
        return Q1

    def bezier_points(self, t: Union[int, float], points: np.ndarray) -> list:
        """Returns a list of points interpolated by the Bezier process."""
        newpoints = []
        for i1 in range(0, len(points) - 1):
            newpoints += [
                self.parameterized_two_points(t, points[i1], points[i1 + 1])
            ]
        return newpoints

    def bezier_point(
        self, t: Union[int, float], points: np.ndarray
    ) -> np.ndarray:
        """Returns a point interpolated by the Bezier process."""
        newpoints = points
        while len(newpoints) > 1:
            newpoints = self.bezier_points(t, newpoints)
        return newpoints[0]

    def bezier_curve(
        self, t_values: Union[tuple, list], points: np.ndarray
    ) -> np.ndarray:
        """Returns a point interpolated by the Bezier process."""
        assert len(t_values) > 0, "t_values must contain at least one value."
        curve = np.array([[0.0] * len(points[0])])
        for t in t_values:
            curve = np.append(curve, [self.bezier_point(t, points)], axis=0)
        curve = np.delete(curve, 0, 0)
        return curve


@contextlib.contextmanager
def stdout_redirected(to=os.devnull):
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield
        finally:
            _redirect_stdout(to=old_stdout)


def toTorchTensor(x, requriesGrad=False, toDouble=False):
    torchX = torch.Tensor(x)
    if toDouble:
        torchX = torchX.double()
    torchX = torchX.view(-1).clone().detach().requires_grad_(requriesGrad)
    return torchX


def get_state(sim: diffcloth.Simulation, to_tensor: bool = False) -> tuple:
    state_info_init = sim.getStateInfo()
    x, v = state_info_init.x, state_info_init.v
    clip_pos = np.array(sim.getStateInfo().x_fixedpoints)
    if to_tensor:
        x_t = toTorchTensor(x, False, False).clone()
        v_t = toTorchTensor(v, False, False).clone()
        a_t = toTorchTensor(clip_pos, False, False).clone()
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


def forward_sim_rand_control(
    x_i: torch.Tensor,
    v_i: torch.Tensor,
    a_t: torch.Tensor,
    pysim: pySim,
    steps: int,
    dilution: float = 0.1,
    action_repeat: int = 4,
) -> list:
    records = []
    for step in tqdm(range(steps)):
        records.append((x_i, v_i))
        da_t = (torch.rand(a_t.shape) - 0.5) * 2 * dilution
        a_t += da_t
        for _ in range(action_repeat):
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
    min_height: float = 1,
    max_height: float = 3,
) -> list:

    start_pos = a_t.clone().numpy()
    tgt_pos = tgt_pos.numpy()

    mid_pos = start_pos - (tgt_pos - start_pos) / 5
    for idx in range(int(tgt_pos.shape[0] / 3)):
        mid_pos[3 * idx + 1] = (
            start_pos[3 * idx + 1]
            + (tgt_pos[3 * idx + 1] - start_pos[3 * idx + 1]) * 2 / 3
        )
    bezier = Bezier()
    t_points = np.arange(0, 1, 1.0 / (steps + 3))
    points = np.array([start_pos, mid_pos, tgt_pos])
    curve_points = bezier.bezier_curve(t_points, points)

    records = []
    for step in tqdm(range(steps)):
        points = curve_points[step]
        records.append((x_i, v_i))
        a_t = toTorchTensor(points, False, False).clone()
        for _ in range(action_repeat):
            x_i, v_i = pysim(x_i, v_i, a_t)
    records.append((x_i, v_i))
    return records


def export_mesh(
    sim: diffcloth.Simulation,
    out_fn: str,
    tmp_fn: str = "untextured.obj",
    cano_fn: str = "textured_flat_cloth.obj",
    dir_prefix: str = "output/cloth_project",
    export_step: int = None,
    renormalize: bool = False,
) -> None:

    if export_step is None:
        export_step = (sim.sceneConfig.stepNum - 1,)
    # export to untextured object
    sim.exportCurrentMeshPos(
        export_step,
        f"{dir_prefix}/{tmp_fn}".replace("output/", "").replace(".obj", ""),
    )

    if renormalize:
        center_pos = get_center_pos(sim)
        center_pos[2] = 0.0  # in-plane normalization
    if os.path.isfile(f"{dir_prefix}/{tmp_fn}"):
        obj_lines = []
        with open(f"{dir_prefix}/{cano_fn}", "r") as fp:
            found_mtl = False
            cano_vpos_idx = []
            for i, line in enumerate(fp.readlines()):
                obj_lines.append(line)
                if line.startswith("v "):
                    cano_vpos_idx.append(i)
                if ".mtl" in line:
                    found_mtl = True
        if found_mtl:
            with open(f"{dir_prefix}/{tmp_fn}", "r") as fp:
                new_vpos_lines = [
                    line for line in fp.readlines() if line.startswith("v ")
                ]
                assert len(new_vpos_lines) == len(
                    cano_vpos_idx
                ), "the numbers of vertices mismatch"
                for i, line_idx in enumerate(cano_vpos_idx):
                    if renormalize:
                        tmp_pos = new_vpos_lines[i].strip().split()[-3:]
                        pos = [
                            float(n) - center_pos[i]
                            for i, n in enumerate(tmp_pos)
                        ]
                        new_vpos_lines[i] = f"v {pos[0]} {pos[1]} {pos[2]}\n"
                    obj_lines[line_idx] = new_vpos_lines[i]
            with open(f"{dir_prefix}/{out_fn}", "w") as fp:
                fp.write("".join(obj_lines))
            print(f"==> Exported textured obj to {dir_prefix}/{out_fn}")
            os.system(f"rm -f {dir_prefix}/{tmp_fn}")
            os.system(f"rm -f {dir_prefix}/*.txt")
        else:
            print("*********[WARNING]: mtl file not found...*************")
    else:
        print("************[ERROR]: in exporting!!!*************")


def wrap(args, out_fn):
    helper = diffcloth.makeOptimizeHelper(args.task_name)
    sim = diffcloth.makeSim(exampleName=args.task_name, runBackward=False)
    sim.forwardConvergenceThreshold = 1e-8
    pysim = pySim(sim, helper, True)

    # Reset the system
    sim.resetSystem()
    x0_t, v0_t, a0_t = get_state(sim, to_tensor=True)
    control_idx = sim.sceneConfig.customAttachmentVertexIdx[0][1]
    control_x0_t = [x0_t.reshape(-1, 3)[idx] for idx in control_idx]
    control_tgt = sum(control_x0_t)
    control_tgt[1] += 2
    control_tgt = torch.cat([control_tgt] * len(control_idx))

    # Cloth control
    # _ = forward_sim_no_control(x0_t, v0_t, a0_t, pysim, 200)
    # _ = forward_sim_rand_control(x0_t, v0_t, a0_t, pysim, 200)
    _ = forward_sim_targeted_control(x0_t, v0_t, a0_t, control_tgt, pysim, 200)
    # Stablise simulation
    x_t, v_t, a_t = get_state(sim, to_tensor=True)
    _ = forward_sim_no_control(x_t, v_t, a_t, pysim, 30)

    # Rendering the simulationg
    if args.render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=True)

    # Export final configuration into wavefront file
    if args.save:
        export_mesh(
            sim,
            obj_fn,
            export_step=sim.getStateInfo().stepIdx,
            renormalize=True,
            dir_prefix=f"output/{args.output_dir}",
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Perturb flat cloth")
    parser.add_argument(
        "-mode", type=int, default=1,
        help="-1: no change, 0: random perturb, 1: bezier purturb",
    )
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--task-name", type=str, default="wrap_dough")
    parser.add_argument("--n-openmp-thread", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="cloth_project/")
    parser.add_argument("--seed", type=int, default=8824325)
    args = parser.parse_args()
    
    diffcloth.enableOpenMP(n_threads=args.n_openmp_thread)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obj_fn = f"perturbed_cloth_{args.mode}.obj"
    wrap(args, obj_fn)