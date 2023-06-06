from typing import Any, Optional, Mapping, Tuple

import torch
import torch.autograd as autograd
from torch import Tensor

import diffcloth_py as diffcloth
import numpy as np
from numpy import linalg

from diffcloth_py import Simulation, ForwardInformation
from pathlib import Path
from IPython import embed


class SimFunction(autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: Tensor, v: Tensor, a: Tensor, cppSim: diffcloth.Simulation, helper: diffcloth.OptimizeHelper) -> Tuple[Tensor, Tensor]:
        ctx.helper = helper
        ctx.simulation = cppSim
        ctx.pastRecord = cppSim.getStateInfo()

        argX = np.float64(x)
        argV = np.float64(v.contiguous().detach().cpu().numpy())
        argA = np.float64(a.contiguous().detach().cpu().numpy())
        cppSim.stepNN(ctx.pastRecord.stepIdx + 1, argX, argV, argA)

        newRecord = cppSim.getStateInfo()
        ctx.newRecord = newRecord

        x_next = torch.as_tensor(newRecord.x).float()
        v_next = torch.as_tensor(newRecord.v).float()

        ctx.save_for_backward(x, v, a, x_next, v_next)

        return x_next, v_next

    @staticmethod
    def backward(ctx: Any, dL_dx_next: Tensor, dL_dv_next: Tensor) -> Tuple[ None, Tensor, Tensor, Tensor]:
        x, v, a, x_next, v_next = ctx.saved_tensors

        cppSim = ctx.simulation
        dL_dxnew_np = dL_dx_next.contiguous().detach().cpu().numpy()
        dL_dvnew_np = dL_dv_next.contiguous().detach().cpu().numpy()
        isLast = ctx.newRecord.stepIdx == cppSim.sceneConfig.stepNum

        if isLast:
            backRecord = cppSim.stepBackwardNN(
                ctx.helper.taskInfo,
                np.zeros_like(dL_dxnew_np),
                np.zeros_like(dL_dvnew_np),
                ctx.newRecord,
                ctx.newRecord.stepIdx == 1, # TODO: check if this should be 0 or 1
                dL_dxnew_np,
                dL_dvnew_np)
        else:
            backRecord = cppSim.stepBackwardNN(
                ctx.helper.taskInfo,
                dL_dxnew_np,
                dL_dvnew_np,
                ctx.newRecord,
                ctx.newRecord.stepIdx == 1, # TODO: check if this should be 0 or 1
                np.zeros_like(dL_dxnew_np),
                np.zeros_like(dL_dvnew_np))

        dL_dx = torch.as_tensor(backRecord.dL_dx)
        dL_dv = torch.as_tensor(backRecord.dL_dv)

        dL_da_norm = np.linalg.norm(backRecord.dL_dxfixed)
        if dL_da_norm > 1e-7:
            maxNorm = 4.0
            normalized = backRecord.dL_dxfixed * (max(min(backRecord.dL_dxfixed.shape[0] * maxNorm, dL_da_norm), 0.05) / dL_da_norm )
            dL_da = normalized
        else:
            dL_da = backRecord.dL_dxfixed
        dL_da = torch.as_tensor(dL_da)

        return dL_dx, dL_dv, dL_da, None, None


# Code from DiffClothAI
class SimFunction_cloth2cube(autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: Tensor, v: Tensor, a: Tensor, cppSim: diffcloth.Simulation, helper: diffcloth.OptimizeHelper) -> Tuple[Tensor, Tensor]:
        ctx.helper = helper
        ctx.simulation = cppSim
        ctx.pastRecord = cppSim.getStateInfo()

        argX = np.float64(x.contiguous().detach().cpu().numpy())
        argV = np.float64(v.contiguous().detach().cpu().numpy())
        argA = np.float64(a.contiguous().detach().cpu().numpy())
        cppSim.stepNN(ctx.pastRecord.stepIdx + 1, argX, argV, argA)

        newRecord = cppSim.getStateInfo()
        ctx.newRecord = newRecord

        x_next = torch.as_tensor(newRecord.x).float()
        v_next = torch.as_tensor(newRecord.v).float()

        ctx.save_for_backward(x, v, a, x_next, v_next)

        return x_next, v_next

    @staticmethod
    def backward(ctx: Any, dL_dx_next: Tensor, dL_dv_next: Tensor) -> Tuple[ None, Tensor, Tensor, Tensor]:
        x, v, a, x_next, v_next = ctx.saved_tensors

        path = Path.home() / 'DiffCloth' / 'cloth2cube' / 'gradient' / 'iter-{}'.format(ctx.newRecord.stepIdx)

        if path.exists():
            dL_dxload_np = np.load(path / 'dL_dx.npy').reshape(dL_dx_next.shape[0])
            dL_drload_np = - np.load(path / 'dL_dv.npy').reshape(dL_dv_next.shape[0])
            dL_dvload_np = dL_drload_np @ ctx.newRecord.M
        else:
            dL_dxload_np = np.zeros(dL_dx_next.shape[0])
            dL_dvload_np = np.zeros(dL_dv_next.shape[0])

        cppSim = ctx.simulation
        # dL_dxnew_np = dL_dx_next.contiguous().detach().cpu().numpy()
        # dL_dvnew_np = dL_dv_next.contiguous().detach().cpu().numpy()
        isLast = ctx.newRecord.stepIdx == cppSim.sceneConfig.stepNum

        if isLast:
            dL_dxnew_np = dL_dxload_np
            dL_dvnew_np = dL_dvload_np
            backRecord = cppSim.stepBackwardNN(
                ctx.helper.taskInfo,
                np.zeros_like(dL_dxnew_np),
                np.zeros_like(dL_dvnew_np),
                ctx.newRecord,
                ctx.newRecord.stepIdx == 1, # TODO: check if this should be 0 or 1
                dL_dxnew_np,
                dL_dvnew_np)
        else:
            dL_dxnew_np = dL_dx_next.contiguous().detach().cpu().numpy() + dL_dxload_np
            dL_dvnew_np = dL_dv_next.contiguous().detach().cpu().numpy() + dL_dvload_np
            backRecord = cppSim.stepBackwardNN(
                ctx.helper.taskInfo,
                dL_dxnew_np,
                dL_dvnew_np,
                ctx.newRecord,
                ctx.newRecord.stepIdx == 1, # TODO: check if this should be 0 or 1
                np.zeros_like(dL_dxnew_np),
                np.zeros_like(dL_dvnew_np))

        dL_dx = torch.as_tensor(backRecord.dL_dx)
        dL_dv = torch.as_tensor(backRecord.dL_dv)

        dL_da_norm = np.linalg.norm(backRecord.dL_dxfixed)
        if dL_da_norm > 1e-7:
            maxNorm = 4.0
            normalized = backRecord.dL_dxfixed * (max(min(backRecord.dL_dxfixed.shape[0] * maxNorm, dL_da_norm), 0.05) / dL_da_norm )
            dL_da = normalized
        else:
            dL_da = backRecord.dL_dxfixed
        dL_da = torch.as_tensor(dL_da)

        return dL_dx, dL_dv, dL_da, None, None


class SimFunction_cube2cloth(autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: Tensor, v: Tensor, a: Tensor, p_control: Tensor, cppSim: diffcloth.Simulation, helper: diffcloth.OptimizeHelper) -> Tuple[Tensor, Tensor]:
        ctx.helper = helper
        ctx.simulation = cppSim
        ctx.pastRecord = cppSim.getStateInfo()

        argX = np.float64(x.contiguous().detach().cpu().numpy())
        argV = np.float64(v.contiguous().detach().cpu().numpy())
        argA = np.float64(a.contiguous().detach().cpu().numpy())
        cppSim.stepNN(ctx.pastRecord.stepIdx + 1, argX, argV, argA)

        newRecord = cppSim.getStateInfo()
        ctx.newRecord = newRecord

        x_next = torch.as_tensor(newRecord.x).float()
        v_next = torch.as_tensor(newRecord.v).float()

        ctx.save_for_backward(a, p_control)

        return x_next, v_next

    @staticmethod
    def backward(ctx: Any, dL_dx_next: Tensor, dL_dv_next: Tensor) -> Tuple[None, Tensor, Tensor, Tensor, Tensor]:
        a, p_control = ctx.saved_tensors

        exp_path = Path.home() / 'DiffCloth' / 'cube2cloth'

        cppSim = ctx.simulation
        dL_dxnew_np = dL_dx_next.contiguous().detach().cpu().numpy()
        dL_dvnew_np = dL_dv_next.contiguous().detach().cpu().numpy()
        isLast = ctx.newRecord.stepIdx == cppSim.sceneConfig.stepNum

        if isLast:
            backRecord = cppSim.stepBackwardNN(
                ctx.helper.taskInfo,
                np.zeros_like(dL_dxnew_np),
                np.zeros_like(dL_dvnew_np),
                ctx.newRecord,
                ctx.newRecord.stepIdx == 1, # TODO: check if this should be 0 or 1
                dL_dxnew_np,
                dL_dvnew_np)
        else:
            backRecord = cppSim.stepBackwardNN(
                ctx.helper.taskInfo,
                dL_dxnew_np,
                dL_dvnew_np,
                ctx.newRecord,
                ctx.newRecord.stepIdx == 1, # TODO: check if this should be 0 or 1
                np.zeros_like(dL_dxnew_np),
                np.zeros_like(dL_dvnew_np))

        dL_dx = torch.as_tensor(backRecord.dL_dx)
        dL_dv = torch.as_tensor(backRecord.dL_dv)

        dL_da_norm = np.linalg.norm(backRecord.dL_dxfixed)
        if dL_da_norm > 1e-7:
            maxNorm = 4.0
            normalized = backRecord.dL_dxfixed * (max(min(backRecord.dL_dxfixed.shape[0] * maxNorm, dL_da_norm), 0.05) / dL_da_norm )
            dL_da = normalized
        else:
            dL_da = backRecord.dL_dxfixed
        dL_da = torch.as_tensor(dL_da)

        dL_dp_control = torch.zeros(p_control.shape, dtype=torch.float64)

        for i, info in enumerate(ctx.newRecord.collisionInfos[0][0]):
            if info.type == 1 or info.type == 2:
                dv_next_dp_control = torch.tensor(np.load(exp_path / 'dv_next_dp_control' / 'step-{}-info-{}.npy'.format(ctx.newRecord.stepIdx, i)))
                dL_dp_control = dL_dp_control + dv_next_dp_control.T @ (dL_dv_next[3*info.particleId : 3*info.particleId+3].double() + cppSim.sceneConfig.timeStep * dL_dx_next[3*info.particleId : 3*info.particleId+3].double())

        np.save(exp_path / 'dL_dx' / 'iter-{}.npy'.format(ctx.newRecord.stepIdx) , dL_dp_control.contiguous().detach().cpu().numpy())

        return dL_dx, dL_dv, dL_da, dL_dp_control, None, None