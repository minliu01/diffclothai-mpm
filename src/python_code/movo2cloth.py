import diffcloth_py as diffcloth
import numpy as np
import utils, common, torch, os
import time
from pySim.pySim import pySim_cube2cloth
from pathlib import Path
import numpy as np
import torch.optim
import torch.nn.functional as F
import torch.autograd as autograd
from common import stages
from scipy.spatial.transform import Rotation as R
from IPython import embed


def getTorchVectors(x0, v0, fixedPointInitPoses):
    toDouble = False

    x0_torch = common.toTorchTensor(x0, False, toDouble)
    v0_torch = common.toTorchTensor(v0, False, toDouble)
    a_torch = common.toTorchTensor(fixedPointInitPoses, True, toDouble)

    return x0_torch, v0_torch, a_torch


def lossFunction(xvPairs):
    target1 = torch.full((x0_torch.shape[0]//3, ), -0.6)
    target2 = torch.full((x0_torch.shape[0]//3, ), 0.1)
    targetLast = 1e3*(F.smooth_l1_loss(xvPairs[-1][0].reshape(-1,3)[:,0],target1)+F.smooth_l1_loss(xvPairs[-1][0].reshape(-1,3)[:,1],target2))
    return targetLast


def steps_forward(sim, x_i, v_i, a_torch, a_control, simModule):
    records = []
    vMin, vMax= -0.1, 0.1

    step_frame = sim.sceneConfig.stepNum // stages

    read_state_time = 0.0
    pure_forward_time = 0.0
    jacobian_time = 0.0

    for step in range(sim.sceneConfig.stepNum):
        session = step // step_frame
        records.append((x_i, v_i))
        controllerOut = a_control[session]

        T1 = time.time()
        pos_vel = np.loadtxt(cur_state_path / ('iter-{}.txt'.format(step+1)))
        for i in range(PRIMITIVE_NUM):
            sim.primitives[i].angular = pos_vel[i*6 : i*6+3]
            sim.primitives[i].center = pos_vel[i*6+3 : i*6+6]
            sim.primitives[i].angularVelocity = pos_vel[(PRIMITIVE_NUM+i)*6 : (PRIMITIVE_NUM+i)*6+3]
            sim.primitives[i].velocity = pos_vel[(PRIMITIVE_NUM+i)*6+3 : (PRIMITIVE_NUM+i)*6+6]
            r = R.from_rotvec(pos_vel[i*6 : i*6+3])
            rotation = r.as_matrix()
            sim.primitives[i].rotation_matrix = rotation
        T2 = time.time()
        read_state_time += (T2-T1)

        torch.clamp(controllerOut, min=-1.0, max=1.0)
        delta_a_torch = (controllerOut + 1.) / 2. * (vMax - vMin) + vMin
        a_torch = a_torch + delta_a_torch

        T1 = time.time()
        x_i, v_i = simModule(x_i, v_i, a_torch, torch.tensor(pos_vel))
        T2 = time.time()
        pure_forward_time += (T2-T1)

        records.append((x_i, v_i))

        T1 = time.time()
        for i in range(len(sim.forwardRecords[step+1].collisionInfos[0][0])):
            info = sim.forwardRecords[-1].collisionInfos[0][0][i]
            if info.type == 1 or info.type == 2:
                jacobian = calculate_gradient(pos_vel, info, sim.primitives[0].mu)
                np.save(jacobian_path / 'step-{}-info-{}.npy'.format(step+1, i), jacobian)
        T2 = time.time()
        jacobian_time += (T2-T1)

    if args.time:
        print('\033[32m[read_state_time] {}s\033[0m'.format(read_state_time))
        print('\033[32m[pure_forward_time] {}s\033[0m'.format(pure_forward_time))
        print('\033[32m[jacobian_time] {}s\033[0m'.format(jacobian_time))

    return records


def calculate_gradient(p, info, mu):
    p = torch.tensor(p, requires_grad=True)
    normal = torch.tensor(info.normal)
    r_ = torch.tensor(info.contact_point) - p[info.primitiveId*6+3 : info.primitiveId*6+6]

    if info.type == 1:
        r_i = p[(info.primitiveId+PRIMITIVE_NUM)*6+3:(info.primitiveId+PRIMITIVE_NUM)*6+6] + torch.cross(p[(info.primitiveId+PRIMITIVE_NUM)*6:(info.primitiveId+PRIMITIVE_NUM)*6+3], r_)
    elif info.type == 2:
        v_out = p[(info.primitiveId+PRIMITIVE_NUM)*6+3:(info.primitiveId+PRIMITIVE_NUM)*6+6] + torch.cross(p[(info.primitiveId+PRIMITIVE_NUM)*6:(info.primitiveId+PRIMITIVE_NUM)*6+3], r_)
        s = - v_out @ normal
        f_n = s * normal
        f_t = - v_out - f_n
        r_i = - f_n - mu * torch.abs(s) * torch.nn.functional.normalize(f_t, dim=0)

    dr0_dp = autograd.grad(r_i[0], p, retain_graph=True)[0]
    dr1_dp = autograd.grad(r_i[1], p, retain_graph=True)[0]
    dr2_dp = autograd.grad(r_i[2], p, retain_graph=True)[0]
    jacobian = torch.vstack((dr0_dp, dr1_dp, dr2_dp)).contiguous().detach().cpu().numpy()
    return jacobian


def simulateAndGetLoss(render=True):
    T1 = time.time()

    simulations = []
    lossTotal = 0.

    sim.forwardConvergenceThreshold = 1e-8
    sim.resetSystem()

    xvPairs = steps_forward(sim, x0_torch.clone(), v0_torch.clone(), a_torch.clone(), a_control, pySim_cube2cloth)
    simulations.append(xvPairs)

    T2 = time.time()
    print('\033[32m[forward] {}s\033[0m'.format(T2-T1))

    if render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=False)

    loss = lossFunction(xvPairs)
    lossTotal += loss

    return lossTotal, simulations


def trainStep(sim, optimizer, render=True):
    loss = 0.
    loss, _ = simulateAndGetLoss(render=render)

    print('[DiffCloth] epoch-{} backward'.format(epoch+1))

    T1 = time.time()
    optimizer.zero_grad()
    loss.backward()
    T2 = time.time()
    print('\033[32m[backward] {}s\033[0m'.format(T2-T1))

    with open(gradient_path / 'done.txt', 'a') as f:
        pass

    return float(loss)


args = common.parseInputs()

common.setRandomSeed(args.randSeed)
sim = diffcloth.makeSim("hang_cloth")
print('[Step number]: ', sim.sceneConfig.stepNum)
PRIMITIVE_NUM = len(sim.primitives)
np.set_printoptions(precision=5)

exp_path = Path.home() / 'DiffCloth' / 'cube2cloth'
jacobian_path = exp_path / 'dv_next_dp_control'
state_path = exp_path / 'state'
gradient_path = exp_path / 'dL_dx'

jacobian_path.mkdir(parents=True, exist_ok=True)

diffcloth.enableOpenMP(n_threads = 8)
helper = diffcloth.makeOptimizeHelper("hang_cloth")
sim.forwardConvergenceThreshold =  1e-8

sim.resetSystem()
pySim_cube2cloth = pySim_cube2cloth(sim, helper, True)

state_info_init = sim.getStateInfo()
CLIP_INIT_POS = np.array(state_info_init.x_fixedpoints)
x0, v0 = state_info_init.x, state_info_init.v
x0_torch, v0_torch, a_torch = getTorchVectors(x0, v0, CLIP_INIT_POS)

step_frame = sim.sceneConfig.stepNum // stages
print('step_frame:', step_frame)

a_control = torch.tensor([[0.,0,0, 0.,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], 
        [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0]], dtype=float, requires_grad=True)

lr = 5e-4
momentum = 0.7
print('learning rate: {}, momentum: {}'.format(lr, momentum))
optimizer = torch.optim.SGD([{'params':a_control, 'lr':lr}], momentum=momentum)

trainLosses = []
best_loss = 1e8
best_epoch = 0

for i in range(PRIMITIVE_NUM):
    sim.primitives[i].isInCollision = False

for epoch in range(args.epochNum):
    cur_state_path = state_path / ('epoch-{}'.format(epoch+1))
    while True:
        if (cur_state_path / 'done.txt').exists():
            break

    print('[epoch-{}]: forward'.format(epoch+1))

    pos_vel = np.loadtxt(cur_state_path / ('iter-0.txt'))
    for i in range(PRIMITIVE_NUM):
        sim.primitives[i].angularInit = pos_vel[i*6 : i*6+3]
        sim.primitives[i].centerInit = pos_vel[i*6+3 : i*6+6]
        sim.primitives[i].angularVelocityInit = pos_vel[(PRIMITIVE_NUM+i)*6 : (PRIMITIVE_NUM+i)*6+3]
        sim.primitives[i].velocityInit = pos_vel[(PRIMITIVE_NUM+i)*6+3 : (PRIMITIVE_NUM+i)*6+6]
        r = R.from_rotvec(pos_vel[i*6 : i*6+3])
        rotation = r.as_matrix()
        sim.primitives[i].rotation_matrixInit = rotation

    loss = trainStep(sim, optimizer, args.render)

    if loss < best_loss:
        best_loss = loss
        best_epoch = epoch + 1

    print('\033[31m[epoch-{}]: [loss] {}, [best epoch] {}, [best loss] {}\033[0m'.format(epoch+1, loss, best_epoch, best_loss))
    trainLosses.append(loss)
    utils.plotLosses_hanger(trainLosses, exp_path)

    os.system("rm -rf {}".format(jacobian_path))
    jacobian_path.mkdir(parents=True, exist_ok=True)

del sim