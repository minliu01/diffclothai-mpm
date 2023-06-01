import diffcloth_py as diffcloth
import numpy as np
import utils, common, torch, os
from pySim.pySim import pySim_cloth2cube
from pathlib import Path
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional
from clothNN import IndClosedController
from scipy.spatial.transform import Rotation as R
from IPython import embed


def getTorchVectors(x0, v0, fixedPointInitPoses):
    toDouble = False

    x0_torch = common.toTorchTensor(x0, False, toDouble)
    v0_torch = common.toTorchTensor(v0, False, toDouble)
    a_torch = common.toTorchTensor(fixedPointInitPoses, True, toDouble)

    return x0_torch, v0_torch, a_torch


def lossFunction(xvPairs):
    target = x0_torch + torch.full(x0_torch.shape, -10)
    targetLast = 50*torch.nn.functional.smooth_l1_loss(xvPairs[-1][0], target)
    return targetLast


def save_nimble(sim, i):
    record = sim.forwardRecords[i]

    path = output_path / 'iter-{}'.format(i)
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / 'x.npy', record.x.reshape(-1, 3))
    np.savetxt(path / 'cube_state.txt', np.concatenate([sim.primitives[0].center, sim.primitives[0].velocity]))

    responce = np.zeros((record.x.shape)).reshape(-1, 3)

    with open(path / "contact.txt", "a") as f:
        for info in record.collisionInfos[0][0]:
            f.write(str(info.particleId)+'\n')
            responce[info.particleId] = -info.r

    np.save(path / 'r.npy', responce)

    with open(path / "done.txt", "a") as f:
        pass


def load_nimble(sim, i):
    print("[load_nimble] begin waiting for simluation iter {}...".format(i))

    while True:
        if (input_path / 'done-{}.txt'.format(i)).exists():
            cube = sim.primitives[0]
            cube_state = np.loadtxt(input_path / 'iter-{}.txt'.format(i))

            cube.angular = cube_state[:3]
            cube.center = cube_state[3:6]
            cube.angularVelocity = cube_state[6:9]
            cube.velocity = cube_state[9:12]
            r = R.from_rotvec(cube_state[:3])
            rotation = r.as_matrix()
            cube.rotation_matrix = rotation
            cube.forwardRecords[-1] = cube.center
            cube.rotationRecords[-1] = cube.rotation_matrix

            break

    print("[load_nimble] end waiting for simluation iter {}...".format(i))


def steps_forward(sim, x_i, v_i, a_torch, a_control, simModule):
    records = []
    vMin, vMax= -0.1, 0.1

    step_frame = sim.sceneConfig.stepNum // stages

    for step in range(sim.sceneConfig.stepNum):
        session = step // step_frame
        records.append((x_i, v_i))
        controllerOut = a_control[session]
        torch.clamp(controllerOut, min=-1.0, max=1.0)
        delta_a_torch = (controllerOut + 1.) / 2. * (vMax - vMin) + vMin

        a_torch = a_torch + delta_a_torch
        x_i, v_i = simModule(x_i, v_i, a_torch)

        save_nimble(sim, step+1)
        load_nimble(sim, step+1)

    with open(exp_path / "done.txt", "a") as f:
        pass

    records.append((x_i, v_i))
    return records


def simulateAndGetLoss(render=True):
    simulations = []
    lossTotal = 0.

    print("[simulationAndGetLoss] running simulation...")
    sim.forwardConvergenceThreshold = 1e-8
    sim.resetSystem()

    xvPairs = steps_forward(sim, x0_torch.clone(), v0_torch.clone(), a_torch.clone(), a_control, pySim)
    simulations.append(xvPairs)

    if render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=False)

    loss = lossFunction(xvPairs)
    lossTotal += loss

    return lossTotal, simulations


def trainStep(sim, optimizer, render=True):
    loss = 0.
    loss, _ = simulateAndGetLoss(render=render)

    optimizer.zero_grad()

    while True:
        if (exp_path / 'gradient' / 'done.txt').exists():
            break

    loss.backward()
    a_control.grad.data.clamp_(-10, 10)
    optimizer.step()

    return float(loss)


args = common.parseInputs()

common.setRandomSeed(args.randSeed)
sim = diffcloth.makeSim("hang_cloth")
print('[Step number]: ', sim.sceneConfig.stepNum)
np.set_printoptions(precision=5)

exp_path = Path.home() / 'DiffCloth' / 'cloth2cube'
input_path = exp_path / 'input_state'
output_path = exp_path / 'ouput_state'
gradient_path = exp_path / 'gradient'

diffcloth.enableOpenMP(n_threads = 8)
helper = diffcloth.makeOptimizeHelper("hang_cloth")
sim.forwardConvergenceThreshold =  1e-8

sim.primitives[0].centerInit = np.array([0, -1.5, -2])

sim.resetSystem()
pySim = pySim_cloth2cube(sim, helper, True)

state_info_init = sim.getStateInfo()
CLIP_INIT_POS = np.array(state_info_init.x_fixedpoints)
x0, v0 = state_info_init.x, state_info_init.v
x0_torch, v0_torch, a_torch = getTorchVectors(x0, v0, CLIP_INIT_POS)

stages = 10
step_frame = sim.sceneConfig.stepNum // stages
print('step_frame:', step_frame)

az = -0.2
a_control = torch.tensor([[0,0,az, 0,0,az], [0,0,az, 0,0,az], [0,0,az, 0,0,az], [0,0,az, 0,0,az], [0,0,az, 0,0,az], 
    [0,0,az, 0,0,az], [0,0,az, 0,0,az], [0,0,az, 0,0,az], [0,0,az, 0,0,az], [0,0,az, 0,0,az]], dtype=float, requires_grad=True)
# a_control = torch.tensor(np.loadtxt(Path.home() / 'DiffCloth' / 'cloth2cube-2' / '5-control.txt'), dtype=float, requires_grad=True)


lr = 1e-4
momentum = 0.7
optimizer = torch.optim.SGD([{'params':a_control, 'lr':lr}], momentum=momentum)

os.system("rm -rf {}".format(exp_path))
exp_path.mkdir(parents=True, exist_ok=True)
os.system("cp cloth2cube.py {}".format(exp_path / 'cloth2cube.py'))
input_path.mkdir(parents=True, exist_ok=True)
output_path.mkdir(parents=True, exist_ok=True)

trainLosses = []
cube_positions = []

for epoch in range(100):
    print('[epoch] {}'.format(epoch+1))

    loss = trainStep(sim, optimizer, args.render)

    trainLosses.append((sim.primitives[0].center[2]+7)**2)
    cube_positions.append(sim.primitives[0].center[2])

    utils.plotLosses_hanger(trainLosses, exp_path)
    utils.plotCubePos(cube_positions, exp_path)
    np.savetxt(exp_path / '{}-control.txt'.format(epoch+1), a_control.contiguous().detach().cpu().numpy())

    os.system("rm -rf {}".format(input_path))
    os.system("rm -rf {}".format(output_path))
    os.system("rm -rf {}".format(gradient_path))
    os.system("rm {}/done.txt".format(exp_path))

    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savetxt(exp_path / 'epoch-{}.txt'.format(epoch+1), sim.primitives[0].center)

del sim