import diffcloth_py as diffcloth
import numpy as np
import utils, common, torch, os
from pySim.pySim import pySim
from pathlib import Path
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional
from clothNN import IndClosedController
from common import stages
from IPython import embed

def lossFunction(xvPairs):
    targetLoss_left, targetLoss_right, targetLoss = 0, 0, 0
    for i in range(4):
        x_i = xvPairs[step_frame*(i+1)][0]
        x_fixed1 = x_i[attachmentIdx[0]*3:attachmentIdx[0]*3+3]
        x_fixed2 = x_i[attachmentIdx[1]*3:attachmentIdx[1]*3+3]
        targetLoss_left += weight[i] * torch.nn.functional.smooth_l1_loss(x_fixed1, clip_left_target[i])
        targetLoss_right += weight[i] * torch.nn.functional.smooth_l1_loss(x_fixed2, clip_right_target[i])
    
    targetLoss = targetLoss_left + targetLoss_right
    targetLast = 50*torch.nn.functional.smooth_l1_loss(xvPairs[-1][0], targetshape_torch)
    succeed = targetLast < 1.0
    loss = {'succeed':succeed, 'total':targetLoss}
    print('[last-frame loss]:{:.5f}  [left-clip loss]:{:.5f}  [right-clip loss]:{:.5f}'.format(targetLast.item(),targetLoss_left.item(),targetLoss_right.item()))
    return loss


def simulateAndGetLoss(render=True):
    simulations = []
    lossTotal = 0.0
    losses = []
    print("[simulationAndGetLoss] running simulation...")
    sim.forwardConvergenceThreshold = 1e-8
    sim.resetSystem()
    xvPairs = common.forwardSimulation2(sim, x0_torch.clone(), v0_torch.clone(), a_torch.clone(), a_control, pySim)

    simulations.append(xvPairs)
    if render:
        diffcloth.render(sim, renderPosPairs=True, autoExit=False)
    loss = lossFunction(xvPairs)
    losses.append(loss)
    simulations.append(xvPairs)
    lossTotal += loss['total']
    return lossTotal, losses, simulations


def trainStep(sim, optimizer, render=True):
    loss = 0
    loss, _, _ = simulateAndGetLoss(render=render)
    optimizer.zero_grad()
    loss.backward()
    a_control.grad.data.clamp_(-10, 10)
    optimizer.step()
    return float(loss)


args = common.parseInputs()
train_resume, eval_mode = args.train_resume, args.eval

example = "hang_cloth"
dt_string = common.getTimeString()
expName = '{}-{}-{}'.format(dt_string, 'SGD', args.randSeed)
common.setRandomSeed(args.randSeed)

sim = diffcloth.makeSim("hang_cloth")
sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
print('[Step number]: ', sim.sceneConfig.stepNum)
np.set_printoptions(precision=5)

root_path = Path(__file__).resolve().parent
parent_path = root_path / 'experiments' / example 
exp_path = parent_path / expName
diffcloth.enableOpenMP(n_threads = 8)
helper = diffcloth.makeOptimizeHelper(example)
sim.forwardConvergenceThreshold =  1e-8

# sim.primitives[0].center = np.array([0., 0., 0.])
# sim.primitives[0].centerInit = np.array([0., 0., 0.])
sim.resetSystem()

pySim = pySim(sim, helper, True)
state_info_init = sim.getStateInfo()
ndof_u = sim.ndof_u
CLIP_INIT_POS = np.array(sim.getStateInfo().x_fixedpoints)
HANGER_CENTER_POS = sim.primitives[0].center.copy()
print('[CLIP_INIT_POS]:', CLIP_INIT_POS, '  [HANGER_CENTER_POS]:', HANGER_CENTER_POS)

x0, v0 = state_info_init.x, state_info_init.v
targetTranslation = helper.lossInfo.targetTranslation
print('[targetTranslation]:', targetTranslation)

target_clip_left = torch.tensor([[0,0.6,0], [-0.5,0.6,0], [-2,0.6,0], [-0.5,1,0]])
target_clip_right = torch.tensor([[-0.1,1.2,0], [-0.6,1.2,0], [-0.6,1.2,0], [-0.5,1,0]])
weight = [10, 2.5, 0.5, 0.1]
x0_torch, v0_torch, a_torch, targetshape_torch, CLIP_REST_DIST, clip_left, clip_right = common.getTorchVectors_hanger(x0, v0, CLIP_INIT_POS, targetTranslation)
clip_left_target, clip_right_target = target_clip_left+clip_left ,target_clip_right+clip_right
step_frame = sim.sceneConfig.stepNum // stages
print(clip_left_target)
print(clip_right_target)
print('step_frame:', step_frame)

attachmentIdx = sim.sceneConfig.customAttachmentVertexIdx[0][1]
trainMinLoss, trainBestEpoch, epochStart  = 10000, 0, 0

a_control = torch.tensor([
    # [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], 
    # [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0], 
    # [0,0,0, 0,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,0]
    # [0.2,0,-1, -0.6,0,-0.6], [1,0.3,0, -0.6,0,-0.6], [0.5,-0.4,-0.2, 0,-0.2,0], [0,-0.1,0.3, 0,0,0], [0,0,0, 0,0,0], 
    # [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0]
    [0.,0,0, 0.,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], 
    [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0], [0,0,0, 0,0,0]
],dtype=float, requires_grad=True)
lr = 1e-3
momentum = 0.7
optimizer = torch.optim.SGD([{'params':a_control, 'lr':lr}], momentum=momentum)

trainLosses = []

exp_path.mkdir(parents=True, exist_ok=True)
os.system("cp ClothHanger.py {}".format(exp_path / 'hangcloth_{}.py'.format(dt_string)))
os.system("cp utils.py {}".format(exp_path / 'utils_{}.py'.format(dt_string)))
os.system("cp common.py {}".format(exp_path / 'common_{}.py'.format(dt_string)))
configFile = open(exp_path / "config.txt", "a")
configFile.write("randSeed: {}\n".format(args.randSeed))
configFile.write("optimizer: {}\n".format('SGD'))
configFile.close()
logFile = open(exp_path / "log.txt", "a")

for epoch in range(epochStart, args.epochNum):
    logFile.write("Epoch {}\n".format(epoch))
    logFile.close()
    logFile = open(exp_path / "log.txt", "a")
    loss = trainStep(sim, optimizer, render= args.render)
    trainLosses.append(loss)
    if (loss < trainMinLoss):
        trainMinLoss, trainBestEpoch = loss, epoch
    utils.log("[Epoch {}] loss:{:.5f}  minLoss:{:.5f}  bestEpoch:{}\n".format(epoch, loss, trainMinLoss, trainBestEpoch), not(eval_mode), logFile)
    utils.plotLosses_hanger(trainLosses, exp_path)
    with open(exp_path / "a_control_param.txt", "a") as f:
        f.write(str(a_control)+"\n")
del sim