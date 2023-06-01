import diffcloth_py as diffcloth
import numpy as np
import utils, common, torch, os
from pySim.pySim import pySim
from pathlib import Path
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional
from torch.optim import Adam
from clothNN import IndClosedController
import IPython

'''def lossFunction(xvPairs):
    stretchPenalty = 0
    for (i,(x_i,v_i)) in enumerate(xvPairs):
        x_fixed1 = x_i[attachmentIdx[0]*3:attachmentIdx[0]*3+3]
        x_fixed2 = x_i[attachmentIdx[1]*3:attachmentIdx[1]*3+3]
        clip_dist = torch.linalg.norm(x_fixed2-x_fixed1)
        stretchPenalty += torch.clamp(torch.abs(clip_dist-CLIP_REST_DIST)-2.0, min=0.0, max=None) * 0.3
    targetLoss, targetLast = 0, 0
    for (x_last,v_i) in xvPairs:
        targetLoss += torch.nn.functional.smooth_l1_loss(x_last, targetshape_torch)
    targetFirst = torch.nn.functional.smooth_l1_loss(xvPairs[0][0], targetshape_torch)
    targetLast = torch.nn.functional.smooth_l1_loss(xvPairs[-1][0], targetshape_torch)
    succeed = targetLast < 1.0
    loss = {'succeed':succeed, 'target':targetLoss, 'stretch':stretchPenalty, 'total':targetLoss+stretchPenalty}
    print('[stretch loss]:{:.5f}   [first-frame loss]:{:.5f}   [last-frame loss]:{:.5f}'.format(loss['stretch'].item(),targetFirst.item(),targetLast.item()))
    return loss'''

def lossFunction(xvPairs):
    targetLoss_left, targetLoss_right, targetLoss = 0, 0, 0
    for i in range(4):
        x_i = xvPairs[step_frame*(i+1)][0]
        x_fixed1 = x_i[attachmentIdx[0]*3:attachmentIdx[0]*3+3]
        x_fixed2 = x_i[attachmentIdx[1]*3:attachmentIdx[1]*3+3]
        targetLoss_left += 500*weight[i]*torch.nn.functional.smooth_l1_loss(x_fixed1, clip_left_target[i])
        targetLoss_right += 500*weight[i]*torch.nn.functional.smooth_l1_loss(x_fixed2, clip_right_target[i])
    
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
    xvPairs = common.forwardSimulation(sim, x0_torch.clone(), v0_torch.clone(), a_torch.clone(), getState, controller, pySim)
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
    nn.utils.clip_grad_norm_(controller.parameters(), 1.0)
    optimizer.step()
    return float(loss)


def getValidationLosses(epoch, expName, render=True, saveImage=True, saveSimulation=True):
    sim.forwardConvergenceThreshold = 1e-6
    hangerPrim = sim.primitives[0]
    hangerVec = hangerPrim.getPointVec()
    lossAvg, losses, simulations = simulateAndGetLoss(render=render)
    for (i, (loss, xvPairs)) in enumerate(zip(losses, simulations)):
        clothVec, clothVecInit = xvPairs[-1][0].detach().numpy(), xvPairs[0][0].detach().numpy()
        vecStack = np.concatenate([clothVec, clothVecInit], axis = 0)
        if saveImage:
            Path(root_path / 'evalImages' / expName).mkdir(parents=True, exist_ok=True)
            identifier = 'epoch-{:d}-loss-{:.2f}'.format(epoch, loss['total'].item())
            savePath = root_path / 'evalImages' / expName / identifier 
            print("saving image to... {}".format(savePath))
            utils.plotPointCloudFromVecs([vecStack, hangerVec], identifier, save=saveImage, path=savePath)
        if saveSimulation:
            savePath = expName + "_eval"
            print("saving simulation to output/{}".format(savePath))
            sim.exportCurrentSimulation(savePath)
    sim.forwardConvergenceThreshold = 1e-8
    return lossAvg


def getState(x, v):
    HANGER_CENTER_POS = common.toTorchTensor(sim.primitives[0].center.copy(), False, False)
    state = [x]
    v_mean = v.reshape(-1, 3).mean(axis=0)
    state.append(v_mean)
    state.append(HANGER_CENTER_POS)
    for idx in attachmentIdx:
        state.append(x[idx*3:idx*3+3])
    return torch.cat(state).float().unsqueeze(0)


def saveEpoch(isBestTrain):
    ckpt = {'epoch': epoch,
            'trainMinLoss': trainMinLoss,
            'trainLosses' : trainLosses,
            'trainBestEpoch' : trainBestEpoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'controller_state_dict': controller.state_dict(),}
    if isBestTrain:
        torch.save(ckpt, exp_path / 'trainBestEpoch.pth')
    else:
        torch.save(ckpt, exp_path / '{}.pth'.format(epoch))


def loadCheckpoint(path, epoch):
    global controller, optimizer
    checkpoint = torch.load(path  / epoch)
    controller.load_state_dict(checkpoint['controller_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochStart, trainMinLoss, trainLosses = checkpoint['epoch'], checkpoint['trainMinLoss'], checkpoint['trainLosses']
    trainBestEpoch = checkpoint['trainBestEpoch']
    print("Loaded checkpoint from epoch {}, trainMinLoss is {}".format(epochStart, trainMinLoss))
    return epochStart, trainMinLoss, trainLosses, trainBestEpoch


args = common.parseInputs()
train_resume, eval_mode = args.train_resume, args.eval

# Experiment
example = "hang_cloth"
if args.eval or args.train_resume:
    expName = args.load_expname
    loadEpoch = args.load_epoch
else:
    dt_string = common.getTimeString()
    expName = '{}-{}-{}'.format(dt_string, 'Adam', args.randSeed)
    common.setRandomSeed(args.randSeed)

# DiffSimulation Settings
sim = diffcloth.makeSim("hang_cloth")
sim.gradientClippingThreshold, sim.gradientClipping = 100.0, False
print('[Step number]: ', sim.sceneConfig.stepNum)
np.set_printoptions(precision=5)

root_path = Path(__file__).resolve().parent
parent_path = root_path / 'experiments' / example 
exp_path = parent_path / expName
diffcloth.enableOpenMP(n_threads = 8)
helper = diffcloth.makeOptimizeHelper(example)
# forwardConvergence needs to be reset after helper is made
sim.forwardConvergenceThreshold =  1e-8

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
target_clip_left = torch.tensor([[-1,0,0], [-1,-1.2,0], [0.6,-1.2,0], [0.8,-1.5,0]])
target_clip_right = torch.tensor([[-0.9,0,0], [-0.9,-0.3,0], [0.6,-0.3,0], [0.8,-1.5,0]])
weight = [1., 0.7, 0.4, 0.1]
x0_torch, v0_torch, a_torch, targetshape_torch, CLIP_REST_DIST, clip_left, clip_right = common.getTorchVectors_hanger(x0, v0, CLIP_INIT_POS, targetTranslation)
clip_left_target, clip_right_target = target_clip_left+clip_left ,target_clip_right+clip_right
step_frame = sim.sceneConfig.stepNum // 4
print(clip_left_target)
print(clip_right_target)
print('step_frame:', step_frame)

attachmentIdx = sim.sceneConfig.customAttachmentVertexIdx[0][1]
trainMinLoss, trainBestEpoch, epochStart  = 10000, 0, 0
controller = IndClosedController(sim, helper, [getState(x0_torch, v0_torch).size(1), 64, 64, ndof_u], dropout=0.0)
controller.reset_parameters(nn.init.calculate_gain('tanh'), 0.001)
optimizer = optim.Adam(controller.parameters(), lr=1e-4*2, weight_decay=0)

trainLosses = []

if not(eval_mode):
    # Train
    if not(train_resume):
        exp_path.mkdir(parents=True, exist_ok=True)
        os.system("cp ClothHanger.py {}".format(exp_path / 'hangcloth_nn_{}.py'.format(dt_string)))
        os.system("cp utils.py {}".format(exp_path / 'utils_{}.py'.format(dt_string)))
        os.system("cp common.py {}".format(exp_path / 'common_{}.py'.format(dt_string)))
        configFile = open(exp_path / "config.txt", "a")
        configFile.write("randSeed: {}\n".format(args.randSeed))
        configFile.write("optimizer: {}\n".format('Adam'))
        configFile.close()
    else:
        epochStart, trainMinLoss, trainLosses, trainBestEpoch = loadCheckpoint(exp_path, '{}.pth'.format(loadEpoch))
    logFile = open(exp_path / "log.txt", "a")

    for epoch in range(epochStart, args.epochNum):
        logFile.write("Epoch {}\n".format(epoch))
        logFile.close()
        logFile = open(exp_path / "log.txt", "a")
        loss = trainStep(sim, optimizer, render= False)
        trainLosses.append(loss)
        if (loss < trainMinLoss):
            trainMinLoss, trainBestEpoch = loss, epoch
            saveEpoch(isBestTrain=True)
        utils.log("[Epoch {}] loss:{:.5f}  minLoss:{:.5f}  bestEpoch:{}  norm:{:.5f}\n".format(epoch, loss, trainMinLoss, trainBestEpoch, nn.utils.clip_grad_norm_(controller.parameters(), 1.0)), not(eval_mode), logFile)
        utils.plotLosses_hanger(trainLosses, exp_path)
        saveEpoch(isBestTrain=False)

else:
    # Eval
    epochStart, trainMinLoss, trainLosses, trainBestEpoch = loadCheckpoint(exp_path, '{}.pth'.format(loadEpoch))
    evalLoss = getValidationLosses(epochStart, expName, render=args.render, saveImage=True, saveSimulation=True)

del sim