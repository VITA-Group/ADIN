from __future__ import print_function, division

import math
import numpy as np
import torch.nn.functional as F
from baseline.training.functions import *
from adin.models.multitaskModels import *
from adin.models.multitaskModelsExt import *
from torch.autograd import Variable

from baseline.datasetUtils.datasetStat import labelDist


# Get label distribution
# ---------------------------
def getLableDist(use_dataset, use_loss):
    if use_loss == "classCamId":
        return labelDist[use_dataset]['camId']
    elif use_loss == "classTimeStamps":
        return labelDist[use_dataset]['ts']
    elif use_loss == 'classCamIdAndTimeStamp':
        res = np.array(labelDist[use_dataset]['comb']).T.tolist()
        return sum(res, [])
    else:
        raise Exception("unknown task", use_loss)


# Define fused model
# ---------------------------
def getModelFused(use_dataset, use_model, use_loss, use_gpu):
    numPidClass = getNumTrainClass(use_dataset, use_loss[0])
    numEnvClass = getNumTrainClass(use_dataset, use_loss[1])

    if use_model == "resnet":
        model = ft_resnet50_dual(numPidClass, numEnvClass)
    elif use_model == "resNet101":
        model = ft_resnet101_dual(numPidClass, numEnvClass)
    elif use_model == "resNet152":
        model = ft_resnet152_dual(numPidClass, numEnvClass)
    elif use_model == "densenet":
        model = ft_densenet121_dual(numPidClass, numEnvClass)
    elif use_model == "denseNet169":
        model = ft_densenet169_dual(numPidClass, numEnvClass)
    elif use_model == "denseNet201":
        model = ft_densenet201_dual(numPidClass, numEnvClass)
    elif use_model == "denseNet161":
        model = ft_densenet161_dual(numPidClass, numEnvClass)
    elif use_model == "multibranch1":
        model = ft_resnet50_np_2_model1_dual(numPidClass, numEnvClass)
    elif use_model == "multibranch2":
        model = ft_resnet50_np_2_model2_dual(numPidClass, numEnvClass)

    else:
        raise Exception('unknown model')

    return model.cuda() if use_gpu else model


def pidLoss(usePosLoss, outputs, labels):
    if usePosLoss == "XE":
        return F.cross_entropy(outputs, labels)


def advLoss(useNegLoss, outputs, labels, labelDist):
    def isNaN(x):
        return int(x.data != x.data)

    def kldiv(curr, target):
        return -sum(sum((target * torch.log(curr / target)))) / outputs.shape[0] / outputs.shape[1]

    def weighted_kldiv(curr, target, weight):
        return -sum(sum((target * torch.log(curr / target)) * weight)) / outputs.shape[0] / outputs.shape[1]

    def focal_kldiv(curr, target, weight):
        return sum(-torch.sum(target * torch.log(curr / target), dim=1) * weight) / outputs.shape[0] / outputs.shape[1]

    if useNegLoss == 'entropy':
        uniform_dist = Variable(torch.Tensor([[float(1 / outputs.shape[1])] * outputs.shape[1]])).cuda()
        loss_1 = sum(sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)))
        loss_2 = - sum(sum(F.softmax(outputs, dim=1) * torch.log(uniform_dist)))
        loss = (0 if isNaN(loss_1) else loss_1) + (0 if isNaN(loss_2) else loss_2)
        return 1.0 * loss / outputs.shape[0] / outputs.shape[1]


    elif useNegLoss == 'KLdiv':
        loss = -sum(sum(F.log_softmax(outputs, dim=1))) / outputs.shape[1]
        return 1.0 * loss / outputs.shape[0] / outputs.shape[1]

    elif useNegLoss == 'JSD':
        uniform_dist = torch.Tensor([[float(1 / outputs.shape[1])] * outputs.shape[1]])
        uniform_dist = Variable(uniform_dist * outputs.shape[0]).cuda()
        m = 0.5 * (F.softmax(outputs, dim=1) + uniform_dist)
        return kldiv(F.softmax(outputs), m) + kldiv(uniform_dist, m)

    elif useNegLoss == 'recReverse':
        rec_reverse = -F.cross_entropy(outputs, labels, reduce=False) * (torch.max(outputs, 1)[1] != labels).float()
        return torch.sum(rec_reverse) / len(rec_reverse)

    elif useNegLoss == 'wtEntropy':
        target_dist = Variable(torch.Tensor([float(n / sum(labelDist)) for n in labelDist])).cuda()

        loss_1 = sum(sum(F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)) * target_dist)
        loss_2 = sum(sum(F.softmax(outputs, dim=1) * target_dist)) * math.log(outputs.shape[1])
        loss = (0 if isNaN(loss_1) else loss_1) + (0 if isNaN(loss_2) else loss_2)
        return 1.0 * loss / outputs.shape[0] / outputs.shape[1]

    elif useNegLoss == 'wtKLdiv':
        target_dist = Variable(torch.Tensor([float(n / sum(labelDist)) for n in labelDist])).cuda()
        loss = -sum(sum(F.log_softmax(outputs, dim=1)) * target_dist) / outputs.shape[1]
        return 1.0 * loss / outputs.shape[0] / outputs.shape[1]

    elif useNegLoss == 'wtJSD':
        target_dist = Variable(torch.Tensor([float(n / sum(labelDist)) for n in labelDist])).cuda()
        uniform_dist = torch.Tensor([[float(1 / outputs.shape[1])] * outputs.shape[1]])
        uniform_dist = Variable(uniform_dist * outputs.shape[0]).cuda()
        m = 0.5 * (F.softmax(outputs, dim=1) + uniform_dist)
        return weighted_kldiv(F.softmax(outputs), m, target_dist) + weighted_kldiv(uniform_dist, m, target_dist)

    elif useNegLoss == 'focalReverse':
        target_dist = Variable(torch.Tensor([float(labelDist[n] / sum(labelDist)) for n in labels.data])).cuda()
        return sum(-F.cross_entropy(outputs, labels, reduce=False) * target_dist) / len(labels)

    elif useNegLoss == 'focalEntropy':
        target_dist = Variable(torch.Tensor([float(labelDist[n] / sum(labelDist)) for n in labels.data])).cuda()
        uniform_dist = Variable(torch.Tensor([[float(1 / outputs.shape[1])] * outputs.shape[1]])).cuda()
        return focal_kldiv(uniform_dist, F.softmax(outputs), target_dist)

    elif useNegLoss == 'focalKLdiv':
        target_dist = Variable(torch.Tensor([float(labelDist[n] / sum(labelDist)) for n in labels.data])).cuda()
        uniform_dist = Variable(torch.Tensor([[float(1 / outputs.shape[1])] * outputs.shape[1]])).cuda()
        return focal_kldiv(F.softmax(outputs), uniform_dist, target_dist)

    elif useNegLoss == 'focalJSD':
        target_dist = Variable(torch.Tensor([float(labelDist[n] / sum(labelDist)) for n in labels.data])).cuda()
        uniform_dist = torch.Tensor([[float(1 / outputs.shape[1])] * outputs.shape[1]])
        uniform_dist = Variable(uniform_dist * outputs.shape[0]).cuda()
        m = 0.5 * (F.softmax(outputs, dim=1) + uniform_dist)
        return focal_kldiv(F.softmax(outputs), m, target_dist) + focal_kldiv(uniform_dist, m, target_dist)

    else:
        return
