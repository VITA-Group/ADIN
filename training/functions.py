from __future__ import print_function, division

from models.models import *
from models.modelsExt import *
from datasetUtils.datasetStat import Nclass, NcamId


# Define model
# ---------------------------
def getModel(use_dataset, use_model, use_loss, use_gpu):
    numClass = getNumTrainClass(use_dataset, use_loss)

    if use_model == "midnet":
        model = ft_resnet50mid(numClass)
    elif use_model == "resnet":
        model = ft_resnet50(numClass)
    elif use_model == "resNet101":
        model = ft_resnet101(numClass)
    elif use_model == "resNet152":
        model = ft_resnet152(numClass)
    elif use_model == "densenet":
        model = ft_densenet121(numClass)
    elif use_model == "denseNet169":
        model = ft_densenet169(numClass)
    elif use_model == "denseNet201":
        model = ft_densenet201(numClass)
    elif use_model == "denseNet161":
        model = ft_densenet161(numClass)
    elif use_model == "multibranch1":
        model = ft_resnet50_np_2_model1(numClass)
    elif use_model == "multibranch2":
        model = ft_resnet50_np_2_model2(numClass)
    else:
        raise Exception('unknown model')

    return model.cuda() if use_gpu else model


# Get numTrainClass
# ---------------------------
def getNumTrainClass(use_dataset, use_loss):
    if use_loss == "crossEntropy":
        numTrainClass = Nclass[use_dataset]['train']
    elif use_loss == "classCamId":
        numTrainClass = NcamId[use_dataset]
    elif use_loss == "classTimeStamps":
        numTrainClass = 3
    elif use_loss == "classCamIdAndTimeStamp":
        numTrainClass = NcamId[use_dataset] * 3
    else:
        raise Exception("unknown loss", use_loss)

    return numTrainClass


# Get labels
# ---------------------------
def getLable(sample, use_loss, use_gpu):
    if use_loss in ["crossEntropy"]:
        labels = Variable(sample['label']).cuda() if use_gpu else Variable(sample['label'])
    elif use_loss == "classCamId":
        camIds = torch.LongTensor([c - 1 for c in sample['camId']])
        labels = Variable(camIds).cuda() if use_gpu else Variable(camIds)
    elif use_loss == "classTimeStamps":
        index_ts = {'morning': 0, 'noon': 1, 'afternoon': 2}
        timestamps = torch.LongTensor([index_ts[t] for t in sample['ts']])
        labels = Variable(timestamps).cuda() if use_gpu else Variable(timestamps)
    elif use_loss == 'classCamIdAndTimeStamp':
        index_ts = {'morning': 0, 'noon': 1, 'afternoon': 2}
        camIdTs = [(sample['camId'][n] - 1) * 3 + index_ts[sample['ts'][n]] for n in range(len(sample['camId']))]
        camIdTs = torch.LongTensor(camIdTs)
        labels = Variable(camIdTs).cuda() if use_gpu else Variable(camIdTs)
    else:
        raise Exception("unknown task", use_loss)

    return labels
