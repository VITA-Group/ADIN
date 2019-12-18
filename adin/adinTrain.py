from __future__ import print_function, division
import os, sys
from torch.optim import lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())

from adin.config.adinConfig import *
from argument import *

args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
name = '%s_%s_%s_%s' % (args.dataset, args.model, args.loss, timestamp)
print(name + '\n')

os.makedirs(args.log_dir, exist_ok=True)
args.log_dir = os.path.join(args.log_dir, args.dataset, name)
os.makedirs(args.log_dir, exist_ok=True)

os.makedirs(args.tfboard_dir, exist_ok=True)
args.tfboard_dir = os.path.join(args.tfboard_dir, args.dataset, name)
os.makedirs(args.tfboard_dir, exist_ok=True)

# define writer
writer = SummaryWriter(log_dir=args.tfboard_dir)
use_gpu = True

######################################################################
model = getModelFused(args.dataset, args.model, args.loss.split('-'), use_gpu)

assert args.resume_checkpoint
args.resume_checkpoint = args.resume_checkpoint.split('/')[-1].split('_')
assert args.resume_checkpoint[1] == args.model

pretrained = getModelFused(args.resume_checkpoint[0], args.resume_checkpoint[1],
                           args.resume_checkpoint[2].split('-'), use_gpu)
pretrained = load_model(pretrained, args.resume_path)

model.model = pretrained.model
model.classifier_pos = pretrained.classifier_pos
model.classifier_neg = pretrained.classifier_neg
del pretrained

model.cuda()

# define dataloader
# ---------------------------
train_dataloaders = getDataloader(args.dataset, args.batch_size)

allSteps = ['pos', 'neg', 'rescue', 'penalty']

rescueFuseRatio = {'pos': 1.0, 'neg': 0.0}
penaltyFuseRatio = {'pos': 0.0, 'neg': 1.0}

lr_update_penalty_dict = {'reverse': 0.1, 'entropy': 0.1, 'KLdiv': 0.005, 'JSD': 0.0005,
                          'recReverse': 0.01, 'wtEntropy': 10, 'wtKLdiv': 0.05, 'wtJSD': 0.005,
                          'focalReverse': 1, 'focalEntropy': 5, 'focalKLdiv': 0.25, 'focalJSD': 0.025}
lr_update = {'rescue': [1, 0, 0], 'penalty': [lr_update_penalty_dict[args.negloss], 0, 0]}

lr_train = {'rescue': [0.001, 0.001, 0], 'penalty': [0, 0, 0], 'pos': [0, 0.001, 0], 'neg': [0, 0, 0.001]}


# define loss for updating
# ------------------
def lossFunction(outputs, labels, tag):
    labelDist = getLableDist(args.dataset, args.loss.split('-')[1])

    if tag in ['pos', 'neg']:
        if isinstance(outputs[tag], list):
            return sum([F.cross_entropy(out, labels[tag]) for out in outputs[tag]])
        else:
            return F.cross_entropy(outputs[tag], labels[tag])

    if tag in ['rescue', 'penalty']:
        if isinstance(outputs['pos'], list):
            pos_loss = sum([pidLoss(args.posloss, out, labels['pos']) for out in outputs['pos']])
        else:
            pos_loss = pidLoss(args.posloss, outputs['pos'], labels['pos'])

        if isinstance(outputs['neg'], list):
            neg_loss = sum([advLoss(args.negloss, out, labels['neg'], labelDist) for out in outputs['neg']])
        else:
            neg_loss = advLoss(args.negloss, outputs['neg'], labels['neg'], labelDist)

        if tag == 'rescue':
            return rescueFuseRatio['pos'] * pos_loss + rescueFuseRatio['neg'] * neg_loss

        elif tag == 'penalty':
            return penaltyFuseRatio['pos'] * pos_loss + penaltyFuseRatio['neg'] * neg_loss


if __name__ == '__main__':
    use_task = {'pos': args.loss.split('-')[0], 'neg': args.loss.split('-')[1]}

    # reset update schedulers and counters
    nUpdate = {k: 0 for k in lr_update}
    nTrain = {k: 0 for k in lr_train}

    nVal, valAcc = validation(model, train_dataloaders, writer, -1, use_task, args.log_dir)

    thresholdRescueGoal = {'pos': float(valAcc['pos']), 'neg': 0}
    thresholdTriggleUpdate = {'pos': thresholdRescueGoal['pos'] - 0.01, 'neg': float(valAcc['neg']) - 0.1}
    thresholdGoal = {'pos': float(valAcc['pos']) + 0.2, 'neg': float(valAcc['neg']) - 0.5}

    scheduler = {tag: restartScheduler(model, lr_update[tag], args.step_size, False) for tag in lr_update}
    optzUpdt = {tag: scheduler[tag][0] for tag in lr_update}
    schdUpdt = {tag: scheduler[tag][1] for tag in lr_update}

    for n in range(args.num_epochs):
        if valAcc['pos'] < thresholdTriggleUpdate['pos']:
            while valAcc['pos'] < thresholdRescueGoal['pos']:
                # joint training feature extraction and identity classifier to rescue
                nTrain['rescue'] = rescueExtractor(model, train_dataloaders, writer, nTrain['rescue'],
                                                   lr_train, use_task, lossFunction)

                # training classifier
                nTrain['neg'], _ = trainSceneClassifier(model, train_dataloaders, writer, nTrain['neg'],
                                                        lr_train, use_task, lossFunction)
                nTrain['pos'], _ = trainIdentityClassifier(model, train_dataloaders, writer, nTrain['pos'],
                                                           lr_train, use_task, lossFunction)

                # validate
                nVal, valAcc = validation(model, train_dataloaders, writer, nVal, use_task, args.log_dir)

            thresholdRescueGoal['pos'] = float(valAcc['pos'])
            thresholdTriggleUpdate['pos'] = thresholdRescueGoal['pos'] - 0.01
            thresholdTriggleUpdate['neg'] = min(thresholdTriggleUpdate['neg'], float(valAcc['neg']) - 0.1)

        elif valAcc['neg'] > thresholdTriggleUpdate['neg']:
            # update via adv back propogation
            nUpdate['penalty'], _ = updateModel(
                model, train_dataloaders, writer, optzUpdt['penalty'], schdUpdt['penalty'],
                lr_update, 'penalty', nUpdate['penalty'], use_task, lossFunction)

            # training scene classifier
            nTrain['neg'], _ = trainSceneClassifier(model, train_dataloaders, writer, nTrain['neg'],
                                                    lr_train, use_task, lossFunction)
            nTrain['pos'], _ = trainIdentityClassifier(model, train_dataloaders, writer, nTrain['pos'],
                                                       lr_train, use_task, lossFunction)

            # validate
            nVal, valAcc = validation(model, train_dataloaders, writer, nVal, use_task, args.log_dir)

        else:
            for tag in ['pos', 'neg']:
                restartClassifier(model, tag)

            # training classifier
            nTrain['neg'], _ = trainSceneClassifier(model, train_dataloaders, writer, nTrain['neg'],
                                                    lr_train, use_task, lossFunction)
            nTrain['pos'], _ = trainIdentityClassifier(model, train_dataloaders, writer, nTrain['pos'],
                                                       lr_train, use_task, lossFunction)

            # validate
            nVal, valAcc = validation(model, train_dataloaders, writer, nVal, use_task, args.log_dir)

            thresholdRescueGoal['pos'] = max(thresholdRescueGoal['pos'], float(valAcc['pos']))
            thresholdTriggleUpdate['pos'] = thresholdRescueGoal['pos'] - 0.01
            thresholdTriggleUpdate['neg'] = min(thresholdTriggleUpdate['neg'], float(valAcc['neg']) - 0.1)

            # Ending Criterion
            if valAcc['neg'] < thresholdGoal['neg'] and valAcc['pos'] > thresholdGoal['pos']:
                break
