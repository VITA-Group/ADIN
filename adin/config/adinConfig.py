from __future__ import print_function, division
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from baseline.datasetUtils.trainDataloaders import *
from adin.config.adinFunctions import *


# Restart classifier
# ---------------------------
def restartClassifier(model, tag):
    if tag == 'pos':
        if isinstance(model.classifier_pos, classifiers):
            model.classifier_pos.reset()
        else:
            model.classifier_pos.apply(weights_init_classifier)
    elif tag == 'neg':
        if isinstance(model.classifier_neg, classifiers):
            model.classifier_neg.reset()
        else:
            model.classifier_neg.apply(weights_init_classifier)
    else:
        raise Exception('unknown tag')


# Restart optimizer and schedule
# ---------------------------
def restartScheduler(model, learning_rate, step_size, gamma=0.1):
    print('\nreset optimizer, scheduler ...')

    posClassifierParams = list(map(id, model.classifier_pos.parameters()))
    negClassifierParams = list(map(id, model.classifier_neg.parameters()))
    base_params = filter(lambda p: id(p) not in posClassifierParams + negClassifierParams, model.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': learning_rate[0]},
        {'params': model.classifier_pos.parameters(), 'lr': learning_rate[1]},
        {'params': model.classifier_neg.parameters(), 'lr': learning_rate[2]}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer, scheduler


# Update model
# ------------------
def updateModel(model, train_dataloaders, writer, optimizer, scheduler, lr_update, tag, epochCount,
                use_task, lossFunction):
    print('\nupdate model via %s process ...' % tag)

    model.model.train() if lr_update[tag][0] else model.model.eval()
    model.classifier_pos.train() if lr_update[tag][1] else model.classifier_pos.eval()
    model.classifier_neg.train() if lr_update[tag][2] else model.classifier_neg.eval()

    # add count to scheduler
    scheduler.step()
    running_loss = 0.0
    epochCount += 1

    # zero the parameter gradients
    optimizer.zero_grad()

    for batch, sample in enumerate(tqdm(train_dataloaders['train'])):
        # get images and labels
        images = Variable(sample['img']).cuda()
        labels = {tag: getLable(sample, use_task[tag], True) for tag in use_task}

        # feed forward
        _, outputs = model(images)

        # calculate loss
        loss = lossFunction(outputs, labels, tag)
        running_loss += 1.0 * loss * len(images)

        # back propagate
        loss.backward()

    # update parameter by gradients
    optimizer.step()

    # statistics
    accumulated_loss = running_loss.data[0] / len(train_dataloaders['train'].dataset)

    # tensorboardX
    writer.add_scalar('adin_update/loss/' + tag, accumulated_loss, epochCount)
    writer.add_scalar('adin_update/lr*loss/' + tag, lr_update[tag][0] * accumulated_loss, epochCount)

    # accumulate loss and update weights and bias
    # print("accumulated_loss", accumulated_loss)
    return epochCount, accumulated_loss


# Training the classifier
# ------------------
def trainModel(model, train_dataloaders, writer, optimizer, scheduler, lr_train, tag, nEpoch, epochCount,
               use_task, lossFunction):
    print('\ntrain model via %s process ...' % tag)

    model.model.train() if lr_train[tag][0] else model.model.eval()
    model.classifier_pos.train() if lr_train[tag][1] else model.classifier_pos.eval()
    model.classifier_neg.train() if lr_train[tag][2] else model.classifier_neg.eval()

    for epoch in range(nEpoch):
        epochCount += 1
        scheduler.step()

        running_loss = 0.0
        running_corrects = {tag: 0 for tag in ['pos', 'neg']}

        for batch, sample in enumerate(tqdm(train_dataloaders['train'])):
            # get images and labels
            images = Variable(sample['img']).cuda()
            labels = {tag: getLable(sample, use_task[tag], True) for tag in use_task}

            # zero the parameter gradients
            if batch == 0: optimizer.zero_grad()

            # feed forward
            features, outputs = model(images)

            # calculate loss
            loss = lossFunction(outputs, labels, tag)

            # calculate accuracy
            # preds = {tag: torch.max(outputs[tag].data, 1)[1] for tag in ['pos', 'neg']}
            preds = {
                tag: torch.max(sum([out.data for out in outputs[tag]]), 1)[1] if isinstance(outputs[tag], list) else
                torch.max(outputs[tag].data, 1)[1] for tag in ['pos', 'neg']}
            correct = {tag: torch.sum(preds[tag] == labels[tag].data) for tag in ['pos', 'neg']}

            # statistics
            running_loss += 1.0 * loss.data[0] * len(images)
            running_corrects = {tag: running_corrects[tag] + correct[tag] for tag in ['pos', 'neg']}

            # back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # statistics
        epoch_loss = 1. * running_loss / len(train_dataloaders['train'].dataset)
        epoch_acc = {tag: 1. * running_corrects[tag] / len(train_dataloaders['train'].dataset)
                     for tag in ['pos', 'neg']}

        # tensorboardX
        writer.add_scalar('adin_train/loss/' + tag, epoch_loss, epochCount)
        writer.add_scalar('adin_train/accuracy/' + tag,
                          epoch_acc['pos'] if tag in ['pos', 'rescue'] else epoch_acc['neg'], epochCount)

        # show epoch log (loss, accuracy ...)
        print('\nTag: {}, Epoch: {}, Loss: {:.6f}, Identity Accuracy: {:.4f}%, Scene Accuracy: {:.4f}%\n'.format(
            tag, epoch + 1, epoch_loss, 100. * epoch_acc['pos'], 100. * epoch_acc['neg']))

    return epochCount


# Validation
# ------------------
def validation(model, train_dataloaders, writer, epochCount, use_task, log_dir):
    def entropy(x):
        return -1.0 * F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    print('\nvalidate model ...')

    model.eval()
    running_corrects = {tag: 0 for tag in use_task}
    running_entropys = {tag: 0 for tag in use_task}
    epochCount += 1

    for batch, sample in enumerate(tqdm(train_dataloaders['val'])):
        # get images
        images = Variable(sample['img'], volatile=True).cuda()
        labels = {tag: getLable(sample, use_task[tag], True) for tag in use_task}

        # feed forward
        features, outputs = model(images)

        # calculate accuracy
        # preds = {tag: torch.max(outputs[tag].data, 1)[1] for tag in use_task}
        preds = {
            tag: torch.max(sum([out.data for out in outputs[tag]]), 1)[1] if isinstance(outputs[tag], list) else
            torch.max(outputs[tag].data, 1)[1] for tag in use_task}

        correct = {tag: torch.sum(preds[tag] == labels[tag].data) for tag in ['pos', 'neg']}
        # entropys = {tag: torch.sum(entropy(outputs[tag])).data[0] for tag in use_task}
        entropys = {
            tag: torch.sum(entropy(sum([out for out in outputs[tag]]))).data[0] if isinstance(outputs[tag], list) else
            torch.sum(entropy(outputs[tag])).data[0] for tag in use_task}

        # statistics
        running_corrects = {tag: running_corrects[tag] + correct[tag] for tag in use_task}
        running_entropys = {tag: running_entropys[tag] + entropys[tag] for tag in use_task}

    # statistics
    epoch_acc = {tag: 1. * running_corrects[tag] / len(train_dataloaders['val'].dataset) for tag in use_task}
    epoch_ent = {tag: 1. * running_entropys[tag] / len(train_dataloaders['val'].dataset) for tag in use_task}

    # tensorboardX
    writer.add_scalar('adin_val/accuracy/pos', epoch_acc['pos'], epochCount)
    writer.add_scalar('adin_val/accuracy/neg', epoch_acc['neg'], epochCount)

    writer.add_scalar('adin_val/entropy/pos', epoch_ent['pos'], epochCount)
    writer.add_scalar('adin_val/entropy/neg', epoch_ent['neg'], epochCount)

    # show epoch log (loss, accuracy ...)
    print('\nIdentity: {:.4f}, {:.4f}%, Scene: {:.4f}, {:.4f}%'.format(
        epoch_ent['pos'], 100. * epoch_acc['pos'], epoch_ent['neg'], 100. * epoch_acc['neg']))

    # save model
    save_model(model, os.path.join(log_dir, 'saved_model_%s.pth' % epochCount))
    return epochCount, epoch_acc


# ------------------
def trainSceneClassifier(model, train_dataloaders, writer, epochCount, lr_train, use_task, lossFunction):
    stepsize, trainepoch = 20, 30

    optz, schl = restartScheduler(model, lr_train['neg'], stepsize, False)
    return trainModel(model, train_dataloaders, writer, optz, schl, lr_train, 'neg',
                      trainepoch, epochCount, use_task, lossFunction)


def trainIdentityClassifier(model, train_dataloaders, writer, epochCount, lr_train, use_task, lossFunction):
    stepsize, trainepoch = 20, 30

    optz, schl = restartScheduler(model, lr_train['pos'], stepsize, False)
    return trainModel(model, train_dataloaders, writer, optz, schl, lr_train, 'pos',
                      trainepoch, epochCount, use_task, lossFunction)


def rescueExtractor(model, train_dataloaders, writer, epochCount, lr_train, use_task, lossFunction):
    stepsize, trainepoch = 10, 15

    optz, schl = restartScheduler(model, lr_train['rescue'], stepsize, False)
    return trainModel(model, train_dataloaders, writer, optz, schl, lr_train, 'rescue',
                      trainepoch, epochCount, use_task, lossFunction)


print("Config ... OK!\n")
