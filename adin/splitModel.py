from __future__ import print_function, division

import os, sys, argparse

sys.path.append(os.getcwd())

from baseline.pridUtils.utils import *
from adin.config.adinFunctions import *

log_dir = './prid_log'
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--checkpoint', default='MSMT17_resnet_crossEntropy-classCamId_20191218042547', type=str,
                    help='resume checkpoint')
args = parser.parse_args()

checkpoint = args.checkpoint.split('_')
use_dataset = checkpoint[0]
use_model = checkpoint[1]
use_loss = checkpoint[2].split('-')
timestamp = checkpoint[-1]

resume_path = os.path.join(log_dir, checkpoint[0], '_'.join(checkpoint))
resume_epoch = max([int(m.split('_')[-1].split('.')[0]) for m in os.listdir(resume_path) if m.endswith(".pth")])
resume_path = os.path.join(resume_path, 'saved_model_%s.pth' % resume_epoch)

pretrained = getModelFused(use_dataset, use_model, use_loss, True)
pretrained = load_model(pretrained, resume_path)

model_pid = getModel(use_dataset, use_model, use_loss[0], True)
model_env = getModel(use_dataset, use_model, use_loss[1], True)


def save(model, use_loss):
    name = '%s_%s_%s_%s' % (use_dataset, use_model, use_loss, timestamp)
    os.makedirs(os.path.join(log_dir, use_dataset, name), exist_ok=True)
    saveModelPath = clean_file(os.path.join(log_dir, use_dataset, name, "save_model_%s.pth" % resume_epoch))
    save_model(model, saveModelPath)


model_pid.model = pretrained.model
model_pid.classifier = pretrained.classifier_pos
save(model_pid, use_loss[0])

model_env.model = pretrained.model
model_env.classifier = pretrained.classifier_neg
save(model_env, use_loss[1])
