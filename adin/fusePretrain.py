from __future__ import print_function, division

import os, sys, argparse
from datetime import datetime

sys.path.append(os.getcwd())

from baseline.pridUtils.utils import *
from adin.config.adinFunctions import *


def get_checkpoint(checkpoint):
    resume_path = os.path.join(log_dir, checkpoint[0], '_'.join(checkpoint))
    resume_epoch = max([int(m.split('_')[-1].split('.')[0]) for m in os.listdir(resume_path) if m.endswith(".pth")])
    resume_path = os.path.join(resume_path, 'saved_model_%s.pth' % resume_epoch)

    pretrained = getModel(checkpoint[0], checkpoint[1], checkpoint[2], True)
    pretrained = load_model(pretrained, resume_path)
    return pretrained


log_dir = './prid_log'
parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--resume-pid', default='MSMT17_resnet_crossEntropy_20191218030500', type=str,
                    help='resume feature extractor and identity classifier')
parser.add_argument('--resume-env', default='MSMT17_resnet_classCamId_20191218023840', type=str,
                    help='resume nuisance classifier')
args = parser.parse_args()

resume_pid = args.resume_pid.split('_')
resume_env = args.resume_env.split('_')

assert all([resume_pid[n] == resume_env[n] for n in [0, 1]])
use_dataset = resume_pid[0]
use_model = resume_pid[1]
use_loss = [resume_pid[2], resume_env[2]]
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
name = '%s_%s_%s_%s' % (use_dataset, use_model, '-'.join(use_loss), timestamp)

model = getModelFused(use_dataset, use_model, use_loss, True)

# copy pretrain model to current model
pretrained_pid = get_checkpoint(resume_pid)
pretrained_env = get_checkpoint(resume_env)

model.model = pretrained_pid.model
model.classifier_pos = pretrained_pid.classifier
model.classifier_neg = pretrained_env.classifier

os.makedirs(os.path.join(log_dir, use_dataset, name), exist_ok=True)
saveModelPath = clean_file(os.path.join(log_dir, use_dataset, name, "saved_model_0.pth"))
save_model(model, saveModelPath)
