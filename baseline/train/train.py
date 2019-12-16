from __future__ import print_function, division

import os, sys

from torch.optim import lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter

from baseline.train.config import *
from baseline.train.trainDataloaders import *
from baseline.train.trainSettings import *

# define writer
args = parse_args()
writer = SummaryWriter(log_dir=args.tfboard_dir)
use_gpu = True

######################################################################
model = getModel(args.dataset, args.model, args.loss, use_gpu)
model = nn.DataParallel(model).cuda()

# define optimizer
if args.optimizer == 'sgd':
    optimizer = optim.SGD([
        {'params': model.module.model.parameters(), 'lr': args.lr_base},
        {'params': model.module.classifier.parameters(), 'lr': args.lr_class}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
elif args.optimizer == 'adam':
    optimizer = optim.Adam([
        {'params': model.module.model.parameters(), 'lr': args.lr_base},
        {'params': model.module.classifier.parameters(), 'lr': args.lr_class}
    ], weight_decay=5e-4, betas=(0.9, 0.999))

# define scheduler
if args.scheduler == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
elif args.scheduler == 'multiStep':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
elif args.scheduler == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True,
                                               threshold=0.02, threshold_mode='rel', cooldown=5, min_lr=1e-06,
                                               eps=1e-08)
else:
    raise Exception('unknown scheduler')

######################################################################
model.module.model.train() if args.lr_base else model.module.model.eval()
model.module.classifier.train() if args.lr_class else model.module.classifier.eval()

print("model.model.train()" if args.lr_base else "model.model.eval()")
print("model.classifier.train()" if args.lr_class else "model.classifier.eval()")

train_datasets, train_dataloaders = getDataloader(args.dataset, args.batch_size, args.log_dir)

train_model(model, optimizer, scheduler, writer, train_dataloaders, args.num_epochs,
            args.model, args.loss, args.log_dir, args.lr_base, args.lr_class, 'multibranch' in args.model)

writer.close()
