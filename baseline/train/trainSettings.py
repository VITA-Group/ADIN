from __future__ import print_function, division
import os, argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--dataset', default='MSMT17', type=str, metavar='DATASET',
                        help='training dataset')
    parser.add_argument('--model', default='resnet', type=str, metavar='MODEL',
                        help='training model')
    parser.add_argument('--loss', default='crossEntropy', type=str, metavar='LOSS',
                        help='training dataset')
    parser.add_argument('--batch-size', default=64, type=int, metavar='BATCHSIZE',
                        help='training dataset')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='training optimizer')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='training scheduler')
    parser.add_argument('--lr-base', default=0.01, type=float,
                        help='training learning rate for base')
    parser.add_argument('--lr-class', default=0.1, type=float,
                        help='training learning rate for classifier')
    parser.add_argument('--num-epochs', default=60, type=int,
                        help='training epoches')
    parser.add_argument('--step-size', default=40, type=int,
                        help='training step size')
    parser.add_argument('--milestones', default='30,45,55', type=str,
                        help='training milestones')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='training gamma for optimizer')

    parser.add_argument('--log-dir', default='./prid_log', type=str)
    parser.add_argument('--tfboard-dir', default='./prid_tfboard', type=str)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    name = '%s_%s_%s_%s' % (args.dataset, args.model, args.loss, timestamp)
    print(name + '\n')

    args.milestones = [int(x) for x in args.milestones.split(',')]

    os.makedirs(args.log_dir, exist_ok=True)
    args.log_dir = os.path.join(args.log_dir, args.dataset, name)
    os.makedirs(args.log_dir, exist_ok=True)

    os.makedirs(args.tfboard_dir, exist_ok=True)
    args.tfboard_dir = os.path.join(args.tfboard_dir, args.dataset, name)
    os.makedirs(args.tfboard_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    print(parse_args())
