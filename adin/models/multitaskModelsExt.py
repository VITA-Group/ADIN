from baseline.models.block import *

import torch.nn as nn
from torchvision import models
import copy


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def main_model(with_reduction=False, np_dim=1024):
    model_ft = models.resnet50(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    layer4 = nn.Sequential(
        Bottleneck(
            1024,
            512,
            stride=1,
            downsample=nn.Sequential(
                nn.Conv2d(
                    1024, 2048, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(2048)
            )
        ),
        Bottleneck(2048, 512),
        Bottleneck(2048, 512)
    )
    layer4.load_state_dict(model_ft.layer4.state_dict())
    model_ft.layer4 = layer4

    model_ft.layer4_global = copy.deepcopy(layer4)
    model_ft.global_fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(True),
    )

    if with_reduction:
        model_ft.reduction = nn.Sequential(
            nn.Conv2d(2048, np_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(np_dim),
            nn.ReLU(inplace=True),
        )
    else:
        model_ft.fc0 = nn.Sequential(
            nn.Linear(2048, np_dim),
            nn.BatchNorm1d(np_dim),
            nn.ReLU(True),
        )
        model_ft.fc1 = nn.Sequential(
            nn.Linear(2048, np_dim),
            nn.BatchNorm1d(np_dim),
            nn.ReLU(True),
        )

    return model_ft


class classifiers(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.classifier_g = ClassBlock_Linear(1024, out_dim, num_bottleneck=1024)
        self.classifier0 = ClassBlock_Linear(in_dim, out_dim, num_bottleneck=in_dim)
        self.classifier1 = ClassBlock_Linear(in_dim, out_dim, num_bottleneck=in_dim)

    def reset(self):
        self.classifier_g.reset()
        self.classifier0.reset()
        self.classifier1.reset()


class ft_resnet50_np_2_model1_dual(nn.Module):

    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()

        np_dim = 1024
        self.model = main_model(with_reduction=True, np_dim=np_dim)
        # self.classifier = classifiers(np_dim, class_num)
        self.classifier_pos = classifiers(np_dim, class_num_pos)
        self.classifier_neg = classifiers(np_dim, class_num_neg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        predict_features = []
        # xent_features = []
        xent_features = {'pos': [], 'neg': []}

        x1 = self.model.layer4_global(x)
        x1 = self.model.avgpool(x1).squeeze()
        x1 = self.model.global_fc(x1)
        predict_features.append(x1)
        # xent_features.append(self.classifier.classifier_g(x1))
        xent_features['pos'].append(self.classifier_pos.classifier_g(x1))
        xent_features['neg'].append(self.classifier_neg.classifier_g(x1))

        x2 = self.model.layer4(x)
        x2 = self.model.reduction(x2)
        margin = x2.size(2) // 2

        x3 = x2[:, :, 0:margin, :]
        x3 = self.model.avgpool(x3).squeeze()
        predict_features.append(x3)
        # xent_features.append(self.classifier.classifier0(x3))
        xent_features['pos'].append(self.classifier_pos.classifier0(x3))
        xent_features['neg'].append(self.classifier_neg.classifier0(x3))

        x3 = x2[:, :, margin:margin * 2, :]
        x3 = self.model.avgpool(x3).squeeze()
        predict_features.append(x3)
        # xent_features.append(self.classifier.classifier1(x3))
        xent_features['pos'].append(self.classifier_pos.classifier1(x3))
        xent_features['neg'].append(self.classifier_neg.classifier1(x3))

        return torch.cat(predict_features, 1), xent_features


class ft_resnet50_np_2_model2_dual(nn.Module):

    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()

        np_dim = 1024
        self.model = main_model(with_reduction=False, np_dim=np_dim)
        # self.classifier = classifiers(np_dim, class_num)
        self.classifier_pos = classifiers(np_dim, class_num_pos)
        self.classifier_neg = classifiers(np_dim, class_num_neg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        predict_features = []
        # xent_features = []
        xent_features = {'pos': [], 'neg': []}

        x1 = self.model.layer4_global(x)
        x1 = self.model.avgpool(x1).squeeze()
        x1 = self.model.global_fc(x1)
        predict_features.append(x1)
        # xent_features.append(self.classifier.classifier_g(x1))
        xent_features['pos'].append(self.classifier_pos.classifier_g(x1))
        xent_features['neg'].append(self.classifier_neg.classifier_g(x1))

        x2 = self.model.layer4(x)
        margin = x2.size(2) // 2

        x3 = self.model.avgpool(x2[:, :, 0:margin, :]).squeeze()
        x3 = self.model.fc0(x3)
        predict_features.append(x3)
        # xent_features.append(self.classifier.classifier0(x3))
        xent_features['pos'].append(self.classifier_pos.classifier0(x3))
        xent_features['neg'].append(self.classifier_neg.classifier0(x3))

        x3 = self.model.avgpool(x2[:, :, margin:margin * 2, :]).squeeze()
        x3 = self.model.fc1(x3)
        predict_features.append(x3)
        # xent_features.append(self.classifier.classifier1(x3))
        xent_features['pos'].append(self.classifier_pos.classifier1(x3))
        xent_features['neg'].append(self.classifier_neg.classifier1(x3))

        return torch.cat(predict_features, 1), xent_features

