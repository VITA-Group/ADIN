from baseline.models.block import *


######################################################################
# Define the ResNet50-based Model
class ft_resnet50_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier_pos = ClassBlock(2048, class_num_pos)
        self.classifier_neg = ClassBlock(2048, class_num_neg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the DenseNet121-based Model
class ft_densenet121_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier_pos = ClassBlock(1024, class_num_pos)
        self.classifier_neg = ClassBlock(1024, class_num_neg)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_resnet50mid_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier_pos = ClassBlock(2048 + 1024, class_num_pos)
        self.classifier_neg = ClassBlock(2048 + 1024, class_num_neg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x0 = self.model.avgpool(x)  # x0  n*1024*1*1
        x = self.model.layer4(x)
        x1 = self.model.avgpool(x)  # x1  n*2048*1*1
        x = torch.cat((x0, x1), 1)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the ResNet101-based Model
class ft_resnet101_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.resnet101(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier_pos = ClassBlock(2048, class_num_pos)
        self.classifier_neg = ClassBlock(2048, class_num_neg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the ResNet152-based Model
class ft_resnet152_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.resnet152(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier_pos = ClassBlock(2048, class_num_pos)
        self.classifier_neg = ClassBlock(2048, class_num_neg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the DenseNet169-based Model
class ft_densenet169_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.densenet169(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier_pos = ClassBlock(1664, class_num_pos)
        self.classifier_neg = ClassBlock(1664, class_num_neg)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the DenseNet201-based Model
class ft_densenet201_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.densenet201(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier_pos = ClassBlock(1920, class_num_pos)
        self.classifier_neg = ClassBlock(1920, class_num_neg)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}


# Define the DenseNet161-based Model
class ft_densenet161_dual(nn.Module):
    def __init__(self, class_num_pos, class_num_neg):
        super().__init__()
        model_ft = models.densenet161(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        self.classifier_pos = ClassBlock(2208, class_num_pos)
        self.classifier_neg = ClassBlock(2208, class_num_neg)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        ypos = self.classifier_pos(x)
        yneg = self.classifier_neg(x)
        return x, {'pos': ypos, 'neg': yneg}
