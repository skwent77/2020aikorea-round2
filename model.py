import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import pretrainedmodels

class EffNet(nn.Module):
    def __init__(self, type, pretrained=True, num_classes=2):
        super(EffNet, self).__init__()
        assert type in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'], 'efficientNet은 [b0, b1, b2, b3, b4, b5, b6, b7] 중에 하나이어야 합니다.'

        if pretrained:
            self.effnet = EfficientNet.from_pretrained(f'efficientnet-{type}')
        else:
            self.effnet = EfficientNet.from_name(f'efficientnet-{type}')

        self.fc1 = nn.Linear(in_features=1000, out_features=num_classes)
        # self.fc2 = nn.Linear(in_features=500, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.effnet(x)
        x = self.relu(x)
        x = self.fc1(x)
        #  x = self.relu(x)
        #  x = self.fc2(x)
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()

        self.net = pretrainedmodels.__dict__['xception'](num_classes=1000, pretrained='imagenet')

        self.fc1 = nn.Linear(in_features=1000, out_features=num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x
!python main.py --mode ensemble --prediction_file./prediction/153_ensemble_final.tsv
class Net(torch.nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=False)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Dropout()
        )
        
        self.last_fc = nn.Sequential(nn.Linear(256, num_classes))

        self._initialize_weights()
        
        
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.conv2(h)

        pool = self.avgpool(h)
        flatten = torch.flatten(pool, 1)
        classifier = self.classifier(flatten)
        classes = self.last_fc(classifier)

        return classes

