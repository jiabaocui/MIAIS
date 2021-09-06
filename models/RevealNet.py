import torch
import torch.nn as nn
from torchvision.models import resnet50


class RevealNet(nn.Module):
    def __init__(self, nc=3, nhf=64, output_function=nn.Sigmoid, embedding_size=128, num_classes=1662):
        super(RevealNet, self).__init__()
        self.ch = nn.Conv2d(64, 3, 3, 1, 1)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(nc, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
            nn.Conv2d(nhf, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf * 4, 3, 1, 1),
            nn.BatchNorm2d(nhf * 4),
            nn.ReLU(True),
            nn.Conv2d(nhf * 4, nhf * 2, 3, 1, 1),
            nn.BatchNorm2d(nhf * 2),
            nn.ReLU(True),
            nn.Conv2d(nhf * 2, nhf, 3, 1, 1),
            nn.BatchNorm2d(nhf),
            nn.ReLU(True),
        )
        self.reveal_net = nn.Sequential(
            nn.Conv2d(nhf, nc, 3, 1, 1),
            output_function()
        )

    def forward_features(self, x):
        x = self.feature_extractor(x)
        return x

    def forward_ch(self, x):
        return self.ch(x)

    def forward_reveal(self, x):
        x = self.reveal_net(x)
        return x