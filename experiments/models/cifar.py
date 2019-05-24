import math
import torch.nn as nn


from models.densenet import DenseNet3


class BasicConvNet(nn.Module):

    def __init__(self, dataset, planes=16):

        super(BasicConvNet, self).__init__()

        n_classes = 10 if dataset == 'cifar10' else 100

        self.p = planes

        conv1_1 = nn.Conv2d(3, self.p, 3, padding=1, bias=False)
        bn1_1 = nn.BatchNorm2d(self.p)
        conv1_2 = nn.Conv2d(self.p, self.p, 3, padding=1, bias=False)
        bn1_2 = nn.BatchNorm2d(self.p)

        conv2_1 = nn.Conv2d(self.p, self.p * 2, 3, padding=1, bias=False)
        bn2_1 = nn.BatchNorm2d(self.p * 2)
        conv2_2 = nn.Conv2d(self.p * 2, self.p * 2, 3, padding=1, bias=False)
        bn2_2 = nn.BatchNorm2d(self.p * 2)

        conv3_1 = nn.Conv2d(self.p * 2, self.p * 4, 3, padding=1, bias=False)
        bn3_1 = nn.BatchNorm2d(self.p * 4)
        conv3_2 = nn.Conv2d(self.p * 4, self.p * 4, 3, padding=1, bias=False)
        bn3_2 = nn.BatchNorm2d(self.p * 4)

        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(2)
        avgpool = nn.AvgPool2d(4)

        self.base = nn.Sequential(conv1_1, bn1_1, relu,
                                  conv1_2, bn1_2, relu, maxpool,
                                  conv2_1, bn2_1, relu,
                                  conv2_2, bn2_2, relu, maxpool,
                                  conv3_1, bn3_1, relu,
                                  conv3_2, bn3_2, relu, maxpool,
                                  avgpool)

        self.fc = nn.Linear(self.p * 4, n_classes)

        for m in self.base.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.fc.bias.data.zero_()

    def forward(self, x):

        x = self.base(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
