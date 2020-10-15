import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .GradCNN import CNN_MNIST, CNN_MNIST_Grad
from .GraphTV import *

CNNGrad = CNN_MNIST_Grad
CNN = CNN_MNIST

__all__ = ['leNet5', 'CNN', 'CNNGrad', 'GraphTV', 'FCNN', 'SoftMaxTV']



class leNet5(nn.Module):
    def __init__(self):
        super(leNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1, padding=2)
        self.pool1 = nn.Conv2d(6, 6, 2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.pool2 = nn.Conv2d(16, 16, 2, 2)
        self.conv3 = nn.Conv2d(16, 120, 5, 1)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 1x28x28
        x = F.relu(self.conv1(x))
        # 6x28x28
        
        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.pool1(x))
        # 6x14x14
        
        x = F.relu(self.conv2(x))
        # 16x10x10

        # x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.pool2(x))
        # 16x5x5

        x = F.relu(self.conv3(x))
        # 120x1x1

        x = x.view(-1, 120)
        x = F.relu(self.fc1(x)) # 120 -> 84
        logit = self.fc2(x) # 84 -> 10

        return logit


class FCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=10, input_size=28, dropout_rate=0., alpha = .05):
        super(FCNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.drop_rate = dropout_rate
        self.channels = [self.in_channels, 16, 64]
        self.n_feature = 1000

        self.feature = None # ...

        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout2d(p=self.drop_rate)

        self.convs = []
        self.maxPools = []
        for i in range(len(self.channels)-1):
            self.convs.append(nn.Conv2d(self.channels[i], self.channels[i+1], kernel_size=3, padding=1))
            self.maxPools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            # add to parameters
            self.add_module('conv_' + str(i+1), self.convs[i])
            self.add_module('maxPool_' + str(i+1), self.maxPools[i])


        CHW = self.channels[-1] * (input_size // (2**len(self.convs)))**2
        self.fc1 = nn.Linear(CHW, self.n_feature)
        self.fc2 = nn.Linear(self.n_feature, self.out_channels)

    def forward(self, x):
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = self.maxPools[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        self.feature = x
        x = self.fc1(x.view(x.shape[0], -1))
        logit = self.fc2(x)

        return logit