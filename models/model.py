import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .GradCNN import CNN_MNIST, CNN_MNIST_Grad
from .GraphTV import *

CNNGrad = CNN_MNIST_Grad
CNN = CNN_MNIST

__all__ = ['leNet5', 'CNN', 'CNNGrad', 'GraphTV',]



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