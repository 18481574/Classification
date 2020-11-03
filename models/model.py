import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .GradCNN import CNN_MNIST, CNN_MNIST_Grad
from .GraphTV import *

CNNGrad = CNN_MNIST_Grad
CNN = CNN_MNIST

_Models = ['leNet5', 'CNN', 'CNNGrad', 'FCNN', 'TripletNet', 'TripletNetTV', 'ClassifierTV',]
_Loss_layers = ['GraphTVLoss',]
_Activation_layers = ['SoftMaxTV', 'ReLUTV',]
_Misc = []

__all__ = _Models + _Loss_layers + _Activation_layers + _Misc


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

        # self.n_feature_low = 3
        self.n_feature = 1000

        self.feature = None # ...

        # self.activation = nn.Sigmoid()
        self.activation = nn.ReLU()
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

        x = x.view(x.shape[0], -1)
        self.feature = x
        x = self.fc1(x)
        x = self.activation(x)
        # self.feature = x

        logit = self.fc2(x)
        
        if self.training:
            return (logit, self.feature)
        else:
            return logit


# 64 -> 10
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.channels = [1, 16, 64]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        # self.relu = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.maxPool2(x)
        x = self.relu(x)

        return x


class Classifier_MNIST(nn.Module):
    def __init__(self):
        super(Classifier_MNIST, self).__init__()
        CHW = 64 * 49
        feature = 1000
        self.fc1 = nn.Linear(CHW, feature)
        self.relu = nn.ReLU()
        # self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(feature, 10)

    def forward(self, x):
        if x.dim()>2:
            x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

class TripletNet(nn.Module):
    def __init__(self, module):
        super(TripletNet, self).__init__()
        self.netWork = module

    def forward(self, data):
        if self.training:
            x0, x1, x2 = data
            anchor, positive, negative = self.netWork(x0), self.netWork(x1), self.netWork(x2)

            return anchor, positive, negative
        else:
            x = data
            y = self.netWork(x)

            return y


class TripletNetTV(nn.Module):
    def __init__(self, embedding=EmbeddingNet(), classifier=Classifier_MNIST(), alpha=1., reg_criterion=None, active=True):
        super(TripletNetTV, self).__init__()
        self.embedding = embedding
        self.classifier = ClassifierTV(classifier)
        self.alpha = alpha

        self.TV_active = active
        self._activate(active)

        if reg_criterion is None:
            self.reg = nn.TripletMarginLoss(margin=1., p=2)
        else:
            self.reg = reg_criterion

    def forward(self, data):
        if self.training:
            x0, x1, x2 = data
            anchor, positive, negtive = self.embedding(x0), self.embedding(x1), self.embedding(x2)

            prev_state = self.TV_active
            self._deactivate()
            logit = self.classifier(anchor)
            self._activate(prev_state)

            # Loss_reg = self.reg(anchor.view(anchor.shape[0], -1), positive.view(positive.shape[0], -1), negtive.view(negtive.shape[0], -1))
            Loss_reg = self.reg(anchor, positive, negtive)
            return logit, Loss_reg*self.alpha
        else:
            feature = self.embedding(data)
            
            if self.TV_active:
                self.classifier.W = GraphTVLoss._get_W(feature.view(feature.shape[0], -1)).to(feature.device)

            logit = self.classifier(feature)

            return logit


    def _deactivate(self):
        self._activate(False)

    def _activate(self, state=True):
        if self.classifier.TV_active != state:
            self.classifier._activate(state)
        self.TV_active = state



class ClassifierTV(nn.Module):
    def __init__(self, classifier, relu=ReLUTV(), softmax=SoftMaxTV()):
        super(ClassifierTV, self).__init__()
        self.W = None
        self.TV_active = False

        self.classifier = classifier
        self.relu_prev = classifier.relu
        self.relu = relu
        self.softmax = softmax

    def forward(self, x):
        if self.TV_active:
            self.relu.W = self.W
            self.softmax.W = self.W
            self.classifier.relu = self.relu
        else:
            self.classifier.relu = self.relu_prev

        logit = self.classifier(x)

        if self.TV_active:
            logit = self.softmax(logit)

        return logit

    def _activate(self, state):
        self.TV_active = state

