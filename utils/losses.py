import numpy as np 
import torch
import torch.nn as nn

__all__ = ['Loss_with_Reg', 'CETVLoss']

class Loss_with_Reg(nn.Module):
    def __init__(self, criterion_L, criterion_Reg):
        super(Loss_with_Reg, self).__init__()
        self.criterion_L = criterion_L
        self.criterion_Reg = criterion_Reg

    def forward(self, data, target):
        x, y = data[0], data[1]
        print('shape = ', x.shape,)
        Loss = self.criterion_L(x, target)

        if isinstance(y, list) or isinstance(y, tuple):
            for v in y:
                Reg = self.criterion_Reg(v)
                Loss += Reg
        else:
            Reg = self.criterion_Reg(y)
            Loss += Reg

        return Loss 

class CrossEntropy_with_GraphTVLoss(nn.Module):
    def __init__(self, graphTVLoss):
        super(CrossEntropy_with_GraphTVLoss, self).__init__()
        self.Reg = graphTVLoss

    def forward(self, x, target):
        # Loss = nn.CrossEntropyLoss()(x, target)
        Loss = nn.NLLLoss()(torch.log(x), target)
        Loss_Reg = self.Reg(x)

        return Loss + Loss_Reg


CETVLoss = CrossEntropy_with_GraphTVLoss