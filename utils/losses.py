import numpy as np 
import torch
import torch.nn as nn

__all__ = ['Loss_with_Reg', 'CETVLoss', 'CE_with_TripletLoss', 'CE_with_RegLoss',]

class Loss_with_Reg(nn.Module):
    def __init__(self, criterion_L, criterion_Reg):
        super(Loss_with_Reg, self).__init__()
        self.criterion_L = criterion_L
        self.criterion_Reg = criterion_Reg

    def forward(self, data, target):
        x, y = data[0], data[1]
        # print('shape = ', x.shape,)
        Loss = self.criterion_L(x, target)

        if isinstance(y, list) or isinstance(y, tuple):
            for v in y:
                Reg = self.criterion_Reg(v, p=2, dim=1).sum() # torch.norm
                Loss += Reg
        else:
            Reg = self.criterion_Reg(y)
            Loss += Reg

        return Loss 

class CrossEntropy_with_GraphTVLoss(nn.Module):
    def __init__(self, graphTVLoss):
        super(CrossEntropy_with_GraphTVLoss, self).__init__()
        self.Reg = graphTVLoss

    def forward(self, data, target):
        if isinstance(data, tuple) or isinstance(data, list):
            x, feature = data[0], nn.functional.normalize(data[1], dim=1)
        else:
            x = data
            feature = data

        Loss = nn.CrossEntropyLoss()(x, target)
        # Loss = nn.NLLLoss()(torch.log(x), target)
        Loss_Reg = self.Reg(feature)


        return Loss + Loss_Reg


CETVLoss = CrossEntropy_with_GraphTVLoss


class CE_with_TripletLoss(nn.Module):
    def __init__(self, alpha=1., margin=1., p=2):
        super(CE_with_TripletLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.Reg = nn.TripletMarginLoss(margin=margin, p=p)

        self.alpha = alpha

    def forward(self, data, target):
        if self.training:
            # x0, x1, x2 = data[0][0], data[1][0], data[2][0]
            x0 = data[0][0]
            f0, f1, f2 = data[0][1], data[1][1], data[2][1]
            # print(len(data), type(data))
            # print(len(x0), type(x0))
            Loss = self.CE(x0, target)
            Reg = self.Reg(f0, f1, f2) * self.alpha

            return Loss + Reg
        else:
            Loss = self.CE(data, target)

            return Loss

class CE_with_RegLoss(nn.Module):
    def __init__(self):
        super(CE_with_RegLoss, self).__init__()

        self.CE = nn.CrossEntropyLoss()

    def forward(self, data, target):
        x, loss_reg = data[0], data[1]

        return self.CE(x, target) + loss_reg
