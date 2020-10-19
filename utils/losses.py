import numpy as np 
import torch
import torch.nn as nn

__all__ = ['Loss_with_Reg', ]

class Loss_with_Reg(nn.Module):
    def __init__(self, criterion_L, criterion_Reg):
        super(RegLoss, self).__init__()
        self.criterion_L = criterion_L
        self.criterion_Reg = criterion_Reg

    forward(self, data):
        x, y = data[0], data[1]

        Loss = self.criterion_L(x)
        Reg = self.criterion_Reg(y)

        return Loss + Reg

