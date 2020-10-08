import numpy as np
import torch
import math

import warnings

class element_avrg(object):
    def __init__(self, s:float = 0., c:int = 0):
        super(element_avrg, self).__init__()

        self.s  = s
        self.c = c

    def reset(self):
        self.s = 0.
        self.c = 0


    def upd(self, ds=0., dc=1):
        self.s += ds * dc
        self.c += dc


    def value(self):
        if self.c==0: 
            warnings.warn('Empty Set')
            return 0
        else:
            return self.s / self.c




