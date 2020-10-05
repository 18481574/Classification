import torch 
import numpy as np 

from attack_impl import *


__all__ == ['Attack', ]

_ATTACK_LIST = {
    'PGD': _PGD,
    'FGSM': _FGSM,
    'IFGSM': _IFGSM,
    'one_pixel':_One_pixel,
}


class Attack(object):

    def __init__(self, name='LPG', **kwargs):
        super(self).__init__() kw
        if name not in _ATTACK_LIST:
            raise ValueError('The specified attack is not supported yet.')

        self.name = name

        # self._init(kwargs) # extract some properties of attack

        self.attack_ = _ATTACK_LIST[name](kwargs)


    # def _init(self, **kwargs):


    # default: attack without target label 
    def attack(self, input, **kwargs):
        return self.attack_.attack(input, kwargs)



    def set(self, **kwargs):
        self.attack_.set(kwargs)




def _PGD(**kwargs):
    return PGD(kwargs)

def _FGSM(**kwargs):
    return FGSM(kwargs)

def _IFGSM(**kwargs):
    return IFGSM(kwargs)

def _One_pixel(**kwargs):
    return one_pixel_attack(kwargs)









