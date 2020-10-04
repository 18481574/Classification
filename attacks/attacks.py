import torch 
import numpy as np 



__all__ == ['Attack', ]

_ATTACK_LIST = {
    'LPG': _LPG,
    'FGSM': _FGSM,
    'IFGSM': _IFGSM,
    'one_pixel':_One_pixel,
    ''
}



class Attack(object):

    def __init__(self, name='LPG', **kwarg):
        super(self).__init__()
        if name not in _ATTACK_LIST:
            raise ValueError('The specified attack is not supported yet.')

        self.name = name

        # self._init(kwarg) # extract some properties of attack

        self.attack_ = _ATTACK_LIST[name](kwarg)


    # def _init(self, **kwarg):


    # default: attack without target label 
    def attack(self, input, **kwargs):
        return self.attack_.attack(input, kwargs)



    def set(self, **kwargs):
        self.attack_.set(kwargs)









