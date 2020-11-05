import torch 
import numpy as np 

from .attack_impl import *


__all__ = ['Attack', ]


def _PGD(**kwargs):
    _info = _extract_PGD(**kwargs)
    return PGD(**_info)

def _FGSM(**kwargs):
    _info = _extract_FGSM(**kwargs)
    return FGSM(**_info)

def _IFGSM(**kwargs):
    _info = _extract_IFGSM(**kwargs)
    return IFGSM(**info)

def _One_pixel(**kwargs):
    _extract_One_pixel(**kwargs)
    return OnePixelAttack(**_info)

_ATTACK_LIST = {
    'PGD': _PGD,
    'FGSM': _FGSM,
    'IFGSM': _IFGSM,
    'one_pixel':_One_pixel,
}


class Attack(object):

    def __init__(self, name='PGD', device=torch.device('cpu'), **kwargs):
        super(Attack, self).__init__() 
        if name not in _ATTACK_LIST:
            raise ValueError('The specified attack is not supported yet.')

        self.name = name
        self.device = device

        self.attack_ = _ATTACK_LIST[name](**kwargs)
        self.set_device(self.device)


    # default: attack without target label 
    def attack(self, **kwargs):
        return self.attack_.attack(**kwargs)



    def set(self, **kwargs):
        self.attack_.set(**kwargs)

    def set_device(self, device=torch.device('cpu')):
        self.attack_.set_device(device)





_INFO_COMMON = ['steps', 'random_start', ]
_INFO_PGD = ['max_norm', 'step_size', ] # L2-Normalization
_INFO_FGSM = ['step_size', ]
_INFO_IFGSM = ['step_size', 'max_norm', ]
_INFO_ONEPIXEL = ['popsize']

def _extract_info(_info_List, kwargs):
    _info = {}
    
    for v in _info_List:
        if v in kwargs:
            _info[v] = kwargs[v]

    return _info

def _extract_PGD(**kwargs):
    _info_List = [*_INFO_COMMON, *_INFO_PGD]
    return _extract_info(_info_List, kwargs)

def _extract_FGSM(**kwargs):
    _info_List = [*_INFO_COMMON, *_INFO_FGSM]
    return _extract_info(_info_List, kwargs)

def _extract_IFGSM(**kwargs):
    _info_List = [*_INFO_COMMON, *_INFO_IFGSM]
    return _extract_info(_info_List, kwargs)

def _extract_One_pixel(**kwargs):
    _info_List = [*_INFO_COMMON, *_INFO_ONEPIXEL]
    return _extract_info(_info_List, kwargs)










