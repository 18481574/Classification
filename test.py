import os, sys

import numpy as np 
import torch
import torch.nn as nn

import argparse

from torch.utils.data import Dataset, DataLoader

from datasets.dataset import *
from attacks.attacks import *

from models.model import *
from utils.losses import *

import torch.optim as optim

from measurement import measure

import collections



def weights_init(m):
    if 'reset_parameters' in dir(m):
        # print(m.__class__.__name__)
        m.reset_parameters()

# Models descriptor
ModelDescriptor = collections.namedtuple(
    'ModelDescriptor',
    [
        'model', 
        'train',
        # 'test',
        'save_dir',
    ])








