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



def test(model, loader, criterion, device=torch.device('cpu')):
    model.eval()

    A, L, N = 0, 0., len(loader.dataset)
    with torch.no_grad():
        for itr, data in enumerate(loader):
            input, target = data[0].to(device), data[1].to(device)

            logit = model(input)
            Loss = criterion(logit, target)

            L += Loss.item() * input.shape[0]

            pred = torch.argmax(logit, dim=1)
            A += pred.eq(target).sum().item()

    return A/N, L/N

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch.manual_seed(1997)

    dataset_name = 'mnist'
    train_split = 'test'

    batch_size = 128
    epochs = 1000

    train_dataset = DataSet(dataset=dataset_name, split=train_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    train_dataset_triplet = Triplet(train_dataset)
    train_loader_triplet = DataLoader(train_dataset_triplet, batch_size=batch_size)

    tries = 10
    cnn = CNN()
    cnnGrad = CNNGrad()
    cnnTriplet = TripletNetTV()
    cnnTriplet._deactivate()

    _Info_CNN = ModelDescriptor(
        model = cnn,
        train = {
            'loader': train_loader,
            'criterion': nn.CrossEntropyLoss(),
        },

        save_dir = './results/cnn',
    )

    _Info_CNNGrad = ModelDescriptor(
        model = cnnGrad,
        train = {
            'loader': train_loader,
            'criterion': nn.CrossEntropyLoss(),
        },

        save_dir = './results/cnn_grad',
    )

    _Info_CNNTriplet = ModelDescriptor(
        model = cnnTriplet,
        train = {
            'loader': train_loader,
            'criterion': nn.CrossEntropyLoss(),
        },

        save_dir = './results/cnn_triplet',

    )

    Information_List = {
        'CNN_base': _Info_CNN,
        'CNN_Grad': _Info_CNNGrad,
        'CNN_Triplet': _Info_CNNTriplet,
    }

    for name in Information_List:
        _INFO = Information_List[name]
        model = _INFO.model
        train_ = _INFO.train
        loader = train_['loader']
        criterion = train_['criterion']

        save_dir = _INFO.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        model = model.to(device)

        loss_test_sum, acc_test_sum = 0., 0.
        for itr in range(tries):
            filename = name + '_' + str(itr) + '.pt'
            save_path = os.path.join(save_dir, filename)
            model.load_state_dict(torch.load(save_path))

            loss_test, acc_test = test(model, loader, criterion, device)
            loss_test_sum += loss_test
            acc_test_sum += acc_test

        print(name, loss_test_sum/tries, acc_test_sum/tries)

if __name__ == '__main__':
    main()







