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



def train(model, loader, criterion, optimizer, device=torch.device('cpu')):
    model.train()

    loss = 0.
    acc = 0.
    n = 0

    for itr, data in enumerate(loader):
        input, target = data[0], data[1]

        if isinstance(input, list) or isinstance(input, tuple):
            for i in range(3):
                noise = torch.randn_like(input[i]) * 0.05
                input[i] = input[i] + noise
                input[i] = input[i].to(device)
        else:
            noise = torch.randn_like(input) * 0.05
            input = input + noise
            input = input.to(device)
        target = target.to(device)

        output = model(input)
        Loss = criterion(output, target)

        if isinstance(output, list) or isinstance(output, tuple):
            logit = output[0]
        else:
            logit = output

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        pred = torch.argmax(logit, dim=1)
        acc += pred.eq(target).sum().item()

        loss += Loss.item() * logit.shape[0]
        n += logit.shape[0]

    return loss / n, acc / n


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch.manual_seed(1997)

    dataset_name = 'mnist'
    train_split = 'train_small'

    batch_size = 32
    epochs = 1234

    train_dataset = DataSet(dataset=dataset_name, split=train_split)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_dataset_triplet = Triplet(train_dataset)
    train_loader_triplet = DataLoader(train_dataset_triplet, batch_size=batch_size)

    tries = 10
    cnn = CNN()
    cnnGrad = CNNGrad()
    cnnTriplet = TripletNetTV()
    
    _Info_CNN = ModelDescriptor(
        model = cnn,
        train = {
            'loader': train_loader,
            'criterion': nn.CrossEntropyLoss(),
        },

        save_dir = './results/cnn(sigmoid)',
    )

    _Info_CNNGrad = ModelDescriptor(
        model = cnnGrad,
        train = {
            'loader': train_loader,
            'criterion': Loss_with_Reg(nn.CrossEntropyLoss(), torch.norm),
        },

        save_dir = './results/cnn_grad(sigmoid)',
    )

    _Info_CNNTriplet = ModelDescriptor(
        model = cnnTriplet,
        train = {
            'loader': train_loader_triplet,
            'criterion': CE_with_RegLoss(),
        },

        save_dir = './results/cnn_triplet(sigmoid)',

    )

    Information_List = {
        # 'CNN_base': _Info_CNN,
        'CNN_Grad': _Info_CNNGrad,
        # 'CNN_Triplet': _Info_CNNTriplet,
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

        loss_train_sum, acc_train_sum = 0., 0.
        for itr in range(tries):
            model.apply(weights_init)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(epochs):
                loss_train, acc_train = train(model, loader, criterion, optimizer, device)

                loss_train_sum += loss_train
                acc_train_sum += acc_train

                if epoch % 10 == 0:
                    print('Epoch [{}/{}] ({}): \nTrain: \tAcc = {:.2f}%, \tLoss = {:.2f}'.format(epoch,
                        epochs, name, acc_train*100., loss_train))

            filename = name + '_' + str(itr) + '.pt'
            save_path = os.path.join(save_dir, filename)

            # torch.save(model, save_path)


        print(name, loss_train_sum/tries, acc_train_sum/tries)

if __name__ == '__main__':
    main()







