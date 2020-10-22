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

# Models descriptor
ModelDescriptor = collections.namedtuple(
    'ModelDescriptor',
    [
        'model', 
        'train',
        'test',
    ])


def train(model: nn.Module, train_loader: DataLoader, criterion, optimizer: torch.optim, epoch:int, last_layer=None, display_inter=1, device=torch.device('cpu'), verbose=False):
    model.train()
    # model.to(device)
    # device = model.device

    if train_loader.drop_last:
        N = train_loader.batch_size * len(train_loader)
    else:
        N = len(train_loader.dataset)

    loss = 0.
    acc = 0.
    n = 0
    for itr, data in enumerate(train_loader):
        input, target = data[0], data[1]

        if isinstance(input, list) or isinstance(input, tuple):
            for i in range(3):
                input[i] = input[i].to(device)
        else:
            input = input.to(device)
        target = target.to(device)

        # noise = torch.randn_like(input) * 0.05
        # input = input + noise
        output = model(input) # y + Reg or (y1, y2, y3)

        if model.__class__.__name__.find('Triplet') > -1:
            Loss = criterion(output, target)
            logit = output[0]
            if isinstance(logit, tuple) or isinstance(logit, list):
                logit = logit[0]
        elif isinstance(output, list) or isinstance(output, tuple):
            output, Reg = output[0], output[1]
            if last_layer is not None:
                output = last_layer(output)
            Loss = criterion((output, Reg), target)
            logit = output
        else:
            if last_layer is not None:
                output = last_layer(output)
            if criterion.__class__.__name__.find('TVLoss') > -1:
                target_ = None
                criterion.Reg.W = GraphTVLoss._get_W(input.view(input.shape[0], -1), target=target_).to(device)
            Loss = criterion(output, target)
            logit = output

        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        pred = torch.argmax(logit, dim=1)
        acc += pred.eq(target).sum().item()

        loss += Loss.item() * logit.shape[0]
        n += logit.shape[0]
        
        if verbose and (itr % display_inter == 0 or n==N):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, n, N, 100. * n / N, Loss.item()))


    return loss / n, acc / n # average loss & acc

def test(model: nn.Module, test_loader: DataLoader, criterion, last_layer=None, device=torch.device('cpu')):
    model.eval()
    # model.to(device)
    # device = model.device

    N = len(test_loader.dataset)

    loss = 0.
    acc = 0.
    n = 0
    with torch.no_grad():
        for itr, data in enumerate(test_loader):
            input, target = data[0].to(device), data[1].to(device)

            output = model(input)

            if last_layer is not None:
                if isinstance(output, list) or isinstance(output, tuple):
                    for i in range(len(output)):
                        output[i] = last_layer(output[i])

            Loss = criterion(output, target)

            loss += Loss.item() * input.shape[0]
            n += input.shape[0]

            pred = torch.argmax(output, dim=1)
            acc += pred.eq(target).sum().item()

    return loss / n, acc / n 


def main():
    parser = argparse.ArgumentParser(description='Case Test for Classification')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--batch-size-small', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--batch-size-test', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')

    parser.add_argument('--noise-sd', default=0.0, type=float,
                    help="standard deviation of Gaussian noise for data augmentation")

    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')

    # parser.add_argument('--save-dir', default='results/', type=str, help='path of save folder')
    # parser.add_argument('--save-model', action='store_true', default=True,
                        # help='For Saving the current Model')


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    torch.manual_seed(args.seed)

    dataset_name = 'mnist'
    train_split = 'train_small'
    test_split = 'test'

    train_dataset = DataSet(dataset=dataset_name, split=train_split)
    test_dataset = DataSet(dataset=dataset_name, split=test_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_small, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    train_dataset_triplet = Triplet(train_dataset)
    train_loader_triplet = DataLoader(train_dataset_triplet, batch_size=args.batch_size_small)

    # Model Initialization
    leNet = leNet5()
    cnnGrad = CNNGrad() 
    graphTV = CNN()
    cnnFeature = TripletNet(FCNN())
    
    _Info_cnnGrad = ModelDescriptor(
        model = cnnGrad,
        train = {
            'loader': train_loader,
            'criterion': Loss_with_Reg(nn.NLLLoss(), torch.norm),
            'last_layer': None,
        },

        test = {
            'loader': test_loader,
            'criterion': nn.NLLLoss(),
            'last_layer': None,
        }
    )

    graphTVLoss = GraphTVLoss(alpha=.1)
    _Info_graphTV = ModelDescriptor(
        model = graphTV,

        train = {
            'loader': train_loader,
            'criterion': CETVLoss(graphTVLoss),
            'last_layer': None, 
        },

        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
            'last_layer': None,
        }
    )

    # nn.TripletMarginLoss(margin=1.0, p=2),
    _Info_Feature = ModelDescriptor(
        model = cnnFeature,
        
        train = {
            'loader': train_loader_triplet,
            'criterion': CE_with_TripletLoss(alpha=1., margin=1.0, p=2),
            'last_layer': None,
        }, 

        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
            'last_layer': SoftMaxTV()
        }
    )

    tripletNetTV = TripletNetTV()
    _Info_TripletTV = ModelDescriptor(
        model = tripletNetTV,

        train = {
            'loader': train_loader_triplet,
            'criterion': CE_with_RegLoss(),
            'last_layer': None
        },

        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
            'last_layer': None
        }
    )

    Information_List = {
        # 'leNet5': _Info_leNet5,
        'cnnGrad': _Info_cnnGrad,
        # 'GraphTV': _Info_graphTV,
        'CNN_Feature': _Info_Feature,
        'TripletTV': _Info_TripletTV,
    }

    results = []

    verbose = False
    for name in Information_List:
        Info_ = Information_List[name]
        model = Info_.model
        train_ = Info_.train
        test_ = Info_.test

        train_loader = train_['loader']
        test_loader = test_['loader']
        
        criterion_train = train_['criterion']
        criterion_test = test_['criterion']

        last_layer_train = train_['last_layer']
        last_layer_test = test_['last_layer']


        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.to(device)
        for epoch in range(args.epochs):
            loss_train, acc_train = train(model, train_loader, criterion_train, optimizer, epoch=epoch, last_layer=last_layer_train, device=device, display_inter=1, verbose=verbose)
            loss_test, acc_test = test(model, test_loader, criterion_test, device=device, last_layer=last_layer_test)

            # model._deactivate()
            # loss_test1, acc_test1 = test(model, test_loader, criterion_test, device=device, last_layer=last_layer_test)
            # model._activate()

            print('Epoch [{}/{}] ({}): \nTrain: \tAcc = {:.2f}%, \tLoss = {:.2f} \nTest: \tAcc = {:.2f}%, \tLoss = {:.2f}'.format(epoch,
                args.epochs, name, acc_train*100., loss_train, acc_test*100., loss_test))


        results.append([(loss_train, acc_train), (loss_test, acc_test)])

    print(results)
if __name__ == '__main__':
    main()
