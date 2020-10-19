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
        input, target = data[0].to(device), data[1].to(device)

        noise = torch.randn_like(input) * 0.05
        input = input + noise
        output = model(input)

        if last_layer is not None:
            if 'GraphTV' in last_layer.__class__.__name__:
                if 'FCNN' in model.__class__.__name__:
                    processed = output
                    last_layer.W = last_layer._get_W(input.view(input.shape[0], -1), last_layer.n_Neigbr, last_layer.n_sig, target).to(device)
                    # last_layer.W = last_layer._get_W(model.feature.view(input.shape[0], -1), last_layer.n_Neigbr, last_layer.n_sig).to(device)
                    
                    Loss_Reg = last_layer(model.feature.view(input.shape[0], -1))
                else:
                    processed = torch.log(output)
                    last_layer.W = last_layer._get_W(input.view(input.shape[0], -1), last_layer.n_Neigbr, last_layer.n_sig, target).to(device)
                    Loss_Reg = last_layer(output)

                Loss = criterion(processed, target) + Loss_Reg
                # print(Loss.item(), Loss_Reg.item())
            else: # GradTV
                processed = last_layer(output)
                if 'GradientLoss' in dir(model):
                    Loss_Reg = model.GradientLoss()
                    Loss = criterion(processed, target) + Loss_Reg
                else:
                    Loss = criterion(processed, target)
                # print(Loss.item(), Loss_Reg.item())
        else:
            Loss = criterion(output, target)


        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        pred = torch.argmax(output, dim=1)
        acc += pred.eq(target).sum().item()

        loss += Loss.item() * input.shape[0]
        n += input.shape[0]
        
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

            if 'SoftMaxTV' in last_layer.__class__.__name__:
                N_ = input.shape[0]//10
                last_layer.W = GraphTV._get_W(model.feature.view(input.shape[0], -1), n_Neigbr=N_, n_sig=7).to(device)
                output = last_layer(output)

            if last_layer is not None:
                if 'GraphTV' or last_layer.__class__.__name__:
                    if 'FCNN' in model.__class__.__name__:
                        processed = output
                    else:
                        processed = torch.log(output)

                    Loss = criterion(processed, target) 
                else: # GradTV
                    processed = last_layer(output)
                    Loss = criterion(processed, target)
            else:
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
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
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

    train_loader_triplet = DataLoader(train_dataset, batch_size=args.batch_size_small)

    # Model Initialization
    leNet = leNet5()
    cnn = CNN()
    cnnGrad = CNNGrad() 
    graphTV = CNN()
    cnnFeature = FCNN()
    
    _Info_cnnGrad = ModelDescriptor(
        model = cnnGrad,
        train = {
            'loader': train_loader,
            'criterion': Loss_with_Reg(nn.NLLLoss(), nn.norm),
            'last_layer': None,
        },

        test = {
            'loader': test_loader,
            'criterion': nn.NLLLoss(),
            'last_layer': None,
        }
    )

    graphTV = GraphTVLoss(alpha=.05)
    _Info_graphTV = ModelDescriptor(
        model = graphTV,

        train = {
            'loader': train_loader,
            'criterion': Loss_with_Reg(nn.CrossEntropyLoss(), graphTV),
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
            'criterion': TripletLoss(alpha=1.)
            'last_layer': None,
        }, 

        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
            'last_layer': SoftMaxTV()
        }
    )

    Information_List = {
        # 'leNet5': _Info_leNet5,
        'cnnGrad': _Info_cnnGrad,
        'GraphTV': _Info_graphTV,
        'CNN_Feature': _Info_Feature,
    }

    results = []

    verbose = False
    for name in Information_List:
        Info_ = Information_List[name]
        model = Info_['model']

        train_loader = Info_['train']['loader']
        test_loader = Info_['test']['loader']
        
        criterion_train = Info_['train']['criterion']
        criterion_test = Info_['test']['criterion']

        last_layer_train = Info_['train']['last_layer']
        last_layer_test = Info_['test']['last_layer']


        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.to(device)
        for epoch in range(args.epochs):
            loss_train, acc_train = train(model, train_loader, criterion, optimizer, epoch=epoch, last_layer=last_layer, device=device, display_inter=1, verbose=verbose)
            loss_test, acc_test = test(model, train_loader, criterion, device=device, last_layer=last_layer_test)

            print('Epoch [{}/{}] ({}): \nTrain: \tAcc = {:.2f}%, \tLoss = {:.2f} \nTest: \tAcc = {:.2f}%, \tLoss = {:.2f}'.format(epoch,
                args.epochs, name, acc_train*100., loss_train, acc_test*100., loss_test))


        results.append([(loss_train, acc_train), (loss_test, acc_test)])

    print(results)
if __name__ == '__main__':
    main()
