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
        # 'train',
        'test',
        'filename',
        'save_dir',
    ])

AttackerDescriptor = collections.namedtuple(
    'AttackerDescriptor',
    [
        'name',
        'device',
        'detail',
    ])

def test(model, loader, attacker, criterion, device):
    model.eval()

    N = len(loader.dataset)
    pred_success, attack_success, loss = 0, 0, 0.
    with torch.no_grad():
        for itr, data in enumerate(loader):
            input, target = data[0], data[1]

            noise = torch.randn_like(input) * 0.05
            input = input + noise
            input = torch.clamp(input, 0., 1.)
            input = input.to(device)
            target = target.to(device)

            logit = model(input)

            Loss = criterion(logit, target)
            loss += Loss.item() * input.shape[0]

            pred = torch.argmax(logit, dim=1)
            pred_success += pred.eq(target).sum().item()

            with torch.enable_grad():
                input_attack = attacker.attack(inputs=input, model=model, labels=target, targeted=None)

            logit_attack = model(input_attack)
            pred_attack = torch.argmax(logit_attack, dim=1)
            attack_success += (~pred_attack.eq(target)).sum().item()

    # pred_succes should > 0
    return pred_success / N, attack_success  / N, loss / N

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = 'mnist'
    test_split = 'test'
    
    tries = 10
    batch_size = 128

    test_dataset = DataSet(dataset=dataset_name, split=test_split)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cnn = CNN()
    cnnGrad = CNNGrad()
    cnnTriplet_active = TripletNetTV()
    cnnTriplet_deactive = TripletNetTV(active=False)

    _Info_CNN = ModelDescriptor(
        model = cnn,
        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
        },
        filename = 'CNN_base',
        save_dir = './results/cnn',
    )

    _Info_CNNGrad = ModelDescriptor(
        model = cnnGrad,
        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss()
        },
        filename = 'CNN_Grad',
        save_dir = './results/cnn_grad',
    )

    _Info_CNNTriplet_deactive = ModelDescriptor(
        model = cnnTriplet_deactive,
        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
        },
        filename = 'CNN_Triplet',
        save_dir = './results/cnn_triplet',
    )

    _Info_CNNTriplet_active = ModelDescriptor(
        model = cnnTriplet_active,
        test = {
            'loader': test_loader,
            'criterion': nn.CrossEntropyLoss(),
        },
        filename = 'CNN_Triplet',
        save_dir = './results/cnn_triplet',
    )

    Information_List = {
        # 'CNN_base': _Info_CNN,
        'CNN_Grad': _Info_CNNGrad,
        # 'CNN_Triplet_deactive': _Info_CNNTriplet_deactive,
        # 'CNN_Triplet_active': _Info_CNNTriplet_active,
    }


# _INFO_COMMON = ['steps', 'random_start', ]
# _INFO_PGD = ['max_norm', 'step_size', ] # L2-Normalization
# _INFO_FGSM = ['step_size', ]
# _INFO_IFGSM = ['step_size', 'max_norm', ]
# _INFO_ONEPIXEL = ['popsize']

    _Info_PGD = AttackerDescriptor(
        name = 'PGD',
        device = device,
        detail = {
            'steps': 40,
            'random_start': True,
            'max_norm': 0.3,
            'step_size':0.01,
            'norm_type': 'Infty',
        }
    )

    _Info_FGSM = AttackerDescriptor(
        name = 'FGSM',
        device = device,
        detail = {
            'step': 1,
            'random_start': False,
            'step_size': 0.1,
        }
    )

    _Info_attackers={
        'FGSM': _Info_FGSM,
        'PGD': _Info_PGD,
    }


    attackers = []
    for attacker_name in _Info_attackers:
        attacker_info = _Info_attackers[attacker_name]
        attackers.append(Attack(attacker_info.name, attacker_info.device, **attacker_info.detail))

    for attacker in attackers:
        for name in Information_List:
            _INFO = Information_List[name]
            model = _INFO.model
            test_ = _INFO.test
            loader = test_['loader']
            criterion = test_['criterion']
            # filename = _INFO.filename

            save_dir = _INFO.save_dir

            model.to(device)

            acc_pred_sum, acc_attack_sum, loss_sum = 0., 0., 0.
            for itr in range(tries):
                filename = _INFO.filename + '_' + str(itr) + '.pt'
                save_path = os.path.join(save_dir, filename)
                model_ = torch.load(save_path, map_location=device)
                model.load_state_dict(state_dict = model_)
                acc_pred, acc_attack, loss_test = test(model, loader, attacker, criterion, device)

                acc_pred_sum += acc_pred
                acc_attack_sum += acc_attack
                loss_sum += loss_test

                print('itr = {}, \tAcc_pred = {:.2f}%, \tAcc_attack = {:.2f}%, loss = {:.2f}'.format(itr, acc_pred*100, acc_attack*100, loss_test))

            print('{}({}): \tAcc_pred = {:.2f}%, \tAcc_attack = {:.2f}%, loss = {:.2f}'.format(name, attacker.name,
                acc_pred_sum/tries*100, acc_attack_sum/tries*100, loss_sum/tries))

if __name__ == '__main__':
    main()

