import numpy as np 
import torch

from torch.utils.data import Dataset, DataLoader

from datasets.dataset import *
from attacks.attacks import *
# from models.models import *
from models import GradCNN

import torch.optim as optim

from measurement import measure




def main():
    parser = argparse.ArgumentParser(description='Case Test for Classification')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--batch-size-small', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--batch-size-test', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    torch.manual_seed(args.seed)

    dataset_name = 'mnist'
    train_split = 'train_small'
    test_split = 'test'

    train_dataset = DataSet(dataset=dataset_name, split=train_split)
    test_dataset = DataSet(dataset=dataset_name, split=test_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size_small, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size_test)


    # Model Initialization
    leNet = leNet5()
    cnn = CNN()
    cnnGrad = GradCNN() 

    Models = {
        'leNet5': leNet,
        'CNN_MNIST': cnn, 
        'CNNGRAD_MNIST': cnnGrad
    }
     





if __name__ == '__main__':
    main()
