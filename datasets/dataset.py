import os
import glob

import numpy as numpy
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import collections

__all__ = ['DataSet', 'print_path']

def print_path():
    path = os.getcwd()
    print(path)



# dateset descriptor
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'unknown_label',  # Unknown label value.
        'save_dir',     # Save directory of Dataset
    ])


_MNIST_INFORMATION = DatasetDescriptor(
    splits_to_sizes = { 'train': 60000,
                        'train_aug': 100000, # randomly choose?
                        'test': 10000,
                        'train_small': 600,},
    num_classes = 10,
    unknown_label = None,

    save_dir = './datasets/data',
)

_CIFAR10_INFORMATION = DatasetDescriptor(
    splits_to_sizes = { 'train': 50000,
                        'train_aug': 20000,
                        'train_small': 5000,
                        'test': 10000},
    num_classes = 10,
    unknown_label = None,

    save_dir = './datasets/data',
)

_DATASETS_INFORMATION = {
    'mnist': _MNIST_INFORMATION,
    'cifar10': _CIFAR10_INFORMATION,
}


transform = transforms.ToTensor()


class DataSet(Dataset):
    '''The class of dataset for networks training ans testing.
    -----------------------------------
    Parameters:
    dataset_name: str
        The name of the dataset
    split: str
        The type of the dataset, e.g. 'train', 'val', 'test'...
    train: bool
        The data for train or not
    aug: bool
        Augmented data or not
    small: bool
        True: only use a part samples of data to train or test
        False: use all the data
        data_small = data[small_idx]
    samples_small: list
        A list contain the index of the samples of small dataset
    data_aug / target_aug: torch.Tensor
        The data of augmented data. --------- add later
    N: int
        The number of sample in the dataset
    num_classes: int
        Number of the classes in the dataset
    unknown_label: int or None
        Unlabeled data or the data cannot be recognized 
    data, target: torch.Tensor
        The data of the input and the target
    data_src / target_src: torch.Tensor
        The source data of the data and target
    classes: dictionary
        The names of each classes in the dataset
    '''
    def __init__(self, dataset='mnist', split='train', files_dir=None, transform=transform):
        super(DataSet, self).__init__()
        if dataset not in _DATASETS_INFORMATION:
            raise ValueError('The specified dataset is not supported yet.')

        self.dataset_name = dataset
        self.split = split 
        self.transform = transform
        
        self.train = _is_train(split)
        self.aug = _is_aug(split)

        self.num_classes = _DATASETS_INFORMATION[dataset].num_classes
        self.unknown_label = _DATASETS_INFORMATION[dataset].unknown_label

        if files_dir is None:
            files_dir = _DATASETS_INFORMATION[dataset].save_dir

        self.data_src, self.targets_src, self.classes = self._load(files_dir) 

        if self.aug:
            self.data_aug = self._aug()
            self.N = len(self.data_aug) 
        else:
            self.N = _DATASETS_INFORMATION[dataset].splits_to_sizes
            if _is_small(split):
                skip = len(self.data.src) / self.N
                self.small_idx = list(range(0, len(sel.data_src), skip))
                
    @property
    def data(self):
        if self.aug:
            return self.data_aug
        elif self.small:
            return self.data_src[self.small_idx]
        return self.data_src

    @property
    def target(self):
        if self.aug:
            return traget_aug
        elif self.small:
            return self.target_src[self.small_idx]
        return self.data_src

    def _load(self, files_dir): 
        if self.dataset_name == 'mnist':
            return _mnist_load(files_dir, self.train, self.transform)
        elif self.dataset_name == 'cifar10':
            return _cifar10_load(files_dir, self.train, self.transform)
        else:
            return _dataset_load(files_dir, dataset, split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        
        # img = Image.fromarray(img.numpy(), mode='L')
        img = Image.fromarray(img.numpy())

        return img, target

    def __len__(self) -> int:
        return self.N

    # def _aug(samples):
        # return samples


def _is_train(split):
    return split.find('train') > -1 

def _is_aug(split):
    return split.find('aug') > -1 

def _is_small(split):
    return split.find('small') > -1

# not complete
def _aug(data, target):

    return data, target


def _mnist_load(files_dir, train, transform):
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    classes = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four',
                5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}

    # train = _is_train(split)

    data_set = datasets.MNIST(files_dir, train=train, download=True, transform=transform)

    data, targets = data_set.data, data_set.targets


    return data, targets, classes


def _cifar10_load(files_dir, train, transform):
    files_dir = os.path.join(files_dir, 'CIFAR10')

    if not os.path.exists(files_dir):
        os.mkdir(files_dir)


    data_set = datasets.CIFAR10(files_dir, train=train, download=True, transform=transform)

    data, targets = data_set.data, data_set.targets

    classes = data_set.classes
    N = len(classes)
    classes = {i: classes[i] for i in range(N)}


    return data, targets, classes

# need to unify the format of dataset
def _dataset_load(files_dir, dataset, split):
    return []
