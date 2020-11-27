import os
import glob

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, Subset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random

import collections

__all__ = ['DataSet', 'Triplet', 'PairedSample']

def print_path():
    path = os.getcwd()
    print(path)


# dateset descriptor
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists).
        'unknown_label',  # Unknown label value.
        'save_dir',     # Save directory of Dataset
    ])


_MNIST_INFORMATION = DatasetDescriptor(
    splits_to_sizes = { 'train': 60000,
                        'train_aug': 100000, # randomly choose?
                        'test': 10000,
                        'train_small': 100,},
    num_classes = 10,
    unknown_label = None,

    save_dir = './datasets/data',
)

_CIFAR10_INFORMATION = DatasetDescriptor(
    splits_to_sizes = { 'train': 50000,
                        'train_aug': 20000,
                        'train_small': 500,
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

# Some Impletementations in building queue...
# {small_sample, data_aug,...}
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
    data_src / target_src: torch.Tensor format:NCHW
        The source data of the data and target
    classes: dictionary
        The names of each classes in the dataset
    '''
    def __init__(self, dataset='mnist', split='train', files_dir=None):
        super(DataSet, self).__init__()
        if dataset not in _DATASETS_INFORMATION:
            raise ValueError('The specified dataset is not supported yet.')

        self.dataset_name = dataset
        self.split = split 
        
        self.train = _is_train(split)
        self.aug = _is_aug(split)
        self.small = _is_small(split)

        self.num_classes = _DATASETS_INFORMATION[dataset].num_classes
        self.unknown_label = _DATASETS_INFORMATION[dataset].unknown_label

        if files_dir is None:
            files_dir = _DATASETS_INFORMATION[dataset].save_dir

        self.data_src, self.target_src, self.classes = self._load(files_dir) 

        if self.aug:
            self.data_aug, self.target_aug = self._aug((self.data_src, self.target_src))
            self.N = len(self.data_aug) 
        else:
            self.N = _DATASETS_INFORMATION[dataset].splits_to_sizes[split]
            if _is_small(split):
                skip = len(self.data_src) // self.N
                self.small_idx = list(range(0, len(self.data_src), skip))[:self.N]
                # print(self.small_idx, type(self.target_src), len(self.target_src), type(self.small_idx))


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
            return self.target_aug
        elif self.small:
            return self.target_src[self.small_idx]
        return self.target_src

    def _load(self, files_dir): 
        if self.dataset_name == 'mnist':
            return _mnist_load(files_dir, self.train)
        elif self.dataset_name == 'cifar10':
            return _cifar10_load(files_dir, self.train)
        else:
            return _dataset_load(files_dir, dataset, split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        
        return img, target

    def __len__(self) -> int:
        return self.N

    # Not Impletement
    def _aug(self, samples):
        return samples


def _is_train(split):
    return split.find('train') > -1 

def _is_aug(split):
    return split.find('aug') > -1 

def _is_small(split):
    return split.find('small') > -1

# not complete
def _aug(data, target):

    return data, target


def _mnist_load(files_dir, train):
    if not os.path.exists(files_dir):
        os.mkdir(files_dir)

    classes = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four',
                5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}

    # train = _is_train(split)

    data_set = datasets.MNIST(files_dir, train=train, download=True) # data: torch.Tensor(uint8), targets: Tensor

    data, targets = data_set.data.type(torch.float32).div(255.).unsqueeze(dim=1), data_set.targets # uint8 -> float32
    # print(data.shape, targets.shape, data.type(), targets.type())

    return data, targets, classes


def _cifar10_load(files_dir, train):
    files_dir = os.path.join(files_dir, 'CIFAR10')

    if not os.path.exists(files_dir):
        os.mkdir(files_dir)


    data_set = datasets.CIFAR10(files_dir, train=train, download=True) # data: np.array(uint8), targets: List(int)

    data, targets = data_set.data.transpose((0, 3, 1, 2)), np.array(data_set.targets, dtype=np.int64)
    data, targets = torch.tensor(data, dtype=torch.float32).div(255.), torch.tensor(targets)
    # print(data.shape, targets.shape, data.type(), targets.type())

    classes = data_set.classes
    N = len(classes)
    classes = {i: classes[i] for i in range(N)}

    return data, targets, classes

# need to unify the format of dataset
def _dataset_load(files_dir, dataset, split):
    return []



class Triplet(Dataset):
    def __init__(self, dataset):
        super(Triplet, self).__init__()
        self.dataset = dataset

        self.num_classes = dataset.num_classes
        self.all_classes = set(range(self.num_classes))    
        self.idxes = [ set((dataset.target.view(-1)==idx).nonzero().view(-1).numpy()) for idx in range(self.num_classes)]

    def __getitem__(self, idx):
        anchor = self.dataset.data[idx]
        target = self.dataset.target[idx]

        label_pos = int(target)
        label_neg = random.choice(tuple(self.all_classes - {label_pos}))

        idx_pos = random.choice(tuple(self.idxes[label_pos] - {idx}))
        idx_neg = random.choice(tuple(self.idxes[label_neg]))

        positive = self.dataset.data[idx_pos]
        negtive = self.dataset.data[idx_neg]

        return (anchor, positive, negtive), target

    def __len__(self):
        return len(self.dataset)


class PairedSample(DataSet):
    def __init__(self, dataset, num_sample=1):
        super(PairedSample, self).__init__()
        self.dataset = dataset

        self.num_sample = num_sample 
        self.num_classes = dataset.num_classes
        self.all_classes = set(range(self.num_classes))
        self.idxes = [ set((dataset.target.view(-1)==idx).nonzero().view(-1).numpy()) for idx in range(self.num_classes)]

    def __getitem__(self, idx):
        anchor = self.dataset.data[idx]
        target = self.dataset.target[idx]
        positives = []

        label_pos = int(target)
        idxes_pos = random.choices(tuple(self.idxes[label_pos]- {idx}), k=self.num_sample)

        for i in idxes_pos:
            positives.append(self.dataset.data[i])


        return (anchor, positives), target

    def __len__(self):
        return len(self.dataset)