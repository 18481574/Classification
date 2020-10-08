import numpy as np 
import torch

from datasets.dataset import *
from attacks.attacks import *
# from models.models import *
from models import GradCNN

from measurement import measure




def main():
    dataset_name = 'mnist'
    train_split = 'train_small'
    test_split = 'test'

    train_dataset = DataSet(dataset=dataset_name, split=train_split)
    test_dataset = DataSet(dataset=dataset_name, split=test_split)


    train_loader = 












if __name__ == '__main__':
    main()