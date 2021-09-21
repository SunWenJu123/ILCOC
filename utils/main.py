# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import ContinualDataset
from datasets import get_dataset
from models import get_model
from utils.training import train
from utils.conf import set_random_seed
import torch

import os

def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    args = parser.parse_known_args()[0]
    args.model = 'ocilfast'
    args.seed = None
    args.validation = True

    args.img_dir = 'img/test'  # 打印图片存储路径
    args.print_file = open('../'+args.img_dir+'/result.txt', mode='w')

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    # seq-tinyimagenet
    args.dataset = 'seq-tinyimg'
    args.lr = 2e-3
    args.batch_size = 32  
    args.buffer_size = 0
    args.minibatch_size = 32  
    args.n_epochs = 100 

    args.nu = 0.7  
    args.eta = 0.04  
    args.eps = 1 
    args.embedding_dim = 250  
    args.weight_decay = 1e-2  
    args.margin = 1  
    args.r = 0.01  
    args.nf = 32
    """

    # seq-cifar10
    args.dataset = 'seq-cifar10'
    args.lr = 1e-3
    args.batch_size = 32
    args.buffer_size = 0
    args.minibatch_size = 32
    args.n_epochs = 50

    args.nu = 0.7
    args.eta = 0.8
    args.eps = 1
    args.embedding_dim = 250
    args.weight_decay = 1e-2
    args.margin = 1
    args.r = 0.1
    args.nf = 32

    # seq-mnist
    # args.dataset = 'seq-mnist'
    # args.buffer_size = 0
    # args.lr = 1e-3
    # args.batch_size = 128
    # args.minibatch_size = 128
    # args.n_epochs = 10
    #
    # args.nu = 0.8
    # args.eta = 1
    # args.eps = 0.1
    # args.embedding_dim = 150
    # args.weight_decay = 0
    # args.margin = 5
    # args.r = 0.1                # 半径
    # args.nf = 32

    if args.seed is not None:
        set_random_seed(args.seed)

    train(args)

if __name__ == '__main__':
    main()
