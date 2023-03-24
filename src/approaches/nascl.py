"""
File        :NAS Continual Learning
Description :
Author      :XXX
Date        :2019/8/1
Version     :v1.0
"""
import sys,time
import numpy as np
import torch

import utils

class Appr(object):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,args=None):
        # todo: initialize the approach

        # the initializer model for continual learning
        self.model = model
        # the number of training epochs for one task
        self.nepochs = nepochs
        # batch size for training
        self.sbatch = sbatch
        # initial learning rate
        self.lr = lr
        # the minimum value of learning rate
        self.lr_min = lr_min

        self.lr_factor = lr_factor

        self.lr_patience = lr_patience

        self.clipgrad = clipgrad
        # criterion of network
        self.criterion = torch.nn.CrossEntropyLoss()
        # optimizer
        self.optimizer = self._get_optimizer()

        return

    def _get_optimizer(self, lr=None):
        # todo: define the optimizer
        optimizer = None
        return optimizer

    def train(self,t,xtrain,ytrain,xvalid,yvalid, device='cuda'):
        # todo: train the network for a new task
        pass

    def train_epoch(self,t,x,y, device='cuda'):
        # todo: training in an epoch
        pass

    def eval(self, t, x, y, device='cuda'):
        # todo: eval the network
        pass
