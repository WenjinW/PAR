import sys,time
import numpy as np
import torch
from torch import nn
from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import utils


class Appr(object):
    """ Class implementing the Individual approach
    
    In the individual approach, each task is learned independently by a individual model
    """

    def __init__(self, model, epochs=50, batch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5, lamb=0.1, writer=None, exp_name=None, device='cuda',
                 args=None):
        self.model = model
        self.initial_model=deepcopy(model)

        self.logger = args.logger
        self.local_rank = args.local_rank
        self.device = torch.device("cuda", self.local_rank)
        self.exp_name = exp_name

        self.epochs = args.epochs
        self.batch_size = args.batch
        self.lr = args.lr
        self.lr_min = lr_min
        self.clipgrad = clipgrad
        self.lamb = args.lamb
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.metric = args.metric

        return

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lamb,
                               momentum=0.9)

    def learn(self, t, train_data, valid_data, device='cuda'):
        if dist.get_rank() == 0:
            self.logger.info("Training stage of task {}".format(t))
            self.model=deepcopy(self.initial_model) # Restart model
            best_acc = 0.0
            best_valid_epoch = -1
            best_model = utils.get_model(self.model)
        lr = self.lr
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        # 2 define the dataloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_data)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, num_workers=4, pin_memory=True,
            sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch_size, num_workers=4, pin_memory=True,
            sampler=valid_sampler)

        # 3 training the model
        for e in range(self.epochs):
            # 3.1 train
            time1 = time.time()
            train_sampler.set_epoch(e)
            train_loss, train_acc, train_acc_5 = self.train_epoch(t, train_loader, device=device)
            time2 = time.time()
            # 3.3 compute valid loss
            valid_sampler.set_epoch(e)
            valid_loss, valid_acc, valid_acc_5 = self.eval(t, valid_loader, mode='train', device=device)
            time3 = time.time()

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if dist.get_rank() == 0:
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_val_epoch = e
                    best_model = utils.get_model(self.model)
                
                # 3.6 logging
                self.logger.info("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% |".format(
                    e, train_loss, 100 * train_acc, 100 * train_acc_5, valid_loss, 100 * valid_acc, 100 * valid_acc_5))
                self.logger.info("| Time: train={:.1f}s, eval={:.1f}s | BestValEpoch: {:3d} |".format(
                    time2-time1, time3-time2, best_val_epoch
                ))


        # 4 Restore best model
        if dist.get_rank() == 0:
            utils.set_model_(self.model, best_model)

        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.prepare(t)
        self.model.train()
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        model = DDP(self.model.to(self.device), device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        # Loop batches
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            length = x.size()[0]
            # Forward current model
            output = model.forward(x)
            loss = self.criterion(t, output, y)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            with torch.no_grad():
                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1)).sum()
                    dist.all_reduce(hits_5)
                    total_5_acc += hits_5.item()

                _, pred_1 = output.max(1)
                hits_1 = ((pred_1 == y).float()).sum()
                dist.all_reduce(hits_1)
                total_1_acc += hits_1.item()

                dist.all_reduce(loss)
                total_loss += loss.item()*length 
                total_num += length
        
        number = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(number, total_num)
        total_num = sum(number)

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def eval(self, t, test_loader, mode, device='cuda'):
        self.model.prepare(t)
        self.model.eval()
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        model = DDP(self.model.to(self.device), device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                length = x.size()[0]
                # Forward
                output = model.forward(x)
                loss = self.criterion(t, output, y)

                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1)).sum()
                    dist.all_reduce(hits_5)
                    total_5_acc += hits_5.item()

                _, pred_1 = output.max(1)
                hits_1 = ((pred_1 == y).float()).sum()
                dist.all_reduce(hits_1)
                total_1_acc += hits_1.item()

                dist.all_reduce(loss)
                total_loss += loss.item()*length 
                total_num += length
        
        number = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(number, total_num)
        total_num = sum(number)
        
        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def criterion(self, t, output, targets):

        return self.ce(output, targets)

