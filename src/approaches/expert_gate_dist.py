"""
File        :
Description :Approach: new method
Author      :XXX
Date        :2021/03/08
Version     :v1.1
"""
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import utils

from copy import deepcopy

class Appr(object):
    """ Class implementing the ExpertGate """
    def __init__(self, model, lr=0.025,
                 writer=None, exp_name="None", device='cuda', clipgrad=5, args=None):
        self.args = args
        
        self.writer = writer
        self.exp_name = exp_name
        self.logger = self.args.logger

        self.metric = args.metric

        # model information
        self.model = model
        
        self.local_rank = args.local_rank
        self.device = torch.device("cuda", self.local_rank)

        # the hyper parameters in training stage
        self.batch_encoder = args.batch_encoder
        self.epochs_encoder = args.epochs_encoder
        self.lr_encoder = args.lr_encoder

        self.epochs = args.epochs
        self.batch = args.batch
        self.lr = args.lr
        self.lamb = args.lamb
        self.clipgrad = clipgrad

        # define optimizer and loss function
        self.optimizer = None
        # self.optimizer_o = None
        self.ce = nn.CrossEntropyLoss()

        self.all_relatedness = []

        if args.eval_no_bound:
            self.eval = self.eval_no_bound
        else:
            self.eval = self.eval_bound

    def _get_optimizer(self, lr):
        # optimizer to train the model parameters
        if lr is None:
            lr = self.lr

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, weight_decay=self.lamb, momentum=0.9)

    def criterion(self, output, targets):

        return self.ce(output, targets)
    
    def learn(self, task_id, train_data, valid_data, device):
        """learn a task 

        """
        if task_id == 0:
            self.learn_first(task_id, train_data, valid_data)
        else:
            self.learn_new(task_id, train_data, valid_data)

    def learn_first(self, task_id, train_data, valid_data):
        """Learn the first task
        """
        # train autoencoder
        self.train_encoder(task_id, train_data, self.batch_encoder, self.epochs_encoder, self.device)
        self.get_relatedness(task_id, train_data, self.batch_encoder, self.device)

        # train expert
        self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)
        
    def learn_new(self, task_id, train_data, valid_data):
        # learn a new task
        # train autoencoder
        self.model.add_encoder()
        self.train_encoder(len(self.model.autoencoder)-1, train_data, self.args.batch_encoder, self.epochs_encoder, self.device)
        
        # compute relatedness
        self.get_relatedness(task_id, train_data, self.args.batch_encoder, self.device)
        
        # find the simliar tasks
        related_expert = np.argmax(np.array(self.all_relatedness[task_id][:-1])).item()
        relatedness = self.all_relatedness[task_id][related_expert]

        self.model.add_expert(task_id, related_expert)
        self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)

    def train_encoder(self, encoder_id, train_data, batch_size, epochs, device='cuda'):
        self.model.prepare_train_encoder(encoder_id)
        best_loss = np.inf
        if dist.get_rank() == 0:
            best_model = utils.get_model(self.model)
        
        self.optimizer = self._get_optimizer(self.lr_encoder)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, num_workers=4, pin_memory=True,
            sampler=train_sampler)
        
        # 3 training
        patience = 0
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_sampler.set_epoch(e)
            train_loss = self.train_encoder_epoch(
                encoder_id, train_loader, device=device)
            time2 = time.time()
            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if train_loss < best_loss:
                best_loss = train_loss
                if dist.get_rank() == 0:
                    best_model = utils.get_model(self.model)
                best_loss_epoch = e
                patience = 0
            else:
                patience += 1
                if patience == 10:
                    break
            if dist.get_rank() == 0:
                self.logger.info(
                    "| Epoch {:3d} | Train loss={:.5f} | Best Epoch: {:3d} | Time: {:.3f}s |".format(
                    e, train_loss, best_loss_epoch, time2-time1)
                )
        
        # 4 Restore best model
        if dist.get_rank() == 0:
            utils.set_model_(self.model, best_model)
        
        return best_loss

    def train_encoder_epoch(self, encoder_id, train_loader, device='cuda'):
        self.model.encoder_train(encoder_id)
        model = DDP(self.model.to(self.device), device_ids=[
                    self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        
        total_loss, total_num = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            length = x.size()[0]
            # forward
            loss = model.module.reconstruct_error(x)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            with torch.no_grad():
                dist.all_reduce(loss)
                total_loss += loss.item()*length
                total_num += length
        
        number = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(number, total_num)
        total_num = sum(number)

        return total_loss/total_num
 
    def eval_encoder_epoch(self, encoder_id, valid_loader, device='cuda'):
        self.model.encoder_eval(encoder_id)
        model = DDP(self.model.to(self.device), device_ids=[
                    self.local_rank], output_device=self.local_rank, find_unused_parameters=True)
        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                length = x.size()[0]
                # forward
                loss = model.module.reconstruct_error(x)
                
                dist.all_reduce(loss)
                total_loss += loss.item()*length
                total_num += length
        
        number = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(number, total_num)
        total_num = sum(number)

        return total_loss/total_num
    
    def get_relatedness(self, task_id, valid_data, batch_size, device='cuda'):
        errors = []
        for encoder_id in range(len(self.model.autoencoder)):
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_data)
            valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=batch_size,
                num_workers=4, pin_memory=True, sampler=valid_sampler)
            
            valid_sampler.set_epoch(0)
            errors.append(self.eval_encoder_epoch(encoder_id, valid_loader, device=device))
        
        # relatedness = (1 - (errors - errors[0, task_id]) / errors[0, task_id])
        # self.all_relatedness[task_id, :task_id+1] = relatedness[0, :task_id+1]
        errors = np.array(errors)
        relatedness = ((1 - (errors - errors[-1]) / errors[-1])).tolist()
        self.all_relatedness.append(relatedness)
        if dist.get_rank() == 0:
            self.logger.info(relatedness)
    
    def train(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.model.prepare_train(t)
        if dist.get_rank() == 0:
            self.logger.info("Training stage of task {}".format(t))
            best_acc = 0.0
            best_valid_epoch = -1
            best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_data)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, num_workers=4, pin_memory=True,
            sampler=train_sampler)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, num_workers=4, pin_memory=True,
            sampler=valid_sampler)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_sampler.set_epoch(e)
            train_loss, train_acc, train_acc_5 = self.train_epoch(
                t, train_loader, device=device)
            # 3.2 eval
            time2 = time.time()
            valid_sampler.set_epoch(e)
            valid_loss, valid_acc, valid_acc_5 = self.eval(
                t, valid_loader, mode='train', device=device)
            time3 = time.time()

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if dist.get_rank() == 0:
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = utils.get_model(self.model)
                    best_valid_epoch = e
            
                self.logger.info("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                    e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
                self.logger.info('| Time: Train={:.3f}s, Eval={:.3f}s | Best valid epoch: {:3d}|'.format(
                    time2-time1, time3-time2, best_valid_epoch
                ))
        if dist.get_rank() == 0:
            # 4 Restore best model
            utils.set_model_(self.model, best_model)
        
        return
    
    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train(t)
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0
        model = DDP(self.model.to(self.device), device_ids=[
                    self.local_rank], output_device=self.local_rank, find_unused_parameters=True)   
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            length = x.size()[0]
            # forward
            output = model.forward(x)
            loss = self.criterion(output, y)

            # backward
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

    def eval_bound(self, t, test_loader, mode, device):
        self.model.eval(t)
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0
        model = DDP(self.model.to(self.device), device_ids=[
                    self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                length = x.size()[0]
                # forward
                output = model.forward(x)
                # compute loss
                loss = self.criterion(output, y)

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

    def eval_no_bound(self, t, test_loader, mode, device):
        best_expert_id = -1
        best_loss = 9e9
        for task_id in range(len(self.model.autoencoder)):
            loss = self.eval_encoder_epoch(task_id, test_loader, device)
            if loss < best_loss:
                best_loss = loss
                best_expert_id = task_id
        if dist.get_rank() == 0:
            self.logger.info("best_expert:{}".format(best_expert_id))
        total_loss, total_1_acc, total_5_acc = self.eval_bound(best_expert_id, test_loader, mode, device)
        return total_loss, total_1_acc, total_5_acc
        