"""
File        :
Description :Multinomial distribution for efficient Neural Architecture Search
Author      :XXX
Date        :2019/9/19
Version     :v1.0
"""
import numpy as np
from sklearn.utils import shuffle
import torch
import logging
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.nn.functional as F
import torch.utils

import automl.darts_utils_cnn as utils
from automl.mdenas_basicmodel import BasicNetwork
from automl.darts_model import Network


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices):
        super(MyDataset).__init__()
        self.data = data
        self.indices = indices
        self.length = len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return self.data[i]

    def __len__(self):
        return self.length


class AutoSearch(object):
    # Implements a NAS methods MdeNAS
    def __init__(self, num_cells, num_class=10, input_size=None, init_channel=36, lr=0.025, lr_a=3e-4, lr_min=0.001, momentum=0.9,
                 weight_decay=3e-4, weight_decay_a=1e-3, grad_clip=5, unrolled=False,
                 device='cuda: 0', writer=None, exp_name=None, save_name='EXP', args=None):
        self.args = args
        self.logger = args.logger
        
        self.num_cells = num_cells
        self.num_classes = num_class
        self.input_size = input_size
        self.init_channel = init_channel

        self.writer = writer
        self.exp_name = exp_name
        self.metric = args.metric

        self.local_rank = args.local_rank
        self.device = torch.device("cuda", self.local_rank)

        self.lr = args.c_lr
        self.lr_a = args.c_lr_a
        self.lr_min = lr_min
        self.momentum = momentum
        self.weight_decay = args.c_lamb
        self.c_epochs = args.c_epochs
        self.c_batch = args.c_batch

        self.grad_clip = grad_clip
        self.save_name = save_name

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.model = BasicNetwork(self.input_size[0], self.init_channel, self.num_classes, self.num_cells, self.criterion,
                                  device=self.device).to(self.device)
        
        # self.logger.info("param size = %fMB",
        #              utils.count_parameters_in_MB(self.model))

        self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    def search(self, t, train_data, valid_data, batch_size, nepochs):
        """ search a model genotype for the given task

        :param train_data: the dataset of training data of the given task
        :param valid_data: the dataset of valid data of the given task
        :param batch_size: the batch s[ize of training
        :param nepochs: the number of training epochs
        :return:
            genotype: the selected architecture for the given task
        """
        # dataloader of training data
        num_train = len(train_data)
        indices = list(range(num_train))
        indices = shuffle(indices)
        split = int(np.floor(0.5 * num_train))

        
        search_train_data = MyDataset(train_data, indices[:split])
        train_sampler = torch.utils.data.distributed.DistributedSampler(search_train_data)
        train_queue = torch.utils.data.DataLoader(
            search_train_data, batch_size=batch_size,
            sampler=train_sampler,
            pin_memory=True, num_workers=4)
        search_valid_data = MyDataset(train_data, indices[split:num_train])
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            search_valid_data)
        valid_queue = torch.utils.data.DataLoader(
            search_valid_data, batch_size=batch_size,
            sampler=valid_sampler,
            pin_memory=True, num_workers=4)

        # train_queue = torch.utils.data.DataLoader(
        #     train_data, batch_size=batch_size,
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(
        #         indices[:split]),
        #     pin_memory=True, num_workers=4)
        # dataloader of valid date
        # valid_queue = torch.utils.data.DataLoader(
        #     train_data, batch_size=batch_size,
        #     sampler=torch.utils.data.sampler.SubsetRandomSampler(
        #         indices[split:num_train]),
        #     pin_memory=True, num_workers=4)
        # the scheduler of learning rate of model parameters optimizer
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, nepochs, eta_min=self.lr_min)
        h_e = {
            'normal': torch.full((self.model.num_edges, self.model.num_ops), 0, dtype=torch.long),
            'reduce': torch.full((self.model.num_edges, self.model.num_ops), 0, dtype=torch.long)
        }
        h_a = {
            'normal': torch.full((self.model.num_edges, self.model.num_ops), 0.0),
            'reduce': torch.full((self.model.num_edges, self.model.num_ops), 0.0)
        }
        for epoch in range(nepochs):
            # 0 prepare
            lr = scheduler.get_last_lr()[0]
            # 1 sample
            p_n = self.model.probability()["normal"]
            p_r = self.model.probability()["reduce"]
            selected_ops = {
                'normal': torch.multinomial(p_n, 1).view(-1),
                'reduce': torch.multinomial(p_r, 1).view(-1)
            }
            # 2 train
            time1 = time.time()
            train_sampler.set_epoch(epoch)
            train_obj, train_acc, train_acc_5 = self.train(
                train_queue, selected_ops)
            time2 = time.time()
            valid_sampler.set_epoch(epoch)
            valid_obj, valid_acc, valid_acc_5 = self.eval(
                valid_queue, selected_ops)
            time3 = time.time()

            # 3 update h_e and h_a
            for cell_type in ['normal', 'reduce']:
                # for each edge
                for i, idx in enumerate(selected_ops[cell_type]):
                    h_e[cell_type][i][idx] += 1
                    h_a[cell_type][i][idx] = valid_acc

            # 4 update the probability
            for k in range(self.model.num_edges):
                dh_e_k = {
                    'normal': torch.reshape(h_e['normal'][k], (1, -1)) - torch.reshape(h_e['normal'][k], (-1, 1)),
                    'reduce': torch.reshape(h_e['reduce'][k], (1, -1)) - torch.reshape(h_e['reduce'][k], (-1, 1))
                }
                dh_a_k = {
                    'normal': torch.reshape(h_a['normal'][k], (1, -1)) - torch.reshape(h_a['normal'][k], (-1, 1)),
                    'reduce': torch.reshape(h_a['reduce'][k], (1, -1)) - torch.reshape(h_a['reduce'][k], (-1, 1))
                }
                for cell_type in ['normal', 'reduce']:
                    vector1 = torch.sum(
                        (dh_e_k[cell_type] < 0) * (dh_a_k[cell_type] > 0), dim=0)
                    vector2 = torch.sum(
                        (dh_e_k[cell_type] > 0) * (dh_a_k[cell_type] < 0), dim=0)
                    self.model.p[cell_type][k] += (self.lr_a *
                                                   (vector1-vector2).float())
                    self.model.p[cell_type][k] = F.softmax(
                        self.model.p[cell_type][k], dim=0)
            time4 = time.time()
            if dist.get_rank() == 0:
                self.logger.info("| Sample {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                    epoch, train_obj, train_acc*100, train_acc_5*100, valid_obj, valid_acc*100, valid_acc_5*100))
                self.logger.info("| lr: {:.4f} | train time: {:.3f}s, eval time: {:.3f}s, update time: {:.3f}s |".format(
                    lr, time2-time1, time3-time2, time4-time3
                ))
                # print("| Sample {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                #     epoch, train_obj, train_acc*100, train_acc_5*100, valid_obj, valid_acc*100, valid_acc_5*100))
                # print("| lr: {:.4f} | train time: {:.3f}s, eval time: {:.3f}s, update time: {:.3f}s |".format(
                #     lr, time2-time1, time3-time2, time4-time3
                # ))

            # adjust learning according the scheduler
            scheduler.step()

        # print("The best architecture is", self.model.genotype())
        return self.model.genotype()

    def train(self, train_queue, selected_ops):
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        self.model.train()
        self.model.set_ops(selected_ops)
        model = DDP(self.model.to(self.device), device_ids=[
                    self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        for step, (x, y) in enumerate(train_queue):
            x, y = x.to(self.device), y.to(self.device)
            length = x.size()[0]
            self.optimizer.zero_grad()
            # output = self.model(x)
            output = model(x)
            loss = self.criterion(output, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            with torch.no_grad():
                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1))
                    dist.all_reduce(hits_5)
                    total_5_acc += hits_5.sum().item()

                _, pred_1 = output.max(1)
                hits_1 = (pred_1 == y).float()
                dist.all_reduce(hits_1)
                total_1_acc += hits_1.sum().item()

                dist.all_reduce(loss)
                total_loss += loss.item()*length
                total_num += length

        number = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(number, total_num)
        total_num = sum(number)

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def eval(self, valid_queue, selected_ops):
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        self.model.eval()
        self.model.set_ops(selected_ops)
        model = DDP(self.model.to(self.device), device_ids=[
                    self.local_rank], output_device=self.local_rank, find_unused_parameters=True)

        with torch.no_grad():
            for step, (x, y) in enumerate(valid_queue):
                x, y = x.to(self.device), y.to(self.device)
                length = x.size()[0]
                output = model(x)
                loss = self.criterion(output, y)

                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1))
                    dist.all_reduce(hits_5)
                    total_5_acc += hits_5.sum().item()

                _, pred_1 = output.max(1)
                hits_1 = (pred_1 == y).float()
                dist.all_reduce(hits_1)
                total_1_acc += hits_1.sum().item()

                dist.all_reduce(loss)
                total_loss += loss.item()*length
                total_num += length
        
        number = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(number, total_num)
        total_num = sum(number)

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def create_model(self, archi, num_cells, input_size, task_classes, init_channel):

        return Network(input_size, task_classes, num_cells, init_channel, archi, device=self.device)
