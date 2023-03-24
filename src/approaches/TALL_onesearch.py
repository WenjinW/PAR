"""
File        :
Description :Approach: new method
Author      :XXX
Date        :2021/01/23
Version     :v1.1
"""
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

import utils

from copy import deepcopy

from automl.mdenas_search import AutoSearch
from models.TALL_model_onesearch import Network


class Appr(object):
    """ Class implementing the TALL """
    def __init__(self, input_size=None, task_class_num=None,
                 c_epochs=100, c_batch=512, c_lr=0.025, c_lr_a=0.01, c_lamb=3e-4, c_lr_min=1e-3,
                 o_epochs=100, o_batch=512, o_lr=0.025, o_lr_a=0.01, o_lamb=3e-4, o_size=1, o_lr_min=1e-3,
                 epochs=20, batch=128,  lr=0.025, lamb=3e-4,
                 lr_factor=3, lr_patience=5, clipgrad=5,
                 writer=None, exp_name="None", device='cuda', schedule=True, args=None):
        # the number of cells
        self.model = None
        self.search_layers = args.search_layers
        self.eval_layers = args.eval_layers
        self.input_size = input_size
        self.task_class_num = task_class_num
        # the best model architecture for each task
        self.archis = []
        self.cell_archis = []
        self.metric = args.metric

        self.schedule = schedule
        # the device and tensorboard writer for training
        self.device = device
        self.writer = writer
        self.exp_name = exp_name

        # the hyper parameters
        # cell search stage for task 0
        # the hyper parameters in cell search stage
        self.c_epochs = c_epochs
        self.c_batch = c_batch
        self.c_lr = c_lr
        self.c_lamb = c_lamb

        self.c_lr_a = c_lr_a
        # the hyper parameters in operation search stage
        self.o_epochs = o_epochs
        self.o_batch = o_batch
        self.o_lr = o_lr
        self.o_lr_a = o_lr_a
        self.o_lamb = o_lamb
        self.o_size = o_size
        # the hyper parameters in training stage
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.lamb = lamb

        self.lr_patience = lr_patience  # for dynamically update learning rate
        self.lr_factor = lr_factor  # for dynamically update learning rate
        self.clipgrad = clipgrad

        self.args = args

        if args.mode == 'search':
            # mode: search the best hyper-parameter
            # the hyper parameters in cell search stage
            self.c_epochs = args.c_epochs
            self.c_batch = args.c_batch
            self.c_lr = args.c_lr
            self.c_lr_a = args.c_lr_a
            self.c_lamb = args.c_lamb
            # the hyper parameters in operation search stage
            self.o_epochs = args.o_epochs
            self.o_batch = args.o_batch
            self.o_lr = args.o_lr
            self.o_lr_a = args.o_lr_a
            self.o_lamb = args.o_lamb
            self.o_size = args.o_size
            # the hyper parameters in training stage
            self.epochs = args.epochs
            self.batch = args.batch
            self.lr = args.lr
            self.lamb = args.lamb

            self.lr_patience = args.lr_patience  # for dynamically update learning rate
            self.lr_factor = args.lr_factor  # for dynamically update learning rate

            self.schedule = args.schedule

        # define the search method
        self.auto_ml = AutoSearch(self.search_layers, self.task_class_num[0][1], self.input_size, 36,
                                  device=self.device, writer=self.writer, exp_name=self.exp_name, args=args)
        # define optimizer and loss function
        self.optimizer = None
        self.optimizer_o = None
        self.ce = nn.CrossEntropyLoss()

    def _get_optimizer(self, lr):
        # optimizer to train the model parameters
        if lr is None:
            lr = self.lr

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, weight_decay=self.lamb, momentum=0.9)

    def _get_optimizer_o(self, lr=None):
        if lr is None:
            lr = self.o_lr

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, weight_decay=self.lamb, momentum=0.9)

    def criterion(self, output, targets):

        return self.ce(output, targets)
    
    def eval(self, t, test_loader, mode, device):
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        self.model.eval()
        self.model.set_task_model(t, self.archis[t])
        self.model.set_stage('normal')

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # forward
                outputs = self.model.forward(x)
                output = outputs[t]
                loss = self.criterion(output, y)
                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1))
                    total_5_acc += hits_5.sum().item()

                _, pred_1 = output.max(1)
                hits_1 = (pred_1 == y).float()
                total_1_acc += hits_1.sum().item()

                total_loss += loss.item()*length
                total_num += length

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def learn(self, task_id, train_data, valid_data, device):
        """learn a task 

        """
        if task_id == 0:
            self.learn_first(task_id, train_data, valid_data)
        else:
            self.learn_new(task_id, train_data, valid_data)
        print("The used time of each cell: ")
        print(self.model.used_num)

    def learn_first(self, task_id, train_data, valid_data):
        print("Learn task 0")
        # learn the first task
        # search a cell by mdenas
        self.g = self.search_cell(train_data, valid_data, self.c_batch, self.c_epochs)
        self.cell_archis.append(self.g)
        # build the task model
        self.model = Network(self.input_size, self.task_class_num, self.eval_layers, 36, self.g, self.device).to(self.device)

        utils.print_model_report(self.model)
        self.archis.append(self.model.arch_init)
        # train the task model
        self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)
        # freeze old cells
        self.model.modify_param(self.model.get_old_cells(task_id), False)

    def learn_new(self, task_id, train_data, valid_data):
        """ learn a new task
        """

        # 1 create search space
        self.model.search_space(task_id, self.schedule)
        # 2 search task model
        self.search_model(task_id, train_data, valid_data, self.o_batch, self.o_epochs)
        
        best_archi = self.model.select(task_id)
        
        self.archis.append(best_archi)

        # 3 train task model
        self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)

        # 4 freeze old model
        self.model.modify_param(self.model.get_old_cells(task_id), False)

    def search_cell(self, train_data, valid_data, batch_size, nepochs):
        print("Search cell for task 0")
        auto_ml = AutoSearch(self.search_layers, self.task_class_num[0][1],
                    self.input_size, device=self.device, writer=self.writer,
                    exp_name=self.exp_name, args=self.args)
        genotype = deepcopy(auto_ml.search(0, train_data, valid_data, batch_size, nepochs))

        return genotype

    def search_model(self, task_id, train_data, valid_data, batch_size, epochs):
        print("Search Task Model of task {}".format(task_id))
        # 1 define optimizers and scheduler
        lr = self.o_lr
        self.optimizer_o = self._get_optimizer_o(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_o, epochs, eta_min=0.001)
        # 2 define the dataloader
        num_train = len(train_data)
        indices = list(range(num_train))
        indices = shuffle(indices)
        split = int(np.floor(0.5 * num_train))
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            num_workers=4, pin_memory=True)

        # 3 define information table
        # 3.1 information of cells
        cell_e = [torch.full(pro.size(), 0, dtype=torch.long) for pro in self.model.p_cell]
        cell_a = [torch.full(pro.size(), 0.0, dtype=torch.float) for pro in self.model.p_cell]
        for cell_i_e in cell_e:
            cell_i_e[:-1] = self.o_size

         # 3 search the best model architecture
        for e in range(epochs):
            # 3.1 sample
            selected_cell = [torch.multinomial(pro, 1).item() for pro in self.model.p_cell]

            # 3.2 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.search_epoch(
                task_id, train_loader, selected_cell, self.device)
            time2 = time.time()
            valid_loss, valid_acc, valid_acc_5 = self.search_eval(
                task_id, valid_loader, selected_cell, self.device)
            time3 = time.time()
            
            print("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
            print("| Time: search time: {:.3f}s, eval time: {:.3f}s |".format(
                time2-time1, time3-time2
            ))

            # 3.3 update e and a
            for i, idx in enumerate(selected_cell):
                cell_a[i][idx] = valid_acc # update acc table
                # if the new unit is used, update epoch table
                if i > 0 and i < len(selected_cell) - 1:
                    if idx == self.model.length['cell'+str(i)]:
                        cell_e[i][idx] += 1
                if i == 0:
                    if idx == self.model.length['stem']:
                        cell_e[i][idx] += 1

            # 3.4 update the probability
            for k in range(len(self.model.p_cell)):
                cell_de_k = torch.reshape(cell_e[k], (1, -1)) - torch.reshape(cell_e[k], (-1, 1))
                cell_da_k = torch.reshape(cell_a[k], (1, -1)) - torch.reshape(cell_a[k], (-1, 1))
                # modify
                vector1 = torch.sum((cell_de_k < 0) * (cell_da_k > 0), dim=0)
                vector2 = torch.sum((cell_de_k > 0) * (cell_da_k < 0), dim=0)
                update = (vector1 - vector2).float()
                self.model.p_cell[k] += (self.o_lr_a * update)
                self.model.p_cell[k] = F.softmax(self.model.p_cell[k], dim=0)
            
            scheduler.step()

        return

    def search_epoch(self, t, train_loader, selected_cell, device='cuda'):
        self.model.train()
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        # set tht mode of models which are reused (BN)
        for i in range(self.model.length['stem']):
            self.model.stem[i].eval()
        for i in range(len(self.model.cells)):
            for k in range(self.model.length['cell' + str(i)]):
                self.model.cells[i][k].eval()
        self.model.set_selected_model(selected_cell)
        self.model.set_stage('search')
        # 2 Loop batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # Forward 
            outputs = self.model.forward(x)
            output = outputs[t]
            loss = self.criterion(output, y)
            # Backward
            self.optimizer_o.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer_o.step()

            with torch.no_grad():
                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1))
                    total_5_acc += hits_5.sum().item()

                _, pred_1 = output.max(1)
                hits_1 = (pred_1 == y).float()
                total_1_acc += hits_1.sum().item()

                total_loss += loss.item()*length
                total_num += length

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def search_eval(self, t, test_loader, selected_cell, device):
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        self.model.eval()
        self.model.set_selected_model(selected_cell)
        self.model.set_stage('search')

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # forward
                outputs = self.model.forward(x)
                output = outputs[t]
                loss = self.criterion(output, y)
                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1))
                    total_5_acc += hits_5.sum().item()

                _, pred_1 = output.max(1)
                hits_1 = (pred_1 == y).float()
                total_1_acc += hits_1.sum().item()

                total_loss += loss.item()*length
                total_num += length
        
        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def train(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        print("Training stage of task {}".format(t))
        best_acc = 0.0
        best_eval_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.train_epoch(
                t, train_loader, device=device)
            # 3.2 eval
            time2 = time.time()
            valid_loss, valid_acc, valid_acc_5 = self.eval(
                t, valid_loader, mode='train', device=device)
            time3 = time.time()

            # 3.3 Adapt learning rate
            scheduler.step()
            # 3.4 update the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
                best_valid_epoch = e
            
            print("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
            print('| Time: Train={:.3f}s, Eval={:.3f}s | Best valid epoch: {:3d}|'.format(
                time2-time1, time3-time2, best_valid_epoch
            ))
        # 4 Restore best model
        utils.set_model_(self.model, best_model)
        return
    
    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train()
        # set tht mode of models which are reused (BN)
        if t > 0:
            for i in range(self.model.length['stem']):
                if i not in self.model.model_to_train['stem']:
                    self.model.stem[i].eval()
            for i in range(len(self.model.cells)):
                for k in range(self.model.length['cell' + str(i)]):
                    if k not in self.model.model_to_train['cell' + str(i)]:
                        self.model.cells[i][k].eval()
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        self.model.set_task_model(t, self.archis[t])
        self.model.set_stage('normal')
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # forward
            outputs = self.model.forward(x)
            output = outputs[t]
            loss = self.criterion(output, y)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            with torch.no_grad():
                if self.metric == 'top5':
                    _, pred_5 = torch.topk(output, 5, sorted=True)
                    hits_5 = (pred_5 == y.reshape(pred_5.size()[0], -1))
                    total_5_acc += hits_5.sum().item()

                _, pred_1 = output.max(1)
                hits_1 = (pred_1 == y).float()
                total_1_acc += hits_1.sum().item()

                total_loss += loss.item()*length
                total_num += length

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num
