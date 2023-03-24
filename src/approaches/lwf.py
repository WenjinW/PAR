import sys
import time
import numpy as np
import torch
from torch import nn
from copy import deepcopy

import utils


class Appr(object):
    """ Class implementing the Learning without Forgetting approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self, model, epochs=200, batch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5, lamb=0.1, lamb_ewc=5000, writer=None, exp_name=None, device='cuda',
                 args=None):
        self.model = model
        self.model_old = None

        self.writer = writer
        self.device = device
        self.exp_name = exp_name

        self.epochs = args.epochs
        self.batch = args.batch
        self.lr = args.lr
        self.lr_min = lr_min
        self.clipgrad = clipgrad
        self.lamb = args.lamb
        self.c_lwf = args.c_lwf
        self.ce = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = None
        self.metric = args.metric
        self.T = 2

        return

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lamb,
                               momentum=0.9)

    def learn(self, t, train_data, valid_data, device='cuda'):
        self.writer.add_text("ModelSize/Task_{}".format(t),
                             "model size = {}".format(utils.get_model_size(self.model)))
        # best_loss = np.inf
        best_acc = 0.0
        best_val_epoch = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch, shuffle=False, num_workers=4, pin_memory=True)
        # 3 training the model
        for e in range(self.epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.train_epoch(t, train_loader, device=device)
            time2 = time.time()
            # 3.3 compute valid loss
            valid_loss, valid_acc, valid_acc_5 = self.eval(t, valid_loader, mode='train', device=device)
            time3 = time.time()
            # 3.4 Adapt learning rate
            scheduler.step()
            # 3.5 update the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_val_epoch = e
                best_model = utils.get_model(self.model)
            # 3.6 logging
            print("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, 100 * train_acc, 100 * train_acc_5, valid_loss, 100 * valid_acc, 100 * valid_acc_5))
            print("| Time: train={:.1f}s, eval={:.1f}s | BestValEpoch: {:3d} |".format(
                time2-time1, time3-time2, best_val_epoch
            ))

        # 4 Restore best model
        utils.set_model_(self.model, best_model)

        # Update old
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)  # Freeze the weights
        
        return

    def train_epoch(self, t, train_loader, device='cuda'):
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        self.model.prepare(t)
        self.model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # Forward current model
            targets_old = None if t == 0 else self.model_old.multihead_forward(x)
            outputs = self.model.multihead_forward(x)
            output = outputs[t]
            loss = self.criterion(t, outputs, y, targets_old) # Note: here we need outputs
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clipgrad)
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

    def eval(self, t, test_loader, mode, device='cuda'):
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        self.model.prepare(t)
        self.model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # Forward
                outputs = self.model.multihead_forward(x)
                output = outputs[t]
                loss = self.criterion(t, outputs, y) # Note: here we need outputs

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

    def criterion(self, t, output, targets, targets_old=None):
        # Knowledge distillation loss for all previous tasks
        loss_dist = 0
        if targets_old is not None:
            for t_old in range(0, t):
                loss_dist += utils.cross_entropy(
                    output[t_old], targets_old[t_old], exp=1/self.T)
        # Cross entropy loss
        loss_ce = self.ce(output[t], targets)

        return loss_ce + self.c_lwf * loss_dist
