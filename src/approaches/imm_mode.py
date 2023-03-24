import sys
import time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

import utils


class Appr(object):
    """ Class implementing the Incremental Moment Matching (mode) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self, model, epochs=100, batch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5, c_l2=0.0001, device='cuda', args=None, writer=None, exp_name=None):
        self.device = device
        self.args = args
        self.logger = args.logger

        self.model = model
        self.model_old = None

        self.epochs = args.epochs
        self.batch = args.batch
        self.lr = args.lr
        # Grid search = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]; best was 1
        self.c_l2 = args.c_l2
        self.lamb = args.lamb
        self.clipgrad = clipgrad

        self.ce = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = self._get_optimizer()

        self.metric = args.metric

        return

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.lamb)

    def learn(self, t, train_data, valid_data, device='cuda'):
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
            train_loss, train_acc, train_acc_5 = self.train_epoch(
                t, train_loader, device=device)
            time2 = time.time()
            # 3.3 compute valid loss
            valid_loss, valid_acc, valid_acc_5 = self.eval(
                t, valid_loader, mode='train', device=device)
            time3 = time.time()

            # 3.4 Adapt learning rate
            scheduler.step()
            # 3.5 update the best model
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
        utils.set_model_(self.model, best_model)

        # 5 Model update
        if t == 0:
            self.fisher = utils.fisher_matrix_diag(t, train_loader, self.model,
                                                   self.criterion, device, self.batch)
        else:
            fisher_new = utils.fisher_matrix_diag(t, train_loader, self.model,
                                                  self.criterion, device, self.batch)
            for (n, p), (_, p_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                p = fisher_new[n]*p+self.fisher[n]*p_old
                self.fisher[n] += fisher_new[n]
                p /= (self.fisher[n] == 0).float()+self.fisher[n]

        # 6 Save old model
        self.model_old = deepcopy(self.model)
        utils.freeze_model(self.model_old)
        self.model_old.eval()

        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.prepare(t)
        self.model.train()
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # Forward current model
            output = self.model.forward(x)
            loss = self.criterion(t, output, y)
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
        self.model.prepare(t)
        self.model.eval()
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # Forward
                output = self.model.forward(x)
                loss = self.criterion(t, output, y)

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

    def criterion(self, t, output, targets):

        # L2 multiplier loss
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum((param_old-param).pow(2))/2

        # Cross entropy loss
        loss_ce = self.ce(output, targets)

        return loss_ce + self.c_l2 * loss_reg
