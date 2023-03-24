import sys,time
import numpy as np
import torch
from torch import nn
from copy import deepcopy

import utils


class Appr(object):
    """ Class implementing the Individual approach
    
    In the individual approach, each task is learned independently by a individual model
    """

    def __init__(self, model, epochs=50, batch=64, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5, lamb=0.1, writer=None, exp_name=None, device='cuda',
                 args=None):
        self.model = model

        self.device = device
        self.exp_name = exp_name
        self.logger = args.logger
        self.args = args

        self.epochs = args.epochs
        self.batch = args.batch
        self.lr = args.lr
        self.lr_min = lr_min

        self.clipgrad = clipgrad
        self.lamb = args.lamb
        self.ce = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.metric = args.metric

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                            lr=lr, weight_decay=self.lamb,
                            momentum=0.9)

    def learn(self, t, train_data, valid_data, device='cuda'):
        best_acc = 0.0
        best_val_epoch = 0
        best_model = utils.get_model(self.model)
        lr = self.lr
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
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
            valid_loss, valid_acc, valid_acc_5 = self.eval(t, valid_loader, mode='valid', device=device)
            time3 = time.time()


            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
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
        # freeze learned parameters
        self.model.freeze(t)

        return

    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.prepare(t)
        self.model.train()
        total_loss = 0
        total_1_acc = 0
        total_5_acc = 0
        total_num = 0
        # Loop batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # Forward current model
            output = self.model.forward(x)
            loss = self.criterion(t, output, y)
            # Backward
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

        return self.ce(output, targets)

    def extract_feat(self, t, data, device='cuda'):
        "compute the mixture gaussian"

        data_loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch, shuffle=False, num_workers=4, pin_memory=True)
        self.model.prepare(t)
        self.model.eval()
        feats = []
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                feat = self.model.extract_feat(x)
                feats.append(feat)
            feats = torch.cat(feats, dim=0).cpu().numpy()
        
        return feats
