import sys, time
import numpy as np
import torch
from torch import nn

import utils

class Appr(object):

    def __init__(self, model, epochs=50, batch=128, lr=0.025, lamb=3e-4,
                 lr_min=1e-4, lr_factor=3, lr_patience=5,
                 clipgrad=5,
                 writer=None, exp_name="None", device='cuda', args=None):
        self.device = device
        self.writer = writer
        self.exp_name = exp_name
        self.args = args
        self.logger = args.logger

        self.model = model

        self.epochs = args.epochs
        self.batch = args.batch
        self.lr = args.lr
        self.lamb = args.lamb
        self.clipgrad = clipgrad

        self.metric = args.metric

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()

        return

    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, momentum=0.9, weight_decay=self.lamb)

    def learn(self, t, train_data, valid_data, device='cuda'):
        best_acc = 0.0
        best_model = utils.get_model(self.model)
        best_valid_epoch = 0
        lr = self.lr

        # train only the column for the current task
        self.model.requires_grad_(False)
        self.model.unfreeze_column(t)
        # 1 define the optimizer and scheduler
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.batch, shuffle=False, num_workers=4, pin_memory=True)

        # Loop epochs
        for e in range(self.epochs):
             # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.train_epoch(
                t, train_loader, device=device)
            # 3.2 eval
            time2 = time.time()
            valid_loss, valid_acc, valid_acc_5 = self.eval(
                t, valid_loader, mode='train', device=device)
            time3 = time.time()
            
            self.writer.add_scalars('Train_Loss/Task: {}'.format(t),
                                    {'train_loss': train_loss, 'valid_loss': valid_loss},
                                    global_step=e)
            self.writer.add_scalars('Train_Accuracy/Task: {}'.format(t),
                                    {'train_acc': train_acc * 100, 'valid_acc': valid_acc * 100},
                                    global_step=e)
            # Adapt lr
            scheduler.step()
            # update the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_valid_epoch = e
                best_model = utils.get_model(self.model)
            
            self.logger.info("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
            self.logger.info('| Time: Train={:.3f}s, Eval={:.3f}s | Best valid epoch: {:3d}|'.format(
                time2-time1, time3-time2, best_valid_epoch
            ))

        utils.set_model_(self.model, best_model)

        return

    def train_epoch(self, t, train_loader, device='cuda'):
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        self.model.eval()
        self.model.set_train_mode(t)
        # Loop batches
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # Forward
            outputs = self.model.forward(x, t)
            output = outputs[t]
            loss = self.criterion(output, y)
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
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        self.model.eval()

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # Forward
                outputs = self.model.forward(x, t)
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
