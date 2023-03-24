"""
File        :
Description :Approach: new method
Author      :XXX
Date        :2022/03/19
Version     :v1.1
"""
import numpy as np
import os
from sklearn.utils import shuffle
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
import utils_draw

from copy import deepcopy

from models.TALL_model_v2 import Network
from automl.mdenas_search import AutoSearch as AutoSearchMDL
from automl.darts_search import AutoSearch as AutoSearchDARTS


class Appr(object):
    """ Class implementing the TALL """
    def __init__(self, model, input_size, task_class_num,
                 writer=None, exp_name="None", device='cuda', args=None):
        self.args = args
        self.exp_name = exp_name
        self.logger = args.logger

        self.task_class_num = task_class_num
        self.input_size = input_size

        self.num_layers = args.num_layers

        # model information
        self.model = model        
        self.metric = args.metric
        # the device and tensorboard writer for training
        self.writer = writer
        self.device = device
        # the hyper parameters in searching stage
        self.sample_epochs = args.sample_epochs


        # the hyper parameters in training stage
        self.epochs = args.epochs
        self.batch = args.batch
        self.eval_batch = args.batch if args.eval_batch == 0 else args.eval_batch

        self.lr = args.lr
        self.lamb = args.lamb
        self.clipgrad = args.clipgrad
        self.coefficient_kl = args.coefficient_kl

        self.task_relatedness_method = args.task_relatedness_method
        self.reuse_threshold = args.reuse_threshold

        # define the search method
        if args.nas == 'mdl':
            self.auto_ml = AutoSearchMDL
        elif args.nas == 'darts':
            self.auto_ml = AutoSearchDARTS

        # define optimizer and loss function
        self.optimizer = None
        self.ce = nn.CrossEntropyLoss()

    def _get_optimizer(self, lr, few_data=False):
        # optimizer to train the model parameters
        if lr is None:
            lr = self.lr
        if few_data:
            lamb = self.lamb * 10
        else:
            lamb = self.lamb

        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                               lr=lr, weight_decay=lamb, momentum=0.9)

    def criterion(self, output, targets):

        return self.ce(output, targets)
    
    def task_relatedness_cos(self, task_id, p_task_id):
        class_start = self.model.task_class_start[task_id]
        class_num = self.task_class_num[task_id][1]

        p_class_start = self.model.task_class_start[p_task_id]
        p_class_num = self.task_class_num[p_task_id][1]

        task_dist = 0
        for i in range(class_num):
            for j in range(p_class_num):
                dist = utils.cosine_similarity(self.model.task2mean[task_id][i], self.model.task2mean[p_task_id][j])
                self.model.class_dist[i + class_start][j + p_class_start] = dist
                self.model.class_dist[j + p_class_start][i + class_start] = dist
                task_dist += dist

        task_dist /= (class_num * p_class_num)

        return task_dist
    
    def task_relatedness_knnkl(self, task_id, p_task_id, all_feats):
        """
        Params:
            all_feat: shape C x N_c x D
        """
        p_feat_means = self.model.task2mean[p_task_id] # C x D
        feat_means = self.model.task2mean[task_id] # C x D

        p_feat_means = torch.stack(p_feat_means, dim=0)

        task_dist = 0
        n = 0
        for i in range(len(feat_means)):

            n += all_feats[i].shape[0]

            dist_in = all_feats[i] - feat_means[i] # N_c x D
            dist_in = torch.sqrt(torch.sum(dist_in ** 2, dim=-1)) # N_c
            
            # N_c x C x D
            dist_out = torch.unsqueeze(all_feats[i], dim=1) - torch.unsqueeze(p_feat_means, dim=0)

            dist_out, _ = torch.min(torch.sqrt(torch.sum(dist_out ** 2, dim=-1)), dim=-1)

            dist = torch.mean(torch.log(dist_out / dist_in))

            task_dist += dist

        return task_dist

    def get_relatedness(self, task_id, feats):
        """Compute relatedness
        
        """
        # for p_task_id in range(task_id):
        for p_task_id in range(task_id + 1):
            # task_dist = self.task_relatedness_cos(task_id, p_task_id)
            task_dist = self.task_relatedness_knnkl(task_id, p_task_id, feats)
            self.model.task_dist[task_id][p_task_id] = task_dist
            self.model.task_dist[p_task_id][task_id] = task_dist
          
        utils_draw.draw_heatmap(
            self.model.task_dist.cpu().numpy(),
            row_labels=np.arange(self.model.task_dist.shape[1]),
            col_labels=np.arange(self.model.task_dist.shape[0]),
            annotate=True,
            annotate_format="{x:.3f}",
            cbarlabel="Task Distance",
            path=os.path.join(self.args.result_path, "task_similarity.pdf"),
            size=self.args.task_dist_image_size,
            format="pdf"
        )
    
    def strategy(self, task_id):
        """ Find the expert to be reused. If not found, return -1.
        
        """
        expert = -1
        min_dist = None
        for expert_id, p_tasks in enumerate(self.model.expert2task):
            d = self.model.task_dist[task_id, p_tasks]
            self.logger.info("shape d: {}".format(d.shape))
            if self.task_relatedness_method == "mean":
                s = torch.mean(d).item()
            elif self.task_relatedness_method == "max":
                s = torch.max(d).item()
            elif self.task_relatedness_method == "min":
                s = torch.min(d).item()
            else:
                raise Exception("Unknown reuse strategy !!!")
            if min_dist is None:
                min_dist = s
                expert = expert_id
            elif s < min_dist:
                min_dist = s
                expert = expert_id

        self.args.logger.info("min_dist: {}".format(min_dist))
        
        if min_dist <= self.args.reuse_threshold:
            return "reuse", expert
        elif min_dist <= self.args.reuse_cell_threshold:
            return "reuse cell", expert
        else:
            return "new", expert
            
    def learn(self, task_id, train_data, valid_data, device):
        """learn a task 

        """
        self.logger.info("Learning task: {}".format(task_id))

        # get mean and feats
        self.logger.info("Get feat means and covs for task: {}".format(task_id))
        feat_means, all_feats = self.get_mean_cov(
            task_id, valid_data, batch_size=self.eval_batch, device=device)
        num_train_samples = len(train_data)

        self.model.add_mean_cov(feat_means)
        self.get_relatedness(task_id, all_feats)

        if task_id == 0:
            # train
            self.logger.info("Searching for task: {}".format(task_id))
            g = self.search_cell(task_id, train_data, self.batch, self.sample_epochs)
            self.model.expand(task_id, g, num_train_samples)
            self.logger.info("Training for task: {}".format(task_id))
            self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)
        else:
            strategy, expert_id = self.strategy(task_id)
            if strategy == "reuse":
                self.logger.info("Reuse expert {} for task {}".format(expert_id, task_id))
                self.model.reuse(task_id, expert_id, num_train_samples)
                if num_train_samples < 0.5 * self.model.expert2max_train_samples[expert_id]:
                    self.logger.info("Directly reuse for task: {}".format(task_id))
                    self.reuse(task_id, train_data,valid_data, self.batch, self.epochs, self.device)
                else:
                    self.logger.info("Transfer for task: {}".format(task_id))
                    self.transfer(task_id, train_data,valid_data, self.batch, self.epochs, self.device)
            elif strategy == "reuse cell":
                self.logger.info("Reuse architecture of expert {} for task {}".format(expert_id, task_id))
                g = self.model.genotypes[self.model.expert2genotype[expert_id]]
                self.model.expand(task_id, g, num_train_samples)
                self.logger.info("Training for task: {}".format(task_id))
                self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)
            else: # new
                self.logger.info("Searching architecture for task {}".format(task_id))
                g = self.search_cell(task_id, train_data, self.batch, self.sample_epochs)
                self.model.expand(task_id, g, num_train_samples)
                self.logger.info("Training for task: {}".format(task_id))
                self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)

    def search_cell(self, t, train_data, batch_size, nepochs):
        # print("Search cell for task")
        auto_ml = self.auto_ml(self.num_layers, self.task_class_num[t][1],
                    self.input_size, init_channel=64, device=self.device,
                    exp_name=self.exp_name, args=self.args)
        genotype = deepcopy(auto_ml.search(t, train_data, batch_size, nepochs))
        
        return genotype
    
    def train(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.logger.info("Training stage of task {}".format(t))
        self.model.set_current_task(t)
        self.model.requires_grad_task(t) # freezing other parameters

        best_acc = 0.0
        best_valid_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr, len(train_data) < 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.eval_batch, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
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

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
                best_valid_epoch = e
            
            self.logger.info("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
            self.logger.info('| Time: Train={:.3f}s, Eval={:.3f}s | Best valid epoch: {:3d}|'.format(
                time2-time1, time3-time2, best_valid_epoch
            ))

            self.writer.add_scalars("Task {}/Loss".format(t), {'train': train_loss, 'valid': valid_loss}, e)
            self.writer.add_scalars("Task {}/Acc".format(t), {'train': train_acc, 'valid': valid_acc}, e)
            self.writer.add_scalars("Task {}/Acc 5".format(t), {'train': train_acc_5, 'valid': valid_acc_5}, e)
        
        # 4 Restore best model
        utils.set_model_(self.model, best_model)

        return
    
    def train_epoch(self, t, train_loader, device='cuda:0'):
        self.model.set_current_task(t)
        self.model.requires_grad_task(t)
        self.model.train_task(t) # replace original model.train()
        
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # forward
            output = self.model.forward(x)
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

    def reuse(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.logger.info("Training stage of task {}".format(t))
        self.model.set_current_task(t)
        # self.model.requires_grad_task(t) 
        self.model.requires_grad_fc(t) # freezing other parameters, only fc is optimized

        best_acc = 0.0
        best_valid_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        
        self.optimizer = self._get_optimizer(lr, len(train_data) < 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.eval_batch, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.reuse_epoch(
                t, train_loader, device=device)
            # 3.2 eval
            time2 = time.time()
            valid_loss, valid_acc, valid_acc_5 = self.eval(
                t, valid_loader, mode='train', device=device)
            time3 = time.time()

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
                best_valid_epoch = e
            
            self.logger.info("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
            self.logger.info('| Time: Train={:.3f}s, Eval={:.3f}s | Best valid epoch: {:3d}|'.format(
                time2-time1, time3-time2, best_valid_epoch
            ))

            self.writer.add_scalars("Task {}/Loss".format(t), {'train': train_loss, 'valid': valid_loss}, e)
            self.writer.add_scalars("Task {}/Acc".format(t), {'train': train_acc, 'valid': valid_acc}, e)
            self.writer.add_scalars("Task {}/Acc 5".format(t), {'train': train_acc_5, 'valid': valid_acc_5}, e)
        
        # 4 Restore best model
        utils.set_model_(self.model, best_model)

        return
    
    def reuse_epoch(self, t, train_loader, device='cuda:0'):
        self.model.set_current_task(t)
        self.model.requires_grad_fc(t)
        self.model.train_fc(t) # replace original model.train()
        
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # forward
            output = self.model.forward(x)
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

    def eval(self, t, test_loader, mode="train", device="cuda:0"):
        self.model.set_current_task(t)
        self.model.eval()
        
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                length = x.size()[0]
                # forward
                output = self.model.forward(x)
                # compute loss
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

    def get_mean_cov(self, t, data, batch_size, device):
        """compute mean and cov for features of data extracted by the expert of task t

        """
        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False,
            num_workers=self.args.num_workers, pin_memory=True)
        self.model.requires_grad_(False)
        self.model.eval()
        # self.model.set_current_task(t)

        class_num = self.task_class_num[t][1]
        labels = torch.arange(class_num).view(-1, 1).to(device)
        all_feats = [[] for _ in range(class_num)]

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                # forward
                feat = self.model.extract_feat(x)

                index = labels == y.view(1, -1) # CxC
                for i in range(class_num):
                    all_feats[i].append(feat[index[i]])
            
            
            all_feats_cat = [torch.cat(feats, axis=0) for feats in all_feats]
            feat_means = [torch.mean(feat, dim=0) for feat in all_feats_cat]

        return feat_means, all_feats_cat

    def transfer(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.old_model = deepcopy(self.model)
        self.old_model.requires_grad_(False)
        self.old_model.eval()
        self.old_model.set_current_task(t)
        self.old_model.set_multihead(True)

        self.model.set_current_task(t)
        self.model.requires_grad_task(t)
        self.model.train_task(t)

        self.logger.info("Training stage of task {}".format(t))
        best_acc = 0.0
        best_valid_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr, len(train_data) < 100)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=self.eval_batch, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.transfer_epoch(
                t, train_loader, device=device)
            # 3.2 eval
            time2 = time.time()
            valid_loss, valid_acc, valid_acc_5 = self.eval(
                t, valid_loader, mode='train', device=device)
            time3 = time.time()

            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_model = utils.get_model(self.model)
                best_valid_epoch = e
            
            self.logger.info("| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}%, acc5={:5.1f}% | Valid: loss={:.3f}. acc={:5.1f}%, acc5={:5.1f}% |".format(
                e, train_loss, train_acc*100, train_acc_5*100, valid_loss, valid_acc*100, valid_acc_5*100))
            self.logger.info('| Time: Train={:.3f}s, Eval={:.3f}s | Best valid epoch: {:3d}|'.format(
                time2-time1, time3-time2, best_valid_epoch
            ))

            self.writer.add_scalars("Task {}/Loss".format(t), {'train': train_loss, 'valid': valid_loss}, e)
            self.writer.add_scalars("Task {}/Acc".format(t), {'train': train_acc, 'valid': valid_acc}, e)
            self.writer.add_scalars("Task {}/Acc 5".format(t), {'train': train_acc_5, 'valid': valid_acc_5}, e)
        
        # 4 Restore best model
        utils.set_model_(self.model, best_model)

        return

    def transfer_epoch(self, t, train_loader, device='cuda'):
        self.model.set_current_task(t)
        self.model.train_task(t) # replace original model.train()
        self.model.set_multihead(True)

        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # forward
            outputs = self.model.forward(x)
            output = outputs[-1]
            with torch.no_grad():
                old_outputs = self.old_model.forward(x)
            loss = self.transfer_criterion(outputs, old_outputs, y)

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

        self.model.set_multihead(False)

        return total_loss/total_num, total_1_acc/total_num, total_5_acc/total_num

    def transfer_criterion(self, outputs, old_outputs, y):
        loss = 0
        for i in range(len(outputs) - 1):
            loss += utils.cross_entropy(outputs[i], old_outputs[i], exp=1/2)
        
        loss += self.ce(outputs[-1], y)

        return loss
