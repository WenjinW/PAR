"""
File        :
Description :Approach: new method
Author      :XXX
Date        :2020/10/20
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

from automl.mdenas_search import AutoSearch as AutoSearchMDL
from automl.darts_search import AutoSearch as AutoSearchDARTS
from automl.darts_genotypes import DARTS_V1
from models.TALL_model_v2 import Network


def train_epoch_p(rank, world_size, model, optimizer, loss_fn, loader):
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])

    # forward pass
    for x, y in loader:
        outputs = ddp_model(x.to(rank))
        labels = y.to(rank)

        # backward pass
        loss = loss_fn(outputs, labels)
        loss.backward()
        # update parameters
        optimizer.step()
        print(loss.item())

class Appr(object):
    """ Class implementing the TALL """
    def __init__(self, input_size=None, task_class_num=None,
                 c_epochs=100, c_batch=512, c_lr=0.025, c_lr_a=0.01, c_lamb=3e-4, c_lr_min=1e-3,
                 o_epochs=100, o_batch=512, o_lr=0.025, o_lr_a=0.01, o_lamb=3e-4, o_size=1, o_lr_min=1e-3,
                 epochs=20, batch=128,  lr=0.025, lamb=3e-4,
                 lr_factor=3, lr_patience=5, clipgrad=5,
                 writer=None, exp_name="None", device='cuda', schedule=True, args=None):
        self.args = args
        self.exp_name = exp_name
        self.logger = args.logger
        # the number of cells
        self.search_layers = args.search_layers
        self.eval_layers = args.eval_layers
        self.input_size = input_size
        self.task_class_num = task_class_num

        # model information
        self.model = None
        self.cell_archis = [] # save the cell architecture of each task
        
        self.metric = args.metric
        # the device and tensorboard writer for training
        self.writer = writer
        self.device = device
        # the hyper parameters in searching stage
        self.c_epochs = args.c_epochs
        self.c_batch = args.c_batch
        self.c_lr = args.c_lr
        self.c_lr_a = args.c_lr_a
        self.c_lamb = args.c_lamb
        # the hyper parameters in training stage
        self.epochs_encoder = args.epochs_encoder
        self.lr_encoder = args.lr_encoder
        self.epochs = args.epochs
        self.batch = args.batch
        self.lr = args.lr
        self.lamb = args.lamb
        self.clipgrad = clipgrad

        # define the search method
        if args.nas == 'mdl':
            self.auto_ml = AutoSearchMDL
        elif args.nas == 'darts':
            self.auto_ml = AutoSearchDARTS
        # define optimizer and loss function
        self.optimizer = None
        # self.optimizer_o = None
        self.ce = nn.CrossEntropyLoss()

        # self.all_relatedness = np.zeros((len(task_class_num), len(task_class_num)))
        self.all_relatedness = []
        self.fisher = []

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
        # search a cell by mdenas
        g = self.search_cell(task_id, train_data, valid_data, self.c_batch, self.c_epochs)        
        self.cell_archis.append(g)
        # build the task model according to the cell
        self.model = Network(self.input_size, self.task_class_num,
                    self.eval_layers, 36, g, self.device, self.args.input_dim_encoder).to(self.device)
        self.model.cells.append(g)

        # train autoencoder
        self.train_encoder(task_id, len(self.model.autoencoder)-1, train_data, self.args.batch_encoder, self.epochs_encoder, self.device)
        self.get_relatedness(task_id, train_data, self.args.batch_encoder, self.device)
        
        # train expert
        self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)

        
    def learn_new(self, task_id, train_data, valid_data):
        # learn a new task
        # train autoencoder
        self.model.add_encoder()
        self.train_encoder(task_id, len(self.model.autoencoder)-1, train_data, self.args.batch_encoder, self.epochs_encoder, self.device)
        
        # compute relatedness
        if self.args.relatedness_type == 'reconstruct_error':
            self.get_relatedness(task_id, train_data, self.args.batch_encoder, self.device)
        elif self.args.relatedness_type == 'tvMF':
            self.get_relatedness_tvMF(task_id, train_data, self.args.batch_encoder, self.device)
        
        # decide according to relatedness
        # find the simliar tasks
        flag = True # if all similarity is larger than 1, search new cell
        for i in self.all_relatedness[task_id]:
            if i < 1:
                flag = False

        related_expert = np.argmax(np.array(self.all_relatedness[task_id][:-1])).item()
        relatedness = self.all_relatedness[task_id][related_expert]

        if relatedness < self.args.s_reuse_g or flag: # search a new cell and build a new model
            self.logger.info("Search a new genotype to build a new expert")
            g = self.search_cell(task_id, train_data, valid_data, self.c_batch, self.c_epochs)
            self.model.add_expert(task_id, self.input_size, self.eval_layers, 36, g)
            self.cell_archis.append(g)
            self.model.expert_cell_id.append(len(self.model.cells))
            self.model.cells.append(g)
            self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)

        elif relatedness < self.args.s_finetune:
            self.logger.info("Reuse genotype of expert {} to build a new expert".format(related_expert))
            g = self.model.task_g[related_expert]
            self.model.add_expert(task_id, self.input_size, self.eval_layers, 36, g)
            self.cell_archis.append(g)
            self.model.expert_cell_id.append(self.model.expert_cell_id[related_expert])
            self.train(task_id, train_data, valid_data, self.batch, self.epochs, self.device)
        
        else:
            self.logger.info("Reuse expert {}".format(related_expert))
            self.model.remove_new_encoder()
            self.model.task_expert_id[task_id] = related_expert
            self.model.expert_to_tasks[related_expert].append(task_id)
            self.finetune(task_id, train_data, valid_data, self.batch, self.epochs, self.device)
            self.model.experts_used_num[related_expert] += 1

    def search_cell(self, t, train_data, valid_data, batch_size, nepochs):
        # print("Search cell for task")
        if self.args.pre_cell:
            genotype = deepcopy(DARTS_V1)
        else:
            auto_ml = self.auto_ml(self.search_layers, self.task_class_num[t][1],
                        self.input_size, init_channel=16, device=self.device,
                        exp_name=self.exp_name, args=self.args)
            genotype = deepcopy(auto_ml.search(0, train_data, valid_data, batch_size, nepochs))
        
        return genotype

    def train_encoder(self, t, encoder_id, train_data, batch_size, epochs, device='cuda'):
        self.model.prepare_train_encoder(encoder_id)
        best_loss = np.inf
        best_encoder = utils.get_model(self.model)
        self.optimizer = self._get_optimizer(self.lr_encoder)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        # 3 training
        patience = 0
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss = self.train_encoder_epoch(
                encoder_id, train_loader, device=device)
            time2 = time.time()
            # 3.5 Adapt learning rate
            scheduler.step()
            # 3.6 update the best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_model = utils.get_model(self.model)
                best_loss_epoch = e
                patience = 0
            else:
                patience += 1
                if patience == 5:
                    break
            
            self.logger.info("| Epoch {:3d} | Train loss={:.5f} | Best Epoch: {:3d} | Time: {:.3f}s |".format(
                e, train_loss, best_loss_epoch, time2-time1))
            self.writer.add_scalar("Task {}/Encoder Loss".format(t), train_loss, e)
            
        
        # 4 Restore best model
        utils.set_model_(self.model, best_model)
        
        return best_loss

    def train_encoder_epoch(self, encoder_id, train_loader, device='cuda'):
        self.model.encoder_train(encoder_id)

        total_loss, total_num = 0.0, 0
        for x, y in train_loader:
            x = x.to(device)
            length = x.size()[0]
            # forward
            loss = self.model.reconstruct_error(x)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            with torch.no_grad():
                total_loss += loss.item()*length
                total_num += length

        return total_loss/total_num
 
    def eval_encoder_epoch(self, encoder_id, valid_loader, device='cuda'):
        self.model.encoder_eval(encoder_id)

        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                length = x.size()[0]
                # forward
                loss = self.model.reconstruct_error(x)
                
                total_loss += loss.item()*length
                total_num += length

        return total_loss/total_num
    
    def encode_embedding(self, encoder_id, valid_loader, device='cuda'):
        self.model.encoder_eval(encoder_id)
        total_loss, total_num = 0.0, 0
        with torch.no_grad():
            emb = None
            for x, y in valid_loader:
                x = x.to(device)
                length = x.size()[0]
                # forward
                if emb is None:
                    emb = torch.sum(self.model.encode_embedding(x), dim=0)
                else:
                    emb += torch.sum(self.model.encode_embedding(x), dim=0)
                total_num += length

        return emb/total_num
    
    def get_relatedness(self, task_id, valid_data, batch_size, device='cuda'):
        errors = []
        for encoder_id in range(len(self.model.autoencoder)):
            valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True)
            
            errors.append(self.eval_encoder_epoch(encoder_id, valid_loader, device=device))
        
        # relatedness = (1 - (errors - errors[0, task_id]) / errors[0, task_id])
        # self.all_relatedness[task_id, :task_id+1] = relatedness[0, :task_id+1]
        errors = np.array(errors)
        relatedness = ((1 - (errors - errors[-1]) / errors[-1])).tolist()
        self.all_relatedness.append(relatedness)
        print(relatedness)
    
    def get_relatedness_tvMF(self, task_id, valid_data, batch_size, device='cuda'):
        emb_feat = []
        for encoder_id in range(len(self.model.autoencoder)):
            valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=batch_size, shuffle=False,
                num_workers=self.args.num_workers, pin_memory=True)
            emb_feat.append(self.encode_embedding(encoder_id, valid_loader, device=device))
            
        relatedness = []
        for encoder_id in range(len(self.model.autoencoder)):
            relatedness.append(utils.t_vMF(emb_feat[encoder_id], emb_feat[-1], self.args.tvMF_k).item())
        
        self.all_relatedness.append(relatedness)
        print(relatedness)

    def train(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.logger.info("Training stage of task {}".format(t))
        self.model.prepare_train(t)
        best_acc = 0.0
        best_valid_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.train_epoch(
                t, train_loader, device=device)
            # 3.2 eval
            time2 = time.time()
            valid_loss, valid_acc, valid_acc_5 = self.eval_bound(
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
        
        # 5 Fisher of expert
        self.fisher.append(self.fisher_matrix_diag(t, train_loader, device))

        return
    
    def train_epoch(self, t, train_loader, device='cuda'):
        self.model.train(t)
        
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

    def eval_bound(self, t, test_loader, mode, device):
        self.model.eval(t)
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

    def eval_no_bound(self, t, test_loader, mode, device):
        best_expert_id = -1
        best_loss = 9e9
        for task_id in range(len(self.model.autoencoder)):
            loss = self.eval_encoder_epoch(task_id, test_loader, device)
            if loss < best_loss:
                best_loss = loss
                best_expert_id = task_id
        self.logger.info("best_expert:{}".format(best_expert_id))
        total_loss, total_1_acc, total_5_acc = self.eval_bound(best_expert_id, test_loader, mode, device)
        return total_loss, total_1_acc, total_5_acc

    def finetune(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.old_model = deepcopy(self.model)
        self.old_model.requires_grad_(True)
        self.old_model.eval(t)
        expert_id = self.model.task_expert_id[t]

        self.logger.info("Training stage of task {}".format(t))
        self.model.prepare_train(t)
        best_acc = 0.0
        best_valid_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.finetune_epoch(
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

        # 5 update fisher
        fisher_old = {}
        for n, _ in self.model.experts[expert_id].named_parameters():
            fisher_old[n] = self.fisher[expert_id][n].clone()
        
        fisher_new = self.fisher_matrix_diag(t, train_loader, device)
        used_num = self.model.experts_used_num[expert_id]
        for n, _ in self.model.experts[expert_id].named_parameters():
            self.fisher[expert_id][n] = (fisher_new[n] + fisher_old[n] * used_num) / used_num

        return
    
    def finetune_ewc(self, t, train_data, valid_data, batch_size, epochs, device='cuda'):
        # training model for task t
        # 0 prepare
        self.old_model = deepcopy(self.model)
        self.old_model.requires_grad_(True)
        expert_id = self.model.task_expert_id[t]

        self.logger.info("Training stage of task {}".format(t))
        self.model.prepare_train(t)
        best_acc = 0.0
        best_valid_epoch = -1
        best_model = utils.get_model(self.model)

        lr = self.lr
        self.optimizer = self._get_optimizer(lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs)
        # 2 define the dataloader
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(
            valid_data, batch_size=batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        # 3 training the model
        for e in range(epochs):
            # 3.1 train
            time1 = time.time()
            train_loss, train_acc, train_acc_5 = self.finetune_epoch(
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
        # 4 Restore best model
        utils.set_model_(self.model, best_model)

        # 5 update fisher
        fisher_old = {}
        for n, _ in self.model.experts[expert_id].named_parameters():
            fisher_old[n] = self.fisher[expert_id][n].clone()
        
        fisher_new = self.fisher_matrix_diag(t, train_loader, device)
        used_num = self.model.experts_used_num[expert_id]
        for n, _ in self.model.experts[expert_id].named_parameters():
            self.fisher[expert_id][n] = (fisher_new[n] + fisher_old[n] * used_num) / used_num

        return

    def finetune_epoch(self, t, train_loader, device='cuda'):
        self.model.train(t)
        # print("Here")
        total_loss, total_1_acc, total_5_acc, total_num = 0.0, 0.0, 0.0, 0        
        # Loop batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size()[0]
            # forward
            # output = self.model.forward(x)
            outputs = self.model.multihead_forward(x)
            output = outputs[-1]
            with torch.no_grad():
                old_outputs = self.old_model.multihead_forward(x)
            # loss = self.finetune_criterion(t, output, y)
            loss = self.finetune_criterion_lwf(t, outputs, old_outputs, y)

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

    def fisher_matrix_diag(self, t, train_loader, device, sbatch=20):
        # Init
        self.model.prepare_train(t)
        expert_id = self.model.task_expert_id[t]
        fisher = {}
        with torch.no_grad():
            for n, p in self.model.experts[expert_id].named_parameters():
                fisher[n] = 0 * p.data
        # Compute
        self.model.train(t)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            length = x.size(0)
            # Forward and backward
            self.model.zero_grad()
            output = self.model.forward(x)
            loss = self.criterion(output, y)
            loss.backward()

            # Get gradients
            with torch.no_grad():
                for n, p in self.model.experts[expert_id].named_parameters():
                    if p.grad is not None:
                        fisher[n] += length * p.grad.data.pow(2)
        # Mean
        for n, _ in self.model.experts[expert_id].named_parameters():
            fisher[n] = fisher[n] / len(train_loader)

        return fisher

    def finetune_criterion_lwf(self, task_id, outputs, old_outputs, target):
        loss_reg = 0
        for i in range(len(outputs)-1):
            loss_reg += utils.cross_entropy(outputs[i], old_outputs[i], exp=1/2)
        
        return self.ce(outputs[-1], target) + self.args.c_finetune * loss_reg

    def finetune_criterion(self, task_id, output, target):
        loss_reg = 0
        expert_id = self.model.task_expert_id[task_id]
        for (name, param), (_, param_old) in zip(self.model.experts[expert_id].named_parameters(),
                self.old_model.experts[expert_id].named_parameters()):
            loss_reg += torch.sum(self.fisher[expert_id][name]*(param_old-param).pow(2))/2

        return self.ce(output, target) + self.args.c_finetune * loss_reg
