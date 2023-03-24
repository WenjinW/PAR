"""
File        :
Description :
Author      :XXX
Date        :2022/03/19
Version     :v1.1
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from copy import deepcopy
from torchsummary import summary

from .pretrained_feat_extractor import get_pretrained_feat_extractor
from .light_operations import LightOperations, LightOperationsSpace

# from automl.darts_genotypes import PRIMITIVES, Genotype
from automl.darts_operation import *


class NewCell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(NewCell, self).__init__()

        # preprocess the input (convert the number of input channel)
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim=1)


class Expert(nn.Module):
    def __init__(self, input_size, cell_nums, init_channel, genotype=None, device='cuda'):
        super(Expert, self).__init__()
        self.device = device
        self.cell_nums = cell_nums # the number of cell layers
        # params about dataset
        self.input_size = input_size 
        self.ncha = input_size[0]
        self.C = init_channel
        self.stem_multiplier = 3

        C_curr = self.stem_multiplier * self.C
        self.stem = nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.cells = nn.ModuleList([])
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = NewCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.last_c = C_prev
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)

        # print("s1 shape: {}".format(s1.shape))
        out = self.global_pooling(s1)
        # print("out shape: {}".format(out.shape))

        return out


class PARModel(nn.Module):
    def __init__(self, task_input_size, task_class_num, args=None):
        super(PARModel, self).__init__()

        self.task_input_size = task_input_size
        self.task_class_num = task_class_num

        self.num_layers = args.num_layers
        self.init_channel = args.init_channel
        self.device = args.device
        self.logger = args.logger
        self.args = args

        self.experts = nn.ModuleList([]).to(self.device)
        self.fc = nn.ModuleList([]).to(self.device)
        self.genotypes = []

        self.task2expert = []
        self.expert2task = []

        self.expert2genotype = []

        self.expert2max_train_samples = []

        # prepare pretrained feature extractor
        # if args.pretrained_feat_extractor == "resnet18":
        #     self.feat_extractor = get_pretrained_feat_extractor(args.pretrained_feat_extractor)
        # else:
        #     self.feat_extractor = None
        self.logger.info("Using relatedness feature extractor: {}".format(args.pretrained_feat_extractor))

        self.task2mean = []
        self.task2cov = []
        
        self.current_task = None
        self.current_expert = None

        self.total_num_class = 0
        self.task_class_start = []
        for _, num_class in self.task_class_num:
            self.task_class_start.append(self.total_num_class)
            self.total_num_class += num_class

        self.class_dist = torch.ones(self.total_num_class, self.total_num_class).to(self.device)
        num_task = len(self.task_class_num)
        self.task_dist = torch.zeros(num_task, num_task).to(self.device)

        # state
        self.is_multihead = False # mutlihead output, used for knowledge distillation
        self.reuse_freeze = False

    def get_info(self):
        infos = {
            "num_experts": len(self.expert2task),
            "num_genotypes": len(self.genotypes),
            'expert2task': self.expert2task,
            'expert2genotype': self.expert2genotype,
            "genotypes": self.genotypes,
            'distance': self.task_dist.cpu().numpy().tolist(),
        }

        self.logger.info(f"{infos}")

        return infos
    
    def forward(self, x):
        x = self.experts[self.current_expert](x)
        x = x.view(x.size(0), -1)
        
        if self.is_multihead:
            output = []
            for i in self.expert2task[self.current_expert]:
                output.append(self.fc[i](x))
        else:
            output = self.fc[self.current_task](x)
        return output
    
    def set_multihead(self, is_multihead=False):
        self.is_multihead = is_multihead
    
    # def extract_feat(self, x):
    #     if self.feat_extractor is None:
    #         x = torch.flatten(x, start_dim=1)
    #     else:
    #         x = self.feat_extractor(x)

    #     return x.view(x.size(0), -1)
    
    def expand(self, task_id, g, num_train_samples):
        if isinstance(g, int):
            # expert id
            genotype_id = self.expert2genotype[g]
            genotype = self.genotypes[genotype_id]
        else:
            # new genotype
            self.genotypes.append(g)
            genotype_id = len(self.genotypes) - 1
            genotype = g

        self.experts.append(
            Expert(self.task_input_size, self.num_layers, self.init_channel, genotype).to(self.device)
        )
        self.logger.info("New expert")
        utils.print_model_report(self.experts[-1], self.logger)
        self.fc.append(
            nn.Linear(self.experts[-1].last_c, self.task_class_num[task_id][1]).to(self.device)
        )

        self.task2expert.append(len(self.experts) - 1)
        self.expert2task.append([task_id])
        self.expert2genotype.append(genotype_id)
        self.expert2max_train_samples.append(num_train_samples)

    def reuse(self, task_id, reused_expert_id, num_train_samples):
        self.expert2task[reused_expert_id].append(task_id)
        self.task2expert.append(reused_expert_id)

        self.fc.append(
            nn.Linear(self.experts[reused_expert_id].last_c, self.task_class_num[task_id][1]).to(self.device)
        )
        self.expert2max_train_samples[reused_expert_id] = max(
            self.expert2max_train_samples[reused_expert_id],
            num_train_samples
        )

    def add_mean_cov(self, mean, cov=None):
        self.task2mean.append(mean)
        self.task2cov.append(cov)

    def set_current_task(self, task_id):
        self.current_task = task_id
        self.current_expert = self.task2expert[task_id]

    def requires_grad_task(self, task_id):
        expert_id = self.task2expert[task_id]

        self.requires_grad_(False)
        self.experts[expert_id].requires_grad_(True)
        self.fc[task_id].requires_grad_(True)

    def train_task(self, task_id):
        expert_id = self.task2expert[task_id]
        self.eval()
        self.experts[expert_id].train()
    
    def requires_grad_fc(self, task_id):
        self.requires_grad_(False)
        self.fc[task_id].requires_grad_(True)
    
    def train_fc(self, task_id):
        self.eval()
        # self.fc[task_id].train()
