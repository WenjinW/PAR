"""
File        :
Description :
Author      :XXX
Date        :2019/9/12
Version     :v1.0
"""
import collections
import numpy as np
import torch
import torch.nn as nn

import utils

from copy import deepcopy

from automl.darts_operation import *


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

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


class Network(nn.Module):

    def __init__(self, input_size, task_classes, cell_nums, init_channel, genotype, device):
        super(Network, self).__init__()
        self.device = device
        self.cell_nums = cell_nums
        self.input_size = input_size
        # channel number of samples
        self.ncha = input_size[0]
        self.C = init_channel
        self.task_classes = task_classes
        self.genotype = [genotype]  # the genotypes which have been used by each column
        self.unique_genotype = [genotype]  # the different genotypes which have been used by the model
        self.arch_init = 0
        self.stem_multiplier = 3
        self.num_columns = 1

        C_curr = self.stem_multiplier * self.C
        self.major = nn.ModuleList([nn.ModuleList([
            nn.Sequential(  # layer 0
                nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            )
        ])])
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            self.major[0].append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.major[0].append(nn.AdaptiveAvgPool2d(1))

        self.classifier = nn.ModuleList([])
        for t, c in self.task_classes:
            self.classifier.append(nn.Linear(C_prev, c))

    def forward(self, x, task_arch):
        """

        :param x: the feature of samples
        :param task_arch: the column number
        :return:
        """

        major_column = self.major[task_arch]

        s0 = s1 = major_column[0](x)

        for i in range(self.cell_nums):
            s0, s1 = s1, major_column[1+i](s0, s1)

        out = major_column[-1](s1)

        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def expand_column(self, genotype, device):
        """ generate a new column in major model according to genotype

        :param genotype: the micro-architecture
        :param device: the device for model
        :return:
        """
        C_curr = self.stem_multiplier * self.C
        new_column = nn.ModuleList([
            nn.Sequential(  # layer 0
                nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
                nn.BatchNorm2d(C_curr)
            ).to(device)
        ])

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
            new_column.append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        new_column.append(nn.AdaptiveAvgPool2d(1).to(device))

        self.major.append(new_column)
        self.num_columns += 1

    def del_column(self, k):
        del self.major[k]
        self.num_columns -= 1

    def unfreeze_column(self, idx):
        """ unfreeze the idx column

        :param idx: the index the of column to be unfrozen
        :return:
        """
        utils.unfreeze_model(self.major[idx])

    def unfreeze_task_layer(self, t):
        """ unfreeze the task specific layer of task t

        :param t: the idx of task
        :return:
        """
        utils.unfreeze_model(self.classifier[t])

    def get_param(self, models):
        params = []
        if 'stem' in models.keys():
            for idx in models['stem']:
                params.append({'params': self.stem[idx].parameters()})

        for i in range(self.cell_nums):
            if 'cell' + str(i) in models.keys():
                for idx in models['cell' + str(i)]:
                    params.append({'params': self.cells[i][idx].parameters()})

        # if 'pool' in models.keys():
        #     for idx in models['pool']:
        #         params.append({'params': self.global_pooling[idx].parameters()})

        if 'fc' in models.keys():
            for idx in models['fc']:
                params.append({'params': self.classifier[idx].parameters()})

        return params

    def modify_param(self, models, requires_grad=True):
        if 'stem' in models.keys():
            for idx in models['stem']:
                # print("Set stem {} as {}".format(idx, requires_grad))
                utils.modify_model(self.stem[idx], requires_grad)

        for i in range(self.cell_nums):
            if 'cell' + str(i) in models.keys():
                for idx in models['cell' + str(i)]:
                    utils.modify_model(self.cells[i][idx], requires_grad)
        # if 'pool' in models.keys():
        #     for idx in models['pool']:
        #         utils.modify_model(self.global_pooling[idx], requires_grad)

        if 'fc' in models.keys():
            for idx in models['fc']:
                # print("Set fc {} as {}".format(idx, requires_grad))
                utils.modify_model(self.classifier[idx], requires_grad)


