"""
File        :
Description :
Author      :XXX
Date        :2019/8/17
Version     :v1.0
"""
import numpy as np
import torch
import torch.nn as nn

import utils

from copy import deepcopy

from automl.darts_operation import *



class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

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
        # todo: test
        # print("num_cell: ", self._steps)
        # print("len_indices", len(self._indices))
        for i in range(self._steps):
            # print("idx: ", self._indices[2 * i])
            h1 = states[self._indices[2 * i]]
            # print("idx: ", self._indices[2 * i + 1])
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
        self.ncha = input_size[0]
        self.C = init_channel
        self.task_classes = task_classes

        self.stem_multiplier = 3
        self.length = {'stem': 1}
        self.arch_init = {'stem': [0], 'fc': [0]}

        C_curr = self.stem_multiplier * self.C
        self.stem = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )])

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.cells = nn.ModuleList([])
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(nn.ModuleList([cell]))
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            self.arch_init['cell' + str(i)] = [0]
            self.length['cell'+str(i)] = 1

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.ModuleList([])
        for t, c in self.task_classes:
            self.classifier.append(nn.Linear(C_prev, c))

        # parameter for architecture search
        self.p = None
        # The new models, model to train
        self.new_models = None
        self.model_to_train = None

    def forward(self, x, t, task_arch=None, path=None):
        arch_stem = None
        if task_arch is not None:
            arch_stem = task_arch['stem'][0]
        elif path is not None:
            arch_stem = path[0]
        s0 = s1 = self.stem[arch_stem](x)

        for i, cell in enumerate(self.cells):
            arch_cell = None
            if task_arch is not None:
                arch_cell = task_arch['cell'+str(i)][0]
            elif path is not None:
                arch_cell = path[i+1]
            s0, s1 = s1, cell[arch_cell](s0, s1)

        out = self.global_pooling(s1)

        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def expand(self, t, genotype, device='cuda'):
        # expand the network to a super model
        # 0 clean the probability
        self.p = []
        # 1 expand stem
        # 1.1 reuse: reuse blocks
        # 1.2 new: create a new block
        C_curr = self.stem_multiplier * self.C
        self.stem.append(nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        ).to(device))
        # 1.3 generate action parameter
        num_l = self.length['stem'] + 1
        self.p.append(torch.full((num_l,), 1 / num_l))

        # 2 expand cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # 2.1 reuse: reuse blocks
            # 2.2 new: create new block according to the new cell
            multiplier = None
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
            multiplier = cell.multiplier
            self.cells[i].append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            # 2.3 generate action parameter
            # for each cell: reuse + create
            num_l = self.length['cell'+str(i)] + 1
            self.p.append(torch.full((num_l,), 1 / num_l))

        # 3 get the new modules
        self.get_new_model(t=t)

    def get_new_model(self, t):
        # get new model (update and search)
        new_models = {'stem': [], 'fc': []}
        # 1 stem
        c = self.length['stem']
        new_models['stem'].append(c)
        # 2 cells
        for i in range(self.cell_nums):
            new_models['cell' + str(i)] = [self.length['cell' + str(i)]]

        # 3 classifier
        new_models['fc'].append(t)
        self.new_models = new_models

    def get_param(self, models):
        params = []
        if 'stem' in models.keys():
            for idx in models['stem']:
                params.append({'params': self.stem[idx].parameters()})

        for i in range(self.cell_nums):
            if 'cell' + str(i) in models.keys():
                for idx in models['cell' + str(i)]:
                    params.append({'params': self.cells[i][idx].parameters()})

        if 'fc' in models.keys():
            for idx in models['fc']:
                params.append({'params': self.classifier[idx].parameters()})

        return params

    def modify_param(self, models, requires_grad=True):
        if 'stem' in models.keys():
            for idx in models['stem']:
                utils.modify_model(self.stem[idx], requires_grad)

        for i in range(self.cell_nums):
            if 'cell' + str(i) in models.keys():
                for idx in models['cell' + str(i)]:
                    utils.modify_model(self.cells[i][idx], requires_grad)

        if 'fc' in models.keys():
            for idx in models['fc']:
                utils.modify_model(self.classifier[idx], requires_grad)

    def search_forward(self, x, selected_ops):
        # 1 stem
        # 1.1 stem
        s0 = s1 = self.stem[selected_ops[0]](x)

        # 2 cells layer
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell[selected_ops[1 + i]](s0, s1)

        out = self.global_pooling(s1)
        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def select(self, t):
        # 1 define the container of new models to train and the best submodel
        model_to_train = {'stem': [], 'fc': []}
        best_archi = {'stem': [], 'fc': []}
        # 2 stem
        # 2.1 select the best architecture for stem
        v, idx = torch.max(self.p[0], dim=0)
        c = self.length['stem']
        if idx < c:  # reuse
            best_archi['stem'].append(idx)
        elif idx == c:  # update
            best_archi['stem'].append(c)
            model_to_train['stem'].append(c)
        # 2.2 delete for stem
        if idx != c:
            del self.stem[c]
        # 2.3 update the length
        self.length['stem'] = len(self.stem)

        # 3 cells layer
        for i in range(self.cell_nums):
            name = 'cell' + str(i)
            model_to_train[name] = []
            best_archi[name] = []

            v, idx = torch.max(self.p[i+1], dim=0)
            c = self.length[name]
            # 3.1 select the best architecture for cell
            if idx != c:  # reuse blocks
                best_archi[name].append(idx)
            elif idx == c:  # create new block
                best_archi[name].append(c)
                model_to_train[name].append(c)
            # 3.2 delete for cell
            if idx != c:
                del self.cells[i][c]

            # 3.3 update the length
            self.length[name] = len(self.cells[i])

        # 4 the classifier and pool layer
        model_to_train['fc'].append(t)
        best_archi['fc'].append(t)

        # 5 update the model to train
        self.model_to_train = model_to_train

        return best_archi
