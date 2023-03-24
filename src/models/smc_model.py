"""
File        :
Description :
Author      :XXX
Date        :2019/9/18
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
        self.ncha = input_size[0]
        self.C = init_channel
        self.task_classes = task_classes
        self.genotype = [genotype]  # the genotypes which has been used by every block on each layer
        # self.unique_genotype = [genotype]  # the different genotypes which have been used by blocks on each layer
        self.real_genotype = []
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
            self.real_genotype.append([genotype])

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.ModuleList([])
        for t, c in self.task_classes:
            self.classifier.append(nn.Linear(C_prev, c))

        # parameter for architecture search
        self.p = None
        # The new models, model to train
        self.new_models = None
        self.model_to_train = None

    def forward(self, x, t, task_arch=None):
        arch_stem = None
        if task_arch is not None:
            arch_stem = task_arch['stem'][0]

        s0 = s1 = self.stem[arch_stem](x)

        for i, cell in enumerate(self.cells):
            arch_cell = None
            if task_arch is not None:
                arch_cell = task_arch['cell'+str(i)][0]
            s0, s1 = s1, cell[arch_cell](s0, s1)

        out = self.global_pooling(s1)

        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def expand(self, t, device='cuda'):
        # expand the network to a super model
        # 0 clean the probability
        self.p = []
        # 1 expand stem
        # 1.1 reuse: reuse parameters and architecture
        # 1.2 update: reuse architecture but update parameter
        C_curr = self.stem_multiplier * self.C
        self.stem.append(nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        ).to(device))
        # 1.3 generate action probability
        num_l = self.length['stem'] + 1
        self.p.append(torch.full((num_l,), 1/num_l))

        # 2 expand cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            # 2.1 reuse: reuse parameters and architecture
            # 2.2 update: reuse architecture but update parameter
            multiplier = None
            for genotype in self.genotype:
                cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
                multiplier = cell.multiplier
                self.cells[i].append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            # 2.3 generate action parameter
            # for each cell: reuse + update + mutate select
            num_l = self.length['cell'+str(i)] + len(self.genotype)
            self.p.append(torch.full((num_l,), 1 / num_l))
            # self.a['cell'+str(i)] = nn.Parameter(torch.rand(num_l).to(device))

        # 3 get the new modules
        self.get_new_model(t=t)

    def expand_column(self, t, genotype, device='cuda'):
        # expand the network to a super model
        new_models = {'stem': [], 'fc': []}
        # 1 expand stem
        C_curr = self.stem_multiplier * self.C
        self.stem.append(nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        ).to(device))
        new_models['stem'].append(self.length['stem'])
        self.length['stem'] += 1

        # 2 expand cells
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
            multiplier = cell.multiplier
            self.cells[i].append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            new_models['cell' + str(i)] = [self.length['cell' + str(i)]]
            self.length['cell' + str(i)] += 1

        new_models['fc'].append(t)
        self.model_to_train = new_models
        self.genotype.append(genotype)

        return new_models

    def get_new_model(self, t):
        # get new model (update and search)
        new_models = {'stem': [], 'fc': []}
        # 1 stem
        c = self.length['stem']
        new_models['stem'].append(c)
        # 2 cells
        for i in range(self.cell_nums):
            c_1 = self.length['cell' + str(i)]
            c_2 = len(self.genotype)
            num_l = c_1 + c_2
            new_models['cell' + str(i)] = []
            for k in range(c_1, num_l):
                new_models['cell' + str(i)].append(k)

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

    def modify_archi_param(self, requires_grad=True):
        params = self.get_archi_param()
        if requires_grad:
            utils.unfreeze_parameter(params)
        else:
            utils.freeze_parameter(params)

    def search_forward(self, x, selected_ops):
        # 1 stem
        # 1.1 stem
        s0 = s1 = self.stem[selected_ops[0]](x)

        # 2 cells layer
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell[selected_ops[1+i]](s0, s1)

        out = self.global_pooling(s1)
        logits = []
        for t, c in self.task_classes:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def select(self, t):
        # 1 define the container of new models to train and the best submodel
        model_to_train = {'stem': [], 'fc': []}
        best_archi = {'stem': [], 'fc': []}
        best_op = {}  # reuse: 0, update: 1, mutate 2
        # 2 stem
        # 2.1 select the best architecture for stem
        v, idx = torch.max(self.p[0], dim=0)
        c = self.length['stem']
        if idx < c:  # reuse
            best_archi['stem'].append(idx)
            best_op['stem'] = 0
        elif idx == c:  # new
            best_archi['stem'].append(c)
            model_to_train['stem'].append(c)
            best_op['stem'] = 1
        # 2.2 delete for stem
        if idx != c:
            del self.stem[c]
        # 2.3 update the length
        self.length['stem'] = len(self.stem)

        # 3 cells layer
        for i in range(self.cell_nums):
            name = 'cell' + str(i)
            g_name = 'genotype' + str(i)
            model_to_train[name] = []
            best_archi[name] = []

            v, idx = torch.max(self.p[i+1], dim=0)
            c_1 = self.length[name]
            c_2 = len(self.genotype)
            num_l = c_1 + c_2
            # 3.1 select the best architecture for cell
            if idx < c_1:  # reuse the genotype of the idx cell in layer i
                best_archi[name].append(idx)
                best_op[name] = 0
                best_op[g_name] = deepcopy(self.real_genotype[i][idx])
                self.real_genotype[i].append(best_op[g_name])
            elif idx < c_1 + c_2:  # update the idx-c_1 genotype in the used genotype types set
                best_archi[name].append(c_1)
                model_to_train[name].append(c_1)
                best_op[name] = 1
                best_op[g_name] = deepcopy(self.genotype[idx-c_1])
                self.real_genotype[i].append(best_op[g_name])

            # 3.2 delete for cell
            cell_s = deepcopy(self.cells[i][idx])
            for k in range(c_1, num_l):
                del self.cells[i][-1]
            if idx >= c_1:
                self.cells[i].append(cell_s)

            # 3.3 update the length
            self.length[name] = len(self.cells[i])

        # 4 the classifier and pool layer
        model_to_train['fc'].append(t)
        best_archi['fc'].append(t)

        # 5 update the model to train
        self.model_to_train = model_to_train

        return best_archi, best_op
