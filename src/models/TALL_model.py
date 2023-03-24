"""
File        :
Description :
Author      :XXX
Date        :2020/10/20
Version     :v1.1
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

from copy import deepcopy

from automl.darts_genotypes import PRIMITIVES, Genotype
from automl.darts_operation import *
from automl.mdenas_basicmodel import Cell
from automl.mdenas_search import AutoSearch


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

    def __init__(self, input_size, task_class_num, cell_nums, init_channel, genotype, device):
        super(Network, self).__init__()
        self.device = device
        self.cell_nums = cell_nums
        # params about dataset
        self.input_size = input_size
        self.ncha = input_size[0]
        self.C = init_channel
        self.task_class_num = task_class_num

        # params for searching the cell architecture for the first task
        self.search_layers = 5

        # params for new expansion cell
        self._steps = 4  # the number of intermediate nodes
        self._multiplier = 4

        self.stem_multiplier = 3
        self.length = {'stem': 1}
        self.arch_init = {'stem': [0], 'fc': [0]}

        C_curr = self.stem_multiplier * self.C
        self.stem = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )])
        # reuse number of each cell
        self.used_num = {'stem': [1]}

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        self.cells = nn.ModuleList([])
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = NewCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(nn.ModuleList([cell]))
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            self.arch_init['cell' + str(i)] = [0]
            self.length['cell'+str(i)] = 1
            self.used_num['cell'+str(i)] = [1]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.ModuleList([])
        for t, c in self.task_class_num:
            self.classifier.append(nn.Linear(C_prev, c))

        # parameter for architecture search
        self.p = None
        self.p_cell = {}
        # The new models, model to train
        self.new_models = None
        self.model_to_train = None

    def forward(self, x):
        if self.stage == 'normal':
            return self.normal_forward(x)
        elif self.stage == 'search':
            return self.search_forward(x)
        else:
            print("The stage must in ['normal', 'search']!!!")

    def search_space(self, task_id, schedule):
        # add an undetermined unit in each layer
        self.p_cell = []
        # 1 expand stem
        # 1.1 reuse: reuse blocks
        # 1.2 new: create a new block
        C_curr = self.stem_multiplier * self.C
        self.stem.append(nn.Sequential(
            nn.Conv2d(self.ncha, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        ).to(self.device))
        # 1.3 generate action parameter
        num_l = self.length['stem'] + 1
        self.p_cell.append(torch.full((num_l,), 1 / num_l))

        # 2 expand cells
        steps, multiplier = 4, 4
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
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(self.device)
            # cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(device)
            # multiplier = cell.multiplier
            self.cells[i].append(cell)
            reduction_prev = reduction
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            # 2.3 generate action parameter
            # for each cell: reuse + create
            num_l = self.length['cell'+str(i)] + 1
            self.p_cell.append(torch.full((num_l,), 1 / num_l))
        
        # schedule for cell
        if schedule != 'no':
            self.schedule(schedule)

        # params of operations
        self.num_ops = len(PRIMITIVES)
        self.num_edges = sum(1 for i in range(steps) for n in range(2 + i))
        self.p_ops = {
            "normal": torch.full((self.num_edges, self.num_ops), 1/self.num_ops),
            "reduce": torch.full((self.num_edges, self.num_ops), 1/self.num_ops)
        }

        self.get_new_cells(task_id)
        
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

    def get_genotype(self):
        
        def _parse(p_ops):
            gene = []
            n = 2
            start = 0
            p, idx = torch.max(p_ops[:, 1:], dim=1)
            # for each intermediate node
            for i in range(self._steps):
                end = start + n
                p_i = deepcopy(p[start:end])
                idx_i = deepcopy(idx[start:end])

                top_k, top_idx = torch.topk(p_i, 2)
                for j in top_idx:
                    gene.append((PRIMITIVES[idx_i[j]+1], j.item()))
                start = end
                n += 1
            return gene

        gene_normal = _parse(self.p_ops['normal'])
        gene_reduce = _parse(self.p_ops['reduce'])

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)

        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def get_new_cells(self, task_id):
        # get new model (update and search)
        new_cells = {'stem': [], 'fc': []}
        # 1 stem
        c = self.length['stem']
        new_cells['stem'].append(c)
        # 2 cells
        for i in range(self.cell_nums):
            new_cells['cell' + str(i)] = [self.length['cell' + str(i)]]

        # 3 classifier
        new_cells['fc'].append(task_id)
        self.new_cells = new_cells
    
    def get_old_cells(self, task_id):
        # get all the old modules up to task task_id
        # 1 stems and fcs
        old_cell_layers = {'stem': [k for k in range(self.length['stem'])],
                            'fc': [k for k in range(task_id+1)]}
        # 2 cells
        for i in range(self.cell_nums):
            old_cell_layers['cell'+str(i)] = [k for k in range(self.length['cell'+str(i)])]
        
        return old_cell_layers

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

    def normal_forward(self, x):
        arch_stem = self.current_archi['stem'][0]
        s0 = s1 = self.stem[arch_stem](x)
        for i, cell in enumerate(self.cells):
            arch_cell = self.current_archi['cell'+str(i)][0]
            s0, s1 = s1, cell[arch_cell](s0, s1)

        out = self.global_pooling(s1)

        logits = []
        for t, c in self.task_class_num:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def schedule(self, schedule):
        # schedule for stem
        for idx in range(self.length['stem']):
            self.p_cell[0][idx] *= self.used_num['stem'][idx]
        self.p_cell[0][:-1] = self.p_cell[0][:-1] / torch.sum(self.p_cell[0][:-1]) * (1-self.p_cell[0][-1])
        if schedule == 'softmax':
            self.p_cell[0] = F.softmax(self.p_cell[0], dim=0)
            
        # schedule for cells
        for i in range(self.cell_nums):
            for idx in range(self.length['cell'+str(i)]):
                self.p_cell[1+i][idx] *= self.used_num['cell'+str(i)][idx]
            self.p_cell[1+i][:-1] = self.p_cell[1+i][:-1] / torch.sum(self.p_cell[1+i][:-1]) * (1-self.p_cell[1+i][-1])
            if schedule == 'softmax':
                self.p_cell[1+i] = F.softmax(self.p_cell[1+i], dim=0)
    
    def search_forward(self, x):
        # 1 stem
        # 1.1 stem
        s0 = s1 = self.stem[self.selected_cell[0]](x)

        # 2 cells layer
        for i, cell in enumerate(self.cells):
            # for each cell layer
            if self.selected_cell[1+i] < self.length['cell'+str(i)]: # if reuse a block
                s0, s1 = s1, cell[self.selected_cell[1 + i]](s0, s1)
            else: # if use new block
                if cell[self.selected_cell[1 + i]].reduction:
                    cell_type = 'reduce'
                else:
                    cell_type = 'normal'
                s0, s1 = s1, cell[self.selected_cell[1 + i]](s0, s1, self.selected_ops[cell_type])

        out = self.global_pooling(s1)
        logits = []
        for t, c in self.task_class_num:
            logits.append(self.classifier[t](out.view(out.size(0), -1)))

        return logits

    def select(self, t):
        genotype = self.get_genotype()

        # 1 define the container of new models to train and the best submodel
        model_to_train = {'stem': [], 'fc': []}
        best_archi = {'stem': [], 'fc': []}
        # 2 stem
        # 2.1 select the best architecture for stem
        v, idx = torch.max(self.p_cell[0], dim=0)
        idx = idx.item()
        c = self.length['stem']
        best_archi['stem'].append(idx)
        if idx < c:  # reuse
            self.used_num['stem'][idx] += 1
            del self.stem[c] # delete unused new unit
        elif idx == c:  # use new unit
            model_to_train['stem'].append(c)
            self.used_num['stem'].append(1)
   
        # 2.2 update the length
        self.length['stem'] = len(self.stem)

        # 3 cells layer
        C_curr = self.stem_multiplier * self.C
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, self.C
        reduction_prev = False
        for i in range(self.cell_nums):
            if i in [self.cell_nums // 3, 2 * self.cell_nums // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            name = 'cell' + str(i)
            model_to_train[name] = []
            best_archi[name] = []

            v, idx = torch.max(self.p_cell[i+1], dim=0)
            idx = idx.item()
            c = self.length[name]
            best_archi[name].append(idx)

            # 3.1 select the best architecture for cell
            if idx != c:  # reuse cell
                del self.cells[i][c] # delete new cell
                C_prev_prev, C_prev = C_prev, self.cells[i][c-1].multiplier * C_curr
                self.used_num[name][idx] += 1 
            elif idx == c:  # use new cell
                model_to_train[name].append(c)
                self.cells[i][c] = NewCell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev).to(self.device)
                C_prev_prev, C_prev = C_prev, self.cells[i][c].multiplier * C_curr
                self.used_num[name].append(1)
            reduction_prev = reduction

            # 3.2 update the length
            self.length[name] = len(self.cells[i])

        # 4 the classifier and pool layer
        model_to_train['fc'].append(t)
        best_archi['fc'].append(t)

        # 5 update the model to train
        self.model_to_train = model_to_train

        return best_archi, genotype

    def set_selected_model(self, selected_cell, selected_ops):
        self.selected_cell = selected_cell
        self.selected_ops = selected_ops

    def set_stage(self, stage='normal'):
        self.stage = stage

    def set_task_model(self, t, archi):
        self.current_task = t
        self.current_archi = archi