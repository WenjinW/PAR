"""
File        :
Description :
Author      :XXX
Date        :2019/9/19
Version     :v1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from automl.darts_genotypes import PAR_PRIMITIVES
from automl.darts_genotypes import Genotype
from automl.darts_operation import *


class MixedOp(nn.Module):
    # mixed operation in every edge
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PAR_PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, selected_op):
        # return sum(w * op(x) for w, op in zip(weights, self._ops))
        return self._ops[selected_op](x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, selected):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            # each intermediate node has multiple inputs
            s = sum(self._ops[offset + j](h, selected[offset + j]) for j, h in enumerate(states))
            offset += len(states)

            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class BasicNetwork(nn.Module):

    def __init__(self, input_c, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3, device='cuda:0'):
        super(BasicNetwork, self).__init__()
        self.input_c = input_c
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps  # the number of intermediate nodes
        self._multiplier = multiplier
        self.device = device
        self.num_ops = len(PAR_PRIMITIVES)
        self.num_edges = sum(1 for i in range(self._steps) for n in range(2 + i))

        stem_multiplier = 1
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_c, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                # C_curr *= 2
                reduction = True
            else:
                reduction = False

            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        # initialize the probabilities
        self.p = None
        self._initialize_p()

    def new(self):
        model_new = BasicNetwork(self.input_c, self._C, self._num_classes, self._layers, self._criterion).to(self.device)
        model_new.p = deepcopy(self.probability())

        return model_new

    def set_ops(self, selected_ops):
        self.selected_ops = selected_ops

    def forward(self, x):
        s0 = s1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                cell_type = 'reduce'
            else:
                cell_type = 'normal'
            s0, s1 = s1, cell(s0, s1, self.selected_ops[cell_type])
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def loss(self, x, y):
        logits = self(x)
        return self._criterion(logits, y)

    def _initialize_p(self):
        k = self.num_edges
        num_ops = self.num_ops
        self.p = {
            "normal": torch.full((k, num_ops), 1/num_ops),
            "reduce": torch.full((k, num_ops), 1/num_ops)
        }

    def probability(self):
        return self.p

    def genotype(self):

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
                    gene.append((PAR_PRIMITIVES[idx_i[j]+1], j.item()))
                start = end
                n += 1
            return gene

        gene_normal = _parse(self.p['normal'])
        gene_reduce = _parse(self.p['reduce'])

        concat = list(range(2 + self._steps - self._multiplier, self._steps + 2))

        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

