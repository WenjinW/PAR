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
import torchvision.models as models

import utils

from copy import deepcopy
from torchsummary import summary

from automl.darts_genotypes import PRIMITIVES, Genotype
from automl.darts_operation import *


class Alexnet_FE(nn.Module):
    """
	Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
	and get the most related model whilst training a new task in a sequence
	"""
    def __init__(self, alexnet_model):
        super(Alexnet_FE, self).__init__()
        self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])
        self.fe_model.eval()
        self.fe_model.requires_grad_(False)
    
    def forward(self, x):
        return self.fe_model(x)


class Autoencoder(nn.Module):
    """
    The class defines the autoencoder model which takes in the features from the last convolutional layer of the 
    Alexnet model. The default value for the input_dims is 256*13*13.
    """
    def __init__(self, input_dims = 256*13*13, code_dims = 100):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dims, code_dims),nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(code_dims, input_dims),nn.Sigmoid())
        
    def forward(self, x):
        # print(x.size())
        
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x
    
    def encode_embedding(self, x):

        return self.encoder(x)


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
                C_curr *= 2
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

        out = self.global_pooling(s1)

        return out

class Network(nn.Module):
    def __init__(self, input_size, task_class_num, cell_nums, init_channel, genotype=None, device='cuda', input_dim_encoder=256):
        super(Network, self).__init__()
        self.input_size = input_size
        self.task_class_num = task_class_num
        self.cell_nums = cell_nums
        self.init_channel = init_channel
        self.device = device
        self.encode_input_dim = input_dim_encoder # 43264 for 224, 9216 for 112, 256 for 32

        self.feat_extractor = Alexnet_FE(models.alexnet(pretrained=True))
        
        self.num_experts = 1 # only have one expert at the begin
        # expert id for each task. 
        # Note: each expert contains: expert network, expert autoencdoer, expert genotype
        self.task_expert_id = [0] * len(task_class_num)
        self.expert_cell_id = [0]
        self.cells = []

        self.expert_to_tasks = [[0]]
        self.task_g = [genotype] 
        self.experts_used_num = [1]
        # self.task_g[0] = genotype
        self.experts = nn.ModuleList(
            [Expert(self.input_size, self.cell_nums, self.init_channel,
            genotype)])
        self.last_c = self.experts[0].last_c
        self.autoencoder = nn.ModuleList([Autoencoder(self.encode_input_dim)])
        # self.task_encoder_id = [0] * len(task_class_num)
        
        self.pred_layer = nn.ModuleList([])
        for _, c in self.task_class_num:
            self.pred_layer.append(nn.Linear(self.last_c, c))
        
        self.current_task = 0
    
    def forward(self, x):
        expert_id = self.task_expert_id[self.current_task]
        x = self.experts[expert_id](x)
        output = self.pred_layer[self.current_task](x.view(x.size(0), -1))
        return output
    
    def reconstruct_error(self, x):
        x = self.feat_extractor(x)
        # preprocessing step
        # inputs_x = torch.sigmoid((x - torch.mean(x, dim=0)) / torch.std(x, dim=0))
        inputs_x = torch.sigmoid(x)
        inputs_x = inputs_x.view(inputs_x.size(0), -1)

        # encoder decoder
        outputs_x = self.autoencoder[self.current_encoder](inputs_x)
        error = F.mse_loss(outputs_x, inputs_x, reduction='mean')

        return error
    
    def encode_embedding(self, x):
        inputs_x = torch.sigmoid(self.feat_extractor(x))
        inputs_x = inputs_x.view(inputs_x.size(0), -1)
        return self.autoencoder[self.current_encoder].encode_embedding(inputs_x)

    def multihead_forward(self, x):
        expert_id = self.task_expert_id[self.current_task]
        x = self.experts[expert_id](x)
        outputs = []
        for  i in self.expert_to_tasks[expert_id]:
            outputs.append(self.pred_layer[i](x.view(x.size(0), -1)))
        
        return outputs

    def add_encoder(self):
        self.autoencoder.append(Autoencoder(self.encode_input_dim).to(self.device))

    def remove_new_encoder(self):
        del self.autoencoder[-1]

    def add_expert(self, t, input_size, cell_nums, init_channel, genotype):
        # add new expert
        self.experts.append(
            Expert(input_size, cell_nums, init_channel, genotype).to(self.device)
        )

        # add genotype of new expert
        self.task_g.append(genotype)

        # save the expert model id of task t
        self.task_expert_id[t] = self.num_experts
        self.expert_to_tasks.append([t])

        # update the number of experts
        self.num_experts += 1
        self.experts_used_num.append(1)
        
        print(self.task_expert_id)
        print(len(self.experts))
    
    def prepare_train_encoder(self, t):
        self.current_encoder = t
        self.freeze()
        self.autoencoder[self.current_encoder].requires_grad_(True)
    
    def prepare_train(self, t):
        self.current_task = t
        self.freeze()
        expert_id = self.task_expert_id[self.current_task]
        self.experts[expert_id].requires_grad_(True)
        self.pred_layer[self.current_task].requires_grad_(True)

    def train(self, t):
        self.eval(t)
        expert_id = self.task_expert_id[self.current_task]
        self.experts[expert_id].train()
        self.pred_layer[t].train()
    
    def encoder_train(self, t):
        self.encoder_eval(t)
        self.autoencoder[t].train()

    def eval(self, t):
        self.current_task = t
        self.experts.eval()
        self.pred_layer.eval()
        self.autoencoder.eval()
        self.feat_extractor.eval()
    
    def encoder_eval(self, t):
        self.current_encoder = t
        self.experts.eval()
        self.pred_layer.eval()
        self.autoencoder.eval()
        self.feat_extractor.eval()

    def freeze(self):
        self.experts.requires_grad_(False)
        self.experts.eval()
        self.pred_layer.requires_grad_(False)
        self.pred_layer.eval()
        self.autoencoder.requires_grad_(False)
        self.autoencoder.eval()
        self.feat_extractor.requires_grad_(False)
        self.feat_extractor.eval()