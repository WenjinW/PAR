"""
File        :
Description :
Author      :XXX
Date        :2021/03/08
Version     :v1.1
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import utils

from copy import deepcopy
from models.resnet import resnet_18_feat
from models.TALL_model_v2 import Alexnet_FE, Autoencoder


class Net(nn.Module):
    def __init__(self, input_size, task_class_num, input_dim_encoder=256, device='cuda'):
        super(Net, self).__init__()
        self.input_size = input_size
        self.task_class_num = task_class_num
        self.device =device
        self.encode_input_dim = input_dim_encoder

        self.feat_extractor = Alexnet_FE(models.alexnet(pretrained=True))
        
        self.num_experts = 1
        self.experts = nn.ModuleList(
            [resnet_18_feat(self.input_size)]
            )
        self.pred_layer = nn.ModuleList(
            [nn.Linear(512, self.task_class_num[0][1])]
        )
        self.autoencoder = nn.ModuleList([Autoencoder(self.encode_input_dim)])
        
        self.current_task = 0
    
    def forward(self, x):
        x = self.experts[self.current_task](x)
        output = self.pred_layer[self.current_task](x)
        return output
    
    def reconstruct_error(self, x):
        inputs_x = torch.sigmoid(self.feat_extractor(x))
        inputs_x = inputs_x.view(inputs_x.size(0), -1)
        outputs_x = self.autoencoder[self.current_encoder](inputs_x)
        error = F.mse_loss(outputs_x, inputs_x, reduction='mean')

        return error

    def add_encoder(self):
        # add encoder
        self.autoencoder.append(Autoencoder(self.encode_input_dim).to(self.device))

    def add_expert(self, t, related_expert):
        # add expert
        self.experts.append(resnet_18_feat(self.input_size).to(self.device))
        # fintune
        initial_model = utils.get_model(self.experts[related_expert])
        utils.set_model_(self.experts[t], initial_model)
        # add head
        self.pred_layer.append(nn.Linear(512, self.task_class_num[t][1]).to(self.device))
    
    def prepare_train_encoder(self, t):
        self.current_encoder = t
        self.freeze()
        self.autoencoder[self.current_encoder].requires_grad_(True)
    
    def prepare_train(self, t):
        self.current_task = t
        self.freeze()
        self.experts[self.current_task].requires_grad_(True)
        self.pred_layer[self.current_task].requires_grad_(True)

    def train(self, t):
        self.eval(t)
        self.experts[t].train()
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

