"""
File        :
Description :resnet 18 individual
Author      :XXX
Date        :2019/9/1
Version     :v1.0
"""
from numpy.ma.core import set_fill_value
import torch
import torch.nn as nn
from .resnet import resnet_18_feat


class IndividualModel(nn.Module):
    def __init__(self, input_size, task_class):
        super(IndividualModel, self).__init__()
        self.input_size = input_size
        self.task_class = task_class

        self.feat_extractor = nn.ModuleList()
        self.fc = nn.ModuleList()
        for t, c in task_class:
            self.feat_extractor.append(resnet_18_feat(input_size=input_size))
            self.fc.append(nn.Linear(512, c))

    def forward(self, x):
        x = self.feat_extractor[self.current_task](x)

        return self.fc[self.current_task](x)
    
    def multihead_forward(self, x):
        x = self.feat_extractor[self.current_task](x)
        outputs = [self.fc[i](x) for i in range(self.current_task+1)]

        return outputs
    
    def extract_feat(self, x):
        x = self.feat_extractor[self.current_task](x)

        return x

    def prepare(self, t):
        self.current_task = t

    def freeze(self, t):
        self.feat_extractor[t].requires_grad_(False)
        self.feat_extractor[t].eval()
        self.fc[t].requires_grad_(False)
        self.fc[t].eval()
        


def Net(input_size, task_class):
    return IndividualModel(input_size, task_class)

