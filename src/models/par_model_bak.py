"""
File        :
Description :
Author      :XXX
Date        :2020/10/20
Version     :v1.1
"""
from time import sleep
import numpy as np
from numpy.lib.function_base import delete
from numpy.ma.core import set_fill_value
import torch
from torch._C import FunctionSchema
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy

from tqdm.std import TRLock
from torchsummary import summary

import utils
from .resnet import resnet_18_feat



class PARModel(nn.Module):
    def __init__(self, input_size, task_class_num, device='cuda'):
        super(PARModel, self).__init__()
        self.input_size = input_size
        self.task_class_num = task_class_num

        self.device = device

        # the id of current classifer and feature extractor
        self.current_classifier = 0
        self.current_extractor = 0
        
        # feature extractors
        self.feat_extractor = nn.ModuleList([
            resnet_18_feat(input_size=input_size)
            ]).to(self.device)    
        # classifers
        self.classifier = nn.ModuleList([
            nn.Linear(512, self.task_class_num[self.current_classifier][1])
            ]).to(self.device)

        # index of classifiers
        self.classifier_index = [0]
        # index of extractors
        self.extractor_index = [0]

        self.default_components = ['classifier', 'extractor']
        self.trained_components = ['classifier', 'extractor']
    
    @property   
    def num_classifiers(self):
        
        return len(self.classifier)

    @property
    def num_extractors(self):

        return len(self.feat_extractor)
    
    def forward(self, x):
        x = self.feat_extractor[self.current_extractor](x)
        x = self.classifier[self.current_classifier](x)
        
        return x

    def multihead_forward(self, x):
        pass        
    
    def add_classifier(self, task_id):
        # add a new classifer for a new task
        if task_id < len(self.classifier):
            raise Exception("The classifer of task {} already exists!".format(task_id))
        
        self.classifier.append(
            nn.Linear(512, self.task_class_num[task_id][1]).to(self.device)
            )
    
    def add_extractor(self):
        # add a new feature extractor
        self.feat_extractor.append(
            resnet_18_feat(input_size=self.input_size).to(self.device)
            )

    def remove_classifier(self, classifier_id):
        if classifier_id >= len(self.classifier):
            raise Exception(
                "The classifer {} dost not exist!".format(
                    classifier_id))
        
        del self.classifier[classifier_id]
    
    def remove_extractor(self, feat_extractor_id):
        if feat_extractor_id >= len(self.feat_extractor):
            raise Exception(
                "The extractor {} dost not exist!".format(
                    feat_extractor_id))

        del self.feat_extractor[feat_extractor_id]

    def set_classifier(self, classifier_id):
        if classifier_id >= len(self.classifier):
            raise Exception(
                "The classifer {} dost not exist!".format(
                    classifier_id))
        
        self.current_classifier = classifier_id
    
    def set_extractor(self, feat_extractor_id):
        if feat_extractor_id >= len(self.feat_extractor):
            raise Exception(
                "The extractor {} dost not exist!".format(
                    feat_extractor_id))

        self.current_extractor = feat_extractor_id
    
    def set_trained_components(self, components):
        if len(components) > 2:
            raise Exception("Components number > 2!")
        for i in components:
            if i not in self.default_components:
                raise Exception("Component should be 'classifier' or 'extractor'!")
        self.trained_components = components

    def requires_grad_classifier_(self, classifier_id, requires_grad=True):
        self.classifier[classifier_id].requires_grad_(requires_grad)
    
    def requires_grad_extractor_(self, extractor_id, requires_grad=True):
        self.feat_extractor[extractor_id].requires_grad_(requires_grad)

    def train_components(self, mode=True):
        if "classifier" in self.trained_components:
            self.train_classifier(mode)
        if "extractor" in self.trained_components:
            self.train_extractor(mode)
            
    def train_classifier(self, mode=True):
        self.classifier[self.current_classifier].train(mode)
    
    def train_extractor(self, mode=True):
        self.feat_extractor[self.current_extractor].train(mode)
