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


def conv3x3(in_channel, out_channel, stride=1, padding=1):
    # 3x3 convolution with padding

    return nn.Conv2d(in_channel, out_channel, kernel_size=3,
                     stride=stride, padding=padding, bias=False)


def conv1x1(in_channel, out_channel, stride=1):
    # 1x1 convolution with padding

    return nn.Conv2d(in_channel, out_channel, kernel_size=1,
                     stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channel, channel, stride=1, downsample=None,
                 norm_layer=None, device="cuda:0"):
        """The basic block of ResNet. It has 2 3x3 convolution layers

        :param in_channel: number of input channels
        :param channel: number of channels in this block
        :param stride: stride of the first convolution layer in this
                       block
        :param downsample: downsample for the input if necessary
        :param norm_layer: the type of normalization layer
        """
        super(BasicBlock, self).__init__()
        # define the norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        self.norm_layer = norm_layer
        self.in_channel = in_channel
        self.channel = channel
        self.device = device

        self.conv1 = nn.ModuleList([conv3x3(in_channel, channel, stride)])
        self.bn1 = nn.ModuleList([self.norm_layer(channel)])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ModuleList([conv3x3(channel, channel)])
        self.bn2 = nn.ModuleList([self.norm_layer(channel)])
        self.downsample = downsample
        self.stride = stride

        # state
        self.is_search = False
        self.task_id = None
        self.expert_id = None

    def forward(self, x):
        if self.is_search:
            output = self.forward_search(x)
        else:
            if self.expert_id is None or self.task_id is None:
                raise Exception("Task id: {}, Expert id: {} !!!".format(self.task_id, self.expert_id))
            if self.expert_id == 0:
                output = self.forward_first_expert(x)
            else:
                output = self.forward_later_expert(x)

        return output
    
    def forward_first_expert(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        output = self.conv1[0](x)
        output = self.bn1[0](output)
        output = self.relu(output)

        output = self.conv2[0](output)
        output = self.bn2[0](output)

        output += identity
        output = self.relu(output)

        return output
    
    def forward_later_expert(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        output = self.conv1[0](x)
        output = self.conv1[self.expert_id](output)
        output = self.bn1[self.task_id](output)

        output = self.relu(output)

        output = self.conv2[0](output)
        output = self.conv2[self.expert_id](output)
        output = self.bn2[self.task_id](output)

        output += identity
        output = self.relu(output)

        return output
    
    def forward_search(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        
        output = self.conv1[0](x)
        output = self.conv1_temp_ops[self.sampled_ops[0]](output)
        output = self.bn1_temp(output)

        output = self.relu(output)

        output = self.conv2[0](output)
        output = self.conv2_temp_ops[self.sampled_ops[1]](output)
        output = self.bn2_temp(output)

        output += identity
        output = self.relu(output)

        return output
    
    def expand_ops(self, new_ops):
        """Add new operations for each new expert
        
        """
        if not isinstance(new_ops, list):
            raise Exception("Tye type of 'new ops' should be list")
        if len(new_ops) != 2:
            raise Exception("Tye length of 'new ops' should be equal to 2, but obtain: {}".format(
                len(new_ops)
            ))

        new_op1 = new_ops[0]
        new_op2 = new_ops[1]
        if new_op1 not in LightOperations.keys():
            raise Exception("Unknown opeartion name: {} for new_op1".format(new_op1))
        self.conv1.append(LightOperations[new_op1](self.channel).to(self.device))
        if new_op2 not in LightOperations.keys():
            raise Exception("Unknown opeartion name: {} for new_op2".format(new_op2))
        self.conv2.append(LightOperations[new_op2](self.channel).to(self.device))
    
    def expand_norm(self):
        """add norm operations for each new task

        """
        self.bn1.append(self.norm_layer(self.channel).to(self.device))
        self.bn2.append(self.norm_layer(self.channel).to(self.device))

    def set_task_expert(self, task_id=None, expert_id=None):
        self.task_id = task_id
        self.expert_id = expert_id

    def set_search(self, is_search=True):
        self.is_search = is_search
        if self.is_search:
            self.conv1_temp_ops = LightOperationsSpace(self.channel).to(self.device)
            self.bn1_temp = self.norm_layer(self.channel).to(self.device)
            self.conv2_temp_ops = LightOperationsSpace(self.channel).to(self.device)
            self.bn2_temp = self.norm_layer(self.channel).to(self.device)
        else:
            del self.conv1_temp_ops
            del self.bn1_temp
            del self.conv2_temp_ops
            del self.bn2_temp
    
    def set_sampled_ops(self, ops):

        self.sampled_ops = ops
        return
    
    def requires_grad_expert(self, expert_id):
        if expert_id == -1: # for search
            self.conv1_temp_ops.requires_grad_(True)
            self.conv2_temp_ops.requires_grad_(True)
        else:
            self.conv1[expert_id].requires_grad_(True)
            self.conv2[expert_id].requires_grad_(True)
            if expert_id == 0 and self.downsample is not None:
                self.downsample.requires_grad_(True)
    
    def train_task(self, task_id):
        if task_id == -1: # for search
            self.bn1_temp.train()
            self.bn2_temp.train()
        else:
            self.bn1[task_id].train()
            self.bn2[task_id].train()
            if task_id == 0 and self.downsample is not None:
                self.downsample.train()


class PARLayers(nn.Module):
    def __init__(self, block, in_channel, channel, num_block,
                stride=1, norm_layer=None, device="cuda:0") -> None:
        super(PARLayers, self).__init__()
        
        self.in_channel = in_channel
        self.channel = channel
        self.norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        self.device = device

        # down_sample belongs to the frist task (freezed when learning later tasks)
        if stride != 1 or self.in_channel != self.channel:
            self.down_sample = nn.Sequential(
                conv1x1(self.in_channel, channel, stride),
                self.norm_layer(channel)
            )
        else:
            self.down_sample = None
        
        self.layers = nn.ModuleList([])
        self.layers.append(block(self.in_channel, self.channel, stride,
                            self.down_sample, norm_layer=self.norm_layer, device=self.device))
        
        for _ in range(1, num_block):
            self.layers.append(block(self.channel, self.channel,
                            norm_layer=self.norm_layer, device=self.device))
    
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)

        return x
    
    def set_task_expert(self, task_id=None, expert_id=None):
        for i in range(len(self.layers)):
            self.layers[i].set_task_expert(task_id, expert_id)
            
    def set_search(self, is_search=True):
        for i in range(len(self.layers)):
            self.layers[i].set_search(is_search)
    
    def set_sampled_ops(self, ops):
        for i in range(len(self.layers)):
            self.layers[i].set_sampled_ops(ops[i * 2: (i + 1) * 2])

        return
    
    def requires_grad_expert(self, expert_id):
        for i in range(len(self.layers)):
            self.layers[i].requires_grad_expert(expert_id)
        # if expert_id == 0 and self.down_sample is not None:
        #     self.down_sample.requires_grad_(True)

    def train_task(self, task_id):
        for i in range(len(self.layers)):
            self.layers[i].train_task(task_id)
        # if task_id == 0 and self.down_sample is not None:
        #     self.down_sample.train()
    
    def expand_ops(self, ops):
        for i in range(len(self.layers)):
            self.layers[i].expand_ops(ops[i * 2: (i + 1) * 2])
    
    def expand_norm(self):
        for i in range(len(self.layers)):
            self.layers[i].expand_norm()


class PARResNet(nn.Module):
    def __init__(self, task_input_size, task_class_num, block, layers, norm_layer=None, args=None):
        super(PARResNet, self).__init__()

        self.task_input_size = task_input_size
        self.task_class_num = task_class_num

        self.layers = layers
        self.device = args.device
        self.logger = args.logger
        self.args = args

        self._norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer
        
        # the input channels of the block
        self.in_channel = 64
        # size = 32 * 32 * 3
        self.conv1 = nn.ModuleList(
            [nn.Conv2d(3, self.in_channel, kernel_size=3,stride=1, padding=1, bias=False)]
        )
        self.bn1 = nn.ModuleList(
            [self._norm_layer(self.in_channel)]
        )
        self.relu = nn.ReLU(inplace=True)

        # size = 32 * 32 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # modified version
        if self.args.resnet_version == "MNTDP":
            # input: 64x16x16, output: 64x16x16
            self.layer1 = PARLayers(block, 64, 64, self.layers[0], device=self.device) 
            # input: 64x16x16, output: 64x8x8
            self.layer2 = PARLayers(block, 64, 64, self.layers[1], stride=2, device=self.device)
            # input: 64x8x8, output: 64x4x4
            self.layer3 = PARLayers(block, 64, 64, self.layers[2], stride=2, device=self.device)
            # input: 64x4x4, output: 64x2x2
            self.layer4 = PARLayers(block, 64, 64, self.layers[3], stride=2, device=self.device)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.ModuleList()
            for _, c in self.task_class_num:
                self.fc.append(nn.Linear(64, c))
        elif self.args.resnet_version == "original":
            # input: 64x16x16, output: 64x16x16
            self.layer1 = PARLayers(block, 64, 64, self.layers[0], device=self.device) 
            # input: 64x16x16, output: 64x8x8
            self.layer2 = PARLayers(block, 64, 128, self.layers[1], stride=2, device=self.device)
            # input: 64x8x8, output: 64x4x4
            self.layer3 = PARLayers(block, 128, 256, self.layers[2], stride=2, device=self.device)
            # input: 64x4x4, output: 64x2x2
            self.layer4 = PARLayers(block, 256, 512, self.layers[3], stride=2, device=self.device)

            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.ModuleList()
            for _, c in self.task_class_num:
                self.fc.append(nn.Linear(512, c))

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.task2expert = [0]
        self.expert2task = [[0]]

        # prepare pretrained feature extractor
        if args.pretrained_feat_extractor == "":
            self.feat_extractor = None
        else:
            self.feat_extractor = get_pretrained_feat_extractor(args.pretrained_feat_extractor)
        self.logger.info("Using relatedness feature extractor: {}".format(args.pretrained_feat_extractor))

        self.task2mean = []
        self.task2cov = []
        
        self.current_task = None

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
        self.is_search = False # search state, used for operation search


    def print_info(self):
        self.logger.info("task2expert: {}".format(self.task2expert))
        self.logger.info("expert2task: {}".format(self.expert2task))
    
    def forward(self, x):
        if self.is_search:
            output = self.forward_search(x)
        else:
            if self.current_expert == 0:
                x = self.conv1[self.current_expert](x)
                x = self.bn1[self.current_expert](x)
            else:
                x = self.conv1[0](x)
                x = self.conv1[self.current_expert](x)
                x = self.bn1[self.current_expert](x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.is_multihead:
                output = []
                for i in self.expert2task[self.current_expert]:
                    output.append(self.fc[i](x))
            else:
                output = self.fc[self.current_task](x)
        return output
    
    def forward_search(self, x):
        x = self.conv1[0](x)
        x = self.conv1_temp_ops[self.sampled_ops[0]](x)
        x = self.bn1_temp(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        output = self.temp_fc(x)

        return output

    def set_multihead(self, is_multihead=False):
        self.is_multihead = is_multihead
    
    def set_search(self, is_search=False):
        self.is_search = is_search
        if self.is_search:
            # prepare for search
            self.conv1_temp_ops = LightOperationsSpace(self.in_channel).to(self.device)
            self.bn1_temp = self._norm_layer(self.in_channel).to(self.device)
            # create a temporary fc
            self.temp_fc = nn.Linear(64, self.task_class_num[self.current_task][1]).to(self.device)

            num_ops = len(LightOperations.keys())
            num_conv = 1 + sum(self.layers) * 2
            self.p_ops = torch.ones(size=(num_conv, num_ops)) / num_ops
        else:
            # clean for search
            del self.conv1_temp_ops
            del self.bn1_temp
            del self.temp_fc
            del self.p_ops

        # create all candidate operations
        self.layer1.set_search(is_search)
        self.layer2.set_search(is_search)
        self.layer3.set_search(is_search)
        self.layer4.set_search(is_search)
    
    def set_sampled_ops(self, ops):
        # ops shape: (num_conv, 1)
        self.sampled_ops = ops
        num_conv = torch.tensor([1] + [i * 2 for i in self.layers])
        interval = torch.cumsum(num_conv, 0)
        # self.logger.info("interval: {}".format(interval))
        self.layer1.set_sampled_ops(ops[interval[0]: interval[1]])
        self.layer2.set_sampled_ops(ops[interval[1]: interval[2]])
        self.layer3.set_sampled_ops(ops[interval[2]: interval[3]])
        self.layer4.set_sampled_ops(ops[interval[3]: interval[4]])

        return
    
    def extract_feat(self, x):
        if self.feat_extractor is None:
            x = torch.flatten(x, start_dim=1)
        else:
            x = self.feat_extractor(x)

        return x.view(x.size(0), -1)
    
    def expand(self, task_id, ops):
        op_id2name = [name for name in LightOperations.keys()]
        ops = [op_id2name[op] for op in ops]

        self.logger.info("Expand ops: {}".format(ops))

        if ops[0] not in LightOperations.keys():
            raise Exception("Unknown opeartion name: {} for layer 0".format(ops[0]))
        self.conv1.append(LightOperations[ops[0]](self.in_channel).to(self.device))
        self.bn1.append(self._norm_layer(self.in_channel).to(self.device))

        self.layer1.expand_ops(ops[1:5])
        self.layer1.expand_norm()
        self.layer2.expand_ops(ops[5:9])
        self.layer2.expand_norm()
        self.layer3.expand_ops(ops[9:13])
        self.layer3.expand_norm()
        self.layer4.expand_ops(ops[13:17])
        self.layer4.expand_norm()

        self.expert2task.append([task_id])
        self.task2expert.append(len(self.expert2task) - 1)
    
    def reuse(self, task_id, reused_expert_id):
        self.expert2task[reused_expert_id].append(task_id)
        self.task2expert.append(reused_expert_id)

    def add_mean_cov(self, mean):
        self.task2mean.append(mean)
        # self.task2cov.append(cov)

    def set_current_task(self, task_id):
        if task_id >= len(self.task2expert): # for search
            return
        
        self.current_task = task_id
        self.current_expert = self.task2expert[task_id]
        # self.layer1.set_task_expert(self.current_task, self.current_expert)
        # self.layer2.set_task_expert(self.current_task, self.current_expert)
        # self.layer3.set_task_expert(self.current_task, self.current_expert)
        # self.layer4.set_task_expert(self.current_task, self.current_expert)

        # 暂时先让同个 expert 的不同 task 共享 bn
        self.layer1.set_task_expert(self.current_expert, self.current_expert)
        self.layer2.set_task_expert(self.current_expert, self.current_expert)
        self.layer3.set_task_expert(self.current_expert, self.current_expert)
        self.layer4.set_task_expert(self.current_expert, self.current_expert)

    def requires_grad_task(self, task_id):
        if task_id >= len(self.task2expert): # for search
            expert_id = -1
        else:
            expert_id = self.task2expert[task_id]
        
        self.requires_grad_(False)
        if expert_id == -1:
            self.conv1_temp_ops.requires_grad_(True)
        else:
            self.conv1[expert_id].requires_grad_(True)
        self.layer1.requires_grad_expert(expert_id)
        self.layer2.requires_grad_expert(expert_id)
        self.layer3.requires_grad_expert(expert_id)
        self.layer4.requires_grad_expert(expert_id)
        if expert_id == -1:
            self.temp_fc.requires_grad_(True)
        else:
            self.fc[task_id].requires_grad_(True)

    def train_task(self, task_id):
        if task_id >= len(self.task2expert): # for search
            expert_id = -1
        else:
            expert_id = self.task2expert[task_id]
        
        self.eval()
        if expert_id == -1:
            self.bn1_temp.train()
        else:
            self.bn1[expert_id].train()
        self.layer1.train_task(expert_id)
        self.layer2.train_task(expert_id)
        self.layer3.train_task(expert_id)
        self.layer4.train_task(expert_id)


def _par_resnet(input_size, block, layers, task_class, args):
    model = PARResNet(input_size, task_class, block, layers, args=args)
    return model


def par_resnet_18(input_size, task_class, args):
    return _par_resnet(input_size, BasicBlock, [2, 2, 2, 2], task_class, args)


