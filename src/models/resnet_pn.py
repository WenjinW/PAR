"""
File        :
Description :resnet_progressive
Author      :XXX
Date        :2020/11/04
Version     :v1.0
"""
import numpy as np
import torch
import torch.nn as nn
import copy

import utils

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
                 norm_layer=None):
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

        self.conv1 = conv3x3(in_channel, channel, stride)
        self.bn1 = norm_layer(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channel, channel)
        self.bn2 = norm_layer(channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)

        output += identity
        output = self.relu(output)

        return output


class ResNet(nn.Module):

    def __init__(self, input_size, task_class, block, layers, norm_layer=None):
        super(ResNet, self).__init__()

        self.input_size = input_size
        self.task_class = task_class

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        # the input channels of the block
        self.in_channel = 64
        # size = 32 * 32 * 3
        self.conv1 = nn.ModuleList()
        self.bn1 = nn.ModuleList()
        # self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # size = 32 * 32 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # size = 16 * 16 * 64
        self.layer1 = nn.ModuleList()
        self.layer1_scale = nn.ModuleList()
        self.layer1_r = nn.ModuleList()
        self.layer1_u = nn.ModuleList()
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # size = 16 * 16 * 64
        self.layer2 = nn.ModuleList()
        self.layer2_scale = nn.ModuleList()
        self.layer2_r = nn.ModuleList()
        self.layer2_u = nn.ModuleList()
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # size = 8 * 8 * 128
        self.layer3 = nn.ModuleList()
        self.layer3_scale = nn.ModuleList()
        self.layer3_r = nn.ModuleList()
        self.layer3_u = nn.ModuleList()
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # size = 4 * 4 * 256
        self.layer4 = nn.ModuleList()
        self.layer4_scale = nn.ModuleList()
        self.layer4_r = nn.ModuleList()
        self.layer4_u = nn.ModuleList()
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # size = 2 * 2 * 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()
        self.fc_scale = nn.ModuleList()
        self.fc_r = nn.ModuleList()
        self.fc_u = nn.ModuleList()
        for t, c in task_class:
            self.in_channel = 64
            self.conv1.append(nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False))
            self.bn1.append(self._norm_layer(self.in_channel))
            # self.layer1.append(self._make_layer(block, 64, layers[0]))
            # self.layer2.append(self._make_layer(block, 128, layers[1], stride=2))
            # self.layer3.append(self._make_layer(block, 256, layers[2], stride=2))
            # self.layer4.append(self._make_layer(block, 512, layers[3], stride=2))
            # self.fc.append(nn.Linear(512, c))
            self.layer1.append(self._make_layer(block, 64, layers[0]))
            self.layer2.append(self._make_layer(block, 64, layers[1], stride=2))
            self.layer3.append(self._make_layer(block, 64, layers[2], stride=2))
            self.layer4.append(self._make_layer(block, 64, layers[3], stride=2))
            self.fc.append(nn.Linear(64, c))
            if t > 0: # lateral connections with previous columns
                self.in_channel = 64
                self.layer1_scale.append(nn.Embedding(1, t))
                self.layer1_r.append(nn.Conv2d(t * 64, 64, kernel_size=1, stride=1))
                self.layer1_u.append(self._make_layer(block, 64, layers[0]))

                # self.layer2_scale.append(nn.Embedding(1, t))
                # self.layer2_r.append(nn.Conv2d(t * 64, 64, kernel_size=1, stride=1))
                # self.layer2_u.append(self._make_layer(block, 128, layers[1],stride=2))
                self.layer2_scale.append(nn.Embedding(1, t))
                self.layer2_r.append(nn.Conv2d(t * 64, 64, kernel_size=1, stride=1))
                self.layer2_u.append(self._make_layer(block, 64, layers[1],stride=2))

                # self.layer3_scale.append(nn.Embedding(1, t))
                # self.layer3_r.append(nn.Conv2d(t * 128, 128, kernel_size=1, stride=1))
                # self.layer3_u.append(self._make_layer(block, 256, layers[2],stride=2))
                self.layer3_scale.append(nn.Embedding(1, t))
                self.layer3_r.append(nn.Conv2d(t * 64, 64, kernel_size=1, stride=1))
                self.layer3_u.append(self._make_layer(block, 64, layers[2],stride=2))

                # self.layer4_scale.append(nn.Embedding(1, t))
                # self.layer4_r.append(nn.Conv2d(t * 256, 256, kernel_size=1, stride=1))
                # self.layer4_u.append(self._make_layer(block, 512, layers[3],stride=2))
                self.layer4_scale.append(nn.Embedding(1, t))
                self.layer4_r.append(nn.Conv2d(t * 64, 64, kernel_size=1, stride=1))
                self.layer4_u.append(self._make_layer(block, 64, layers[3],stride=2))
                
                # self.fc_scale.append(nn.Embedding(1, t))
                # self.fc_r.append(nn.Linear(t*512, 512))
                # self.fc_u.append(nn.Linear(512, c))
                self.fc_scale.append(nn.Embedding(1, t))
                self.fc_r.append(nn.Linear(t*64, 64))
                self.fc_u.append(nn.Linear(64, c))
                
        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, channel, num_block, stride=1):
        norm_layer = self._norm_layer
        down_sample = None

        if stride != 1 or self.in_channel != channel:
            down_sample = nn.Sequential(
                conv1x1(self.in_channel, channel, stride),
                norm_layer(channel)
            )

        layers = []
        layers.append(block(self.in_channel, channel, stride,
                            down_sample, norm_layer=norm_layer))
        self.in_channel = channel
        for i in range(1, num_block):
            layers.append(block(self.in_channel, channel,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, t):
        h = self.maxpool(self.relu(self.bn1[t](self.conv1[t](x))))
        if t > 0:
            h_prev = [self.maxpool(self.relu(self.bn1[j](self.conv1[j](x)))) for j in range(t)]

        h = self.layer1[t](h)
        if t > 0:
            later_connect = self.relu(self.layer1_r[t-1](torch.cat([self.layer1_scale[t-1].weight[0][j] * h_prev[j] for j in range(t)], 1)))
            h = h + self.layer1_u[t-1](later_connect)
            h_prev = [self.layer1[j](h_prev[j]) for j in range(t)]
        
        h = self.layer2[t](h)
        if t > 0:
            later_connect = self.relu(self.layer2_r[t-1](torch.cat([self.layer2_scale[t-1].weight[0][j] * h_prev[j] for j in range(t)], 1)))
            h = h + self.layer2_u[t-1](later_connect)
            h_prev = [self.layer2[j](h_prev[j]) for j in range(t)]
        
        h = self.layer3[t](h)
        if t > 0:
            later_connect = self.relu(self.layer3_r[t-1](torch.cat([self.layer3_scale[t-1].weight[0][j] * h_prev[j] for j in range(t)], 1)))
            h = h + self.layer3_u[t-1](later_connect)
            h_prev = [self.layer3[j](h_prev[j]) for j in range(t)]
        
        h = self.layer4[t](h)
        if t > 0:
            later_connect = self.relu(self.layer4_r[t-1](torch.cat([self.layer4_scale[t-1].weight[0][j] * h_prev[j] for j in range(t)], 1)))
            h = h + self.layer4_u[t-1](later_connect)
            h_prev = [self.layer4[j](h_prev[j]) for j in range(t)]
        
        h = torch.flatten(self.avgpool(h), 1)
        h_prev = [torch.flatten(self.avgpool(h_prev[j]), 1) for j in range(t)]
        output = []
        for task_id, c in self.task_class:
            if t > 0 and task_id < t:
                h_task = h_prev[task_id]
                h_task = self.fc[task_id](h_task)
                if task_id > 0:
                    later_connect = self.relu(self.fc_r[task_id-1](torch.cat([self.fc_scale[task_id-1].weight[0][j] * h_prev[j] for j in range(task_id)], 1)))
                    h_task = h_task + self.fc_u[task_id-1](later_connect)
                output.append(h_task)
            else:
                h_task = h
                h_task = self.fc[t](h_task)
                if t > 0:
                    later_connect = self.relu(self.fc_r[t-1](torch.cat([self.fc_scale[t-1].weight[0][j] * h_prev[j] for j in range(t)], 1)))
                    h_task = h_task + self.fc_u[t-1](later_connect)
                output.append(h_task)

        return output

    def unfreeze_column(self, t):
        # utils.set_req_grad(self.conv1[t], True)
        self.conv1[t].requires_grad_(True)
        self.layer1[t].requires_grad_(True)
        self.layer2[t].requires_grad_(True)
        self.layer3[t].requires_grad_(True)
        self.layer4[t].requires_grad_(True)
        self.fc[t].requires_grad_(True)

        if t > 0:
            self.layer1_scale[t-1].requires_grad_(True)
            self.layer1_r[t-1].requires_grad_(True)
            self.layer1_u[t-1].requires_grad_(True)
            self.layer2_scale[t-1].requires_grad_(True)
            self.layer2_r[t-1].requires_grad_(True)
            self.layer2_u[t-1].requires_grad_(True)
            self.layer3_scale[t-1].requires_grad_(True)
            self.layer3_r[t-1].requires_grad_(True)
            self.layer3_u[t-1].requires_grad_(True)
            self.layer4_scale[t-1].requires_grad_(True)
            self.layer4_r[t-1].requires_grad_(True)
            self.layer4_u[t-1].requires_grad_(True)
            self.fc_scale[t-1].requires_grad_(True)
            self.fc_r[t-1].requires_grad_(True)
            self.fc_u[t-1].requires_grad_(True)
            
        return

    def set_train_mode(self, t):
        self.bn1[t].train()
        self.layer1[t].train()
        self.layer2[t].train()
        self.layer3[t].train()
        self.layer4[t].train()
        if t > 0:
            self.layer1_u.train()
            self.layer2_u.train()
            self.layer3_u.train()
            self.layer4_u.train()
        
    def get_model_size(self):
        self.model_size = []
        count = 0
        for i in range(len(self.task_class)):
            for p in self.conv1[i].parameters():
                count += np.prod(p.size())
            for p in self.bn1[i].parameters():
                count += np.prod(p.size())
            for p in self.layer1[i].parameters():
                count += np.prod(p.size())
            for p in self.layer2[i].parameters():
                count += np.prod(p.size())
            for p in self.layer3[i].parameters():
                count += np.prod(p.size())             
            for p in self.layer4[i].parameters():
                count += np.prod(p.size())
            for p in self.fc[i].parameters():
                count += np.prod(p.size())
            if i > 0:
                for p in self.layer1_scale[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer2_scale[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer3_scale[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer4_scale[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.fc_scale[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer1_r[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer1_u[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer2_r[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer2_u[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer3_r[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer3_u[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer4_r[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.layer4_u[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.fc_r[i-1].parameters():
                    count += np.prod(p.size())
                for p in self.fc_u[i-1].parameters():
                    count += np.prod(p.size())
            # count = count / 1000000
            self.model_size.append(count.item() / 1000000)

        model_size = copy.deepcopy(self.model_size)

        return model_size


def _resnet(input_size, block, layers, task_class, **kwargs):
    model = ResNet(input_size, task_class, block, layers, **kwargs)
    return model


def resnet_26(input_size, task_class):
    return _resnet(input_size, BasicBlock, [3, 3, 3, 3], task_class)


def resnet_18(input_size, task_class):
    return _resnet(input_size, BasicBlock, [2, 2, 2, 2], task_class)


def Net(input_size, task_class):
    return resnet_18(input_size, task_class)

