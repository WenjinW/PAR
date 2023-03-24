"""
File        :
Description :resnet 18, learn to grow
Author      :XXX
Date        :2019/8/9
Version     :v2.0
"""
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

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
        self.downsample = downsample
        self.init_arch = {'conv1': [[0, 0]], 'conv2': [[0, 0]],
                          'bn1': [0], 'bn2': [0]}
        self.length = {'conv1': 1, 'conv2': 1}
        # define the norm layer
        if norm_layer is None:
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = norm_layer
        if self.downsample is not None:
            self.downsample_conv = nn.ModuleList([nn.ModuleList([
                conv1x1(self.downsample[0], self.downsample[1], self.downsample[2])
            ])])
            self.downsample_bn = nn.ModuleList([self.norm_layer(self.downsample[1])])
            self.init_arch['d_conv'] = [[0, 0]]
            self.init_arch['d_bn'] = [0]
            self.length['d_conv'] = 1

        self.conv1 = nn.ModuleList([nn.ModuleList([conv3x3(in_channel,
                                                           channel, stride)])])
        self.bn1 = nn.ModuleList([self.norm_layer(channel)])

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.ModuleList([nn.ModuleList([conv3x3(channel,
                                                           channel)])])
        self.bn2 = nn.ModuleList([self.norm_layer(channel)])

        self.stride = stride
        self.in_channel = in_channel
        self.channel = channel
        # action parameter
        self.a = nn.ParameterDict()
        # new model
        self.new_models = None

    def expand(self, t, device='cuda'):
        # expand the network to a super model
        # 1 expand d_conv and d_bn
        if self.downsample is not None:
            # 1.1 expand d_conv
            # 1.1.1 action: new
            self.downsample_conv.append(nn.ModuleList([
                conv1x1(self.downsample[0], self.downsample[1],
                        self.downsample[2])]).to(device))
            # 1.1.2 action: adaption
            for i in range(self.length['d_conv']):
                self.downsample_conv[i].append(
                    conv1x1(self.downsample[0], self.downsample[1],
                            self.downsample[2]).to(device))
            # 1.2 expand d_bn
            self.downsample_bn.append(self.norm_layer(self.downsample[1]).to(device))
            # 1.3 generate action parameter
            # the action number of d_conv = 2c+1
            num_l = self.length['d_conv'] * 2 + 1
            # the action parameter of conv1
            # reuse (0:c-1), adaption (c:2c-1), new (2c)
            self.a['d_conv'] = nn.Parameter(torch.rand(num_l).to(device))

        # 2 expand conv1 and bn1
        # 2.1 expand conv1
        # 2.1.1 action: new
        self.conv1.append(nn.ModuleList([conv3x3(self.in_channel, self.channel, self.stride)]).to(device))
        # 2.1.2 action: adaption
        for i in range(self.length['conv1']):
            self.conv1[i].append(conv1x1(self.in_channel, self.channel, self.stride).to(device))
        # 2.2 expand bn1
        self.bn1.append(self.norm_layer(self.channel).to(device))
        # 2.3 generate action parameter
        num_l = self.length['conv1'] * 2 + 1
        self.a['conv1'] = nn.Parameter(torch.rand(num_l).to(device))

        # 3 expand conv2 and bn2
        # 3.1 expand conv2
        # 3.1.1 action: new
        self.conv2.append(nn.ModuleList([conv3x3(self.channel, self.channel)]).to(device))
        # 3.1.2 action: adaption
        for i in range(self.length['conv2']):
            self.conv2[i].append(conv1x1(self.channel, self.channel).to(device))
        # 3.2 expand bn2
        self.bn2.append(self.norm_layer(self.channel).to(device))
        # 3.3 generate action parameter
        num_l = self.length['conv2'] * 2 + 1
        self.a['conv2'] = nn.Parameter(torch.rand(num_l).to(device))

        self.get_new_model(t=t)

    def forward(self, x, t, task_arch):
        # 1 deal with the downsample
        if self.downsample is not None:
            d_conv = task_arch['d_conv'][0]
            d_bn = task_arch['d_bn'][0]
            identity = self.downsample_conv[d_conv[0]][d_conv[1]](x)
            identity = self.downsample_bn[d_bn](identity)
        else:
            identity = x
        # 2 deal with conv1
        conv1 = task_arch['conv1'][0]
        x = self.conv1[conv1[0]][conv1[1]](x)
        # 3 deal with bn1
        bn1 = task_arch['bn1'][0]
        x = self.bn1[bn1](x)
        # 4 relu
        x = self.relu(x)
        # 5 conv2
        conv2 = task_arch['conv2'][0]
        x = self.conv2[conv2[0]][conv2[1]](x)
        # 6 bn2
        bn2 = task_arch['bn2'][0]
        x = self.bn2[bn2](x)

        x += identity
        output = self.relu(x)

        return output

    def get_new_model(self, t):
        # new model. (adaption and new)
        new_models = {'conv1': [], 'bn1': [], 'conv2': [], 'bn2': []}
        # 1 d_conv and d_bn
        if self.downsample is not None:
            c = self.length['d_conv']
            # 1.1 d_conv: new
            new_models['d_conv1'] = [[c, 0]]
            # 1.2 d_conv: adaption
            for i in range(c):
                new_models['d_conv1'].append([i, len(self.downsample_conv[i])-1])
            # 1.3 d_bn
            new_models['d_bn'] = [t]

        # 2 conv1 and bn1
        c = self.length['conv1']
        # 2.1 new
        new_models['conv1'].append([c, 0])
        # 2.2 adaption
        for i in range(c):
            new_models['conv1'].append([i, len(self.conv1[i])-1])
        # 2.3 bn1
        new_models['bn1'].append(t)

        # 3 conv2 and bn2
        c = self.length['conv2']
        # 3.1 new
        new_models['conv2'].append([c, 0])
        # 3.2 adaption
        for i in range(c):
            new_models['conv2'].append([i, len(self.conv2[i]) - 1])
        # 3.3 bn2
        new_models['bn2'].append(t)

        self.new_models = new_models

    def search_forward(self, x, t):
        # 1 d_conv
        if self.downsample is not None:
            g_conv_d = torch.exp(self.a['d_conv']) / torch.sum(torch.exp(self.a['d_conv']))
            # 1.1 d_conv: new
            out_ = g_conv_d[-1] * self.downsample_conv[-1][0](x)
            # 1.2 d_conv: reuse and adaption
            c = self.length['d_conv']
            for i in range(c):
                for j in range(len(self.downsample_conv[i]) - 1):
                    # reuse for submodel i
                    out_ += g_conv_d[i] * self.downsample_conv[i][j](x)
                    # adaption for the submodel i
                    out_ += g_conv_d[c + i] * self.downsample_conv[i][j](x)
                # adaption for the submodel j
                out_ += g_conv_d[c + i] * self.downsample_conv[i][-1](x)
            # 2 d_bn
            identify = self.downsample_bn[t](out_)
        else:
            identify = x
        # 3 conv1
        g_conv_1 = torch.exp(self.a['conv1']) / torch.sum(torch.exp(self.a['conv1']))
        # 3.1 conv1: new
        out_ = g_conv_1[-1] * self.conv1[-1][0](x)
        # 3.2 conv1: reuse and adaption
        c = self.length['conv1']
        for i in range(c):
            for j in range(len(self.conv1[i]) - 1):
                # reuse for submodel i
                out_ += g_conv_1[i] * self.conv1[i][j](x)
                # adaption for the submodel i
                out_ += g_conv_1[c + i] * self.conv1[i][j](x)
            # adaption for the submodel j
            out_ += g_conv_1[c + i] * self.conv1[i][-1](x)
        # 4 bn1
        x = self.bn1[t](out_)
        # 5 relu
        x = self.relu(x)
        # 6 conv2
        g_conv_2 = torch.exp(self.a['conv2']) / torch.sum(torch.exp(self.a['conv2']))
        # 3.1 conv1: new
        out_ = g_conv_2[-1] * self.conv2[-1][0](x)
        # 3.2 conv1: reuse and adaption
        c = self.length['conv2']
        for i in range(c):
            for j in range(len(self.conv2[i]) - 1):
                # reuse for submodel i
                out_ += g_conv_2[i] * self.conv2[i][j](x)
                # adaption for the submodel i
                out_ += g_conv_2[c + i] * self.conv2[i][j](x)
            # adaption for the submodel j
            out_ += g_conv_2[c + i] * self.conv2[i][-1](x)
        # 7 bn2
        x = self.bn2[t](out_)
        # 8 sum and relu
        x += identify
        x = self.relu(x)

        return x

    def select(self, t):
        # 1 define the container of new models to train and the best submodel
        model_to_train = {'conv1': [], 'bn1': [], 'conv2': [], 'bn2': []}
        best_archi = {'conv1': [], 'bn1': [], 'conv2': [], 'bn2': []}
        # 2 d_conv
        if self.downsample is not None:
            model_to_train['d_conv'] = []
            best_archi['d_conv'] = []
            # 2.1 select the best architecture for d_conv
            v, arg_v = torch.max(self.a['d_conv'].data, dim=0)
            idx = deepcopy(arg_v.item())
            c = self.length['d_conv']
            if idx < c:
                # reuse
                best_archi['d_conv'].append([idx, len(self.downsample_conv[idx]) - 2])
            elif idx < 2 * c:
                # adaption
                model_to_train['d_conv'].append([idx - c, len(self.downsample_conv[idx - c]) - 1])
                best_archi['d_conv'].append([idx - c, len(self.downsample_conv[idx - c]) - 1])
            elif idx == 2 * c:
                # new
                model_to_train['d_conv'].append([c, 0])
                best_archi['d_conv'].append([c, 0])
            # 2.2 delete for conv1
            for i in range(2 * c + 1):
                if i != idx:  # do not select the action
                    if c <= i < 2 * c:
                        # adaption
                        del self.downsample_conv[i - c][-1]
                    elif i == 2 * c:
                        # new
                        del self.downsample_conv[-1]
            # 2.3 d_bn
            model_to_train['d_bn'] = [t]
            best_archi['d_bn'] = [t]

        # 3 conv1
        # 3.1 select the best architecture for conv1
        v, arg_v = torch.max(self.a['conv1'].data, dim=0)
        idx = deepcopy(arg_v.item())
        c = self.length['conv1']
        if idx < c:
            # reuse
            best_archi['conv1'].append([idx, len(self.conv1[idx]) - 2])
        elif idx < 2 * c:
            # adaption
            model_to_train['conv1'].append([idx - c, len(self.conv1[idx - c]) - 1])
            best_archi['conv1'].append([idx - c, len(self.conv1[idx - c]) - 1])
        elif idx == 2 * c:
            # new
            model_to_train['conv1'].append([c, 0])
            best_archi['conv1'].append([c, 0])
        # 3.2 delete for conv1
        for i in range(2 * c + 1):
            if i != idx:  # do not select the action
                if c <= i < 2 * c:
                    # adaption
                    del self.conv1[i - c][-1]
                elif i == 2 * c:
                    # new
                    del self.conv1[-1]
        # 3.3 bn1
        model_to_train['bn1'].append(t)
        best_archi['bn1'].append(t)

        # 4 conv2
        # 4.1 select the best architecture for conv2
        v, arg_v = torch.max(self.a['conv2'].data, dim=0)
        idx = deepcopy(arg_v.item())
        c = self.length['conv2']
        if idx < c:
            # reuse
            best_archi['conv2'].append([idx, len(self.conv2[idx]) - 2])
        elif idx < 2 * c:
            # adaption
            model_to_train['conv2'].append([idx - c, len(self.conv2[idx - c]) - 1])
            best_archi['conv2'].append([idx - c, len(self.conv2[idx - c]) - 1])
        elif idx == 2 * c:
            # new
            model_to_train['conv2'].append([c, 0])
            best_archi['conv2'].append([c, 0])
        # 4.2 delete for conv2
        for i in range(2 * c + 1):
            if i != idx:  # do not select the action
                if c <= i < 2 * c:
                    # adaption
                    del self.conv2[i - c][-1]
                elif i == 2 * c:
                    # new
                    del self.conv2[-1]
        # 4.3 bn2
        model_to_train['bn2'].append(t)
        best_archi['bn2'].append(t)

        # 5 update the length of every layer
        if self.downsample is not None:
            self.length['d_conv'] = len(self.downsample_conv)
        self.length['conv1'] = len(self.conv1)
        self.length['conv2'] = len(self.conv2)

        return model_to_train, best_archi


class ResNet(nn.Module):
    """ A small variant of the ResNet (MNTDP) for Learn to grow.

    """
    def __init__(self, input_size, task_class, block, layers, norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()

        self.input_size = input_size
        self.task_class = task_class
        self._norm_layer = norm_layer

        # the input channels of the block
        self.in_channel = 64
        # the channels of every layer
        self.channels = {}

        # size = 32 * 32 * 3, a double nested module list
        self.conv1 = nn.ModuleList([nn.ModuleList([nn.Conv2d(self.input_size[0], self.in_channel,
                                                             kernel_size=3, stride=1, padding=1, bias=False)])])

        self.channels['conv1'] = deepcopy(self.in_channel)
        self.bn1 = nn.ModuleList([self._norm_layer(self.channels['conv1'])])

        self.relu = nn.ReLU(inplace=True)
        # size = 32 * 32 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # size = 16 * 16 * 64
        self.layer1, arch_1 = self._make_layer(block, 64, layers[0])
        # size = 16 * 16 * 64
        self.layer2, arch_2 = self._make_layer(block, 64, layers[1], stride=2)
        # size = 8 * 8 * 128
        self.layer3, arch_3 = self._make_layer(block, 64, layers[2], stride=2)
        # size = 4 * 4 * 256
        self.layer4, arch_4 = self._make_layer(block, 64, layers[3], stride=2)
        # size = 2 * 2 * 512
        self.layer5, arch_5 = self._make_layer(block, 64, layers[4], stride=2)
        # size = 1 * 1 * 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()

        for t, c in task_class:
            self.fc.append(nn.Linear(512, c))
            # self.fc.append(nn.Linear(64, c))

        # initialize the length of every module list
        self.length = {'conv1': 1}
        # model parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # parameter for architecture search
        self.a = torch.nn.ParameterDict({})
        # The new models, model to train
        self.new_models = None
        self.model_to_train = None
        # initial architecture
        self.init_archi = {'conv1': [[0, 0]], 'bn1': [0], 'fc': [0],
                           'layer1': arch_1, 'layer2': arch_2, 'layer3': arch_3,
                           'layer4': arch_4}

    def _make_layer(self, block, channel, num_block, stride=1):
        # layer由相同的block组成
        norm_layer = self._norm_layer
        down_sample = None
        # architecture
        archi = []

        if stride != 1 or self.in_channel != channel:
            down_sample = [deepcopy(self.in_channel), deepcopy(channel), deepcopy(stride)]

        layers = []
        b = block(self.in_channel, channel, stride,
                          down_sample, norm_layer=norm_layer)
        layers.append(b)
        archi.append(b.init_arch)
        self.in_channel = channel
        for i in range(1, num_block):
            b = block(self.in_channel, channel, norm_layer=norm_layer)
            layers.append(b)
            archi.append(b.init_arch)

        return nn.ModuleList(layers), archi

    def expand(self, t, device='cuda'):
        # expand the network to a super model
        # 1 expand conv1 and bn1
        # 1.1 action: new
        self.conv1.append(nn.ModuleList([nn.Conv2d(self.input_size[0], self.channels['conv1'], kernel_size=3,
                                                   stride=1, padding=1, bias=False)]).to(device))
        # 1.2 action adaption
        for i in range(self.length['conv1']):
            self.conv1[i].append(nn.Conv2d(self.input_size[0], self.channels['conv1'], kernel_size=1,
                                           stride=1, bias=False).to(device))

        # 1.3 expand bn1
        self.bn1.append(self._norm_layer(self.channels['conv1']).to(device))
        # 1.4 generate action parameter
        # the action number of conv1, = 2c+1
        num_l = self.length['conv1'] * 2 + 1
        # the action parameter of conv1
        # reuse (0:c-1), adaption (c:2c-1), new (2c)
        self.a['conv1'] = nn.Parameter(torch.rand(num_l).to(device))
        # 2 expand layer 1-4
        for layer_n in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in layer_n:
                b.expand(t, device)

        self.get_new_model(t=t)

    def forward(self, x, t, task_arch):
        # 1 deal with the conv1
        # 1.1 architecture of conv1
        arch_conv1 = task_arch['conv1'][0]

        out_conv1 = self.conv1[arch_conv1[0]][arch_conv1[1]](x)
        h = self.bn1[task_arch['bn1'][0]](out_conv1)
        h = self.relu(h)
        h = self.maxpool(h)
        # 2 deal with layer 1 to 4
        for (name, layer_n) in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                 [self.layer1, self.layer2, self.layer3, self.layer4]):

            layer_l = len(layer_n)
            for i in range(layer_l):
                h = layer_n[i](h, t, task_arch[name][i])

        h = self.avgpool(h)
        h = torch.flatten(h, 1)

        output = []
        for u, c in self.task_class:
            output.append(self.fc[u](h))

        return output

    def get_archi_param(self):
        params = []
        params.append(self.a)
        for layer_n in [self.layer1, self.layer2, self.layer3,
                        self.layer4]:
            for b in layer_n:
                params.append(b.a)

        return params

    def get_new_model(self, t):
        # new model. (adaption and new)
        # 1 conv1
        new_models = {'conv1': [], 'bn1': [], 'fc': []}
        c = self.length['conv1']
        # 1.1 new
        new_models['conv1'].append([c, 0])
        # 1.2 adaption
        for i in range(c):
            new_models['conv1'].append([i, len(self.conv1[i])-1])
        # 2 bn1
        new_models['bn1'].append(t)
        # 3 layer 1
        for (name, layer_n) in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                   [self.layer1, self.layer2, self.layer3,
                                    self.layer4]):

            new_models[name] = []
            for i in range(len(layer_n)):
                new_models[name].append(layer_n[i].new_models)

        # task specific layer
        new_models['fc'].append(t)

        self.new_models = new_models

    def get_param(self, models):
        params = []
        if 'conv1' in models.keys():
            for idx in models['conv1']:
                params.append({'params': self.conv1[idx[0]][idx[1]].parameters()})

        if 'bn1' in models.keys():
            for idx in models['bn1']:
                params.append({'params': self.bn1[idx].parameters()})

        if 'fc' in models.keys():
            for idx in models['fc']:
                params.append({'params': self.fc[idx].parameters()})

        for (name, layer_n) in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                   [self.layer1, self.layer2, self.layer3,
                                    self.layer4]):

            if name in models.keys():
                for i in range(len(models[name])):
                    model_b = models[name][i]
                    # for every block:
                    if 'd_conv' in model_b.keys():
                        for idx in model_b['d_conv']:
                            params.append({'params': layer_n[i].downsample_conv[idx[0]][idx[1]].parameters()})
                    if 'd_bn' in model_b.keys():
                        for idx in model_b['d_bn']:
                            params.append({'params': layer_n[i].downsample_bn[idx].parameters()})
                    if 'conv1' in model_b.keys():
                        for idx in model_b['conv1']:
                            params.append({'params': layer_n[i].conv1[idx[0]][idx[1]].parameters()})
                    if 'bn1' in model_b.keys():
                        for idx in model_b['bn1']:
                            params.append({'params': layer_n[i].bn1[idx].parameters()})
                    if 'conv2' in model_b.keys():
                        for idx in model_b['conv2']:
                            params.append({'params': layer_n[i].conv2[idx[0]][idx[1]].parameters()})
                    if 'bn2' in model_b.keys():
                        for idx in model_b['bn2']:
                            params.append({'params': layer_n[i].bn2[idx].parameters()})

        return params

    def modify_param(self, models, requires_grad=True):
        """freeze or unfreeze the new model. (adaption and new)

        :param models: a dict of submodel
        :param requires_grad:
        :return:
        """
        if 'conv1' in models.keys():
            for idx in models['conv1']:
                utils.modify_model(self.conv1[idx[0]][idx[1]], requires_grad)

        if 'bn1' in models.keys():
            for idx in models['bn1']:
                utils.modify_model(self.bn1[idx], requires_grad)

        if 'fc' in models.keys():
            for idx in models['fc']:
                utils.modify_model(self.fc[idx], requires_grad)

        for (name, layer_n) in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                   [self.layer1, self.layer2, self.layer3,
                                    self.layer4]):

            if name in models.keys():
                for i in range(len(models[name])):
                    model_b = models[name][i]
                    # for every block:
                    if 'd_conv' in model_b.keys():
                        for idx in model_b['d_conv']:
                            utils.modify_model(layer_n[i].downsample_conv[idx[0]][idx[1]], requires_grad)
                    if 'd_bn' in model_b.keys():
                        for idx in model_b['d_bn']:
                            utils.modify_model(layer_n[i].downsample_bn[idx], requires_grad)
                    if 'conv1' in model_b.keys():
                        for idx in model_b['conv1']:
                            utils.modify_model(layer_n[i].conv1[idx[0]][idx[1]], requires_grad)
                    if 'bn1' in model_b.keys():
                        for idx in model_b['bn1']:
                            utils.modify_model(layer_n[i].bn1[idx], requires_grad)
                    if 'conv2' in model_b.keys():
                        for idx in model_b['conv2']:
                            utils.modify_model(layer_n[i].conv2[idx[0]][idx[1]], requires_grad)
                    if 'bn2' in model_b.keys():
                        for idx in model_b['bn2']:
                            utils.modify_model(layer_n[i].bn2[idx], requires_grad)

    def modify_archi_param(self, requires_grad=True):
        params = self.get_archi_param()
        for param in params:
            if requires_grad:
                utils.unfreeze_parameter(param)
            else:
                utils.freeze_parameter(param)

    def regular_loss(self):
        loss = 0.0
        # conv 1
        c = self.length['conv1']
        g_conv = torch.exp(self.a['conv1']) / torch.sum(torch.exp(self.a['conv1']))
        for i in range(c):
            loss += g_conv[c+i] * utils.model_size(self.conv1[i][-1])
        loss += g_conv[2*c] * utils.model_size(self.conv1[c][-1])
        # layers
        for layer_n in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for b in layer_n:
                # d_conv
                if b.downsample is not None:
                    c = b.length['d_conv']
                    g_dconv = torch.exp(b.a['d_conv']) / torch.sum(torch.exp(b.a['d_conv']))
                    for i in range(c):
                        loss += g_dconv[c+i] * utils.model_size(b.downsample_conv[i][-1])
                    loss += g_dconv[2*c] * utils.model_size(b.downsample_conv[c][-1])
                # conv1
                c = b.length['conv1']
                g_conv1 = torch.exp(b.a['conv1']) / torch.sum(torch.exp(b.a['conv1']))
                for i in range(c):
                    loss += g_conv1[c + i] * utils.model_size(b.conv1[i][-1])
                loss += g_conv1[2 * c] * utils.model_size(b.conv1[c][-1])
                # conv2
                c = b.length['conv2']
                g_conv2 = torch.exp(b.a['conv2']) / torch.sum(torch.exp(b.a['conv2']))
                for i in range(c):
                    loss += g_conv2[c + i] * utils.model_size(b.conv2[i][-1])
                loss += g_conv2[2 * c] * utils.model_size(b.conv2[c][-1])

        return loss

    def search_forward(self, x, t):
        # 1 conv1
        g_conv = torch.exp(self.a['conv1']) / torch.sum(torch.exp(self.a['conv1']))
        # 1.1 conv1: new
        out_ = g_conv[-1] * self.conv1[-1][0](x)
        # 1.2 conv1: reuse and adaption
        c = self.length['conv1']
        for i in range(c):
            for j in range(len(self.conv1[i]) - 1):
                # reuse for submodel i
                out_ += g_conv[i] * self.conv1[i][j](x)
                # adaption for the submodel i
                out_ += g_conv[c + i] * self.conv1[i][j](x)
            # adaption for the submodel j
            out_ += g_conv[c + i] * self.conv1[i][-1](x)

        x = self.bn1[t](out_)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2 deal with layer 1 to 4
        for (name, layer_n) in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                   [self.layer1, self.layer2, self.layer3, self.layer4]):

            layer_l = len(layer_n)
            for i in range(layer_l):
                x = layer_n[i].search_forward(x, t)

        # 3 the common non-expandable section
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = []
        for u, c in self.task_class:
            output.append(self.fc[u](x))

        return output

    def select(self, t):
        # select the best model for task t from super model

        # 1 define the container of new models to train and the best submodel
        model_to_train = {'conv1': [], 'bn1': [], 'fc': []}
        best_archi = {'conv1': [], 'bn1': [], 'fc': []}
        # 2 conv1
        # 2.1 select the best architecture for conv1
        v, arg_v = torch.max(self.a['conv1'].data, dim=0)
        idx = deepcopy(arg_v.item())
        c = self.length['conv1']
        if idx < c:
            # reuse
            best_archi['conv1'].append([idx, len(self.conv1[idx]) - 2])
        elif idx < 2 * c:
            # adaption
            model_to_train['conv1'].append([idx - c, len(self.conv1[idx - c]) - 1])
            best_archi['conv1'].append([idx - c, len(self.conv1[idx - c]) - 1])
        elif idx == 2 * c:
            # new
            model_to_train['conv1'].append([c, 0])
            best_archi['conv1'].append([c, 0])
        # 2.2 delete for conv1
        for i in range(2 * c + 1):
            if i != idx:  # do not select the action
                if c <= i < 2 * c:
                    # adaption
                    del self.conv1[i - c][-1]
                elif i == 2 * c:
                    # new
                    del self.conv1[-1]

        # 3 bn1
        model_to_train['bn1'].append(t)
        best_archi['bn1'].append(t)
        # 4 layer 1 to 4
        for (name, layer_n) in zip(['layer1', 'layer2', 'layer3', 'layer4'],
                                   [self.layer1, self.layer2, self.layer3, self.layer4]):
            model_to_train[name], best_archi[name] = [], []
            for b in layer_n:
                model_to_train_b, best_archi_b = b.select(t)
                model_to_train[name].append(model_to_train_b)
                best_archi[name].append(best_archi_b)

        # 5 the task specific layer
        model_to_train['fc'].append(t)
        best_archi['fc'].append(t)

        # update the length of every layer
        self.length['conv1'] = len(self.conv1)
        # update the model to train
        self.model_to_train = model_to_train

        return best_archi


def Net(inputsize, taskcla):
    # ResNet 18

    return ResNet(inputsize, taskcla, BasicBlock, layers=[2, 2, 2, 2])
