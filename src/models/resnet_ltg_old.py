"""
File        :
Description :resnet 18, learn to grow
Author      :XXX
Date        :2019/9/1
Version     :v1.0
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
        # define the norm layer
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.ModuleList([conv3x3(in_channel, channel, stride)])
        self.bn1 = norm_layer(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ModuleList([conv3x3(channel, channel)])
        self.bn2 = norm_layer(channel)
        self.downsample = downsample
        self.stride = stride

    def expand(self, a):
        pass

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

    def __init__(self, input_size, block, layers, task_class, norm_layer=None):
        super(ResNet, self).__init__()

        self.input_size = input_size
        self.task_class = task_class

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        # the input channels of the block
        self.in_channel = 64
        # the channels of every layer
        self.channels = {}
        # size = 32 * 32 * 3, a double nested module list
        self.conv1 = nn.ModuleList([nn.ModuleList([nn.Conv2d(3, self.in_channel,
                                    kernel_size=3, stride=1, padding=1, bias=False)])])
        self.channels['conv1'] = deepcopy(self.in_channel)

        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # size = 32 * 32 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # size = 16 * 16 * 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        # size = 16 * 16 * 64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # size = 8 * 8 * 128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # size = 4 * 4 * 256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # size = 2 * 2 * 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()
        for t, c in task_class:
            # TODO: modify to 512
            # self.fc.append(nn.Linear(512, c))
            self.fc.append(nn.Linear(64, c))

        # initialize the length of every module list
        self.length = {'conv1': 1}
        for i in range(1, 5):
            self.length['layer'+str(i)] = [[1, 1], [1, 1]]

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
        # The new models and new paramemters
        self.new_models = None
        self.new_params = None
        self.params_to_train = None
        # TODO: for test
        self.old_out = None

    def _make_layer(self, block, channel, num_block, stride=1):
        # layer由相同的block组成
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

        return layers

    def forward(self, x, t, task_arch=None, is_test=False):
        # task_arch: 任务对应的结构参数，训练阶段需要训练，推理阶段则是固定的
        # 1 training stage of the expandable section
        if task_arch is None:
            # 1 forward: conv1
            if t == 0:  # task 1
                out_ = self.conv1[0][0](x)
            else:  # task 2 to n
                # softmax a
                g_conv = torch.exp(self.a['conv1']) / torch.sum(torch.exp(self.a['conv1']))
                # self.a['conv1'].data = torch.exp(self.a['conv1'].data) / torch.sum(torch.exp(self.a['conv1'].data))
                # 1.1 new
                out_ = g_conv[-1] * self.conv1[-1][0](x)
                # 1.2 reuse and adaption
                c = self.length['conv1']
                for i in range(c):
                    for j in range(len(self.conv1[i])-1):
                        # reuse for submodel i
                        out_ += g_conv[i] * self.conv1[i][j](x)
                        # adaption for the submodel i
                        out_ += g_conv[c+i] * self.conv1[i][j](x)
                    # adaption for the submodel j
                    out_ += g_conv[c+i] * self.conv1[i][-1](x)

            # 2 forward: common
            x = self.bn1(out_)
            x = self.relu(x)
            x = self.maxpool(x)

            # 3 TODO: forward: layer 1
            if t == 0:  # task 1
                pass
            else:
                pass
                # self.a['layer1'].data = torch.exp(self.a['layer1'].data) / torch.sum(torch.exp(self.a['layer1'].data))

            # TODO: forward for layers

        # 2 testing stage of the expandable section
        else:
            out_ = None
            arch_conv1 = task_arch[0]
            for i in range(arch_conv1[1]+1):
                if i == 0:
                    out_ = self.conv1[arch_conv1[0]][i](x)
                else:
                    out_ += self.conv1[arch_conv1[0]][i](x)
            # TODO: test
            # if is_test:
            #     print("="*5, "Output is Equal?", "="*5)
            #
            #     tmp = out_.detach().cpu().numpy()
            #     if self.old_out is None:
            #         self.old_out = tmp
            #     else:
            #         print(np.sum(self.old_out == tmp))
            #         self.old_out = tmp
            # forward
            x = self.bn1(out_)
            x = self.relu(x)
            x = self.maxpool(x)
            # TODO: forward layers

        # the common non-expandable section
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        output = []
        for t, c in self.task_class:
            output.append(self.fc[t](x))

        return output

    def expand(self, t, device='cuda'):
        # TODO: expand the network according the actions for task t
        # 1 expand conv1
        # 1.1 action: new, add a new submodel, insert in the outer module list
        self.conv1.append(nn.ModuleList([nn.Conv2d(3, self.channels['conv1'], kernel_size=3,
                                                   stride=1, padding=1, bias=False)]).to(device))
        # 1.2 action: reuse and adaption
        for i in range(self.length['conv1']):
            block = self.conv1[i]
            # take actions for every submodel in super network
            # action: reuse, freeze the old submodel
            utils.freeze_model(block)
            # action: adaption
            block.append(nn.Conv2d(3, self.channels['conv1'], kernel_size=1,
                                   stride=1, bias=False).to(device))
        # 1.3 generate the architecture parameter of conv1
        # the action number of conv1, = 2c+1
        num_l = self.length['conv1'] * 2 + 1
        # the action parameter of conv1
        # reuse (0:c-1), adaption (c:2c-1), new (2c)
        self.a['conv1'] = nn.Parameter(torch.rand(num_l, requires_grad=True).to(device))
        # TODO: 2 expand layer 1
        # TODO: 3 expand layer 2
        # ...

        # get the new models and parameters
        self.get_new_model(t=t)
        self.new_params = self.get_param(self.new_models)

        return

    def get_param(self, models):
        params = []
        if 'conv1' in models.keys():
            for idx in models['conv1']:
                params.append({'params': self.conv1[idx[0]][idx[1]].parameters()})

        if 'fc' in models.keys():
            for idx in models['fc']:
                params.append({'params': self.fc[idx].parameters()})

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
        if 'fc' in models.keys():
            for idx in models['fc']:
                utils.modify_model(self.fc[idx], requires_grad)

    def get_new_model(self, t):
        # new model. (adaption and new)
        # conv1
        new_models = {'conv1': [], 'fc': []}
        c = self.length['conv1']
        # adaption
        for i in range(c):
            # for every submodel in conv1
            new_models['conv1'].append([i, -1])
        # new
        new_models['conv1'].append([c, 0])

        # task specific layer
        new_models['fc'].append(t)

        self.new_models = new_models

    def regular_loss(self):
        loss = 0.0
        c = self.length['conv1']
        # print("="*10)
        for i in range(c):
            # print(utils.model_size(self.conv1[i][-1]))
            # print(self.a['conv1'][c+i])
            loss += self.a['conv1'][c+i] * utils.model_size(self.conv1[i][-1])
            # print(loss)
        # print(loss)
        loss += self.a['conv1'][2*c] * utils.model_size(self.conv1[c][-1])
        # print(utils.model_size(self.conv1[c][-1]))
        # print(loss)
        # print("=" * 10)
        return loss

    def select(self, t):
        # TODO: select the best architecture from the expanded network
        # TODO: delete the redundant parts from the expanded network
        # TODO: select the models need to training for task and their parameters

        # define the container of models to train
        model_to_train = {'conv1': [], 'fc': []}

        # select the best architecture for conv1
        archi = []
        print(self.a['conv1'].data)
        v, arg_v = torch.max(self.a['conv1'].data, dim=0)
        idx = deepcopy(arg_v.item())
        c = self.length['conv1']
        if idx < c:
            # reuse
            archi.append([idx, len(self.conv1[idx])-2])
        elif idx < 2*c:
            # adaption
            model_to_train['conv1'].append([idx-c, -1])
            archi.append([idx-c, len(self.conv1[idx-c])-1])
        elif idx == 2*c:
            # new
            model_to_train['conv1'].append([c, 0])
            archi.append([c, len(self.conv1[c]) - 1])

        # delete
        for i in range(2*c+1):
            if i != idx:  # do not select the action
                if c <= i < 2*c:
                    # adaption
                    del self.conv1[i-c][-1]
                elif i == 2*c:
                    # new
                    del self.conv1[-1]

        self.length['conv1'] = len(self.conv1)
        print("="*10)
        print("length: {}".format(self.length['conv1']))

        # add the task specific layer
        model_to_train['fc'].append(t)

        # params to train
        params_to_train = self.get_param(model_to_train)

        return archi, model_to_train, params_to_train


def _resnet(input_size, block, layers, task_class, **kwargs):
    model = ResNet(input_size, block, layers, task_class, **kwargs)
    return model


def resnet_18(input_size, task_class):
    return _resnet(input_size, BasicBlock, [2, 2, 2, 2], task_class)


def Net(input_size, task_class):
    return resnet_18(input_size, task_class)

