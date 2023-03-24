"""
File        :
Description :resnet 18
Author      :XXX
Date        :2019/9/1
Version     :v1.0
"""
import torch
import torch.nn as nn


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
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        # size = 32 * 32 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # size = 16 * 16 * 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        # size = 16 * 16 * 64
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        # size = 8 * 8 * 128
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # size = 4 * 4 * 256
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        # size = 2 * 2 * 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.ModuleList()
        for t, c in task_class:
            self.fc.append(nn.Linear(64, c))

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.fc[self.current_task](x)
    
    def multihead_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        outputs = [self.fc[i](x) for i in range(self.current_task+1)]

        return outputs
    
    def extract_feat(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def prepare(self, t):
        self.current_task = t


class ResNetFeat(nn.Module):

    def __init__(self, input_size, block, layers, norm_layer=None):
        super(ResNetFeat, self).__init__()

        self.input_size = input_size

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer
        # the input channels of the block
        self.in_channel = 64
        # size = 32 * 32 * 3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def _resnet(input_size, block, layers, task_class, **kwargs):
    model = ResNet(input_size, task_class, block, layers, **kwargs)
    return model

def resnet_18(input_size, task_class):
    return _resnet(input_size, BasicBlock, [2, 2, 2, 2], task_class)

def resnet_18_feat(input_size):
    return ResNetFeat(input_size, BasicBlock, [2, 2, 2, 2])

def resnet_26(input_size, task_class):
    return _resnet(input_size, BasicBlock, [3, 3, 3, 3], task_class)

def resnet_34(input_size, task_class):
    return _resnet(input_size, BasicBlock, [3, 4, 6, 3], task_class)

def Net(input_size, task_class):
    return resnet_18(input_size, task_class)

